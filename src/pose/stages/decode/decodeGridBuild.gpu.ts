import { GRID_STEP } from '../../../sphereLab/constants.ts';
import { C, R, torus } from '../../../sphereLab/floorPattern.ts';
import { type DecodeGridLayout } from './decodeGrid.ts';
import { type DecodeSampleGrid, type DecodeSamplePoint, type VoteResult } from '../../results.ts';
import {
  DECODE_RESULT_BYTES, encodeDecodeArgmax, encodeDecodeTally, unpackDecodeResult,
} from './decodeTally.gpu.ts';
import { dispatchCount, getGPUDevice, uploadFloat32, uploadUint32 } from '../../gpu/device.ts';
import { type Alloc, type Arena, type Slice, bind, sliceRange } from '../../gpu/arena.ts';
import { readSliceF32, readSliceU32 } from '../../gpu/device.ts';
import { gpuTimelineSlot } from '../../gpu/gpuTimeline.ts';
import { poseArena } from '../lsd/chain.ts';
import { DECODE_CORRECTNESS_WGSL, DECODE_GRID_BUILD_WGSL } from './decodeGridBuild.wgsl.ts';

interface Pipelines {
  buildLayout: GPUBindGroupLayout; build: GPUComputePipeline;
  correctLayout: GPUBindGroupLayout; correctness: GPUComputePipeline;
}

const pipelineCache = new WeakMap<GPUDevice, Pipelines>();
function getPipelines(device: GPUDevice): Pipelines {
  let p = pipelineCache.get(device);
  if (!p) {
    const storage = (i: number, ro: boolean): GPUBindGroupLayoutEntry => ({
      binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: ro ? 'read-only-storage' : 'storage' },
    });
    const uniform = (i: number): GPUBindGroupLayoutEntry => ({
      binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' },
    });
    // Explicit layouts throughout -- see growRegions.ts's getPipelines for what
    // `layout: 'auto'` cost us when one buffer set is shared across entry
    // points whose binding usage differs.
    const buildLayout = device.createBindGroupLayout({
      label: 'decodeGridBuild', entries: [uniform(0), storage(1, true), storage(2, false), storage(3, false)],
    });
    const correctLayout = device.createBindGroupLayout({
      label: 'decodeCorrectness', entries: [uniform(0), storage(1, true), storage(2, true), storage(3, false)],
    });
    const buildModule = device.createShaderModule({ code: DECODE_GRID_BUILD_WGSL, label: 'decodeGridBuild' });
    const correctModule = device.createShaderModule({ code: DECODE_CORRECTNESS_WGSL, label: 'decodeCorrectness' });
    p = {
      buildLayout,
      build: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [buildLayout] }),
        compute: { module: buildModule, entryPoint: 'build' }, label: 'decodeGridBuild.build',
      }),
      correctLayout,
      correctness: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [correctLayout] }),
        compute: { module: correctModule, entryPoint: 'correctness' }, label: 'decodeGridBuild.correctness',
      }),
    };
    pipelineCache.set(device, p);
  }
  return p;
}

// The De Bruijn torus as a flat u32 0/1 array. Cached per device like
// decodeTally.ts's hash table -- it's ~256KB for the default 256x256 board and
// only changes when the board-size slider rebuilds the pattern, which is what
// the generation counter guards.
const torusCache = new WeakMap<GPUDevice, { buf: GPUBuffer; r: number; c: number; sample: number }>();
function getTorusBuffer(device: GPUDevice): GPUBuffer {
  const cached = torusCache.get(device);
  // R/C changing is the coarse signal; torus[0][0] catches a same-size rebuild.
  if (cached && cached.r === R && cached.c === C && cached.sample === torus[0][0]) return cached.buf;
  cached?.buf.destroy();
  const flat = new Uint32Array(R * C);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) flat[r * C + c] = torus[r][c];
  const buf = uploadUint32(device, flat, 0, 'decode:torus', 'decode.fused');
  torusCache.set(device, { buf, r: R, c: C, sample: torus[0][0] });
  return buf;
}

// ── decode.build, ENCODE-ONLY ────────────────────────────────────────────
//
// One thread per lattice cell. Allocates its two outputs from the caller's arena
// and encodes one pass; no submit, no await. `gray` arrives as a binding
// resource rather than a slice because it may be the LSD chain's own arena slice
// or a standalone upload this stage's caller made -- see buildAndTallyDecodeGPU.
function encodeDecodeBuild(
  arena: Arena, alloc: Alloc, enc: GPUCommandEncoder,
  inp: { gray: GPUBindingResource; layout: DecodeGridLayout; w: number; h: number },
): { packed: Slice; geom: Slice } {
  const { gray, layout, w, h } = inp;
  const device = arena.device;
  const p = getPipelines(device);
  const cells = layout.rows * layout.cols;

  const packed = alloc(cells * 4, 'decode.packed');
  const geom = alloc(cells * 16, 'decode.geom');
  const uni = alloc(128, 'decode.buildUniforms');
  const r = sliceRange(uni, arena);
  device.queue.writeBuffer(r.buffer, r.offset, buildUniforms(layout, w, h));

  const pass = enc.beginComputePass(gpuTimelineSlot('decode:build'));
  pass.setPipeline(p.build);
  pass.setBindGroup(0, device.createBindGroup({
    layout: p.buildLayout,
    entries: [
      { binding: 0, resource: bind(uni, arena) },
      { binding: 1, resource: gray },
      { binding: 2, resource: bind(packed, arena) },
      { binding: 3, resource: bind(geom, arena) },
    ],
  }));
  pass.dispatchWorkgroups(dispatchCount(layout.rows), dispatchCount(layout.cols));
  pass.end();
  return { packed, geom };
}

// ── decode.correctness, ENCODE-ONLY ──────────────────────────────────────
//
// Reduces the whole rotated grid to the two counts behind `consistency`, adding
// them into the result block the argmax already wrote the winner into.
//
// INDIRECT, and that is the second half of what removed the fence. This pass's
// extent depends on the winning orientation (rows and cols swap at 1 and 3), so
// even with the winner in a buffer the host would still have had to know it to
// size the dispatch. The argmax writes those workgroup counts too.
function encodeDecodeCorrectness(
  arena: Arena, alloc: Alloc, enc: GPUCommandEncoder,
  inp: { packed: Slice; result: Slice; dispatchArgs: Slice; rows: number; cols: number },
): void {
  const { packed, result, dispatchArgs, rows, cols } = inp;
  const device = arena.device;
  const p = getPipelines(device);

  const uni = alloc(32, 'decode.correctnessUniforms');
  {
    const b = new ArrayBuffer(32);
    const dv = new DataView(b);
    dv.setUint32(0, rows, true); dv.setUint32(4, cols, true);
    dv.setUint32(16, R, true); dv.setUint32(20, C, true);
    const r = sliceRange(uni, arena);
    device.queue.writeBuffer(r.buffer, r.offset, b);
  }

  const pass = enc.beginComputePass(gpuTimelineSlot('decode:correctness'));
  pass.setPipeline(p.correctness);
  pass.setBindGroup(0, device.createBindGroup({
    layout: p.correctLayout,
    entries: [
      { binding: 0, resource: bind(uni, arena) },
      { binding: 1, resource: bind(packed, arena) },
      { binding: 2, resource: { buffer: getTorusBuffer(device) } },
      { binding: 3, resource: bind(result, arena) },
    ],
  }));
  const args = sliceRange(dispatchArgs, arena);
  pass.dispatchWorkgroupsIndirect(args.buffer, args.offset);
  pass.end();
}

function buildUniforms(layout: DecodeGridLayout, w: number, h: number): ArrayBuffer {
  const buf = new ArrayBuffer(128);
  const f = new Float32Array(buf), u32 = new Uint32Array(buf), i32 = new Int32Array(buf);
  const { Drow, Dcol, normal, invQuat } = layout;
  f[0] = Drow.x; f[1] = Drow.y; f[2] = Drow.z;
  f[4] = Dcol.x; f[5] = Dcol.y; f[6] = Dcol.z;
  f[8] = normal.x; f[9] = normal.y; f[10] = normal.z;
  f[12] = invQuat.x; f[13] = invQuat.y; f[14] = invQuat.z; f[15] = invQuat.w;
  f[16] = layout.distance; f[17] = Math.tan(layout.halfV); f[18] = layout.aspect; f[19] = layout.minGrazingCos;
  f[20] = layout.uPhase; f[21] = layout.vPhase; f[22] = GRID_STEP; f[23] = layout.binThreshold;
  u32[24] = layout.rows; u32[25] = layout.cols; u32[26] = w; u32[27] = h;
  i32[28] = layout.kMinU; i32[29] = layout.kMinV;
  return buf;
}

interface FusedDecodeResult {
  winner: VoteResult;
  correctCount: number;
  wrongCount: number;
  // Materializes the full DecodeSampleGrid by reading gridGeom + gridPacked
  // back. NOTHING on the pose critical path calls this -- it exists for the
  // projected-cam overlay and the phone's AR readout, which is the entire
  // reason the grid is split across two buffers. Calling it is what turns a
  // ~24-byte readback into a ~1.5MB one, so call it only when a view is
  // actually going to draw the result.
  //
  // There is no `release` beside it any more, and its absence is the point.
  // The grid lives in ARENA SLICES now, so it is freed by the next
  // reconstruction's reset along with everything else -- which deletes the
  // second deferral handle in the library (the first was PendingIntermediates)
  // and, with it, the rule that readGrid-after-release must throw. A caller
  // that reads too late gets a StaleSliceError naming the slice, from the one
  // mechanism that guards every other buffer here.
  readGrid: () => Promise<DecodeSampleGrid>;
}

// The whole GPU decode: build -> tally -> argmax -> correctness, in ONE encoder,
// ONE submit, and ONE 32-byte readback.
//
// ── What "fused" used to mean, and what it means now ──
//
// It never meant one submit. It meant one FUNCTION, and inside it were three
// submits with awaits between them, because the host sat in the middle of the
// chain twice: once to argmax the histogram (331KB down, a winner back up in a
// uniform) and once to size the correctness dispatch from the winning
// orientation. Both of those are on device now (decodeTally.wgsl.ts's argmax,
// and the indirect args it writes), so the four stages are four encode calls in
// sequence and the only fence left is reading the answer.
//
// The stages are separate functions rather than one kernel, which is the point:
// the packed grid still never crosses the bus -- passes in one encoder share
// device memory for free -- so un-fusing them cost nothing and each is now
// something a harness can call on its own.
//
// What comes back on the pose path is the result block: the winner (5 numbers)
// and the two correctness counts. The reference point's u/v are recomputed on
// the host from `layout` (pure arithmetic in i/j, see decodeGridCellUV) rather
// than read back.
//
// Returns null if WebGPU is unavailable, a dispatch failed validation, or no
// anchor scored a vote; the caller falls back to the CPU pair, which stays the
// source of truth.
//
// `sharedGray`, when given, is a gray buffer ALREADY on the device -- the LSD
// chain's own, handed down by pose/poseCompute.ts. Passing it skips a second
// 1.19MB upload of numbers that are already sitting in GPU memory, plus the
// f64->f32 narrowing loop in front of it; measured together at ~0.8ms of the
// ~2.1ms of byte-proportional cost in the whole reconstruction, i.e. this one
// duplicate was about a third of it. Ownership does NOT transfer -- a shared
// slice belongs to the arena, and only the buffer this function uploaded itself
// gets dropped.
export async function buildAndTallyDecodeGPU(
  layout: DecodeGridLayout, gray: Float64Array, w: number, h: number,
  // The LSD chain's device-resident gray, as an ARENA SLICE rather than a bare
  // buffer -- so the binding goes through bind() and inherits the generation
  // check. A slice from a previous run throws here instead of feeding the decode
  // another capture's pixels.
  sharedGray?: { arena: Arena; slice: Slice } | null,
): Promise<FusedDecodeResult | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  device.pushErrorScope('validation');

  const { rows, cols } = layout;
  const cells = rows * cols;
  // The SAME arena the LSD chain used, deliberately NOT reset here: decode runs
  // after the chain within one reconstruction, so it appends to that frame's
  // allocation and both are freed together at the next run.
  const arena = poseArena(device);
  const alloc = arena.alloc;
  const ownGray = sharedGray ? null : uploadFloat32(device, new Float32Array(gray), 0, 'decode:gray', 'decode.fused');
  const grayResource: GPUBindingResource = sharedGray
    ? bind(sharedGray.slice, sharedGray.arena)
    : { buffer: ownGray! };

  // The four stages, in order, into one encoder. Each pass sees the previous
  // pass's writes -- WebGPU orders passes within a submit -- which is the whole
  // reason none of this has to come back to the host in between.
  const enc = device.createCommandEncoder();
  const { packed: packedBuf, geom: geomBuf } = encodeDecodeBuild(arena, alloc, enc, { gray: grayResource, layout, w, h });
  const { hist, totalWindows } = encodeDecodeTally(arena, alloc, enc, { grid: packedBuf, gr: rows, gc: cols });
  const { result, dispatchArgs } = encodeDecodeArgmax(arena, alloc, enc, { hist, totalWindows, gr: rows, gc: cols });
  encodeDecodeCorrectness(arena, alloc, enc, { packed: packedBuf, result, dispatchArgs, rows, cols });
  device.queue.submit([enc.finish()]);
  // Only ever OUR copy -- a shared slice belongs to the arena. Safe the moment
  // the commands referencing it are submitted.
  ownGray?.destroy();

  const raw = await readSliceU32(arena, result, DECODE_RESULT_BYTES, 'decode:result', 'decode.fused');

  // Checked BEFORE the result is believed, and it has to be: a validation
  // failure makes every command that used the offending bind group a silent
  // no-op, so the block would read back all zeros -- which is exactly what an
  // ordinary undecodable frame looks like. Without this, a real bug would
  // present as "nothing decoded", quietly, and be papered over by the CPU
  // fallback.
  const err = await device.popErrorScope();
  if (err) {
    console.error('buildAndTallyDecodeGPU: WebGPU validation error, falling back to CPU --', err.message);
    return null;
  }
  const unpacked = unpackDecodeResult(raw);
  if (!unpacked) return null; // no anchor scored a vote -- an undecodable frame
  const { winner, correctCount, wrongCount } = unpacked;

  {
    const readGrid = async (): Promise<DecodeSampleGrid> => {
      const [geom, packed] = await Promise.all([
        readSliceF32(arena, geomBuf, cells * 16, 'decode:gridGeom', 'pose.drain'),
        readSliceU32(arena, packedBuf, cells * 4, 'decode:gridPacked', 'pose.drain'),
      ]);
      const points: DecodeSamplePoint[][] = [];
      for (let i = 0; i < rows; i++) {
        const row: DecodeSamplePoint[] = [];
        for (let j = 0; j < cols; j++) {
          const idx = i * cols + j, g = idx * 4, pk = packed[idx];
          row.push({
            u: geom[g], v: geom[g + 1], px: geom[g + 2], py: geom[g + 3],
            valid: (pk & 1) === 1, bit: (pk >> 1) & 1,
          });
        }
        points.push(row);
      }
      return { rows, cols, zeroI: layout.zeroI, zeroJ: layout.zeroJ, points };
    };
    return { winner, correctCount, wrongCount, readGrid };
  }
}
