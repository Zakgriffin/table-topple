import { GRID_STEP } from '../../../sphereLab/constants.ts';
import { C, R, torus } from '../../../sphereLab/floorPattern.ts';
import { type DecodeGridLayout } from './decodeGrid.ts';
import { type DecodeSampleGrid, type DecodeSamplePoint, type VoteResult } from '../../results.ts';
import { tallyFromDeviceGrid } from './decodeTally.gpu.ts';
import {
  createStorageBuffer, dispatchCount, getGPUDevice, readUint32,
  uploadFloat32, uploadUint32, uploadUniform,
} from '../../gpu/device.ts';
import { type Arena, type Slice, bind } from '../../gpu/arena.ts';
import { readSliceF32, readSliceU32 } from '../../gpu/device.ts';
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

// Fused GPU counterpart to buildDecodeSampleGrid + tallyPositionVotes.
//
// The grid is BUILT on device and the tally consumes it in place, so the packed
// grid never crosses the bus -- at a 270x276 lattice that upload alone is 298KB
// per call, and it was the single biggest reason to fuse these rather than port
// them separately. What comes back on the pose path is the winner (5 numbers)
// and two correctness counts. The reference point's u/v are recomputed on the
// host from `layout` (pure arithmetic in i/j, see decodeGridCellUV) rather than
// read back.
//
// Returns null if WebGPU is unavailable or a dispatch failed validation; the
// caller falls back to the CPU pair, which stays the source of truth.
// `sharedGray`, when given, is a gray buffer ALREADY on the device -- the LSD
// chain's own, handed down by pose/poseCompute.ts. Passing it skips a second
// 1.19MB upload of numbers that are already sitting in GPU memory, plus the
// f64->f32 narrowing loop in front of it; measured together at ~0.8ms of the
// ~2.1ms of byte-proportional cost in the whole reconstruction, i.e. this one
// duplicate was about a third of it.
//
// Ownership does NOT transfer: a shared buffer belongs to the residency and is
// destroyed with it, so this must not destroy it. Only the buffer it uploaded
// itself gets dropped below.
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
  const p = getPipelines(device);

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
  const packedBuf = alloc(cells * 4, 'decode.packed');
  const geomBuf = alloc(cells * 16, 'decode.geom');
  const uniBuf = uploadUniform(device, buildUniforms(layout, w, h), 'decode.fused');

  {
    const bg = device.createBindGroup({
      layout: p.buildLayout,
      entries: [
        { binding: 0, resource: { buffer: uniBuf } },
        { binding: 1, resource: grayResource },
        { binding: 2, resource: bind(packedBuf, arena) },
        { binding: 3, resource: bind(geomBuf, arena) },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(p.build);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(dispatchCount(rows), dispatchCount(cols));
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
  // gray is only needed by the build pass; drop it before the tally so the
  // largest buffer in flight isn't held across the rest of the sequence. Only
  // ever OUR copy -- a shared buffer is the residency's to free.
  ownGray?.destroy();

  const winner = await tallyFromDeviceGrid(device, arena, packedBuf, rows, cols);
  if (!winner) {
    await device.popErrorScope();
    return null;
  }

  // Correctness counts, reduced on device so `consistency` costs 8 bytes rather
  // than a grid readback.
  const countsBuf = createStorageBuffer(device, 8);
  const counts = await (async () => {
    const cu = new ArrayBuffer(32);
    const dv = new DataView(cu);
    dv.setUint32(0, rows, true); dv.setUint32(4, cols, true); dv.setUint32(8, winner.orientation, true);
    dv.setUint32(16, R, true); dv.setUint32(20, C, true);
    dv.setUint32(24, winner.anchorRow, true); dv.setUint32(28, winner.anchorCol, true);
    const cuBuf = uploadUniform(device, cu, 'decode.fused');
    const bg = device.createBindGroup({
      layout: p.correctLayout,
      entries: [
        { binding: 0, resource: { buffer: cuBuf } },
        { binding: 1, resource: bind(packedBuf, arena) },
        { binding: 2, resource: { buffer: getTorusBuffer(device) } },
        { binding: 3, resource: { buffer: countsBuf } },
      ],
    });
    const swap = winner.orientation === 1 || winner.orientation === 3;
    const rr = swap ? cols : rows, cc = swap ? rows : cols;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(p.correctness);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(dispatchCount(rr), dispatchCount(cc));
    pass.end();
    device.queue.submit([encoder.finish()]);
    const out = await readUint32(device, countsBuf, 8, 'decode:counts', 'decode.fused');
    cuBuf.destroy(); countsBuf.destroy();
    return out;
  })();

  const err = await device.popErrorScope();
  if (err) {
    console.error('buildAndTallyDecodeGPU: WebGPU validation error, falling back to CPU --', err.message);
    return null;
  }

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
    return { winner, correctCount: counts[0], wrongCount: counts[1], readGrid };
  }
}
