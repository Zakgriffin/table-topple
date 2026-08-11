import { ORDER, R, C, debruijnLookup } from '../../../sphereLab/floorPattern.ts';
import { rotatedDims } from './decodeGrid.ts';
import { type DecodeSampleGrid, type VoteResult } from '../../results.ts';
import { dispatchCount, getGPUDevice, readSliceU32, uploadUint32, writeSlice } from '../../gpu/device.ts';
import { type Alloc, type Arena, type Slice, allocZeroed, bind, sliceRange } from '../../gpu/arena.ts';
import { poseArena } from '../lsd/chain.ts';
import { gpuTimelineSlot } from '../../gpu/gpuTimeline.ts';
import { DECODE_ARGMAX_WGSL, DECODE_TALLY_WGSL } from './decodeTally.wgsl.ts';

// The correctness kernel's workgroup edge (it is @workgroup_size(8, 8)), needed
// HERE because the argmax computes that pass's workgroup counts on device. Same
// value dispatchCount uses; passed through the uniform rather than hard-coded in
// the shader so the two cannot drift apart silently.
const CORRECTNESS_WG = 8;

const NOT_FOUND = 0xffffffff;

// Same 32-bit finisher as decodeTally.wgsl.ts's hashU32 -- Math.imul wraps
// mod 2^32 identically to WGSL's u32 multiply, so this MUST stay byte-for-
// byte in sync with the WGSL version, or GPU lookups silently miss entries
// the CPU-built table actually has.
function hashU32(x: number): number {
  x = x >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

// debruijnLookup never changes at runtime (it's the shared floor pattern's
// static decode table) -- built into a flat open-addressing hash table once
// per device and cached forever after, same reasoning as positionLM.ts's
// torus-brightness buffer cache. Load factor 0.5 (table sized to 2x entry
// count, rounded up to a power of two) keeps linear-probe chains short.
interface HashTable { keysBuf: GPUBuffer; valuesBuf: GPUBuffer; size: number }
let hashTableCache = new WeakMap<GPUDevice, HashTable>();
// Call whenever debruijnLookup actually changes (currently: the "De Bruijn
// board size" global slider, see ui/cameraPanel.ts) -- the assumption this
// cache was built on (comment above) no longer holds once that's possible
// at runtime. Existing GPUBuffers held by any cached table are intentionally
// left to GC, not explicitly destroy()'d -- they're tiny and destroying a
// buffer that a still-in-flight GPU command might reference is worse than
// leaking it briefly.
export function invalidateHashTableCache() {
  hashTableCache = new WeakMap<GPUDevice, HashTable>();
}
function getHashTable(device: GPUDevice): HashTable {
  let table = hashTableCache.get(device);
  if (table) return table;

  const entries = Array.from(debruijnLookup.entries());
  const size = nextPowerOfTwo(entries.length * 2);
  const keys = new Uint32Array(size).fill(0);
  const values = new Uint32Array(size).fill(NOT_FOUND);
  for (const [key, value] of entries) {
    let slot = hashU32(key) % size;
    while (values[slot] !== NOT_FOUND) slot = (slot + 1) % size;
    keys[slot] = key;
    values[slot] = value;
  }
  const keysBuf = uploadUint32(device, keys, 0, 'tally:keys', 'decode.fused');
  const valuesBuf = uploadUint32(device, values, 0, 'tally:values', 'decode.fused');
  table = { keysBuf, valuesBuf, size };
  hashTableCache.set(device, table);
  return table;
}

interface Pipelines { tally: GPUComputePipeline; argmax: GPUComputePipeline }
const pipelineCache = new WeakMap<GPUDevice, Pipelines>();
function getPipelines(device: GPUDevice): Pipelines {
  let p = pipelineCache.get(device);
  if (!p) {
    const tallyModule = device.createShaderModule({ code: DECODE_TALLY_WGSL, label: 'decodeTally' });
    const argmaxModule = device.createShaderModule({ code: DECODE_ARGMAX_WGSL, label: 'decodeArgmax' });
    p = {
      tally: device.createComputePipeline({
        layout: 'auto', compute: { module: tallyModule, entryPoint: 'main' }, label: 'decodeTally',
      }),
      argmax: device.createComputePipeline({
        layout: 'auto', compute: { module: argmaxModule, entryPoint: 'argmax' }, label: 'decodeArgmax',
      }),
    };
    pipelineCache.set(device, p);
  }
  return p;
}

// ── THE DECODE RESULT BLOCK ──────────────────────────────────────────────
//
// Eight u32 shared by the argmax pass (which writes 0..5) and the correctness
// pass (which adds into 6..7), and the ONLY thing decode reads back on the pose
// path. It is one buffer rather than three so the whole stage costs one map
// instead of three -- the readbacks it replaced were the 331KB histogram, a
// 4-byte totalWindows and an 8-byte counts pair, each behind its own fence.
//
// `found` is slot 0 because "no anchor got a single vote" is an ordinary
// outcome, not an error: the host returns null and the caller decodes on CPU.
export const DECODE_RESULT_SLOTS = 8;
export const DECODE_RESULT_BYTES = DECODE_RESULT_SLOTS * 4;
export const RESULT_FOUND = 0, RESULT_ORIENT = 1, RESULT_ANCHOR_ROW = 2, RESULT_ANCHOR_COL = 3;
export const RESULT_VOTES = 4, RESULT_TOTAL_WINDOWS = 5, RESULT_CORRECT = 6, RESULT_WRONG = 7;

// The decode result read out of the block above, once. Slots 6/7 are only
// meaningful when a correctness pass was encoded after the argmax -- the unfused
// entry point does not encode one, and leaves them zero.
export function unpackDecodeResult(raw: Uint32Array): {
  winner: VoteResult; correctCount: number; wrongCount: number;
} | null {
  if (raw[RESULT_FOUND] === 0) return null;
  return {
    winner: {
      orientation: raw[RESULT_ORIENT], anchorRow: raw[RESULT_ANCHOR_ROW], anchorCol: raw[RESULT_ANCHOR_COL],
      votes: raw[RESULT_VOTES], totalWindows: raw[RESULT_TOTAL_WINDOWS],
    },
    correctCount: raw[RESULT_CORRECT], wrongCount: raw[RESULT_WRONG],
  };
}

function buildUniforms(gr: number, gc: number, orient: number, tableSize: number): ArrayBuffer {
  const buf = new ArrayBuffer(32);
  const dv = new DataView(buf);
  dv.setUint32(0, gr, true); dv.setUint32(4, gc, true); dv.setUint32(8, orient, true); dv.setUint32(12, ORDER, true);
  dv.setUint32(16, R, true); dv.setUint32(20, C, true); dv.setUint32(24, tableSize, true);
  return buf;
}

// GPU-resident counterpart to pose/stages/decode/decodeGrid.ts's tallyPositionVotes --
// see decodeTally.wgsl.ts's header for the design (dense u32 atomic
// histogram over the bounded (orientation, anchorRow, anchorCol) key space,
// no float-atomics workaround needed since this is pure counting). Returns
// null if WebGPU isn't available; caller falls back to the CPU version,
// which stays the source of truth.
//
// The crossover this used to speculate about has now been MEASURED (live, over
// the dev bridge, simulated camera, 2026-08-02) and the prediction held:
//
//        grid          this path   tallyPositionVotes (CPU)
//        60x59           1.60ms      0.85ms
//        270x276         1.86ms     18.20ms
//
// i.e. this is essentially FLAT in grid size while the CPU version is linear,
// so it wins ~10x as soon as the grid is large and costs under a millisecond
// when it isn't. Winners agreed exactly at both sizes. That is why
// this runs as the fused decode's fallback.
//
// An older version of this comment claimed ~40-70ms here vs ~0.4-1ms on CPU,
// off a much smaller (~22x29) saved capture. Those numbers do not reproduce --
// the readback latency it blamed (a "4-byte readback observed anywhere from
// ~5ms to ~55ms") is not visible at all now. Treat that note as superseded.
export async function tallyPositionVotesGPU(grid: DecodeSampleGrid): Promise<VoteResult | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const gr = grid.rows, gc = grid.cols;
  const gridData = new Uint32Array(gr * gc);
  for (let i = 0; i < gr; i++) {
    for (let j = 0; j < gc; j++) {
      const pt = grid.points[i][j];
      gridData[i * gc + j] = pt.valid ? (1 | (pt.bit << 1)) : 0;
    }
  }
  // Into an arena slice like everything else, so the two entry points hand
  // `tallyFromDeviceGrid` the same kind of thing. Freed by the next
  // reconstruction's reset rather than by a finally.
  const arena = poseArena(device);
  const gridSlice = arena.alloc(gridData.byteLength, 'tally.gridUpload');
  writeSlice(arena, gridSlice, gridData, 'tally:grid', 'decode.fused');
  return await tallyFromDeviceGrid(device, arena, gridSlice, gr, gc);
}

// ── The tally, ENCODE-ONLY ───────────────────────────────────────────────
//
// Four passes over a grid buffer this function does NOT own, into a histogram it
// allocates from the caller's arena. No submit, no await, no readback -- which
// is what lets decodeGridBuild.gpu.ts put build, tally, argmax and correctness
// in ONE encoder.
//
// Split from the grid build so the packed grid never crosses the bus: at a
// 270x276 lattice that upload is 298KB per call, and it was the single biggest
// reason these were fused. Under encode-only stages sharing an encoder the
// saving is free, so the fusion is gone and the saving stayed.
export function encodeDecodeTally(
  arena: Arena, alloc: Alloc, enc: GPUCommandEncoder,
  inp: { grid: Slice; gr: number; gc: number },
): { hist: Slice; totalWindows: Slice } {
  const { grid, gr, gc } = inp;
  const device = arena.device;
  const pipeline = getPipelines(device).tally;
  const { keysBuf, valuesBuf, size: tableSize } = getHashTable(device);

  // ZEROED EXPLICITLY. Both are atomicAdd targets, and both used to rely on
  // createBuffer returning zeroes -- the comment here said so. An arena slice is
  // last frame's bytes, so without these clears the histogram would accumulate
  // across reconstructions and every frame would decode to the previous frame's
  // winner with a growing vote count. See allocZeroed's header in arena.ts.
  const hist = allocZeroed(arena, alloc, enc, 4 * R * C * 4, 'tally.hist');
  const totalWindows = allocZeroed(arena, alloc, enc, 4, 'tally.totalWindows');

  for (let o = 0; o < 4; o++) {
    const [rr, cc] = rotatedDims(gr, gc, o);
    // The uniforms come out of the ARENA rather than a per-call createBuffer,
    // which is what makes this stage encode-only: a buffer created here would
    // have to outlive a submit this function does not make, so there would be
    // nowhere to destroy it. The arena frees them at the next reset with
    // everything else.
    const uni = alloc(32, 'tally.uniforms');
    const r = sliceRange(uni, arena);
    device.queue.writeBuffer(r.buffer, r.offset, buildUniforms(gr, gc, o, tableSize));
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bind(uni, arena) },
        { binding: 1, resource: bind(grid, arena) },
        { binding: 2, resource: { buffer: keysBuf } },
        { binding: 3, resource: { buffer: valuesBuf } },
        { binding: 4, resource: bind(hist, arena) },
        { binding: 5, resource: bind(totalWindows, arena) },
      ],
    });
    // One label for all four orientations rather than `tally:o0..o3`: they are
    // the same kernel over the same grid at four rotations, so gpuTimeline's
    // per-label aggregation reports them as `tally:orient x4` -- which is the
    // useful reading, and four rows of one quantum each is not (see
    // gpuTimeline.ts on the resolution limit).
    const pass = enc.beginComputePass(gpuTimelineSlot('tally:orient'));
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchCount(rr), dispatchCount(cc));
    pass.end();
    // Every uniform is a distinct slice, so all four stay valid until the
    // submit. Reusing one and rewriting it between passes would be a
    // write-after-encode race with commands that have not run yet.
  }
  return { hist, totalWindows };
}

// ── The argmax, ENCODE-ONLY ──────────────────────────────────────────────
//
// The stage that used to run on the host on both backends. It writes the winner
// into the shared result block and the correctness pass's workgroup counts into
// `dispatchArgs`, so nothing about "which anchor won" or "how big is the rotated
// grid" has to come back to the CPU and be sent down again.
//
// `gr`/`gc` are only for those dispatch args -- the argmax itself is a reduction
// over the histogram and knows nothing about the lattice.
export function encodeDecodeArgmax(
  arena: Arena, alloc: Alloc, enc: GPUCommandEncoder,
  inp: { hist: Slice; totalWindows: Slice; gr: number; gc: number },
): { result: Slice; dispatchArgs: Slice } {
  const { hist, totalWindows, gr, gc } = inp;
  const device = arena.device;
  const pipeline = getPipelines(device).argmax;

  // ZEROED: slots 6/7 are the correctness pass's atomicAdd targets. The argmax
  // itself fully writes 0..5, so only the tail actually needs it -- but the
  // whole block is one slice and one clear, and a partial clear here would be a
  // trap for whoever adds slot 8.
  const result = allocZeroed(arena, alloc, enc, DECODE_RESULT_BYTES, 'decode.result');
  // Zeroed too, though this pass fully writes it. The case it covers is this
  // pass NOT RUNNING -- a bind group that failed validation makes every command
  // using it a silent no-op, and the indirect dispatch that reads these args is
  // a different command with its own valid buffer. Without the clear it would
  // launch over the PREVIOUS reconstruction's extent.
  const dispatchArgs = allocZeroed(arena, alloc, enc, 12, 'decode.correctnessArgs');

  const uni = alloc(32, 'argmax.uniforms');
  {
    const b = new ArrayBuffer(32);
    const dv = new DataView(b);
    dv.setUint32(0, 4 * R * C, true);
    dv.setUint32(4, R, true); dv.setUint32(8, C, true);
    dv.setUint32(12, gr, true); dv.setUint32(16, gc, true);
    dv.setUint32(20, CORRECTNESS_WG, true);
    const r = sliceRange(uni, arena);
    device.queue.writeBuffer(r.buffer, r.offset, b);
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: bind(uni, arena) },
      { binding: 1, resource: bind(hist, arena) },
      { binding: 2, resource: bind(totalWindows, arena) },
      { binding: 3, resource: bind(result, arena) },
      { binding: 4, resource: bind(dispatchArgs, arena) },
    ],
  });
  const pass = enc.beginComputePass(gpuTimelineSlot('decode:argmax'));
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1); // one workgroup by design -- see DECODE_ARGMAX_WGSL
  pass.end();
  return { result, dispatchArgs };
}

// The unfused route's tally: encode, submit, read the 32-byte result. Used when
// the grid came from the CPU builder, so there is nothing to share an encoder
// with. It reads the SAME block the fused path does and simply leaves the
// correctness slots at zero, because it encodes no correctness pass.
//
// What it no longer does is read the histogram. That was 4*R*C u32 -- 331KB at
// the default board -- brought down so the host could scan it for a maximum.
export async function tallyFromDeviceGrid(
  device: GPUDevice, arena: Arena, gridSlice: Slice, gr: number, gc: number,
): Promise<VoteResult | null> {
  // Scoped HERE rather than in the entry point above, so a validation failure
  // becomes a null and the caller's CPU fallback, in one place.
  device.pushErrorScope('validation');
  const enc = device.createCommandEncoder();
  const { hist, totalWindows } = encodeDecodeTally(arena, arena.alloc, enc, { grid: gridSlice, gr, gc });
  const { result } = encodeDecodeArgmax(arena, arena.alloc, enc, { hist, totalWindows, gr, gc });
  device.queue.submit([enc.finish()]);

  const raw = await readSliceU32(arena, result, DECODE_RESULT_BYTES, 'decode:result', 'decode.tally');

  // Checked BEFORE the result is believed. An all-zero block reads as "no
  // anchor won", which is an ordinary undecodable frame -- so without this a
  // validation failure would be indistinguishable from one, silently, with
  // nothing in the console.
  const err = await device.popErrorScope();
  if (err) {
    console.error('tallyFromDeviceGrid: WebGPU validation error, falling back to CPU --', err.message);
    return null;
  }
  return unpackDecodeResult(raw)?.winner ?? null;
}
