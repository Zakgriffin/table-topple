import { ORDER, R, C, debruijnLookup } from '../floorPattern.ts';
import { rotatedDims } from '../pipeline/decodeGrid.ts';
import { DecodeSampleGrid, VoteResult } from '../types.ts';
import { attachGPUKernelBreakdown, profilerEnabled, spanEnd, spanStart } from '../profiling/profiler.ts';
import { createStorageBuffer, createTimestampQuerySet, dispatchCount, getGPUDevice, readUint32, resolveTimestamps, supportsTimestampQuery, uploadUint32, uploadUniform } from './device.ts';
import { DECODE_TALLY_WGSL } from './decodeTally.wgsl.ts';

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
  const keysBuf = uploadUint32(device, keys);
  const valuesBuf = uploadUint32(device, values);
  table = { keysBuf, valuesBuf, size };
  hashTableCache.set(device, table);
  return table;
}

const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();
function getPipeline(device: GPUDevice): GPUComputePipeline {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: DECODE_TALLY_WGSL, label: 'decodeTally' });
    p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: 'decodeTally' });
    pipelineCache.set(device, p);
  }
  return p;
}

function buildUniforms(gr: number, gc: number, orient: number, tableSize: number): ArrayBuffer {
  const buf = new ArrayBuffer(32);
  const dv = new DataView(buf);
  dv.setUint32(0, gr, true); dv.setUint32(4, gc, true); dv.setUint32(8, orient, true); dv.setUint32(12, ORDER, true);
  dv.setUint32(16, R, true); dv.setUint32(20, C, true); dv.setUint32(24, tableSize, true);
  return buf;
}

// GPU-resident counterpart to pipeline/decodeGrid.ts's tallyPositionVotes --
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
// globalState.useGPUDecode defaults on.
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
  const gridBuf = uploadUint32(device, gridData);
  try {
    return await tallyFromDeviceGrid(device, gridBuf, gr, gc);
  } finally {
    gridBuf.destroy(); // owned here, unlike the fused path's caller-owned buffer
  }
}

// The tally proper, over a grid buffer this function does NOT own. Split out so
// pipelineGPU/decodeGridBuild.ts can hand it a grid the GPU just BUILT, instead
// of one packed and uploaded from CPU -- at a 270x276 grid that upload is 298KB
// per call, and it is the single biggest reason to fuse the two stages.
export async function tallyFromDeviceGrid(
  device: GPUDevice, gridBuf: GPUBuffer, gr: number, gc: number,
): Promise<VoteResult | null> {
  const pipeline = getPipeline(device);
  const { keysBuf, valuesBuf, size: tableSize } = getHashTable(device);

  const tallyBuf = createStorageBuffer(device, 4 * R * C * 4); // zero-initialized per WebGPU spec
  const totalWindowsBuf = createStorageBuffer(device, 4);

  const wantTimestamps = profilerEnabled() && supportsTimestampQuery(device);
  const querySet = wantTimestamps ? createTimestampQuerySet(device, 4) : null;

  const dispatchSpan = spanStart('GPU dispatch (4 orientations)');
  const encoder = device.createCommandEncoder();
  const uniformBufs: GPUBuffer[] = [];
  for (let o = 0; o < 4; o++) {
    const [rr, cc] = rotatedDims(gr, gc, o);
    const uniformBuf = uploadUniform(device, buildUniforms(gr, gc, o, tableSize));
    uniformBufs.push(uniformBuf);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: gridBuf } },
        { binding: 2, resource: { buffer: keysBuf } },
        { binding: 3, resource: { buffer: valuesBuf } },
        { binding: 4, resource: { buffer: tallyBuf } },
        { binding: 5, resource: { buffer: totalWindowsBuf } },
      ],
    });
    const pass = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: o * 2, endOfPassWriteIndex: o * 2 + 1 } } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchCount(rr), dispatchCount(cc));
    pass.end();
  }
  device.queue.submit([encoder.finish()]);
  if (querySet) {
    const durations = await resolveTimestamps(device, querySet, 4);
    attachGPUKernelBreakdown(durations.map((durationMs, o) => ({ name: `orientation ${o} kernel`, durationMs })));
    querySet.destroy();
  }
  spanEnd(dispatchSpan);

  const [tallyRaw, totalWindowsRaw] = await Promise.all([
    readUint32(device, tallyBuf, 4 * R * C * 4),
    readUint32(device, totalWindowsBuf, 4),
  ]);
  for (const b of [tallyBuf, totalWindowsBuf, ...uniformBufs]) b.destroy(); // gridBuf is the caller's

  let bestIdx = -1, bestVotes = 0;
  for (let i = 0; i < tallyRaw.length; i++) {
    if (tallyRaw[i] > bestVotes) { bestVotes = tallyRaw[i]; bestIdx = i; }
  }
  if (bestIdx < 0) return null;
  const orientation = Math.floor(bestIdx / (R * C));
  const rem = bestIdx % (R * C);
  const anchorRow = Math.floor(rem / C), anchorCol = rem % C;
  return { orientation, anchorRow, anchorCol, votes: bestVotes, totalWindows: totalWindowsRaw[0] };
}
