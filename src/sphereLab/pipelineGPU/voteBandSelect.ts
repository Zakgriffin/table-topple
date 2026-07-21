import * as THREE from 'three';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { Vote } from '../types.ts';
import { createStorageBuffer, getGPUDevice, readFloat32, readUint32, uploadFloat32, uploadUniform } from './device.ts';
import { COMPACT_FILTER_WGSL, HISTOGRAM_WGSL } from './voteBandSelect.wgsl.ts';

// 4096 buckets over ~270k votes averages ~66 votes/bucket -- more than
// precise enough for a weight-sharpening heuristic filter, see this file's
// .wgsl.ts header for why exact-rank precision was never actually needed.
const NUM_BUCKETS = 4096;
const WORKGROUP_SIZE_1D = 64;

const histPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();
function getHistPipeline(device: GPUDevice): GPUComputePipeline {
  let p = histPipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: HISTOGRAM_WGSL, label: 'voteBandHistogram' });
    p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: 'voteBandHistogram' });
    histPipelineCache.set(device, p);
  }
  return p;
}

const filterPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();
function getFilterPipeline(device: GPUDevice): GPUComputePipeline {
  let p = filterPipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: COMPACT_FILTER_WGSL, label: 'voteBandCompactFilter' });
    p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: 'voteBandCompactFilter' });
    filterPipelineCache.set(device, p);
  }
  return p;
}

function votesToFloat32(votes: Vote[]): Float32Array {
  const data = new Float32Array(votes.length * 4);
  for (let i = 0; i < votes.length; i++) {
    const o = i * 4;
    data[o] = votes[i].n.x; data[o + 1] = votes[i].n.y; data[o + 2] = votes[i].n.z; data[o + 3] = votes[i].weight;
  }
  return data;
}

// GPU-resident counterpart to pipeline/votes.ts's votesInMagnitudeBand --
// see voteBandSelect.wgsl.ts's header for the two-pass histogram+compact
// design. Returns null if WebGPU isn't available; caller falls back to the
// CPU version, which stays the source of truth.
export async function votesInMagnitudeBandGPU(votes: Vote[], minPercent: number, maxPercent: number): Promise<Vote[] | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const n = votes.length;
  if (n === 0) return [];

  let maxWeight = 0;
  for (const { weight } of votes) if (weight > maxWeight) maxWeight = weight;

  const votesBuf = uploadFloat32(device, votesToFloat32(votes));
  const numWorkgroups = Math.ceil(n / WORKGROUP_SIZE_1D);

  // ── Pass 1: histogram ────────────────────────────────────────────────
  const histPipeline = getHistPipeline(device);
  const histBuf = createStorageBuffer(device, NUM_BUCKETS * 4); // zero-initialized per WebGPU spec
  const histUniformData = new ArrayBuffer(16);
  {
    const dv = new DataView(histUniformData);
    dv.setUint32(0, n, true); dv.setUint32(4, NUM_BUCKETS, true); dv.setFloat32(8, maxWeight, true);
  }
  const histUniformBuf = uploadUniform(device, histUniformData);
  const histBindGroup = device.createBindGroup({
    layout: histPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: histUniformBuf } },
      { binding: 1, resource: { buffer: votesBuf } },
      { binding: 2, resource: { buffer: histBuf } },
    ],
  });
  const histSpan = spanStart('GPU dispatch (histogram)');
  {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(histPipeline);
    pass.setBindGroup(0, histBindGroup);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
  spanEnd(histSpan);
  const histRaw = await readUint32(device, histBuf, NUM_BUCKETS * 4);

  // ── CPU: translate the two rank cutoffs into weight thresholds by
  // walking the (tiny) histogram from the highest-weight bucket down. ────
  const scanSpan = spanStart('CPU (histogram scan)');
  const lo = Math.round(n * (minPercent / 100));
  const hi = Math.round(n * (maxPercent / 100));
  let cum = 0;
  let hiThresh = maxWeight, loThresh = 0;
  let hiSet = false;
  for (let b = NUM_BUCKETS - 1; b >= 0; b--) {
    const c = histRaw[b];
    if (!hiSet && cum + c >= lo) { hiThresh = ((b + 1) / NUM_BUCKETS) * maxWeight; hiSet = true; }
    cum += c;
    if (cum >= hi) { loThresh = (b / NUM_BUCKETS) * maxWeight; break; }
  }
  spanEnd(scanSpan);

  if (hi <= lo) {
    for (const b of [votesBuf, histBuf, histUniformBuf]) b.destroy();
    return [];
  }

  // ── Pass 2: compact filter (real stream compaction via an atomic index
  // counter -- integer, so no float-atomics workaround needed here either) ─
  const filterPipeline = getFilterPipeline(device);
  const votesOutBuf = createStorageBuffer(device, n * 16); // upper bound; only outCount*16 bytes ever get read back
  const outCountBuf = createStorageBuffer(device, 4); // zero-initialized
  const filterUniformData = new ArrayBuffer(16);
  {
    const dv = new DataView(filterUniformData);
    dv.setUint32(0, n, true); dv.setFloat32(8, loThresh, true); dv.setFloat32(12, hiThresh, true);
  }
  const filterUniformBuf = uploadUniform(device, filterUniformData);
  const filterBindGroup = device.createBindGroup({
    layout: filterPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: filterUniformBuf } },
      { binding: 1, resource: { buffer: votesBuf } },
      { binding: 2, resource: { buffer: votesOutBuf } },
      { binding: 3, resource: { buffer: outCountBuf } },
    ],
  });
  const filterSpan = spanStart('GPU dispatch (compact filter)');
  {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(filterPipeline);
    pass.setBindGroup(0, filterBindGroup);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
  spanEnd(filterSpan);

  const outCountRaw = await readUint32(device, outCountBuf, 4);
  const count = outCountRaw[0];
  const outRaw = count > 0 ? await readFloat32(device, votesOutBuf, count * 16) : new Float32Array(0);

  for (const b of [votesBuf, histBuf, histUniformBuf, votesOutBuf, outCountBuf, filterUniformBuf]) b.destroy();

  const result: Vote[] = new Array(count);
  for (let i = 0; i < count; i++) {
    const o = i * 4;
    result[i] = { n: new THREE.Vector3(outRaw[o], outRaw[o + 1], outRaw[o + 2]), weight: outRaw[o + 3] };
  }
  return result;
}
