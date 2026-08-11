import { type PlaneTriad, planesFromScatter } from './votes.ts';
import { spanEnd } from '../../../sphereLab/profiling/profiler.ts';
import { poseSpan } from '../../timing/stages.ts';
import { type Vote } from '../../results.ts';
import { createStorageBuffer, getGPUDevice, readFloat32, uploadFloat32, uploadUniform } from '../../gpu/device.ts';
import { gpuTimelineSlot } from '../../gpu/gpuTimeline.ts';
import { FIT_PLANES_WGSL } from './fitPlanes.wgsl.ts';

const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();
function getPipeline(device: GPUDevice): GPUComputePipeline {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: FIT_PLANES_WGSL, label: 'fitPlanes' });
    p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: 'fitPlanes' });
    pipelineCache.set(device, p);
  }
  return p;
}

const WORKGROUP_SIZE_1D = 64;

// `fit.ata` on GPU, then `fit.eigen` on the host -- the SAME `planesFromScatter`
// the CPU path calls, not a copy of it. Only the scatter accumulation is
// offloaded, because only it grows with the vote count; the decomposition is
// fixed-size arithmetic and stays where it is on both backends (see
// fitPlanes.wgsl.ts's header).
//
// Returns null if WebGPU isn't available; caller falls back to the CPU version,
// which stays the source of truth.
export async function fitPairOfPlanesGPU(votes: Vote[]): Promise<PlaneTriad | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  if (votes.length === 0) return null;
  // After the early returns, so no path leaves a scope pushed. See the pop.
  device.pushErrorScope('validation');
  const pipeline = getPipeline(device);

  const n = votes.length;
  let maxW = 0;
  for (const { weight } of votes) if (weight > maxW) maxW = weight;

  const voteData = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    voteData[o] = votes[i].n.x; voteData[o + 1] = votes[i].n.y; voteData[o + 2] = votes[i].n.z; voteData[o + 3] = votes[i].weight;
  }
  const voteBuf = uploadFloat32(device, voteData, 0, 'fit:votes', 'pose.fit');

  const numWorkgroups = Math.ceil(n / WORKGROUP_SIZE_1D);
  const outBuf = createStorageBuffer(device, numWorkgroups * 21 * 4);

  const uniformData = new ArrayBuffer(8);
  const dv = new DataView(uniformData);
  dv.setUint32(0, n, true); dv.setFloat32(4, maxW, true);
  const uniformBuf = uploadUniform(device, uniformData, 'pose.fit');

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: voteBuf } },
      { binding: 2, resource: { buffer: outBuf } },
    ],
  });

  const ataSpan = poseSpan('fit.ata', { backend: 'gpu' });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(gpuTimelineSlot('fit:ATA'));
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(numWorkgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
  // Closes at SUBMIT, deliberately -- the readback stall below belongs to the
  // declared edge into `fit.eigen`, not inside this span. See timing/stages.ts.
  spanEnd(ataSpan);

  const raw = await readFloat32(device, outBuf, numWorkgroups * 21 * 4, 'fit:ATApartials', 'pose.fit');
  for (const b of [voteBuf, outBuf, uniformBuf]) b.destroy();

  // Before `raw` is believed. A validation failure is reported asynchronously
  // and silently no-ops every command that uses the offending resource, so
  // without this the readback would be all zeros, the eigen-decomposition below
  // would run on an all-zero ATA, and this would return an arbitrary basis
  // instead of null -- with the caller's `?? fitPairOfPlanes(...)` fallback
  // never firing, because nothing reported a failure.
  const err = await device.popErrorScope();
  if (err) {
    console.error('fitPairOfPlanesGPU: WebGPU validation error, falling back to CPU --', err.message);
    return null;
  }

  // The partial sum sits in `fit.eigen` rather than `fit.ata`, and that is a
  // deliberate impurity: it is arithmetically the tail of the ATA reduction, but
  // it happens AFTER the readback, and putting it in `fit.ata` would mean that
  // span had to span the fence too -- burying the pipeline's stall inside a
  // stage instead of on the edge that reports it.
  const eigenSpan = poseSpan('fit.eigen', { backend: 'gpu' });
  try {
    const packed = new Float64Array(21);
    for (let g = 0; g < numWorkgroups; g++) {
      const base = g * 21;
      for (let k = 0; k < 21; k++) packed[k] += raw[base + k];
    }
    // Unpack in the exact a<=b order fitPlanes.wgsl.ts packed them in.
    const ATA: number[][] = Array.from({ length: 6 }, () => new Array(6).fill(0));
    let idx = 0;
    for (let a = 0; a < 6; a++) {
      for (let b = a; b < 6; b++) {
        ATA[a][b] = packed[idx]; ATA[b][a] = packed[idx];
        idx++;
      }
    }
    // THE SHARED TAIL. This used to be twenty lines copied verbatim out of
    // fitPairOfPlanes -- the same smallestEigenvector, the same jacobi, the same
    // b1/b2 handedness -- so the two backends could disagree about the fit
    // without either file changing. Now the GPU path differs from the CPU path
    // in exactly one thing, which is where its ATA came from.
    return planesFromScatter(ATA);
  } finally {
    spanEnd(eigenSpan);
  }
}
