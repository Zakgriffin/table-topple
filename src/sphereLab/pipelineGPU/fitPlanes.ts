import * as THREE from 'three';
import { jacobiEigenSymmetric, smallestEigenvector } from '../../linalg.ts';
import { attachGPUKernelBreakdown, profilerEnabled, spanEnd, spanStart } from '../profiling/profiler.ts';
import { Vote } from '../types.ts';
import { createStorageBuffer, createTimestampQuerySet, getGPUDevice, readFloat32, resolveTimestamps, supportsTimestampQuery, uploadFloat32, uploadUniform } from './device.ts';
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

// GPU-resident counterpart to pipeline/votes.ts's fitPairOfPlanes -- see
// fitPlanes.wgsl.ts's header for exactly what's offloaded (the ATA
// reduction) vs what stays on CPU (the eigendecomposition, fixed-size
// regardless of vote count). Returns null if WebGPU isn't available; caller
// falls back to the CPU version, which stays the source of truth.
export async function fitPairOfPlanesGPU(
  votes: Vote[], power: number,
): Promise<{ Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  if (votes.length === 0) return null;
  const pipeline = getPipeline(device);

  const n = votes.length;
  let maxW = 0;
  for (const { weight } of votes) if (weight > maxW) maxW = weight;

  const voteData = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    voteData[o] = votes[i].n.x; voteData[o + 1] = votes[i].n.y; voteData[o + 2] = votes[i].n.z; voteData[o + 3] = votes[i].weight;
  }
  const voteBuf = uploadFloat32(device, voteData);

  const numWorkgroups = Math.ceil(n / WORKGROUP_SIZE_1D);
  const outBuf = createStorageBuffer(device, numWorkgroups * 21 * 4);

  const uniformData = new ArrayBuffer(16);
  const dv = new DataView(uniformData);
  dv.setUint32(0, n, true); dv.setFloat32(4, maxW, true); dv.setFloat32(8, power, true);
  const uniformBuf = uploadUniform(device, uniformData);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: voteBuf } },
      { binding: 2, resource: { buffer: outBuf } },
    ],
  });

  const wantTimestamps = profilerEnabled() && supportsTimestampQuery(device);
  const querySet = wantTimestamps ? createTimestampQuerySet(device, 1) : null;
  const dispatchSpan = spanStart('GPU dispatch (ATA reduction)');
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(numWorkgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
  if (querySet) {
    const [durationMs] = await resolveTimestamps(device, querySet, 1);
    attachGPUKernelBreakdown([{ name: 'ATA reduction kernel', durationMs }]);
    querySet.destroy();
  }
  spanEnd(dispatchSpan);

  const raw = await readFloat32(device, outBuf, numWorkgroups * 21 * 4);
  for (const b of [voteBuf, outBuf, uniformBuf]) b.destroy();

  const finishSpan = spanStart('CPU finish (sum partials + eigen)');
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

  // From here down, identical to fitPairOfPlanes' own tail (votes.ts) --
  // a fixed-size 6x6 -> 3x3 eigendecomposition, not worth porting.
  const m = smallestEigenvector(ATA);
  const M = [
    [m[0], m[3] / 2, m[4] / 2],
    [m[3] / 2, m[1], m[5] / 2],
    [m[4] / 2, m[5] / 2, m[2]],
  ];
  const { values, vectors } = jacobiEigenSymmetric(M);
  let zeroIdx = 0;
  for (let i = 1; i < 3; i++) if (Math.abs(values[i]) < Math.abs(values[zeroIdx])) zeroIdx = i;
  const others = [0, 1, 2].filter((i) => i !== zeroIdx);
  const b1 = new THREE.Vector3(vectors[others[0]][0], vectors[others[0]][1], vectors[others[0]][2]);
  const b2 = new THREE.Vector3(vectors[others[1]][0], vectors[others[1]][1], vectors[others[1]][2]);
  const Dnormal = new THREE.Vector3(vectors[zeroIdx][0], vectors[zeroIdx][1], vectors[zeroIdx][2]).normalize();
  const Drow = b1.clone().add(b2);
  const Dcol = b1.clone().sub(b2);
  spanEnd(finishSpan);
  if (Drow.lengthSq() < 1e-9 || Dcol.lengthSq() < 1e-9) return null;
  return { Drow: Drow.normalize(), Dcol: Dcol.normalize(), Dnormal };
}
