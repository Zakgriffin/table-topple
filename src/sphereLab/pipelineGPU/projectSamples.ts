import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { MATH_QUAT } from '../constants.ts';
import { cornerDir } from '../math/geometry.ts';
import { getAnalysisVFovRad } from '../pipeline/capture.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { ProjectedSamplesDense } from '../types.ts';
import { createStorageBuffer, dispatchCount, getGPUDevice, readFloat32, readUint32, uploadFloat32, uploadUniform } from './device.ts';
import { GRADIENT_WGSL } from './voteGeneration.wgsl.ts';
import { PROJECT_SAMPLES_WGSL } from './projectSamples.wgsl.ts';

// Stage 2 (bucket accumulation) deliberately stays on CPU for now -- it's a
// scatter-add of FLOAT sums (color, gradient covector) into a data-dependent
// bucket per sample, and WGSL has no atomic<f32> (only atomic<u32>/
// atomic<i32>, which is why decodeTally's and voteBandSelect's histograms
// worked cleanly -- they're pure counting). Doing this on GPU would need
// fixed-point atomic<i32> accumulation, a real design commitment (picking a
// scale factor that doesn't overflow/lose precision for this data's range)
// intentionally deferred. Stage 1 (this file) is a pure per-pixel map with
// no such problem -- every thread owns its own dense output slot.

interface Pipelines { gradient: GPUComputePipeline; project: GPUComputePipeline }
const pipelineCache = new WeakMap<GPUDevice, Pipelines>();
function getPipelines(device: GPUDevice): Pipelines {
  let p = pipelineCache.get(device);
  if (!p) {
    const gradModule = device.createShaderModule({ code: GRADIENT_WGSL, label: 'projectSamplesGradient' });
    const projModule = device.createShaderModule({ code: PROJECT_SAMPLES_WGSL, label: 'projectSamples' });
    p = {
      gradient: device.createComputePipeline({ layout: 'auto', compute: { module: gradModule, entryPoint: 'main' }, label: 'projectSamplesGradient' }),
      project: device.createComputePipeline({ layout: 'auto', compute: { module: projModule, entryPoint: 'main' }, label: 'projectSamples' }),
    };
    pipelineCache.set(device, p);
  }
  return p;
}

function buildProjectUniforms(
  w: number, h: number, minGrazingCos: number, distance: number, vFovRad: number, aspect: number,
  quat: THREE.Quaternion, drow: THREE.Vector3, dcol: THREE.Vector3, normal: THREE.Vector3,
): ArrayBuffer {
  const buf = new ArrayBuffer(96);
  const dv = new DataView(buf);
  dv.setUint32(0, w, true); dv.setUint32(4, h, true);
  dv.setFloat32(16, minGrazingCos, true); dv.setFloat32(20, distance, true); dv.setFloat32(24, vFovRad, true); dv.setFloat32(28, aspect, true);
  dv.setFloat32(32, quat.x, true); dv.setFloat32(36, quat.y, true); dv.setFloat32(40, quat.z, true); dv.setFloat32(44, quat.w, true);
  dv.setFloat32(48, drow.x, true); dv.setFloat32(52, drow.y, true); dv.setFloat32(56, drow.z, true);
  dv.setFloat32(64, dcol.x, true); dv.setFloat32(68, dcol.y, true); dv.setFloat32(72, dcol.z, true);
  dv.setFloat32(80, normal.x, true); dv.setFloat32(84, normal.y, true); dv.setFloat32(88, normal.z, true);
  return buf;
}

// GPU-resident counterpart to decodeGrid.ts's projectSamplesCPU (stage 1 of
// castAndBucketProjectedSamples). Returns null if WebGPU isn't available;
// caller falls back to the CPU version, which stays the source of truth.
export async function projectSamplesGPU(camera: Camera): Promise<ProjectedSamplesDense | null> {
  if (!camera.lastRecoveredAxes) return null;
  const device = await getGPUDevice();
  if (!device) return null;
  const { gradient, project } = getPipelines(device);

  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const vFovRad = getAnalysisVFovRad(camera);
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const MIN_GRAZING_COS = 0.15;
  const n = w * h;

  // Gradient field (reuses voteGeneration.ts's own GRADIENT_WGSL kernel --
  // same finite-difference computation, radius 1, just a different input
  // image) -- only needed if there's a captured frame to differentiate.
  const gray = camera.lastNoisedPreviewGray;
  const fxBuf = createStorageBuffer(device, n * 4);
  const fyBuf = createStorageBuffer(device, n * 4);
  const uploadSpan = spanStart('CPU→GPU upload phase (gray + uniforms)');
  let grayBuf: GPUBuffer | null = null;
  if (gray) {
    grayBuf = uploadFloat32(device, new Float32Array(gray));
  }
  const gradDimsBuf = uploadUniform(device, new Uint32Array([w, h, 1, 0]).buffer);
  const projUniformBuf = uploadUniform(device, buildProjectUniforms(
    w, h, MIN_GRAZING_COS, distance, vFovRad, camera.aspect, MATH_QUAT, Drow, Dcol, normal,
  ));
  spanEnd(uploadSpan);

  const dispatchSpan = spanStart('GPU dispatch (gradient + project)');
  const encoder = device.createCommandEncoder();
  if (grayBuf) {
    const gradBindGroup = device.createBindGroup({
      layout: gradient.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gradDimsBuf } },
        { binding: 1, resource: { buffer: grayBuf } },
        { binding: 2, resource: { buffer: fxBuf } },
        { binding: 3, resource: { buffer: fyBuf } },
      ],
    });
    const gradPass = encoder.beginComputePass();
    gradPass.setPipeline(gradient);
    gradPass.setBindGroup(0, gradBindGroup);
    gradPass.dispatchWorkgroups(dispatchCount(w), dispatchCount(h));
    gradPass.end();
  }
  // fxBuf/fyBuf stay zero-initialized (per WebGPU spec) if there's no gray
  // frame to differentiate -- matches the CPU path's `srcGrad ? ... : null`
  // (zero fx/fy means every sample's gradient covector comes out as 0,0,
  // same as the CPU version's cxAtSample/cyAtSample defaults).

  const sampleOutBuf = createStorageBuffer(device, n * 16);
  const validOutBuf = createStorageBuffer(device, n * 4);
  const projBindGroup = device.createBindGroup({
    layout: project.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: projUniformBuf } },
      { binding: 1, resource: { buffer: fxBuf } },
      { binding: 2, resource: { buffer: fyBuf } },
      { binding: 3, resource: { buffer: sampleOutBuf } },
      { binding: 4, resource: { buffer: validOutBuf } },
    ],
  });
  const projPass = encoder.beginComputePass();
  projPass.setPipeline(project);
  projPass.setBindGroup(0, projBindGroup);
  projPass.dispatchWorkgroups(dispatchCount(w), dispatchCount(h));
  projPass.end();
  device.queue.submit([encoder.finish()]);
  spanEnd(dispatchSpan);

  const [sampleRaw, validRaw] = await Promise.all([
    readFloat32(device, sampleOutBuf, n * 16),
    readUint32(device, validOutBuf, n * 4),
  ]);

  const buffersToDestroy = [fxBuf, fyBuf, gradDimsBuf, projUniformBuf, sampleOutBuf, validOutBuf];
  if (grayBuf) buffersToDestroy.push(grayBuf);
  for (const b of buffersToDestroy) b.destroy();

  const finishSpan = spanStart('CPU finish (unpack + min/max)');
  const uArr = new Float32Array(n), vArr = new Float32Array(n), cxArr = new Float32Array(n), cyArr = new Float32Array(n);
  const validArr = new Uint8Array(n);
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (let i = 0; i < n; i++) {
    if (!validRaw[i]) continue;
    const o = i * 4;
    const u = sampleRaw[o], v = sampleRaw[o + 1];
    uArr[i] = u; vArr[i] = v; cxArr[i] = sampleRaw[o + 2]; cyArr[i] = sampleRaw[o + 3];
    validArr[i] = 1;
    if (u < minU) minU = u; if (u > maxU) maxU = u;
    if (v < minV) minV = v; if (v > maxV) maxV = v;
  }
  spanEnd(finishSpan);
  if (!isFinite(minU) || !isFinite(minV)) return null;
  return { u: uArr, v: vArr, cx: cxArr, cy: cyArr, valid: validArr, minU, maxU, minV, maxV };
}
