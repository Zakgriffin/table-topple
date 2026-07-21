import * as THREE from 'three';
import { CameraSettingsCommon } from '../camera/settings.ts';
import { attachGPUKernelBreakdown, profilerEnabled, spanEnd, spanStart } from '../profiling/profiler.ts';
import { Vote } from '../types.ts';
import { createStorageBuffer, createTimestampQuerySet, dispatchCount, getGPUDevice, readFloat32, resolveTimestamps, supportsTimestampQuery, uploadFloat32, uploadUniform, WORKGROUP_SIZE } from './device.ts';
import { BOX_BLUR_H_WGSL, BOX_BLUR_V_WGSL, DOUBLE_ANGLE_WGSL, EFFECTIVE_WGSL, GRADIENT_WGSL, WALK_AND_VOTE_WGSL } from './voteGeneration.wgsl.ts';

interface Pipelines {
  gradient: GPUComputePipeline;
  doubleAngle: GPUComputePipeline;
  blurH: GPUComputePipeline;
  blurV: GPUComputePipeline;
  effective: GPUComputePipeline;
  walkAndVote: GPUComputePipeline;
}

// One pipeline set per GPUDevice (a device lost/reacquired would need fresh
// ones) -- shader compilation isn't free, so this is built once and reused
// across every reconstruction call, not per call.
const pipelineCache = new WeakMap<GPUDevice, Pipelines>();

function makePipeline(device: GPUDevice, code: string, label: string): GPUComputePipeline {
  const module = device.createShaderModule({ code, label });
  return device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
}

function getPipelines(device: GPUDevice): Pipelines {
  let p = pipelineCache.get(device);
  if (!p) {
    p = {
      gradient: makePipeline(device, GRADIENT_WGSL, 'gradient'),
      doubleAngle: makePipeline(device, DOUBLE_ANGLE_WGSL, 'doubleAngle'),
      blurH: makePipeline(device, BOX_BLUR_H_WGSL, 'blurH'),
      blurV: makePipeline(device, BOX_BLUR_V_WGSL, 'blurV'),
      effective: makePipeline(device, EFFECTIVE_WGSL, 'effective'),
      walkAndVote: makePipeline(device, WALK_AND_VOTE_WGSL, 'walkAndVote'),
    };
    pipelineCache.set(device, p);
  }
  return p;
}

function dispatch(
  encoder: GPUCommandEncoder, device: GPUDevice, pipeline: GPUComputePipeline,
  buffers: GPUBuffer[], w: number, h: number,
  timestampWrites?: GPUComputePassTimestampWrites,
) {
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
  });
  const pass = encoder.beginComputePass(timestampWrites ? { timestampWrites } : undefined);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchCount(w), dispatchCount(h));
  pass.end();
}

function buildWalkUniforms(
  w: number, h: number, r: number, maxSteps: number,
  devCos: number, magFraction: number, graceSamples: number, aspect: number,
  vFovRad: number, quat: THREE.Quaternion,
): ArrayBuffer {
  const buf = new ArrayBuffer(64);
  const dv = new DataView(buf);
  dv.setUint32(0, w, true); dv.setUint32(4, h, true); dv.setUint32(8, r, true); dv.setUint32(12, maxSteps, true);
  dv.setFloat32(16, devCos, true); dv.setFloat32(20, magFraction, true); dv.setUint32(24, graceSamples, true); dv.setFloat32(28, aspect, true);
  dv.setFloat32(32, vFovRad, true); // 36/40/44 left as padding
  dv.setFloat32(48, quat.x, true); dv.setFloat32(52, quat.y, true); dv.setFloat32(56, quat.z, true); dv.setFloat32(60, quat.w, true);
  return buf;
}

// GPU-resident counterpart to pipeline/votes.ts's computeWorldVotes -- see
// voteGeneration.wgsl.ts's header comment for the one deliberate numeric
// deviation (skipped agreement-field normalization constant) and why it's
// provably inconsequential downstream. Returns null if WebGPU isn't
// available at all; the caller falls back to the CPU version in that case
// (see pipeline/votes.ts's computeWorldVotes, still the source of truth).
export async function computeWorldVotesGPU(
  settings: CameraSettingsCommon,
  gray: Float64Array, w: number, h: number,
  gradientRadius: number, agreementRadius: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Promise<Vote[] | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const pipelines = getPipelines(device);
  const n = w * h;

  const uploadSpan = spanStart('CPU→GPU upload phase (gray + uniforms)');
  const grayBuf = uploadFloat32(device, new Float32Array(gray));
  const fxBuf = createStorageBuffer(device, n * 4);
  const fyBuf = createStorageBuffer(device, n * 4);
  const cxBuf = createStorageBuffer(device, n * 4);
  const cyBuf = createStorageBuffer(device, n * 4);
  const tmpXBuf = createStorageBuffer(device, n * 4);
  const tmpYBuf = createStorageBuffer(device, n * 4);
  const sxBuf = createStorageBuffer(device, n * 4);
  const syBuf = createStorageBuffer(device, n * 4);
  const effFxBuf = createStorageBuffer(device, n * 4);
  const effFyBuf = createStorageBuffer(device, n * 4);
  const voteBuf = createStorageBuffer(device, n * 16); // vec4<f32> per pixel

  const r = Math.round(gradientRadius);
  const aggR = Math.round(agreementRadius);
  const dimsBuf = uploadUniform(device, new Uint32Array([w, h, r, 0]).buffer);
  const blurDimsBuf = uploadUniform(device, new Uint32Array([w, h, aggR, 0]).buffer);
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(settings.tangentWalkDeviationDeg));
  const walkUniformsBuf = uploadUniform(device, buildWalkUniforms(
    w, h, r, Math.round(settings.tangentWalkMaxSteps),
    devCos, settings.tangentWalkMagFraction, Math.round(settings.tangentWalkGraceSamples),
    aspect, vFovRad, quat,
  ));
  spanEnd(uploadSpan);

  const STAGE_NAMES = ['gradient', 'doubleAngle', 'blurH', 'blurV', 'effective', 'walkAndVote'];
  const wantTimestamps = profilerEnabled() && supportsTimestampQuery(device);
  const querySet = wantTimestamps ? createTimestampQuerySet(device, STAGE_NAMES.length) : null;
  const tw = (i: number): GPUComputePassTimestampWrites | undefined =>
    querySet ? { querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 } : undefined;

  const dispatchSpan = spanStart('GPU dispatch (6 kernels)');
  const encoder = device.createCommandEncoder();
  dispatch(encoder, device, pipelines.gradient, [dimsBuf, grayBuf, fxBuf, fyBuf], w, h, tw(0));
  dispatch(encoder, device, pipelines.doubleAngle, [dimsBuf, fxBuf, fyBuf, cxBuf, cyBuf], w, h, tw(1));
  dispatch(encoder, device, pipelines.blurH, [blurDimsBuf, cxBuf, cyBuf, tmpXBuf, tmpYBuf], w, h, tw(2));
  dispatch(encoder, device, pipelines.blurV, [blurDimsBuf, tmpXBuf, tmpYBuf, sxBuf, syBuf], w, h, tw(3));
  dispatch(encoder, device, pipelines.effective, [dimsBuf, fxBuf, fyBuf, sxBuf, syBuf, effFxBuf, effFyBuf], w, h, tw(4));
  dispatch(encoder, device, pipelines.walkAndVote, [walkUniformsBuf, effFxBuf, effFyBuf, voteBuf], w, h, tw(5));
  device.queue.submit([encoder.finish()]);

  if (querySet) {
    const durations = await resolveTimestamps(device, querySet, STAGE_NAMES.length);
    attachGPUKernelBreakdown(STAGE_NAMES.map((name, i) => ({ name, durationMs: durations[i] })));
    querySet.destroy();
  }
  spanEnd(dispatchSpan);

  const raw = await readFloat32(device, voteBuf, n * 16); // readFloat32 self-spans (device.ts) -- see attachGPUKernelBreakdown's comment for why this is split from kernel time
  const votes: Vote[] = [];
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const weight = raw[o + 3];
    if (weight === 0) continue;
    votes.push({ n: new THREE.Vector3(raw[o], raw[o + 1], raw[o + 2]), weight });
  }

  for (const b of [grayBuf, fxBuf, fyBuf, cxBuf, cyBuf, tmpXBuf, tmpYBuf, sxBuf, syBuf, effFxBuf, effFyBuf, voteBuf, dimsBuf, blurDimsBuf, walkUniformsBuf]) {
    b.destroy();
  }
  return votes;
}
