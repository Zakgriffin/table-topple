import { createStorageBuffer, dispatchCount, uploadUniform } from '../../gpu/device.ts';
import { FieldResidency } from '../../gpu/fieldResidency.ts';
import { GRADIENT_2X2_WGSL } from './gradient2x2.wgsl.ts';
import { gpuTimelineSlot } from '../../gpu/gpuTimeline.ts';

const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();
function getPipeline(device: GPUDevice): GPUComputePipeline {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: GRADIENT_2X2_WGSL, label: 'gradient2x2' });
    p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: 'gradient2x2' });
    pipelineCache.set(device, p);
  }
  return p;
}

// GPU-resident counterpart to pose/stages/gradient/gradientField.ts's
// computeGradient2x2Field -- a pure per-pixel forward-difference map with
// no cross-thread dependency, same shape as projectSamples.ts's stage 1.
// The float32 GPU result is widened to Float64Array only if somebody actually
// asks for the CPU side, so it's not bit-identical to the CPU path -- fine for
// the forward difference itself (precision-insensitive, near-integer grayscale
// inputs), but downstream hard-threshold consumers (computeBucketFillRegions'
// flood-fill comparisons) aren't guaranteed to produce identically-shaped
// segments, only equivalent ones -- see this session's chat.
//
// Stage 1 of the LSD chain, and the last stage to stop transferring by hand.
// It used to take a Float64Array and return a GradientField, which meant it
// uploaded gray and read fx/fy straight back every frame -- and then
// growRegionsCCLGPU uploaded those same two arrays again a moment later, so a
// fully-GPU chain paid FIVE crossings to move numbers that never needed to
// leave the device. Now gray comes out of the residency and fx/fy are published
// into it, so with stages 3/3b/4 on GPU too the only traffic left is the gray
// upload -- and when a downstream stage is on CPU the residency does the
// readback itself, in one place, and records it.
//
// Returns false if the residency has no device or a dispatch failed validation;
// caller falls back to the CPU version, which stays the source of truth. On
// that path nothing has been published, so the fallback is free to publish its
// own fx/fy into the same residency.
export async function computeGradient2x2FieldGPU(res: FieldResidency, w: number, h: number): Promise<boolean> {
  const device = res.device;
  if (!device) return false;
  // Same reasoning as growRegionsCCLGPU's scope: WebGPU reports validation
  // failures asynchronously, so without this an invalid bind group makes every
  // command a silent no-op and stage 1 publishes an all-zero field that looks
  // entirely plausible three stages downstream.
  device.pushErrorScope('validation');
  const pipeline = getPipeline(device);

  const n = w * h;
  const grayBuf = res.gpu('gray');
  const fxBuf = createStorageBuffer(device, n * 4);
  const fyBuf = createStorageBuffer(device, n * 4);
  const dimsBuf = uploadUniform(device, new Uint32Array([w, h, 0, 0]).buffer);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: dimsBuf } },
      { binding: 1, resource: { buffer: grayBuf } },
      { binding: 2, resource: { buffer: fxBuf } },
      { binding: 3, resource: { buffer: fyBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(gpuTimelineSlot('gradient2x2'));
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchCount(w), dispatchCount(h));
  pass.end();
  device.queue.submit([encoder.finish()]);

  // Popped BEFORE publishing, for the same reason growRegionsCCLGPU pops
  // before its provideGPU: the CPU fallback publishes into this same residency,
  // and a half-published failure would turn a recoverable fault into a
  // single-assignment throw. gray is NOT destroyed here on either path -- the
  // residency owns it and this stage is only borrowing.
  const err = await device.popErrorScope();
  if (err) {
    console.error('computeGradient2x2FieldGPU: WebGPU validation error, falling back to CPU --', err.message);
    for (const b of [fxBuf, fyBuf, dimsBuf]) b.destroy();
    return false;
  }

  dimsBuf.destroy();
  res.provideGPU('fx', fxBuf);
  res.provideGPU('fy', fyBuf);
  return true;
}
