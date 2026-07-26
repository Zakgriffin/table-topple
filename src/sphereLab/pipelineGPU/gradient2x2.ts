import { GradientField } from '../types.ts';
import { createStorageBuffer, dispatchCount, getGPUDevice, readFloat32, uploadFloat32, uploadUniform } from './device.ts';
import { GRADIENT_2X2_WGSL } from './gradient2x2.wgsl.ts';

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

// GPU-resident counterpart to pipeline/gradientField.ts's
// computeGradient2x2Field -- a pure per-pixel forward-difference map with
// no cross-thread dependency, same shape as projectSamples.ts's stage 1.
// Returns null if WebGPU isn't available; caller falls back to the CPU
// version, which stays the source of truth. The float32 GPU result gets
// widened into this GradientField's own Float64Array storage, so it's not
// bit-identical to the CPU path -- fine for the forward difference itself
// (precision-insensitive, near-integer grayscale inputs), but downstream
// hard-threshold/sort consumers (computeTopGradientAlpha's percentile cut
// -- since removed as dead code, see this session's chat -- and
// computeBucketFillRegions' flood-fill comparisons, still live) aren't guaranteed to
// produce identically-shaped segments, only equivalent ones -- see this
// session's chat.
export async function computeGradient2x2FieldGPU(gray: Float64Array, w: number, h: number): Promise<GradientField | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const pipeline = getPipeline(device);

  const n = w * h;
  const grayBuf = uploadFloat32(device, new Float32Array(gray));
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
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchCount(w), dispatchCount(h));
  pass.end();
  device.queue.submit([encoder.finish()]);

  const [fx32, fy32] = await Promise.all([
    readFloat32(device, fxBuf, n * 4),
    readFloat32(device, fyBuf, n * 4),
  ]);

  for (const b of [grayBuf, fxBuf, fyBuf, dimsBuf]) b.destroy();

  return { fx: new Float64Array(fx32), fy: new Float64Array(fy32), w, h, r: 1 };
}
