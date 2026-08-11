import { type Alloc, type Arena, type Slice, bind } from '../../gpu/arena.ts';
import { dispatchCount } from '../../gpu/device.ts';
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
// computeGradient2x2Field -- a pure per-pixel forward-difference map with no
// cross-thread dependency.
//
// The float32 GPU result is widened to Float64Array only if somebody actually
// reads it back, so it is not bit-identical to the CPU path -- fine for the
// forward difference itself (precision-insensitive, near-integer grayscale
// inputs), but downstream hard-threshold consumers are only guaranteed to
// produce equivalent segments, not identically-shaped ones.
//
// ── ENCODE ONLY ──
//
// This function allocates from the caller's arena, encodes into the caller's
// encoder, and returns. It does not submit, does not await, does not read
// anything back, and does not know what runs after it.
//
// It used to take a FieldResidency: it asked that object for `gray` and
// published fx/fy into it, which meant the gradient stage had to know it was
// stage 1 of an LSD chain. It is not -- it is a gradient. The residency also
// made this the only place that could decide whether gray needed uploading,
// which is a scheduling decision that belongs to whoever owns the frame.
//
// No error scope either, and that is a deliberate move rather than a loss. A
// scope here forced an `await popErrorScope()` mid-chain, which is a suspension
// point in the middle of what should be one encode; the chain pushes ONE scope
// around all of its stages and pops it once (see stages/lsd/chain.ts). Validation
// failures are reported per-device, not per-pass, so one scope catches exactly
// what the per-stage scopes caught between them.
export function sizeOfGradient(w: number, h: number): { fx: number; fy: number } {
  const n = w * h;
  return { fx: n * 4, fy: n * 4 };
}

export function encodeGradient2x2(
  arena: Arena, alloc: Alloc, enc: GPUCommandEncoder,
  inp: { gray: Slice; w: number; h: number },
): { fx: Slice; fy: Slice } {
  const { gray, w, h } = inp;
  const sizes = sizeOfGradient(w, h);
  const fx = alloc(sizes.fx, 'gradient.fx');
  const fy = alloc(sizes.fy, 'gradient.fy');

  // A uniform is a slice like anything else now. It used to be its own
  // createBuffer + mappedAtCreation per call -- ~14 of those per reconstruction,
  // each one a buffer the caller then had to remember to destroy.
  const dims = alloc(16, 'gradient.dims');
  arena.device.queue.writeBuffer(
    dims.buffer, dims.offset, new Uint32Array([w, h, 0, 0]).buffer,
  );

  const pipeline = getPipeline(arena.device);
  const bindGroup = arena.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: bind(dims, arena) },
      { binding: 1, resource: bind(gray, arena) },
      { binding: 2, resource: bind(fx, arena) },
      { binding: 3, resource: bind(fy, arena) },
    ],
  });

  const pass = enc.beginComputePass(gpuTimelineSlot('gradient2x2'));
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchCount(w), dispatchCount(h));
  pass.end();

  return { fx, fy };
}
