import { createStorageBuffer, readFloat32, uploadUniform } from './device.ts';
import { FieldResidency } from './fieldResidency.ts';
import { LSD_FIT_WGSL } from './lsdFit.wgsl.ts';

const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();
function getPipeline(device: GPUDevice): GPUComputePipeline {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: LSD_FIT_WGSL, label: 'lsdFit' });
    p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: 'lsdFit' });
    pipelineCache.set(device, p);
  }
  return p;
}

const WORKGROUP_SIZE_1D = 64;

export interface LsdFitResult {
  cx: number; cy: number; theta: number; length: number; width: number;
  nfaLog10: number; accepted: boolean;
  // The kernel's own NFA counts -- pixels inside the rectangle footprint, and
  // how many of those were aligned. Diagnostic only (nothing in the production
  // path reads them); see lsdFitVerify.ts for why they're worth carrying.
  n: number; k: number;
}

// GPU-resident counterpart to pipeline/lsdSegments.ts's fitRectangle +
// countRectanglePixels + logBinomialTail (stage 4 + stage 5's first pass,
// no retry -- see lsdFit.wgsl.ts's own header for why the retry loop stays
// CPU-only). One region's members never overlap another's (stage 3's own
// disjoint-partition guarantee), so this is a plain one-thread-per-region
// parallel map, no cross-region synchronization needed at all.
//
// VERIFIED against the CPU path (pipelineGPU/lsdFitVerify.ts, on a real
// 2931-region capture): zero disagreements on n, k, or accept/reject, and a
// max nfaLog10 delta of 7.7e-6 -- pure f32-vs-f64 rounding. Geometry agrees to
// ~4e-5. The mod-π/directed parity problem that originally pinned this is gone
// (both sides are directed again, see the level-line vector block in lsdSegments.ts).
//
// Getting there required one real fix on BOTH sides: countRectanglePixels'
// inclusion test now carries a BOUNDARY_EPS, because a rectangle's extent is
// defined by its own extreme members, so those members land exactly ON the
// boundary and were being included or dropped by whichever way the last ulp
// rounded. That disagreed across arithmetic on 180/2931 regions for n and
// 277 for k, moving nfaLog10 by up to 4.7 decades.
//
// The upload that used to pin this default-off is GONE. Warm-cache timing on
// that same capture was CPU 3.5ms vs GPU 8ms, and essentially all of the gap
// was this function re-uploading mag/theta (which growRegions.ts had ALSO just
// uploaded, separately) and packing the CSR by hand on CPU. Both now come from
// the residency: when stage 3/3b ran on GPU it binds their buffers directly and
// uploads nothing at all; when they ran on CPU the residency does the upload
// once, in one place, and records it.
//
// Returns null if the residency has no device; caller falls back to the CPU
// version, which stays the source of truth. Returns [] (not null) for an empty
// region set -- a valid empty-input result, not a GPU failure.
export async function fitAndTestRegionsGPU(
  res: FieldResidency, w: number, h: number,
  rho: number, toleranceDeg: number, logNTests: number, logEpsilon: number,
): Promise<LsdFitResult[] | null> {
  const device = res.device;
  if (!device) return null;
  const rs = res.regionsGPU();
  const regionCount = rs.regionCount;
  if (regionCount === 0) return [];
  // Opened AFTER the two early returns above, so no path can leave a scope
  // pushed and unpopped. See the pop below for why this is here at all.
  device.pushErrorScope('validation');
  const pipeline = getPipeline(device);

  const fxBuf = res.gpu('fx');
  const fyBuf = res.gpu('fy');
  const outBuf = createStorageBuffer(device, regionCount * 10 * 4);

  const toleranceRad = (toleranceDeg * Math.PI) / 180;
  const uniformData = new ArrayBuffer(32);
  const dv = new DataView(uniformData);
  dv.setUint32(0, w, true); dv.setUint32(4, h, true); dv.setUint32(8, regionCount, true); dv.setUint32(12, 0, true);
  // Squared, matching lsdSegments.ts's eligibilityThresholdSq (negative case
  // included), so the shader's per-pixel test needs no sqrt.
  dv.setFloat32(16, rho >= 0 ? rho * rho : -Infinity, true);
  dv.setFloat32(20, toleranceRad, true);
  dv.setFloat32(24, logNTests, true); dv.setFloat32(28, logEpsilon, true);
  const uniformBuf = uploadUniform(device, uniformData);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: fxBuf } },
      { binding: 2, resource: { buffer: fyBuf } },
      { binding: 3, resource: { buffer: rs.offsets } },
      { binding: 4, resource: { buffer: rs.members } },
      { binding: 5, resource: { buffer: rs.meanDirs } },
      { binding: 6, resource: { buffer: outBuf } },
      { binding: 7, resource: { buffer: rs.sizes } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(regionCount / WORKGROUP_SIZE_1D));
  pass.end();
  device.queue.submit([encoder.finish()]);

  // Only outBuf and the uniform are ours to free -- fx/fy and the whole
  // region CSR belong to the residency, which is still holding them for any
  // other consumer this frame.
  const raw = await readFloat32(device, outBuf, regionCount * 10 * 4, 'lsdFit:rects');
  for (const b of [outBuf, uniformBuf]) b.destroy();

  // Checked BEFORE `raw` is believed. WebGPU reports validation failures
  // asynchronously: an invalid bind group or pipeline does not throw, it makes
  // every command that uses it a silent no-op, so the readback above would be
  // all zeros and this function would return a full set of plausible-looking
  // rectangles at the origin rather than null -- and the caller's CPU fallback
  // would never fire, because nothing reported a failure. Returning null is
  // what turns that class of bug into an honest fall back to the CPU fitter.
  const err = await device.popErrorScope();
  if (err) {
    console.error('fitAndTestRegionsGPU: WebGPU validation error, falling back to CPU --', err.message);
    return null;
  }

  const results: LsdFitResult[] = new Array(regionCount);
  for (let i = 0; i < regionCount; i++) {
    const o = i * 10;
    results[i] = {
      cx: raw[o], cy: raw[o + 1], theta: raw[o + 2], length: raw[o + 3], width: raw[o + 4],
      nfaLog10: raw[o + 5], accepted: raw[o + 6] !== 0, n: raw[o + 8], k: raw[o + 9],
    };
  }
  return results;
}
