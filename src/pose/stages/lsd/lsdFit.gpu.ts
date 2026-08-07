import { createStorageBuffer, readFloat32, uploadUniform } from '../../gpu/device.ts';
import { FieldResidency } from '../../gpu/fieldResidency.ts';
import { gpuTimelineSlot } from '../../gpu/gpuTimeline.ts';
import { LSD_FIT_WGSL } from './lsdFit.wgsl.ts';
import { spanEnd, spanStart } from '../../../sphereLab/profiling/profiler.ts';

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

interface LsdFitResult {
  cx: number; cy: number; theta: number; length: number; width: number;
  nfaLog10: number; accepted: boolean;
  // The kernel's own NFA counts -- pixels inside the rectangle footprint, and
  // how many of those were aligned. Diagnostic only (nothing in the production
  // path reads them); see lsdFitVerify.ts for why they're worth carrying.
  n: number; k: number;
}

// GPU-resident counterpart to pose/stages/lsd/rectangles.cpu.ts's fitRectangle +
// countRectanglePixels + logBinomialTail (stage 4 + stage 5's first pass,
// no retry -- see lsdFit.wgsl.ts's own header for why the retry loop stays
// CPU-only). One region's members never overlap another's (stage 3's own
// disjoint-partition guarantee), so this is a plain one-thread-per-region
// parallel map, no cross-region synchronization needed at all.
//
// VERIFIED against the CPU path (harness/lsdFitVerify.ts, on a real
// 2931-region capture): zero disagreements on n, k, or accept/reject, and a
// max nfaLog10 delta of 7.7e-6 -- pure f32-vs-f64 rounding. Geometry agrees to
// ~4e-5. The mod-π/directed parity problem that originally pinned this is gone
// (both sides are directed again, see the level-line vector block in the LSD stage).
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
// How many rectangles the readback copies OPTIMISTICALLY, before anyone knows
// how many there are.
//
// This constant exists because of a circularity the indirect-dispatch rewrite
// exposes: a copyBufferToBuffer size is fixed at ENCODE time, so to copy exactly
// regionCount rectangles the host must already know regionCount -- which is the
// fence the rewrite removes. The alternatives are to copy the PROVABLE bound
// (n/minRegionSize = 155520 rectangles = 6.2MB at this resolution, against the
// ~0.2MB actually used -- a byte regression far larger than the fence saving) or
// to copy a cap.
//
// A cap, but NOT a correctness policy: `counts` comes back in the same stall, so
// the host always learns the true count, and if it exceeds this it simply reads
// the remainder in a second pass. Overflow is a PERFORMANCE CLIFF on a rare
// frame, never a truncated answer. That is the property the decode-grid cap and
// the growRegions round budget do not have, and it is why this one is
// acceptable where those were held back.
//
// 8192 against ~5200 observed at 480x648 -- 1.6x headroom, 328KB. Deliberately
// NOT generous: overshoot is paid in bytes on EVERY frame while overflow is paid
// in one extra fence on frames that never happen, so the asymmetry favours a
// tight cap. 16384 was tried first and cost +0.45MB per reconstruction against
// the ~0.2MB actually used, which is real byte cost bought for nothing.
const RECT_COPY_CAP = 8192;
const RECT_STRIDE_BYTES = 10 * 4;

export async function fitAndTestRegionsGPU(
  res: FieldResidency, w: number, h: number,
  rho: number, toleranceDeg: number, logNTests: number, logEpsilon: number,
): Promise<LsdFitResult[] | null> {
  const device = res.device;
  if (!device) return null;
  const dispatchSpan = spanStart('lsdFit dispatch+fence');
  const rs = res.regionsGPU();
  // No `regionCount === 0` early return any more -- the host does not know the
  // count here, by design. An empty capture falls out correctly anyway:
  // regionMeta writes 0 workgroups, the dispatch does nothing, and the readback
  // below reports 0 and builds an empty array.
  device.pushErrorScope('validation');
  const pipeline = getPipeline(device);

  const fxBuf = res.gpu('fx');
  const fyBuf = res.gpu('fy');
  // The PROVABLE bound, not a cap: this one is only allocated, never copied, so
  // its size costs address space rather than bandwidth. rs.maxRegions is exact
  // (a kept region holds >= minRegionSize pixels), so there is no overflow case.
  const outBuf = createStorageBuffer(device, rs.maxRegions * RECT_STRIDE_BYTES);

  const toleranceRad = (toleranceDeg * Math.PI) / 180;
  const uniformData = new ArrayBuffer(32);
  const dv = new DataView(uniformData);
  dv.setUint32(0, w, true); dv.setUint32(4, h, true);
  // Squared, matching levelLine.ts's eligibilityThresholdSq (negative case
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
      { binding: 8, resource: { buffer: rs.counts } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(gpuTimelineSlot('lsdFit:fitRegion'));
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  // INDIRECT, off the same triple collect's finalize used -- both are
  // workgroup_size(64) over regions, so one device-written count serves both.
  pass.dispatchWorkgroupsIndirect(rs.dispatchArgs, 0);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // ONE stall for both: the counts and the optimistic rectangle slice are read
  // off the same submit, concurrently, so this costs one fence rather than the
  // count-then-rectangles pair the old code paid.
  const capped = Math.min(RECT_COPY_CAP, rs.maxRegions);
  const [counts, head] = await Promise.all([
    res.regionCounts(),
    readFloat32(device, outBuf, capped * RECT_STRIDE_BYTES, 'lsdFit:rects'),
  ]);
  const regionCount = counts.regionCount;

  // The cliff, and it is a cliff rather than a bug: everything past the cap is
  // read in a second pass. Costs one extra fence on a frame that produced more
  // regions than any observed capture, and reports itself so RECT_COPY_CAP can
  // be raised rather than silently eaten.
  let raw = head;
  if (regionCount > capped) {
    console.warn(`fitAndTestRegionsGPU: ${regionCount} regions exceeded RECT_COPY_CAP ${RECT_COPY_CAP} -- second readback taken, consider raising it`);
    raw = await readFloat32(device, outBuf, regionCount * RECT_STRIDE_BYTES, 'lsdFit:rectsOverflow');
  }
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
    spanEnd(dispatchSpan);
    return null;
  }
  // Everything inside the span above: encode, submit, the fence, and the error
  // scope. Everything below: pure JS over an array already in host memory. The
  // split is here rather than at the readback because popErrorScope also awaits,
  // so folding it into the unpack span would put a second fence inside a span
  // documented as containing none.
  spanEnd(dispatchSpan);

  // ~5200 objects per frame, of which the pose path's consumer keeps 893 -- the
  // specific hypothesis the perf TODO names for the unattributed ~7ms, and until
  // now invisible because the stage boundary hid it inside `votes`. NO await in
  // this loop, so whatever it reports is real executing JS (or a GC pause, which
  // is the same hypothesis rather than a confound). Measure before rewriting: the
  // fix, if one is warranted, is to filter on `accepted` here or to hand the flat
  // Float32Array down and skip the array entirely.
  const unpackSpan = spanStart('lsdFit unpack (objects)', true);
  const results: LsdFitResult[] = new Array(regionCount);
  for (let i = 0; i < regionCount; i++) {
    const o = i * 10;
    results[i] = {
      cx: raw[o], cy: raw[o + 1], theta: raw[o + 2], length: raw[o + 3], width: raw[o + 4],
      nfaLog10: raw[o + 5], accepted: raw[o + 6] !== 0, n: raw[o + 8], k: raw[o + 9],
    };
  }
  spanEnd(unpackSpan);
  return results;
}
