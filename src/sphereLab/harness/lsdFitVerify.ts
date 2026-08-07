import { computeGradient2x2Field } from '../../pose/stages/gradient/gradientField.ts';
import { countRectanglePixels, fitRegionOnce } from '../../pose/stages/lsd/rectangles.cpu.ts';
import { growRegionsCCL } from '../../pose/stages/lsd/regions.cpu.ts';
import type { LsdRectangle } from '../../pose/stages/lsd/types.ts';
import { FieldResidency } from '../../pose/gpu/fieldResidency.ts';
import { fitAndTestRegionsGPU } from '../../pose/stages/lsd/lsdFit.gpu.ts';
import { type InputProvenance, provenance } from './input.ts';
import type { HarnessInput } from './input.ts';

// ── Dev harness: is lsdFit.wgsl.ts's output still the CPU path's output? ──
//
// The GPU fitter was pinned off for a long time, since the shader and
// pose/stages/lsd/rectangles.cpu.ts's countRectanglePixels disagreed about whether the
// NFA alignment test was directed or mod-PI. That disagreement is gone (both
// are directed again -- see the level-line vector block's  own comment there), so the shader
// SHOULD be correct as written. This turns "should" into evidence.
//
// Not part of any pipeline. Call it from the devtools console on a real
// capture -- main.ts re-exports every module onto globalThis:
//
//   await verifyLsdFit(cameraInput())
//   await verifyLsdFit(await fixtureInput('default'))
//
// Compares STAGE 4 + STAGE 5, which is now the whole fitter on both sides --
// the retry loop is retired and the CPU reference is fitRegionOnce, so the two
// paths have identical scope by construction rather than by pinning a setting.
// Both sides are handed the SAME regions from a single growRegionsCCL call, so
// any difference is the fit/NFA kernel and not the grower.
//
// fitAndTestRegionsGPU is called DIRECTLY rather than through
// computeLsdRectanglesGPU. That mattered more when the wrapper re-ran every
// GPU-rejected region on CPU (which would have masked rejection disagreements
// as agreement); it no longer does, but calling the kernel directly is still
// what keeps this a test of the kernel rather than of the wrapper.
// Every report in this directory carries where it came from. See
// harness/input.ts's InputProvenance: six of these recorded nothing at all
// about their input, which made their deltas exactly as un-re-derivable as the
// timing numbers the config-pinning work exists to replace.
interface LsdFitVerifyReport extends InputProvenance {
  regions: number;
  regionSizes: { min: number; median: number; p95: number; max: number; singletons: number };
  acceptedCpu: number;
  acceptedGpu: number;
  acceptDisagreements: number; // regions where the two paths disagree on accept/reject -- the number that actually matters
  maxAbs: { cx: number; cy: number; theta: number; length: number; width: number; nfaLog10: number };
  worstNfaRegion: { index: number; cpu: number; gpu: number; cpuN: number; gpuN: number; cpuK: number; gpuK: number } | null;
  // Splits an nfaLog10 disagreement into its two possible causes. If the two
  // paths agree on n and k but not on the tail, the log-sum-exp arithmetic
  // diverged (f32 precision). If they disagree on n or k, they are counting
  // DIFFERENT PIXELS -- geometry landing on a boundary, not arithmetic -- and
  // no amount of precision work in the tail would fix it.
  countMismatches: { n: number; k: number }; // how many regions differ in each
  maxCountDelta: { n: number; k: number };
  cpuMs: number;
  gpuMs: number;
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))];
}

export async function verifyLsdFit(input: HarnessInput): Promise<LsdFitVerifyReport | string> {
  const { gray, w, h, settings: s } = input;

  const field = computeGradient2x2Field(gray, w, h);
  const { fx, fy } = field;
  const { regions } = growRegionsCCL(
    fx, fy, w, h, s.lsdToleranceDeg, s.lsdRhoNoiseThreshold, s.lsdRhoHighThreshold, s.lsdCclSteps, s.lsdMinRegionSize,
  );
  if (regions.length === 0) return 'grower produced no regions -- nothing to compare';

  const maxDim = Math.max(w, h);
  const logNTests = s.lsdNfaTestExponent * Math.log(maxDim);
  const logEpsilon = Math.log(s.lsdNfaEpsilon);
  // The retry fields are inert now -- fitRegionOnce is attempt-0-only by
  // construction, which is exactly the shader's scope, so there is no longer a
  // maxRetries: 0 pin needed to make the comparison fair. They stay in the
  // literal only because LsdSettings still declares them for the retired
  // fitRegionWithRetries.
  const cpuSettings = {
    toleranceDeg: s.lsdToleranceDeg, rhoNoiseThreshold: s.lsdRhoNoiseThreshold,
    rhoHighThreshold: s.lsdRhoHighThreshold, cclSteps: s.lsdCclSteps, minRegionSize: s.lsdMinRegionSize,
    nfaEpsilon: s.lsdNfaEpsilon, nfaTestExponent: s.lsdNfaTestExponent,
  };

  const cpuStart = performance.now();
  const cpu: (LsdRectangle | null)[] = regions.map((r) => fitRegionOnce(r, fx, fy, w, h, cpuSettings, logNTests, logEpsilon));
  const cpuMs = performance.now() - cpuStart;

  // Everything is handed to the residency on the CPU side, which makes this the
  // ISOLATED cost of the stage: the timing below still includes the mag/theta
  // upload and the CSR pack, exactly as it did before the residency existed, so
  // these numbers stay comparable to the 3.5-vs-8ms that pinned the flag off.
  // What the production path now saves is precisely the part this harness is
  // deliberately still paying.
  const res = await FieldResidency.create(w * h, true);
  res.provideCPU('fx', fx);
  res.provideCPU('fy', fy);
  res.provideRegionsCPU(regions);

  const gpuStart = performance.now();
  const gpu = await fitAndTestRegionsGPU(
    res, w, h, s.lsdRhoNoiseThreshold, s.lsdToleranceDeg, logNTests, logEpsilon,
  );
  const gpuMs = performance.now() - gpuStart;
  res.destroy();
  if (!gpu) return 'WebGPU unavailable -- nothing to compare against';

  const maxAbs = { cx: 0, cy: 0, theta: 0, length: 0, width: 0, nfaLog10: 0 };
  let acceptedCpu = 0, acceptedGpu = 0, acceptDisagreements = 0;
  let worstNfaRegion: LsdFitVerifyReport['worstNfaRegion'] = null;
  let worstNfaDelta = -1;
  const countMismatches = { n: 0, k: 0 }, maxCountDelta = { n: 0, k: 0 };
  const toleranceRad = (s.lsdToleranceDeg * Math.PI) / 180;

  for (let i = 0; i < regions.length; i++) {
    const c = cpu[i], g = gpu[i];
    // A degenerate region (<2 members) produces null on CPU and an
    // all-zero/rejected row on GPU. Both mean "no rectangle"; there are no
    // geometry values to compare, so skip rather than count a false mismatch.
    if (!c) { if (g.accepted) acceptDisagreements++; continue; }
    if (c.accepted) acceptedCpu++;
    if (g.accepted) acceptedGpu++;
    if (c.accepted !== g.accepted) acceptDisagreements++;
    maxAbs.cx = Math.max(maxAbs.cx, Math.abs(c.cx - g.cx));
    maxAbs.cy = Math.max(maxAbs.cy, Math.abs(c.cy - g.cy));
    // Wrapped, so a legitimate +-2pi representation difference doesn't read as
    // a huge error. The PCA sign resolution can also land theta a full PI apart
    // if the two paths' near-zero dot products fall opposite sides of 0 -- that
    // IS a real disagreement and stays visible here.
    let dTheta = Math.abs(c.theta - g.theta) % (2 * Math.PI);
    if (dTheta > Math.PI) dTheta = 2 * Math.PI - dTheta;
    maxAbs.theta = Math.max(maxAbs.theta, dTheta);
    maxAbs.length = Math.max(maxAbs.length, Math.abs(c.length - g.length));
    maxAbs.width = Math.max(maxAbs.width, Math.abs(c.width - g.width));
    // Both can be +-Infinity for a degenerate tail; only compare finite pairs.
    // Recount on CPU using the CPU's OWN fitted rectangle, mirroring exactly
    // what fitRegionOnce did internally -- so this is the same n/k that
    // produced c.nfaLog10, not an independent estimate.
    const { n: cpuN, k: cpuK } = countRectanglePixels(c, fx, fy, w, h, s.lsdRhoNoiseThreshold, toleranceRad);
    if (cpuN !== g.n) { countMismatches.n++; maxCountDelta.n = Math.max(maxCountDelta.n, Math.abs(cpuN - g.n)); }
    if (cpuK !== g.k) { countMismatches.k++; maxCountDelta.k = Math.max(maxCountDelta.k, Math.abs(cpuK - g.k)); }
    if (Number.isFinite(c.nfaLog10) && Number.isFinite(g.nfaLog10)) {
      const d = Math.abs(c.nfaLog10 - g.nfaLog10);
      maxAbs.nfaLog10 = Math.max(maxAbs.nfaLog10, d);
      if (d > worstNfaDelta) {
        worstNfaDelta = d;
        worstNfaRegion = { index: i, cpu: c.nfaLog10, gpu: g.nfaLog10, cpuN, gpuN: g.n, cpuK, gpuK: g.k };
      }
    }
  }

  const sizes = regions.map((r) => r.members.length).sort((a, b) => a - b);
  return {
    ...provenance(input),
    regions: regions.length,
    regionSizes: {
      min: sizes[0], median: quantile(sizes, 0.5), p95: quantile(sizes, 0.95), max: sizes[sizes.length - 1],
      singletons: sizes.filter((n) => n < 2).length,
    },
    acceptedCpu, acceptedGpu, acceptDisagreements, maxAbs, worstNfaRegion, countMismatches, maxCountDelta, cpuMs, gpuMs,
  };
}
