import { activeCamera } from '../camera/store.ts';
import { Camera } from '../camera/model.ts';
import { computeGradient2x2Field } from '../pipeline/gradientField.ts';
import { computeMagTheta, countRectanglePixels, fitRegionWithRetries, growRegionsCCL, LsdRectangle } from '../pipeline/lsdSegments.ts';
import { fitAndTestRegionsGPU } from './lsdFit.ts';

// ── Dev harness: is lsdFit.wgsl.ts's output still the CPU path's output? ──
//
// globalState.useGPULsdFit has been pinned false since the shader and
// pipeline/lsdSegments.ts's countRectanglePixels disagreed about whether the
// NFA alignment test was directed or mod-PI. That disagreement is gone (both
// are directed again -- see levelLineAngle's own comment there), so the shader
// SHOULD be correct as written. This turns "should" into evidence.
//
// Not part of any pipeline. Call it from the devtools console on a real
// capture -- main.ts re-exports every module onto globalThis:
//
//   await verifyLsdFit()
//
// Compares STAGE 4 + STAGE 5-ATTEMPT-0 ONLY, which is exactly the shader's
// scope. Both sides are handed the SAME regions from a single growRegionsCCL
// call, so any difference is the fit/NFA kernel and not the grower, and both
// run with maxRetries forced to 0 so the CPU side stops where the GPU side
// does (the retry loop is deliberately unported -- retry 2+ needs a per-region
// partial sort). fitAndTestRegionsGPU is called DIRECTLY rather than through
// computeLsdRectanglesGPU, because that wrapper silently re-runs every
// GPU-rejected region on CPU -- which is the right production behavior and
// exactly the wrong thing for a comparison, since it would mask disagreements
// on rejection as agreement.
export interface LsdFitVerifyReport {
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

export async function verifyLsdFit(camera?: Camera | null): Promise<LsdFitVerifyReport | string> {
  camera = camera ?? activeCamera() ?? null;
  if (!camera) return 'no active camera';
  const gray = camera.lastNoisedPreviewGray;
  if (!gray) return 'no capture yet -- run a capture first';
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const s = camera.settings;

  const field = computeGradient2x2Field(gray, w, h);
  const { mag, theta } = computeMagTheta(field);
  const { regions } = growRegionsCCL(
    mag, theta, w, h, s.lsdToleranceDeg, s.lsdRhoNoiseThreshold, s.lsdRhoHighThreshold, s.lsdCclSteps,
  );
  if (regions.length === 0) return 'grower produced no regions -- nothing to compare';

  const maxDim = Math.max(w, h);
  const logNTests = s.lsdNfaTestExponent * Math.log(maxDim);
  const logEpsilon = Math.log(s.lsdNfaEpsilon);
  // maxRetries: 0 pins the CPU side to attempt 0, the shader's whole scope.
  const cpuSettings = {
    toleranceDeg: s.lsdToleranceDeg, rhoNoiseThreshold: s.lsdRhoNoiseThreshold,
    rhoHighThreshold: s.lsdRhoHighThreshold, cclSteps: s.lsdCclSteps,
    nfaEpsilon: s.lsdNfaEpsilon, nfaTestExponent: s.lsdNfaTestExponent,
    maxRetries: 0, retryToleranceFactor: s.lsdRetryToleranceFactor, retryShrinkFraction: s.lsdRetryShrinkFraction,
  };

  const cpuStart = performance.now();
  const cpu: (LsdRectangle | null)[] = regions.map((r) => fitRegionWithRetries(r, mag, theta, w, h, cpuSettings, logNTests, logEpsilon));
  const cpuMs = performance.now() - cpuStart;

  const gpuStart = performance.now();
  const gpu = await fitAndTestRegionsGPU(
    mag, theta, w, h, regions, s.lsdRhoNoiseThreshold, s.lsdToleranceDeg, logNTests, logEpsilon,
  );
  const gpuMs = performance.now() - gpuStart;
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
    // what fitRegionWithRetries did internally -- so this is the same n/k that
    // produced c.nfaLog10, not an independent estimate.
    const { n: cpuN, k: cpuK } = countRectanglePixels(c, mag, theta, w, h, s.lsdRhoNoiseThreshold, toleranceRad);
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
    regions: regions.length,
    regionSizes: {
      min: sizes[0], median: quantile(sizes, 0.5), p95: quantile(sizes, 0.95), max: sizes[sizes.length - 1],
      singletons: sizes.filter((n) => n < 2).length,
    },
    acceptedCpu, acceptedGpu, acceptDisagreements, maxAbs, worstNfaRegion, countMismatches, maxCountDelta, cpuMs, gpuMs,
  };
}
