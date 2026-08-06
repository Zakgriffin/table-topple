import { computeGradient2x2Field } from '../pipeline/gradientField.ts';
import { type GrownRegion, growRegionsCCL } from '../pipeline/lsdSegments.ts';
import { growRegionsCCLGPUToCPU } from '../pipelineGPU/growRegions.ts';
import type { HarnessInput } from './input.ts';

// ── Dev harness: does growRegions.wgsl.ts's labeling agree with the CPU's? ──
//
// Not part of any pipeline. Call it from the devtools console on a real
// capture -- main.ts re-exports every module onto globalThis:
//
//   await verifyGrowRegions(cameraInput())
//   await verifyGrowRegions(await fixtureInput('default'))
//
// Unlike verifyLsdFit, this CANNOT assert exact equality, and the report is
// shaped around that. The edge predicate is `cos(theta_i - theta_j) >= cosTol`,
// evaluated in f32 on the GPU and f64 here, so a neighbour pair sitting within
// ~1e-7 of exactly tau can fall on opposite sides in the two paths. There is no
// epsilon fix for it the way there was for lsdFit's boundary pixels: the
// rectangle case was BOUNDED (a miscounted pixel perturbs one region's own n
// and k), whereas one flipped EDGE here merges or splits whole components and
// the difference cascades outward.
//
// So `borderlinePairs` is the number that decides how to read everything else.
// It counts neighbour pairs whose f64 dot lands within BORDERLINE_EPS of
// cosTol -- i.e. the pairs actually EXPOSED to a f32/f64 flip. If it is 0 then
// zero disagreement below is structural, not luck; if it is nonzero, expect
// roughly that many split/merge events and judge by whether the region
// population is materially the same.
interface GrowRegionsVerifyReport {
  cpuRegions: number;
  gpuRegions: number;
  cpuLabeledPixels: number; // pixels surviving hysteresis, i.e. belonging to some region
  gpuLabeledPixels: number;
  pixelsOnlyCpu: number; // labeled by one path and not the other -- should be 0 unless an edge flip killed a component's only rhoHigh pixel
  pixelsOnlyGpu: number;
  // A CPU region is an EXACT match when all of its pixels land in one GPU
  // region AND that region is the same size (containment + equal size == set
  // equality). Anything else is classified by how it failed.
  exactMatches: number;
  cpuRegionsSplit: number; // one CPU region's pixels spread across >1 GPU region
  gpuRegionsMerged: number; // one GPU region absorbed pixels from >1 CPU region
  maxMeanAngleDelta: number; // radians, over exactly-matched regions only
  cpuSizes: { min: number; median: number; p95: number; max: number; singletons: number };
  gpuSizes: { min: number; median: number; p95: number; max: number; singletons: number };
  borderlinePairs: number; // see header -- the exposure count that contextualizes everything above
  cpuRounds: number;
  gpuRounds: number; // batched, so an UPPER BOUND on rounds that did work -- not comparable to cpuRounds
  cpuConverged: boolean;
  gpuConverged: boolean;
  cpuMs: number;
  gpuMs: number;
}

// How near the tolerance boundary a pair has to be to count as exposed. f32 has
// ~7 decimal digits, and the dot is a sum of two products, so 1e-6 is a
// deliberately generous over-estimate of the flip window rather than a tight one.
const BORDERLINE_EPS = 1e-6;

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))];
}

function sizeStats(regions: GrownRegion[]): GrowRegionsVerifyReport['cpuSizes'] {
  const sizes = regions.map((r) => r.members.length).sort((a, b) => a - b);
  if (sizes.length === 0) return { min: 0, median: 0, p95: 0, max: 0, singletons: 0 };
  return {
    min: sizes[0], median: quantile(sizes, 0.5), p95: quantile(sizes, 0.95), max: sizes[sizes.length - 1],
    singletons: sizes.filter((n) => n < 2).length,
  };
}

// Counts neighbour pairs whose compatibility decision sits inside the f32 flip
// window. Only pairs where BOTH pixels are eligible (> rhoLow) matter -- an
// ineligible pixel never participates in an edge regardless of its angle. Each
// unordered pair is visited twice by the full 8-neighbourhood; that's left
// as-is so the number is directly comparable to the hook pass's own edge tests.
function countBorderlinePairs(
  mag: Float64Array, theta: Float64Array, w: number, h: number, rhoLow: number, cosTol: number,
): number {
  const NDX = [1, 1, 0, -1, -1, -1, 0, 1], NDY = [0, 1, 1, 1, 0, -1, -1, -1];
  let count = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!(mag[i] > rhoLow)) continue;
      const ci = Math.cos(theta[i]), si = Math.sin(theta[i]);
      for (let k = 0; k < 8; k++) {
        const nx = x + NDX[k], ny = y + NDY[k];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const j = ny * w + nx;
        if (!(mag[j] > rhoLow)) continue;
        const dot = ci * Math.cos(theta[j]) + si * Math.sin(theta[j]);
        if (Math.abs(dot - cosTol) < BORDERLINE_EPS) count++;
      }
    }
  }
  return count;
}

export async function verifyGrowRegions(input: HarnessInput): Promise<GrowRegionsVerifyReport | string> {
  const { gray, w, h, settings: s } = input;

  // Both paths are handed the SAME mag/theta, so any difference below is the
  // round loop and not the gradient stage.
  const field = computeGradient2x2Field(gray, w, h);
  const { fx, fy } = field;
  const args = [
    w, h, s.lsdToleranceDeg, s.lsdRhoNoiseThreshold, s.lsdRhoHighThreshold, s.lsdCclSteps, s.lsdMinRegionSize,
  ] as const;

  const cpuStart = performance.now();
  const cpu = growRegionsCCL(fx, fy, ...args);
  const cpuMs = performance.now() - cpuStart;

  const gpuStart = performance.now();
  // collectOnGPU: FALSE, so both sides collect on CPU and the only difference
  // left is the round loop -- which is what this harness's own header claims it
  // isolates.
  //
  // It did not, until this argument became explicit. The parameter used to
  // default to `!globalState.forceCPU`, so on the production configuration
  // (forceCPU off) this call collected on GPU while `growRegionsCCL` above
  // collected on CPU: the comparison was (CPU grow + CPU collect) vs (GPU grow +
  // GPU collect), conflating the two stages it exists to tell apart. Worse, it
  // silently changed SHAPE with a UI checkbox -- turn forceCPU on and the same
  // function suddenly compared like against like. That is the ambient-config
  // failure in miniature: the harness could not state what it was measuring.
  const gpu = await growRegionsCCLGPUToCPU(fx, fy, ...args, false);
  const gpuMs = performance.now() - gpuStart;
  // Null means either no WebGPU at all or a validation error the grower's own
  // error scope caught -- in the latter case it has already logged the message.
  if (!gpu) return 'GPU grower returned null (WebGPU unavailable, or a validation error -- check the console)';

  const n = w * h;
  let cpuLabeledPixels = 0, gpuLabeledPixels = 0, pixelsOnlyCpu = 0, pixelsOnlyGpu = 0;
  // partner[c]: which GPU region CPU region c's pixels land in. -2 unset,
  // -1 means "more than one", i.e. c was split.
  const partner = new Int32Array(cpu.regions.length).fill(-2);
  // How many distinct CPU regions each GPU region absorbed, tracked the same way.
  const absorbedFrom = new Int32Array(gpu.regions.length).fill(-2);
  for (let i = 0; i < n; i++) {
    const c = cpu.regionId[i], g = gpu.regionId[i];
    if (c >= 0) cpuLabeledPixels++;
    if (g >= 0) gpuLabeledPixels++;
    if (c >= 0 && g < 0) { pixelsOnlyCpu++; continue; }
    if (g >= 0 && c < 0) { pixelsOnlyGpu++; continue; }
    if (c < 0) continue;
    if (partner[c] === -2) partner[c] = g; else if (partner[c] !== g) partner[c] = -1;
    if (absorbedFrom[g] === -2) absorbedFrom[g] = c; else if (absorbedFrom[g] !== c) absorbedFrom[g] = -1;
  }

  let exactMatches = 0, cpuRegionsSplit = 0, maxMeanAngleDelta = 0;
  for (let c = 0; c < cpu.regions.length; c++) {
    const g = partner[c];
    if (g === -1) { cpuRegionsSplit++; continue; }
    if (g < 0) continue; // every pixel of this region was unlabeled on the GPU side
    if (gpu.regions[g].members.length !== cpu.regions[c].members.length) continue; // contained but merged into something larger
    exactMatches++;
    // The ANGLE BETWEEN the two mean directions, via atan2(cross, dot) rather
    // than acos(dot): acos loses most of its precision exactly where these
    // land, next to zero. Same quantity the old meanAngle delta reported, so
    // the numbers stay comparable across the vector-space change.
    const a = cpu.regions[c], b = gpu.regions[g];
    const d = Math.abs(Math.atan2(a.meanUx * b.meanUy - a.meanUy * b.meanUx, a.meanUx * b.meanUx + a.meanUy * b.meanUy));
    maxMeanAngleDelta = Math.max(maxMeanAngleDelta, d);
  }
  let gpuRegionsMerged = 0;
  for (let g = 0; g < gpu.regions.length; g++) if (absorbedFrom[g] === -1) gpuRegionsMerged++;

  return {
    cpuRegions: cpu.regions.length,
    gpuRegions: gpu.regions.length,
    cpuLabeledPixels, gpuLabeledPixels, pixelsOnlyCpu, pixelsOnlyGpu,
    exactMatches, cpuRegionsSplit, gpuRegionsMerged, maxMeanAngleDelta,
    cpuSizes: sizeStats(cpu.regions),
    gpuSizes: sizeStats(gpu.regions),
    borderlinePairs: countBorderlinePairs(
      fx, fy, w, h, s.lsdRhoNoiseThreshold, Math.cos((s.lsdToleranceDeg * Math.PI) / 180),
    ),
    cpuRounds: cpu.roundsRun, gpuRounds: gpu.roundsRun,
    cpuConverged: cpu.converged, gpuConverged: gpu.converged,
    cpuMs, gpuMs,
  };
}
