import { activeCamera } from '../camera/store.ts';
import { Camera } from '../camera/model.ts';
import { circularFit } from '../pipeline/gridPeriodPhase.ts';
import { sweepResultantsGPU } from './periodSweep.ts';

// ── Dev harness: does periodSweep.wgsl.ts's sweep match the CPU's? ────────
//
// Not part of any pipeline. Call it from the devtools console on a real
// capture -- main.ts re-exports every module onto globalThis:
//
//   await verifyPeriodSweep()
//
// Reads the candidate set (and the per-family values/weights) off the last
// reconstruction's stored result, then scores it BOTH ways here -- the CPU side
// is recomputed rather than taken from the stored scores, so the harness is
// valid whether or not that capture ran the GPU sweep. `rowLines`/
// `colLines` are a 1:1 map of the sweep's own rowSamples/colSamples, so both
// paths get byte-identical input.
//
// What actually matters here is NOT maxAbsDelta. The sweep's job is to feed an
// interior-local-maximum scan and a SIGNIFICANCE cut, so the question is
// whether the two paths pick out the same peaks -- which is why the report
// leads with peak agreement and with the argmax period, and treats the raw
// score deltas as context for those.
interface PeriodSweepVerifyReport {
  // Whether the STORED scores matched the recomputed CPU ones -- i.e. whether
  // the capture ran on the CPU path. Purely informational: the comparison below
  // uses the recomputed CPU scores either way, so false does not weaken it.
  hadCpuBaseline: boolean;
  candidates: number;
  lines: { row: number; col: number };
  maxAbsDelta: number; // largest |cpuScore - gpuScore| over all candidates
  meanAbsDelta: number;
  maxRelDelta: number; // same, relative to the CPU score (guarded for tiny scores)
  argmaxPeriodCpu: number;
  argmaxPeriodGpu: number;
  argmaxAgrees: boolean;
  // Interior local maxima clearing SIGNIFICANCE x global best -- reproduced
  // exactly as computeGridPeriodPhase's own peak scan does it, on each path's
  // scores. Disagreement here is the only kind that can change the answer.
  peakCountCpu: number;
  peakCountGpu: number;
  peakPeriodsMatch: boolean;
  // Both time the SWEEP ALONE over the same candidate set -- not the whole of
  // computeGridPeriodPhase, which also does the gnomonic projection, the
  // distinctness test and the golden-section refinement. gpuMs includes upload
  // and readback, which is the honest comparison since the CPU path has neither.
  cpuMs: number;
  gpuMs: number;
}

const SIGNIFICANCE = 0.5; // must track computeGridPeriodPhase's own constant

function findPeakPeriods(samples: { period: number; score: number }[]): number[] {
  let gMax = 0;
  for (const s of samples) if (s.score > gMax) gMax = s.score;
  const out: number[] = [];
  for (let i = 1; i < samples.length - 1; i++) {
    const s = samples[i];
    if (s.score > samples[i - 1].score && s.score >= samples[i + 1].score && s.score >= SIGNIFICANCE * gMax) {
      out.push(s.period);
    }
  }
  return out;
}

export async function verifyPeriodSweep(camera?: Camera | null): Promise<PeriodSweepVerifyReport | string> {
  camera = camera ?? activeCamera() ?? null;
  if (!camera) return 'no active camera';
  const gpp = camera.lastGridPeriodPhase;
  if (!gpp) return 'no grid period/phase result yet -- run a capture first';
  const { coarseSamples } = gpp.debug;
  if (coarseSamples.length < 3) return 'too few coarse samples to compare';
  // rowLines/colLines are the same per-family samples the sweep folded, in the
  // same order -- computeGridPeriodPhase builds both from one rowSamples/
  // colSamples pass -- so value/weight can be read straight back off them
  // rather than storing a second copy in debug.
  const rowValues = gpp.rowLines.map((s) => s.value), rowWeights = gpp.rowLines.map((s) => s.weight);
  const colValues = gpp.colLines.map((s) => s.value), colWeights = gpp.colLines.map((s) => s.weight);

  // Recover the exact candidate set from the stored samples rather than
  // re-deriving nMin/nMax: period = spread/n in ASCENDING period order, so
  // n runs nMax down to nMin and spread = period * n at every index.
  const spread = Math.max(
    ...[rowValues, colValues].map((vals) => {
      if (vals.length < 2) return 0;
      let lo = Infinity, hi = -Infinity;
      for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
      return hi - lo;
    }),
  );
  const nMax = Math.round(spread / coarseSamples[0].period);
  const nMin = Math.round(spread / coarseSamples[coarseSamples.length - 1].period);
  if (nMax - nMin + 1 !== coarseSamples.length) {
    return `candidate set didn't round-trip (nMax ${nMax}, nMin ${nMin}, samples ${coarseSamples.length}) -- spread derivation is off`;
  }

  const total = rowValues.length + colValues.length;
  const scaled = new Float32Array(total), weights = new Float32Array(total);
  let at = 0;
  for (const [vals, wts] of [[rowValues, rowWeights], [colValues, colWeights]] as const) {
    let lo = Infinity;
    for (const v of vals) if (v < lo) lo = v;
    for (let i = 0; i < vals.length; i++) { scaled[at] = (vals[i] - lo) / spread; weights[at] = wts[i]; at++; }
  }

  const gpuStart = performance.now();
  const scores = await sweepResultantsGPU(scaled, weights, rowValues.length, colValues.length, nMin, nMax);
  const gpuMs = performance.now() - gpuStart;
  if (!scores) return 'GPU sweep returned null (WebGPU unavailable, or a validation error -- check the console)';

  // Recompute the CPU side here rather than trusting the stored scores, so the
  // harness is valid even when the capture was taken with the GPU path on.
  const cpuAt = (P: number) => circularFit(rowValues, rowWeights, P).resultant + circularFit(colValues, colWeights, P).resultant;
  const cpuStart = performance.now();
  const cpuSamples = coarseSamples.map((s) => ({ period: s.period, score: cpuAt(s.period) }));
  const cpuMs = performance.now() - cpuStart;
  const gpuSamples = coarseSamples.map((s, i) => ({ period: s.period, score: scores[i] }));

  let maxAbsDelta = 0, sumAbs = 0, maxRelDelta = 0;
  let argmaxCpu = 0, argmaxGpu = 0;
  for (let i = 0; i < cpuSamples.length; i++) {
    const d = Math.abs(cpuSamples[i].score - gpuSamples[i].score);
    maxAbsDelta = Math.max(maxAbsDelta, d);
    sumAbs += d;
    if (cpuSamples[i].score > 1e-6) maxRelDelta = Math.max(maxRelDelta, d / cpuSamples[i].score);
    if (cpuSamples[i].score > cpuSamples[argmaxCpu].score) argmaxCpu = i;
    if (gpuSamples[i].score > gpuSamples[argmaxGpu].score) argmaxGpu = i;
  }

  const peaksCpu = findPeakPeriods(cpuSamples), peaksGpu = findPeakPeriods(gpuSamples);
  const peakPeriodsMatch = peaksCpu.length === peaksGpu.length
    && peaksCpu.every((p, i) => p === peaksGpu[i]);

  // The stored samples came from the GPU iff that path was on for the capture
  // that produced them -- in which case `coarseSamples`' own scores are not an
  // independent baseline. The recomputed cpuSamples above are, which is why the
  // comparison uses those; this flag just says whether the STORED ones agreed.
  const hadCpuBaseline = coarseSamples.every((s, i) => Math.abs(s.score - cpuSamples[i].score) < 1e-12);

  return {
    hadCpuBaseline,
    candidates: cpuSamples.length,
    lines: { row: rowValues.length, col: colValues.length },
    maxAbsDelta, meanAbsDelta: sumAbs / cpuSamples.length, maxRelDelta,
    argmaxPeriodCpu: cpuSamples[argmaxCpu].period,
    argmaxPeriodGpu: gpuSamples[argmaxGpu].period,
    argmaxAgrees: argmaxCpu === argmaxGpu,
    peakCountCpu: peaksCpu.length, peakCountGpu: peaksGpu.length, peakPeriodsMatch,
    cpuMs, gpuMs,
  };
}
