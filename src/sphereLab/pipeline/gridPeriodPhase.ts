import * as THREE from 'three';
import { CameraSettingsCommon } from '../camera/settings.ts';
import { cornerDir } from '../math/geometry.ts';
import { computeCompositeLines, computeJoinWalk, computeMergeGroups } from './bucketFillJoin.ts';
import { computeBucketFillRegions } from './bucketFillSegments.ts';
import { computeEffectiveGradientField, computeGradientAgreementField, computeGradientField } from './gradientField.ts';
import { computeTopGradientAlpha } from './gradientHighlight.ts';

// ── Grid period/phase recovery from composite lines (pure) ────────────────
//
// Given the orientation already recovered by fitPairOfPlanes (Drow, Dcol,
// Dnormal), figures out WHICH row/column of the physical grid each detected
// composite line actually is -- the piece fitPairOfPlanes doesn't attempt
// (it only recovers the grid's ORIENTATION, not its registration/spacing).
//
// The derivation (worked out in conversation, not copied from any existing
// reference) has two equivalent forms: one using a line's own vote NORMAL,
// one using an actual POINT/ray on the line, gnomonically projected onto the
// plane tangent to the sphere at -Dnormal. Both are affine-linear in the
// line's true physical offset, but they're numerically DIFFERENT quantities
// (different slope/intercept) -- so this file deliberately uses ONLY the
// point-projection form throughout, via `gnomonic()` below, even though the
// normal-vector form is still used to CLASSIFY a line as row/column-type
// (that test only cares about orientation, not position, so either form
// works there identically). Using one consistent form everywhere means the
// period/phase values computed here land in the exact same coordinate space
// a caller would use to draw the rectified composite lines and a derived
// sample lattice -- mixing the two forms would silently misalign them.
//
// gnomonic(r) = (-(r.Drow)/(r.Dnormal), -(r.Dcol)/(r.Dnormal)) for a ray r:
// every point on a given row line shares the same xCol (that's what makes a
// row line project to a straight, constant-xCol line at all), and that
// shared xCol is affine-linear in the row's true offset -- symmetrically,
// xRow for column lines. That's "period and phase" reduced to: classify
// each line (free, same |n.Drow| vs |n.Dcol| test the plane-pair fit itself
// relies on), gnomonically project one representative point per line, then
// find the period/phase of the resulting 1D point set.
//
// Finding that period is NOT a further closed-form step (frequency
// parameters don't reduce to one polynomial eigen-solve the way
// fitPairOfPlanes' orientation did) -- it's a small, deliberately NARROW
// bracketed search around a pairwise-gap seed estimate, scored at each
// candidate period by the weighted circular-mean resultant length (the same
// "fold onto a circle, measure how tightly it clusters" construction
// already used by bucketFillJoin.ts's groupDisplayColors for hue blending,
// and by computeGradientAgreementField's own double-angle folding). Row and
// column periods are pooled into one shared search (square cells force the
// same physical period on both axes), which sharpens the search and is more
// robust than fitting each axis independently.

export interface GnomonicPoint { xRow: number; xCol: number }

export interface GridLineSample {
  root: number; // this line's merge-group root (pipeline/bucketFillJoin.ts) -- lets a caller map a sample back to its actual composite line/pixels
  value: number; // rectified periodic coordinate -- dimensionless (a ratio of unit-vector dot products, effectively tan of an angle), NOT pixels or degrees. xCol for a row line, xRow for a column line (see gnomonic() below)
  weight: number; // projected arc length, same weighting fitPairOfPlanes already uses for its own votes
  index: number; // recovered integer row/column index: round((value - phase) / period)
  p1: GnomonicPoint; p2: GnomonicPoint; // this line's own two endpoints, gnomonically projected -- for drawing the rectified line directly, in the SAME coordinate space `value`/period/phase are expressed in
}

export interface PeriodSearchSample { period: number; score: number }

export interface GridPeriodPhaseResult {
  period: number;
  phiRow: number;
  phiCol: number;
  height: number | null; // cellPitch / period, null if cellPitch wasn't supplied
  rowLines: GridLineSample[];
  colLines: GridLineSample[];
  // Everything a debug visualization needs to show its work -- the pooled
  // pairwise gaps (for a histogram), the seed period derived from them, the
  // search bracket, and every coarse sample's own (period, score) pair (for
  // plotting the search curve and marking where it landed).
  debug: {
    pooledGaps: number[];
    seedPeriod: number;
    bracket: [number, number];
    coarseSamples: PeriodSearchSample[];
  };
}

// Folds `values` onto the unit circle at the given candidate `period` and
// returns how tightly they cluster (resultant, 0..1: 0 = no periodicity at
// this period, 1 = perfect) and the weighted circular-mean phase, in the
// SAME units as `values` (not radians -- inverted back through the same
// theta = 2*pi*value/period relationship used to fold them in).
// Exported so overlays/gridPeriodPhaseOverlays.ts can extend the search
// curve visualization outside the actual searched bracket PURELY for
// display (e.g. when the user pans/zooms past it) without duplicating this
// math or pretending it was ever part of the real bracketed search.
export function circularFit(values: number[], weights: number[], period: number): { resultant: number; phase: number } {
  let sumCos = 0, sumSin = 0, sumW = 0;
  for (let i = 0; i < values.length; i++) {
    const theta = (2 * Math.PI * values[i]) / period;
    sumCos += weights[i] * Math.cos(theta);
    sumSin += weights[i] * Math.sin(theta);
    sumW += weights[i];
  }
  if (sumW < 1e-9) return { resultant: 0, phase: 0 };
  const resultant = Math.hypot(sumCos, sumSin) / sumW;
  const phaseTheta = Math.atan2(sumSin, sumCos);
  const phase = (phaseTheta * period) / (2 * Math.PI);
  return { resultant, phase };
}

// Gnomonic projection of a ray direction onto the plane tangent to the unit
// sphere at -Dnormal ("the bottom of the sphere") -- Drow/Dcol double as the
// tangent plane's own in-plane basis for free, since {Drow,Dcol,Dnormal}
// being orthonormal means "perpendicular to Dnormal" already IS span{Drow,
// Dcol}. null for a ray nearly parallel to the tangent plane (r.Dnormal ~
// 0), which shouldn't happen for a real ray toward the floor but is checked
// defensively.
function gnomonic(r: THREE.Vector3, Drow: THREE.Vector3, Dcol: THREE.Vector3, Dnormal: THREE.Vector3): GnomonicPoint | null {
  const dn = r.dot(Dnormal);
  if (Math.abs(dn) < 1e-9) return null;
  return { xRow: -r.dot(Drow) / dn, xCol: -r.dot(Dcol) / dn };
}

export function computeGridPeriodPhase(
  settings: CameraSettingsCommon,
  gray: Float64Array, w: number, h: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
  Drow: THREE.Vector3, Dcol: THREE.Vector3, Dnormal: THREE.Vector3,
  cellPitch: number | null,
): GridPeriodPhaseResult | null {
  // Rebuilds bucket-fill segments -> join walk -> merge groups -> composite
  // lines from scratch (same steps computeSegmentVotes already runs) rather
  // than threading identity through the existing anonymous Vote[] path --
  // keeps this module fully self-contained and leaves the working vote/fit
  // pipeline untouched. The extra recompute only happens while this debug
  // pipeline is actually enabled.
  const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
  const agreement = computeGradientAgreementField(field, Math.round(settings.coherenceRadius));
  const effective = computeEffectiveGradientField(field, agreement);
  const seedEligible = computeTopGradientAlpha(effective, 0, 100);
  const { regionId, segments } = computeBucketFillRegions(effective, settings.bucketFillToleranceDeg, seedEligible, settings.bucketFillMagnitudeThreshold, settings.bucketFillMaxSteps);
  const { merges } = computeJoinWalk(
    segments, regionId, w, h, settings.bucketFillMergeMinSimilarity, settings.bucketFillJoinSteps, settings.bucketFillMinLengthPx,
    settings.bucketFillMaxTravelFactor,
  );
  const groupOf = computeMergeGroups(segments.length, merges);
  const composites = computeCompositeLines(segments, groupOf);

  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];

  // Step 1+2: classify each composite line as row- or column-type via its
  // own vote normal (orientation only -- same |n.Drow| vs |n.Dcol| test the
  // plane-pair fit itself relies on), then rectify to one scalar via the
  // gnomonic projection of its own two endpoints -- NOT the normal-vector
  // ratio, see this file's header for why the two forms don't mix.
  const rowSamples: { root: number; value: number; weight: number; p1: GnomonicPoint; p2: GnomonicPoint }[] = [];
  const colSamples: { root: number; value: number; weight: number; p1: GnomonicPoint; p2: GnomonicPoint }[] = [];
  for (const [root, line] of composites) {
    const [u1, v1] = toNDC(line.x1, line.y1);
    const [u2, v2] = toNDC(line.x2, line.y2);
    const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
    const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
    const nRaw = ray1.clone().cross(ray2);
    const arcLen = nRaw.length();
    if (arcLen < 1e-12) continue;
    const n = nRaw.divideScalar(arcLen);

    const p1 = gnomonic(ray1, Drow, Dcol, Dnormal);
    const p2 = gnomonic(ray2, Drow, Dcol, Dnormal);
    if (!p1 || !p2) continue; // ray parallel to the tangent plane -- shouldn't happen for a real floor ray, skip defensively

    if (Math.abs(n.dot(Drow)) < Math.abs(n.dot(Dcol))) {
      // Row-type: every point on this line shares the same xCol (that's
      // what "projects to a straight, constant-xCol line" means) --
      // averaging both endpoints is a cheap noise-robustness bonus over
      // picking just one, since in theory they'd already be identical.
      rowSamples.push({ root, value: (p1.xCol + p2.xCol) / 2, weight: arcLen, p1, p2 });
    } else {
      colSamples.push({ root, value: (p1.xRow + p2.xRow) / 2, weight: arcLen, p1, p2 });
    }
  }
  if (rowSamples.length === 0 && colSamples.length === 0) return null;

  // Step 3: seed a period guess from the MODE of the pooled pairwise-gap
  // distribution (row and column values pooled together -- same physical
  // period forced on both axes by square cells), not the smallest few gaps
  // directly. A near-duplicate line detection (e.g. a segment the join walk
  // didn't fully merge -- two composite lines that are really the same
  // physical grid line) produces a near-zero, essentially random gap that
  // would completely dominate a "smallest few" average -- found live via
  // dev-bridge: one such gap corrupted the seed by ~500x. The true period,
  // in contrast, is shared by many genuinely-adjacent pairs (n-1 of them
  // for n evenly spaced lines, more than any other spacing achieves), so it
  // shows up as the single most heavily-populated bin in a coarse histogram
  // of ALL pairwise gaps -- isolated duplicate-noise gaps scatter across
  // many separate near-empty bins instead of piling into one, so a handful
  // of them being individually tiny doesn't let them win the mode.
  const allValues = [...rowSamples.map((s) => s.value), ...colSamples.map((s) => s.value)];
  const pooledGaps: number[] = [];
  for (let i = 0; i < allValues.length; i++) {
    for (let j = i + 1; j < allValues.length; j++) {
      const g = Math.abs(allValues[i] - allValues[j]);
      if (g > 1e-9) pooledGaps.push(g);
    }
  }
  if (pooledGaps.length === 0) return null;
  pooledGaps.sort((a, b) => a - b);
  const maxGap = pooledGaps[pooledGaps.length - 1];
  if (maxGap < 1e-9) return null;
  const HIST_BINS = Math.min(1000, Math.max(20, Math.floor(pooledGaps.length / 10)));
  const histCounts = new Array(HIST_BINS).fill(0);
  for (const g of pooledGaps) {
    const bi = Math.min(HIST_BINS - 1, Math.floor((g / maxGap) * HIST_BINS));
    histCounts[bi]++;
  }
  let modeBin = 0;
  for (let i = 1; i < HIST_BINS; i++) if (histCounts[i] > histCounts[modeBin]) modeBin = i;
  const binLo = (modeBin / HIST_BINS) * maxGap, binHi = ((modeBin + 1) / HIST_BINS) * maxGap;
  const modeGaps = pooledGaps.filter((g) => g >= binLo && g < binHi);
  const seedPeriod = modeGaps.reduce((a, b) => a + b, 0) / modeGaps.length;
  if (seedPeriod < 1e-9) return null;

  // Step 4: bracketed coarse-to-fine search for the period, scored by the
  // COMBINED (row + column) circular resultant -- deliberately narrow
  // (0.5x-1.5x the seed) so the search structurally never evaluates the
  // sub-multiple periods (P0/2, P0/3, ...) that would otherwise alias as
  // false peaks -- every true lattice point trivially also sits on any
  // finer sub-lattice, so a wide/blind search risks locking onto one of
  // those instead of the true, fundamental period.
  const bracket: [number, number] = [seedPeriod * 0.5, seedPeriod * 1.5];
  const COARSE_SAMPLES = 40;
  const rowValues = rowSamples.map((s) => s.value), rowWeights = rowSamples.map((s) => s.weight);
  const colValues = colSamples.map((s) => s.value), colWeights = colSamples.map((s) => s.weight);
  const coarseSamples: PeriodSearchSample[] = [];
  let bestP = seedPeriod, bestScore = -1;
  for (let i = 0; i < COARSE_SAMPLES; i++) {
    const P = bracket[0] + ((bracket[1] - bracket[0]) * i) / (COARSE_SAMPLES - 1);
    const rowFit = circularFit(rowValues, rowWeights, P);
    const colFit = circularFit(colValues, colWeights, P);
    const score = rowFit.resultant + colFit.resultant;
    coarseSamples.push({ period: P, score });
    if (score > bestScore) { bestScore = score; bestP = P; }
  }
  // Parabolic polish around the winning coarse sample using its two
  // neighbors -- cheap precision refinement, no extra full search needed.
  const bestIdx = coarseSamples.findIndex((s) => s.period === bestP);
  if (bestIdx > 0 && bestIdx < coarseSamples.length - 1) {
    const p0 = coarseSamples[bestIdx - 1], p1 = coarseSamples[bestIdx], p2 = coarseSamples[bestIdx + 1];
    const denom = p0.score - 2 * p1.score + p2.score;
    if (Math.abs(denom) > 1e-12) {
      const offset = (0.5 * (p0.score - p2.score)) / denom;
      const refinedP = p1.period + offset * (p1.period - p0.period);
      if (refinedP > bracket[0] && refinedP < bracket[1]) {
        const rowFit = circularFit(rowValues, rowWeights, refinedP);
        const colFit = circularFit(colValues, colWeights, refinedP);
        const refinedScore = rowFit.resultant + colFit.resultant;
        if (refinedScore > bestScore) { bestScore = refinedScore; bestP = refinedP; }
      }
    }
  }

  // Step 5: final phases at the winning period.
  const finalRow = circularFit(rowValues, rowWeights, bestP);
  const finalCol = circularFit(colValues, colWeights, bestP);

  // Step 6: assign an integer row/column index to every detected line, and
  // convert the period to a camera-height estimate if the physical cell
  // pitch is known (h = cellPitch / period).
  const rowLines: GridLineSample[] = rowSamples.map((s) => ({
    root: s.root, value: s.value, weight: s.weight, index: Math.round((s.value - finalRow.phase) / bestP), p1: s.p1, p2: s.p2,
  }));
  const colLines: GridLineSample[] = colSamples.map((s) => ({
    root: s.root, value: s.value, weight: s.weight, index: Math.round((s.value - finalCol.phase) / bestP), p1: s.p1, p2: s.p2,
  }));

  return {
    period: bestP, phiRow: finalRow.phase, phiCol: finalCol.phase,
    height: cellPitch !== null ? cellPitch / bestP : null,
    rowLines, colLines,
    debug: { pooledGaps, seedPeriod, bracket, coarseSamples },
  };
}
