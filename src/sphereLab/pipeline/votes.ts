import * as THREE from 'three';
import { CameraSettingsCommon } from '../camera/settings.ts';
import { jacobiEigenSymmetric, smallestEigenvector } from '../../linalg.ts';
import { cornerDir } from '../math/geometry.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { GradientField, Vote } from '../types.ts';
import { CompositeLine, compositeLineLength, computeCompositeLines, computeJoinWalk, computeMergeGroups } from './bucketFillJoin.ts';
import { computeEffectiveGradientField, computeGradientAgreementField, computeGradientField } from './gradientField.ts';
import { computeLsdRectangles, computeLsdRectanglesAuto, lsdRectanglesToBucketFillShape } from './lsdSegments.ts';
import { fourFoldResidual } from './orientationLM.ts';
import { guidedTangentDirectionForWalk } from './tangentWalk.ts';

// gray is expected to already be captureDistortedGrayscale's output.
export function computeWorldVotes(
  settings: CameraSettingsCommon,
  gray: Float64Array, w: number, h: number,
  gradientRadius: number, agreementRadius: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Vote[] {
  const votes: Vote[] = [];
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  const gradSpan = spanStart('gradientField');
  const field = computeGradientField(gray, w, h, gradientRadius);
  spanEnd(gradSpan);
  const agreeSpan = spanStart('agreementField');
  const agreement = computeGradientAgreementField(field, agreementRadius);
  spanEnd(agreeSpan);
  const effSpan = spanStart('effectiveField');
  const effective = computeEffectiveGradientField(field, agreement);
  spanEnd(effSpan);
  const { fx, fy, r } = effective;
  const walkSpan = spanStart('walk+vote loop');
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      if (fx[i] === 0 && fy[i] === 0) continue;
      const walked = guidedTangentDirectionForWalk(settings, fx, fy, w, h, x, y, fx[i], fy[i]);
      let theta = Math.atan2(walked.fy, walked.fx);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      const tdx = -Math.sin(theta), tdy = Math.cos(theta);
      const [u1, v1] = toNDC(x, y);
      const [u2, v2] = toNDC(x + tdx, y + tdy);
      const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
      const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
      const n = ray1.clone().cross(ray2);
      if (n.lengthSq() < 1e-12) continue;
      n.normalize();
      votes.push({ n, weight: Math.hypot(walked.fx, walked.fy) });
    }
  }
  spanEnd(walkSpan);
  return votes;
}

// Alternative to computeWorldVotes above: one vote per pixel, cast straight
// off computeGradient2x2Field's own per-pixel direction -- no tangent walk,
// no agreement/effective-field pass. Those existed to fight two problems
// (per-pixel noise, and corners contributing a spurious ~45-degree-off-pole
// direction) that refineOrientationIRLS's residual-based reweighting below
// is meant to handle instead, directly against the emerging fit rather than
// via a separately-tuned local-neighborhood field -- see this session's
// chat. Takes the field pre-computed (same reasoning as
// computeGradient2x2Composites above) so the caller picks CPU/GPU without
// this needing to know which.
export function computePixelVotes2x2(
  field: GradientField, w: number, h: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Vote[] {
  const { fx, fy, r } = field;
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  const votes: Vote[] = [];
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const gx = fx[i], gy = fy[i];
      if (gx === 0 && gy === 0) continue;
      let theta = Math.atan2(gy, gx);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      const tdx = -Math.sin(theta), tdy = Math.cos(theta);
      const [u1, v1] = toNDC(x, y);
      const [u2, v2] = toNDC(x + tdx, y + tdy);
      const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
      const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
      const n = ray1.clone().cross(ray2);
      if (n.lengthSq() < 1e-12) continue;
      n.normalize();
      votes.push({ n, weight: Math.hypot(gx, gy) });
    }
  }
  return votes;
}

// Just the LSD/join-walk tuning knobs computeGradient2x2Composites/
// compositesFromLsdRectangles actually read off `settings` -- narrowed off
// the full CameraSettingsCommon (which also carries dozens of display-only
// toggles a real Camera has but a bare phone-side PoseComputeState never
// will) so both functions stay callable with either a real camera's
// settings OR pipeline/poseCompute.ts's PoseComputeState.settings, which
// only carries this subset. A real CameraSettingsCommon still satisfies
// this structurally (extra fields are fine), so every existing desktop call
// site is unaffected.
export interface LsdCompositeSettings {
  lsdToleranceDeg: number; lsdRhoNoiseThreshold: number; lsdRhoHighThreshold: number; lsdCclSteps: number;
  lsdMinRegionSize: number;
  lsdNfaEpsilon: number;
  lsdNfaTestExponent: number; lsdMaxRetries: number; lsdRetryToleranceFactor: number; lsdRetryShrinkFraction: number;
  lsdMergeMinSimilarity: number; lsdJoinSteps: number; lsdMinLengthPx: number; lsdMaxTravelFactor: number;
}

// One composite LINE per LSD-rectangle-and-JOIN-WALK merge group (see
// pipeline/lsdSegments.ts and pipeline/bucketFillJoin.ts), tagged with that
// group's root -- the shared first stage behind computeSegmentVotes below
// AND pipeline/gridPeriodPhase.ts. Runs the full traditional LSD pipeline
// (region growing -> rectangle fit -> NFA validation, see
// pipeline/lsdSegments.ts's own header) over the 2x2 gradient field `field`
// (computeGradient2x2Field or its GPU twin, computeGradient2x2FieldGPU --
// see pipeline/axesReconstruction.ts's gradient2x2Field for the CPU/GPU
// choice), adapts the accepted rectangles into the join walk's expected
// input shape, then runs the join walk (merge groups + composite lines).
// Deliberately NOT the radius-driven gradient field x local-agreement
// "effective" field this used to run through (computeGradientField/
// computeGradientAgreementField/computeEffectiveGradientField, still
// defined in ./gradientField.ts for computeWorldVotes and other debug
// views, just not called from here) -- see this session's chat for why.
// Pulled out as its own function (and called exactly ONCE per
// reconstruction pass, see pipeline/axesReconstruction.ts) so every
// downstream consumer -- vote casting here, row/col family classification
// in gridPeriodPhase.ts, and the "color composite lines by row/col family"
// debug overlay -- sees the exact same lines under the exact same root
// numbers. Takes the field pre-computed (rather than computing it from
// `gray` itself) so the caller can pick CPU or GPU without this function --
// or any of its downstream consumers -- needing to know or care which one
// produced it.
export async function computeGradient2x2Composites(
  settings: LsdCompositeSettings,
  field: GradientField, w: number, h: number,
): Promise<{ root: number; line: CompositeLine }[]> {
  const rects = await computeLsdRectanglesAuto(field, {
    toleranceDeg: settings.lsdToleranceDeg,
    rhoNoiseThreshold: settings.lsdRhoNoiseThreshold,
    rhoHighThreshold: settings.lsdRhoHighThreshold,
    cclSteps: settings.lsdCclSteps,
    minRegionSize: settings.lsdMinRegionSize,
    nfaEpsilon: settings.lsdNfaEpsilon,
    nfaTestExponent: settings.lsdNfaTestExponent,
    maxRetries: settings.lsdMaxRetries,
    retryToleranceFactor: settings.lsdRetryToleranceFactor,
    retryShrinkFraction: settings.lsdRetryShrinkFraction,
  });
  return compositesFromLsdRectangles(rects, w, h, settings);
}

// The join-walk half of computeGradient2x2Composites, split out so
// overlays/lsdOverlay.ts's live debug preview (which already has its own
// freshly-computed LsdRectangle[] for the rectangle/rejected/raw-region
// views) can produce composite lines from that SAME array instead of
// re-running stages 2-5 a second time just to get here.
export function compositesFromLsdRectangles(
  rects: ReturnType<typeof computeLsdRectangles>, w: number, h: number, settings: LsdCompositeSettings,
): { root: number; line: CompositeLine }[] {
  const { regionId, segments } = lsdRectanglesToBucketFillShape(rects, w, h);

  const { merges } = computeJoinWalk(
    segments, regionId, w, h, settings.lsdMergeMinSimilarity, settings.lsdJoinSteps, settings.lsdMinLengthPx,
    settings.lsdMaxTravelFactor,
  );
  const groupOf = computeMergeGroups(segments.length, merges);
  const composites = computeCompositeLines(segments, groupOf);

  const result: { root: number; line: CompositeLine }[] = [];
  for (const [root, line] of composites) {
    if (compositeLineLength(line) < settings.lsdMinLengthPx) continue; // an unmerged singleton whose own segment was already too short
    result.push({ root, line });
  }
  return result;
}

// Experimental alternative to computeWorldVotes: one vote per composite LINE
// (computeGradient2x2Composites above) instead of one per pixel -- casts
// a ray to each GROUP's composite endpoints, not each raw segment's own
// endpoints, so a chain of several short segments joined into one long line
// votes as ONE long, well-conditioned ray instead of several separate short,
// noisier ones.
export function computeSegmentVotes(
  composites: { root: number; line: CompositeLine }[],
  w: number, h: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Vote[] {
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  const votes: Vote[] = [];
  for (const { line } of composites) {
    const [u1, v1] = toNDC(line.x1, line.y1);
    const [u2, v2] = toNDC(line.x2, line.y2);
    const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
    const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
    const n = ray1.clone().cross(ray2);
    const arcLen = n.length(); // sin(angle between the two endpoint rays) -- the composite line's own PROJECTED ARC LENGTH on the unit sphere
    if (arcLen < 1e-12) continue;
    n.divideScalar(arcLen); // normalize using the length already computed, instead of a second hypot
    // Weighted by that same projected arc length, not pixel count or average
    // magnitude -- a short/close-together line gives a small, easily-
    // corrupted cross product; a long one gives a large, well-conditioned
    // one, which is exactly the property we want a fit-confidence weight to
    // track (same reasoning computeWorldVotes's own per-pixel cross product
    // relies on implicitly, just now visible as an explicit per-composite-
    // line quantity instead of buried in a 1-pixel step).
    votes.push({ n, weight: arcLen });
  }
  return votes;
}

// The TRUE [minPercent, maxPercent) band by magnitude rank, out of every vote.
export function votesInMagnitudeBand(votes: Vote[], minPercent: number, maxPercent: number): Vote[] {
  const sorted = Array.from(votes).sort((a, b) => b.weight - a.weight);
  const lo = Math.round(sorted.length * (minPercent / 100));
  const hi = Math.round(sorted.length * (maxPercent / 100));
  if (hi <= lo) return [];
  return sorted.slice(lo, hi);
}

// Shared by fitPairOfPlanes and refineOrientationIRLS below: accumulates
// the weighted 6x6 ATA scatter matrix ("pair of planes through the origin")
// from a set of votes, given an arbitrary per-vote weight function -- a
// plain magnitude-based sharpen for a one-shot fit, or a residual-against-
// the-current-fit reweight for an IRLS refinement pass.
function accumulateATA(votes: Vote[], weightOf: (v: Vote) => number): number[][] {
  const ATA: number[][] = Array.from({ length: 6 }, () => new Array(6).fill(0));
  for (const v of votes) {
    const w = weightOf(v);
    if (w <= 0) continue;
    const { n } = v;
    const row = [n.x * n.x, n.y * n.y, n.z * n.z, n.x * n.y, n.x * n.z, n.y * n.z];
    for (let a = 0; a < 6; a++) {
      const wra = w * row[a];
      for (let b = 0; b < 6; b++) ATA[a][b] += wra * row[b];
    }
  }
  return ATA;
}

// The eigendecomposition half of fitPairOfPlanes -- fixed-size (6x6 -> 3x3),
// independent of vote count, shared by every caller of accumulateATA above.
function solveQuadricFromATA(ATA: number[][]): { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null {
  const m = smallestEigenvector(ATA);
  const M = [
    [m[0], m[3] / 2, m[4] / 2],
    [m[3] / 2, m[1], m[5] / 2],
    [m[4] / 2, m[5] / 2, m[2]],
  ];
  const { values, vectors } = jacobiEigenSymmetric(M);
  let zeroIdx = 0;
  for (let i = 1; i < 3; i++) if (Math.abs(values[i]) < Math.abs(values[zeroIdx])) zeroIdx = i;
  const others = [0, 1, 2].filter((i) => i !== zeroIdx);
  const b1 = new THREE.Vector3(vectors[others[0]][0], vectors[others[0]][1], vectors[others[0]][2]);
  const b2 = new THREE.Vector3(vectors[others[1]][0], vectors[others[1]][1], vectors[others[1]][2]);
  const Dnormal = new THREE.Vector3(vectors[zeroIdx][0], vectors[zeroIdx][1], vectors[zeroIdx][2]).normalize();
  const Drow = b1.clone().add(b2);
  const Dcol = b1.clone().sub(b2);
  if (Drow.lengthSq() < 1e-9 || Dcol.lengthSq() < 1e-9) return null;
  return { Drow: Drow.normalize(), Dcol: Dcol.normalize(), Dnormal };
}

// Fits the degenerate quadric ("pair of planes through the origin") that
// best explains "every vote lies on one plane or the other" -- see
// pre-Stage-A history for the full derivation. `power` is the caller's
// current weightSharpenPower setting.
export function fitPairOfPlanes(votes: Vote[], power: number): { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null {
  let maxW = 0;
  for (const { weight } of votes) if (weight > maxW) maxW = weight;
  return solveQuadricFromATA(accumulateATA(votes, (v) => (maxW > 0 ? Math.pow(v.weight / maxW, power) : 0)));
}

// Iteratively reweighted least squares on top of fitPairOfPlanes: after the
// same single-shot fit above, every vote's angular residual against the
// CURRENT poles (fourFoldResidual, from orientationLM.ts -- exactly 0 when
// a vote's normal sits on either pole, +-1 at the worst 45-degree-off
// corner case) folds multiplicatively into its weight for the next refit.
// A vote that's genuinely row/col-aligned has its residual shrink toward 0
// as the fit sharpens across iterations and keeps its full weight; a corner
// vote's residual stays large and roughly constant, so it gets suppressed
// on its own -- no separately-tuned agreement/effective-field muting step
// needed. Stops early once Drow stops rotating by more than
// IRLS_CONVERGE_EPS between iterations (same convergence pattern
// orientationLM.ts's own LM loop uses), so `maxIterations` is a worst-case
// cap, not a fixed cost -- 0 skips refinement entirely, returning exactly
// fitPairOfPlanes' own result, for direct A/B comparison.
const IRLS_CONVERGE_EPS = 1e-5;
export function refineOrientationIRLS(
  votes: Vote[], power: number, maxIterations: number,
): { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3; iterations: number } | null {
  let maxW = 0;
  for (const { weight } of votes) if (weight > maxW) maxW = weight;
  const magnitudeWeight = (v: Vote) => (maxW > 0 ? Math.pow(v.weight / maxW, power) : 0);

  let fit = solveQuadricFromATA(accumulateATA(votes, magnitudeWeight));
  if (!fit) return null;

  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    const { Drow, Dcol } = fit;
    const residualWeight = (v: Vote) =>
      magnitudeWeight(v) * THREE.MathUtils.clamp(1 - Math.abs(fourFoldResidual(v.n, Drow, Dcol)), 0, 1);
    const next = solveQuadricFromATA(accumulateATA(votes, residualWeight));
    if (!next) break; // degenerate refit (e.g. reweighting zeroed out too much) -- keep the last good fit
    // Undirected angle (min against the negated axis too) -- a sign flip
    // between iterations is the same physical axis, not real movement, same
    // reasoning as axesReconstruction.ts's own axisErr helper.
    const angleChange = Math.min(next.Drow.angleTo(fit.Drow), next.Drow.angleTo(fit.Drow.clone().negate()));
    fit = next;
    if (angleChange < IRLS_CONVERGE_EPS) { iterations++; break; }
  }
  return { ...fit, iterations };
}

