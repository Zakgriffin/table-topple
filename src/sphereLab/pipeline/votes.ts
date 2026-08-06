import * as THREE from 'three';
import { jacobiEigenSymmetric, smallestEigenvector } from '../../linalg.ts';
import { cornerDir, fourFoldResidual } from '../math/geometry.ts';
import { FieldResidency } from '../pipelineGPU/fieldResidency.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { CompositeLine, GradientField, Vote } from '../types.ts';
import { computeLsdRectangles, runLsdChain } from './lsdSegments.ts';


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

// Just the LSD tuning knobs computeGradient2x2Composites/
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
  lsdNfaTestExponent: number;
  lsdMinLengthPx: number;
}

// One LINE per accepted LSD rectangle, tagged with its root -- the shared
// first stage behind computeSegmentVotes below AND pipeline/gridPeriodPhase.ts.
// Runs the full traditional LSD pipeline (region growing -> rectangle fit ->
// NFA validation, see pipeline/lsdSegments.ts's own header) over the gray `res`
// was created around, then turns each accepted rectangle into a line.
// It used to run a JOIN WALK between those two steps, merging collinear
// rectangles into composites; that is retired, see compositesFromLsdRectangles.
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
// numbers. It used to take the gradient field pre-computed, so that the caller
// could pick CPU or GPU without this function -- or any of its downstream
// consumers -- needing to know or care which one produced it. The residency
// now carries that same ignorance one stage earlier and better: runLsdChain
// computes the gradient too, on whichever side the toggle says, and holds
// fx/fy there. Nothing between here and the fitter has to name a side, and the
// gradient no longer has to land on the CPU just to be passed along.
export async function computeGradient2x2Composites(
  settings: LsdCompositeSettings,
  res: FieldResidency, w: number, h: number,
): Promise<{ root: number; line: CompositeLine }[]> {
  const rects = await runLsdChain(res, w, h, {
    toleranceDeg: settings.lsdToleranceDeg,
    rhoNoiseThreshold: settings.lsdRhoNoiseThreshold,
    rhoHighThreshold: settings.lsdRhoHighThreshold,
    cclSteps: settings.lsdCclSteps,
    minRegionSize: settings.lsdMinRegionSize,
    nfaEpsilon: settings.lsdNfaEpsilon,
    nfaTestExponent: settings.lsdNfaTestExponent,
  });
  // Its own span inside this function's: it walks all ~5200 rectangles and keeps
  // ~893, and until it was split out that walk was indistinguishable from the
  // chain that produced them. Contains no await, so its duration is host CPU.
  const filterSpan = spanStart('compositesFromLsdRectangles', true);
  const out = compositesFromLsdRectangles(rects, w, h, settings);
  spanEnd(filterSpan);
  return out;
}

// One line per ACCEPTED LSD rectangle -- its own two endpoints, no merging.
// Split out from computeGradient2x2Composites so overlays/lsdOverlay.ts's live
// debug preview (which already has its own freshly-computed LsdRectangle[] for
// the rectangle/rejected/raw-region views) can produce lines from that SAME
// array instead of re-running the chain a second time just to get here.
//
// ── The join walk was removed here, and it was a NO-OP when it went ──
//
// This used to merge rectangles into composite lines and emit one line per
// merge group. It ran at joinSteps 0, where the step loop never executes: every
// segment was already its own root and the length filter compared the same
// endpoint pair `r.length` compares below. So every line, root number and
// filter decision is unchanged by its removal.
//
// The consequence worth remembering is what that means for the OPEN question:
// "does voting per raw segment rather than per merged composite hurt the fit?"
// was never actually being tested, because at joinSteps 0 there was no merged
// composite to lose. See lsdSegments.ts's growth-rule block for the one place
// this genuinely bites -- a polarity flip mid-line yields two antiparallel
// halves that nothing reassembles.
export function compositesFromLsdRectangles(
  rects: ReturnType<typeof computeLsdRectangles>, w: number, h: number, settings: LsdCompositeSettings,
): { root: number; line: CompositeLine }[] {
  const result: { root: number; line: CompositeLine }[] = [];
  let root = 0;
  for (const r of rects) {
    if (!r.accepted) continue;
    const id = root++; // consumed even when the length filter drops this line, to keep numbering stable
    if (r.length < settings.lsdMinLengthPx) continue;
    const ax = Math.cos(r.theta), ay = Math.sin(r.theta);
    const hl = r.length / 2;
    result.push({
      root: id,
      line: { x1: r.cx + hl * ax, y1: r.cy + hl * ay, x2: r.cx - hl * ax, y2: r.cy - hl * ay },
    });
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
// CURRENT poles (fourFoldResidual, from math/geometry.ts -- exactly 0 when
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

