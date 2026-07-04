// Level 3 of the line-based rectification redesign: turns two indexed line
// families into an actual homography, by treating every (row-line, col-line)
// crossing as a lattice corner with a KNOWN (relative) integer address —
// which is exactly the input format src/homography.ts's fitHomographyDLT
// already expects.
//
// This deliberately reuses fitHomographyDLT rather than hand-deriving a
// separate "scale from intersection spacing" formula: homography.ts's own
// comment on fitHomographyFromVPAndPatch explains that plain DLT was
// abandoned for the OLD architecture because its correspondences came from a
// small, spatially CLUSTERED local anchor patch, which barely constrains a
// homography's perspective terms. That objection doesn't apply here — these
// correspondences span however much of the grid Level 1 actually detected
// (a wide baseline by construction), which is exactly what DLT needs to be
// well-conditioned. If Level 1 only ever sees a tiny patch in practice this
// may need revisiting, but it should be validated against real detection
// spans, not assumed to fail the same way.

import type { LineCandidate } from './lines.ts';
import type { LineFamily, VanishingPoint } from './vp.ts';
import { crossLines, toAbsoluteLine, vpIsFinite, vpToPoint } from './vp.ts';
import type { PointCorrespondence } from './homography.ts';
import { smallestEigenvector } from './linalg.ts';

export interface IndexedLine { index: number; line: LineCandidate; }

// A 1D projective (Mobius) transform t = (p*v+q)/(r*v+s), up to overall
// scale of (p,q,r,s). This is EXACTLY what relates a family's true integer
// grid index v to its crossing position t along a fixed transversal: a
// homography restricted to any line is itself a 1D projective transform, and
// intersecting a pencil of lines with a transversal is exactly that
// restriction — same object as the 2D homography fit in homography.ts, one
// dimension down (a 2x2 matrix up to scale = 3 DOF, vs 3x3 = 8 DOF).
interface Mobius { p: number; q: number; r: number; s: number; }

function applyMobius(m: Mobius, v: number): number {
  return (m.p * v + m.q) / (m.r * v + m.s);
}
function invertMobius(m: Mobius, t: number): number {
  // t*(r*v+s) = p*v+q  =>  v*(t*r-p) = q-t*s
  return (m.q - t * m.s) / (t * m.r - m.p);
}

// Fits a Mobius transform from (v,t) correspondences via the same
// homogeneous-linear-system + smallest-eigenvector approach as
// estimateVanishingPoint/fitHomographyDLT: t*(r*v+s)-(p*v+q)=0 is linear in
// (p,q,r,s), so each correspondence contributes one row [v,1,-t*v,-t] to a
// 4-unknown homogeneous system. Needs at least 3 correspondences (matches
// the transform's 3 DOF); more just makes it a least-squares fit instead of
// exact.
function fitMobius(pairs: { v: number; t: number }[]): Mobius | null {
  if (pairs.length < 3) return null;
  const M = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (const { v, t } of pairs) {
    const row = [v, 1, -t * v, -t];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) M[i][j] += row[i] * row[j];
  }
  const [p, q, r, s] = smallestEigenvector(M);
  if (Math.abs(p * s - q * r) < 1e-9) return null; // degenerate (singular Mobius)
  return { p, q, r, s };
}

// Recovers each line's TRUE relative integer grid index from its position
// along a transversal — robust to Level 1 having MISSED some lines (a real,
// confirmed issue on real De Bruijn content: adjacent cells differ in bit
// value only ~50% of the time, so a true grid line can have too little
// visible edge to clear Level 1's peak threshold; see scripts/test-lines-decode.ts's
// investigation). Naively assuming "consecutive in sorted order == consecutive
// integers" breaks the instant one line is missing: every following index
// silently shifts, corrupting the whole correspondence set (confirmed via
// DLT residual jumping to 15-48px on real content, vs ~0 on gap-free
// synthetic tests).
//
// Fix: fit the Mobius model directly and INVERT it per line, rather than
// counting forward from a neighbor — a missing line then just means an
// integer is absent from the recovered set (harmless), not a corruption of
// everything after it. Bootstrapping the model (before any index is known)
// uses the same RANSAC shape as splitIntoTwoFamilies: try small candidate
// windows of the sorted sequence, fit a trial model from each, and see how
// many OTHER lines round-trip through it cleanly.
//
// A window can itself straddle a gap (or several), so it's NOT enough to
// assume each window is gap-free consecutive integers (0,1,2,3) — confirmed
// the hard way: with gaps spaced roughly one per 3-4 detected lines, EVERY
// possible 4-line window straddled at least one gap, so the gap-free
// assumption was never even available to try, and the least-bad wrong guess
// won by accident (a Mobius transform has enough freedom to fit 4 arbitrary
// points closely regardless, then extrapolate confidently wrong elsewhere).
// The fix: also search over small candidate step sizes (1, 2, or 3 missing
// lines) between each pair of consecutive seed lines, not just assume 1 —
// cheap (a few dozen combinations per window) since real consecutive misses
// should be rare.
//
// A DEEPER issue, found from real-footage testing (not just synthetic): "no
// gaps" and "a uniform pattern of missing lines" are only distinguishable by
// comparing raw inlier count when the data is exactly noiseless. Reproduced
// directly (scripts/test-lattice-alias-robustness.ts): under 1.5px realistic
// position noise, a genuinely GAP-FREE family gets mis-detected as aliased
// (step 2, 3, or more) over 50% of the time — different candidate windows
// fit INDEPENDENTLY, so under noise they aren't exactly tied in count, and
// the span-based tie-break above never gets a chance to fire before a
// sparser-but-noise-luckier candidate already won outright on raw count.
// This isn't a tunable-away edge case: from relative line positions ALONE,
// "1 step apart" and "2 steps apart with one missed" are mathematically
// indistinguishable — same Mobius shape, different assumed multiplier.
// Nothing in the (v,t) data itself can settle that.
//
// The fix needs INDEPENDENT information outside this family's own point
// positions: expectedSpacingPx (the coarse apparent-pitch estimate already
// computed for Hough bin-sizing, see src/main.ts) measures roughly how many
// pixels one real cell spans, via a totally different method (autocorrelation)
// that doesn't depend on which lines Level 1 happened to detect. A candidate
// model's IMPLIED per-step pixel spacing (measured directly from its seed
// window's own real data: total pixel span / total assumed index span)
// should roughly match that — an aliased "believes there's a missing line"
// model has an implied spacing that's a FRACTION (half,
// a third, ...) of the true cell size, which is directly checkable and
// rejected before it ever gets to compete on inlier count.
function recoverIndicesFromTransversal(
  scored: { line: LineCandidate; t: number }[], inlierPx: number, expectedSpacingPx?: number,
  maxGap = 3, scaleTolerance = 1.6, spanCapMultiplier = 3,
): { line: LineCandidate; index: number }[] {
  const n = scored.length;
  const WINDOW = 4;
  const GAP_CHOICES = Array.from({ length: maxGap }, (_, i) => i + 1); // [1, 2, ..., maxGap]
  if (n < WINDOW) {
    // Not enough lines to fit a Mobius model at all — fall back to the
    // naive consecutive assumption (only wrong if this tiny set itself has
    // a gap, which a 4-DOF model couldn't have caught anyway with this
    // little data).
    return scored.map((s, i) => ({ line: s.line, index: i }));
  }

  function scoreModel(m: Mobius): { inliers: boolean[]; count: number; totalResidual: number; span: number } {
    const inliers: boolean[] = [];
    const vs: number[] = [];
    let count = 0, totalResidual = 0;
    for (const s of scored) {
      const vRaw = invertMobius(m, s.t);
      const vRound = Math.round(vRaw);
      vs.push(vRound);
      const residual = Math.abs(applyMobius(m, vRound) - s.t);
      const ok = residual < inlierPx;
      inliers.push(ok);
      if (ok) { count++; totalResidual += residual; }
    }
    const inlierVs = vs.filter((_, i) => inliers[i]);
    const span = inlierVs.length ? Math.max(...inlierVs) - Math.min(...inlierVs) : Infinity;
    return { inliers, count, totalResidual, span };
  }

  let best: { model: Mobius; count: number; totalResidual: number; span: number } | null = null;
  for (let start = 0; start + WINDOW <= n; start++) {
    const window = scored.slice(start, start + WINDOW);
    for (const g1 of GAP_CHOICES) for (const g2 of GAP_CHOICES) for (const g3 of GAP_CHOICES) {
      const vs = [0, g1, g1 + g2, g1 + g2 + g3];
      const model = fitMobius(window.map((s, i) => ({ v: vs[i], t: s.t })));
      if (!model) continue;
      if (expectedSpacingPx !== undefined) {
        // Average spacing directly from the window's own REAL measured
        // positions (total pixel span / total assumed index span), not the
        // fitted model's analytic derivative — the latter trusts the tiny
        // 4-point fit's local curvature/pole location, which can be wildly
        // unstable exactly where it'd be evaluated (an interpolated
        // midpoint, not a real sample) even for an objectively correct gap
        // guess. This average is cruder but numerically robust, and the
        // tolerance band below is already loose enough not to need more.
        const impliedSpacing = Math.abs(window[3].t - window[0].t) / (vs[3] - vs[0]);
        const ratio = impliedSpacing / expectedSpacingPx;
        if (ratio < 1 / scaleTolerance || ratio > scaleTolerance) continue; // implausible scale -- almost certainly a step-size alias, not a real fit
      }
      const { count, totalResidual, span } = scoreModel(model);
      // A tiny 4-point fit can be near-degenerate (r close to 0, an almost-
      // linear map with implausibly broad "reach") and spuriously claim a
      // higher raw inlier count than any sane candidate simply by rounding
      // many real, unrelated t-values into SOME nearby integer over a huge
      // assumed span — found via scripts/test-lattice-alias-robustness.ts's
      // multi-gap scenario: the winning "best" candidate needed a span of
      // ~4900 to explain 21 real lines. A real family shouldn't need more
      // than a handful of missed lines per real one, so span is capped at a
      // generous multiple of the actual line count — this is a sanity bound
      // on the search, not a claim about real gap frequency.
      if (span > n * spanCapMultiplier) continue;
      const better = !best || count > best.count
        || (count === best.count && span < best.span)
        || (count === best.count && span === best.span && totalResidual < best.totalResidual);
      if (better) best = { model, count, totalResidual, span };
    }
  }
  if (!best) return scored.map((s, i) => ({ line: s.line, index: i })); // degenerate fallback

  // Refit using ALL inliers under the winning model (not just its 4-line
  // seed) for a more accurate final model, then assign final indices.
  const { inliers } = scoreModel(best.model);
  const inlierPairs = scored
    .map((s, i) => ({ s, i }))
    .filter(({ i }) => inliers[i])
    .map(({ s }) => ({ v: Math.round(invertMobius(best!.model, s.t)), t: s.t }));
  const finalModel = fitMobius(inlierPairs) ?? best.model;

  const withIndex = scored
    .map(s => ({ line: s.line, vRaw: invertMobius(finalModel, s.t), t: s.t }))
    .map(s => ({ line: s.line, index: Math.round(s.vRaw), residual: Math.abs(applyMobius(finalModel, Math.round(s.vRaw)) - s.t) }))
    .filter(s => s.residual < inlierPx * 3); // drop lines that don't fit ANY nearby integer well — likely spurious, non-grid detections rather than gaps

  const minIndex = Math.min(...withIndex.map(s => s.index));
  return withIndex.map(s => ({ line: s.line, index: s.index - minIndex }));
}

// Assigns each line in a family a consistent integer index (consecutive
// grid-row or grid-column number, WITH gaps where Level 1 missed a line —
// see recoverIndicesFromTransversal), by sorting them along a FIXED
// reference transversal: the line through the image center pointing toward
// the OTHER family's vanishing point. Any transversal recovers the correct
// ORDER (a pencil of concurrent lines crosses any fixed line in the same
// order as their angular order around the vertex) — the other family's VP
// direction is used specifically because it guarantees the transversal isn't
// close to parallel with THIS family's own lines, which would make
// intersections numerically unstable (or, exactly parallel, nonexistent).
export function indexFamilyLines(
  family: LineFamily, otherVp: VanishingPoint, w: number, h: number, inlierPx = 4, expectedSpacingPx?: number,
  maxGap = 3, scaleTolerance = 1.6, spanCapMultiplier = 3,
): IndexedLine[] {
  const cx = w / 2, cy = h / 2;
  let dx: number, dy: number;
  if (vpIsFinite(otherVp)) {
    const p = vpToPoint(otherVp);
    dx = p.x - cx; dy = p.y - cy;
  } else {
    dx = otherVp.x; dy = otherVp.y;
  }
  const len = Math.hypot(dx, dy) || 1;
  dx /= len; dy /= len;

  let normalAngle = Math.atan2(dy, dx) + Math.PI / 2;
  normalAngle = ((normalAngle % Math.PI) + Math.PI) % Math.PI;
  const refAbs = toAbsoluteLine({ theta: normalAngle, rho: 0, weight: 1 }, cx, cy);

  const scored = family.lines.map(line => {
    const l = toAbsoluteLine(line, cx, cy);
    const p = crossLines(l, refAbs);
    const px = p.x / p.w, py = p.y / p.w;
    const t = (px - cx) * dx + (py - cy) * dy;
    return { line, t };
  });
  scored.sort((a, b) => a.t - b.t);

  return recoverIndicesFromTransversal(scored, inlierPx, expectedSpacingPx, maxGap, scaleTolerance, spanCapMultiplier);
}

// Every (row-line, col-line) crossing is a lattice corner with a known
// (relative) integer address — builds the full correspondence set for
// fitHomographyDLT.
export function buildLatticeCorrespondences(
  rows: IndexedLine[], cols: IndexedLine[], w: number, h: number,
): PointCorrespondence[] {
  const cx = w / 2, cy = h / 2;
  const out: PointCorrespondence[] = [];
  for (const r of rows) {
    const lr = toAbsoluteLine(r.line, cx, cy);
    for (const c of cols) {
      const lc = toAbsoluteLine(c.line, cx, cy);
      const p = crossLines(lr, lc);
      if (Math.abs(p.w) < 1e-9) continue; // parallel (degenerate) — skip
      out.push({ u: r.index, v: c.index, x: p.x / p.w, y: p.y / p.w });
    }
  }
  return out;
}
