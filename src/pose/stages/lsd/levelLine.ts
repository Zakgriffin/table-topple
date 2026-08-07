import * as THREE from 'three';

// ── The level line: the definition the whole LSD stage is built on ────────
//
// This file exists because the block below is the thing every other module in
// this directory points AT. It used to sit a third of the way down a
// 1074-line the LSD stage, and half the comments in the stage said some
// variant of "see the level-line vector block". Now they can name a file.
//
// It holds exactly the shared rule and nothing else: the definition, the
// growth predicate built on it, the 8-neighborhood that predicate is applied
// over, and the eligibility test. Stage 2+3 (regions.cpu.ts) and stage 4+5
// (rectangles.cpu.ts) both depend on this; neither depends on the other.

// The level-line angle (LSD's own convention): the gradient rotated -90
// degrees, a DIRECTED angle -- it encodes which side is darker. The two edges
// of one bright stripe have level-line angles ~180 degrees apart.
//
// EVERY orientation comparison in this file is directed (mod 2π): growth
// compatibility, countRectanglePixels' NFA alignment count, and
// and the fitter's own alignment test all use a plain SIGNED cos-dot
// against cos(τ). This reverts a previous mod-π (theta and theta+π both count)
// experiment, and the reason for the revert is worth recording, because the
// original motivation for mod-π was real and is still real:
//
// WHY MOD-π WAS TRIED: a single De Bruijn grid line separates cells whose
// VALUES vary along its own length, so the exact same physical line can
// legitimately flip which side is darker partway along it. Directed growth
// splits that line in two at the flip.
//
// WHY IT WAS REVERTED ANYWAY: directed angles were silently buying a second
// thing nobody had accounted for -- FREE SEPARATION OF THE TWO EDGES OF A
// THIN STRIPE. Those two edges are geometrically parallel but antiparallel as
// vectors, so a directed test rejects merging them by definition. Under mod-π
// they are indistinguishable, and at a 1-2px stripe width they are 8-ADJACENT,
// so nothing stops them fusing into one wrongly-fattened region. The old
// strided search masked this by only ever probing along the tangent (it rarely
// OFFERED a perpendicular neighbor as a candidate), but that was an accident
// of the search geometry, not a guarantee -- and with growth now testing an
// honest 8-neighborhood at stride 1, the mask is gone.
//
// The two concerns are separable by WHERE the flip happens: the flip we need
// to tolerate happens ALONG a line, the flip we must reject sits ACROSS a
// stripe. That could be encoded as a mod-π test plus a "polarity may only flip
// for tangential connections" side condition -- but since bridging disjoint
// segments is bucketFillJoin.ts's job now (see this file's header), the
// simpler resolution is to not tolerate flips here at all. Which makes the
// growth rule a single signed dot product, and makes it canonical LSD's own
// rule again rather than a variant.
//
// THE STANDING CONSEQUENCE, and it is a real limitation rather than a note
// about a stage that used to follow: a line whose polarity flips mid-length
// comes out as two COLLINEAR, ABUTTING, ANTIPARALLEL segments, and NOTHING
// reassembles them. The join walk that was once responsible for merging them
// is deleted. Both halves still vote (each contributes its own weighted n n^T,
// and the sub-lengths nearly sum to the whole), so the orientation fit barely
// moves -- but each half is fitted from a shorter baseline, so its normal is
// noisier, and that error enters gridPeriodPhase's row/col classification
// quadratically. Grazing angles, where lines are already short, are where this
// would show up first.
//
// One simplification falls out for free: with no polarity flips inside a
// region, a plain raw sum of member level-line vectors can no longer partially
// cancel, so a region's mean direction needs none of the
// sign-resolve-against-a-fixed-reference machinery the mod-π version required.
//
// ── THE LEVEL-LINE VECTOR (the canonical definition -- comments elsewhere
//    point here) ──────────────────────────────────────────────────────────
//
// The level line at a pixel runs perpendicular to the gradient, i.e. a quarter
// turn from it. Written as a unit vector that is simply
//
//   (ux, uy) = (-fy, fx) / ‖(fx, fy)‖
//
// and that vector is what every consumer in this file actually wants.
//
// GET THE COMPONENT ORDER RIGHT -- it was wrong here from faf55f6 until
// 2026-08-04, and the way it was wrong is worth keeping because the same slip
// is one keystroke away at every site below. The angle this replaced was
// `theta = atan2(fx, -fy)`, and the vector was written as `(fx, -fy)` by
// reading those two arguments left to right. But atan2 takes Y FIRST, so
// (cos theta, sin theta) = (-fy, fx). The buggy form is the correct one with
// its components SWAPPED, i.e. reflected about the 45-degree line rather than
// rotated a quarter turn.
//
// WHY IT SURVIVED SO LONG, which is the genuinely instructive part: swapping
// components is an ORTHOGONAL map, and every growth test is a dot product
// between two pixels' vectors. dot(Mu, Mv) == dot(u, v) for orthogonal M, so
// region growing was BIT-IDENTICAL under the bug -- the clumps looked perfect,
// and faf55f6's own verification (0 edge-predicate flips over 187630 pairs,
// max dot delta 5.6e-16) was measuring a quantity that is INVARIANT under the
// error it was meant to catch. The CPU-vs-GPU harnesses were equally blind,
// since both sides carried the same reflection.
//
// It only became observable where a level-line vector meets something that is
// NOT reflected with it: the fitted rectangle's own axis, which comes from a
// PCA of member POSITIONS. countRectanglePixels' alignment count is that
// meeting point, and dot(Mu, a) == dot(u, Ma) means the count was scored
// against an axis mirrored about 45 degrees. A vertical edge (fx>0, fy=0) has
// level line (0,1) but scored as (1,0) -- exactly perpendicular, k=0, rejected
// every time. Only edges near +/-45 degrees, where the mirror is close to the
// identity, survived at all.
//
// It used to be materialized as an ANGLE: computeMagTheta stored
// theta = atan2(fx, -fy) per pixel, and then every single consumer immediately
// undid it. The grower precomputed cos(theta)/sin(theta) for every pixel;
// collect summed cos/sin over a region's members and atan2'd the result back
// into an angle; fitRectangle took cos/sin of THAT to get a direction again;
// countRectanglePixels took a cos/sin pair per pixel of every rectangle
// footprint. Not one site used the angle as an angle. So the angle is gone and
// the vector is carried directly, straight off the gradient field the previous
// stage already produces -- which deletes a whole per-pixel pass along with
// every transcendental in the chain.
//
// The one atan2 that remains is the one producing a fitted rectangle's OWN
// theta (see fitRectangle), because that is a real output votes.ts consumes
// as an angle.
//
// Two forms of the magnitude test show up below, and the split is deliberate:
// eligibility compares SQUARED magnitude against a squared threshold, so a
// pixel that fails costs no sqrt at all -- and most pixels fail -- while only a
// pixel that passes pays the single sqrt needed to normalize.

// The full 8-neighborhood. All eight are tested, not a steered subset: the
// old scheme picked a handful of long offsets aimed along the region's
// aggregate direction because most of 8 directions would have been wasted
// perpendicular probes at a large stride. At stride 1 that reasoning
// evaporates -- a perpendicular neighbor is one pixel away, testing it is
// free, and it is exactly the test that lets a genuinely 2px-thick ridge grow
// across its own width instead of splitting into two parallel components.
export const NEIGHBOR_DX = [1, 1, 0, -1, -1, -1, 0, 1];
export const NEIGHBOR_DY = [0, 1, 1, 1, 0, -1, -1, -1];

// THE predicate -- the single rule the whole of stage 2+3 is built on.
// Exported so overlays/lsdOverlay.ts's edge-connectivity hover view tests the
// exact same condition the real algorithm does, rather than an independently
// reimplemented copy that could quietly drift out of sync (the same reason
// the JFA version exported computeGrowthCandidates).
//
// Directed, so a plain SIGNED dot: cos(θi - θj) >= cos(τ). No abs(), which is
// what keeps the two antiparallel edges of a thin stripe from fusing -- see
// the level-line vector block (the LSD stage).
export function levelLinesCompatible(
  cosA: number, sinA: number, cosB: number, sinB: number, cosTol: number,
): boolean {
  return cosA * cosB + sinA * sinB >= cosTol;
}

// Which of pixel i's 8 neighbors it shares a graph edge with. Used by the
// hover debug view; the growth loop below inlines the same test rather than
// allocating an array per pixel per round.
export function computeEdgeNeighbors(
  fx: Float64Array, fy: Float64Array, w: number, h: number, i: number,
  toleranceDeg: number, rhoLow: number,
): number[] {
  const out: number[] = [];
  const mi = Math.hypot(fx[i], fy[i]);
  if (mi <= rhoLow) return out;
  const cosTol = Math.cos(THREE.MathUtils.degToRad(toleranceDeg));
  const x = i % w, y = (i / w) | 0;
  // The quarter turn plus normalize -- see the level-line vector block above.
  const ci = -fy[i] / mi, si = fx[i] / mi;
  for (let k = 0; k < 8; k++) {
    const nx = x + NEIGHBOR_DX[k], ny = y + NEIGHBOR_DY[k];
    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
    const j = ny * w + nx;
    const mj = Math.hypot(fx[j], fy[j]);
    if (mj <= rhoLow) continue;
    if (levelLinesCompatible(ci, si, -fy[j] / mj, fx[j] / mj, cosTol)) out.push(j);
  }
  return out;
}

// Squared eligibility threshold, so the per-pixel test below costs no sqrt.
// Exact rather than approximately equivalent: for rhoLow >= 0, ‖g‖ > rhoLow iff
// ‖g‖² > rhoLow², and the negative case (which no slider produces, but which
// would silently invert the test) is mapped to "everything is eligible", which
// is what comparing against a negative threshold means.
export function eligibilityThresholdSq(rho: number): number {
  return rho >= 0 ? rho * rho : -Infinity;
}
