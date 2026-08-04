import * as THREE from 'three';
import { BucketFillSegment } from './bucketFillSegments.ts';
import { FieldResidency } from '../pipelineGPU/fieldResidency.ts';
import { computeGradient2x2FieldGPU } from '../pipelineGPU/gradient2x2.ts';
import { fitAndTestRegionsGPU } from '../pipelineGPU/lsdFit.ts';
import { growRegionsCCLGPU } from '../pipelineGPU/growRegions.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { globalState } from '../state.ts';
import { GradientField } from '../types.ts';
import { computeGradient2x2Field } from './gradientField.ts';

// ── LSD (Line Segment Detector, von Gioi/Jakubowicz/Morel/Randall 2010) ───
// ── from scratch ────────────────────────────────────────────────────────
//
// A genuinely traditional reimplementation. This is now the PRODUCTION
// composite-line source -- pipeline/votes.ts's computeGradient2x2Composites
// calls lsdRectanglesToBucketFillShape (bottom of this file) to feed
// pipeline/bucketFillJoin.ts's join walk, replacing pipeline/
// bucketFillSegments.ts's own BFS growing, which stays defined and correct
// but is no longer invoked from anywhere in the live pipeline -- kept in
// the repo as a reference/comparison implementation, not deleted.
//
// Pipeline: hysteresis-gated, directed-angle CONNECTED-COMPONENT region
// growing (stage 2+3, see growRegionsCCL below -- replaces the JFA-strided
// competitive relabeling that replaced the original magnitude-sorted serial
// BFS) -> magnitude-weighted PCA rectangle fit per region (stage 4) -> NFA
// statistical validation with a bounded tighten-then-shrink retry loop
// (stage 5).
//
// This file now forms segments ONLY. Bridging genuinely disjoint segments
// (across a gradient dropout, or across a level-line POLARITY flip -- see
// below) is exclusively pipeline/bucketFillJoin.ts's job. The previous design
// tried to do both at once via long strided jumps, which is where all of its
// complexity and all of its failure modes lived: a jump of tens or hundreds
// of pixels only checks ANGLE agreement, so it could silently land on a
// completely different parallel ridge (an adjacent De Bruijn grid row) that
// merely shares a direction, and a bad merge like that compounds. Every pixel
// comparison here is now strictly between 8-NEIGHBORS, so that class of error
// is not merely guarded against, it is unreachable.
//
// GPU-friendliness note: stage 4+5's FIRST NFA pass HAS a GPU port
// (pipelineGPU/lsdFit.ts), and it is VERIFIED identical to this file's CPU
// path -- zero disagreements on n, k or accept/reject across a 2931-region
// capture, max nfaLog10 delta 7.7e-6 (see pipelineGPU/lsdFitVerify.ts, which
// is how to re-check it after any change to either side). It defaults OFF
// anyway, for a PERFORMANCE reason: fitAndTestRegionsGPU still uploads
// mag/theta and the region CSR from CPU on every call, so it currently ADDS a
// round trip and measures slower (3.5ms CPU vs 8ms GPU). That inverts once
// stage 2+3 is GPU-resident and the labeling never lands on CPU.
// Stage 2+3 ALSO has a GPU port now (pipelineGPU/growRegions.ts, toggled by
// globalState.useGPUGrowRegions, dispatched independently of the fit stage in
// computeLsdRectanglesAuto). Like the JFA version it replaces, it was
// architecturally GPU-ready by construction -- each round is two
// frozen-buffer-in/fresh-buffer-out passes with no same-round cross-pixel
// dependency -- and it needed strictly LESS GPU machinery than that version
// would have: the per-round segmented reduction over region sums is gone
// entirely, leaving only the propagate step itself. It is the one stage that is
// NOT bit-identical to its CPU counterpart and cannot be made so; see its own
// header, and pipelineGPU/growRegionsVerify.ts for how to measure the exposure
// on a given capture.
//
// Stage 5's retry loop (tighten-then-shrink on NFA rejection) is RETIRED --
// fitRegionOnce is the live fitter, fitRegionWithRetries stays below as
// unreferenced reference code. It was never ported (retry 2+ needs a per-region
// partial sort, the hardest GPU problem in this file), and keeping it CPU-side
// meant every GPU-rejected region fell back to CPU and dragged mag/theta along
// with it -- the last dependency preventing stages 1-4 from becoming one
// GPU-resident run. Both fitters are now attempt-0-only, so the CPU and GPU
// paths have identical scope by construction rather than by pinning a setting.
//
// Regions below settings.minRegionSize are dropped in
// collectRegionsFromLabels, before either fitter sees them. At the default
// floor of 2 that is free (a 1-member region has no axis to fit) and removes
// ~40% of the fit stage's input on a real capture.

// The level-line angle (LSD's own convention): the gradient rotated -90
// degrees, a DIRECTED angle -- it encodes which side is darker. The two edges
// of one bright stripe have level-line angles ~180 degrees apart.
//
// EVERY orientation comparison in this file is directed (mod 2π): growth
// compatibility, countRectanglePixels' NFA alignment count, and
// fitRegionWithRetries' retry-1 refilter all use a plain SIGNED cos-dot
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
// simpler resolution is to not tolerate flips here at all and let the join
// walk reassemble them. Which makes the growth rule a single signed dot
// product, and makes it canonical LSD's own rule again rather than a variant.
//
// CONSEQUENCE FOR THE JOIN WALK: a line whose polarity flips mid-length now
// arrives at bucketFillJoin.ts as two COLLINEAR, ABUTTING, ANTIPARALLEL
// segments. Merging those is a hard requirement on that stage, not an
// optional nicety -- if its merge test rejects antiparallel candidates, these
// lines stay fragmented with nothing downstream to recover them.
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
// theta (see fitRectangle), because that is a real output the join walk
// downstream consumes as an angle.
//
// Two forms of the magnitude test show up below, and the split is deliberate:
// eligibility compares SQUARED magnitude against a squared threshold, so a
// pixel that fails costs no sqrt at all -- and most pixels fail -- while only a
// pixel that passes pays the single sqrt needed to normalize.

// ── Stage 2+3: directed connected-component region growing ───────────────
//
// The segments are, by definition, the CONNECTED COMPONENTS of one fixed
// undirected graph: a node per eligible pixel, an edge between 8-neighbors
// whose level-line angles agree within τ (directed -- see the level-line vector block's // own comment). That is the entire specification. Everything below is just
// how it gets computed.
//
// This replaces a round-synchronous, JFA-strided COMPETITIVE relabeling
// scheme (each pixel re-evaluating every round whether some neighbor's label
// was "more compatible AND stronger" than its own, where strength was the
// label's summed member magnitude, recomputed each round by a segmented
// reduction, and the neighbor offsets were long steered jumps whose stride
// grew with the region's own accumulated angular coherence). The single
// biggest thing that buys is CONFLUENCE.
//
// WHY CONFLUENCE MATTERS HERE. Because the edge predicate is SYMMETRIC, the
// answer is the transitive closure of a fixed relation, so it does not depend
// on round order, round COUNT, seeding, or which pixel happened to win a
// contested tie first. Concretely:
//   - There is a real termination condition. The old growSteps was "how many
//     rounds to run", with the honest admission in its own comment that there
//     was no closed form for how far growth would get in N rounds -- it was a
//     tuning knob wearing an iteration count's clothes. Here the loop runs to
//     a FIXPOINT and the round count is a pure debug scrubber.
//   - Two completely different implementations can be checked against each
//     other. A serial union-find over the same edge set must produce a
//     byte-identical labeling; any divergence is a bug rather than a tuning
//     difference. (Not implemented here -- see this file's tabled notes --
//     but the property is what makes it worth writing when it is.)
//   - No flapping. The old scheme needed a strictly-stronger tiebreak purely
//     to stop two comparably-sized regions trading a boundary pixel forever.
//   - Far less state. Gone: the per-round segmented reduction over
//     sumCos/sumSin/sumMag, the per-pixel strideDivider shrink-on-stall
//     counter and its wrap constant, largestStride, the steered candidate fan
//     and its lateral-drift bound. A round is now two flat passes over two
//     buffers.
//
// HOW IT CONVERGES FAST. Naive label propagation ("take the min label among
// my compatible neighbors") moves information exactly one pixel per round, so
// a 300px-long line would need ~300 rounds -- which is precisely the problem
// the strided JFA jumps existed to solve. The fix is NOT to reintroduce
// spatial strides. It is to do the doubling in LABEL SPACE instead:
//
//   hook:     next[i]  = min(label[i], min over edge-connected neighbors j of label[j])
//   compress: label[i] = next[next[i]]        <- pointer jumping
//
// Labels are pixel indices, so `next` is a forest of pointers and the
// compress pass halves every node's distance to its root each round. That is
// classic Shiloach-Vishkin hooking-and-shortcutting, and it recovers the
// logarithmic round count JFA had (O(log L) in practice) WITHOUT ever
// comparing two non-adjacent pixels. This is the crux of the whole redesign:
// long-range shortcutting can only ever compress connectivity that
// adjacent-pixel tests already proved, so it is structurally incapable of
// jumping onto a different parallel ridge -- the failure mode the old
// maxLateralSearchPx bound existed to (partially) contain.
//
// Both passes keep the frozen-buffer-in/fresh-buffer-out discipline the JFA
// version had, so each is still one trivially-parallel compute dispatch for a
// future GPU port -- two dispatches per round, and no reduction between them.
//
// WHY THE FIXPOINT IS THE RIGHT ANSWER. At the fixpoint, for every pixel i and
// every edge-connected neighbor j, label[i] <= label[j] and label[j] <=
// label[i], hence label[i] === label[j]; by induction across the component
// every member shares one label, and the compress pass's own fixpoint forces
// that label to be a self-pointing root. Since labels only ever propagate
// from members, that root is the component's MINIMUM pixel index. So the
// converged output is exactly "min member index per connected component" --
// the same thing union-find would produce, which is what makes the two
// checkable against each other.
//
// SEEDING is dense and uninteresting: every eligible pixel starts as its own
// singleton. There is no "which pixels get to seed" decision to make at all,
// because with a symmetric predicate the seed set cannot influence the result.
//
// CHAINING (accepted, deliberately unfixed). A pairwise-only predicate lets a
// gently curving arc chain end to end -- each adjacent pair agrees within τ
// while the two ends differ by far more. The classic guard is to compare each
// candidate against the REGION's running mean angle instead of its neighbor's
// pixel angle, but that is exactly what would destroy confluence and drag the
// per-round reduction back in. It is left unfixed on purpose: under lens
// distortion a physically straight line genuinely IS a gentle arc, so a
// grower that follows it is reporting the truth about the image, and stage
// 4+5's PCA fit and NFA test already measure and judge the resulting shape.
// Tabled remedies if it ever does bite: split-on-fit-failure (strictly better
// than stage 5's current shrink, since it keeps BOTH halves as real
// detections); or angle-bucketed CCL (run the whole thing independently
// inside each of B overlapping angle bins, which kills chaining by
// construction and stays embarrassingly parallel across bins).
// meanU is the region's mean level-line DIRECTION, normalized. It was an angle
// (meanAngle) until the vector-space pass; its only consumer, fitRectangle,
// used it purely to pick which of the PCA axis's two opposite directions to
// keep, so only the direction has ever mattered and the normalization is for
// comparability rather than correctness.
export interface GrownRegion { members: Int32Array; meanUx: number; meanUy: number }

// The full 8-neighborhood. All eight are tested, not a steered subset: the
// old scheme picked a handful of long offsets aimed along the region's
// aggregate direction because most of 8 directions would have been wasted
// perpendicular probes at a large stride. At stride 1 that reasoning
// evaporates -- a perpendicular neighbor is one pixel away, testing it is
// free, and it is exactly the test that lets a genuinely 2px-thick ridge grow
// across its own width instead of splitting into two parallel components.
const NEIGHBOR_DX = [1, 1, 0, -1, -1, -1, 0, 1];
const NEIGHBOR_DY = [0, 1, 1, 1, 0, -1, -1, -1];

// A hard ceiling on rounds, purely so a debug tool can never hang. Hook alone
// (no compression) propagates one pixel per round, so the longest possible
// chain sets a true upper bound; compression makes the real count
// logarithmic, so this is never reached in practice -- if it ever IS, that is
// a bug in the fixpoint detection, not a legitimately slow image.
function roundHardCap(w: number, h: number): number { return w + h + 64; }

// THE predicate -- the single rule the whole of stage 2+3 is built on.
// Exported so overlays/lsdOverlay.ts's edge-connectivity hover view tests the
// exact same condition the real algorithm does, rather than an independently
// reimplemented copy that could quietly drift out of sync (the same reason
// the JFA version exported computeGrowthCandidates).
//
// Directed, so a plain SIGNED dot: cos(θi - θj) >= cos(τ). No abs(), which is
// what keeps the two antiparallel edges of a thin stripe from fusing -- see
// the level-line vector block (lsdSegments.ts).
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
function eligibilityThresholdSq(rho: number): number {
  return rho >= 0 ? rho * rho : -Infinity;
}

// rhoLow/rhoHigh implement CANNY-STYLE HYSTERESIS, which is what stands in
// for the magnitude-priority ordering this design dropped. The old serial-BFS
// and JFA schemes both leaned on magnitude in an ordering-dependent way (grow
// the strongest seed first; let the higher summed-magnitude label win a
// contested pixel) specifically so a weak noisy pixel could not out-compete a
// real ridge. A symmetric predicate has no notion of "wins", so without a
// replacement a chain of barely-above-threshold noise could bridge two
// genuine ridges into one region.
//
// Hysteresis restores that guarantee without restoring any ordering: any pixel
// above rhoLow may participate in edges, but a finished component only
// SURVIVES if it contains at least one pixel above rhoHigh. That is a
// per-component OR-reduction over an already-finalized labeling -- computed
// after the fact, so it cannot influence what merged with what, and it stays
// confluent. It also strictly dominates a single threshold: a faint but real
// line anchored anywhere by one strong pixel survives intact, while a
// same-strength blob of pure noise with no strong pixel anywhere is dropped
// whole. Setting rhoHigh <= rhoLow degrades gracefully to plain single-
// threshold behavior (every eligible pixel trivially clears the high bar).
//
// maxRounds is a DEBUG SCRUBBER, not a tuning parameter: 0 means run to the
// fixpoint (the real algorithm, and the production default), while 1..N caps
// the round count so an overlay can watch components coalesce. Unlike the
// growSteps it replaces, changing it cannot change the converged answer --
// only how much of the way there you are looking at.
// Hysteresis survival + the collect/relabel pass, shared by growRegionsCCL and
// its GPU counterpart (pipelineGPU/growRegions.ts) so the two can never drift
// in how a finished labeling turns into regions.
//
// This is the CPU route AND the fallback; pipelineGPU/collectRegions.ts is the
// GPU one. An earlier version of this comment called the step "inherently
// serial", which was simply wrong: every stage is a standard parallel pattern
// -- labelSurvives is a scatter of a constant (no atomic needed, every writer
// writes 1), the grouping is a histogram + prefix sum + scatter CSR build,
// ascending label order falls out of the scan for free because labels ARE pixel
// indices, and meanAngle is a segmented reduction over the resulting slices.
// What actually blocked it was the missing prefix-sum primitive, now
// pipelineGPU/prefixSum.ts.
export function collectRegionsFromLabels(
  label: Int32Array, fx: Float64Array, fy: Float64Array, rhoHigh: number, n: number,
  minRegionSize: number,
): { regionId: Int32Array; regions: GrownRegion[] } {
  // Which components contain a pixel above rhoHigh. Indexed by label value (a
  // pixel index, so bounded by n), the same convention the round loop uses.
  const rhoHighSq = eligibilityThresholdSq(rhoHigh);
  const labelSurvives = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const lab = label[i];
    if (lab !== -1 && fx[i] * fx[i] + fy[i] * fy[i] > rhoHighSq) labelSurvives[lab] = 1;
  }

  const membersByLabel = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const lab = label[i];
    if (lab === -1 || !labelSurvives[lab]) continue;
    let list = membersByLabel.get(lab);
    if (!list) { list = []; membersByLabel.set(lab, list); }
    list.push(i);
  }
  const sortedLabels = Array.from(membersByLabel.keys()).sort((a, b) => a - b); // deterministic order

  const regionId = new Int32Array(n).fill(-1);
  const regions: GrownRegion[] = [];
  for (const lab of sortedLabels) {
    const members = membersByLabel.get(lab)!;
    // Prefilter, here rather than at the fitter, so an undersized component is
    // never materialized as a region on EITHER path -- no GrownRegion object,
    // no CSR row, no GPU thread. At the default floor of 2 this is purely work
    // removed: a 1-member component has no axis to fit, so the fitter could
    // only ever return null for it and the caller drop it. Its pixels are left
    // at regionId -1, which is what the raw-region debug overlay paints as "no
    // region" -- an isolated pixel stops showing as a region of one.
    if (members.length < minRegionSize) continue;
    const id = regions.length;
    for (const p of members) regionId[p] = id;
    // A plain raw sum of member level-line vectors -- no sign resolution
    // against a reference member. With directed growth every member is within
    // tau of every other member it is connected THROUGH, so there is no
    // polarity flip inside a region for the sum to cancel against. (Chaining
    // can still walk the direction a long way around a curve, which is a real
    // effect, but it walks CONTINUOUSLY and never jumps by pi.) The result
    // stays a genuine directed value, which is what fitRectangle's PCA-sign
    // resolution needs.
    let sc = 0, ss = 0;
    for (const p of members) {
      const m = Math.hypot(fx[p], fy[p]);
      // Every member cleared rhoLow to be labeled at all, so m > 0 here.
      sc += -fy[p] / m; ss += fx[p] / m;
    }
    // Normalized for comparability across paths; only the DIRECTION is load-
    // bearing. The degenerate all-cancel case can't arise from directed growth
    // but is pinned to a fixed axis rather than left as NaN.
    const sLen = Math.hypot(sc, ss);
    regions.push({
      members: Int32Array.from(members),
      meanUx: sLen > 0 ? sc / sLen : 1, meanUy: sLen > 0 ? ss / sLen : 0,
    });
  }
  return { regionId, regions };
}

export function growRegionsCCL(
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  toleranceDeg: number, rhoLow: number, rhoHigh: number, maxRounds: number, minRegionSize: number,
): { regionId: Int32Array; regions: GrownRegion[]; roundsRun: number; converged: boolean } {
  const n = w * h;
  const cosTol = Math.cos(THREE.MathUtils.degToRad(toleranceDeg));

  // The level-line vector never changes across rounds -- only which LABEL owns
  // a pixel does -- so it is normalized once here rather than recomputed by
  // every round's neighbor tests. This is where the old cos(theta)/sin(theta)
  // precompute lived, and where computeMagTheta's atan2 used to be undone one
  // stage after it was applied. Raw direction, NOT a doubled-angle pair: the
  // predicate is a signed dot, and doubling is precisely what would throw away
  // the sign it depends on.
  //
  // Eligibility is folded into the same pass, against a SQUARED threshold, so
  // an ineligible pixel costs no sqrt -- and most pixels are ineligible.
  // Ineligible entries keep (0,0), which no round ever reads: label -1 is
  // checked first everywhere.
  const rhoLowSq = eligibilityThresholdSq(rhoLow);
  const ux = new Float64Array(n), uy = new Float64Array(n);
  let label = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const gx = fx[i], gy = fy[i];
    const m2 = gx * gx + gy * gy;
    if (m2 > rhoLowSq) {
      const inv = 1 / Math.sqrt(m2);
      ux[i] = -gy * inv; uy[i] = gx * inv;
      label[i] = i; // dense: every eligible pixel is its own singleton
    } else {
      label[i] = -1;
    }
  }
  let next = new Int32Array(n);

  const cap = maxRounds > 0 ? Math.min(maxRounds, roundHardCap(w, h)) : roundHardCap(w, h);
  let roundsRun = 0, converged = false;

  for (let round = 0; round < cap; round++) {
    let changed = false;

    // ── Hook: read ONLY the frozen `label`, write ONLY `next` ─────────────
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const own = label[i];
        if (own === -1) { next[i] = -1; continue; } // below rhoLow -- never participates
        let best = own;
        const ci = ux[i], si = uy[i];
        for (let k = 0; k < 8; k++) {
          const nx = x + NEIGHBOR_DX[k], ny = y + NEIGHBOR_DY[k];
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const j = ny * w + nx;
          const nlab = label[j];
          if (nlab === -1 || nlab >= best) continue; // -1 ineligible; >= best can't lower our min
          if (!levelLinesCompatible(ci, si, ux[j], uy[j], cosTol)) continue;
          best = nlab;
        }
        next[i] = best;
      }
    }

    // ── Compress: pointer-jump one level, `next` in / `label` out ─────────
    // next[i] is always either -1 or the index of an ELIGIBLE pixel (labels
    // only ever originate from eligible pixels), so next[next[i]] is always a
    // defined, non-(-1) entry -- no second guard needed past the -1 check.
    for (let i = 0; i < n; i++) {
      const l = next[i];
      const jumped = l === -1 ? -1 : next[l];
      if (jumped !== label[i]) changed = true;
      label[i] = jumped;
    }

    roundsRun++;
    if (!changed) { converged = true; break; }
  }

  const { regionId, regions } = collectRegionsFromLabels(label, fx, fy, rhoHigh, n, minRegionSize);
  return { regionId, regions, roundsRun, converged };
}

// ── Stage 4: magnitude-weighted PCA rectangle fit ─────────────────────────

export interface RectangleCandidate { cx: number; cy: number; theta: number; length: number; width: number }

// The rectangle's angle comes from PCA on the region's own PIXEL
// COORDINATES (weighted by gradient magnitude) -- the standard image-
// moments technique for "what's this region's own axis of elongation,"
// which is exactly what a thin, long region wants for its rectangle's long
// axis. PCA only determines this axis up to a 180-degree ambiguity (an
// inertia matrix has no notion of "which way" along its own principal
// axis) -- resolved using the region's OWN directed growth-mean direction
// (meanUx/meanUy, already computed in stage 3): whichever of the PCA axis's two
// directions agrees with it is the one kept. This also matters for
// correctness, not just cosmetics -- stage 5's NFA alignment test compares
// pixel directions against the rectangle's theta directly, so a 180-degree-
// flipped theta would make every genuinely-aligned pixel look
// anti-aligned instead.
//
// NOTE this function deliberately KEEPS its atan2/cos/sin, even though the
// vector-space pass removed them everywhere else. The reason is that all of it
// is PER REGION (~1800 calls) rather than per pixel, so replacing the PCA angle
// with a direct half-angle eigenvector would buy nothing measurable while
// perturbing the one quantity that genuinely escapes this file as an angle.
// The per-pixel transcendentals were the whole cost, and those are gone.
function fitRectangle(
  members: Int32Array, fx: Float64Array, fy: Float64Array, w: number,
  meanUx: number, meanUy: number,
): RectangleCandidate {
  let sumW = 0, sumX = 0, sumY = 0;
  for (let mi = 0; mi < members.length; mi++) {
    const i = members[mi];
    const m = Math.hypot(fx[i], fy[i]);
    sumW += m; sumX += m * (i % w); sumY += m * ((i / w) | 0);
  }
  const wcx = sumX / sumW, wcy = sumY / sumW;

  let Ixx = 0, Iyy = 0, Ixy = 0;
  for (let mi = 0; mi < members.length; mi++) {
    const i = members[mi];
    const m = Math.hypot(fx[i], fy[i]);
    const x = (i % w) - wcx, y = ((i / w) | 0) - wcy;
    Ixx += m * x * x; Iyy += m * y * y; Ixy += m * x * y;
  }
  let theta = 0.5 * Math.atan2(2 * Ixy, Ixx - Iyy);
  if (Math.cos(theta) * meanUx + Math.sin(theta) * meanUy < 0) theta += Math.PI;

  const ax = Math.cos(theta), ay = Math.sin(theta);
  const px = -ay, py = ax;
  let minProj = Infinity, maxProj = -Infinity, minPerp = Infinity, maxPerp = -Infinity;
  for (let mi = 0; mi < members.length; mi++) {
    const i = members[mi];
    const x = (i % w) - wcx, y = ((i / w) | 0) - wcy;
    const proj = x * ax + y * ay, perp = x * px + y * py;
    if (proj < minProj) minProj = proj; if (proj > maxProj) maxProj = proj;
    if (perp < minPerp) minPerp = perp; if (perp > maxPerp) maxPerp = perp;
  }
  // Rectangle center = midpoint of the projected extent, NOT the weighted
  // centroid directly -- the centroid isn't generally at the geometric
  // middle of the region's own extent along either axis, so placing the
  // rectangle there could clip real member pixels on one side while
  // overshooting empty space on the other.
  const midProj = (minProj + maxProj) / 2, midPerp = (minPerp + maxPerp) / 2;
  return {
    cx: wcx + midProj * ax + midPerp * px,
    cy: wcy + midProj * ay + midPerp * py,
    theta, length: maxProj - minProj, width: maxPerp - minPerp,
  };
}

// ── Stage 5: NFA validation ────────────────────────────────────────────────

// log(sum_{j=k}^{n} C(n,j) p^j (1-p)^(n-j)), natural log -- the standard
// stable approach: each term as log-choose + j*log(p) + (n-j)*log(1-p),
// with log-choose built INCREMENTALLY (C(n,j) = C(n,j-1) * (n-j+1)/j)
// rather than restarting an O(j) computation per term, then combined via
// log-sum-exp (subtract the running max before exponentiating) since terms
// can span many orders of magnitude and would overflow/underflow computed
// directly.
function logBinomialTail(n: number, k: number, p: number): number {
  if (k <= 0) return 0; // log(1) -- any n pixels trivially have >=0 aligned
  if (k > n) return -Infinity;
  const logP = Math.log(p), log1mP = Math.log(1 - p);
  let logChoose = 0;
  for (let i = 1; i <= k; i++) logChoose += Math.log((n - k + i) / i);
  const logTerms: number[] = [];
  let logTerm = logChoose + k * logP + (n - k) * log1mP;
  logTerms.push(logTerm);
  let maxLog = logTerm;
  for (let j = k + 1; j <= n; j++) {
    logChoose += Math.log((n - j + 1) / j);
    logTerm = logChoose + j * logP + (n - j) * log1mP;
    logTerms.push(logTerm);
    if (logTerm > maxLog) maxLog = logTerm;
  }
  let sumExp = 0;
  for (const t of logTerms) sumExp += Math.exp(t - maxLog);
  return maxLog + Math.log(sumExp);
}

// Squashes nfaLog10 (unbounded, more negative = more confident) into the
// shared [0, 1] "how line-y is this" scale bucketFillJoin.ts's join walk
// reads uniformly off every segment, regardless of which producer built it
// (see BucketFillSegment's own lineScore comment). A logistic curve centered
// exactly on the accept/reject threshold: a rectangle that JUST clears NFA
// scores 0.5, one an order of magnitude past it scores ~0.91, and it
// saturates smoothly toward 1 from there -- no separate span constant to
// tune, since the threshold itself (already a real per-settings quantity) is
// the only anchor this needs.
function nfaLog10ToLineScore(nfaLog10: number, logEpsilon: number): number {
  const thresholdLog10 = logEpsilon / Math.LN10;
  return 1 / (1 + Math.pow(10, nfaLog10 - thresholdLog10));
}

// n = pixels whose center falls inside the rectangle's actual rotated
// footprint (scanned via its axis-aligned bounding box, not just the
// region's original flood-fill members -- the fitted rectangle's shape can
// differ slightly from the grown blob). k = of those, how many have a
// level-line angle within toleranceRad of the rectangle's own theta,
// DIRECTED: a plain signed cos-dot >= cos(toleranceRad), matching
// levelLinesCompatible's growth test exactly.
//
// This briefly used abs(cos-dot) instead, to match a mod-π growth experiment
// -- and that combination was quietly WRONG in a way worth recording, because
// it is easy to reintroduce. The NFA null model below uses p = τ/π, which is
// the probability that a uniformly-random DIRECTED angle lands within ±τ of a
// fixed direction (measure 2τ out of 2π). An abs() acceptance test admits TWO
// such windows (±τ of theta and of theta+π), i.e. measure 4τ out of 2π = 2τ/π
// -- twice as permissive as the model it was scored against. Every region
// therefore looked more statistically significant than it was, and the
// effective accept threshold was looser than nfaEpsilon claimed. Going
// directed here makes p = τ/π correct as-written rather than needing to be
// doubled. If mod-π counting is ever reintroduced, p MUST double with it.
//
// Sub-rho pixels count toward n (they were geometrically tested) but never
// toward k (too weak to trust their angle at all).
export function countRectanglePixels(
  rect: RectangleCandidate, fx: Float64Array, fy: Float64Array, w: number, h: number, rho: number, toleranceRad: number,
): { n: number; k: number } {
  const { cx, cy, theta: rectTheta, length } = rect;
  const ax = Math.cos(rectTheta), ay = Math.sin(rectTheta);
  const px = -ay, py = ax;
  const cosTol = Math.cos(toleranceRad);
  const rhoSq = eligibilityThresholdSq(rho);
  // Inclusion is tested with a small tolerance because the rectangle's own
  // extent is DEFINED by its extreme members: fitRectangle sets length =
  // maxProj - minProj, so those members land at |proj| exactly == hl, and on a
  // thin region every member lands at |perp| exactly == hw (which pins to the
  // 0.5 floor). Exact-boundary pixels are therefore guaranteed to exist in
  // every region, not a rare edge case -- and `Math.abs(proj) > hl` decides
  // them on whichever side the last ulp of rounding falls. That made the count
  // non-deterministic across arithmetic: the GPU port (f32) and this path
  // (f64) disagreed on n for 180 of 2931 regions and on k for 277, by up to 4
  // pixels each, which on a 12-pixel region moved nfaLog10 by 4.7 decades and
  // flipped 6 accept/reject decisions. The epsilon is far above f32 geometry
  // error (~1e-4 at image coordinates) and far below a pixel, so it includes
  // the boundary deterministically without admitting anything that isn't
  // genuinely inside.
  const BOUNDARY_EPS = 1e-3;
  const hl = length / 2;
  // Floor the tested half-width so a near-zero-width fit (a nearly
  // 1-pixel-wide ridge, common on a clean synthetic edge) still tests a
  // real strip of pixels instead of degenerating to nothing.
  const hw = Math.max(rect.width / 2, 0.5);

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [a, b] of [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]]) {
    const x = cx + a * ax + b * px, y = cy + a * ay + b * py;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const x0 = Math.max(0, Math.floor(minX)), x1 = Math.min(w - 1, Math.ceil(maxX));
  const y0 = Math.max(0, Math.floor(minY)), y1 = Math.min(h - 1, Math.ceil(maxY));

  let n = 0, k = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx, dy = y - cy;
      const proj = dx * ax + dy * ay, perp = dx * px + dy * py;
      if (Math.abs(proj) > hl + BOUNDARY_EPS || Math.abs(perp) > hw + BOUNDARY_EPS) continue;
      n++;
      const i = y * w + x;
      // The hot loop of the whole fitter -- every pixel of every candidate
      // rectangle's footprint passes through here. The squared test rejects a
      // sub-rho pixel with no sqrt at all, and a pixel that survives pays
      // exactly one: alignDot is the level-line unit vector dotted with the
      // rectangle axis, and the unit vector is (-fy, fx)/‖g‖, so the whole
      // thing is one divide by the magnitude. This used to be a cos and a sin
      // per pixel, on an angle that had itself cost an atan2 to build.
      const gx = fx[i], gy = fy[i];
      const m2 = gx * gx + gy * gy;
      if (m2 <= rhoSq) continue;
      const alignDot = (-gy * ax + gx * ay) / Math.sqrt(m2); // = cos(levelLine(i) - rectTheta)
      if (alignDot >= cosTol) k++;
    }
  }
  return { n, k };
}

export interface LsdRectangle {
  cx: number; cy: number; theta: number; length: number; width: number;
  accepted: boolean;
  retries: number; // how many tighten/shrink attempts were taken before the final accept/reject
  nfaLog10: number; // log10(NFA) -- more negative = more statistically confident
  lineScore: number; // nfaLog10 squashed to [0, 1] via nfaLog10ToLineScore -- see that function's own comment
  // Stage 3's ORIGINAL grown-region membership (pixel indices into the
  // field), before any retry loop tightened/shrank it -- region.members
  // itself is never mutated by the retry loop below (only the local
  // `members` variable gets reassigned), so this is always the true,
  // complete flood-fill result this rectangle came from. For debug display
  // only (overlays/hoverDebugOverlays.ts's raw-region-pixels toggle) -- not
  // read anywhere in the accept/reject decision itself.
  rawMembers: Int32Array;
}

export interface LsdSettings {
  toleranceDeg: number;
  // rhoNoiseThreshold is hysteresis' LOW bar (participate in edges);
  // rhoHighThreshold is its HIGH bar (a component must contain at least one
  // pixel above it to survive at all). Kept under the original name so every
  // existing caller/persisted control id stays valid -- see growRegionsCCL's
  // own comment for what the pair does.
  rhoNoiseThreshold: number; rhoHighThreshold: number;
  cclSteps: number; // debug round scrubber only -- 0 = run to fixpoint, see growRegionsCCL
  minRegionSize: number; // components smaller than this never become regions -- see camera/settings.ts's lsdMinRegionSize
  nfaEpsilon: number; nfaTestExponent: number;
  // Read only by the retired fitRegionWithRetries -- no live path touches them.
  maxRetries: number; retryToleranceFactor: number; retryShrinkFraction: number;
}

// RETIRED-NOTE: computeMagTheta used to live here, turning the gradient field
// into per-pixel (magnitude, angle) arrays for every stage below to consume.
// It is gone rather than moved. Its output was a lossy re-encoding of the
// gradient the previous stage already produces -- every consumer immediately
// converted the angle back into the vector it came from -- so stages 2 through
// 5 now read fx/fy directly and the pass does not exist. See the level-line
// vector block near the top of this file for the identity that makes that
// work, and the notes on fitRectangle for the one angle that survives.

// Stage 4 + stage 5 for ONE region, attempt 0 only -- the LIVE fitter, used by
// the CPU path for every region and by pipelineGPU/lsdFitVerify.ts as the
// per-region reference the GPU kernel is compared against. Exactly the scope
// lsdFit.wgsl.ts implements, which is the point: with no retry concept on
// either side there is nothing left for the two paths to disagree about
// structurally.
//
// Returns null only for a region below the 2-member floor. In practice that
// never happens now -- collectRegionsFromLabels' minRegionSize prefilter
// already dropped those -- but the guard stays because fitRectangle genuinely
// has no axis to fit with fewer than 2 points, and the floor is a slider.
export function fitRegionOnce(
  region: GrownRegion, fx: Float64Array, fy: Float64Array, w: number, h: number,
  settings: LsdSettings, logNTests: number, logEpsilon: number,
): LsdRectangle | null {
  const members = region.members;
  if (members.length < 2) return null;
  const rect = fitRectangle(members, fx, fy, w, region.meanUx, region.meanUy);
  const toleranceRad = THREE.MathUtils.degToRad(settings.toleranceDeg);
  const { n: rn, k: rk } = countRectanglePixels(rect, fx, fy, w, h, settings.rhoNoiseThreshold, toleranceRad);
  const logNfa = logNTests + logBinomialTail(rn, rk, toleranceRad / Math.PI);
  const nfaLog10 = logNfa / Math.LN10;
  return {
    cx: rect.cx, cy: rect.cy, theta: rect.theta, length: rect.length, width: rect.width,
    accepted: logNfa < logEpsilon, retries: 0, nfaLog10,
    lineScore: nfaLog10ToLineScore(nfaLog10, logEpsilon), rawMembers: region.members,
  };
}

// ── RETIRED-NOTE: NOT CALLED FROM ANYWHERE ───────────────────────────────
// Stage 5's original accept/RETRY loop (tighten tau, then shrink the region by
// dropping its farthest-from-center members). fitRegionOnce above replaced it
// as the live fitter; this is kept as reference, not deleted, alongside
// growRegionsJFA at the bottom of this file.
//
// Retired deliberately rather than because it was wrong. It was the last thing
// forcing mag/theta to stay available on CPU during the GPU path -- every
// GPU-rejected region fell through to here for a CPU re-attempt -- which is
// exactly the dependency that has to go for stages 1-4 to become one
// GPU-resident run. Retry 1 (angle refilter) would have ported easily; retry 2+
// needs a per-region partial sort (drop the farthest fraction), which is a
// genuinely harder GPU problem than anything else in this file. With the live
// path running attempt-0-only, the GPU's rejected candidates are simply used as
// returned, and nothing re-derives them on CPU.
//
// Its three settings (lsdMaxRetries/lsdRetryToleranceFactor/
// lsdRetryShrinkFraction) still exist and their sliders are disabled, not
// removed -- see camera/settings.ts.
export function fitRegionWithRetries(
  region: GrownRegion, fx: Float64Array, fy: Float64Array, w: number, h: number,
  settings: LsdSettings, logNTests: number, logEpsilon: number,
): LsdRectangle | null {
  let members = region.members;
  let toleranceDeg = settings.toleranceDeg;
  let retries = 0;
  let accepted = false;
  let rect: RectangleCandidate | null = null;
  let nfaLog10 = Infinity;

  for (;;) {
    if (members.length < 2) break; // degenerate -- no meaningful axis, leave as rejected
    rect = fitRectangle(members, fx, fy, w, region.meanUx, region.meanUy);
    const toleranceRad = THREE.MathUtils.degToRad(toleranceDeg);
    const p = toleranceRad / Math.PI;
    const { n: rn, k: rk } = countRectanglePixels(rect, fx, fy, w, h, settings.rhoNoiseThreshold, toleranceRad);
    const logNfa = logNTests + logBinomialTail(rn, rk, p);
    nfaLog10 = logNfa / Math.LN10;
    if (logNfa < logEpsilon) { accepted = true; break; }
    if (retries >= settings.maxRetries) break;
    retries++;
    if (retries === 1) {
      // Retighten first (LSD's own first move): a simplified stand-in for
      // re-growing the region from its seed under a stricter tolerance --
      // re-filters the CURRENT members down to those still within the
      // new, tighter tolerance of the region's own (fixed) mean angle,
      // rather than a full re-grow from scratch. Same intent (shrink to
      // the self-consistent "core" of the region) with far less
      // machinery than repeating stage 3 per retry. Signed (not abs) cos-dot,
      // matching levelLinesCompatible's growth test and countRectanglePixels'
      // alignment count -- see the level-line vector block on why every
      // orientation comparison in this file is directed.
      toleranceDeg *= settings.retryToleranceFactor;
      const cosTol = Math.cos(THREE.MathUtils.degToRad(toleranceDeg));
      const kept: number[] = [];
      for (let mi = 0; mi < members.length; mi++) {
        const i = members[mi];
        const m = Math.hypot(fx[i], fy[i]);
        if ((-fy[i] * region.meanUx + fx[i] * region.meanUy) / m >= cosTol) kept.push(i);
      }
      members = Int32Array.from(kept);
    } else {
      // Subsequent retries: drop the farthest-from-center fraction of
      // members -- LSD's own "reduce region radius," aimed at cutting
      // away a spuriously-attached side branch (e.g. two near-parallel
      // lines the growth pass glued together) rather than tightening
      // angle further.
      const { cx, cy } = rect;
      const withDist: { i: number; d: number }[] = [];
      for (let mi = 0; mi < members.length; mi++) {
        const i = members[mi];
        const x = (i % w) - cx, y = ((i / w) | 0) - cy;
        withDist.push({ i, d: x * x + y * y });
      }
      withDist.sort((a, b) => a.d - b.d);
      const keep = Math.max(2, Math.round(members.length * (1 - settings.retryShrinkFraction)));
      members = Int32Array.from(withDist.slice(0, keep).map((e) => e.i));
    }
  }

  if (!rect) return null;
  const lineScore = nfaLog10ToLineScore(nfaLog10, logEpsilon);
  return { cx: rect.cx, cy: rect.cy, theta: rect.theta, length: rect.length, width: rect.width, accepted, retries, nfaLog10, lineScore, rawMembers: region.members };
}

// The two NFA log terms, derived once per call from the settings + image size
// so the stage-4 helpers below can't drift in how they compute them.
function nfaLogTerms(w: number, h: number, settings: LsdSettings): { logNTests: number; logEpsilon: number } {
  return {
    logNTests: settings.nfaTestExponent * Math.log(Math.max(w, h)),
    logEpsilon: Math.log(settings.nfaEpsilon),
  };
}

// Stage 4+5 on CPU, for an ALREADY-GROWN region set. Split out from
// computeLsdRectangles so the grower and the fitter can be dispatched to
// CPU/GPU independently of each other (see computeLsdRectanglesAuto).
function fitRegionsCPU(
  regions: readonly GrownRegion[], fx: Float64Array, fy: Float64Array,
  w: number, h: number, settings: LsdSettings,
): LsdRectangle[] {
  const { logNTests, logEpsilon } = nfaLogTerms(w, h, settings);
  const results: LsdRectangle[] = [];
  for (const region of regions) {
    const r = fitRegionOnce(region, fx, fy, w, h, settings, logNTests, logEpsilon);
    if (r) results.push(r);
  }
  return results;
}

// Stage 4 + stage 5 on GPU (pipelineGPU/lsdFit.ts). Null only if WebGPU itself
// is unavailable.
//
// REJECTED candidates are taken straight from the GPU's own output rather than
// re-fitted on CPU. That fallback existed only to run the retry loop on regions
// the first pass rejected; with the fitter attempt-0-only and the two paths
// verified to agree on n/k/accept, re-running a rejection on CPU would spend
// real time reproducing the identical numbers. The shader already returns full
// geometry for rejected regions, which is all the "show rejected candidates"
// overlay wants. Dropping it is also what frees stages 1-4 from needing
// the gradient field on CPU at all.
//
// The one thing this still needs on CPU is each region's member list, for
// rawMembers -- and that is NOT a residency shortfall to be optimized away
// later. lsdRectanglesToBucketFillShape rebuilds a per-pixel regionId from
// rawMembers to seed the join walk, so the members have to land whatever
// happens. What the residency removes is the DOUBLE handling: they used to come
// down from collect and then get re-packed and re-uploaded here as a CSR. Now
// they come down once and the fitter reads the device-side copy.
async function fitRegionsGPU(
  res: FieldResidency, w: number, h: number, settings: LsdSettings,
): Promise<LsdRectangle[] | null> {
  const { logNTests, logEpsilon } = nfaLogTerms(w, h, settings);
  // Concurrent, not sequential: fitAndTestRegionsGPU binds the region CSR
  // synchronously before its first await, so starting it first and letting the
  // members readback run alongside puts that transfer under the kernel instead
  // of after it. Promise.all rather than two bare awaits so a failure in either
  // one can't leave the other rejecting unobserved.
  const [gpuResults, regions] = await Promise.all([
    fitAndTestRegionsGPU(res, w, h, settings.rhoNoiseThreshold, settings.toleranceDeg, logNTests, logEpsilon),
    res.regionsCPU(),
  ]);
  if (!gpuResults) return null;

  const results: LsdRectangle[] = [];
  for (let i = 0; i < regions.length; i++) {
    const g = gpuResults[i];
    results.push({
      cx: g.cx, cy: g.cy, theta: g.theta, length: g.length, width: g.width,
      accepted: g.accepted, retries: 0, nfaLog10: g.nfaLog10,
      lineScore: nfaLog10ToLineScore(g.nfaLog10, logEpsilon), rawMembers: regions[i].members,
    });
  }
  return results;
}

// Fully-CPU path, and the source of truth every GPU stage is verified against.
export function computeLsdRectangles(field: GradientField, settings: LsdSettings): LsdRectangle[] {
  const { w, h, fx, fy } = field;
  const { regions } = growRegionsCCL(
    fx, fy, w, h, settings.toleranceDeg, settings.rhoNoiseThreshold, settings.rhoHighThreshold, settings.cclSteps, settings.minRegionSize,
  );
  return fitRegionsCPU(regions, fx, fy, w, h, settings);
}

// CPU growing + GPU fitting, i.e. the useGPULsdFit half of the dispatch in
// isolation. Kept as a named entry point because it is the exact pairing
// pipelineGPU/lsdFitVerify.ts's numbers were measured against.
export async function computeLsdRectanglesGPU(field: GradientField, settings: LsdSettings): Promise<LsdRectangle[] | null> {
  const { w, h, fx, fy } = field;
  const res = await FieldResidency.create(w * h, true);
  try {
    res.provideCPU('fx', fx);
    res.provideCPU('fy', fy);
    const { regionId, regions } = growRegionsCCL(
      fx, fy, w, h, settings.toleranceDeg, settings.rhoNoiseThreshold, settings.rhoHighThreshold, settings.cclSteps, settings.minRegionSize,
    );
    res.provideCPU('regionId', regionId);
    res.provideRegionsCPU(regions);
    return await fitRegionsGPU(res, w, h, settings);
  } finally {
    res.destroy();
  }
}

// Single dispatch point every caller uses -- centralizes the globalState GPU
// checks once instead of duplicating them at each call site.
//
// The two stages dispatch INDEPENDENTLY: useGPUGrowRegions picks stage 2+3,
// useGPULsdFit picks stage 4+5's first pass, and either can fall back to CPU on
// its own without disturbing the other. That independence is the point -- the
// grower is the one stage whose GPU output is NOT bit-identical to its CPU
// output (see pipelineGPU/growRegions.ts's header), so it has to stay
// separately switchable from a stage that is.
//
// Takes the residency (pipelineGPU/fieldResidency.ts) rather than a
// GradientField, and does NOT own it. That is deliberate: this function used to
// create one per call, which capped the chain at stage 3 and forced stage 1's
// output down to CPU and straight back up again no matter what either toggle
// said. The caller owning it is what lets the gradient join the chain -- see
// pipeline/poseCompute.ts, and computeLsdRectanglesFromField below for callers
// that genuinely do start from a CPU field.
//
// Stages transfer nothing themselves; they publish on the side they produced on
// and ask for the side they want, and the residency moves data only when a
// producer and a consumer are actually on opposite sides of the bus. That is
// what makes these toggles cost what the configuration implies rather than what
// each module hardcoded: with the whole chain on GPU, fx/fy are never uploaded
// at all and the labeling never crosses either.
export async function computeLsdRectanglesAuto(
  res: FieldResidency, w: number, h: number, settings: LsdSettings,
): Promise<LsdRectangle[]> {
  const growArgs = [
    w, h, settings.toleranceDeg, settings.rhoNoiseThreshold, settings.rhoHighThreshold, settings.cclSteps, settings.minRegionSize,
  ] as const;
  const grown = globalState.useGPUGrowRegions ? await growRegionsCCLGPU(res, ...growArgs) : null;
  if (!grown) {
    // Either the toggle is off or the GPU grower bailed; on both paths it
    // published nothing, so the CPU grower owns these slots. Asking the
    // residency for the CPU side of fx/fy is what pulls stage 1's output back
    // down if -- and only if -- it was produced on the device.
    const cpu = growRegionsCCL(await res.cpuF64('fx'), await res.cpuF64('fy'), ...growArgs);
    res.provideCPU('regionId', cpu.regionId);
    res.provideRegionsCPU(cpu.regions);
  }

  if (globalState.useGPULsdFit) {
    const gpu = await fitRegionsGPU(res, w, h, settings);
    if (gpu) return gpu;
  }
  return fitRegionsCPU(await res.regionsCPU(), await res.cpuF64('fx'), await res.cpuF64('fy'), w, h, settings);
}

// Whether any stage of the chain needs a device. No device is requested at all
// when every stage is on CPU, so an all-CPU frame still never touches
// navigator.gpu -- and useGPUGradient counts, because stage 1 can want a device
// when nothing downstream does.
export function lsdChainWantsGPU(): boolean {
  return globalState.useGPUGradient || globalState.useGPUGrowRegions || globalState.useGPULsdFit;
}

// ── Stage 1: the 2x2 forward-difference gradient ──
//
// Publishes into the residency instead of returning a field, which is what
// makes stage 1 part of the chain rather than a thing that happens before it.
// Neither branch transfers anything: the GPU one leaves fx/fy on the device and
// the CPU one leaves them in JS, and whether either ever crosses is decided
// later, by whoever asks for the other side. Before this, the GPU branch
// uploaded gray and read fx/fy straight back, and then growRegionsCCLGPU
// uploaded those same two arrays again a moment later.
async function runGradient2x2Stage(res: FieldResidency, w: number, h: number): Promise<void> {
  const useGPU = globalState.useGPUGradient && res.device !== null;
  const s = spanStart(useGPU ? 'gradient2x2 (GPU)' : 'gradient2x2 (CPU)');
  if (!(useGPU && await computeGradient2x2FieldGPU(res, w, h))) {
    // gray is CPU-resident by construction (createLsdChainResidency put it
    // there), so this is a lookup and never a readback.
    const field = computeGradient2x2Field(await res.cpuF64('gray'), w, h);
    res.provideCPU('fx', field.fx);
    res.provideCPU('fy', field.fy);
  }
  spanEnd(s);
}

// The chain's two entry points, always used as a pair. Split rather than fused
// because the caller has to be able to reach into the residency AFTER the
// rectangles come out -- pipeline/poseCompute.ts's useWorldVoteOrientation
// branch still wants fx/fy on the CPU -- and because the caller owns the
// destroy. Anything that runs the chain should go through these two and nothing
// else, so that a residency-plumbing mistake is visible to the dev harness
// (pipelineGPU/lsdChainVerify.ts) rather than only to production.
export async function createLsdChainResidency(gray: Float64Array, w: number, h: number): Promise<FieldResidency> {
  const res = await FieldResidency.create(w * h, lsdChainWantsGPU());
  res.provideCPU('gray', gray);
  return res;
}

// Stages 1 through 4: gray in, rectangles out, every intermediate left wherever
// its producer put it.
export async function runLsdChain(
  res: FieldResidency, w: number, h: number, settings: LsdSettings,
): Promise<LsdRectangle[]> {
  await runGradient2x2Stage(res, w, h);
  return await computeLsdRectanglesAuto(res, w, h, settings);
}

// computeLsdRectanglesAuto for callers that already hold a CPU gradient field
// and have no interest in where anything lives: it wraps the field in a
// residency of its own for the duration of the call. The production pose path
// does NOT come through here -- it owns a residency spanning stage 1 as well,
// which is the whole point (see pipeline/poseCompute.ts) -- but the debug
// overlay and the phone both compute their gradient on CPU and would gain
// nothing from threading one through.
export async function computeLsdRectanglesFromField(field: GradientField, settings: LsdSettings): Promise<LsdRectangle[]> {
  const { w, h, fx, fy } = field;
  // Not lsdChainWantsGPU(): stage 1 has already happened on CPU here, so
  // useGPUGradient says nothing about whether this call needs a device.
  const res = await FieldResidency.create(w * h, globalState.useGPUGrowRegions || globalState.useGPULsdFit);
  try {
    res.provideCPU('fx', fx);
    res.provideCPU('fy', fy);
    return await computeLsdRectanglesAuto(res, w, h, settings);
  } finally {
    res.destroy();
  }
}

// ── Adapter: LSD rectangles -> bucketFillJoin.ts's expected input shape ──
//
// pipeline/bucketFillJoin.ts's computeJoinWalk was built against
// bucketFillSegments.ts's own BucketFillSegment[] + regionId output, but
// only reads FIVE things off a segment: endAlongX/Y, endAgainstX/Y (via
// segmentLength and spawnPair) and lineScore (seeds each segment's merge
// confidence, see computeJoinWalk's own header) -- count/cx/cy/avgFx/avgFy
// are present in the type but unused by the join walk or
// computeCompositeLines. That makes this a thin, honest adapter rather than
// a real behavioral bridge:
// only ACCEPTED rectangles become segments (a rejected candidate isn't a
// real detection, shouldn't be treated as one to merge); a rectangle's own
// two tangent-axis ends (its long axis, already what LSD fits the line
// ALONG) become endAlong/endAgainst directly, no re-derivation needed the
// way bucketFillSegments.ts's own post-pass needed one; and regionId is
// built from EACH rectangle's rawMembers (stage 3's actual flood-filled
// pixels, not the rectangle's geometric footprint) -- this is what the join
// walk seeds its buffer with, so a front reacts the moment it enters
// another segment's real grown blob, matching exactly what the join walk's
// own header comment describes bucketFillSegments.ts's regionId as
// providing.
//
// avgFx/avgFy: approximated as the PERPENDICULAR of the rectangle's own
// tangent axis (LSD's theta is a line's long axis; bucketFillSegments.ts's
// avgFx/avgFy was specifically the GRADIENT direction, perpendicular to
// that), scaled by member count so its magnitude still roughly tracks
// region "mass" the way the original did -- populated for type completeness
// and in case a future consumer reads it, not because anything currently
// downstream (computeJoinWalk, computeCompositeLines) actually does.
export function lsdRectanglesToBucketFillShape(
  rects: readonly LsdRectangle[], w: number, h: number,
): { regionId: Int32Array; segments: BucketFillSegment[] } {
  const regionId = new Int32Array(w * h).fill(-1);
  const segments: BucketFillSegment[] = [];
  for (const r of rects) {
    if (!r.accepted) continue;
    const id = segments.length;
    for (let mi = 0; mi < r.rawMembers.length; mi++) regionId[r.rawMembers[mi]] = id;

    const ax = Math.cos(r.theta), ay = Math.sin(r.theta);
    const hl = r.length / 2;
    const count = r.rawMembers.length;
    segments.push({
      count, cx: r.cx, cy: r.cy,
      avgFx: -ay * count, avgFy: ax * count,
      endAlongX: r.cx + hl * ax, endAlongY: r.cy + hl * ay,
      endAgainstX: r.cx - hl * ax, endAgainstY: r.cy - hl * ay,
      lineScore: r.lineScore,
    });
  }
  return { regionId, segments };
}

// ═══════════════════════════════════════════════════════════════════════════
// ── RETIRED: JFA-strided competitive region growing ────────────────────────
// ═══════════════════════════════════════════════════════════════════════════
//
// NOT CALLED FROM ANYWHERE. growRegionsCCL above replaced this as stage 2+3;
// nothing in the live pipeline, the overlays, or the settings panel reaches
// it, and it has no UI (its three tuning parameters -- grow steps, max
// lateral search, exact-tangent-only -- were removed from camera/settings.ts
// and sphere-lab.html rather than left as dead controls, so the only way to
// run this now is to call it directly with literal arguments).
//
// Kept in the repo deliberately, as the same kind of reference/comparison
// implementation pipeline/bucketFillSegments.ts's computeBucketFillRegions
// already is: it is a genuinely different answer to "how do you grow a line
// segment in parallel" -- long steered strides that let ONE stage both form
// AND bridge segments, versus growRegionsCCL's strictly-adjacent connected
// components that form only and leave bridging to bucketFillJoin.ts. The
// design notes below are the real value here; they record why the strided
// approach was built and, by implication, what would have to be true to want
// it back. See growRegionsCCL's own header for the counter-argument that won.
//
// If this is ever revived, note that it was written against the MOD-π
// convention this file has since reverted (see the level-line vector block's  own comment):
// its compatibility test folds to doubled angles and would need revisiting
// against directed angles, and it would no longer separate the two edges of a
// thin stripe for free.

// The original design notes, preserved verbatim from when this WAS the live
// stage 2+3, except where a claim about the surrounding code has since gone
// stale (those are corrected inline and marked RETIRED-NOTE):
//
// A round-synchronous, JFA-strided competitive relabeling scheme --
// replaces the old magnitude-sorted serial BFS (grow one region to
// completion, strictly in magnitude-priority order, before even looking at
// the next seed). Every round reads ONE frozen "current labels" buffer and
// writes a brand-new one -- no pixel's update depends on another pixel's
// update from the SAME round, so a round is trivially parallel (one round =
// one compute dispatch, whenever this gets a GPU port), unlike the BFS
// version's shared priority queue. See this session's chat for the full
// derivation.
//
// Seeding is DENSE: every pixel above rho starts as its own singleton
// region (label = its own pixel index). This isn't a design compromise --
// it's what makes the JFA framing apply cleanly at all: with dense seeding
// there's no separate "which pixels get to seed" decision (the old sparse
// local-maxima seeding this used to lean on needed computeGradientLocalMaxima,
// deleted along with the rest of label-prop in f27dfdf, and would need
// rebuilding from scratch) -- growing is just a pure per-pixel competition
// for "whose territory is this," classic JFA's own Voronoi framing, just
// with a direction+strength compatibility test standing in for Euclidean
// distance.
//
// Each round, EVERY pixel (not just "unlabeled" ones -- under dense seeding
// every pixel already has SOME label from round 0, so there's no meaningful
// unlabeled state, and no monotonic "once absorbed, stays" rule the way
// sparse seeding needed) re-evaluates whether a NEIGHBOR's label is more
// compatible AND stronger than its own current label. "Stronger" is a
// label's summed member magnitude (not member count) -- a soft
// approximation of LSD's own strict magnitude-priority seed order (a real
// ridge should reliably out-compete a weak noisy singleton for a contested
// pixel), recomputed fresh each round via a segmented reduction over the
// frozen label buffer, the one other GPU-friendly primitive this needs
// besides the propagate step itself. A pixel only ever switches to a
// STRICTLY stronger neighbor, never an equal one -- avoids pointless
// flapping between comparably-sized regions near a boundary.
//
// Neighbor offsets follow a stride that GROWS WITH THE REGION'S OWN
// COHERENCE, not a fixed round-indexed schedule: stride = r, the group's own
// doubled-angle resultant length hypot(sumCos[ownLabel], sumSin[ownLabel])
// (clamped to [1, largestStride]) -- the SAME quantity growRegionsJFA's own
// body already computes per pixel per round to derive the search direction
// (see the "complex square root" comment below), just reused here as a
// magnitude too, not just a direction. sumCos/sumSin are sums of RAW UNIT
// doubled-angle vectors (not magnitude-weighted -- that's sumMag, a separate
// accumulator), so r is bounded between 0 and the region's own member count,
// and only approaches that count when members' angles actually agree -- the
// standard circular-statistics "resultant length," a coherence-weighted
// size, not a raw count or raw mass.
//
// This replaces an earlier ascending-Fibonacci-with-per-pixel-cooldown
// schedule (ascend a fixed 1,1,2,3,5,8,... table, reset to the smallest
// stride on every label switch) that served the same purpose -- keep a
// pixel from attempting a long, hard-to-reverse jump before local consensus
// has formed -- but did so indirectly, via "how many consecutive rounds has
// THIS PIXEL personally gone without switching," a proxy for trustworthiness
// rather than trustworthiness itself. r is the direct measure: a region that
// has grown LARGE but stayed geometrically INCOHERENT (members disagreeing
// in angle) gets a small r regardless of member count, so it can't earn a
// big stride either way -- exactly the "wrongly-fattened, geometrically
// incoherent clump" failure mode the old cooldown existed to prevent, now
// prevented by the same signal that already gates everything else here
// (growth compatibility, NFA alignment) rather than a separate proxy for it.
// See this session's chat for the full back-and-forth.
//
// One real behavioral consequence of this swap: stride is now a per-REGION
// quantity (every pixel currently sharing a label gets the identical stride
// this round, since it depends only on sumCos[ownLabel]/sumSin[ownLabel]),
// not per-PIXEL the way the old schedule's own curStrideIdx was tracked --
// a pixel that just joined a large, coherent region inherits that region's
// already-earned stride immediately, rather than having to individually
// re-earn it. Intentional: what should gate a big jump is "is the group
// this pixel now belongs to trustworthy," not "has this specific pixel
// personally been stable a while."
//
// No separate per-pixel state survives across rounds for this anymore
// (contrast the old curStrideIdx/nextStrideIdx pair) -- r is recomputed
// fresh every round from sumCos/sumSin, which the round loop already
// recomputes fresh via its own segmented reduction regardless. One less
// buffer to carry/double-buffer round to round, and one less thing a future
// GPU port of this stage would need to synchronize.
//
// The offsets themselves are STEERED, not isotropic: classic JFA samples
// all 8 directions because ANY 2D displacement must be reachable for a
// Voronoi/area result, but our target is a 1D ridge -- most of those 8
// directions would just be wasted perpendicular checks. Each candidate
// pixel instead samples offsets near its OWN CURRENT LABEL's group-
// aggregate direction (both signs, since a level line is axial -- see
// the level-line vector block (lsdSegments.ts)) -- a denoised, multi-pixel average once a
// region has grown some size, rather than this one pixel's own individual
// (noisier) reading, fanned by +-fanRad(stride) (3 angles per side: 0,
// +fanRad, -fanRad). See growRegionsJFA's own body for how this is derived
// straight from the group's doubled-angle vector (no atan2 round-trip);
// fanRad/its cos/sin are plain inline trig calls now (stride is a continuous
// r rather than one of a fixed set of table-indexed values, so there's no
// longer a small fixed set of (stride, fanRad) pairs worth precomputing).
//
// fanRad is bounded by ABSOLUTE lateral pixel drift, not a fixed angle:
// fanRad(stride) = min(toleranceRad, atan(maxLateralSearchPx / stride))
// (settings.lsdMaxLateralSearchPx, default 2 -- on the order of the level
// line's own 1-2px ridge width, see countRectanglePixels' own half-width
// floor for the same idea). A fixed angular fan (the first version of this
// function used exactly +-toleranceDeg at every stride) drifts sideways by
// stride*sin(fan) pixels -- at a large stride (up to ~half the image's
// longest side, reached in the LATER rounds now) even a modest angle drifts
// tens to hundreds of pixels off the true tangent line, easily far enough
// to land on a COMPLETELY DIFFERENT parallel ridge (e.g. an adjacent De Bruijn grid row)
// that merely shares the same direction -- exactly the "two adjacent rows
// have identical direction vectors despite being unrelated" failure
// bucketFillJoin.ts's own mergeAt already has a dedicated connecting-vector
// check for, just unguarded here. Since the compatibility test only checks
// ANGLE agreement, a wrongly-reached parallel ridge passes it trivially
// and, if currently stronger, wins the pixel -- and a bad merge like that
// compounds (the wrongly-fattened region has even more mass to keep winning
// with next round). Scaling fanRad so lateral drift stays bounded at
// maxLateralSearchPx keeps every jump -- however far -- searching a thin
// cone hugging the true tangent line, while still clamping at toleranceRad
// for small strides (so this never searches wider than the angle the
// compatibility test would go on to accept anyway).
//
// RETIRED-NOTE: growSteps was a live production setting (settings.lsdGrowSteps,
// mirroring lsdJoinSteps' own dual role in that one respect). That setting is
// gone -- this function now takes it as a plain argument with no UI behind it.
// growSteps is purely "how many rounds to run". It no longer ALSO caps how far a
// stride can reach -- that ceiling is just largestStride (image-derived,
// see below), applied directly to r every round regardless of growSteps.
// growSteps still shapes how far growth can practically get in the time
// available (a region needs enough rounds to accumulate a large resultant
// length before it can search far), but there's no closed-form "rounds to
// reach stride X" the way the old Fibonacci table gave -- how fast r grows
// depends on how quickly a region actually accumulates coherent members,
// which is data-dependent, not schedule-dependent.
export interface GrowthCandidate { dx: number; dy: number }

// The per-pixel candidate-offset derivation growRegionsJFA's own inner loop
// runs every round -- pulled out as its own function so overlays/lsdOverlay.ts's
// growth-candidate-preview arrow (hover debug view, "where would this pixel
// look next") could call the EXACT same logic instead of an independently
// reimplemented copy that could quietly drift out of sync with the real
// algorithm. RETIRED-NOTE: that overlay is gone too -- it was replaced by
// drawEdgeConnectivityPreview, which asks the stride-1 equivalent question
// ("which of my 8 neighbors did I actually connect to, and why not the
// others") against growRegionsCCL's own predicate.
//
// See growRegionsJFA's own header for the reasoning behind stride = r and the
// steered/fanned search directions.
//
// ownSumCos/ownSumSin: the pixel's own CURRENT LABEL's group doubled-angle
// sum (sumCos[ownLabel]/sumSin[ownLabel] in growRegionsJFA's own round loop).
// ownCos2Theta/ownSin2Theta: this pixel's OWN doubled-angle reading (cos2theta[i]/
// sin2theta[i]), used only as the degenerate r===0 fallback.
// strideDivider: per-PIXEL (not per-region) shrink factor -- growRegionsJFA's
// own round loop doubles this every consecutive round a pixel's label
// DOESN'T change (wrapping back to 1 at STRIDE_DIVIDER_MAX rather than
// growing forever), and resets it to 1 the instant it does change. See that
// function's own header for why: stride = r alone can only ever grow as a
// region strengthens, so a genuinely closer gap smaller than the current
// stride would otherwise become permanently unreachable once r outgrows it
// -- a pixel that keeps finding nothing progressively narrows its OWN search
// radius to get a shot at exactly that, then wraps back to the full stride
// periodically rather than staying shrunk forever. Applied to the ALREADY
// largestStride-clamped value (not to raw r) so a very large, long-stable r
// divides down to the same result a fresh r would at the same divider. This
// function itself doesn't need to enforce the wrap -- the max(1, ...) floor
// below already handles whatever divider it's handed gracefully, bottoming
// out at stride 1 even for a divider larger than STRIDE_DIVIDER_MAX.
export function computeGrowthCandidates(
  ownSumCos: number, ownSumSin: number, ownCos2Theta: number, ownSin2Theta: number,
  toleranceRad: number, maxLateralSearchPx: number, largestStride: number, exactTangentOnly: boolean,
  strideDivider: number = 1,
): { stride: number; candidates: GrowthCandidate[] } {
  let Cx = ownSumCos, Cy = ownSumSin;
  let r = Math.hypot(Cx, Cy);
  if (r === 0) { Cx = ownCos2Theta; Cy = ownSin2Theta; r = 1; }
  const ux = Math.sqrt(Math.max(0, (r + Cx) / (2 * r)));
  const uy = (Cy >= 0 ? 1 : -1) * Math.sqrt(Math.max(0, (r - Cx) / (2 * r)));

  const stride = Math.max(1, Math.min(largestStride, r) / strideDivider);
  const fanRad = Math.min(toleranceRad, Math.atan(maxLateralSearchPx / stride));
  const fc = Math.cos(fanRad), fs = Math.sin(fanRad);

  const candidates: GrowthCandidate[] = [];
  for (const axisSign of [1, -1]) {
    const bx = axisSign * ux, by = axisSign * uy;
    candidates.push({ dx: bx, dy: by });
    if (!exactTangentOnly) {
      candidates.push({ dx: bx * fc - by * fs, dy: bx * fs + by * fc });
      candidates.push({ dx: bx * fc + by * fs, dy: -bx * fs + by * fc });
    }
  }
  return { stride, candidates };
}

export function growRegionsJFA(
  mag: Float64Array, theta: Float64Array, w: number, h: number,
  toleranceDeg: number, rho: number, growSteps: number, maxLateralSearchPx: number, exactTangentOnly: boolean,
): { regionId: Int32Array; regions: GrownRegion[]; strideDivider: Float64Array } {
  const n = w * h;
  const toleranceRad = THREE.MathUtils.degToRad(toleranceDeg);
  // Double-angle tolerance (mod π: theta and theta+π both pass) -- see this
  // file's header on why level-line polarity is now allowed to flip mid-
  // region (a De Bruijn grid line's own value encoding), and
  // bucketFillSegments.ts's own computeBucketFillRegions for the identical
  // cosTol = cos(2*toleranceRad) convention applied to a genuinely axial
  // quantity.
  const cosTol = Math.cos(2 * toleranceRad);

  let label = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) if (mag[i] > rho) label[i] = i; // dense: every eligible pixel is its own seed

  let largestStride = 1;
  while (largestStride * 2 < Math.max(w, h)) largestStride *= 2;

  // theta[i] never changes across rounds -- only which LABEL owns pixel i
  // does -- so its doubled-angle vector is precomputed ONCE here rather
  // than recomputed by every round's reduction pass (the original version
  // of this function called Math.cos/sin(2*theta[i]) fresh every round for
  // the exact same fixed value). See this session's chat.
  const cos2theta = new Float64Array(n), sin2theta = new Float64Array(n);
  for (let i = 0; i < n; i++) { cos2theta[i] = Math.cos(2 * theta[i]); sin2theta[i] = Math.sin(2 * theta[i]); }

  // Reused across rounds, indexed directly by label value (a pixel index,
  // so bounded by n) -- reset with .fill(0) each round rather than
  // reallocated, since the reset itself is already the same O(n) order as
  // the rest of a round. sumCos/sumSin accumulate at DOUBLE angle (2*theta)
  // -- unlike a sign-resolved raw-angle sum, a double-angle sum needs no
  // running reference to resolve against and can't be corrupted by which
  // order members happened to join in, the same reason
  // computeGradientAgreementField/computeBucketFillRegions fold by 2*theta
  // for their own online tolerance tests.
  const sumCos = new Float64Array(n), sumSin = new Float64Array(n), sumMag = new Float64Array(n);

  // Per-PIXEL (not per-region -- unlike stride's own r source, this is
  // exactly the kind of individually-earned/lost state stride moved away
  // from) shrink factor: doubles every consecutive round a pixel's label
  // DOESN'T change, resets to 1 the instant it does. Complements stride = r
  // (which can only ever GROW as a region strengthens, never shrink on its
  // own) -- without this, a real gap smaller than the current stride would
  // become permanently unreachable the moment r outgrows it, since every
  // future round's candidates only probe at r or its fan variants. A pixel
  // that keeps finding nothing progressively narrows its own search radius
  // instead, independent of what stride the rest of its region is using.
  //
  // WRAPS back to 1 at STRIDE_DIVIDER_MAX rather than doubling forever: an
  // unbounded divider is a one-way ratchet -- once a long-stalled pixel has
  // shrunk all the way down, it would never attempt the full r-based stride
  // again unless it happens to change labels first, even though r itself
  // keeps evolving round to round as OTHER pixels join or leave its region.
  // Cycling 1,2,4,8,16,1,... gives a stalled pixel a repeated shot at the
  // full stride every 5 rounds, not just once before shrinking permanently.
  // See this session's chat.
  const STRIDE_DIVIDER_MAX = 16;
  let strideDivider = new Float64Array(n).fill(1);

  for (let round = 0; round < growSteps; round++) {
    sumCos.fill(0); sumSin.fill(0); sumMag.fill(0);
    for (let i = 0; i < n; i++) {
      const lab = label[i];
      if (lab === -1) continue;
      sumCos[lab] += cos2theta[i]; sumSin[lab] += sin2theta[i]; sumMag[lab] += mag[i];
    }

    const nextLabel = label.slice(); // read ONLY from `label` (this round's frozen input) below -- never from nextLabel
    const nextStrideDivider = strideDivider.slice();
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const ownLabel = label[i];
        if (ownLabel === -1) continue; // below rho -- never eligible to participate at all
        const c2 = cos2theta[i], s2 = sin2theta[i]; // this PIXEL's own raw reading -- still what the compatibility test below checks

        // Search direction (denoised group aggregate, half-angle-recovered
        // from the doubled vector) and stride (= r / this pixel's own
        // strideDivider) both come from computeGrowthCandidates -- see that
        // function's own comment and this file's header for the reasoning.
        const { stride, candidates } = computeGrowthCandidates(
          sumCos[ownLabel], sumSin[ownLabel], c2, s2, toleranceRad, maxLateralSearchPx, largestStride, exactTangentOnly,
          strideDivider[i],
        );

        let bestLabel = ownLabel, bestStrength = sumMag[ownLabel];
        for (const { dx, dy } of candidates) {
          const nx = Math.round(x + stride * dx), ny = Math.round(y + stride * dy);
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const nlab = label[ny * w + nx];
          if (nlab === -1 || nlab === ownLabel) continue;
          const avgLen = Math.hypot(sumCos[nlab], sumSin[nlab]);
          const cosDeviation = avgLen > 0 ? (c2 * sumCos[nlab] + s2 * sumSin[nlab]) / avgLen : 1;
          if (cosDeviation < cosTol) continue;
          if (sumMag[nlab] > bestStrength) { bestStrength = sumMag[nlab]; bestLabel = nlab; }
        }
        nextLabel[i] = bestLabel;
        nextStrideDivider[i] = bestLabel !== ownLabel || strideDivider[i] >= STRIDE_DIVIDER_MAX ? 1 : strideDivider[i] * 2;
      }
    }
    label = nextLabel;
    strideDivider = nextStrideDivider;
  }

  // ── Collect: group by final label, remap to dense ids ──────────────────
  // Inherently serial, same role as the old BFS version's own post-pass --
  // runs once after every round has finished, not once per round, so it
  // isn't part of what needs to parallelize.
  const membersByLabel = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const lab = label[i];
    if (lab === -1) continue;
    let list = membersByLabel.get(lab);
    if (!list) { list = []; membersByLabel.set(lab, list); }
    list.push(i);
  }
  const sortedLabels = Array.from(membersByLabel.keys()).sort((a, b) => a - b); // deterministic order

  const regionId = new Int32Array(n).fill(-1);
  const regions: GrownRegion[] = [];
  for (const lab of sortedLabels) {
    const members = membersByLabel.get(lab)!;
    const id = regions.length;
    for (const p of members) regionId[p] = id;

    // meanAngle needs to be a genuine DIRECTED value (see GrownRegion's own
    // consumer, fitRectangle's PCA-sign resolution below) -- a plain sum of
    // raw cos/sin would let a region that legitimately spans a polarity
    // flip (allowed now, see this file's header) partially or fully cancel
    // itself out instead of reinforcing. Sign-resolve each member's raw
    // angle against ONE fixed reference (label `lab`'s own raw angle, a
    // real pixel -- not the running sum, since this is a flat one-time pass
    // over an already-finalized member list, not an online growth loop)
    // before adding -- same technique bucketFillSegments.ts's own online
    // growth loop uses for its genuinely-axial fx/fy, just against a fixed
    // reference instead of an evolving one, matching how the deleted
    // labelPropagationSegments.ts's own flat collection pass did this.
    //
    // RETIRED-NOTE: this pass still works in ANGLE space (it takes a `theta`
    // array and cos/sin's it) because it is unreferenced reference code that
    // predates the vector-space conversion. The live path carries the
    // level-line vector directly and has no theta array to hand it -- so
    // reviving this would mean converting it too, not just re-wiring a call.
    // Only the GrownRegion it emits was updated, to keep the file compiling.
    const refX = Math.cos(theta[lab]), refY = Math.sin(theta[lab]);
    let sc = 0, ss = 0;
    for (const p of members) {
      const px = Math.cos(theta[p]), py = Math.sin(theta[p]);
      if (px * refX + py * refY < 0) { sc -= px; ss -= py; } else { sc += px; ss += py; }
    }
    const sLen = Math.hypot(sc, ss);
    regions.push({
      members: Int32Array.from(members),
      meanUx: sLen > 0 ? sc / sLen : 1, meanUy: sLen > 0 ? ss / sLen : 0,
    });
  }
  return { regionId, regions, strideDivider };
}
