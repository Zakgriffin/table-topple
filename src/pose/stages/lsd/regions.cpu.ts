import * as THREE from 'three';
import { NEIGHBOR_DX, NEIGHBOR_DY, eligibilityThresholdSq, levelLinesCompatible } from './levelLine.ts';
import { type GrownRegion } from './types.ts';

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

// A hard ceiling on rounds, purely so a debug tool can never hang. Hook alone
// (no compression) propagates one pixel per round, so the longest possible
// chain sets a true upper bound; compression makes the real count
// logarithmic, so this is never reached in practice -- if it ever IS, that is
// a bug in the fixpoint detection, not a legitimately slow image.
function roundHardCap(w: number, h: number): number { return w + h + 64; }

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
// The pair is SPLIT ACROSS THE TWO STAGES below, which is worth stating since
// the block above describes them together: rhoLow gates who may participate in
// an edge and is read by the round loop, while rhoHigh gates which finished
// components survive and is read only by the collector.

// Hysteresis survival + the collect/relabel pass: the `lsd.collect` STAGE, and
// the CPU counterpart of encodeCollectRegions. It takes a finished labeling and
// nothing else, which is what lets both backends call grow and collect as two
// steps in the same order. It used to be called from inside `growRegionsCCL`,
// where it was reachable only by growing first.
//
// This is the CPU route AND the fallback; pose/stages/lsd/collectRegions.gpu.ts is the
// GPU one. An earlier version of this comment called the step "inherently
// serial", which was simply wrong: every stage is a standard parallel pattern
// -- labelSurvives is a scatter of a constant (no atomic needed, every writer
// writes 1), the grouping is a histogram + prefix sum + scatter CSR build,
// ascending label order falls out of the scan for free because labels ARE pixel
// indices, and meanAngle is a segmented reduction over the resulting slices.
// What actually blocked it was the missing prefix-sum primitive, now
// pose/gpu/prefixSum.ts.
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

// maxRounds is a DEBUG SCRUBBER, not a tuning parameter: 0 means run to the
// fixpoint (the real algorithm, and the production default), while 1..N caps
// the round count so an overlay can watch components coalesce. Unlike the
// growSteps it replaces, changing it cannot change the converged answer --
// only how much of the way there you are looking at.
//
// The ROUND LOOP, and nothing else. It returns the finished labeling; turning
// that into regions is `collectRegionsFromLabels`, which the caller calls
// itself -- the same two-step the GPU path has always had (`encodeGrowInit` +
// `encodeGrowRound`, then `encodeCollectRegions`). This used to end by calling
// the collector, which made `label` the one stage output that existed on only
// one backend, cost `lsd.collect` its CPU span, and left the two decompositions
// disagreeing about where a stage boundary is.
//
// Note what left the signature with the collector: `rhoHigh` and
// `minRegionSize` are hysteresis and prefilter parameters that no round ever
// read. The remaining three are exactly `encodeGrowInit`'s inputs plus the cap.
export function growRegionsCCL(
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  toleranceDeg: number, rhoLow: number, maxRounds: number,
): { label: Int32Array; roundsRun: number; converged: boolean } {
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

  return { label, roundsRun, converged };
}
