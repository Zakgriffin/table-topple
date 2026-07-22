import { BucketFillSegment, segmentLength } from './bucketFillSegments.ts';
import { hsvToRgb, rgbToHueDeg } from './distortion.ts';

// ── Line-segment joining (pure) ───────────────────────────────────────────
//
// A second, separate stepped process on top of the bucket-fill segments
// (pipeline/bucketFillSegments.ts): from each segment's two endpoints, walk
// straight out along that segment's own tangent direction (fixed for the
// whole walk -- there's no more field data to re-steer by once we're past
// the segment's actual pixels, unlike the original flood fill). The
// UNDERLYING data structure is conceptually a texture of SEGMENT INDICES,
// not colors -- each cell holds which segment currently claims it (-1 =
// unclaimed) -- so a front walking into a cell already claimed by a
// DIFFERENT segment can look that segment up directly and test whether the
// two are plausibly the same line (their gradient directions are close to
// collinear) before deciding to merge. Internally this is one level of
// indirection: a cell actually stores a HANDLE, resolved to a segment index
// through a small per-episode table (see computeJoinWalk's own comment) --
// this is what lets a front's pre-redirect trail be erased in O(1) (just
// invalidate its old handle) instead of walking back over every pixel it
// claimed. This buffer is SEEDED with the flood fill's own pixel
// footprint (regionId, from bucketFillSegments.ts's computeBucketFillRegions)
// before any front takes a step -- so a front reacts the moment it enters
// ANOTHER segment's real blob shape, not just when it happens to cross that
// segment's own thin front-trace line. Only segments that pass minLengthPx
// (the same set that gets fronts at all) get seeded, so a too-short segment
// stays exactly as invisible to the join walk as it already is everywhere
// else. A collision that FAILS the similarity test (almost
// always a crossing perpendicular segment on this grid) gets passed through
// without claiming the cell, not treated as a dead end, so a front can keep
// searching past unrelated crossings for its true collinear partner further
// along. A collision that PASSES only actually MERGES if the two specific
// fronts involved were walking HEAD-ON (roughly opposite directions) -- the
// expected case for two ends of one line meeting in the middle: both stop,
// they found their neighbor, this point is now internal to the composite,
// not an extremity -- but each one's SIBLING (the other front on that same
// segment, still headed outward toward the group's other, still-unexplored
// end) gets redirected instead: it SNAPS to the group's own composite
// endpoint on that side (not wherever it currently is) and keeps walking
// outward from there, now along the merged group's own composite direction
// (see the live union-find below) rather than its own original segment's
// estimate -- safe to jump straight there since its entire pre-redirect
// trail gets erased anyway (see the handle-indirection comment below), so
// nothing stale is left behind between the old and new positions. If instead the
// two fronts were walking the SAME direction (the DISCOVERING front caught
// up to/overtook the DISCOVERED front's own trail, not closing a gap), this
// is NOT treated as a merge at all: only the discovering front stops, the
// discovered one is left completely untouched, no group/composite update
// happens. So after a real (head-on) merge, exactly the two fronts still
// facing outward survive, sharing one jointly-informed direction -- this
// lets a chain of 3+ segments keep discovering further collinear neighbors,
// correcting course toward the group's actual line as more evidence comes
// in, rather than a fixed pair only ever finding one match each. Colors are
// purely a rendering concern, see paintJoinOverlay.
//
// Deterministic and fully recomputed from scratch for whatever step count is
// requested (see numSteps) -- "step N" always means the same walk state, so
// moving the step slider back and forth doesn't need any incremental/stateful
// tracking, just a pure re-run.

export interface SegmentMerge { a: number; b: number }

interface JoinFront {
  seg: number;
  fi: number; // this front's own index into the `fronts` array -- lets a redirect re-derive its own identity (for allocateHandle) without a linear search
  startX: number; startY: number;
  dx: number; dy: number; // fixed unit tangent direction for this front's whole walk
  k: number; // steps taken so far -- position is always startX + k*dx, startY + k*dy (recomputed fresh each step, not accumulated, to avoid drift)
  active: boolean;
  handle: number; // this front's CURRENT episode handle -- every cell it claims while walking this episode is tagged with this number, see computeJoinWalk's own comment
}

export function computeJoinWalk(
  segments: BucketFillSegment[], regionId: Int32Array, w: number, h: number, minSimilarity: number, numSteps: number, minLengthPx: number,
): {
  joinBuffer: Int32Array; merges: SegmentMerge[];
  sameDirMergePoints: { x: number; y: number }[]; oppositeDirMergePoints: { x: number; y: number }[];
} {
  const n = w * h;
  const merges: SegmentMerge[] = [];
  // Merge points split by the two colliding fronts' own TRAVEL directions
  // (JoinFront.dx/dy, not the segments' orientation-only gradient axis) --
  // walking the SAME way vs head-on/opposite -- see this function's own
  // dot-product check below for what that distinguishes and why.
  const sameDirMergePoints: { x: number; y: number }[] = [];
  const oppositeDirMergePoints: { x: number; y: number }[] = [];
  if (segments.length === 0) {
    return { joinBuffer: new Int32Array(n).fill(-1), merges, sameDirMergePoints, oppositeDirMergePoints };
  }

  // -- Handle indirection ---------------------------------------------------
  // cellHandle (the actual working buffer the walk reads/writes -- NOT the
  // joinBuffer this function returns, see the resolve pass at the bottom)
  // stores a HANDLE per cell, not a segment index directly. episodeOwner[h]
  // resolves a handle to the segment index that currently owns it, or -1 once
  // that handle has been invalidated. Handles 0..segments.length-1 are
  // PERMANENT, one per segment (episodeOwner[i] = i, never invalidated) --
  // reserved for the flood-fill blob seed below, since a segment's own real
  // footprint never goes away. Every FRONT additionally gets its own fresh
  // handle for each "episode" of its walk (from creation, or from its most
  // recent redirect, until its next redirect) -- erasing a front's
  // pre-redirect trail is then just `episodeOwner[oldHandle] = -1`, an O(1)
  // invalidation instead of walking back over every pixel it claimed: every
  // cell still holding that handle silently resolves to -1 (unclaimed) the
  // next time anything reads it. This also lets "which front owns this cell"
  // (episodeFront, replacing what used to be a full-image-resolution
  // claimedByFront: Int32Array) live as a tiny per-EPISODE table instead of a
  // per-PIXEL one -- bounded by how many episodes ever exist (roughly 2 per
  // segment plus one per redirect), not by image size.
  const cellHandle = new Int32Array(n).fill(-1);
  const episodeOwner: number[] = [];
  const episodeFront: number[] = [];
  for (let i = 0; i < segments.length; i++) { episodeOwner.push(i); episodeFront.push(-1); }
  function allocateHandle(seg: number, fi: number): number {
    const h = episodeOwner.length;
    episodeOwner.push(seg);
    episodeFront.push(fi);
    return h;
  }

  // minSimilarity is compared DIRECTLY against |dot(unitA, unitB)| -- both
  // sides are already-normalized unit vectors before the dot product, so
  // this is already a clean cosine similarity in [0,1], no angle conversion
  // needed. |dot| (not the raw signed dot) ranges 0 (perpendicular) to 1
  // (parallel OR antiparallel -- the abs() is what makes this correctly
  // axial, same reasoning as bucketFillSegments.ts's own sign-resolution,
  // just applied to a single pairwise comparison instead of an online
  // running average). gradUX/gradUY below are each segment's own individual
  // gradient direction, used as: (a) the walk's fixed per-front tangent
  // direction (unaffected by any of this -- a front's own step direction
  // never changes except via an explicit redirect), and (b) the FALLBACK
  // similarity-test direction for a segment that hasn't merged with anything
  // yet. Once a segment IS part of a merged group, the similarity test
  // instead uses groupGradientAxis() below, which reads that group's live
  // composite line direction -- not this raw per-segment value -- so the
  // test always reflects the group's current, jointly-informed direction
  // rather than whichever single original member happens to still own the
  // pixel a new front collided with.
  // Same eligibility test every other consumer of minLengthPx uses (base
  // raster, markers, votes) -- computed once up front since it's needed both
  // for the blob-seeding pass below and for which segments get fronts at all.
  const eligible = segments.map((seg) => segmentLength(seg) >= minLengthPx);

  // Seed the buffer with the flood fill's own footprint (see this file's
  // header) BEFORE any front takes a step -- ineligible segments' pixels are
  // left at -1, same as if they'd never been flood-filled at all. A
  // segment's own permanent handle IS its regionId value (both reserved as
  // 0..segments.length-1, see above), so this is a direct write, no
  // allocation needed.
  for (let i = 0; i < n; i++) {
    const rid = regionId[i];
    if (rid >= 0 && eligible[rid]) cellHandle[i] = rid;
  }

  const gradUX = new Float64Array(segments.length), gradUY = new Float64Array(segments.length);
  const fronts: JoinFront[] = [];
  // Each segment's two front INDICES (not the fronts themselves) -- needed
  // at merge time to find a colliding front's SIBLING (the other front on
  // the same segment, still headed toward the group's other, unexplored
  // end) without a linear search.
  const frontsBySegment = new Map<number, [number, number]>();
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (!eligible[i]) continue; // too short to trust its endpoint-to-endpoint direction -- no fronts at all, can't grow OR be merged into
    const gLen = Math.hypot(seg.avgFx, seg.avgFy);
    const gux = gLen > 0 ? seg.avgFx / gLen : 1, guy = gLen > 0 ? seg.avgFy / gLen : 0;
    gradUX[i] = gux; gradUY[i] = guy;
    const tx = -guy, ty = gux; // tangent = gradient rotated 90 degrees, same convention as the marker drawing
    // A front's very first claimed pixel would just be its own segment's own
    // endpoint -- already covered by the permanent blob seed above (same
    // segment, same value), so there's no separate "claim my starting pixel"
    // step here anymore; each front's own handle only ever gets used once it
    // actually steps somewhere new.
    const alongFi = fronts.length;
    fronts.push({
      seg: i, fi: alongFi, startX: seg.endAlongX, startY: seg.endAlongY, dx: tx, dy: ty, k: 0, active: true,
      handle: allocateHandle(i, alongFi),
    });
    const againstFi = fronts.length;
    fronts.push({
      seg: i, fi: againstFi, startX: seg.endAgainstX, startY: seg.endAgainstY, dx: -tx, dy: -ty, k: 0, active: true,
      handle: allocateHandle(i, againstFi),
    });
    frontsBySegment.set(i, [alongFi, againstFi]);
  }

  // Live union-find with satellite data: same idea as computeMergeGroups
  // below, but maintained WHILE the walk runs (not after) so a redirect can
  // ask "what's this segment's group's composite line RIGHT NOW". Each
  // root's own compositeX1/Y1/X2/Y2 entry holds that group's current
  // farthest-apart endpoint pair (see computeCompositeLines for the same
  // "farthest pair among candidates" idea, done here incrementally: a union
  // only ever has 4 candidate points to consider -- the two groups' existing
  // composite pairs -- not a rescan of every member). Always read/write
  // through find(root), never a cached index, since path compression can
  // change which index IS the root as more unions happen.
  const parent = new Int32Array(segments.length);
  for (let i = 0; i < segments.length; i++) parent[i] = i;
  function find(x: number): number {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  const compositeX1 = new Float64Array(segments.length), compositeY1 = new Float64Array(segments.length);
  const compositeX2 = new Float64Array(segments.length), compositeY2 = new Float64Array(segments.length);
  for (const f of fronts) {
    // Both of a segment's fronts initialize the SAME composite pair (its own
    // two endpoints) -- harmless double-write, avoids needing a separate
    // per-segment init pass.
    compositeX1[f.seg] = segments[f.seg].endAlongX; compositeY1[f.seg] = segments[f.seg].endAlongY;
    compositeX2[f.seg] = segments[f.seg].endAgainstX; compositeY2[f.seg] = segments[f.seg].endAgainstY;
  }
  function union(a: number, b: number): number {
    const ra = find(a), rb = find(b);
    if (ra === rb) return ra;
    let bestX1 = compositeX1[ra], bestY1 = compositeY1[ra], bestX2 = compositeX2[ra], bestY2 = compositeY2[ra];
    let bestDistSq = (bestX1 - bestX2) ** 2 + (bestY1 - bestY2) ** 2;
    const candidates: [number, number][] = [
      [compositeX1[ra], compositeY1[ra]], [compositeX2[ra], compositeY2[ra]],
      [compositeX1[rb], compositeY1[rb]], [compositeX2[rb], compositeY2[rb]],
    ];
    for (let p = 0; p < candidates.length; p++) {
      for (let q = p + 1; q < candidates.length; q++) {
        const dx = candidates[p][0] - candidates[q][0], dy = candidates[p][1] - candidates[q][1];
        const distSq = dx * dx + dy * dy;
        if (distSq > bestDistSq) {
          bestDistSq = distSq;
          bestX1 = candidates[p][0]; bestY1 = candidates[p][1]; bestX2 = candidates[q][0]; bestY2 = candidates[q][1];
        }
      }
    }
    parent[ra] = rb;
    compositeX1[rb] = bestX1; compositeY1[rb] = bestY1; compositeX2[rb] = bestX2; compositeY2[rb] = bestY2;
    return rb;
  }
  // Redirects a SIBLING front -- not one of the two that just collided (those
  // stop, see below), but the OTHER front on each of those two segments,
  // still headed toward the group's other, still-unexplored extremity. SNAPS
  // to the composite's own known endpoint on that side (not wherever the
  // front currently happens to be mid-walk) -- the group's actual, currently-
  // known reach -- then continues outward from there along the group's just-
  // updated composite axis instead of its own original segment's estimate,
  // so from here on both segments' outward-facing rays share one jointly-
  // informed direction. Sign chosen to keep it moving roughly the way it
  // already was (dot product against its OLD direction), not double back on
  // itself; the snapped start point is whichever composite endpoint that
  // final direction points AWAY from (continuing past the group's own known
  // extremity, not back into the line it just came from). Safe to jump
  // straight there -- rather than continuing from the front's current
  // position and leaving the skipped stretch behind as a stale trail -- since
  // the whole pre-redirect episode gets erased below anyway (see the handle-
  // indirection comment above), regardless of where the front resumes from.
  function redirectSibling(front: JoinFront, root: number) {
    if (!front.active) return; // already stopped (out of bounds, or already consumed by an earlier merge) -- nothing to redirect
    let ndx = compositeX2[root] - compositeX1[root], ndy = compositeY2[root] - compositeY1[root];
    const ndLen = Math.hypot(ndx, ndy);
    if (ndLen < 1e-9) return; // degenerate (shouldn't happen for any group with 2+ distinct points) -- leave direction as-is
    ndx /= ndLen; ndy /= ndLen;
    const sign = (ndx * front.dx + ndy * front.dy) >= 0 ? 1 : -1;
    if (sign > 0) { front.startX = compositeX2[root]; front.startY = compositeY2[root]; }
    else { front.startX = compositeX1[root]; front.startY = compositeY1[root]; }
    front.dx = ndx * sign; front.dy = ndy * sign;
    front.k = 0;
    // Erase this front's pre-redirect trail (its whole prior episode, not
    // just the stretch between its old position and the new snapped one) --
    // O(1), no pixel-walking needed: every cell it claimed under its OLD
    // handle now silently resolves to -1 (unclaimed) the next time anything
    // reads it. It then gets a fresh handle for its continuing walk from the
    // new snapped position/direction just set above.
    episodeOwner[front.handle] = -1;
    front.handle = allocateHandle(front.seg, front.fi);
  }

  // The direction used for the merge-similarity test -- a segment's current
  // GROUP's live composite line direction (via find(seg)'s root), not just
  // that one segment's own raw gradient. This makes the test see the same
  // answer regardless of which specific member pixel a front happens to
  // collide with (a 3-segment chain's trail is a mix of all 3 members'
  // pixels, but should always be tested as "one C", not "whichever of A/B/C
  // this particular pixel happens to still say"). compositeX1/Y1/X2/Y2 are
  // valid even for a still-unmerged singleton (initialized to its own
  // endAlong/endAgainst), so this same lookup works uniformly whether or not
  // a merge has happened yet -- no separate merged/unmerged branch needed.
  // The composite pair is a TANGENT-axis direction (along the line, see
  // computeCompositeLines); rotated back to the gradient axis here (inverse
  // of the tx=-guy,ty=gux convention above) purely so it's directly
  // comparable to gradUX/gradUY's convention -- since the test only ever
  // uses |dot|, rotating both sides by the same 90 degrees wouldn't actually
  // change the result, but keeping one consistent axis convention throughout
  // the file avoids confusion.
  function groupGradientAxis(seg: number): [number, number] {
    const root = find(seg);
    const tx0 = compositeX2[root] - compositeX1[root], ty0 = compositeY2[root] - compositeY1[root];
    const len = Math.hypot(tx0, ty0);
    if (len < 1e-9) return [gradUX[seg], gradUY[seg]]; // degenerate composite (shouldn't happen once past the length filter) -- fall back to the raw per-segment gradient
    return [ty0 / len, -tx0 / len];
  }

  for (let step = 1; step <= numSteps; step++) {
    for (let fi = 0; fi < fronts.length; fi++) {
      const f = fronts[fi];
      if (!f.active) continue;
      f.k++;
      const nx = Math.round(f.startX + f.k * f.dx), ny = Math.round(f.startY + f.k * f.dy);
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) { f.active = false; continue; }
      const idx = ny * w + nx;
      const rawOccupant = cellHandle[idx];
      // -1 here covers BOTH a cell that's never been touched AND one whose
      // handle was invalidated by an earlier redirect (see the handle-
      // indirection comment above) -- both read as plain "unclaimed".
      const occupant = rawOccupant === -1 ? -1 : episodeOwner[rawOccupant];
      if (occupant === -1) {
        cellHandle[idx] = f.handle;
      } else if (occupant !== f.seg && find(f.seg) === find(occupant)) {
        // Already the same group, just reached via a different path through
        // the chain (e.g. both ends of a 3-segment chain sweeping back
        // toward each other) -- nothing new to learn, pass through exactly
        // like the own-segment case below.
      } else if (occupant !== f.seg) {
        const [gux, guy] = groupGradientAxis(f.seg);
        const [oux, ouy] = groupGradientAxis(occupant);
        const dot = gux * oux + guy * ouy;
        if (Math.abs(dot) >= minSimilarity) {
          // -1 for a permanent blob-seed handle (no owning front); a real
          // front index otherwise.
          const otherFi = episodeFront[rawOccupant];
          // Distinguishes head-on ("closing the gap", the expected case for
          // two ends of one line meeting in the middle) from same-direction
          // collisions (the DISCOVERING front f caught up to/overtook the
          // DISCOVERED front's own trail, walking the same way -- not two
          // ends closing a gap at all). This checks the two SPECIFIC FRONTS'
          // own travel directions, not the segments' sign-insensitive
          // orientation similarity above -- that's already what let this
          // collision reach here, so this is strictly additional
          // information. Only meaningful when a specific front (not a bare
          // flood-fill blob seed, see this file's header) owns the cell --
          // otherFi < 0 always falls to the head-on branch below, since
          // there's no "discovered front" to distinguish from in that case.
          const isSameDirection = otherFi >= 0 && f.dx * fronts[otherFi].dx + f.dy * fronts[otherFi].dy >= 0;
          if (isSameDirection) {
            // Same-direction ("red X") -- NOT treated as a real merge: no
            // group union, no composite update, no redirect. Only the
            // DISCOVERING front (f, the one that just stepped into this
            // cell) stops; the DISCOVERED front (occupant's own front,
            // otherFi) is left completely untouched, still walking its own
            // way as if this never happened.
            sameDirMergePoints.push({ x: nx, y: ny });
            f.active = false;
          } else {
            if (otherFi >= 0) oppositeDirMergePoints.push({ x: nx, y: ny });
            merges.push({ a: f.seg, b: occupant });
            // The two fronts that just met are done -- they found their
            // neighbor, this point is now internal to the composite, not an
            // extremity. f's SIBLING (the other front on f's own segment,
            // still headed outward toward the group's real, still-unexplored
            // end on that side) is the one that keeps walking, see
            // redirectSibling's own comment.
            f.active = false;
            const root = union(f.seg, occupant);
            const [aAlong, aAgainst] = frontsBySegment.get(f.seg)!;
            redirectSibling(fronts[fi === aAlong ? aAgainst : aAlong], root);
            if (otherFi >= 0) {
              // Front-to-front: otherFi pins down EXACTLY which of
              // occupant's two fronts we hit, so its sibling (the other one)
              // is unambiguously "still headed outward on that side" -- same
              // redirect treatment as f's own sibling above.
              fronts[otherFi].active = false;
              const [bAlong, bAgainst] = frontsBySegment.get(occupant)!;
              redirectSibling(fronts[otherFi === bAlong ? bAgainst : bAlong], root);
            } else {
              // Blob hit: there's no specific front that owns the collided
              // cell, so unlike the front-to-front case there's no way to
              // tell which of occupant's own two fronts is "the sibling" of
              // whichever one we notionally ran into -- redirecting either
              // would be a guess. Stopping BOTH instead of guessing keeps
              // the "at most 2 active fronts per group" invariant intact.
              // The group's composite line stays fully correct either way
              // (union() reads it from occupant's own real endpoints, not
              // from whether either front is still alive) -- the only cost
              // is not continuing to actively explore past occupant's
              // original extent on this side.
              const [bAlong, bAgainst] = frontsBySegment.get(occupant)!;
              fronts[bAlong].active = false;
              fronts[bAgainst].active = false;
            }
          }
        }
        // else: NOT a match (e.g. a crossing perpendicular segment) -- pass
        // through without claiming the cell (leave the occupant's own claim
        // alone) and keep walking. Stopping unconditionally here was the
        // actual bug: on this grid, a front headed toward its true collinear
        // partner runs into unrelated crossing segments constantly, and
        // stopping at the FIRST one (regardless of match) meant most fronts
        // never got anywhere near far enough to find a real match.
      }
      // occupant === f.seg: this front (or its sibling front on the same
      // segment) already owns this cell -- pass through as a no-op.
    }
  }

  // Resolve the internal handle buffer down to plain segment indices for the
  // return value -- keeps this function's external contract (a segment-index
  // buffer, -1 = unclaimed) identical to before the handle indirection was
  // added, so every consumer (painting, votes.ts) is unaffected.
  const joinBuffer = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const h = cellHandle[i];
    joinBuffer[i] = h === -1 ? -1 : episodeOwner[h];
  }

  return { joinBuffer, merges, sameDirMergePoints, oppositeDirMergePoints };
}

// Union-find over the merge pairs -- two segments joined only indirectly
// (A merges with B, B merges with C) still end up in the same group, not
// just directly-touching pairs. Returns, per segment, the representative
// (root) segment index of its group -- an UNmerged segment is its own root.
export function computeMergeGroups(numSegments: number, merges: readonly SegmentMerge[]): Int32Array {
  const parent = new Int32Array(numSegments);
  for (let i = 0; i < numSegments; i++) parent[i] = i;
  function find(x: number): number {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  for (const m of merges) {
    const ra = find(m.a), rb = find(m.b);
    if (ra !== rb) parent[ra] = rb;
  }
  const groupOf = new Int32Array(numSegments);
  for (let i = 0; i < numSegments; i++) groupOf[i] = find(i);
  return groupOf;
}

// Every segment in a group displays with the BLEND of every member
// segment's own already-assigned random color (see randomSegmentColors) --
// not just the root's color, so a 3+-way chain blends all of them, not only
// the first two that happened to touch. Blended in HUE space (circular
// mean, so e.g. 350deg and 10deg correctly average to 0deg, not 180deg),
// not by averaging RGB directly -- averaging RGB pulls opposite-ish hues
// (red+cyan, etc) toward gray, since it's not accounting for hue being
// circular; segment colors are all generated at the same fixed
// saturation/value (randomSegmentColors), so re-emitting the blended hue at
// that same s/v keeps the result exactly as vivid as any individual
// segment's own color. Degenerates to a segment's own unchanged color when
// it never merged with anything (its group has exactly one member: itself).
export function groupDisplayColors(
  groupOf: Int32Array, colors: readonly [number, number, number][],
): [number, number, number][] {
  const n = groupOf.length;
  const sumCos = new Float64Array(n), sumSin = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const root = groupOf[i];
    const [r, g, b] = colors[i];
    const rad = (rgbToHueDeg(r, g, b) * Math.PI) / 180;
    sumCos[root] += Math.cos(rad); sumSin[root] += Math.sin(rad);
  }
  const blendedByRoot = new Map<number, [number, number, number]>();
  const out: [number, number, number][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const root = groupOf[i];
    let blended = blendedByRoot.get(root);
    if (!blended) {
      let meanDeg = (Math.atan2(sumSin[root], sumCos[root]) * 180) / Math.PI;
      if (meanDeg < 0) meanDeg += 360;
      blended = hsvToRgb(meanDeg, 0.85, 1);
      blendedByRoot.set(root, blended);
    }
    out[i] = blended;
  }
  return out;
}

export interface CompositeLine { x1: number; y1: number; x2: number; y2: number }

// Pixel-space length of a composite line -- same idea as
// bucketFillSegments.ts's segmentLength, for the group-level composite
// instead of one raw segment's own endpoints.
export function compositeLineLength(line: CompositeLine): number {
  return Math.hypot(line.x1 - line.x2, line.y1 - line.y2);
}

// The "composite line" for each merge group: the two points, among EVERY
// member segment's own two endpoints, that end up farthest apart from each
// other. This is deliberately computed as a flat scan over the group's whole
// candidate set (every member's endAlong + endAgainst) rather than
// propagated incrementally merge-by-merge ("this pair collided, so keep the
// other two") -- both give the identical final answer once a chain settles,
// but the flat scan is order-independent and doesn't need special-casing
// for a segment that's already merged on one end merging again on its
// other end (which would otherwise need its OWN prior composite carried
// forward rather than its raw endpoints). A group of k member segments has
// 2k candidate points, so this is O(k^2) per group -- fine given how small
// groups actually get in practice (see computeMergeGroups), but would be
// worth revisiting if one single group ever swallowed a large fraction of
// all segments.
export function computeCompositeLines(segments: BucketFillSegment[], groupOf: Int32Array): Map<number, CompositeLine> {
  const candidatesByRoot = new Map<number, { x: number; y: number }[]>();
  for (let i = 0; i < segments.length; i++) {
    const root = groupOf[i];
    let list = candidatesByRoot.get(root);
    if (!list) { list = []; candidatesByRoot.set(root, list); }
    const seg = segments[i];
    list.push({ x: seg.endAlongX, y: seg.endAlongY });
    list.push({ x: seg.endAgainstX, y: seg.endAgainstY });
  }
  const result = new Map<number, CompositeLine>();
  for (const [root, points] of candidatesByRoot) {
    let best: CompositeLine = { x1: points[0].x, y1: points[0].y, x2: points[0].x, y2: points[0].y };
    let bestDistSq = 0;
    for (let a = 0; a < points.length; a++) {
      for (let b = a + 1; b < points.length; b++) {
        const dx = points[a].x - points[b].x, dy = points[a].y - points[b].y;
        const distSq = dx * dx + dy * dy;
        if (distSq > bestDistSq) { bestDistSq = distSq; best = { x1: points[a].x, y1: points[a].y, x2: points[b].x, y2: points[b].y }; }
      }
    }
    result.set(root, best);
  }
  return result;
}

export interface CompositeLineDisplay extends CompositeLine { color: [number, number, number] }

const JOIN_ALPHA = 130; // "faded" relative to the base fill's fully-opaque 255

export function paintJoinOverlay(joinBuffer: Int32Array, colors: readonly [number, number, number][], out: Uint8Array) {
  for (let i = 0; i < joinBuffer.length; i++) {
    const o = i * 4;
    const id = joinBuffer[i];
    if (id < 0) { out[o + 3] = 0; continue; }
    const [rr, gg, bb] = colors[id];
    out[o] = rr; out[o + 1] = gg; out[o + 2] = bb; out[o + 3] = JOIN_ALPHA;
  }
}
