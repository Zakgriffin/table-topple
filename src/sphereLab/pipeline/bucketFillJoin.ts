import { BucketFillSegment, segmentLength } from './bucketFillSegments.ts';
import { hsvToRgb, rgbToHueDeg } from './distortion.ts';

// ── Line-segment joining (pure) ───────────────────────────────────────────
//
// A second, separate stepped process on top of the bucket-fill segments
// (pipeline/bucketFillSegments.ts): from each segment's two endpoints, walk
// straight out along that segment's own tangent direction (fixed for the
// whole walk -- there's no more field data to re-steer by once we're past
// the segment's actual pixels, unlike the original flood fill). The
// UNDERLYING data structure is a texture of SEGMENT INDICES, not colors --
// each cell holds which segment currently claims it (-1 = unclaimed) -- so
// a front walking into a cell already claimed by a DIFFERENT segment can
// look that segment up directly and test whether the two are plausibly the
// same line (their gradient directions are close to collinear) before
// deciding to merge. A collision that FAILS the similarity test (almost
// always a crossing perpendicular segment on this grid) gets passed through
// without claiming the cell, not treated as a dead end, so a front can keep
// searching past unrelated crossings for its true collinear partner further
// along. A collision that PASSES stops the two fronts that actually met --
// they found their neighbor, this point is now internal to the composite,
// not an extremity -- but each one's SIBLING (the other front on that same
// segment, still headed outward toward the group's other, still-unexplored
// end) gets redirected instead: it keeps walking from wherever it currently
// is, now along the merged group's own composite direction (see the live
// union-find below) rather than its own original segment's estimate. So
// after a merge, exactly the two fronts still facing outward survive,
// sharing one jointly-informed direction -- this lets a chain of 3+
// segments keep discovering further collinear neighbors, correcting course
// toward the group's actual line as more evidence comes in, rather than a
// fixed pair only ever finding one match each. Colors are purely a
// rendering concern, see paintJoinOverlay.
//
// Deterministic and fully recomputed from scratch for whatever step count is
// requested (see numSteps) -- "step N" always means the same walk state, so
// moving the step slider back and forth doesn't need any incremental/stateful
// tracking, just a pure re-run.

export interface SegmentMerge { a: number; b: number }

interface JoinFront {
  seg: number;
  startX: number; startY: number;
  dx: number; dy: number; // fixed unit tangent direction for this front's whole walk
  k: number; // steps taken so far -- position is always startX + k*dx, startY + k*dy (recomputed fresh each step, not accumulated, to avoid drift)
  active: boolean;
}

export function computeJoinWalk(
  segments: BucketFillSegment[], w: number, h: number, minSimilarity: number, numSteps: number, minLengthPx: number,
): { joinBuffer: Int32Array; merges: SegmentMerge[] } {
  const n = w * h;
  const joinBuffer = new Int32Array(n).fill(-1);
  // Which FRONT (not just which segment) claimed each cell -- needed so that
  // when a merge succeeds, we can stop the SPECIFIC ray on the other side
  // too, not an arbitrary front belonging to that segment (it has two).
  const claimedByFront = new Int32Array(n).fill(-1);
  const merges: SegmentMerge[] = [];
  if (segments.length === 0) return { joinBuffer, merges };

  // minSimilarity is compared DIRECTLY against |dot(unitA, unitB)| -- both
  // gradUX/gradUY below are already normalized (divided by gLen) before the
  // dot product, so this is already a clean cosine similarity in [0,1], no
  // angle conversion needed. |dot| (not the raw signed dot) ranges 0
  // (perpendicular) to 1 (parallel OR antiparallel -- the abs() is what
  // makes this correctly axial, same reasoning as bucketFillSegments.ts's
  // own sign-resolution, just applied to a single pairwise comparison
  // instead of an online running average).
  const gradUX = new Float64Array(segments.length), gradUY = new Float64Array(segments.length);
  const fronts: JoinFront[] = [];
  // Each segment's two front INDICES (not the fronts themselves) -- needed
  // at merge time to find a colliding front's SIBLING (the other front on
  // the same segment, still headed toward the group's other, unexplored
  // end) without a linear search.
  const frontsBySegment = new Map<number, [number, number]>();
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (segmentLength(seg) < minLengthPx) continue; // too short to trust its endpoint-to-endpoint direction -- no fronts at all, can't grow OR be merged into
    const gLen = Math.hypot(seg.avgFx, seg.avgFy);
    const gux = gLen > 0 ? seg.avgFx / gLen : 1, guy = gLen > 0 ? seg.avgFy / gLen : 0;
    gradUX[i] = gux; gradUY[i] = guy;
    const tx = -guy, ty = gux; // tangent = gradient rotated 90 degrees, same convention as the marker drawing
    const alongFi = fronts.length;
    fronts.push({ seg: i, startX: seg.endAlongX, startY: seg.endAlongY, dx: tx, dy: ty, k: 0, active: true });
    const againstFi = fronts.length;
    fronts.push({ seg: i, startX: seg.endAgainstX, startY: seg.endAgainstY, dx: -tx, dy: -ty, k: 0, active: true });
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
  // still headed toward the group's other, still-unexplored extremity.
  // Continues from wherever it CURRENTLY is (not the collision point) along
  // the group's just-updated composite axis, so from here on both segments'
  // outward-facing rays share one jointly-informed direction instead of each
  // independently trusting its own original segment's estimate. Sign chosen
  // to keep it moving roughly the way it already was (dot product against
  // its OLD direction), not double back on itself.
  function redirectSibling(front: JoinFront, root: number) {
    if (!front.active) return; // already stopped (out of bounds, or already consumed by an earlier merge) -- nothing to redirect
    let ndx = compositeX2[root] - compositeX1[root], ndy = compositeY2[root] - compositeY1[root];
    const ndLen = Math.hypot(ndx, ndy);
    if (ndLen < 1e-9) return; // degenerate (shouldn't happen for any group with 2+ distinct points) -- leave direction as-is
    ndx /= ndLen; ndy /= ndLen;
    const sign = (ndx * front.dx + ndy * front.dy) >= 0 ? 1 : -1;
    front.startX = front.startX + front.k * front.dx; // front's own CURRENT position
    front.startY = front.startY + front.k * front.dy;
    front.dx = ndx * sign; front.dy = ndy * sign;
    front.k = 0;
  }

  // Claim each front's own starting pixel (its segment's own endpoint)
  // unconditionally -- this is the segment's own already-established
  // territory, not a new encounter, so no collision/merge test here. Last
  // write wins if two fronts happen to start on the same pixel; harmless.
  for (let fi = 0; fi < fronts.length; fi++) {
    const f = fronts[fi];
    const sx = Math.round(f.startX), sy = Math.round(f.startY);
    if (sx >= 0 && sx < w && sy >= 0 && sy < h) { joinBuffer[sy * w + sx] = f.seg; claimedByFront[sy * w + sx] = fi; }
  }

  for (let step = 1; step <= numSteps; step++) {
    for (let fi = 0; fi < fronts.length; fi++) {
      const f = fronts[fi];
      if (!f.active) continue;
      f.k++;
      const nx = Math.round(f.startX + f.k * f.dx), ny = Math.round(f.startY + f.k * f.dy);
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) { f.active = false; continue; }
      const idx = ny * w + nx;
      const occupant = joinBuffer[idx];
      if (occupant === -1) {
        joinBuffer[idx] = f.seg;
        claimedByFront[idx] = fi;
      } else if (occupant !== f.seg && find(f.seg) === find(occupant)) {
        // Already the same group, just reached via a different path through
        // the chain (e.g. both ends of a 3-segment chain sweeping back
        // toward each other) -- nothing new to learn, pass through exactly
        // like the own-segment case below.
      } else if (occupant !== f.seg) {
        const dot = gradUX[f.seg] * gradUX[occupant] + gradUY[f.seg] * gradUY[occupant];
        if (Math.abs(dot) >= minSimilarity) {
          merges.push({ a: f.seg, b: occupant });
          // The two fronts that just met are done -- they found their
          // neighbor, this point is now internal to the composite, not an
          // extremity. Their SIBLINGS (the other front on each segment) are
          // still headed outward toward the group's real, still-unexplored
          // ends -- those are the ones that keep walking, see
          // redirectSibling's own comment.
          const otherFi = claimedByFront[idx];
          f.active = false;
          if (otherFi >= 0) fronts[otherFi].active = false;
          const root = union(f.seg, occupant);
          const [aAlong, aAgainst] = frontsBySegment.get(f.seg)!;
          redirectSibling(fronts[fi === aAlong ? aAgainst : aAlong], root);
          if (otherFi >= 0) {
            const [bAlong, bAgainst] = frontsBySegment.get(occupant)!;
            redirectSibling(fronts[otherFi === bAlong ? bAgainst : bAlong], root);
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

  return { joinBuffer, merges };
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
