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
// deciding to merge. A front only stops on an actual merge or leaving the
// image -- a collision that FAILS the similarity test (almost always a
// crossing perpendicular segment on this grid) gets passed through without
// claiming the cell, not treated as a dead end, so a front can keep
// searching past unrelated crossings for its true collinear partner
// further along. Colors are purely a rendering concern, see
// paintJoinOverlay.
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
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (segmentLength(seg) < minLengthPx) continue; // too short to trust its endpoint-to-endpoint direction -- no fronts at all, can't grow OR be merged into
    const gLen = Math.hypot(seg.avgFx, seg.avgFy);
    const gux = gLen > 0 ? seg.avgFx / gLen : 1, guy = gLen > 0 ? seg.avgFy / gLen : 0;
    gradUX[i] = gux; gradUY[i] = guy;
    const tx = -guy, ty = gux; // tangent = gradient rotated 90 degrees, same convention as the marker drawing
    fronts.push({ seg: i, startX: seg.endAlongX, startY: seg.endAlongY, dx: tx, dy: ty, k: 0, active: true });
    fronts.push({ seg: i, startX: seg.endAgainstX, startY: seg.endAgainstY, dx: -tx, dy: -ty, k: 0, active: true });
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
      } else if (occupant !== f.seg) {
        const dot = gradUX[f.seg] * gradUX[occupant] + gradUY[f.seg] * gradUY[occupant];
        if (Math.abs(dot) >= minSimilarity) {
          merges.push({ a: f.seg, b: occupant });
          f.active = false; // found its match -- done walking
          // Stop the SPECIFIC ray that claimed this cell too, not just any
          // front belonging to `occupant` (it has two) -- the two rays met,
          // neither needs to keep searching past each other.
          const otherFi = claimedByFront[idx];
          if (otherFi >= 0) fronts[otherFi].active = false;
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
