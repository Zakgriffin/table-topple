import { BucketFillSegment, segmentLength } from './bucketFillSegments.ts';
import { hsvToRgb, rgbToHueDeg } from './distortion.ts';

// ── Line-segment joining (pure) ───────────────────────────────────────────
//
// A second, separate stepped process on top of the bucket-fill segments
// (pipeline/bucketFillSegments.ts): every merge GROUP (initially just one
// raw segment; later, a chain of several) has exactly two live "fronts",
// walking straight outward from its two known extremity points -- there's
// no more field data to re-steer by once we're past the segment's actual
// pixels, unlike the original flood fill, so a front's direction is fixed
// until it's REGENERATED (see below), not continuously re-evaluated.
//
// The UNDERLYING data structure is conceptually a texture of SEGMENT
// INDICES, not colors -- each cell holds which segment currently claims it
// (-1 = unclaimed) -- so a front walking into a cell already claimed by a
// DIFFERENT segment can look that segment up directly and test whether the
// two are plausibly the same line (their current GROUP axes are close to
// collinear) before deciding to merge. Internally this is one level of
// indirection: a cell actually stores a HANDLE, resolved to a segment index
// through a small per-episode table (episodeOwner, see below) -- this is
// what lets a front's pre-merge trail be erased in O(1) (just invalidate its
// old handle) instead of walking back over every pixel it claimed. This
// buffer is SEEDED with the flood fill's own pixel footprint (regionId, from
// bucketFillSegments.ts's computeBucketFillRegions) before any front takes a
// step -- so a front reacts the moment it enters ANOTHER segment's real blob
// shape, not just when it happens to cross that segment's own thin
// front-trace line. Only segments that pass minLengthPx (the same set that
// gets fronts at all) get seeded, so a too-short segment stays exactly as
// invisible to the join walk as it already is everywhere else.
//
// A collision that FAILS the orientation-similarity test (almost always a
// crossing perpendicular segment on this grid) gets passed through without
// claiming the cell, not treated as a dead end, so a front can keep
// searching past unrelated crossings for its true collinear partner further
// along -- this part is unchanged from before. What happens when it PASSES
// is deliberately indifferent to HOW the collision happened (another
// front's own trace line, or a bare flood-fill blob pixel -- both resolve to
// "this segment" via the exact same episodeOwner lookup, so the merge logic
// never needs to know or care which) and indifferent to any front's current
// travel DIRECTION: the two segments' current GROUPS each have exactly two
// known extremity points (their live composite pair) -- 4 candidate points
// total -- and the two of those 4 that end up FARTHEST APART become the
// merged group's new composite. Two fresh fronts spawn there, walking
// straight away from each other along the line between them; whichever of
// the other two candidate points didn't win just has its (if any) existing
// front stopped, no replacement -- if both winners happen to come from the
// SAME original group, the other group's whole contribution is discarded
// (its shorter reach didn't extend the group's known extent at all). This
// keeps "exactly two active fronts per group" true by construction, with no
// separate case analysis for front-vs-front vs front-vs-blob collisions, and
// no need to reason about any front's travel direction to decide who
// survives -- see mergeAt below for the actual mechanics, and
// spawnPair/the handle-indirection comments for how "erase the old trail,
// start fresh at the winning point" is implemented.
//
// Deterministic and fully recomputed from scratch for whatever step count is
// requested (see numSteps) -- "step N" always means the same walk state, so
// moving the step slider back and forth doesn't need any incremental/stateful
// tracking, just a pure re-run.

export interface SegmentMerge { a: number; b: number }

interface JoinFront {
  seg: number; // a raw segment index whose find() resolves to this front's CURRENT group root -- not necessarily the segment this front's own point originally came from once it's part of a merged group, see mergeAt
  startX: number; startY: number;
  dx: number; dy: number; // fixed unit direction for this front's whole walk, set once at spawn time from the two points it was spawned between (see spawnPair) -- never re-steered except by a fresh spawn
  k: number; // steps taken so far -- position is always startX + k*dx, startY + k*dy (recomputed fresh each step, not accumulated, to avoid drift)
  maxK: number; // this front stops once k reaches this -- the length of the pair it was spawned from (see spawnPair), so a front never walks farther than the evidence that produced it
  active: boolean;
  handle: number; // this front's CURRENT episode handle -- every cell it claims while walking this episode is tagged with this number, see computeJoinWalk's own comment
}

export function computeJoinWalk(
  segments: BucketFillSegment[], regionId: Int32Array, w: number, h: number, minSimilarity: number, numSteps: number, minLengthPx: number,
): {
  joinBuffer: Int32Array; merges: SegmentMerge[];
  blueMergePoints: { x: number; y: number }[]; orangeMergePoints: { x: number; y: number }[]; redMergePoints: { x: number; y: number }[];
} {
  const n = w * h;
  const merges: SegmentMerge[] = [];
  // Cosmetic classification of each real merge (see mergeAt) -- doesn't feed
  // back into the walk itself. Blue/orange: the winning pair took one point
  // from EACH of the two groups, and those two points' own fronts happened
  // to be walking roughly opposite (blue, "closing a gap") or roughly the
  // same (orange) direction. Red: the winning pair took BOTH points from the
  // SAME group -- the other group's own best point lost outright and its
  // whole contribution was discarded.
  const blueMergePoints: { x: number; y: number }[] = [];
  const orangeMergePoints: { x: number; y: number }[] = [];
  const redMergePoints: { x: number; y: number }[] = [];
  if (segments.length === 0) {
    return { joinBuffer: new Int32Array(n).fill(-1), merges, blueMergePoints, orangeMergePoints, redMergePoints };
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
  // handle for each "episode" of its walk (from its most recent spawn until
  // it's replaced by a future merge) -- erasing a front's pre-merge trail is
  // then just `episodeOwner[oldHandle] = -1`, an O(1) invalidation instead of
  // walking back over every pixel it claimed: every cell still holding that
  // handle silently resolves to -1 (unclaimed) the next time anything reads
  // it. There's no need for a per-pixel "which front claimed this" table
  // (unlike an earlier version of this function) -- merge decisions are made
  // entirely at the GROUP level (see frontAtSlot1/2 below), never by asking
  // "which specific front is sitting on this exact pixel".
  const cellHandle = new Int32Array(n).fill(-1);
  const episodeOwner: number[] = [];
  for (let i = 0; i < segments.length; i++) episodeOwner.push(i);
  function allocateHandle(seg: number): number {
    const h = episodeOwner.length;
    episodeOwner.push(seg);
    return h;
  }

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

  // Live union-find with satellite data. Each root's own
  // compositeX1/Y1/X2/Y2 entry holds that group's current two known
  // extremity points, and frontAtSlot1/2 holds which FRONT (index into
  // `fronts`, or -1) currently represents each of those two points --
  // together these fully describe "this group's current live state".
  // Always read/write through find(root), never a cached index, since path
  // compression can change which index IS the root as more unions happen.
  const parent = new Int32Array(segments.length);
  for (let i = 0; i < segments.length; i++) parent[i] = i;
  function find(x: number): number {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  }
  const compositeX1 = new Float64Array(segments.length), compositeY1 = new Float64Array(segments.length);
  const compositeX2 = new Float64Array(segments.length), compositeY2 = new Float64Array(segments.length);
  const frontAtSlot1 = new Int32Array(segments.length).fill(-1);
  const frontAtSlot2 = new Int32Array(segments.length).fill(-1);

  const fronts: JoinFront[] = [];

  // Spawns a fresh pair of fronts representing `root`'s two current
  // extremity points -- one at each point, walking directly away from the
  // other along the line between them. Direction comes purely from this
  // pair's own geometry (not a gradient average) -- exactly as valid for a
  // lone segment's own two endpoints (used by the initial per-segment setup
  // below) as it is for a freshly merged group's winning pair (used by
  // mergeAt). This is the ONLY place compositeX1/Y1/X2/Y2 and
  // frontAtSlot1/2 ever get written, so a group's tracked composite always
  // exactly matches wherever its two live fronts actually started from.
  // Each front's maxK is capped to this SAME pair's own length -- a front
  // never walks farther out than the length of the evidence (single segment,
  // or already-merged composite) that just produced it, so a short segment
  // only ever reaches a proportionally short distance looking for its
  // partner, not an arbitrary fixed number of steps.
  function spawnPair(root: number, p1x: number, p1y: number, p2x: number, p2y: number) {
    const ddx = p1x - p2x, ddy = p1y - p2y;
    const len = Math.hypot(ddx, ddy);
    const ux = len > 1e-9 ? ddx / len : 1, uy = len > 1e-9 ? ddy / len : 0;
    const maxK = Math.round(len);
    const fi1 = fronts.length;
    fronts.push({ seg: root, startX: p1x, startY: p1y, dx: ux, dy: uy, k: 0, maxK, active: true, handle: allocateHandle(root) });
    const fi2 = fronts.length;
    fronts.push({ seg: root, startX: p2x, startY: p2y, dx: -ux, dy: -uy, k: 0, maxK, active: true, handle: allocateHandle(root) });
    compositeX1[root] = p1x; compositeY1[root] = p1y; frontAtSlot1[root] = fi1;
    compositeX2[root] = p2x; compositeY2[root] = p2y; frontAtSlot2[root] = fi2;
  }

  for (let i = 0; i < segments.length; i++) {
    if (!eligible[i]) continue; // too short to trust its endpoint-to-endpoint direction -- no fronts at all, can't grow OR be merged into
    spawnPair(i, segments[i].endAlongX, segments[i].endAlongY, segments[i].endAgainstX, segments[i].endAgainstY);
  }

  // The GROUP's current axis direction (via find(seg)'s root) -- used ONLY
  // to gate whether two groups are plausibly the same line at all before a
  // merge is even considered; the actual choice of which points survive a
  // real merge (mergeAt) is purely geometric and doesn't use this. |dot|
  // against minSimilarity, same axial (sign-insensitive) reasoning as
  // bucketFillSegments.ts's own gradient averaging -- a line's direction is
  // undirected.
  function groupAxis(seg: number): [number, number] {
    const root = find(seg);
    const dx0 = compositeX2[root] - compositeX1[root], dy0 = compositeY2[root] - compositeY1[root];
    const len = Math.hypot(dx0, dy0);
    if (len < 1e-9) return [0, 0]; // degenerate (shouldn't happen once past the length filter) -- dot comes out 0, correctly never matches
    return [dx0 / len, dy0 / len];
  }

  // The core merge operation, called once the orientation-similarity gate
  // above has already passed for segA (the segment whose front just
  // stepped) and segB (whatever it ran into -- via another front's trace OR
  // a bare blob pixel, doesn't matter, both resolve to a segment index the
  // same way, see the collision loop below). Gathers the two groups'
  // current 4 candidate points (2 each) and finds both the FARTHEST cross
  // pair (one point from each group -- the two same-group pairs are never
  // considered, so the "red" classification below can never fire, left in
  // place as dead code rather than deleted) and the NEAREST cross pair.
  //
  // Orientation similarity alone can't tell "these are the same line,
  // extended" apart from "these are two different but merely PARALLEL
  // lines" (e.g. two adjacent rows of the De Bruijn grid have identical
  // direction vectors despite being unrelated) -- so before committing to
  // anything, this also checks that the NEAREST pair's own connecting
  // vector (the two points actually facing each other across whatever gap
  // just got bridged) aligns with BOTH groups' own axis directions. A real
  // collinear merge has that connecting vector running right along the
  // shared line; a lateral offset between two merely-parallel lines bends it
  // away from one or both axes even though the axes themselves still agree.
  // Reuses minSimilarity rather than a separate tunable -- same shape of
  // quantity (a cosine-similarity threshold on a unit-vector dot product),
  // just applied to a second vector pair.
  //
  // Returns false (no merge, no state mutated) if this check fails --
  // treated exactly like a failed orientation test by the caller.
  function mergeAt(segA: number, segB: number, cx: number, cy: number): boolean {
    const ra = find(segA), rb = find(segB);
    const candidates = [
      { x: compositeX1[ra], y: compositeY1[ra], root: ra, fi: frontAtSlot1[ra] },
      { x: compositeX2[ra], y: compositeY2[ra], root: ra, fi: frontAtSlot2[ra] },
      { x: compositeX1[rb], y: compositeY1[rb], root: rb, fi: frontAtSlot1[rb] },
      { x: compositeX2[rb], y: compositeY2[rb], root: rb, fi: frontAtSlot2[rb] },
    ];
    const crossPairs: [number, number][] = [[0, 2], [0, 3], [1, 2], [1, 3]];
    let bestP = 0, bestQ = 2, bestDistSq = -1;
    let nearP = 0, nearQ = 2, nearDistSq = Infinity;
    for (const [p, q] of crossPairs) {
      const dx = candidates[p].x - candidates[q].x, dy = candidates[p].y - candidates[q].y;
      const distSq = dx * dx + dy * dy;
      if (distSq > bestDistSq) { bestDistSq = distSq; bestP = p; bestQ = q; }
      if (distSq < nearDistSq) { nearDistSq = distSq; nearP = p; nearQ = q; }
    }
    const winA = candidates[bestP], winB = candidates[bestQ];
    const nearA = candidates[nearP], nearB = candidates[nearQ];

    if (nearDistSq > 1e-9) {
      const cvx = (nearB.x - nearA.x) / Math.sqrt(nearDistSq), cvy = (nearB.y - nearA.y) / Math.sqrt(nearDistSq);
      const [gaX, gaY] = groupAxis(segA), [gbX, gbY] = groupAxis(segB);
      const cvDotA = Math.abs(cvx * gaX + cvy * gaY), cvDotB = Math.abs(cvx * gbX + cvy * gbY);
      if (cvDotA < minSimilarity || cvDotB < minSimilarity) return false; // parallel but laterally offset -- not actually the same line
    }
    // nearDistSq ~ 0: the near points already coincide, nothing to check --
    // trivially collinear (can't be laterally offset from itself).

    if (winA.root === winB.root) {
      redMergePoints.push({ x: cx, y: cy });
    } else if (winA.fi >= 0 && winB.fi >= 0) {
      const fa = fronts[winA.fi], fb = fronts[winB.fi];
      const fdot = fa.dx * fb.dx + fa.dy * fb.dy;
      (fdot < 0 ? blueMergePoints : orangeMergePoints).push({ x: cx, y: cy });
    }

    // Every one of the (up to 4) fronts currently at these candidate points
    // is being replaced -- the 2 winners get a fresh spawn at the exact same
    // point (new direction, new handle -- see spawnPair), the 2 losers just
    // stop with no replacement. Only the WINNERS' pre-merge trails get
    // erased (they're the ones starting a fresh episode); a loser's trail is
    // genuine, already-walked territory and stays as-is.
    for (const cand of candidates) {
      if (cand.fi < 0) continue;
      const isWinner = cand === winA || cand === winB;
      if (isWinner) episodeOwner[fronts[cand.fi].handle] = -1;
      fronts[cand.fi].active = false;
    }

    parent[ra] = rb; // standard union-by-arbitrary-root, matches computeMergeGroups' own convention
    spawnPair(rb, winA.x, winA.y, winB.x, winB.y);
    return true;
  }

  for (let step = 1; step <= numSteps; step++) {
    // Snapshot the front count at the START of this step -- a merge can
    // spawn brand-new fronts mid-step (see mergeAt/spawnPair), but they
    // shouldn't get a bonus step within the same step count they were born
    // in; they wait for the next step, same as every other front.
    const frontCount = fronts.length;
    for (let fi = 0; fi < frontCount; fi++) {
      const f = fronts[fi];
      if (!f.active) continue;
      f.k++;
      if (f.k > f.maxK) { f.active = false; continue; } // reached the length of the evidence that spawned it -- see spawnPair
      const nx = Math.round(f.startX + f.k * f.dx), ny = Math.round(f.startY + f.k * f.dy);
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) { f.active = false; continue; }
      const idx = ny * w + nx;
      const rawOccupant = cellHandle[idx];
      // -1 here covers BOTH a cell that's never been touched AND one whose
      // handle was invalidated by an earlier merge (see the handle-
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
        const [gux, guy] = groupAxis(f.seg);
        const [oux, ouy] = groupAxis(occupant);
        const dot = gux * oux + guy * ouy;
        if (Math.abs(dot) >= minSimilarity && mergeAt(f.seg, occupant, nx, ny)) {
          merges.push({ a: f.seg, b: occupant });
        }
        // else: NOT a match (e.g. a crossing perpendicular segment, or a
        // merely-parallel-but-offset line rejected by mergeAt's own
        // connecting-vector check) -- pass through without claiming the
        // cell (leave the occupant's own claim alone) and keep walking.
        // Stopping unconditionally here was the original bug this whole
        // design fixed: on this grid, a front headed
        // toward its true collinear partner runs into unrelated crossing
        // segments constantly, and stopping at the FIRST one (regardless of
        // match) meant most fronts never got anywhere near far enough to
        // find a real match.
      }
      // occupant === f.seg: this front's own group's territory -- pass
      // through as a no-op.
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

  return { joinBuffer, merges, blueMergePoints, orangeMergePoints, redMergePoints };
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
