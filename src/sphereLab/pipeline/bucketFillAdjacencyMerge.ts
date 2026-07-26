import { BucketFillSegment } from './bucketFillSegments.ts';
import { SegmentMerge } from './bucketFillJoin.ts';

// ── Adjacency-based merge detection (pure) ────────────────────────────────
//
// A lighter-weight companion to bucketFillJoin.ts's own front-walking merge
// search: that phase exists to bridge a genuine GAP between two segments (a
// dashed line, a low-magnitude break) by simulating two fronts walking
// outward looking for a partner. This one instead catches the ZERO-GAP
// case -- two segments that are directly touching because flood fill (BFS or
// label propagation) split one continuous line into fragments purely by
// which region's growth claimed a given pixel first, not because there's
// any real break between them. No simulated walking needed: since the two
// fragments already touch, the same "is this actually the same line, not
// just a parallel one" test bucketFillJoin.ts's mergeAt applies to two live
// fronts colliding can be applied directly, once, to the two segments'
// already-final endpoints.
//
// Candidate pairs come from a single O(n) scan over the finished regionId
// buffer (which segment borders which), NOT an O(k^2) all-pairs scan over
// every segment -- segments that never actually touch have no adjacency
// evidence they're the same line at all (that's exactly what the gap-
// bridging join walk is for instead), so they're never even tested here.

// Each segment's own TANGENT axis, derived from its two tracked endpoints --
// same convention as bucketFillJoin.ts's groupAxis, and deliberately NOT
// avgFx/avgFy (the GRADIENT direction, perpendicular to the line): the
// connecting vector between two touching fragments runs ALONG the line, so
// it needs to be compared against the line's own tangent, not the axis
// perpendicular to it. Degenerate (single-pixel, zero-length) segments have
// no defined axis and always fail the dot-product test below, same as
// groupAxis's own [0,0] fallback for a degenerate composite.
function segmentAxis(seg: BucketFillSegment): [number, number] {
  const dx = seg.endAlongX - seg.endAgainstX, dy = seg.endAlongY - seg.endAgainstY;
  const len = Math.hypot(dx, dy);
  return len > 1e-9 ? [dx / len, dy / len] : [0, 0];
}

// Same "nearest cross pair's connecting vector must align with BOTH
// segments' own axes" check bucketFillJoin.ts's mergeAt uses to reject
// merely-parallel-but-offset lines -- see that function's own comment for
// the full reasoning. Here the 4 candidates are just each segment's own two
// already-tracked endpoints (no live fronts/handles involved, since both
// segments are already finished).
function segmentsCollinear(a: BucketFillSegment, b: BucketFillSegment, minSimilarity: number): boolean {
  const [axAx, axAy] = segmentAxis(a);
  const [axBx, axBy] = segmentAxis(b);
  // Axial (sign-insensitive) direction agreement, same convention as every
  // other direction comparison in this pipeline.
  if (Math.abs(axAx * axBx + axAy * axBy) < minSimilarity) return false;

  const candA: [number, number][] = [[a.endAlongX, a.endAlongY], [a.endAgainstX, a.endAgainstY]];
  const candB: [number, number][] = [[b.endAlongX, b.endAlongY], [b.endAgainstX, b.endAgainstY]];
  let nearDistSq = Infinity, nearAx = 0, nearAy = 0, nearBx = 0, nearBy = 0;
  for (const [ax, ay] of candA) {
    for (const [bx, by] of candB) {
      const dx = bx - ax, dy = by - ay;
      const distSq = dx * dx + dy * dy;
      if (distSq < nearDistSq) { nearDistSq = distSq; nearAx = ax; nearAy = ay; nearBx = bx; nearBy = by; }
    }
  }
  if (nearDistSq < 1e-9) return true; // near points already coincide -- trivially collinear
  const cvx = (nearBx - nearAx) / Math.sqrt(nearDistSq), cvy = (nearBy - nearAy) / Math.sqrt(nearDistSq);
  const cvDotA = Math.abs(cvx * axAx + cvy * axAy), cvDotB = Math.abs(cvx * axBx + cvy * axBy);
  return cvDotA >= minSimilarity && cvDotB >= minSimilarity;
}

export function computeAdjacencyMerges(
  regionId: Int32Array, segments: readonly BucketFillSegment[], w: number, h: number, minSimilarity: number,
): SegmentMerge[] {
  const seen = new Set<number>();
  const merges: SegmentMerge[] = [];
  const numSegments = segments.length;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = regionId[y * w + x];
      if (a < 0) continue;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const b = regionId[ny * w + nx];
          if (b < 0 || b === a) continue;
          // Dedupe so a long shared border between two regions only gets
          // ONE collinearity test, not one per touching pixel pair.
          const lo = Math.min(a, b), hi = Math.max(a, b);
          const key = lo * numSegments + hi;
          if (seen.has(key)) continue;
          seen.add(key);
          if (segmentsCollinear(segments[a], segments[b], minSimilarity)) merges.push({ a, b });
        }
      }
    }
  }
  return merges;
}
