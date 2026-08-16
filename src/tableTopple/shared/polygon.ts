import * as THREE from 'three';

// Pure polygon math: no THREE scene graph, no DOM, so it can be exercised
// straight from node (see scripts/check-centroid.ts) rather than only by
// drawing shapes with a mouse and squinting at the result.

/**
 * Area centroid of the closed polygon traced by `pts` (the standard shoelace
 * centroid).
 *
 * Deliberately NOT the mean of the vertices: mouse points are sampled per
 * pointer event, so vertex density follows how fast the cursor was moving. A
 * circle drawn with a slow start and a quick finish would drag the vertex mean
 * toward the slow arc, while the center of the SHAPE is what the player is
 * pointing at. The area centroid depends only on the outline's geometry, so it
 * lands in the same place no matter how the shape was drawn.
 *
 * Works for any simple polygon, convex or not, in either winding direction
 * (the signed area cancels the sign out of the result).
 *
 * Returns null when the path encloses essentially no area -- a straight
 * scribble, a stray flick, a doubled-back line. Shoelace divides by that area,
 * so the caller needs a fallback here rather than a NaN destination.
 */
export function polygonCentroid(pts: THREE.Vector2[], out: THREE.Vector2): THREE.Vector2 | null {
  if (pts.length < 3) return null;
  let twiceArea = 0, cx = 0, cy = 0;
  // j trails i by one, starting wrapped around to the last point -- that pairs
  // every edge including the closing one (last -> first) without a special case.
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const a = pts[j], b = pts[i];
    const cross = a.x * b.y - b.x * a.y;
    twiceArea += cross;
    cx += (a.x + b.x) * cross;
    cy += (a.y + b.y) * cross;
  }
  if (Math.abs(twiceArea) < 1e-6) return null;
  // twiceArea is 2A and the sums carry another factor of 3: C = sum / (6A).
  return out.set(cx / (3 * twiceArea), cy / (3 * twiceArea));
}
