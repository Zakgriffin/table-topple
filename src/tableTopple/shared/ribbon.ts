import * as THREE from 'three';

// A polyline with real width, lying flat in the ground plane -- a ribbon of
// triangles, not a THREE.Line. Line's `linewidth` is a dead end (the WebGL
// renderer ignores it on every platform, and even where it worked the width
// would be in SCREEN pixels: the outline would change apparent thickness as
// you zoom, which is exactly wrong for something meant to be drawn ON the
// board). Building the strip ourselves gives a width in world units, so the
// outline is a stripe of paint on the floor that zooms with everything else.
//
// The strip is two vertices per path point -- one offset to each side along
// the point's miter normal -- stitched into quads.

/** Cap on how sharp a corner can be before its miter is truncated. A miter
 *  length grows as 1/cos(theta/2), so a hairpin turn would otherwise fire a
 *  spike off to infinity. Past this the corner is left slightly blunt, which
 *  nobody notices on a hand-drawn shape. */
const MITER_LIMIT = 3;

export interface Ribbon {
  mesh: THREE.Mesh;
  /** Rebuild the strip along `pts`. `closed` adds the segment from the last
   *  point back to the first, mitered as a real joint. */
  update(pts: THREE.Vector2[], closed: boolean): void;
  hide(): void;
}

/**
 * @param maxPoints buffer capacity; update() ignores points past it.
 * @param y height above the ground plane to float the ribbon at.
 * @param opacity defaults to path mode's own 0.9 -- roads.ts passes something
 *   lower for a blueprint that reads as a translucent preview rather than
 *   paint already committed to the floor.
 */
export function createRibbon(maxPoints: number, color: number, y: number, halfWidth: number, opacity = 0.9): Ribbon {
  // One vertex pair per point, plus one spare pair: closing the loop writes a
  // duplicate of the first pair at the end, which lets the closing quad use
  // the same index pattern as every other quad (see update()).
  const capacity = maxPoints + 1;
  const positions = new Float32Array(capacity * 2 * 3);
  const indices = new Uint32Array(maxPoints * 6);

  // The index pattern never changes -- quad s spans vertex pairs s and s+1 --
  // so it's written once here and only the draw range moves afterwards.
  for (let s = 0; s < maxPoints; s++) {
    const a = s * 2, o = s * 6;
    indices[o] = a; indices[o + 1] = a + 1; indices[o + 2] = a + 3;
    indices[o + 3] = a; indices[o + 4] = a + 3; indices[o + 5] = a + 2;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.setDrawRange(0, 0);

  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color,
    // DoubleSide is not optional here: the strip's winding flips with the
    // direction the shape was drawn in, so a clockwise stroke and a
    // counterclockwise one face opposite ways. With backface culling, half of
    // all drawings would be invisible from above.
    side: THREE.DoubleSide,
    transparent: true,
    opacity,
    // Basic, not Standard: this is a graphic overlay, not a lit surface. It
    // should read the same regardless of where the sun is.
  }));
  mesh.visible = false;
  // Nothing else is drawn at this height and it must not occlude the denizens,
  // so it takes no part in the depth buffer.
  mesh.material.depthWrite = false;
  mesh.renderOrder = 1;

  // Scratch, reused across calls -- update() runs per pointermove.
  const prevDir = new THREE.Vector2();
  const nextDir = new THREE.Vector2();
  const tangent = new THREE.Vector2();
  const normal = new THREE.Vector2();
  const prevNormal = new THREE.Vector2();

  function dirTo(from: THREE.Vector2, to: THREE.Vector2, out: THREE.Vector2) {
    out.subVectors(to, from);
    const len = out.length();
    if (len > 1e-9) out.divideScalar(len);
    return len > 1e-9;
  }

  // Left normal of a 2D direction. (x, y) here is really (x, z) in world space.
  const perp = (d: THREE.Vector2, out: THREE.Vector2) => out.set(-d.y, d.x);

  function update(pts: THREE.Vector2[], closed: boolean) {
    const n = Math.min(pts.length, maxPoints);
    // One point is a dot with no direction; nothing to extrude.
    if (n < 2) { mesh.visible = false; return; }

    for (let i = 0; i < n; i++) {
      const p = pts[i];

      // The two segments meeting at this point. At an open path's ends one of
      // them doesn't exist; on a closed path both always do, by wrapping.
      const hasPrev = i > 0 ? dirTo(pts[i - 1], p, prevDir)
        : closed ? dirTo(pts[n - 1], p, prevDir) : false;
      const hasNext = i < n - 1 ? dirTo(p, pts[i + 1], nextDir)
        : closed ? dirTo(p, pts[0], nextDir) : false;

      let miter = halfWidth;
      if (hasPrev && hasNext) {
        // Interior joint: offset along the angle bisector, lengthened by
        // 1/cos(theta/2) so the ribbon's OUTER edges meet exactly at the
        // corner. Offsetting by plain halfWidth instead would pinch the
        // outside of every bend.
        tangent.addVectors(prevDir, nextDir);
        if (tangent.lengthSq() < 1e-12) {
          // A 180-degree doubling-back: the bisector is undefined. Any
          // perpendicular is as good as another here.
          perp(prevDir, normal);
        } else {
          perp(tangent.normalize(), normal);
          perp(prevDir, prevNormal);
          // cos(theta/2), via the angle between the joint normal and the
          // incoming segment's own normal.
          const cos = Math.abs(normal.dot(prevNormal));
          miter = halfWidth / Math.max(cos, 1 / MITER_LIMIT);
        }
      } else if (hasNext) {
        perp(nextDir, normal);
      } else if (hasPrev) {
        perp(prevDir, normal);
      } else {
        // Coincident with its neighbours -- no direction to work from.
        perp(prevDir.set(1, 0), normal);
      }

      const ox = normal.x * miter, oz = normal.y * miter;
      const v = i * 6;
      positions[v] = p.x + ox; positions[v + 1] = y; positions[v + 2] = p.y + oz;
      positions[v + 3] = p.x - ox; positions[v + 4] = y; positions[v + 5] = p.y - oz;
    }

    if (closed) {
      // Duplicate the first pair one slot past the end. The closing quad then
      // spans pairs n-1 and n using the ordinary index pattern -- and because
      // pair 0 was mitered WITH the wraparound above, the seam is a proper
      // joint rather than two butted ends.
      const first = 0, dup = n * 6;
      for (let k = 0; k < 6; k++) positions[dup + k] = positions[first + k];
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.setDrawRange(0, (closed ? n : n - 1) * 6);
    // The strip only ever grows toward the camera-facing plane; skipping the
    // bounding-sphere recompute would make it vanish under frustum culling
    // once the shape extends past the sphere computed for its first points.
    geometry.computeBoundingSphere();
    mesh.visible = true;
  }

  function hide() {
    mesh.visible = false;
    geometry.setDrawRange(0, 0);
  }

  return { mesh, update, hide };
}
