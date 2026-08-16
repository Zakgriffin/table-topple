import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

// Shared kit for building scenery out of boxes -- the idiom character.ts and
// castle.ts established, factored out once landmarks.ts and forest.ts both
// wanted it.
//
// Pure: no scene.ts, so anything built on top of this stays runnable under
// node for checking.

/**
 * Accumulates boxes and merges them into ONE geometry at the end.
 *
 * Scenery is static and never moves relative to itself, so there's nothing to
 * gain from keeping its parts as separate meshes and a lot to lose: fifteen
 * structures of twenty-odd boxes would be ~300 draw calls. Merging is also how
 * per-block colour variation is paid for -- one merged geometry means one
 * material, so each block's shade is baked into vertex colours instead.
 */
export class Blocks {
  private parts: THREE.BufferGeometry[] = [];

  /**
   * @param yaw turn about the vertical, applied before the translate.
   * @param roll tilt about Z, for anything leaning. Applied before yaw, so it
   *   reads as "tip it over, then turn it" rather than the other way round.
   */
  add(
    w: number, h: number, d: number,
    x: number, y: number, z: number,
    color: number, yaw = 0, roll = 0,
  ) {
    const g = new THREE.BoxGeometry(w, h, d);
    if (roll) g.rotateZ(roll);
    if (yaw) g.rotateY(yaw);
    g.translate(x, y, z);

    const c = new THREE.Color(color);
    const count = g.getAttribute('position').count;
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    this.parts.push(g);
  }

  /** A box standing ON the ground: y is its base, not its center. Most of a
   *  structure is built this way, and halving heights at every call site is
   *  where sunken and floating blocks come from. */
  standing(w: number, h: number, d: number, x: number, z: number, color: number, yaw = 0) {
    this.add(w, h, d, x, h / 2, z, color, yaw);
  }

  build(): THREE.BufferGeometry {
    const merged = mergeGeometries(this.parts);
    if (!merged) throw new Error('blocks: geometries failed to merge (mismatched attributes)');
    // Needed for fitting scenery to its patch, and so frustum culling uses the
    // real extent rather than whatever the first box happened to be.
    merged.computeBoundingBox();
    return merged;
  }
}

/** One material for all scenery. Vertex colours carry the actual shades; flat
 *  shading because this is faceted rock, stone and foliage, and smooth normals
 *  on a box just make its corners look soft. */
export const sceneryMaterial = new THREE.MeshStandardMaterial({
  roughness: 0.94, metalness: 0, vertexColors: true, flatShading: true,
});

export const pick = (rng: () => number, palette: number[]) => palette[Math.floor(rng() * palette.length)];

/** Uniform in [lo, hi). */
export const between = (rng: () => number, lo: number, hi: number) => lo + rng() * (hi - lo);
