import * as THREE from 'three';
import { Blocks, between, pick, sceneryMaterial } from './blocks.ts';
import { rightFromYaw, yawFromDirection } from './frame.ts';
import { makeRng } from './noise.ts';
import { SOLDIER_HEIGHT } from './constants.ts';

// A confirmed road's permanent geometry: a mottled dirt bed with a row of
// individually-toned cobblestones along each edge, built from the same box
// kit castle.ts/landmarks.ts use. Pure -- no scene.ts, same reasoning as
// those two: this builds and hands back a Mesh, and roads.ts is what puts it
// in the world.
//
// A road is always a straight segment (roads.ts's own header explains why),
// so unlike castle.ts's multi-wall plans this only ever builds one run.
//
// ── DETERMINISTIC, NOT RANDOM ──
//
// Every colour/size/position jitter below is seeded from the road's own id
// (makeRng, noise.ts -- the same "reproducible randomness" tool board.ts's
// terrain mottling uses), never Math.random(). This is what lets a confirmed
// road look IDENTICAL whether it was just built on the desktop or just
// arrived over the network on a phone: writeRoadSnapshot (roads.ts) calls
// this with the SAME id, independently, on a different machine, and has to
// get the SAME mesh out of it or the two hosts would visibly disagree about
// what the road looks like.

/** ~4.8x a soldier's own half-width (character.ts's CHARACTER_HALF_WIDTH is
 *  0.25) -- wide enough to read as a proper thoroughfare several figures
 *  could walk abreast on, not a garden path. Every interaction radius in
 *  roads.ts derives from this one number, per its own header, so widening a
 *  road here automatically loosens its snap zones to match. */
export const ROAD_HALF_WIDTH = 1.2 * SOLDIER_HEIGHT;

const BED_HEIGHT = 0.08 * SOLDIER_HEIGHT;
/** Roughly square bed tiles -- long enough to read as a few strides of dirt,
 *  short enough that even a modest road gets several, for real colour
 *  variation rather than one tile lightly tinted. */
const BED_SEGMENT_LENGTH = ROAD_HALF_WIDTH * 2.2;
const DIRT_PALETTE = [0x6b5842, 0x7a6449, 0x5f4d3a, 0x71593f, 0x63513c];

const COBBLE_HEIGHT = BED_HEIGHT * 1.6;
const COBBLE_SIZE = ROAD_HALF_WIDTH * 0.34;
/** Target spacing between cobble centers -- actual spacing is fitted to each
 *  edge's length, same reasoning as castle.ts's crenellate(): the end stones
 *  land flush with the road's own ends instead of the row stopping short. */
const COBBLE_PITCH = ROAD_HALF_WIDTH * 0.6;
const COBBLE_PALETTE = [0x8a887f, 0x76746c, 0x949186, 0x817e74, 0x6f6c62];
/** How far a cobble row sits off the road's own centerline -- shared by both
 *  buildRoadMesh and buildRoadGhostMesh so a preview's curb lines up exactly
 *  with the confirmed road's own once it's built. */
const COBBLE_EDGE_OFFSET = ROAD_HALF_WIDTH - COBBLE_SIZE / 2;

/** How far a junction plaza reaches past a road's own half-width -- wide
 *  enough to swallow every connecting road's cobble row, not just its bed. */
const JUNCTION_HALF = ROAD_HALF_WIDTH * 1.3;
/** Taller than a cobble, not just the bed -- see buildJunctionPatch's own
 *  comment on why a raised plaza, not trimmed/mitred road ends, is what
 *  cleans up a junction: anything shorter is simply enclosed underneath it,
 *  no coplanar overlap and so no z-fighting. */
const JUNCTION_HEIGHT = COBBLE_HEIGHT * 1.15;
const JUNCTION_PALETTE = [0x807d73, 0x716e64];

const right = new THREE.Vector2();

/**
 * Builds one road's mesh from its two endpoints and its own id (the
 * determinism seed -- see this file's header). `start`/`end` are
 * ground-plane (x, z) points, same convention as everywhere else in this
 * project (frame.ts's Vector2-is-(x,z) rule).
 */
export function buildRoadMesh(start: THREE.Vector2, end: THREE.Vector2, id: number): THREE.Mesh {
  const dx = end.x - start.x, dz = end.y - start.y;
  const length = Math.max(Math.hypot(dx, dz), 1e-3);
  // yawFromDirection/rightFromYaw (frame.ts) rather than re-deriving this
  // inline -- see that file's own header on why nothing else should.
  const yaw = yawFromDirection(dx, dz);
  rightFromYaw(yaw, right);
  const midX = (start.x + end.x) / 2, midZ = (start.y + end.y) / 2;
  // `* 2 + 1`: keeps a road's seed from ever landing on the exact bit
  // pattern of its own id, which buildJunctionPatch's unrelated seeding
  // scheme could otherwise coincide with for a low id by pure chance.
  const rng = makeRng(id * 2 + 1);

  const b = new Blocks();

  // ── The dirt bed, tiled rather than one flat colour ──────────────────
  const segCount = Math.max(1, Math.round(length / BED_SEGMENT_LENGTH));
  const segLength = length / segCount;
  for (let i = 0; i < segCount; i++) {
    const t = (i + 0.5) * segLength - length / 2;
    const cx = midX + Math.sin(yaw) * t, cz = midZ + Math.cos(yaw) * t;
    // A touch of height jitter too -- a dirt road is trodden, not poured.
    // *1.02 on the length: a hairline overlap between neighbouring tiles so
    // floating-point rounding can never leave a visible gap at the seam.
    b.standing(ROAD_HALF_WIDTH * 2, between(rng, 0.8, 1.15) * BED_HEIGHT, segLength * 1.02, cx, cz, pick(rng, DIRT_PALETTE), yaw);
  }

  // ── Cobblestones along each edge, in place of one solid curb ──────────
  const cobbleCount = Math.max(2, Math.round(length / COBBLE_PITCH));
  const usable = length - COBBLE_SIZE;
  const step = cobbleCount > 1 ? usable / (cobbleCount - 1) : 0;
  for (const side of [-1, 1]) {
    for (let i = 0; i < cobbleCount; i++) {
      const t = -usable / 2 + i * step + between(rng, -1, 1) * COBBLE_PITCH * 0.12;
      const cx = midX + Math.sin(yaw) * t + right.x * COBBLE_EDGE_OFFSET * side;
      const cz = midZ + Math.cos(yaw) * t + right.y * COBBLE_EDGE_OFFSET * side;
      const size = COBBLE_SIZE * between(rng, 0.75, 1.15);
      b.standing(size, COBBLE_HEIGHT * between(rng, 0.8, 1.2), size, cx, cz, pick(rng, COBBLE_PALETTE), between(rng, -0.4, 0.4));
    }
  }

  return new THREE.Mesh(b.build(), sceneryMaterial);
}

// ── Ghost preview: a live-drag-safe approximation of the same look ────────
//
// buildRoadMesh (above) distributes its segments and cobblestones EVENLY
// across the whole final length, all drawn from one running rng stream --
// fine for something built exactly once, at a length that will never change
// again. A drag's own preview rebuilds every pointermove with a length
// that's ALWAYS changing, and evenly redistributing everything across that
// length (or worse, letting a changed segment/cobble COUNT shift every later
// draw's place in the shared rng stream) would make the whole preview
// visibly reshuffle on every mouse move instead of growing smoothly the way
// a real drag should feel.
//
// This builds the same look from FIXED intervals measured from `start`
// instead, each with its own INDEX-seeded rng (seedAt below) rather than a
// shared running stream -- an interval near the fixed start never changes
// once it exists, however far the other end still has left to travel; only
// the growing tip ever gains or loses one. It won't match buildRoadMesh
// exactly once confirmed (that one centers its run and fits it exactly to
// the final length; this one only ever grows from `start` and can fall a
// little short of `end`) -- an acceptable, deliberate trade for a live
// preview that has to redraw itself dozens of times a second without
// visibly jittering.

/** Combines a base seed with a handful of small integers into a fresh RNG
 *  seed -- the same coordinate-hashing trick buildJunctionPatch already uses
 *  below (large, unrelated odd multipliers XORed together), reused here to
 *  turn (road id, which interval) into its own independent seed instead of
 *  a position in one shared running stream. */
function seedAt(base: number, ...indices: number[]): number {
  let h = (base * 2 + 1) >>> 0;
  for (const i of indices) h = (Math.imul(h, 73856093) ^ Math.imul(i + 1, 19349663)) >>> 0;
  return h;
}

/** Translucent preview material -- its own instance, not sceneryMaterial:
 *  that one is shared by every piece of BUILT scenery in the game, and
 *  turning it transparent would fade all of it, not just the road actually
 *  being dragged. */
const GHOST_OPACITY = 0.225;
export const roadGhostMaterial = new THREE.MeshStandardMaterial({
  roughness: 0.94, metalness: 0, vertexColors: true, flatShading: true,
  transparent: true, opacity: GHOST_OPACITY,
});
/** Brightened the same way sceneryHighlightMaterial is (blocks.ts) -- what
 *  roads.ts swaps a ghost mesh's own `.material` to while it's the current
 *  snap target, same swap-not-mutate reasoning that material's own header
 *  gives. */
export const roadGhostHighlightMaterial = new THREE.MeshStandardMaterial({
  roughness: 0.94, metalness: 0, vertexColors: true, flatShading: true,
  transparent: true, opacity: GHOST_OPACITY, emissive: 0xffffff, emissiveIntensity: 0.35,
});

/**
 * A live preview of buildRoadMesh's own look, safe to rebuild every
 * pointermove of an active drag -- see this section's own header for why its
 * layout algorithm differs from the confirmed one. `seed` is the road's own
 * id, same convention buildRoadMesh uses.
 */
export function buildRoadGhostMesh(start: THREE.Vector2, end: THREE.Vector2, seed: number): THREE.Mesh {
  const dx = end.x - start.x, dz = end.y - start.y;
  const length = Math.max(Math.hypot(dx, dz), 1e-3);
  const yaw = yawFromDirection(dx, dz);
  rightFromYaw(yaw, right);

  const b = new Blocks();

  // ── The dirt bed, fixed-length tiles measured from `start` ────────────
  let segIndex = 0;
  for (let t = 0; t < length; t += BED_SEGMENT_LENGTH, segIndex++) {
    const segLength = Math.min(BED_SEGMENT_LENGTH, length - t);
    const mid = t + segLength / 2;
    const cx = start.x + Math.sin(yaw) * mid, cz = start.y + Math.cos(yaw) * mid;
    // `0` as the leading index tags this a BED seed, disjoint from the
    // cobblestone loop's own leading index (always -1 or 1 below) -- the two
    // loops can never accidentally land on the same seed.
    const rng = makeRng(seedAt(seed, 0, segIndex));
    b.standing(
      ROAD_HALF_WIDTH * 2, between(rng, 0.8, 1.15) * BED_HEIGHT, segLength * 1.02,
      cx, cz, pick(rng, DIRT_PALETTE), yaw,
    );
  }

  // ── Cobblestones, fixed pitch from `start` on each edge ───────────────
  for (const side of [-1, 1]) {
    let i = 0;
    for (let t = COBBLE_SIZE / 2; t <= length - COBBLE_SIZE / 2; t += COBBLE_PITCH, i++) {
      const rng = makeRng(seedAt(seed, side, i));
      const cx = start.x + Math.sin(yaw) * t + right.x * COBBLE_EDGE_OFFSET * side;
      const cz = start.y + Math.cos(yaw) * t + right.y * COBBLE_EDGE_OFFSET * side;
      const size = COBBLE_SIZE * between(rng, 0.75, 1.15);
      b.standing(size, COBBLE_HEIGHT * between(rng, 0.8, 1.2), size, cx, cz, pick(rng, COBBLE_PALETTE), between(rng, -0.4, 0.4));
    }
  }

  return new THREE.Mesh(b.build(), roadGhostMaterial);
}

/**
 * A small raised plaza at a point where two or more roads meet, built from
 * the connection graph rather than each road pretending the others don't
 * exist (see roads.ts's own comment on why this exists at all). Sitting
 * ABOVE every cobble that might reach into its footprint -- not mitring or
 * trimming any road's own geometry to fit -- is what makes this safe to
 * build independently of however many roads actually converge here: nothing
 * below it needs to know it's there.
 *
 * Two overlapping squares rather than one, offset 45 degrees, for a rough
 * octagon instead of a bare diamond -- cheap, and this kit has no curves to
 * reach for instead (same trick landmarks.ts's bush() uses turned boxes for).
 */
export function buildJunctionPatch(point: { x: number; y: number }): THREE.Mesh {
  // Seeded off the point itself, not a road id -- a junction belongs to no
  // single road (roads.ts writes a connection symmetrically onto BOTH
  // sides), and whichever host happens to build it first has to land on the
  // same colours as whichever host builds it second.
  const rng = makeRng((Math.round(point.x * 100) * 73856093) ^ (Math.round(point.y * 100) * 19349663));
  const b = new Blocks();
  b.standing(JUNCTION_HALF * 2, JUNCTION_HEIGHT, JUNCTION_HALF * 2, point.x, point.y, pick(rng, JUNCTION_PALETTE));
  // The second square shares its ENTIRE footprint with the first everywhere
  // the two overlap (the whole point of stacking them at 45deg for a rounder
  // silhouette) -- two coplanar top faces at the exact same height is a
  // textbook z-fight, flickering between them per pixel. A hair taller is
  // enough to win the depth test outright across the whole overlap, and at
  // 2% of JUNCTION_HEIGHT it reads as one flat plaza, not a step.
  b.standing(JUNCTION_HALF * 2, JUNCTION_HEIGHT * 1.02, JUNCTION_HALF * 2, point.x, point.y, pick(rng, JUNCTION_PALETTE), Math.PI / 4);
  return new THREE.Mesh(b.build(), sceneryMaterial);
}
