import * as THREE from 'three';
import { SOLDIER_HEIGHT } from './constants.ts';

// A court's castle, built from a PLAN rather than modelled as one lump.
//
// The structure is a graph: towers are nodes, walls are edges between them. A
// tower can carry any number of walls; a wall connects exactly two towers, and
// that invariant is enforced (see validatePlan) rather than merely intended.
// Everything else -- wall lengths, angles, how many merlons fit along a given
// run -- is derived from where the towers are, so a different plan is a
// different castle with no new geometry code. The square keep below is just
// the first plan; a pentagon, a long curtain wall, or a broken siege ruin with
// a wall missing are all expressible today.
//
// In the middle of the enclosure stands the keep, which is what flies the flag.
//
// Same idiom as the characters: axis-aligned boxes, no bevels, no curves, all
// sized in multiples of SOLDIER_HEIGHT so the whole scene shares one scale.
// Built facing +Z (the shared convention, see frame.ts), so a castle set to a
// court's own facing presents its gate toward the middle of the board.

const H = SOLDIER_HEIGHT;

// Walls stand two characters tall, as asked; towers overtop them, and the keep
// overtops the towers, so the silhouette steps up toward the flag.
const WALL_H = 2 * H;
const WALL_T = 0.45 * H;
const TOWER_W = 1.15 * H, TOWER_H = 2.8 * H;
const KEEP_W = 2.7 * H, KEEP_D = 2.7 * H, KEEP_H = 3.5 * H;

const MERLON_W = 0.24 * H, MERLON_H = 0.3 * H, MERLON_D = 0.44 * H;
/** Target spacing between merlon centers. Actual spacing is fitted to each
 *  run's length so the end merlons always land flush with the corners. */
const MERLON_PITCH = 0.62 * H;

const GATE_W = 1.3 * H, GATE_H = 1.5 * H;
const TRIM_H = 0.1 * H;
const POLE_W = 0.06 * H, POLE_H = 1.5 * H;
const FLAG_W = 0.85 * H, FLAG_H = 0.5 * H, FLAG_D = 0.03 * H;

/** Side of the default square enclosure, tower center to tower center. */
export const CASTLE_SPAN = 9 * H;
/** Half the footprint of a default castle, including the towers' own width --
 *  what a caller needs to know to place one without hanging it off the board. */
export const CASTLE_HALF_EXTENT = CASTLE_SPAN / 2 + TOWER_W / 2;

const STONE = new THREE.Color(0x8b90a0);
const GATE_WOOD = 0x2b2119;
const POLE_WOOD = 0x4a4038;

/** A wall runs between exactly two towers, named by index into the plan's
 *  tower list. `gate` puts the castle's entrance in this wall. */
export interface WallSpec {
  from: number;
  to: number;
  gate?: boolean;
}

export interface CastlePlan {
  /** Tower positions in castle-local space (x, z). */
  towers: THREE.Vector2[];
  walls: WallSpec[];
}

export interface Castle {
  group: THREE.Group;
  flag: THREE.Mesh;
}

/** Square enclosure: four towers at the corners, four walls joining them, the
 *  gate in the +Z wall so it faces whatever the castle faces. */
export function squareCastlePlan(span = CASTLE_SPAN): CastlePlan {
  const h = span / 2;
  return {
    // Wound so that consecutive towers are adjacent corners.
    towers: [
      new THREE.Vector2(-h, h), // front left
      new THREE.Vector2(h, h), // front right
      new THREE.Vector2(h, -h), // back right
      new THREE.Vector2(-h, -h), // back left
    ],
    walls: [
      { from: 0, to: 1, gate: true }, // front wall, facing the board
      { from: 1, to: 2 },
      { from: 2, to: 3 },
      { from: 3, to: 0 },
    ],
  };
}

/** Enforces the structural rule: a wall joins two DISTINCT, existing towers.
 *  Thrown rather than warned -- a malformed plan is a bug in the caller, and a
 *  wall dangling off a nonexistent tower would otherwise render as a silent
 *  NaN-positioned slab. */
function validatePlan(plan: CastlePlan) {
  if (plan.towers.length < 2) throw new Error('castle plan needs at least 2 towers');
  plan.walls.forEach((w, i) => {
    const ok = (n: number) => Number.isInteger(n) && n >= 0 && n < plan.towers.length;
    if (!ok(w.from) || !ok(w.to)) {
      throw new Error(`castle wall ${i} references a tower that does not exist (${w.from} -> ${w.to}, ${plan.towers.length} towers)`);
    }
    if (w.from === w.to) throw new Error(`castle wall ${i} starts and ends at the same tower (${w.from})`);
  });
}

// Shared across every castle; only materials differ per court.
const towerGeo = new THREE.BoxGeometry(TOWER_W, TOWER_H, TOWER_W);
const keepGeo = new THREE.BoxGeometry(KEEP_W, KEEP_H, KEEP_D);
const merlonGeo = new THREE.BoxGeometry(MERLON_W, MERLON_H, MERLON_D);
const gateGeo = new THREE.BoxGeometry(GATE_W, GATE_H, WALL_T + 0.02 * H);
const trimGeo = new THREE.BoxGeometry(KEEP_W + 0.02 * H, TRIM_H, KEEP_D + 0.02 * H);
const poleGeo = new THREE.BoxGeometry(POLE_W, POLE_H, POLE_W);
const flagGeo = new THREE.BoxGeometry(FLAG_W, FLAG_H, FLAG_D);

// Wall slabs are the one thing that can't be shared: every wall is a different
// length. Cached by length so a plan with equal-length walls (every regular
// polygon, including the default square) still builds just one.
const wallGeoCache = new Map<number, THREE.BoxGeometry>();
function wallGeometry(length: number): THREE.BoxGeometry {
  const key = Math.round(length * 1e4);
  let geo = wallGeoCache.get(key);
  if (!geo) {
    geo = new THREE.BoxGeometry(length, WALL_H, WALL_T);
    wallGeoCache.set(key, geo);
  }
  return geo;
}

/**
 * Lays merlons along a horizontal run, fitted so the first and last land flush
 * with its ends.
 *
 * @param out receives one matrix per merlon, for the shared InstancedMesh.
 */
function crenellate(
  out: THREE.Matrix4[],
  centerX: number, centerZ: number, y: number,
  length: number, yaw: number,
) {
  // At least two, so a run always reads as crenellated rather than as a single
  // lonely block. Spacing stretches or squeezes from MERLON_PITCH to fit.
  const count = Math.max(2, Math.round(length / MERLON_PITCH));
  const usable = length - MERLON_W;
  const step = usable / (count - 1);
  const dirX = Math.cos(yaw), dirZ = -Math.sin(yaw);
  for (let i = 0; i < count; i++) {
    const t = -usable / 2 + i * step;
    const m = new THREE.Matrix4();
    m.makeRotationY(yaw);
    m.setPosition(centerX + dirX * t, y, centerZ + dirZ * t);
    out.push(m);
  }
}

/**
 * Builds a castle in a court's colors.
 *
 * The stone is tinted 18% toward the owner's color rather than left neutral
 * gray. It's subtle up close, but it's what lets you tell whose corner you're
 * looking at from across a board this size, where the flag is only a few
 * pixels tall. The flag and the keep's trim band carry the color at full
 * strength.
 */
export function buildCastle(color: number, plan: CastlePlan = squareCastlePlan()): Castle {
  validatePlan(plan);

  const own = new THREE.Color(color);
  const tinted = STONE.clone().lerp(own, 0.18);
  const wallMat = new THREE.MeshStandardMaterial({ color: tinted, roughness: 0.95, metalness: 0 });
  // Towers a shade darker than the walls they join. Flat-lit boxes of one
  // identical color merge into a single silhouette; a little separation keeps
  // the structure legible.
  const towerMat = new THREE.MeshStandardMaterial({
    color: tinted.clone().multiplyScalar(0.86), roughness: 0.95, metalness: 0,
  });
  const keepMat = new THREE.MeshStandardMaterial({
    color: tinted.clone().multiplyScalar(0.94), roughness: 0.95, metalness: 0,
  });
  const trimMat = new THREE.MeshStandardMaterial({
    color: own.clone().multiplyScalar(0.8), roughness: 0.7, metalness: 0,
  });
  const flagMat = new THREE.MeshStandardMaterial({
    color: own, roughness: 0.6, metalness: 0, side: THREE.DoubleSide,
  });
  const gateMat = new THREE.MeshStandardMaterial({ color: GATE_WOOD, roughness: 1 });
  const poleMat = new THREE.MeshStandardMaterial({ color: POLE_WOOD, roughness: 0.9 });

  const group = new THREE.Group();
  // Every merlon in the castle, collected as instance matrices and drawn in a
  // single call at the end. A castle this size carries around 70 of them, and
  // four castles of individual meshes would be ~300 extra draw calls for what
  // is one box repeated.
  const merlons: THREE.Matrix4[] = [];

  // ── Walls ───────────────────────────────────────────────────────────────
  for (const wall of plan.walls) {
    const a = plan.towers[wall.from], b = plan.towers[wall.to];
    const dx = b.x - a.x, dz = b.y - a.y;
    const length = Math.hypot(dx, dz);
    if (length < 1e-6) continue; // coincident towers: nothing to span

    // A mesh's local +X maps to world (cos y, 0, -sin y) under rotation.y, so
    // this is the yaw that points the slab's length down the wall's run.
    const yaw = Math.atan2(-dz, dx);
    const midX = (a.x + b.x) / 2, midZ = (a.y + b.y) / 2;

    // Spans tower center to tower center: the towers overlap the wall's ends,
    // which is what guarantees no seam or gap at a corner regardless of the
    // angle the two walls meet at.
    const slab = new THREE.Mesh(wallGeometry(length), wallMat);
    slab.position.set(midX, WALL_H / 2, midZ);
    slab.rotation.y = yaw;
    group.add(slab);

    crenellate(merlons, midX, midZ, WALL_H + MERLON_H / 2, length, yaw);

    if (wall.gate) {
      const gate = new THREE.Mesh(gateGeo, gateMat);
      gate.position.set(midX, GATE_H / 2, midZ);
      gate.rotation.y = yaw;
      group.add(gate);
    }
  }

  // ── Towers ──────────────────────────────────────────────────────────────
  for (const t of plan.towers) {
    const tower = new THREE.Mesh(towerGeo, towerMat);
    tower.position.set(t.x, TOWER_H / 2, t.y);
    group.add(tower);

    // Merlons at the four top corners rather than a row across one face: a
    // tower is seen from every angle as the camera orbits, and corner merlons
    // read as a battlement from all of them.
    const inset = (TOWER_W - MERLON_W) / 2;
    for (const mx of [-1, 1]) {
      for (const mz of [-1, 1]) {
        const m = new THREE.Matrix4();
        m.setPosition(t.x + mx * inset, TOWER_H + MERLON_H / 2, t.y + mz * inset);
        merlons.push(m);
      }
    }
  }

  // ── Keep, in the middle of the enclosure ────────────────────────────────
  // Placed at the centroid of the towers rather than at the local origin, so
  // an asymmetric plan still puts its keep inside its own walls.
  const center = plan.towers
    .reduce((acc, t) => acc.add(t), new THREE.Vector2())
    .divideScalar(plan.towers.length);

  const keep = new THREE.Mesh(keepGeo, keepMat);
  keep.position.set(center.x, KEEP_H / 2, center.y);
  group.add(keep);

  // Colored band below the keep's crenellations -- the heraldic stripe that
  // says whose castle this is even when the flag is edge-on to the camera.
  // Slightly oversized so it stands proud of the walls and can't z-fight.
  const trim = new THREE.Mesh(trimGeo, trimMat);
  trim.position.set(center.x, KEEP_H - TRIM_H / 2 - 0.12 * H, center.y);
  group.add(trim);

  crenellate(merlons, center.x, center.y, KEEP_H + MERLON_H / 2, KEEP_W, 0);
  crenellate(merlons, center.x, center.y, KEEP_H + MERLON_H / 2, KEEP_D, Math.PI / 2);

  const pole = new THREE.Mesh(poleGeo, poleMat);
  pole.position.set(center.x, KEEP_H + POLE_H / 2, center.y);
  group.add(pole);

  // The flag hangs off one side of the pole, just below its top. DoubleSide,
  // because a flat plane is otherwise invisible from behind -- and with an
  // orbit camera, "behind" is half the views.
  const flag = new THREE.Mesh(flagGeo, flagMat);
  flag.position.set(
    center.x + POLE_W / 2 + FLAG_W / 2,
    KEEP_H + POLE_H - FLAG_H / 2 - 0.1 * H,
    center.y,
  );
  group.add(flag);

  // ── The battlements, in one draw call ───────────────────────────────────
  const battlements = new THREE.InstancedMesh(merlonGeo, wallMat, merlons.length);
  merlons.forEach((m, i) => battlements.setMatrixAt(i, m));
  battlements.instanceMatrix.needsUpdate = true;
  // Required: an InstancedMesh's bounding box is null until asked for, and
  // without it frustum culling (and any Box3.setFromObject on the castle)
  // falls back to the single un-instanced box at the origin.
  battlements.computeBoundingBox();
  battlements.computeBoundingSphere();
  group.add(battlements);

  return { group, flag };
}
