import * as THREE from 'three';
import { BOARD_SIZE, DEATH_FADE_TIME, DEATH_SHAKE_TIME, SOLDIER_HEIGHT } from './constants.ts';
import { onCourtGround, terrainAt } from './terrain.ts';
import { fbm, makeRng } from './noise.ts';
import { Blocks, between, pick, sceneryMaterial } from './blocks.ts';
import type { Interactable } from './interactable.ts';
import { deathShakeOffset, setOpacity } from './health.ts';

// Trees scattered over the forest terrain, which is most of the board.
//
// The floor already tells you where the forest is; this is what makes it feel
// like one. Density comes from a noise field rather than being uniform, so the
// board gets thickets and clearings -- an even sprinkle reads as an orchard,
// and the clearings are what give the open ground somewhere to be.
//
// Drawn as a handful of INSTANCED variants: a few hundred trees would be a few
// hundred draw calls otherwise, and they're the same object repeated. Each
// variant is modelled once (as a merged block structure, blocks.ts) and then
// stamped out with a per-tree position, yaw and scale, so trees of the same
// variant never look copy-pasted.
//
// Like landmarks.ts, this imports nothing from scene.ts -- board.ts is what
// puts the result in the world -- so it stays checkable under node.
//
// Scenery only: no collision, so a denizen walks through a trunk today.

const H = SOLDIER_HEIGHT;
const FOREST_SEED = 0x7ee5;

const BARK = [0x3a2e22, 0x2e2519, 0x453729];
/** Conifer needles: the darkest foliage on the board, barely above the forest
 *  floor's own colour, so a stand of them reads as mass rather than as
 *  individual trees. */
const NEEDLE = [0x1f3a26, 0x24462b, 0x1a3120, 0x28492c];
/** Broadleaf canopy, a step lighter and more varied -- these are the trees you
 *  pick out individually against the dark. */
const LEAF = [0x2b4f2c, 0x335c33, 0x24421f, 0x3a6433];

// ── The variants ───────────────────────────────────────────────────────────

/** A conifer: a short trunk under tiers that narrow toward the top. The taper
 *  is the whole silhouette, so the widths are what matter, not the count. */
function conifer(b: Blocks, rng: () => number) {
  const height = between(rng, 3.0, 4.6) * H;
  const trunkH = height * 0.3;
  b.standing(0.24 * H, trunkH, 0.24 * H, 0, 0, pick(rng, BARK));

  const tiers = 4 + Math.floor(rng() * 3);
  // Tiers start below the trunk's top so the lowest skirt hides the join --
  // a gap there makes the tree look like a bush balanced on a stick.
  const bottom = trunkH * 0.6;
  const span = height - bottom;
  for (let i = 0; i < tiers; i++) {
    const t = i / (tiers - 1);
    const w = (2.0 - 1.55 * t) * between(rng, 0.9, 1.1) * H;
    // Overlapping (1.7x the even spacing) so the tiers read as one mass with
    // stepped edges rather than as separate plates floating apart.
    const tierH = (span / tiers) * 1.7;
    b.add(
      w, tierH, w,
      between(rng, -0.08, 0.08) * H, bottom + t * (span - tierH) + tierH / 2, between(rng, -0.08, 0.08) * H,
      pick(rng, NEEDLE), between(rng, 0, Math.PI / 2),
    );
  }
}

/** A broadleaf: a taller bare trunk carrying a rounded mass of overlapping
 *  lumps. Rounder and lighter than a conifer, which is what makes a mixed
 *  forest look mixed at a distance. */
function broadleaf(b: Blocks, rng: () => number) {
  const height = between(rng, 2.9, 4.3) * H;
  const trunkH = height * between(rng, 0.4, 0.5);
  b.standing(0.3 * H, trunkH, 0.3 * H, 0, 0, pick(rng, BARK));

  const lumps = 4 + Math.floor(rng() * 3);
  const crown = height - trunkH;
  for (let i = 0; i < lumps; i++) {
    const w = between(rng, 1.2, 2.1) * H;
    const lh = between(rng, 0.7, 1.2) * H;
    b.add(
      w, lh, w * between(rng, 0.8, 1.15),
      between(rng, -0.55, 0.55) * H,
      trunkH + between(rng, 0.1, 0.9) * crown,
      between(rng, -0.55, 0.55) * H,
      pick(rng, LEAF), between(rng, 0, Math.PI / 2),
    );
  }
}

/** A dead tree: trunk and bare branches. A few of these among the living ones
 *  do more for the forest reading as old growth than another green variant
 *  would, and they're the only trees you can see through. */
function deadTree(b: Blocks, rng: () => number) {
  const height = between(rng, 2.4, 3.6) * H;
  b.standing(0.28 * H, height, 0.28 * H, 0, 0, pick(rng, BARK), between(rng, 0, Math.PI));

  const branches = 3 + Math.floor(rng() * 3);
  for (let i = 0; i < branches; i++) {
    const len = between(rng, 0.7, 1.4) * H;
    const yaw = (i / branches) * Math.PI * 2 + between(rng, -0.5, 0.5);
    // Rolled off vertical, then yawed, so the branch reaches outward and up
    // from the trunk (blocks.ts applies roll first for exactly this).
    const roll = between(rng, 0.9, 1.35);
    const y = height * between(rng, 0.45, 0.92);
    b.add(
      0.14 * H, len, 0.14 * H,
      Math.sin(yaw) * len * 0.4, y, Math.cos(yaw) * len * 0.4,
      pick(rng, BARK), yaw, roll,
    );
  }
}

const SHAPES = [conifer, conifer, conifer, broadleaf, broadleaf, broadleaf, deadTree, deadTree];

// ── Where they grow ────────────────────────────────────────────────────────

/** Grid pitch for candidate positions, world units. A jittered grid rather
 *  than free random placement with a rejection test: the grid guarantees a
 *  minimum spacing for free, where uniform random would clump two trunks into
 *  the same spot and need an O(n^2) check to stop it. */
const GRID_STEP = 4.0;
/** Jitter either way. Under half the step, so two neighbours can never swap
 *  order or collide -- worst case leaves them GRID_STEP - 2*JITTER apart. */
const GRID_JITTER = 1.2;

/** World units per feature of the density field -- the size of a thicket or a
 *  clearing. Large: this should read as terrain-scale structure, not as noise
 *  on the tree spacing. */
const DENSITY_SCALE = 34;
/** Chance a candidate becomes a tree, at the emptiest and thickest parts of
 *  the density field. The low end is deliberately not zero: a completely bare
 *  clearing looks cut out, while a few scattered trees read as thinning. */
const DENSITY_MIN = 0.08;
const DENSITY_MAX = 0.9;

/** Kept clear around each court, beyond the keep-out the patches already
 *  respect: the courts need open ground to form up and fight in, and the
 *  opening camera looks straight down the red court's line. Cut from 5 --
 *  this only pushes trees back from the box's edge nearest the battlefield
 *  (the box itself, all the way to the rim, is already off-limits and
 *  covers the castle regardless of this number), so shrinking it brings
 *  trees nearer the court without ever letting one grow inside a castle's
 *  own footprint. */
const COURT_TREE_MARGIN = 1;
/** Kept clear of the board's rim, so no tree is half over the edge. */
const EDGE_MARGIN = 2.5;

interface Placement {
  shape: number;
  x: number;
  z: number;
  yaw: number;
  scale: number;
}

function scatter(rng: () => number): Placement[] {
  const out: Placement[] = [];
  const limit = BOARD_SIZE / 2 - EDGE_MARGIN;

  for (let gz = -limit; gz <= limit; gz += GRID_STEP) {
    for (let gx = -limit; gx <= limit; gx += GRID_STEP) {
      const x = gx + between(rng, -GRID_JITTER, GRID_JITTER);
      const z = gz + between(rng, -GRID_JITTER, GRID_JITTER);
      if (Math.abs(x) > limit || Math.abs(z) > limit) continue;

      // Trees grow in the forest and nowhere else. That's what keeps them off
      // the patches without any separate test -- a patch is, by definition,
      // not forest -- and it means a landmark can never end up with a tree
      // standing inside it.
      if (terrainAt(x, z) !== 'forest') continue;
      if (onCourtGround(x, z, COURT_TREE_MARGIN)) continue;

      const density = fbm(x, z, DENSITY_SCALE, 2, FOREST_SEED);
      if (rng() > DENSITY_MIN + (DENSITY_MAX - DENSITY_MIN) * density) continue;

      out.push({
        shape: Math.floor(rng() * SHAPES.length),
        x, z,
        yaw: rng() * Math.PI * 2,
        scale: between(rng, 0.8, 1.25),
      });
    }
  }
  return out;
}

// ── Assembly ───────────────────────────────────────────────────────────────

/** Every tree, in one Group ready to be added to the scene. Not added here --
 *  see the note at the top about staying free of scene.ts. */
export const forest = new THREE.Group();

/** How many trees were placed. Exported for checking, and because it's the
 *  number to look at when tuning the density constants above. */
export let treeCount = 0;

/** One entry per placed tree -- unlike landmarks.ts's structures, trees were
 *  never individually tracked before act.ts needed to pick one out: they're
 *  drawn as InstancedMesh purely for draw-call count, and an instance has no
 *  identity of its own to a raycaster. This is the identity, built alongside
 *  the instancing loop below from the same Placement data. */
export interface PlacedTree {
  id: number;
  x: number;
  z: number;
  /** World-space canopy reach, same "read back off the geometry's own
   *  bounds" reasoning landmarks.ts gives for its own radius. */
  radius: number;
}
export const placedTrees: PlacedTree[] = [];

/** Where each tree's own instance actually lives -- its InstancedMesh and its
 *  index within it -- kept apart from PlacedTree (above) since nothing that
 *  reads placedTrees (act.ts, interactable.ts) needs to know this, only
 *  hideTree (below) does. */
const treeInstance = new Map<number, { mesh: THREE.InstancedMesh; index: number }>();

{
  const rng = makeRng(FOREST_SEED);

  // Shapes first, so every variant exists before anything is placed into it.
  const variants = SHAPES.map((shape) => {
    const b = new Blocks();
    shape(b, rng);
    return b.build();
  });

  const placements = scatter(rng);
  treeCount = placements.length;

  // Grouped by variant, because an InstancedMesh draws exactly one geometry.
  const byShape: Placement[][] = variants.map(() => []);
  for (const p of placements) byShape[p.shape].push(p);

  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const position = new THREE.Vector3();
  const scale = new THREE.Vector3();
  let nextTreeId = 0;

  variants.forEach((geometry, i) => {
    const group = byShape[i];
    if (group.length === 0) return;

    // Same "read the reach back off the geometry's own bounds" idiom
    // landmarks.ts uses, per-variant rather than per-structure since every
    // tree of this shape shares one geometry.
    const box = geometry.boundingBox!;
    const reach = Math.max(-box.min.x, box.max.x, -box.min.z, box.max.z);

    const mesh = new THREE.InstancedMesh(geometry, sceneryMaterial, group.length);
    group.forEach((p, j) => {
      quaternion.setFromAxisAngle(up, p.yaw);
      position.set(p.x, 0, p.z);
      scale.setScalar(p.scale);
      mesh.setMatrixAt(j, matrix.compose(position, quaternion, scale));
      const id = nextTreeId++;
      placedTrees.push({ id, x: p.x, z: p.z, radius: reach * p.scale });
      treeInstance.set(id, { mesh, index: j });
    });
    mesh.instanceMatrix.needsUpdate = true;
    // Required: an InstancedMesh's bounds are null until asked for, and without
    // them frustum culling falls back to the single un-instanced geometry at
    // the origin -- which would cull the entire forest the moment the board's
    // center left the view.
    mesh.computeBoundingBox();
    mesh.computeBoundingSphere();
    forest.add(mesh);
  });
}

/** Every placed tree, wrapped for act.ts's generic pick query -- see
 *  interactable.ts. `distanceTo` is distance to the trunk's own canopy
 *  boundary, same convention landmarks.ts's own wrapper uses. `setHighlighted`
 *  is a deliberate no-op: every tree of one shape shares ONE material on its
 *  InstancedMesh, so brightening it would brighten that whole variant across
 *  the board, not just this trunk. A real per-tree highlight would need
 *  per-instance colour (InstancedMesh.setColorAt) and isn't built -- nothing
 *  needs it yet, since act.ts doesn't hover-highlight targets, only the
 *  denizen being commanded. */
export function treeInteractables(): Interactable[] {
  const p = new THREE.Vector2();
  return placedTrees.map((t): Interactable => ({
    kind: 'tree',
    id: t.id,
    resource: 'wood',
    pos: new THREE.Vector2(t.x, t.z),
    radius: t.radius,
    distanceTo: (pt) => Math.max(0, pt.distanceTo(p.set(t.x, t.z)) - t.radius),
    setHighlighted: () => {},
  }));
}

const ZERO_SCALE = new THREE.Matrix4().makeScale(0, 0, 0);

// ── Falling ────────────────────────────────────────────────────────────────
// A chopped tree doesn't just vanish: it shakes, falls, and fades, in that
// order -- the same three beats a denizen's own death goes through
// (health.ts's renderVitals), plus a fourth act a denizen never needed (the
// fall itself). The shake and the fade are literally the SAME code
// (deathShakeOffset/setOpacity, imported above), so "something got
// destroyed" reads as one consistent effect across both, not two similar
// ones.
//
// An InstancedMesh instance can't be animated on its own -- one shared
// matrix buffer, one shared material, no such thing as "this one instance is
// mid-fade". So the moment a tree starts falling, its instance is zeroed
// (same trick the old instant-despawn used) and a genuine standalone Mesh
// takes its place for the length of the sequence, sharing the variant's
// geometry (never disposed here -- other live trees of that shape still use
// it) but with its OWN cloned material, since only this one tree's opacity
// should move.

const SHAKE_TIME = DEATH_SHAKE_TIME;
/** How long the topple itself takes, once the shake is done. */
const FALL_TIME = 0.5;
const FADE_TIME = DEATH_FADE_TIME;
/** Straight down to horizontal -- a felled tree ends up lying flat. */
const FALL_ANGLE = Math.PI / 2;
/** Ease-in: the exponent on the fall's own time fraction, so the topple
 *  starts slow and is moving fastest right as it reaches FALL_ANGLE. There's
 *  no ease-OUT to match -- the timer simply stops advancing the angle past
 *  that point, which is the "sudden stop" the ground would actually give a
 *  falling trunk, rather than a smooth deceleration into one. */
const FALL_EASE_POWER = 3;

interface FallingTree {
  mesh: THREE.Mesh;
  /** Where it was planted -- the shake phase jitters around this, the fall
   *  and fade phases sit exactly on it. */
  basePosition: THREE.Vector3;
  baseQuaternion: THREE.Quaternion;
  /** A random horizontal direction to topple in, chosen once at the start of
   *  the fall -- see beginFall. */
  fallAxis: THREE.Vector3;
  /** The tree's own placement scale, read back off its instance transform so
   *  the shake amplitude scales with the tree's actual size instead of
   *  assuming every tree is the same one. */
  scale: number;
  timer: number;
}
const fallingTrees: FallingTree[] = [];

const decomposedPos = new THREE.Vector3();
const decomposedQuat = new THREE.Quaternion();
const decomposedScale = new THREE.Vector3();

/** Spawns the standalone falling copy of a tree at exactly the transform its
 *  instance had the moment it was chopped -- the swap between the two has to
 *  be frame-perfect, or the tree visibly jumps the instant it starts to
 *  fall. */
function beginFall(geometry: THREE.BufferGeometry, instanceMatrix: THREE.Matrix4) {
  instanceMatrix.decompose(decomposedPos, decomposedQuat, decomposedScale);

  const mesh = new THREE.Mesh(geometry, sceneryMaterial.clone());
  mesh.position.copy(decomposedPos);
  mesh.quaternion.copy(decomposedQuat);
  mesh.scale.copy(decomposedScale);
  forest.add(mesh);

  // Cosmetic and one-off -- same reasoning denizens.ts's own blinkTimer
  // stagger gives for using Math.random() rather than the board's seeded rng:
  // this happens on demand, from a player's click, not at deterministic
  // board-build time.
  const azimuth = Math.random() * Math.PI * 2;

  fallingTrees.push({
    mesh,
    basePosition: decomposedPos.clone(),
    baseQuaternion: decomposedQuat.clone(),
    fallAxis: new THREE.Vector3(Math.cos(azimuth), 0, Math.sin(azimuth)),
    scale: decomposedScale.x,
    timer: 0,
  });
}

/**
 * Advances every tree mid-fall by dt. Called once a frame from sim.ts's
 * step(), same as updateActions/updateIdle -- order doesn't matter relative
 * to either, since a falling tree is its own standalone object touching
 * nothing a denizen's frame also writes.
 */
export function updateFallingTrees(dt: number) {
  for (let i = fallingTrees.length - 1; i >= 0; i--) {
    const f = fallingTrees[i];
    f.timer += dt;
    const t = f.timer;

    if (t < SHAKE_TIME) {
      const shake = deathShakeOffset(t, SHAKE_TIME, f.scale);
      f.mesh.position.x = f.basePosition.x + shake.x;
      f.mesh.position.z = f.basePosition.z + shake.z;
    } else {
      f.mesh.position.copy(f.basePosition);
      const fallT = Math.min(1, (t - SHAKE_TIME) / FALL_TIME);
      const angle = FALL_ANGLE * fallT ** FALL_EASE_POWER;
      f.mesh.quaternion.copy(f.baseQuaternion);
      f.mesh.rotateOnWorldAxis(f.fallAxis, angle);
    }

    const fadeStart = SHAKE_TIME + FALL_TIME;
    if (t < fadeStart) continue;

    const fadeT = Math.min(1, (t - fadeStart) / FADE_TIME);
    setOpacity(f.mesh, 1 - fadeT);
    if (fadeT >= 1) {
      forest.remove(f.mesh);
      (f.mesh.material as THREE.Material).dispose();
      fallingTrees.splice(i, 1);
    }
  }
}

/**
 * Chops a tree down: its instance collapses to zero scale (an InstancedMesh
 * has no per-instance visibility flag, so a degenerate transform is how one
 * instance disappears without rebuilding the buffer or touching its
 * neighbours) while a standalone copy takes over for the shake/fall/fade
 * sequence (beginFall, above), and it's dropped from placedTrees so it can
 * never be picked as a target again. Returns false if `id` was already gone
 * -- actions.ts uses that to guard against two denizens racing to the same
 * tree both getting paid.
 */
export function hideTree(id: number): boolean {
  const i = placedTrees.findIndex((t) => t.id === id);
  if (i === -1) return false;
  placedTrees.splice(i, 1);

  const instance = treeInstance.get(id);
  if (instance) {
    // Read the instance's own transform BEFORE zeroing it -- beginFall needs
    // it to start the standalone copy from the exact same spot.
    const matrix = new THREE.Matrix4();
    instance.mesh.getMatrixAt(instance.index, matrix);
    beginFall(instance.mesh.geometry, matrix);

    instance.mesh.setMatrixAt(instance.index, ZERO_SCALE);
    instance.mesh.instanceMatrix.needsUpdate = true;
  }
  return true;
}
