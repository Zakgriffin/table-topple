import * as THREE from 'three';
import { BOARD_SIZE, SOLDIER_HEIGHT } from './constants.ts';
import { onCourtGround, terrainAt } from './terrain.ts';
import { fbm, makeRng } from './noise.ts';
import { Blocks, between, pick, sceneryMaterial } from './blocks.ts';

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
 *  opening camera looks straight down the red court's line. */
const COURT_TREE_MARGIN = 5;
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

  variants.forEach((geometry, i) => {
    const group = byShape[i];
    if (group.length === 0) return;

    const mesh = new THREE.InstancedMesh(geometry, sceneryMaterial, group.length);
    group.forEach((p, j) => {
      quaternion.setFromAxisAngle(up, p.yaw);
      position.set(p.x, 0, p.z);
      scale.setScalar(p.scale);
      mesh.setMatrixAt(j, matrix.compose(position, quaternion, scale));
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
