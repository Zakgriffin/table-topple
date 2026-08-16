import * as THREE from 'three';
import { SOLDIER_HEIGHT } from './constants.ts';
import { type Patch, type Terrain, patches } from './terrain.ts';
import { makeRng } from './noise.ts';
import { Blocks, between, pick, sceneryMaterial } from './blocks.ts';

// One structure per terrain patch, so a patch is somewhere you go rather than
// just a stain on the floor: a clump of bushes on a field, a mine mouth on a
// cave, a wrecked church on hallowed ground. Forest has none -- it's the
// board's baseline, and a landmark on every cell would be no landmark at all.
//
// Same idiom as character.ts and castle.ts: axis-aligned boxes, no bevels, no
// curves, everything a multiple of SOLDIER_HEIGHT so the whole scene keeps one
// scale. Each structure is built facing +Z (frame.ts's convention) around an
// origin on the ground between its feet, so placing one is a position and a
// yaw with no half-height offset to remember.
//
// Nothing here imports scene.ts -- like castle.ts, this builds and hands back
// a Group, and board.ts is what puts it in the world. That keeps the module
// runnable under node for checking.
//
// These are scenery only: they have no collision, so a denizen walks straight
// through a wall today. Whenever that changes, the footprint each builder
// already computes for fitting is the thing to hang it on.

const H = SOLDIER_HEIGHT;
const LANDMARK_SEED = 0xb0a12;

// ── Palettes ───────────────────────────────────────────────────────────────
// Local to this module, following castle.ts's STONE/GATE_WOOD rather than the
// shared palette in constants.ts: these are one structure's materials, not the
// game's colour scheme. Several shades per material so neighbouring blocks
// differ slightly -- flat-lit boxes of one identical colour merge into a
// single silhouette, which is the same reason castle.ts darkens its towers.

const FOLIAGE = [0x2f5c31, 0x386b39, 0x27502a, 0x436f3a];
const BARK = [0x4a3a28, 0x3b2e20];
const ROCK = [0x5a5c66, 0x4e505a, 0x666974, 0x53555e];
const TIMBER = [0x53412c, 0x45361f];
/** The dark inside a mine. Near black rather than actually black, so it still
 *  takes a little light and reads as depth instead of a hole in the render. */
const CAVE_DARK = 0x07080b;
/** Ruin stone: paler and warmer than castle.ts's STONE, which is the point --
 *  these are old, sun-bleached, and standing on ground the game calls
 *  hallowed. */
const RUIN_STONE = [0x7f7d73, 0x6e6c63, 0x8b897e, 0x757269];

// ── Field: a shrubbery ─────────────────────────────────────────────────────

/** One bush: a squat base with two or three smaller lumps piled on and around
 *  it, each turned off-axis. The turn is what matters -- a bush of
 *  axis-aligned boxes reads as a stack of crates, and a few degrees of yaw on
 *  each lump is enough to break that up without leaving the blocky idiom. */
function bush(b: Blocks, rng: () => number, cx: number, cz: number, size: number) {
  const baseW = 1.1 * size * H, baseH = 0.55 * size * H;
  b.standing(baseW, baseH, baseW * 0.9, cx, cz, pick(rng, FOLIAGE), between(rng, 0, Math.PI));

  const lumps = 2 + Math.floor(rng() * 2);
  for (let i = 0; i < lumps; i++) {
    const w = between(rng, 0.45, 0.8) * size * H;
    const h = between(rng, 0.35, 0.7) * size * H;
    b.add(
      w, h, w * between(rng, 0.8, 1.1),
      cx + between(rng, -0.35, 0.35) * size * H,
      baseH * between(rng, 0.7, 1.1) + h / 2,
      cz + between(rng, -0.35, 0.35) * size * H,
      pick(rng, FOLIAGE), between(rng, 0, Math.PI),
    );
  }
}

function buildShrubbery(rng: () => number): THREE.BufferGeometry {
  const b = new Blocks();

  // A loose ring rather than a grid: the bushes are placed by angle so the
  // clump has a middle and an edge, which is what makes it read as one thing
  // instead of several bushes that happen to be near each other.
  //
  // Spread wide enough to hold its own next to the mine and the church, which
  // are both about seven units across. A botanically sensible little thicket
  // of three looked like litter dropped in the middle of a patch seventeen
  // units wide -- these have to read from across a 144-unit board.
  const count = 7 + Math.floor(rng() * 4);
  for (let i = 0; i < count; i++) {
    const angle = (i / count) * Math.PI * 2 + between(rng, -0.4, 0.4);
    const dist = between(rng, 0.6, 2.7) * H;
    bush(b, rng, Math.sin(angle) * dist, Math.cos(angle) * dist, between(rng, 0.9, 1.6));
  }
  // One in the middle, taller than the ring, so the clump has a peak.
  bush(b, rng, between(rng, -0.2, 0.2), between(rng, -0.2, 0.2), between(rng, 1.3, 1.7));

  // A few saplings poking out of the top -- the vertical accent that stops the
  // whole thing being one low mound.
  const saplings = 2 + Math.floor(rng() * 3);
  for (let i = 0; i < saplings; i++) {
    const x = between(rng, -2.2, 2.2) * H, z = between(rng, -2.2, 2.2) * H;
    const trunkH = between(rng, 1.5, 2.6) * H;
    b.standing(0.13 * H, trunkH, 0.13 * H, x, z, pick(rng, BARK));
    for (let j = 0; j < 3; j++) {
      const w = between(rng, 0.5, 0.85) * H;
      b.add(
        w, between(rng, 0.3, 0.5) * H, w,
        x + between(rng, -0.2, 0.2) * H,
        trunkH - between(rng, -0.15, 0.35) * H,
        z + between(rng, -0.2, 0.2) * H,
        pick(rng, FOLIAGE), between(rng, 0, Math.PI),
      );
    }
  }

  return b.build();
}

// ── Cave: a mine ───────────────────────────────────────────────────────────

function buildMine(rng: () => number): THREE.BufferGeometry {
  const b = new Blocks();

  const MOUTH_W = 1.9 * H; // clear opening -- a king is 2H tall, so he stoops
  const MOUTH_H = 2.4 * H;
  const MASS_D = 3.6 * H;
  const SIDE_W = 2.2 * H;

  // The hillside is built as two flanking masses with a genuine GAP between
  // them, spanned by a third above -- not as one solid block with a dark
  // rectangle painted on its face. Boxes can't be boolean-subtracted, so the
  // only way to get an opening that still looks like an opening from an angle
  // is to leave one.
  const sideX = MOUTH_W / 2 + SIDE_W / 2;
  for (const s of [-1, 1]) {
    b.standing(SIDE_W, between(rng, 2.7, 3.1) * H, MASS_D, s * sideX, 0, pick(rng, ROCK));
    // A smaller mass set back and out, so the hill has a shoulder instead of
    // ending in a clean vertical edge.
    b.standing(
      between(rng, 1.2, 1.8) * H, between(rng, 1.4, 2.2) * H, between(rng, 1.6, 2.4) * H,
      s * (sideX + SIDE_W * 0.55), -between(rng, 0.2, 0.9) * H,
      pick(rng, ROCK), between(rng, -0.3, 0.3),
    );
  }

  // The lintel mass, bridging the gap, and a cap on top of that.
  const overW = MOUTH_W + SIDE_W * 2;
  b.add(overW, 1.5 * H, MASS_D, 0, MOUTH_H + 0.75 * H, 0, pick(rng, ROCK));
  b.add(
    between(rng, 3.4, 4.4) * H, between(rng, 0.9, 1.4) * H, between(rng, 2.2, 2.8) * H,
    between(rng, -0.5, 0.5) * H, MOUTH_H + 1.9 * H, -between(rng, 0.2, 0.6) * H,
    pick(rng, ROCK), between(rng, -0.4, 0.4),
  );

  // The dark. Set back inside the gap so the front of the tunnel is genuinely
  // empty space and you see INTO something, then blackness a couple of paces
  // in. A dark slab flush with the mouth would just look painted on.
  b.add(MOUTH_W, MOUTH_H, MASS_D * 0.55, 0, MOUTH_H / 2, -MASS_D * 0.22, CAVE_DARK);

  // Timber frame around the mouth: two posts and a lintel. This is what says
  // "mine" rather than "hole" -- the structure is worked, not natural.
  const postX = MOUTH_W / 2 + 0.13 * H;
  const frameZ = MASS_D / 2 - 0.25 * H;
  for (const s of [-1, 1]) {
    b.standing(0.26 * H, MOUTH_H, 0.3 * H, s * postX, frameZ, pick(rng, TIMBER));
    // A brace leaning in off each post.
    b.add(
      0.18 * H, 1.5 * H, 0.22 * H,
      s * (postX + 0.4 * H), MOUTH_H * 0.45, frameZ,
      pick(rng, TIMBER), 0, s * between(rng, 0.25, 0.45),
    );
  }
  b.add(MOUTH_W + 0.9 * H, 0.32 * H, 0.34 * H, 0, MOUTH_H + 0.16 * H, frameZ, pick(rng, TIMBER));

  // A track running out of the mouth: sleepers with two rails over them.
  // Cheap, and it points the eye at the entrance from across the board.
  const trackEnd = MASS_D / 2 + 2.8 * H;
  for (let z = frameZ + 0.5 * H; z < trackEnd; z += 0.55 * H) {
    b.standing(1.5 * H, 0.09 * H, 0.2 * H, between(rng, -0.05, 0.05) * H, z, pick(rng, TIMBER));
  }
  for (const s of [-1, 1]) {
    const from = frameZ, to = trackEnd;
    b.add(0.09 * H, 0.1 * H, to - from, s * 0.5 * H, 0.13 * H, (from + to) / 2, 0x3a3c42);
  }

  // Spoil heaped either side of the mouth, from digging it out.
  for (let i = 0; i < 7; i++) {
    const s = i % 2 ? 1 : -1;
    const w = between(rng, 0.3, 0.7) * H;
    b.standing(
      w, between(rng, 0.25, 0.6) * H, w * between(rng, 0.8, 1.2),
      s * between(rng, 1.1, 2.6) * H, between(rng, 1.9, 3.4) * H,
      pick(rng, ROCK), between(rng, 0, Math.PI),
    );
  }

  return b.build();
}

// ── Hallowed ground: a ruined church ───────────────────────────────────────

function buildRuins(rng: () => number): THREE.BufferGeometry {
  const b = new Blocks();

  const NAVE_HALF_W = 2.3 * H; // wall centerlines, either side of the aisle
  const WALL_T = 0.45 * H; // same thickness as a castle wall, deliberately
  const FRONT_Z = 3.6 * H, BACK_Z = -3.6 * H;

  // The side walls are a RUN OF SEGMENTS at varying heights, with gaps where
  // the wall has come down entirely. That's the whole trick to making a wreck
  // instead of a building: the outline is broken and uneven, so the eye reads
  // collapse rather than a low wall someone built on purpose.
  const SEGMENTS = 9;
  const segD = (FRONT_Z - BACK_Z) / SEGMENTS;
  for (const s of [-1, 1]) {
    for (let i = 0; i < SEGMENTS; i++) {
      // Fully collapsed here -- an opening, not a low bit.
      if (rng() < 0.22) continue;
      const z = BACK_Z + segD * (i + 0.5);
      // Taller toward the back, which is the part of a church that survives:
      // it's the heaviest masonry and it isn't holding up a tower.
      const lean = 1 - (z - BACK_Z) / (FRONT_Z - BACK_Z);
      const h = between(rng, 0.5, 1.4) * H + lean * between(rng, 0.6, 1.7) * H;
      b.standing(WALL_T, h, segD * 0.98, s * NAVE_HALF_W, z, pick(rng, RUIN_STONE));
      // A stub of a fallen course lying beside the wall.
      if (rng() < 0.3) {
        b.standing(
          between(rng, 0.4, 0.8) * H, between(rng, 0.15, 0.35) * H, between(rng, 0.4, 0.9) * H,
          s * (NAVE_HALF_W + between(rng, 0.5, 1.1) * H), z,
          pick(rng, RUIN_STONE), between(rng, 0, Math.PI),
        );
      }
    }
  }

  // Back wall, standing highest, with a window left in it: two piers and a
  // lintel. The gap is the window -- again a real hole rather than a dark
  // patch, so daylight shows through from the far side.
  const backH = between(rng, 3.4, 4.2) * H;
  const winW = 1.2 * H, winBase = 1.3 * H, winTop = backH - 0.5 * H;
  for (const s of [-1, 1]) {
    const pierW = NAVE_HALF_W - winW / 2;
    b.standing(pierW, backH * between(rng, 0.85, 1), WALL_T, s * (NAVE_HALF_W - pierW / 2 + 0.05 * H), BACK_Z, pick(rng, RUIN_STONE));
  }
  b.standing(winW + 0.2 * H, winBase, WALL_T, 0, BACK_Z, pick(rng, RUIN_STONE));
  b.add(winW + 0.5 * H, backH - winTop + 0.3 * H, WALL_T, 0, winTop, BACK_Z, pick(rng, RUIN_STONE));

  // The tower, at the front corner, snapped off partway up: three shafts of
  // decreasing footprint, the last one offset and short, so the top reads as
  // broken rather than finished.
  const towerW = 2.1 * H;
  const tx = -(NAVE_HALF_W - towerW / 2 + 0.3 * H), tz = FRONT_Z - towerW / 2;
  const shaft1 = between(rng, 2.6, 3.2) * H;
  b.standing(towerW, shaft1, towerW, tx, tz, pick(rng, RUIN_STONE));
  const shaft2 = between(rng, 1.2, 1.9) * H;
  b.add(towerW * 0.92, shaft2, towerW * 0.92, tx, shaft1 + shaft2 / 2, tz, pick(rng, RUIN_STONE));
  // The snapped-off remnant: half the width, shoved to one side.
  const stubH = between(rng, 0.5, 1.1) * H;
  b.add(
    towerW * 0.5, stubH, towerW * 0.8,
    tx + towerW * between(rng, -0.25, 0.25), shaft1 + shaft2 + stubH / 2, tz,
    pick(rng, RUIN_STONE),
  );

  // A cross still up there, leaning. The lean is the point: upright reads as
  // maintained, tilted reads as abandoned.
  const crossBase = shaft1 + shaft2 + stubH;
  const tilt = between(rng, 0.12, 0.3) * (rng() < 0.5 ? -1 : 1);
  b.add(0.14 * H, 1.0 * H, 0.14 * H, tx, crossBase + 0.5 * H, tz, RUIN_STONE[2], 0, tilt);
  b.add(0.62 * H, 0.14 * H, 0.14 * H, tx, crossBase + 0.72 * H, tz, RUIN_STONE[2], 0, tilt);

  // Column stumps down the aisle -- what's left of the arcade. Broken to
  // different heights, because they didn't all go at once.
  for (let i = 0; i < 4; i++) {
    for (const s of [-1, 1]) {
      if (rng() < 0.25) continue;
      b.standing(
        0.42 * H, between(rng, 0.4, 1.7) * H, 0.42 * H,
        s * NAVE_HALF_W * 0.55, BACK_Z + 1.4 * H + i * 1.6 * H,
        pick(rng, RUIN_STONE),
      );
    }
  }

  // Rubble through the nave and spilling outside it.
  for (let i = 0; i < 14; i++) {
    const w = between(rng, 0.25, 0.7) * H;
    b.standing(
      w, between(rng, 0.15, 0.45) * H, w * between(rng, 0.7, 1.3),
      between(rng, -1, 1) * (NAVE_HALF_W + 1.2 * H),
      between(rng, BACK_Z - 1 * H, FRONT_Z + 1 * H),
      pick(rng, RUIN_STONE), between(rng, 0, Math.PI),
    );
  }

  return b.build();
}

// ── Placement ──────────────────────────────────────────────────────────────

const BUILDERS: Partial<Record<Terrain, (rng: () => number) => THREE.BufferGeometry>> = {
  field: buildShrubbery,
  cave: buildMine,
  hallowed: buildRuins,
};

/** Fraction of a patch's radius a structure is allowed to span. The patch edge
 *  is lobed and pulls in to 0.8 of the radius in places, so staying well
 *  inside that is what keeps a ruin from poking out onto the forest. */
const FIT_FRACTION = 0.62;
/** How far off a patch's center a structure may sit, as a fraction of the
 *  radius. Small: "roughly centered" is the look, and a landmark shoved to one
 *  side stops reading as belonging to its patch. */
const OFFSET_FRACTION = 0.15;

/** Everything built, in one Group ready to be added to the scene. Not added
 *  here -- see the note at the top about staying free of scene.ts. */
export const landmarks = new THREE.Group();

/** The structures actually placed, alongside the patch each belongs to. */
export const placed: { patch: Patch; mesh: THREE.Mesh }[] = [];

{
  const rng = makeRng(LANDMARK_SEED);

  for (const patch of patches) {
    const build = BUILDERS[patch.terrain];
    if (!build) continue;

    const geometry = build(rng);
    const mesh = new THREE.Mesh(geometry, sceneryMaterial);

    // Scale is read back off the geometry's own bounds rather than from a
    // per-builder constant. A hand-maintained footprint number is exactly the
    // kind of fact that silently stops matching the moment a builder grows a
    // wing, and being wrong here means scenery hanging off its patch.
    const box = geometry.boundingBox!;
    const reach = Math.max(-box.min.x, box.max.x, -box.min.z, box.max.z);
    const allowed = patch.radius * FIT_FRACTION;
    // Only ever shrinks: structures are modelled at the size they should read
    // at, and a big patch shouldn't get a giant church.
    const fit = Math.min(1, allowed / reach) * between(rng, 0.9, 1.05);
    mesh.scale.setScalar(fit);

    const angle = rng() * Math.PI * 2;
    const dist = rng() * patch.radius * OFFSET_FRACTION;
    mesh.position.set(patch.x + Math.sin(angle) * dist, 0, patch.z + Math.cos(angle) * dist);
    // Free rotation: none of these has a front that needs to face anything, and
    // four mines all square to the board's axes would look placed by a level
    // editor.
    mesh.rotation.y = rng() * Math.PI * 2;

    landmarks.add(mesh);
    placed.push({ patch, mesh });
  }
}
