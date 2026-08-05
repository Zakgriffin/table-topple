// What each board cell is made of.
//
// Today terrain does exactly one thing -- tints the floor (board.ts) -- but the
// per-cell grid is the actual product, not the colour. It's a fact any future
// rule can ask about ("does the hallowed ground heal?", "does the cave block
// line of sight?") without that rule having to know how the board is drawn.
// That's why this module holds data and a query and imports nothing from three
// or scene.ts: like frame.ts and motion.ts, it stays directly runnable under
// node for checking.
//
// The layout is generated rather than authored, from a fixed seed, so the board
// is the same every reload and a headless check sees what the page sees. Change
// TERRAIN_SEED to get a different board.

import { BOARD_CELLS, BOARD_SIZE, START_RADIUS } from './constants.ts';
import { makeRng } from './noise.ts';

export type Terrain = 'forest' | 'field' | 'cave' | 'hallowed';

/** Storage order for the grid: an index into this array is what's actually
 *  stored per cell, so the grid is a byte per cell rather than a string. */
const TERRAINS: Terrain[] = ['forest', 'field', 'cave', 'hallowed'];

/** The board's default -- everything the patches below don't claim. Index 0,
 *  which is what a freshly zeroed grid already is. */
export const BASE_TERRAIN: Terrain = 'forest';

/** The scattered terrains, and how many patches of each. */
export const PATCH_TERRAINS: Terrain[] = ['field', 'cave', 'hallowed'];
export const PATCHES_PER_TERRAIN = 5;

// ── Patch shape and placement ──────────────────────────────────────────────

const TERRAIN_SEED = 0x5eed1;

/** Patch size, in cells. Big enough to be a place you walk across rather than
 *  a tile you step over; small enough that 15 of them still leave forest as
 *  the board's character. At these radii the patches take up about a seventh
 *  of the board. */
const PATCH_MIN_RADIUS = 6;
const PATCH_MAX_RADIUS = 11;

/** Cells of forest guaranteed between any two patches, so two of them never
 *  touch and read as one odd bicoloured blob. */
const PATCH_SEPARATION = 3;

/** How far a patch's edge deviates from a circle, as a fraction of its radius.
 *  Purely cosmetic: a field of exact circles reads as a bug. */
const PATCH_WOBBLE = 0.2;
/** Lobes around the patch's edge. Odd, so opposite sides don't mirror. */
const PATCH_LOBES = 3;

/** Half-width of the area patches may occupy, in world units -- about halfway
 *  from the courts' standing radius out to the rim. Patches reach well past
 *  the courts now, mostly filling in the four corners, but stop short of the
 *  board's actual edge: a patch running off the side reads as the board having
 *  been cropped rather than as terrain. */
const PLAYFIELD_HALF = BOARD_SIZE / 2 - 8.5;

/** Boxes no patch may touch: each court's own ground, from a little in front
 *  of its line back to the board edge behind its castle. Now that patches
 *  reach out this far, this is what keeps one from appearing inside a castle
 *  or under a starting formation -- terrain under a building nobody can walk
 *  into is just a colour mistake. */
interface KeepOut { minX: number; maxX: number; minZ: number; maxZ: number }

/** Half the width of a court's ground: its line is 8.8 either side of center
 *  (FORMATION spans 8 gaps of RANK_SPACING), plus room for the figures
 *  themselves. */
const COURT_HALF_WIDTH = 11;
/** Open ground left in FRONT of a line before terrain may start. */
const COURT_FRONT_CLEARANCE = 6;
/** Breathing room between a patch's edge and a court box. */
const COURT_GAP = 2;

const courtKeepOuts: KeepOut[] = (() => {
  const rim = BOARD_SIZE / 2;
  const front = START_RADIUS - COURT_FRONT_CLEARANCE;
  const w = COURT_HALF_WIDTH;
  return [
    { minX: -w, maxX: w, minZ: -rim, maxZ: -front }, // red, at -Z
    { minX: -w, maxX: w, minZ: front, maxZ: rim }, // blue, at +Z
    { minX: -rim, maxX: -front, minZ: -w, maxZ: w }, // green, at -X
    { minX: front, maxX: rim, minZ: -w, maxZ: w }, // yellow, at +X
  ];
})();

/** Closest-point test: clamp the center into the box and measure. Cheaper and
 *  less error-prone than case-splitting on which side of the box the disk is. */
function reaches(x: number, z: number, radius: number, box: KeepOut): boolean {
  const nx = Math.min(Math.max(x, box.minX), box.maxX);
  const nz = Math.min(Math.max(z, box.minZ), box.maxZ);
  return (x - nx) ** 2 + (z - nz) ** 2 < radius * radius;
}

/** Whether a disk of this radius touches any court's ground. Exported because
 *  patches aren't the only thing that has to stay off it -- scenery scattered
 *  over the open board (forest.ts) needs the same answer, and there should be
 *  exactly one description of where the courts stand. */
export function onCourtGround(x: number, z: number, radius = 0): boolean {
  return courtKeepOuts.some((box) => reaches(x, z, radius, box));
}

/** Placement is rejection sampling; this caps it. Generous -- the geometry
 *  above places 15 patches with room to spare, so exhausting this means the
 *  constants were changed into something that doesn't fit, which is a bug to
 *  be told about rather than a board to render half-populated. */
const MAX_PLACEMENT_ATTEMPTS = 4000;

export interface Patch {
  terrain: Terrain;
  /** Center in world units. */
  x: number;
  z: number;
  /** Maximum reach from the center, world units. The wobble only ever pulls
   *  the edge inward from this, so it doubles as the separation radius. */
  radius: number;
  /** Rotation of the lobe pattern, radians. */
  phase: number;
}

/** The patches that were placed, in placement order. Exported for checking and
 *  for anything that later wants to talk about a patch as a whole ("the cave
 *  nearest the red castle") rather than cell by cell. */
export const patches: Patch[] = [];

function layOutPatches() {
  const rng = makeRng(TERRAIN_SEED);

  // Round-robin rather than all five fields, then all five caves: placement
  // rejects against what's already down, so going type by type would hand the
  // roomiest board to whichever type happened to go first.
  const order: Terrain[] = [];
  for (let i = 0; i < PATCHES_PER_TERRAIN; i++) order.push(...PATCH_TERRAINS);

  for (const terrain of order) {
    const radius = PATCH_MIN_RADIUS + rng() * (PATCH_MAX_RADIUS - PATCH_MIN_RADIUS);
    // Kept a full radius inside the playfield so no patch is clipped by its
    // boundary -- a patch cut off in a straight line looks like a mistake.
    const span = PLAYFIELD_HALF - radius;

    let placed = false;
    for (let attempt = 0; attempt < MAX_PLACEMENT_ATTEMPTS && !placed; attempt++) {
      const x = (rng() * 2 - 1) * span;
      const z = (rng() * 2 - 1) * span;
      const clear =
        !courtKeepOuts.some((box) => reaches(x, z, radius + COURT_GAP, box)) &&
        patches.every((p) => {
          const need = p.radius + radius + PATCH_SEPARATION;
          return (p.x - x) ** 2 + (p.z - z) ** 2 >= need * need;
        });
      if (clear) {
        patches.push({ terrain, x, z, radius, phase: rng() * Math.PI * 2 });
        placed = true;
      }
    }

    if (!placed) {
      throw new Error(
        `terrain: could not place a ${terrain} patch in ${MAX_PLACEMENT_ATTEMPTS} attempts ` +
        `(${patches.length} of ${order.length} placed) -- the patch radii or the playfield are ` +
        'no longer compatible',
      );
    }
  }
}

// ── The grid ───────────────────────────────────────────────────────────────

/** One byte per cell, indexed row-major as row * BOARD_CELLS + col, holding an
 *  index into TERRAINS. Zero-filled, which is already every cell forest. */
export const terrainGrid = new Uint8Array(BOARD_CELLS * BOARD_CELLS);

function paintPatches() {
  for (const p of patches) {
    // Only the cells the patch could possibly reach get tested.
    const [colLo, colHi] = cellRange(p.x, p.radius);
    const [rowLo, rowHi] = cellRange(p.z, p.radius);
    const value = TERRAINS.indexOf(p.terrain);

    for (let row = rowLo; row <= rowHi; row++) {
      for (let col = colLo; col <= colHi; col++) {
        const dx = cellCenter(col) - p.x;
        const dz = cellCenter(row) - p.z;
        // Lobed edge: the radius the patch actually reaches in this direction.
        const angle = Math.atan2(dz, dx);
        const reach = p.radius * (1 - PATCH_WOBBLE + PATCH_WOBBLE * Math.sin(PATCH_LOBES * angle + p.phase));
        if (dx * dx + dz * dz <= reach * reach) {
          terrainGrid[row * BOARD_CELLS + col] = value;
        }
      }
    }
  }
}

/** Inclusive cell range covering [center - radius, center + radius] on one
 *  axis, clamped to the board. */
function cellRange(center: number, radius: number): [number, number] {
  return [
    Math.max(0, worldToCell(center - radius)),
    Math.min(BOARD_CELLS - 1, worldToCell(center + radius)),
  ];
}

// One world unit is one cell (constants.ts keeps that true on purpose), so
// both conversions below are a shift by half the board and a floor -- no scale
// factor. If that invariant ever breaks, these two functions are the only
// places that need to learn about it.
export function worldToCell(w: number): number {
  return Math.floor(w + BOARD_SIZE / 2);
}
export function cellCenter(cell: number): number {
  return cell + 0.5 - BOARD_SIZE / 2;
}

layOutPatches();
paintPatches();

// ── Queries ────────────────────────────────────────────────────────────────

/** The terrain of a cell. Out-of-range cells read as the base terrain rather
 *  than throwing: callers ask about world positions that can drift off the
 *  board edge, and "forest" is a truthful answer for the void beyond it. */
export function terrainAtCell(col: number, row: number): Terrain {
  if (col < 0 || col >= BOARD_CELLS || row < 0 || row >= BOARD_CELLS) return BASE_TERRAIN;
  return TERRAINS[terrainGrid[row * BOARD_CELLS + col]];
}

/** The terrain under a world position. `worldZ` is the board's z axis, which
 *  is the grid's row axis -- denizens store position as a Vector2 of (x, z),
 *  so this takes them in that same order. */
export function terrainAt(worldX: number, worldZ: number): Terrain {
  return terrainAtCell(worldToCell(worldX), worldToCell(worldZ));
}
