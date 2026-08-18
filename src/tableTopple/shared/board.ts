import * as THREE from 'three';
import { scene } from './scene.ts';
import {
  BOARD_CELLS, BOARD_SIZE, COLOR_CAVE, COLOR_FIELD, COLOR_FOREST, COLOR_HALLOWED,
  // TILE_CELLS -- only used by the commented-out tileGrid below; uncomment together.
} from './constants.ts';
import { type Terrain, cellCenter, terrainAtCell, worldToCell } from './terrain.ts';
import { tileIndexFromCell } from './gameTile.ts';
import { fbm, noise2 } from './noise.ts';
import { landmarks } from './landmarks.ts';
import { forest } from './forest.ts';

// The playing surface: a single lit plane. No De Bruijn pattern here -- that
// belongs to the tracking work, and this page is a clean room for game
// mechanics.
//
// The floor is a MOSAIC: every cell is one flat colour, edge to edge. There is
// no grid helper over it any more -- the cells are legible from their own
// colours, and drawn lines on top of a textured floor read as a wireframe
// laid over terrain rather than as the terrain being made of tiles. That puts
// the whole burden of showing the grid on the per-cell variation below.
//
// What varies is which colour each cell gets. Two
// things move it off its terrain's base colour, and both work cell to cell
// rather than within a cell:
//
//   blending  a cell near a patch's edge mixes in its neighbours' terrain, so
//             the boundary steps across several tiles instead of switching in
//             one, while each tile stays flat.
//   texture   a brightness field over the board, so no two stretches of forest
//             are quite the same shade.
//
// All of that is APPEARANCE ONLY. The terrain grid is cells with hard edges and
// stays that way -- a denizen is standing on cave or it isn't, and a tile that
// looks half-cave is still wholly whatever terrain.ts says it is. Nothing
// should ever read a terrain back out of these colours.

const TERRAIN_COLOR: Record<Terrain, number> = {
  forest: COLOR_FOREST,
  field: COLOR_FIELD,
  cave: COLOR_CAVE,
  hallowed: COLOR_HALLOWED,
};

// ── Blending ───────────────────────────────────────────────────────────────

/** How many cells the blend reaches. 3 puts a patch edge across roughly five
 *  tiles -- a few paces of transition, enough to read as a border region
 *  rather than a cut line, while a patch's middle stays solidly its own
 *  colour. */
const BLEND_CELLS = 3;
/** Falloff of the blend weights, in cells. Well inside BLEND_CELLS so the
 *  kernel has faded to almost nothing by its rim; a kernel still heavy at its
 *  edge just moves the hard step outward instead of removing it. */
const BLEND_SIGMA = 1.5;

/** The blend kernel: one entry per neighbour cell offset, with its weight.
 *  Built once -- it's the same for all 20,736 cells, and recomputing an exp()
 *  per sample per cell was a million calls for 49 distinct answers. */
const kernel: { dcol: number; drow: number; w: number }[] = [];
for (let drow = -BLEND_CELLS; drow <= BLEND_CELLS; drow++) {
  for (let dcol = -BLEND_CELLS; dcol <= BLEND_CELLS; dcol++) {
    kernel.push({ dcol, drow, w: Math.exp(-(dcol * dcol + drow * drow) / (2 * BLEND_SIGMA * BLEND_SIGMA)) });
  }
}

// ── Texture ────────────────────────────────────────────────────────────────

// Three layers at different sizes, rather than one field turned up: a single
// scale of noise cranked high reads as one repeating blotch pattern, while
// broad tone + mid detail + per-tile grain reads as ground. Peak swings add to
// well over half, but the three fields are independent, so all three peaking
// on the same cell is rare -- the typical cell lands nearer ±20%.

/** Broad sweeps of lighter and darker ground, several patches wide. */
const BROAD_SCALE = 52;
const BROAD_AMOUNT = 0.2;
/** Mid-size mottling -- the size of a clearing. */
const TEXTURE_SCALE = 17;
const TEXTURE_AMOUNT = 0.24;
/** A per-cell jitter, uncorrelated between neighbours: what makes the mosaic
 *  read as individual tiles rather than a smooth wash. This one is doing more
 *  work now that the grid lines are gone -- tile edges have to come from the
 *  colour itself. */
const GRAIN_AMOUNT = 0.11;
const BROAD_SEED = 0x3f11;
const TEXTURE_SEED = 0x7e44;
const GRAIN_SEED = 0x1d0b;

// ── The mesh ───────────────────────────────────────────────────────────────

// One quad per board cell, NOT indexed. An indexed plane shares each interior
// vertex between the four cells meeting there, which forces one colour for all
// four and smears every cell into its neighbours. Split, each triangle owns its
// vertices, so a cell can be flat and its edges land exactly on the grid lines.
// The cost is ~118k vertices for a 140x140 board, built once at load.
const geometry = new THREE.PlaneGeometry(BOARD_SIZE, BOARD_SIZE, BOARD_CELLS, BOARD_CELLS).toNonIndexed();
const position = geometry.getAttribute('position');
const colors = new THREE.BufferAttribute(new Float32Array(position.count * 3), 3);
geometry.setAttribute('color', colors);

export const floor = new THREE.Mesh(
  geometry,
  // White base: vertexColors multiplies the material colour, so anything other
  // than white here would tint every terrain at once. Flat shading, because a
  // mosaic wants its lighting constant across a face too, not just its colour.
  new THREE.MeshStandardMaterial({
    color: 0xffffff, roughness: 0.95, metalness: 0, vertexColors: true, flatShading: true,
  }),
);
floor.rotation.x = -Math.PI / 2;
scene.add(floor);

/** Every terrain's colour in the LINEAR working space, decoded once.
 *  Blending has to happen in linear: averaging sRGB values directly darkens
 *  the midpoint of every transition and would ring each patch with a dirty
 *  seam. THREE.Color's hex setter is what does the decode, and it's the same
 *  conversion a material's own `color` gets -- writing raw hex bytes into a
 *  colour attribute instead renders everything washed out next to the rest of
 *  the scene. */
const linear = new Map<Terrain, [number, number, number]>();
{
  const c = new THREE.Color();
  for (const [terrain, hex] of Object.entries(TERRAIN_COLOR) as [Terrain, number][]) {
    c.setHex(hex);
    linear.set(terrain, [c.r, c.g, c.b]);
  }
}

// Pulled halfway toward forest, in linear space, so a patch keeps its own
// hue up close but no longer competes with the forest surrounding it from
// across the board -- the constants above (a field's, a cave's, hallowed
// ground's own colour) are untouched; only what actually gets painted is
// softened. Forest itself is left alone -- there's nothing to blend it
// toward.
const FOREST_BLEND = 0.5;
{
  const [fr, fg, fb] = linear.get('forest')!;
  for (const terrain of Object.keys(TERRAIN_COLOR) as Terrain[]) {
    if (terrain === 'forest') continue;
    const [r, g, b] = linear.get(terrain)!;
    linear.set(terrain, [
      r + (fr - r) * FOREST_BLEND,
      g + (fg - g) * FOREST_BLEND,
      b + (fb - b) * FOREST_BLEND,
    ]);
  }
}

// ── Game-tile checkerboard ──────────────────────────────────────────────────
// A darker shade on alternating game tiles (gameTile.ts) -- the same grid the
// (visually faint) tileGrid lines below mark out, so the checker's own edges
// double as a second, colour-based way to read "N tiles that way" without
// leaning on the lines at all. Applied last and per-cell, with no blend
// kernel of its own: the patch-edge blending above exists to soften a
// TERRAIN change, but a tile boundary isn't one -- it should read as a crisp
// square, landing exactly on the grid line, not a smeared gradient.
/** How much darker the alternate tiles are. Multiplicative, so it darkens
 *  whatever colour is already there -- a patch's own hue included -- rather
 *  than mixing in a second one. */
const TILE_DARK_FACTOR = 0.72;
function tileShade(col: number, row: number): number {
  const tileCol = tileIndexFromCell(col);
  const tileRow = tileIndexFromCell(row);
  return (tileCol + tileRow) % 2 === 0 ? 1 : TILE_DARK_FACTOR;
}

/** One flat colour per cell, row-major like the terrain grid. */
const cellColor = new Float32Array(BOARD_CELLS * BOARD_CELLS * 3);

function computeCellColors() {
  for (let row = 0; row < BOARD_CELLS; row++) {
    for (let col = 0; col < BOARD_CELLS; col++) {
      let r = 0, g = 0, b = 0, weight = 0;
      for (const k of kernel) {
        // Off-board neighbours read as the base terrain (terrain.ts), so cells
        // along the rim blend against forest rather than against nothing --
        // no bright fringe around the board's edge.
        const [sr, sg, sb] = linear.get(terrainAtCell(col + k.dcol, row + k.drow))!;
        r += sr * k.w; g += sg * k.w; b += sb * k.w;
        weight += k.w;
      }

      // Sampled at the cell's center, once, and applied to the whole cell --
      // this is what keeps the tile flat. Sampling per vertex or per triangle
      // instead would put a gradient inside every cell and, worse, a visible
      // seam along the diagonal where its two triangles disagree.
      const x = cellCenter(col), z = cellCenter(row);
      // fbm and noise2 both return ~[0, 1); centred so the texture brightens as
      // often as it darkens and the board's overall value stays where the
      // palette put it.
      const broad = (fbm(x, z, BROAD_SCALE, 2, BROAD_SEED) - 0.5) * 2 * BROAD_AMOUNT;
      const smooth = (fbm(x, z, TEXTURE_SCALE, 3, TEXTURE_SEED) - 0.5) * 2 * TEXTURE_AMOUNT;
      // Sampled at whole-cell coordinates, so every cell lands exactly on a
      // lattice point of the noise and gets an independent value -- the one
      // place fine detail is wanted rather than smoothed away.
      const grain = (noise2(col, row, GRAIN_SEED) - 0.5) * 2 * GRAIN_AMOUNT;
      // Normalizing by the kernel's total weight and applying the brightness
      // are one multiply, since both are just a scale on the accumulated
      // colour. Floored: the swings are large enough now that three dark peaks
      // stacking could otherwise drive a cell to black, which would read as a
      // hole in the board rather than as shadow.
      const scale = (Math.max(0.35, 1 + broad + smooth + grain) / weight) * tileShade(col, row);

      const i = (row * BOARD_CELLS + col) * 3;
      cellColor[i] = r * scale;
      cellColor[i + 1] = g * scale;
      cellColor[i + 2] = b * scale;
    }
  }
}

/** Re-reads the terrain grid and recolours the floor. Called once at load;
 *  exported so terrain that changes during a game (a field burned to ash, a
 *  cave collapsed) has one thing to call rather than rebuilding the mesh. */
export function paintFloor() {
  computeCellColors();

  // The plane is built in its own space and then rotated flat, so a vertex's
  // (x, y) is not the board's (x, z). Rather than re-deriving that mapping by
  // hand -- the exact kind of orientation convention that's easy to get
  // mirrored, and invisible once wrong -- each triangle is pushed through the
  // mesh's own world matrix. Whatever transform the mesh actually has is the
  // transform the colours are placed with.
  floor.updateMatrixWorld();
  const centroid = new THREE.Vector3();

  for (let tri = 0; tri < position.count; tri += 3) {
    // A triangle is looked up by which CELL its centroid falls in, not by the
    // centroid's own position: the two triangles of a quad have different
    // centroids but the same cell, which is what guarantees they come out the
    // same colour and the quad reads as one tile.
    centroid.set(0, 0, 0);
    for (let v = 0; v < 3; v++) {
      centroid.x += position.getX(tri + v) / 3;
      centroid.y += position.getY(tri + v) / 3;
      centroid.z += position.getZ(tri + v) / 3;
    }
    centroid.applyMatrix4(floor.matrixWorld);

    const col = worldToCell(centroid.x), row = worldToCell(centroid.z);
    const i = (row * BOARD_CELLS + col) * 3;
    for (let v = 0; v < 3; v++) colors.setXYZ(tri + v, cellColor[i], cellColor[i + 1], cellColor[i + 2]);
  }
  colors.needsUpdate = true;
}

paintFloor();

// ── Game-tile grid ───────────────────────────────────────────────────────────
// A much coarser grid than the one this file's own header above explains
// away: one line every TILE_CELLS cells (35 divisions across the full
// 140-cell board), not one per cell. At that spacing it reads as a reference
// grid laid over the mosaic -- somewhere to say "N tiles that way" -- rather
// than as a wireframe standing in for terrain colour, which is the per-cell
// case the header is about and doesn't apply here. terrain.ts snaps patches
// to this same grid's line intersections, and gameTile.ts's whole "game tile"
// concept is built on it, so it's the shared constant from constants.ts
// rather than a number local to this file. Kept faint now that the
// checkerboard above marks the same squares by colour -- the lines are a
// secondary cue, not the primary way the grid reads.
// Commented out rather than deleted -- not wanted on screen for now, but the
// checkerboard above still marks the same squares by colour, and this is the
// quickest way back if the lines turn out to be missed.
// const tileGrid = new THREE.GridHelper(BOARD_SIZE, BOARD_CELLS / TILE_CELLS, 0xffffff, 0xffffff);
// tileGrid.position.y = 0.006; // clear of the floor itself; well under the mode overlays' own 0.02+
// const tileGridMaterial = tileGrid.material as THREE.LineBasicMaterial;
// tileGridMaterial.transparent = true;
// tileGridMaterial.opacity = 0.08;
// tileGridMaterial.depthWrite = false;
// scene.add(tileGrid);

// Scenery. Both modules build and hand back a Group without touching the
// scene, the way castle.ts does; this is where they enter the world.
scene.add(landmarks); // one structure per terrain patch
scene.add(forest); // trees over the forest terrain
