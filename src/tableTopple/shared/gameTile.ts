import * as THREE from 'three';
import { TILE_CELLS } from './constants.ts';
import { cellCenter, worldToCell } from './terrain.ts';

// The board's "game tiles" -- TILE_CELLS-wide grid squares, given a name and
// a home of their own because they now drive real rules, not just a look.
// Several things already agreed on this grid before it had a shared module:
// terrain patches snap their centers onto its line intersections
// (terrain.ts), the floor checkerboards it (board.ts), a denizen's act-mode
// reach is measured in it (act.ts), and a road may only span so many of them
// (roads.ts). This is the one place that converts between world space and
// that grid, so none of them can quietly disagree about where its lines
// fall -- built on terrain.ts's own worldToCell/cellCenter (one cell is one
// world unit; a tile is TILE_CELLS of those) rather than re-deriving the
// board's half-extent a second time.
//
// Depends on terrain.ts, which must never depend back: terrain.ts is the
// zero-dependency, node-runnable root every board-space module already
// builds on (see its own header), and this module pulls in THREE for the
// geometry builder at the bottom, which terrain.ts deliberately does not.

export interface GameTile {
  col: number;
  row: number;
}

/** Which tile index a cell index falls in, one axis. */
export function tileIndexFromCell(cell: number): number {
  return Math.floor(cell / TILE_CELLS);
}

/** Which tile index a world coordinate falls in, one axis. */
export function tileIndex(w: number): number {
  return tileIndexFromCell(worldToCell(w));
}

/** The tile a world point falls in, both axes. `z` takes the board's z axis
 *  (a Vector2's `.y`), the same (x, z)-as-(x, y) convention terrain.ts's own
 *  terrainAt uses. */
export function tileAt(x: number, z: number): GameTile {
  return { col: tileIndex(x), row: tileIndex(z) };
}

/** A tile index's own MIN edge, in world units, one axis. */
export function tileOrigin(index: number): number {
  return cellCenter(index * TILE_CELLS) - 0.5;
}

export function sameTile(a: GameTile, b: GameTile): boolean {
  return a.col === b.col && a.row === b.row;
}

/** Tile-to-tile Manhattan distance between two world points -- standing
 *  anywhere in a tile is the same distance from anywhere in another tile,
 *  which is what makes reach (act.ts) a property of the GRID rather than of
 *  exactly where within a tile someone happens to be standing. */
export function tileManhattan(a: THREE.Vector2, b: THREE.Vector2): number {
  const ta = tileAt(a.x, a.y), tb = tileAt(b.x, b.y);
  return Math.abs(ta.col - tb.col) + Math.abs(ta.row - tb.row);
}

/** Every tile within `reach` Manhattan tile-steps of `center`, `center`
 *  itself included -- a blocky diamond made of whole tiles (act.ts's reach
 *  indicator), not a continuous circle, since reach is decided tile by tile,
 *  not by raw distance. */
export function tilesWithinReach(center: GameTile, reach: number): GameTile[] {
  const out: GameTile[] = [];
  for (let dCol = -reach; dCol <= reach; dCol++) {
    const maxRow = reach - Math.abs(dCol);
    for (let dRow = -maxRow; dRow <= maxRow; dRow++) {
      out.push({ col: center.col + dCol, row: center.row + dRow });
    }
  }
  return out;
}

/** Every tile whose square footprint overlaps a circle -- what a structure
 *  or tree "occupies" for highlighting purposes (act.ts). Tested by clamping
 *  the circle's center into each candidate tile's own box and comparing to
 *  its radius, the same closest-point-in-a-box trick terrain.ts's own
 *  onCourtGround uses for a disk-vs-box test. */
export function tilesInCircle(cx: number, cz: number, radius: number): GameTile[] {
  const colLo = tileIndex(cx - radius), colHi = tileIndex(cx + radius);
  const rowLo = tileIndex(cz - radius), rowHi = tileIndex(cz + radius);
  const out: GameTile[] = [];
  for (let row = rowLo; row <= rowHi; row++) {
    for (let col = colLo; col <= colHi; col++) {
      const minX = tileOrigin(col), maxX = minX + TILE_CELLS;
      const minZ = tileOrigin(row), maxZ = minZ + TILE_CELLS;
      const nx = Math.min(Math.max(cx, minX), maxX);
      const nz = Math.min(Math.max(cz, minZ), maxZ);
      if ((cx - nx) ** 2 + (cz - nz) ** 2 <= radius * radius) out.push({ col, row });
    }
  }
  return out;
}

/** A tile-grid walk from `a` to `b`, ORTHOGONAL steps only -- never both axes
 *  in the same step, so two consecutive tiles always share a full edge and
 *  never just a corner. True Manhattan distance apart (|dCol| + |dRow| + 1
 *  tiles), not Chebyshev: a diagonal reach would let a road cross the board
 *  corner-to-corner far cheaper than crossing it edge-to-edge, and would
 *  show up on screen as tiles touching only at a point. This is what a
 *  road's own reach is measured along (roads.ts): the tiles a drag can
 *  actually get to, not whatever distinct tiles a raw straight segment
 *  happens to graze (which could undercount a shallow diagonal's true tile
 *  distance and let a drag sneak an extra tile past its span limit).
 *
 *  Bresenham's own line algorithm, but with its diagonal shortcut removed
 *  (only ONE of the two branches ever fires per iteration, not both) --
 *  otherwise it's the same error-accumulation trick, which is what keeps a
 *  shallow diagonal an even staircase instead of every column step landing
 *  before any row step. */
export function tileWalk(a: GameTile, b: GameTile): GameTile[] {
  let col = a.col, row = a.row;
  const path: GameTile[] = [{ col, row }];
  const dCol = Math.abs(b.col - a.col), sCol = Math.sign(b.col - a.col);
  const dRow = -Math.abs(b.row - a.row), sRow = Math.sign(b.row - a.row);
  let err = dCol + dRow;
  while (col !== b.col || row !== b.row) {
    const e2 = 2 * err;
    if (e2 >= dRow && col !== b.col) { err += dRow; col += sCol; }
    else { err += dCol; row += sRow; }
    path.push({ col, row });
  }
  return path;
}

/**
 * Clamps `end` so it can never reach past the `maxTiles`th tile of the
 * guided walk (tileWalk, above) from the tile `start` sits in -- a road's
 * own span limit (roads.ts's MAX_ROAD_TILES), enforced on the GRID rather
 * than by distance along the raw drag segment. Returns both the clamped
 * world point (free to move anywhere inside that last reachable tile, not
 * snapped to its center) and the exact tile path to highlight, so a drag's
 * highlight and its clamp can never disagree about which tiles are actually
 * in reach.
 */
export function clampToTileWalk(
  start: THREE.Vector2, end: THREE.Vector2, maxTiles: number,
): { point: THREE.Vector2; tiles: GameTile[] } {
  const walk = tileWalk(tileAt(start.x, start.y), tileAt(end.x, end.y));
  if (walk.length <= maxTiles) return { point: end.clone(), tiles: walk };
  const tiles = walk.slice(0, maxTiles);
  const last = tiles[tiles.length - 1];
  const x = THREE.MathUtils.clamp(end.x, tileOrigin(last.col), tileOrigin(last.col) + TILE_CELLS);
  const z = THREE.MathUtils.clamp(end.y, tileOrigin(last.row), tileOrigin(last.row) + TILE_CELLS);
  return { point: new THREE.Vector2(x, z), tiles };
}

/**
 * Builds a flat mosaic of unit quads, one per tile, directly in WORLD space
 * (XY plane; rotate -90 about X to lay flat, the same convention board.ts's
 * own floor and act.ts's reach indicator use) -- the one shared visual every
 * "these tiles are relevant right now" highlight is built from (act.ts's
 * structure/tree hover, roads.ts's guided-walk highlight), so the effect
 * reads as the same thing everywhere it shows up rather than a different
 * look per caller.
 */
export function buildTileHighlightGeometry(tiles: readonly GameTile[]): THREE.BufferGeometry {
  const positions: number[] = [];
  for (const t of tiles) {
    const x0 = tileOrigin(t.col), x1 = x0 + TILE_CELLS;
    // Pre-negated: every caller lays this geometry flat with rotation.x =
    // -Math.PI/2 (board.ts's own floor/tileGrid convention), which maps
    // local Y to world -Z, not world Z. Writing world Z straight into local
    // Y here would land each tile mirrored across the board's own center
    // once that rotation applies -- negating it up front cancels the
    // rotation's own negation back out.
    const y0 = -tileOrigin(t.row), y1 = y0 - TILE_CELLS;
    positions.push(
      x0, y0, 0, x1, y0, 0, x1, y1, 0,
      x0, y0, 0, x1, y1, 0, x0, y1, 0,
    );
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  return geometry;
}
