// Grid decode pipeline — pure logic, no DOM/camera dependency, so it can run
// both in the browser (src/main.ts) and under Node for testing
// (scripts/test-decode.ts) against a known image.
//
// Stage 2: rotation + uniform scale (still assumes the mat is planar / camera
// faces it straight-on — no perspective/skew handling).
//   1. Convert a square crop to grayscale, then binarize it (global mean
//      threshold — the pattern is pure black/white, so this is robust as
//      long as lighting is roughly even).
//   2. Estimate the grid's rotation via a gradient-orientation histogram
//      (see estimateRotationRad) on that raw (still-rotated) crop. Edge
//      orientation alone only pins the angle down modulo 90 degrees (a
//      square grid looks the same rotated by any multiple of 90) — so this
//      gives 4 candidate full angles: theta, theta+90, theta+180, theta+270.
//   3. The caller derotates the ORIGINAL grayscale/RGBA crop by EACH of
//      those 4 candidate angles (browser: via canvas ctx.rotate + drawImage,
//      for smooth resampling; Node test: via manual resampling) into 4
//      separate "aligned" buffers, then re-binarizes each.
//   4. For each aligned buffer, find the cell pitch + phase via
//      autocorrelation of an "edge energy" profile (sum of adjacent-pixel
//      differences) — periodic peaks land at cell boundaries. Since the
//      buffer is now (assumed) axis-aligned, this is the same detector as
//      Stage 1, just running on rotation-corrected pixels.
//   5. Sample every fully-visible cell in each candidate into a grid, tile
//      into discrete, non-overlapping order x order patches, and decode
//      each patch's bits (see decodePatches — which internally also tries
//      all 4 *reading* orientations per candidate, since a wrong-by-90
//      derotation angle shifts tile boundaries too, not just content, so
//      reading-orientation and derotation-angle ambiguity must both be
//      resolved together).
//   6. Each patch's exact-match lookup gives a candidate SEED anchor
//      position — fast, narrows the search from "every torus position" to a
//      handful, but weak evidence on its own: with only 16 bits of key
//      space at order 4, 65535 of 65536 possible windows are valid, so a
//      single patch finding "a" match doesn't mean much. pickBestCandidate
//      scores every distinct seed by CORRELATING the entire sampled grid
//      (not just one patch) against the actual known pattern at the
//      position that seed implies (see scoreCorrelation), and keeps
//      whichever seed — and whichever of the 4 angle candidates — scores
//      highest. This tolerates individual misread bits gracefully (a
//      genuinely correct anchor stays close to 1.0 even with some noise)
//      while a wrong anchor sits close to 0.5 (uncorrelated with a random
//      binary pattern) — a much better-separated signal than checking
//      patches only agree with each other.

import type { Mesh } from './mesh.ts';

export interface GridDetection {
  px: number; py: number; // phase (px offset of first cell boundary)
  pitchX: number; pitchY: number;
}

// Finds the pitch (dominant period) of a 1D energy profile via autocorrelation.
//
// A sub-pixel (parabolic-interpolation) refinement was attempted here, on
// the theory that a real camera's cell size is essentially never an exact
// integer number of pixels, so rounding to the nearest integer lag should
// compound into visible drift over many cells. In practice it made the
// rotated-input synthetic test regress badly (300-trial pass rate dropped
// from ~99% to ~1%), including with a stricter validity threshold and with
// pre-smoothing the score profile — nearest-neighbor resampling during
// derotation introduces staircase aliasing that the parabolic fit reacts to.
// Reverted rather than ship something destabilized; worth retrying later
// with a test setup whose ground-truth pitch isn't coincidentally an exact
// integer (this test's source PNG is rendered at exactly 8px/cell, which
// hid whether the refinement was helping or hurting on real-world input).
export function findPitch(energy: Float64Array, minLag: number, maxLag: number): number {
  let bestLag = minLag, bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let score = 0;
    for (let x = 0; x + lag < energy.length; x++) score += energy[x] * energy[x + lag];
    if (score > bestScore) { bestScore = score; bestLag = lag; }
  }
  return bestLag;
}

// Finds the phase in [0, pitch) whose boundary positions (phase, phase+pitch,
// phase+2*pitch, ...) best align with peaks in the energy profile.
export function findPhase(energy: Float64Array, pitch: number): number {
  let bestPhase = 0, bestScore = -Infinity;
  for (let phase = 0; phase < pitch; phase++) {
    let score = 0;
    for (let x = phase; x < energy.length; x += pitch) score += energy[x];
    if (score > bestScore) { bestScore = score; bestPhase = phase; }
  }
  return bestPhase;
}

// Runs pitch/phase detection restricted to the sub-rectangle [x0,x1) x
// [y0,y1) of the buffer (still indexed in full-buffer coordinates in the
// result). detectGrid is just this called on the whole buffer.
function detectGridInRegion(bin: Uint8Array, w: number, x0: number, y0: number, x1: number, y1: number): GridDetection {
  const rw = x1 - x0, rh = y1 - y0;
  const colEnergy = new Float64Array(rw);
  for (let x = 1; x < rw; x++) {
    let e = 0;
    for (let y = y0; y < y1; y++) e += Math.abs(bin[y * w + (x0 + x)] - bin[y * w + (x0 + x - 1)]);
    colEnergy[x] = e;
  }
  const rowEnergy = new Float64Array(rh);
  for (let y = 1; y < rh; y++) {
    let e = 0;
    for (let x = x0; x < x1; x++) e += Math.abs(bin[(y0 + y) * w + x] - bin[(y0 + y - 1) * w + x]);
    rowEnergy[y] = e;
  }

  const minLag = 4, maxLagX = Math.floor(rw / 4), maxLagY = Math.floor(rh / 4);
  const pitchX = findPitch(colEnergy, minLag, maxLagX);
  const pitchY = findPitch(rowEnergy, minLag, maxLagY);
  let px = findPhase(colEnergy, pitchX) + x0;
  let py = findPhase(rowEnergy, pitchY) + y0;

  // Re-anchor the phase to the boundary nearest the REGION's center, rather
  // than leaving it wherever findPhase's [0, pitch) search happened to land
  // it (always near the region's own index 0). Cell positions are
  // extrapolated outward from this anchor (see sampleFullGrid), so any
  // residual pitch error compounds with distance from it — anchoring near
  // one edge means the far edge carries the full accumulated error;
  // anchoring near the center means every visible edge is at most half that
  // distance away, roughly halving the worst-case drift and making it
  // symmetric instead of one-sided.
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  px += pitchX * Math.round((cx - px) / pitchX);
  py += pitchY * Math.round((cy - py) / pitchY);

  return { px, py, pitchX, pitchY };
}

export function detectGrid(bin: Uint8Array, w: number, h: number): GridDetection {
  return detectGridInRegion(bin, w, 0, 0, w, h);
}

// Estimates the grid's rotation, folded into [0, PI/2) radians, via a
// gradient-orientation histogram. Edges at cell boundaries produce gradient
// vectors perpendicular to the boundary line; a square grid's two line
// families (rows, columns) are perpendicular to each other, so their
// gradient directions are too — folding all edge orientations modulo 90
// degrees collapses both families onto the same histogram peak, directly
// giving the grid's rotation mod 90 in one shot (weighted by edge strength
// so faint/noisy gradients don't skew the estimate).
//
// Operates on blurred GRAYSCALE, not the binarized image: a rotated hard
// edge in an already-thresholded binary image becomes a staircase, whose
// micro-edges are biased toward 0/90 degrees rather than the true diagonal —
// same reason edge detectors like Canny blur before computing gradients.
function boxBlur(gray: Float64Array, w: number, h: number, radius: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          sum += gray[yy * w + xx];
          count++;
        }
      }
      out[y * w + x] = sum / count;
    }
  }
  return out;
}

// Uses a circular mean rather than a histogram peak search: orientation data
// is periodic mod PI/2 (both edge families and the +-direction ambiguity of
// each gradient all collapse together), so scaling each angle by 4 maps that
// periodicity onto the full circle (4 * PI/2 = 2*PI). Averaging as vectors in
// that scaled space, then dividing the result back down by 4, gives a
// continuous estimate with no bin-quantization error — a histogram's ~0.5
// degree bin width was enough residual error to visibly drift pixel sampling
// across a couple hundred pixels of buffer.
export function estimateRotationRad(gray: Float64Array, w: number, h: number): number {
  const blurred = boxBlur(gray, w, h, 2);
  let sumX = 0, sumY = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx = blurred[y * w + x + 1] - blurred[y * w + x - 1];
      const gy = blurred[(y + 1) * w + x] - blurred[(y - 1) * w + x];
      const mag = gx * gx + gy * gy; // squared magnitude — emphasizes strong edges
      if (mag < 1) continue;
      const angle = Math.atan2(gy, gx); // -PI..PI
      sumX += mag * Math.cos(4 * angle);
      sumY += mag * Math.sin(4 * angle);
    }
  }
  const meanAngle4 = Math.atan2(sumY, sumX); // -PI..PI
  let theta = meanAngle4 / 4; // -PI/4..PI/4
  if (theta < 0) theta += Math.PI / 2;
  return theta; // [0, PI/2)
}

// Reinterprets an estimateRotationRad() result as a signed correction near 0
// rather than an unsigned angle in [0, PI/2) — for refining a buffer that's
// already nearly aligned, where the true residual is expected to be small
// but wraps to near PI/2 if it was actually a small negative correction.
// Lets a caller derotate once with a coarse estimate, then call
// estimateRotationRad again on that mostly-aligned result to correct
// residual error (which grows with distance from the rotation pivot, so it
// matters more the larger the capture region is) — reusing the same
// estimator rather than adding a second one.
export function asSignedResidual(angle: number): number {
  return angle > Math.PI / 4 ? angle - Math.PI / 2 : angle;
}

// valid is false when a cell couldn't actually be sampled — only relevant
// for mesh-based sampling (see sampleFromMesh), where gaps exist because not
// every grid lattice point is 2D-localizable (see src/cornerdetect.ts); x/y/
// bit are meaningless placeholders in that case. sampleFullGrid's constant-
// pitch model has no gaps, so valid is always true there.
//
// cornerCount is how many of the cell's 4 mesh corners were actually known
// when sampleFromMesh estimated its center (2, 3, or 4 — see its diagonal-
// pair comment for why 2 is enough) — a rough per-cell confidence signal,
// mainly for debug display (src/main.ts's overlay). 0 for sampleFullGrid,
// which isn't corner-based at all.
export interface SampledCell { x: number; y: number; bit: number; valid: boolean; cornerCount: number; }

// Walks outward from `anchor` in both directions at a constant `pitch`,
// accumulating cell boundary positions.
export function buildBoundaries(anchor: number, pitch: number, minPos: number, maxPos: number): { boundaries: number[]; anchorIndex: number } {
  const right: number[] = [anchor];
  let x = anchor;
  while (x < maxPos) { x += pitch; right.push(x); }
  const left: number[] = [];
  x = anchor;
  while (x > minPos) { x -= pitch; left.unshift(x); }
  return { boundaries: [...left, ...right], anchorIndex: left.length };
}

// Samples every fully-visible cell in the (assumed axis-aligned) buffer into
// a 2D grid, cells[row][col], row 0 = top. Extends in BOTH directions from
// (px, py) — not just rightward/downward — since detectGrid re-anchors
// (px, py) near the buffer's center rather than near index 0. originRow/
// originCol give the array index of the cell whose top-left corner sits at
// exactly (px, py), for callers that need to map back to pixel positions
// (e.g. drawing overlay lines) without recomputing the anchor themselves.
export interface SampledGrid {
  rows: number; cols: number; cells: SampledCell[][];
  originRow: number; originCol: number;
}

export function sampleFullGrid(bin: Uint8Array, w: number, h: number, grid: GridDetection): SampledGrid {
  const { px, py, pitchX, pitchY } = grid;

  const { boundaries: xB, anchorIndex: colsLeft } = buildBoundaries(px, pitchX, 0, w);
  const { boundaries: yB, anchorIndex: rowsUp } = buildBoundaries(py, pitchY, 0, h);
  const cols = xB.length - 1, rows = yB.length - 1;

  const cells: SampledCell[][] = [];
  for (let i = 0; i < rows; i++) {
    const cy = (yB[i] + yB[i + 1]) / 2;
    const by = Math.max(2, Math.floor((yB[i + 1] - yB[i]) * 0.2));
    const rowCells: SampledCell[] = [];
    for (let j = 0; j < cols; j++) {
      const cx = (xB[j] + xB[j + 1]) / 2;
      const bx = Math.max(2, Math.floor((xB[j + 1] - xB[j]) * 0.2));
      let sum = 0, count = 0;
      for (let dy = -by; dy <= by; dy++) {
        const yy = Math.round(cy + dy);
        if (yy < 0 || yy >= h) continue;
        for (let dx = -bx; dx <= bx; dx++) {
          const xx = Math.round(cx + dx);
          if (xx < 0 || xx >= w) continue;
          sum += bin[yy * w + xx];
          count++;
        }
      }
      rowCells.push({ x: cx, y: cy, bit: count > 0 && sum / count > 0.5 ? 1 : 0, valid: true, cornerCount: 0 });
    }
    cells.push(rowCells);
  }
  return { rows, cols, cells, originRow: rowsUp, originCol: colsLeft };
}

// Samples cell-fill bits from a corner mesh (src/mesh.ts) instead of a
// constant-pitch grid — each cell (row,col) is bounded by the 4 mesh nodes
// at (row,col), (row,col+1), (row+1,col), (row+1,col+1), and its bit is
// sampled at the cell's actual detected center, so cell size and shape
// follow whatever local perspective distortion the mesh captured, not an
// assumed constant pitch.
//
// Only 5/8 of lattice points are ever directly detectable (see
// src/cornerdetect.ts), so requiring all 4 corners is a much stricter bar
// than it sounds: at a representative ~55% per-corner mesh-linking rate,
// P(all 4 present) is only ~9%, which measured out as true on real captures
// and left far too few valid cells for either decodePatches's tiling or
// findSlidingSeeds's exhaustive search to find enough complete windows.
//
// The fix uses a property of the (locally-approximated) parallelogram each
// cell forms: its center is the midpoint of EITHER diagonal, since both
// diagonals of a parallelogram bisect each other. So only one complete
// diagonal pair — {TL,BR} or {TR,BL} — is needed, not all 4 corners; a cell
// with exactly 3 known corners always has one complete diagonal pair
// automatically (whichever diagonal doesn't include the missing corner).
// At the same ~55% per-corner rate this raises P(valid) to ~51% (2p^2-p^4
// vs p^4) — order-of-magnitude more cells to work with. Genuinely
// under-determined cells (0-2 corners known, or 2 known but ADJACENT rather
// than diagonal) are still marked SampledCell.valid=false rather than
// guessed at with a weaker assumption.
export function sampleFromMesh(bin: Uint8Array, w: number, h: number, mesh: Mesh): SampledGrid {
  if (mesh.nodes.length === 0) return { rows: 0, cols: 0, cells: [], originRow: 0, originCol: 0 };

  let minRow = Infinity, maxRow = -Infinity, minCol = Infinity, maxCol = -Infinity;
  for (const n of mesh.nodes) {
    minRow = Math.min(minRow, n.row); maxRow = Math.max(maxRow, n.row);
    minCol = Math.min(minCol, n.col); maxCol = Math.max(maxCol, n.col);
  }
  const rows = maxRow - minRow, cols = maxCol - minCol; // one fewer cell than the node span, each direction
  if (rows <= 0 || cols <= 0) return { rows: 0, cols: 0, cells: [], originRow: 0, originCol: 0 };

  const cells: SampledCell[][] = [];
  for (let i = 0; i < rows; i++) {
    const gridRow = minRow + i;
    const rowCells: SampledCell[] = [];
    for (let j = 0; j < cols; j++) {
      const gridCol = minCol + j;
      const tl = mesh.byCoord.get(`${gridRow},${gridCol}`);
      const tr = mesh.byCoord.get(`${gridRow},${gridCol + 1}`);
      const bl = mesh.byCoord.get(`${gridRow + 1},${gridCol}`);
      const br = mesh.byCoord.get(`${gridRow + 1},${gridCol + 1}`);

      const diag1 = tl && br ? { ax: tl.x, ay: tl.y, bx: br.x, by: br.y } : null; // TL-BR
      const diag2 = tr && bl ? { ax: tr.x, ay: tr.y, bx: bl.x, by: bl.y } : null; // TR-BL
      if (!diag1 && !diag2) {
        rowCells.push({ x: NaN, y: NaN, bit: 0, valid: false, cornerCount: 0 });
        continue;
      }
      const cornerCount = [tl, tr, bl, br].filter(c => c !== undefined).length;
      // Average both diagonal midpoints when both happen to be known (all 4
      // corners present) for slightly better precision; otherwise use
      // whichever one is available.
      const mids = [diag1, diag2].filter((d): d is NonNullable<typeof d> => d !== null)
        .map(d => [(d.ax + d.bx) / 2, (d.ay + d.by) / 2]);
      const cx = mids.reduce((s, m) => s + m[0], 0) / mids.length;
      const cy = mids.reduce((s, m) => s + m[1], 0) / mids.length;
      // Cell size: half of whichever diagonal(s) are known (a diagonal
      // spans corner-to-corner, i.e. sqrt(2) side lengths for a square
      // cell) — reasonable even without direct edge measurements.
      const diagLen = mids.length === 2
        ? (Math.hypot(diag1!.bx - diag1!.ax, diag1!.by - diag1!.ay) + Math.hypot(diag2!.bx - diag2!.ax, diag2!.by - diag2!.ay)) / 2
        : Math.hypot((diag1 ?? diag2)!.bx - (diag1 ?? diag2)!.ax, (diag1 ?? diag2)!.by - (diag1 ?? diag2)!.ay);
      const halfSide = diagLen / (2 * Math.SQRT2);
      const bx = Math.max(2, Math.floor(halfSide * 0.4)), by = bx;

      let sum = 0, count = 0;
      for (let dy = -by; dy <= by; dy++) {
        const yy = Math.round(cy + dy);
        if (yy < 0 || yy >= h) continue;
        for (let dx = -bx; dx <= bx; dx++) {
          const xx = Math.round(cx + dx);
          if (xx < 0 || xx >= w) continue;
          sum += bin[yy * w + xx];
          count++;
        }
      }
      rowCells.push({ x: cx, y: cy, bit: count > 0 && sum / count > 0.5 ? 1 : 0, valid: true, cornerCount });
    }
    cells.push(rowCells);
  }
  return { rows, cols, cells, originRow: -minRow, originCol: -minCol };
}

// Packs an order x order cell block into a lookup key, reading it under one
// of 4 orientations (0/90/180/270 degree rotation of the reading order).
// Orientation 0 matches src/debruijn.ts's windowKey order exactly
// (top-to-bottom, left-to-right, MSB-first); the others apply the standard
// NxN matrix rotation index mapping before packing.
function packPatchCells(cells: SampledCell[][], order: number, orientation: number): number {
  let key = 0;
  for (let i = 0; i < order; i++) {
    for (let j = 0; j < order; j++) {
      let bit: number;
      if (orientation === 1) bit = cells[order - 1 - j][i].bit;       // 90 CW
      else if (orientation === 2) bit = cells[order - 1 - i][order - 1 - j].bit; // 180
      else if (orientation === 3) bit = cells[j][order - 1 - i].bit;  // 270 CW
      else bit = cells[i][j].bit;                                    // 0
      key = (key << 1) | bit;
    }
  }
  return key >>> 0;
}

export interface Patch {
  tileRow: number; tileCol: number; // position in the tile grid (not torus coords)
  cells: SampledCell[][]; // order x order
  match: { row: number; col: number } | null;
  // Per-cell ground-truth correctness (does this cell's bit match the actual
  // known pattern at the position the frame's winning anchor implies?), for
  // visual debugging — see pickBestCandidate, which populates this only for
  // the winning candidate (needs a resolved anchor to compare against).
  correct: boolean[][] | null;
}

export interface PatchDecodeResult {
  patches: Patch[];
  orientation: number | null; // resolved 0/90/180/270 reading orientation, shared across all patches this frame
  consistency: number; // fraction of adjacent-tile pairs whose decoded positions agree, 0..1
}

// Scores how self-consistent a set of decoded patches is: for each pair of
// tiles that are adjacent in the tile grid (share an edge), their decoded
// torus positions should be close together (within a couple of cells) if the
// decode is genuinely correct. With only 16 bits of key space (order 4),
// 65535 of 65536 possible windows are valid, so a single patch's "found a
// match" is very weak evidence on its own — cross-checking neighbors against
// each other is what actually distinguishes a correct decode from noise that
// happened to hash to *some* valid-but-wrong position.
function scoreConsistency(patches: Patch[], order: number, R: number, C: number): number {
  const byTile = new Map<string, Patch>();
  for (const p of patches) byTile.set(`${p.tileRow},${p.tileCol}`, p);
  const wrapDist = (a: number, b: number, mod: number) => Math.min(Math.abs(a - b), mod - Math.abs(a - b));

  let agree = 0, total = 0;
  for (const p of patches) {
    if (!p.match) continue;
    for (const [dI, dJ] of [[1, 0], [0, 1]] as const) {
      const neighbor = byTile.get(`${p.tileRow + dI},${p.tileCol + dJ}`);
      if (!neighbor || !neighbor.match) continue;
      total++;
      const dr = wrapDist(p.match.row, neighbor.match.row, R);
      const dc = wrapDist(p.match.col, neighbor.match.col, C);
      if (dr <= order * 2 && dc <= order * 2) agree++;
    }
  }
  return total > 0 ? agree / total : 0;
}

interface Tile { tileRow: number; tileCol: number; cells: SampledCell[][]; }

function buildTiles(sg: SampledGrid, order: number): Tile[] {
  const tileRows = Math.floor(sg.rows / order);
  const tileCols = Math.floor(sg.cols / order);
  const tiles: Tile[] = [];
  for (let I = 0; I < tileRows; I++) {
    for (let J = 0; J < tileCols; J++) {
      const cells: SampledCell[][] = [];
      for (let i = 0; i < order; i++) cells.push(sg.cells[I * order + i].slice(J * order, J * order + order));
      tiles.push({ tileRow: I, tileCol: J, cells });
    }
  }
  return tiles;
}

function patchesForOrientation(tiles: Tile[], order: number, lookup: Int32Array, C: number, o: number): Patch[] {
  return tiles.map(tile => {
    // A gap anywhere in the tile (see SampledCell.valid — only relevant for
    // mesh-based sampling) means an exact-match lookup can't be trusted;
    // skip straight to no-match rather than packing a bogus bit into the
    // key. The tile still contributes to scoreCorrelation later via
    // whichever of its cells ARE valid, just not as an exact-match seed
    // candidate here.
    const complete = tile.cells.every(row => row.every(c => c.valid));
    const packed = complete ? lookup[packPatchCells(tile.cells, order, o)] : -1;
    const match = packed === -1 ? null : { row: Math.floor(packed / C), col: packed % C };
    return { tileRow: tile.tileRow, tileCol: tile.tileCol, cells: tile.cells, match, correct: null };
  });
}

// Tiles the sampled grid into discrete, non-overlapping order x order
// patches, resolves the shared 0/90/180/270 reading-orientation ambiguity
// (the gradient-histogram rotation estimate only pins down the grid angle
// modulo 90 degrees) by trying every orientation and picking whichever gives
// the most self-consistent set of decoded patches (see scoreConsistency —
// picking the first orientation with any valid hit is NOT reliable here,
// since almost any noise hashes to *some* valid window).
export function decodePatches(sg: SampledGrid, order: number, lookup: Int32Array, R: number, C: number): PatchDecodeResult {
  const tiles = buildTiles(sg, order);
  let best: PatchDecodeResult = { patches: [], orientation: null, consistency: -1 };
  for (let o = 0; o < 4; o++) {
    const patches = patchesForOrientation(tiles, order, lookup, C, o);
    const consistency = scoreConsistency(patches, order, R, C);
    if (consistency > best.consistency) best = { patches, orientation: o, consistency };
  }
  return best;
}

// Applies the same orientation-dependent rotation packPatchCells uses (see
// its comment) to a (sampled-grid row shift, col shift) vector, giving the
// torus (row shift, col shift) a correct decode should show between two
// windows offset by that amount in sampled-grid space. Derived by tracking
// how packPatchCells's per-orientation index mapping responds to shifting
// its window's reference point by one cell in each sampled-grid direction —
// same underlying rotation, just applied to a shift vector instead of a
// cell index.
export function rotateShift(da: number, db: number, orientation: number): [number, number] {
  if (orientation === 1) return [db, -da];
  if (orientation === 2) return [-da, -db];
  if (orientation === 3) return [-db, da];
  return [da, db];
}

// Alternative seed generation to decodePatches's non-overlapping tiling —
// needed for mesh-based sampling (see sampleFromMesh), where only 5/8 of
// lattice points are ever detectable so a randomly-placed order x order
// TILE landing on a fully-valid block of cells is rare even when plenty of
// individually-valid cells exist scattered through the grid (a real
// pattern's mesh coverage tested out around only ~5-10% of tiles being
// fully valid — decodePatches found literally zero seeds there despite
// ~10% of individual cells being valid). Sliding a window across EVERY
// possible position, not just tile-aligned ones, finds a valid window far
// more often. Only called when the grid actually has gaps (see
// pickBestCandidate) — the old dense sampleFullGrid path never has any, so
// its behavior and cost are completely unchanged.
function findSlidingSeeds(sg: SampledGrid, order: number, lookup: Int32Array, R: number, C: number, orientation: number): { row: number; col: number }[] {
  const seen = new Set<string>();
  const anchors: { row: number; col: number }[] = [];
  for (let i0 = 0; i0 + order <= sg.rows; i0++) {
    for (let j0 = 0; j0 + order <= sg.cols; j0++) {
      let complete = true;
      for (let i = 0; i < order && complete; i++) {
        for (let j = 0; j < order; j++) {
          if (!sg.cells[i0 + i][j0 + j].valid) { complete = false; break; }
        }
      }
      if (!complete) continue;
      const window = sg.cells.slice(i0, i0 + order).map(row => row.slice(j0, j0 + order));
      const packed = lookup[packPatchCells(window, order, orientation)];
      if (packed === -1) continue;
      const matchRow = Math.floor(packed / C), matchCol = packed % C;
      const [dr, dc] = rotateShift(i0, j0, orientation);
      const anchorRow = ((matchRow - dr) % R + R) % R, anchorCol = ((matchCol - dc) % C + C) % C;
      const key = `${anchorRow},${anchorCol}`;
      if (seen.has(key)) continue;
      seen.add(key);
      anchors.push({ row: anchorRow, col: anchorCol });
    }
  }
  return anchors;
}

// Scores a candidate torus anchor (the torus position implied for the
// sampled grid's own cell (0,0), i.e. its top-left corner) by comparing
// EVERY sampled cell — not just one patch, not just adjacent pairs — against
// the actual torus content at the positions that anchor implies. This is a
// strictly stronger signal than checking patches agree with EACH OTHER
// (the previous approach, scoreDenseConsistency): it checks agreement with
// the known pattern directly, so a genuinely correct anchor stays close to
// 1.0 even with a handful of misread bits (graceful degradation, rather than
// exact-match's all-or-nothing), while a wrong anchor — even one some patch
// found via a "valid" exact-match lookup, which is weak evidence on its own
// given only 16 bits of key space at order 4 (65535 of 65536 possible
// windows are valid) — should sit close to 0.5 (uncorrelated with a random
// binary pattern), giving much better separation than the old metric.
function scoreCorrelation(sg: SampledGrid, torus: Uint8Array[], R: number, C: number, anchorRow: number, anchorCol: number, orientation: number): number {
  let agree = 0, total = 0;
  for (let i = 0; i < sg.rows; i++) {
    for (let j = 0; j < sg.cols; j++) {
      if (!sg.cells[i][j].valid) continue; // gap in mesh-based sampling — no data to compare
      total++;
      const [dr, dc] = rotateShift(i, j, orientation);
      const torusRow = ((anchorRow + dr) % R + R) % R;
      const torusCol = ((anchorCol + dc) % C + C) % C;
      if (sg.cells[i][j].bit === torus[torusRow][torusCol]) agree++;
    }
  }
  return total === 0 ? 0 : agree / total;
}

// Runs across several rotation candidates (typically the gradient-histogram
// estimate plus 90/180/270-degree offsets of it — see this module's header
// comment for why the full angle isn't pinned down by edge orientation
// alone), and — unlike an earlier version of this function — across all 4
// tile-reading orientations independently via scoreCorrelation directly,
// rather than delegating orientation choice to decodePatches's own
// scoreConsistency. That was fine for the dense sampleFullGrid case (patch
// matches exist for scoreConsistency to compare), but for mesh-based
// sampling (see sampleFromMesh), where the grid has gaps, scoreConsistency
// has nothing to compare and returns 0 for every orientation identically —
// it can't actually discriminate the right one. scoreCorrelation, the
// stronger signal already used for anchor selection, does not have this
// problem, so it now picks orientation too; this is a superset of the old
// behavior for the dense case (never worse, since it tries everything the
// old path did plus more).
//
// Seeds come from two sources: decodePatches's non-overlapping tiles (fast,
// works well when most cells are valid) and, when the grid has any gaps,
// findSlidingSeeds's exhaustive window search (needed because a randomly
// tile-aligned window landing on an all-valid block is rare once a
// meaningful fraction of cells are gaps — see findSlidingSeeds). Both feed
// the same scoreCorrelation-based selection, so a mesh-sampled grid pays the
// extra sliding-window cost only when it actually needs it. match is the
// torus position of the sampled grid's CENTER cell (roughly where the
// camera is actually pointed), not (0,0), which could be anywhere depending
// on how far the phase anchor extends in each direction (see detectGrid).
export interface CandidateResult extends PatchDecodeResult { candidateIndex: number; match: { row: number; col: number } | null; }

export function pickBestCandidate(sampledGrids: SampledGrid[], order: number, lookup: Int32Array, torus: Uint8Array[], R: number, C: number): CandidateResult {
  let best: CandidateResult | null = null;
  let winningAnchor: { row: number; col: number } | null = null;

  for (let i = 0; i < sampledGrids.length; i++) {
    const sg = sampledGrids[i];
    const tiles = buildTiles(sg, order);
    const hasGaps = sg.cells.some(row => row.some(c => !c.valid));

    for (let o = 0; o < 4; o++) {
      const patches = patchesForOrientation(tiles, order, lookup, C, o);
      const seenAnchors = new Set<string>();
      let bestAnchor: { row: number; col: number } | null = null;
      let bestScore = -1;

      for (const patch of patches) {
        if (!patch.match) continue;
        const [dr, dc] = rotateShift(patch.tileRow * order, patch.tileCol * order, o);
        const anchorRow = ((patch.match.row - dr) % R + R) % R;
        const anchorCol = ((patch.match.col - dc) % C + C) % C;
        const key = `${anchorRow},${anchorCol}`;
        if (seenAnchors.has(key)) continue;
        seenAnchors.add(key);
        const score = scoreCorrelation(sg, torus, R, C, anchorRow, anchorCol, o);
        if (score > bestScore) { bestScore = score; bestAnchor = { row: anchorRow, col: anchorCol }; }
      }

      if (hasGaps) {
        for (const anchor of findSlidingSeeds(sg, order, lookup, R, C, o)) {
          const key = `${anchor.row},${anchor.col}`;
          if (seenAnchors.has(key)) continue;
          seenAnchors.add(key);
          const score = scoreCorrelation(sg, torus, R, C, anchor.row, anchor.col, o);
          if (score > bestScore) { bestScore = score; bestAnchor = anchor; }
        }
      }

      const consistency = Math.max(0, bestScore);
      let match: { row: number; col: number } | null = null;
      if (bestAnchor) {
        const centerI = Math.floor(sg.rows / 2), centerJ = Math.floor(sg.cols / 2);
        const [dr, dc] = rotateShift(centerI, centerJ, o);
        match = { row: ((bestAnchor.row + dr) % R + R) % R, col: ((bestAnchor.col + dc) % C + C) % C };
      }

      if (!best || consistency > best.consistency) {
        best = { patches, orientation: o, consistency, candidateIndex: i, match };
        winningAnchor = bestAnchor;
      }
    }
  }

  if (!best) return { patches: [], orientation: null, consistency: 0, candidateIndex: -1, match: null };

  // Per-cell ground-truth correctness, computed only for the winning
  // candidate (see the Patch.correct comment) — cheap enough (one pass over
  // the sampled cells) to not bother gating behind whether a caller actually
  // wants it for display.
  if (winningAnchor && best.orientation !== null) {
    const o = best.orientation, anchor = winningAnchor;
    best.patches = best.patches.map(patch => ({
      ...patch,
      correct: patch.cells.map((row, i) => row.map((cell, j) => {
        const globalI = patch.tileRow * order + i, globalJ = patch.tileCol * order + j;
        const [dr, dc] = rotateShift(globalI, globalJ, o);
        const torusRow = ((anchor.row + dr) % R + R) % R;
        const torusCol = ((anchor.col + dc) % C + C) % C;
        return cell.bit === torus[torusRow][torusCol];
      })),
    }));
  }

  return best;
}

export function toGrayscale(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Float64Array {
  const gray = new Float64Array(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return gray;
}

// Binarizes a grayscale buffer via global mean threshold (dark -> 1, i.e. a
// black cell).
export function binarize(gray: Float64Array): Uint8Array {
  let mean = 0;
  for (let i = 0; i < gray.length; i++) mean += gray[i];
  mean /= gray.length;
  const bin = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) bin[i] = gray[i] < mean ? 1 : 0;
  return bin;
}
