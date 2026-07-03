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
//      (see estimateRotationRad) on that raw (still-rotated) crop.
//   3. The caller derotates the ORIGINAL grayscale/RGBA crop by that angle
//      (browser: via canvas ctx.rotate + drawImage, for smooth resampling;
//      Node test: via manual nearest-neighbor resampling) into a fresh
//      "aligned" buffer, then re-binarizes that.
//   4. Find the cell pitch + phase in the aligned buffer via autocorrelation
//      of an "edge energy" profile (sum of adjacent-pixel differences) —
//      periodic peaks in that profile land at cell boundaries. Since the
//      buffer is now (assumed) axis-aligned, this is the same detector as
//      Stage 1, just running on rotation-corrected pixels.
//   5. Sample every fully-visible cell into a grid, then tile that into
//      discrete, non-overlapping order x order patches (see decodePatches).
//   6. The gradient-histogram angle is only known modulo 90 degrees (a
//      square grid's edges alone can't distinguish 0/90/180/270 rotation),
//      so decodePatches resolves that by trying each patch in all 4 reading
//      orientations against the lookup table until one hits, then reuses
//      that orientation for the rest of the frame's patches.

export interface GridDetection {
  px: number; py: number; // phase (px offset of first cell boundary)
  pitchX: number; pitchY: number;
}

// Finds the pitch (dominant period) of a 1D energy profile via autocorrelation.
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

export function detectGrid(bin: Uint8Array, w: number, h: number): GridDetection {
  const colEnergy = new Float64Array(w);
  for (let x = 1; x < w; x++) {
    let e = 0;
    for (let y = 0; y < h; y++) e += Math.abs(bin[y * w + x] - bin[y * w + x - 1]);
    colEnergy[x] = e;
  }
  const rowEnergy = new Float64Array(h);
  for (let y = 1; y < h; y++) {
    let e = 0;
    for (let x = 0; x < w; x++) e += Math.abs(bin[y * w + x] - bin[(y - 1) * w + x]);
    rowEnergy[y] = e;
  }

  const minLag = 4, maxLagX = Math.floor(w / 4), maxLagY = Math.floor(h / 4);
  const pitchX = findPitch(colEnergy, minLag, maxLagX);
  const pitchY = findPitch(rowEnergy, minLag, maxLagY);
  const px = findPhase(colEnergy, pitchX);
  const py = findPhase(rowEnergy, pitchY);
  return { px, py, pitchX, pitchY };
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

export function estimateRotationRad(gray: Float64Array, w: number, h: number): number {
  const blurred = boxBlur(gray, w, h, 2);
  const BINS = 180;
  const HALF_PI = Math.PI / 2;
  const hist = new Float64Array(BINS);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx = blurred[y * w + x + 1] - blurred[y * w + x - 1];
      const gy = blurred[(y + 1) * w + x] - blurred[(y - 1) * w + x];
      const mag = gx * gx + gy * gy; // squared magnitude — emphasizes strong edges
      if (mag < 1) continue;
      const angle = Math.atan2(gy, gx); // -PI..PI
      const folded = ((angle % HALF_PI) + HALF_PI) % HALF_PI; // [0, PI/2)
      const idx = Math.min(BINS - 1, Math.floor((folded / HALF_PI) * BINS));
      hist[idx] += mag;
    }
  }
  // 3-bin smoothing (circular, since angle mod PI/2 wraps) to avoid locking
  // onto a single noisy spike.
  let bestIdx = 0, bestVal = -Infinity;
  for (let i = 0; i < BINS; i++) {
    const smoothed = hist[(i - 1 + BINS) % BINS] + hist[i] + hist[(i + 1) % BINS];
    if (smoothed > bestVal) { bestVal = smoothed; bestIdx = i; }
  }
  return (bestIdx / BINS) * HALF_PI;
}

export interface SampledCell { x: number; y: number; bit: number; }

// Samples every fully-visible cell in the (assumed axis-aligned) buffer into
// a 2D grid, cells[row][col], row 0 = top.
export interface SampledGrid { rows: number; cols: number; cells: SampledCell[][]; }

export function sampleFullGrid(bin: Uint8Array, w: number, h: number, grid: GridDetection): SampledGrid {
  const { px, py, pitchX, pitchY } = grid;
  const cols = Math.floor((w - px) / pitchX);
  const rows = Math.floor((h - py) / pitchY);
  const bx = Math.max(2, Math.floor(pitchX * 0.2));
  const by = Math.max(2, Math.floor(pitchY * 0.2));

  const cells: SampledCell[][] = [];
  for (let i = 0; i < rows; i++) {
    const cy = py + pitchY * (i + 0.5);
    const rowCells: SampledCell[] = [];
    for (let j = 0; j < cols; j++) {
      const cx = px + pitchX * (j + 0.5);
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
      rowCells.push({ x: cx, y: cy, bit: count > 0 && sum / count > 0.5 ? 1 : 0 });
    }
    cells.push(rowCells);
  }
  return { rows, cols, cells };
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
}

export interface PatchDecodeResult {
  patches: Patch[];
  orientation: number | null; // resolved 0/90/180/270 reading orientation, shared across all patches this frame
}

// Tiles the sampled grid into discrete, non-overlapping order x order
// patches. The gradient-histogram rotation estimate only pins down the grid
// angle modulo 90 degrees, so this resolves the remaining 4-fold reading
// ambiguity by trying patches (and all 4 orientations per patch) until one
// produces a valid lookup hit — false positives across unrelated windows
// should be astronomically unlikely given the uniqueness guarantee — then
// reuses that same orientation for every other patch in the frame.
export function decodePatches(sg: SampledGrid, order: number, lookup: Int32Array, C: number): PatchDecodeResult {
  const tileRows = Math.floor(sg.rows / order);
  const tileCols = Math.floor(sg.cols / order);

  const tiles: { tileRow: number; tileCol: number; cells: SampledCell[][] }[] = [];
  for (let I = 0; I < tileRows; I++) {
    for (let J = 0; J < tileCols; J++) {
      const cells: SampledCell[][] = [];
      for (let i = 0; i < order; i++) cells.push(sg.cells[I * order + i].slice(J * order, J * order + order));
      tiles.push({ tileRow: I, tileCol: J, cells });
    }
  }

  let orientation: number | null = null;
  outer:
  for (const tile of tiles) {
    for (let o = 0; o < 4; o++) {
      if (lookup[packPatchCells(tile.cells, order, o)] !== -1) { orientation = o; break outer; }
    }
  }

  const patches: Patch[] = tiles.map(tile => {
    let match: { row: number; col: number } | null = null;
    if (orientation !== null) {
      const packed = lookup[packPatchCells(tile.cells, order, orientation)];
      if (packed !== -1) match = { row: Math.floor(packed / C), col: packed % C };
    }
    return { tileRow: tile.tileRow, tileCol: tile.tileCol, cells: tile.cells, match };
  });

  return { patches, orientation };
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
