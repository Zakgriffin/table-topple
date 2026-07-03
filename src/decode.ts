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
//   6. Pick whichever of the 4 angle candidates gives the most *mutually
//      consistent* patches (adjacent tiles decoding to nearby torus
//      positions — see scoreConsistency and pickBestCandidate). This
//      matters because with only 16 bits of key space at order 4, 65535 of
//      65536 possible windows are valid, so a single patch finding "a"
//      match is very weak evidence; only cross-patch agreement reliably
//      distinguishes a correct decode from noise that happened to hash to
//      some valid-but-wrong position.

export interface GridDetection {
  px: number; py: number; // phase (px offset of first cell boundary)
  pitchX: number; pitchY: number;
}

// Finds the pitch (dominant period) of a 1D energy profile via autocorrelation,
// refined to sub-pixel precision via parabolic interpolation of the
// autocorrelation score around the best integer lag. Without this, findPitch
// could only ever return whole-pixel cell sizes — any true (real-world) pitch
// that isn't an exact integer of pixels gets rounded, and that rounding error
// compounds with distance from the phase anchor when cell positions are
// extrapolated (see detectGrid / sampleFullGrid), producing visible drift
// toward the far side of the visible grid.
export function findPitch(energy: Float64Array, minLag: number, maxLag: number): number {
  const scores = new Float64Array(maxLag - minLag + 1);
  let bestIdx = 0, bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let score = 0;
    for (let x = 0; x + lag < energy.length; x++) score += energy[x] * energy[x + lag];
    scores[lag - minLag] = score;
    if (score > bestScore) { bestScore = score; bestIdx = lag - minLag; }
  }
  const bestLag = bestIdx + minLag;

  if (bestIdx > 0 && bestIdx < scores.length - 1) {
    const yMinus = scores[bestIdx - 1], y0 = scores[bestIdx], yPlus = scores[bestIdx + 1];
    const denom = yMinus - 2 * y0 + yPlus;
    if (Math.abs(denom) > 1e-9) {
      const delta = 0.5 * (yMinus - yPlus) / denom;
      if (Math.abs(delta) < 1) return bestLag + delta;
    }
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
  let px = findPhase(colEnergy, pitchX);
  let py = findPhase(rowEnergy, pitchY);

  // Re-anchor the phase to the boundary nearest the buffer's center, rather
  // than leaving it wherever findPhase's [0, pitch) search happened to land
  // it (always near index 0). Cell positions are extrapolated outward from
  // this anchor (see sampleFullGrid), so any residual pitch error compounds
  // with distance from it — anchoring near one edge means the far edge
  // carries the full accumulated error; anchoring near the center means
  // every visible edge is at most half that distance away, roughly halving
  // the worst-case drift and making it symmetric instead of one-sided.
  px += pitchX * Math.round((w / 2 - px) / pitchX);
  py += pitchY * Math.round((h / 2 - py) / pitchY);

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

export interface SampledCell { x: number; y: number; bit: number; }

// Samples every fully-visible cell in the (assumed axis-aligned) buffer into
// a 2D grid, cells[row][col], row 0 = top. Extends in BOTH directions from
// (px, py) — not just rightward/downward — since detectGrid now re-anchors
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
  const colsLeft = Math.floor(px / pitchX), colsRight = Math.floor((w - px) / pitchX);
  const rowsUp = Math.floor(py / pitchY), rowsDown = Math.floor((h - py) / pitchY);
  const cols = colsLeft + colsRight, rows = rowsUp + rowsDown;
  const bx = Math.max(2, Math.floor(pitchX * 0.2));
  const by = Math.max(2, Math.floor(pitchY * 0.2));

  const cells: SampledCell[][] = [];
  for (let i = 0; i < rows; i++) {
    const cy = py + pitchY * (i - rowsUp + 0.5);
    const rowCells: SampledCell[] = [];
    for (let j = 0; j < cols; j++) {
      const cx = px + pitchX * (j - colsLeft + 0.5);
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
  return { rows, cols, cells, originRow: rowsUp, originCol: colsLeft };
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

// Tiles the sampled grid into discrete, non-overlapping order x order
// patches, resolves the shared 0/90/180/270 reading-orientation ambiguity
// (the gradient-histogram rotation estimate only pins down the grid angle
// modulo 90 degrees) by trying every orientation and picking whichever gives
// the most self-consistent set of decoded patches (see scoreConsistency —
// picking the first orientation with any valid hit is NOT reliable here,
// since almost any noise hashes to *some* valid window).
export function decodePatches(sg: SampledGrid, order: number, lookup: Int32Array, R: number, C: number): PatchDecodeResult {
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

  let best: PatchDecodeResult = { patches: [], orientation: null, consistency: -1 };
  for (let o = 0; o < 4; o++) {
    const patches: Patch[] = tiles.map(tile => {
      const packed = lookup[packPatchCells(tile.cells, order, o)];
      const match = packed === -1 ? null : { row: Math.floor(packed / C), col: packed % C };
      return { tileRow: tile.tileRow, tileCol: tile.tileCol, cells: tile.cells, match };
    });
    const consistency = scoreConsistency(patches, order, R, C);
    if (consistency > best.consistency) best = { patches, orientation: o, consistency };
  }
  return best;
}

// Runs decodePatches across several rotation candidates (typically the
// gradient-histogram estimate plus 90/180/270-degree offsets of it — see
// this module's header comment for why the full angle isn't pinned down by
// edge orientation alone) and returns whichever candidate scores the most
// self-consistent set of patches.
export interface CandidateResult extends PatchDecodeResult { candidateIndex: number; }

export function pickBestCandidate(sampledGrids: SampledGrid[], order: number, lookup: Int32Array, R: number, C: number): CandidateResult {
  let best: CandidateResult | null = null;
  for (let i = 0; i < sampledGrids.length; i++) {
    const result = decodePatches(sampledGrids[i], order, lookup, R, C);
    if (!best || result.consistency > best.consistency) best = { ...result, candidateIndex: i };
  }
  return best!;
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
