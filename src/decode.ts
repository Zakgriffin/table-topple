// Grid decode pipeline — pure logic, no DOM/camera dependency, so it can run
// both in the browser (src/main.ts) and under Node for testing
// (scripts/test-decode.ts) against a known image.
//
// Stage 1: axis-aligned only (no rotation search yet).
//   1. Binarize a square crop (global mean threshold — the pattern is pure
//      black/white, so this is robust as long as lighting is roughly even).
//   2. Find the cell pitch in each axis via autocorrelation of an "edge
//      energy" profile (sum of adjacent-pixel differences) — periodic peaks
//      in that profile land at cell boundaries, spaced by the pitch.
//   3. Find the phase (sub-pitch offset) that best aligns with those boundaries.
//   4. Sample the order x order grid of cells nearest the crop center,
//      average brightness per cell, threshold to a bit.
//   5. Pack into a window key (top-to-bottom, left-to-right, MSB-first — must
//      match src/debruijn.ts's windowKey order) for lookup.

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

// Samples the order x order cells nearest the crop center and packs them into
// a window key, or returns null if not enough cells are visible.
export function sampleWindow(bin: Uint8Array, w: number, h: number, grid: GridDetection, order: number): number | null {
  const { px, py, pitchX, pitchY } = grid;
  const numCellsX = Math.floor((w - px) / pitchX);
  const numCellsY = Math.floor((h - py) / pitchY);
  if (numCellsX < order || numCellsY < order) return null;

  const startX = Math.floor((numCellsX - order) / 2);
  const startY = Math.floor((numCellsY - order) / 2);

  let key = 0;
  for (let i = 0; i < order; i++) {
    const cy = py + pitchY * (startY + i + 0.5);
    for (let j = 0; j < order; j++) {
      const cx = px + pitchX * (startX + j + 0.5);
      // Average a small box around the cell center to reduce edge noise.
      const bx = Math.max(2, Math.floor(pitchX * 0.2));
      const by = Math.max(2, Math.floor(pitchY * 0.2));
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
      const bit = count > 0 && sum / count > 0.5 ? 1 : 0;
      key = (key << 1) | bit;
    }
  }
  return key >>> 0;
}

// Converts an RGBA buffer to a binary (0/1, dark -> 1) array via global mean
// threshold.
export function binarizeRGBA(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Uint8Array {
  const gray = new Float64Array(w * h);
  let mean = 0;
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const luma = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
    gray[i] = luma;
    mean += luma;
  }
  mean /= gray.length;
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < gray.length; i++) bin[i] = gray[i] < mean ? 1 : 0;
  return bin;
}
