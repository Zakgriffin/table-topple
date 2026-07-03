// Synthetic self-test for the decode pipeline: crops a known region out of
// the actual generated pattern PNG (simulating a camera capturing that
// region), runs it through the exact same decodeFrame logic the browser app
// uses, and checks the recovered (row, col) matches where the crop was
// actually taken from. Catches pipeline bugs (bit order, threshold
// direction, off-by-ones) without needing a live camera.
//
// Usage: node scripts/test-decode.ts [order]

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus, buildLookupTable } from '../src/debruijn.ts';
import { detectGrid, sampleWindow, binarizeRGBA } from '../src/decode.ts';

const order = parseInt(process.argv[2] ?? '4', 10);
const pngPath = `samples/order${order}.png`;

console.log(`Building order-${order} torus + lookup table...`);
const debruijn = generateTorus(order);
const lookup = buildLookupTable(debruijn);
const { R, C } = debruijn;
console.log(`Torus: ${R}x${C} cells.`);

const png = PNG.sync.read(readFileSync(pngPath));
console.log(`Loaded ${pngPath}: ${png.width}x${png.height}px`);

// The PNG was rendered at a fixed cell size (see samples generation: 8px/cell).
const cellPx = png.width / C;
if (!Number.isInteger(cellPx) || png.width / C !== png.height / R) {
  throw new Error(`Unexpected PNG dimensions for order ${order}: ${png.width}x${png.height} vs ${R}x${C} cells`);
}
console.log(`Cell size in PNG: ${cellPx}px`);

// Crops a CROP x CROP region starting at a given cell (testRow, testCol),
// simulating a camera capture centered there, and returns an RGBA buffer
// matching what cropCtx.getImageData would hand the browser decoder.
const CROP = 200;
function cropAt(testRow: number, testCol: number): Uint8ClampedArray {
  // Leave margin so the ORDER x ORDER window comfortably fits mid-crop.
  const srcX = ((testCol * cellPx) - CROP / 2 + cellPx / 2 + C * cellPx) % (C * cellPx);
  const srcY = ((testRow * cellPx) - CROP / 2 + cellPx / 2 + R * cellPx) % (R * cellPx);
  const out = new Uint8ClampedArray(CROP * CROP * 4);
  for (let y = 0; y < CROP; y++) {
    const sy = (Math.floor(srcY) + y) % png.height;
    for (let x = 0; x < CROP; x++) {
      const sx = (Math.floor(srcX) + x) % png.width;
      const srcIdx = (png.width * sy + sx) << 2;
      const dstIdx = (CROP * y + x) << 2;
      out[dstIdx] = png.data[srcIdx];
      out[dstIdx + 1] = png.data[srcIdx + 1];
      out[dstIdx + 2] = png.data[srcIdx + 2];
      out[dstIdx + 3] = 255;
    }
  }
  return out;
}

function decode(rgba: Uint8ClampedArray): { row: number; col: number } | null {
  const bin = binarizeRGBA(rgba, CROP, CROP);
  const grid = detectGrid(bin, CROP, CROP);
  const key = sampleWindow(bin, CROP, CROP, grid, order);
  if (key === null) return null;
  const packed = lookup[key];
  if (packed === -1) return null;
  return { row: Math.floor(packed / C), col: packed % C };
}

const trials = 200;
let hits = 0, misses = 0, wrong = 0;
for (let t = 0; t < trials; t++) {
  const testRow = Math.floor(Math.random() * R);
  const testCol = Math.floor(Math.random() * C);
  const rgba = cropAt(testRow, testCol);
  const result = decode(rgba);
  if (!result) { misses++; console.log(`MISS at true (${testRow},${testCol}) — no lock`); continue; }
  // The decoded window is centered in the crop, not necessarily exactly at
  // (testRow, testCol) since sampleWindow picks the centered ORDER-window —
  // just check the result is a cell that's actually near the crop center.
  const dr = Math.min(Math.abs(result.row - testRow), R - Math.abs(result.row - testRow));
  const dc = Math.min(Math.abs(result.col - testCol), C - Math.abs(result.col - testCol));
  if (dr <= 1 && dc <= 1) hits++;
  else { wrong++; console.log(`WRONG: true (${testRow},${testCol}) -> decoded (${result.row},${result.col})`); }
}
console.log(`\n${hits}/${trials} correct, ${misses} no-lock, ${wrong} wrong.`);
if (wrong > 0) { console.error('FAIL: some decodes were wrong (not just missed).'); process.exit(1); }
console.log(hits === trials ? 'ALL PASSED' : 'Some no-locks, but zero wrong decodes.');
