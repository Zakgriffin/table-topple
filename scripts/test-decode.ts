// Synthetic self-test for the rotation-aware decode pipeline: crops a known,
// ROTATED region out of the actual generated pattern PNG (simulating a
// camera capturing that region at some angle), runs it through the exact
// same decode.ts logic the browser app uses (rotation estimation, manual
// derotation here in place of canvas's ctx.rotate, grid detection, patch
// decoding), and checks that at least one patch recovers a position near
// where the crop was actually taken from. Catches transform-math bugs
// without needing a live camera.
//
// Usage: node scripts/test-decode.ts [order]

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus, buildLookupTable } from '../src/debruijn.ts';
import { detectGrid, sampleFullGrid, decodePatches, toGrayscale, binarize, estimateRotationRad } from '../src/decode.ts';

const order = parseInt(process.argv[2] ?? '4', 10);
const pngPath = `samples/order${order}.png`;

console.log(`Building order-${order} torus + lookup table...`);
const debruijn = generateTorus(order);
const lookup = buildLookupTable(debruijn);
const { R, C } = debruijn;
console.log(`Torus: ${R}x${C} cells.`);

const png = PNG.sync.read(readFileSync(pngPath));
console.log(`Loaded ${pngPath}: ${png.width}x${png.height}px`);

const cellPx = png.width / C;
if (!Number.isInteger(cellPx) || png.width / C !== png.height / R) {
  throw new Error(`Unexpected PNG dimensions for order ${order}: ${png.width}x${png.height} vs ${R}x${C} cells`);
}
console.log(`Cell size in PNG: ${cellPx}px`);

// RAW must be large enough that after any rotation, ALIGNED is fully covered
// by real (derotated) content — RAW >= ALIGNED * sqrt(2) suffices, see
// src/decode.ts's module comment for the transform this mirrors.
const RAW = 360;
const ALIGNED = 240;

// Simulates capturing a RAW x RAW crop centered at (testRow, testCol) as if
// photographed at rotation phi (radians) relative to the pattern.
function cropAtRotated(testRow: number, testCol: number, phi: number): Uint8ClampedArray {
  const cosP = Math.cos(phi), sinP = Math.sin(phi);
  const centerX = testCol * cellPx + cellPx / 2;
  const centerY = testRow * cellPx + cellPx / 2;
  const out = new Uint8ClampedArray(RAW * RAW * 4);
  for (let ry = 0; ry < RAW; ry++) {
    const dy = ry - RAW / 2;
    for (let rx = 0; rx < RAW; rx++) {
      const dx = rx - RAW / 2;
      const sx = Math.round(centerX + (dx * cosP - dy * sinP));
      const sy = Math.round(centerY + (dx * sinP + dy * cosP));
      const wx = ((sx % png.width) + png.width) % png.width;
      const wy = ((sy % png.height) + png.height) % png.height;
      const srcIdx = (png.width * wy + wx) << 2;
      const dstIdx = (RAW * ry + rx) << 2;
      out[dstIdx] = png.data[srcIdx];
      out[dstIdx + 1] = png.data[srcIdx + 1];
      out[dstIdx + 2] = png.data[srcIdx + 2];
      out[dstIdx + 3] = 255;
    }
  }
  return out;
}

// Manual nearest-neighbor derotation, standing in for the browser's
// canvas ctx.rotate(-theta) + drawImage. Must match that transform exactly
// (see src/decode.ts's module comment): a RAW-buffer point is mapped to
// ALIGNED-buffer space by rotating by -theta around RAW's center and
// re-centering on ALIGNED; this is the inverse of that map, used to pull
// each ALIGNED pixel from its source RAW pixel.
function derotate(gray: Float64Array, theta: number): Float64Array {
  const out = new Float64Array(ALIGNED * ALIGNED);
  const cosT = Math.cos(theta), sinT = Math.sin(theta);
  for (let ay = 0; ay < ALIGNED; ay++) {
    const relY = ay - ALIGNED / 2;
    for (let ax = 0; ax < ALIGNED; ax++) {
      const relX = ax - ALIGNED / 2;
      const rx = Math.round(relX * cosT - relY * sinT + RAW / 2);
      const ry = Math.round(relX * sinT + relY * cosT + RAW / 2);
      out[ay * ALIGNED + ax] = (rx >= 0 && rx < RAW && ry >= 0 && ry < RAW) ? gray[ry * RAW + rx] : 255;
    }
  }
  return out;
}

function decode(rgba: Uint8ClampedArray): { anyMatch: boolean; matches: { row: number; col: number }[] } {
  const rawGray = toGrayscale(rgba, RAW, RAW);
  const rawBin = binarize(rawGray);
  const theta = estimateRotationRad(rawBin, RAW, RAW);

  const alignedGray = derotate(rawGray, theta);
  const alignedBin = binarize(alignedGray);

  const grid = detectGrid(alignedBin, ALIGNED, ALIGNED);
  const sg = sampleFullGrid(alignedBin, ALIGNED, ALIGNED, grid);
  const { patches } = decodePatches(sg, order, lookup, C);

  const matches = patches.map(p => p.match).filter((m): m is { row: number; col: number } => m !== null);
  return { anyMatch: matches.length > 0, matches };
}

function within(target: number, start: number, span: number, mod: number): boolean {
  const rel = ((target - start) % mod + mod) % mod;
  return rel <= span || rel >= mod - 1;
}

const trials = 60;
const testAngles = [0, 15, 30, 45, -20, -40, 60, 80];
let hits = 0, misses = 0, wrong = 0;
for (let t = 0; t < trials; t++) {
  const testRow = Math.floor(Math.random() * R);
  const testCol = Math.floor(Math.random() * C);
  const phiDeg = testAngles[t % testAngles.length];
  const rgba = cropAtRotated(testRow, testCol, phiDeg * Math.PI / 180);
  const { anyMatch, matches } = decode(rgba);

  if (!anyMatch) { misses++; console.log(`MISS at true (${testRow},${testCol}) angle ${phiDeg}deg — no patch matched`); continue; }

  const anyCorrect = matches.some(m => within(testRow, m.row, order, R) && within(testCol, m.col, order, C));
  if (anyCorrect) hits++;
  else { wrong++; console.log(`WRONG at true (${testRow},${testCol}) angle ${phiDeg}deg — patches decoded: ${JSON.stringify(matches)}`); }
}

console.log(`\n${hits}/${trials} correct, ${misses} no-lock, ${wrong} wrong.`);
const wrongRate = wrong / trials;
if (wrongRate > 0.1) { console.error(`FAIL: wrong-decode rate ${(wrongRate * 100).toFixed(1)}% is too high.`); process.exit(1); }
console.log(`PASS (wrong-decode rate ${(wrongRate * 100).toFixed(1)}%).`);
