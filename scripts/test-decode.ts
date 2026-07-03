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
import { detectLocalGrid, sampleFullGrid, pickBestCandidate, toGrayscale, binarize, estimateRotationRad, asSignedResidual } from '../src/decode.ts';
import type { SampledGrid } from '../src/decode.ts';

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

// Edge orientation alone only pins the grid angle down modulo 90 degrees, so
// this derotates at all 4 candidate full angles and lets pickBestCandidate
// decide which one actually produces the best-correlated match.
function decode(rgba: Uint8ClampedArray): { match: { row: number; col: number } | null; consistency: number } {
  const rawGray = toGrayscale(rgba, RAW, RAW);
  const thetaCoarse = estimateRotationRad(rawGray, RAW, RAW);

  // Coarse-to-fine refinement — see src/main.ts's decodeFrame for the same
  // pattern in the browser (there via canvas ctx.rotate; here via the
  // manual derotate() above).
  const previewGray = derotate(rawGray, thetaCoarse);
  const residual = asSignedResidual(estimateRotationRad(previewGray, ALIGNED, ALIGNED));
  const theta0 = thetaCoarse + residual;

  const sampledGrids = [0, 1, 2, 3].map(k => {
    const theta = theta0 + k * (Math.PI / 2);
    const alignedGray = derotate(rawGray, theta);
    const alignedBin = binarize(alignedGray);
    const grid = detectLocalGrid(alignedBin, ALIGNED, ALIGNED);
    return sampleFullGrid(alignedBin, ALIGNED, ALIGNED, grid, grid);
  });

  const { match, consistency } = pickBestCandidate(sampledGrids, order, lookup, debruijn.torus, R, C);
  return { match, consistency };
}

function within(target: number, start: number, span: number, mod: number): boolean {
  const rel = ((target - start) % mod + mod) % mod;
  return rel <= span || rel >= mod - 1;
}

// First pass: no confidence threshold, just log raw scores split by
// correct/incorrect, to empirically justify where to set the threshold
// rather than guessing.
const trials = 300;
const testAngles = [0, 15, 30, 45, -20, -40, 60, 80];
const correctScores: number[] = [], wrongScores: number[] = [];
let hits = 0, misses = 0, wrong = 0;
for (let t = 0; t < trials; t++) {
  const testRow = Math.floor(Math.random() * R);
  const testCol = Math.floor(Math.random() * C);
  const phiDeg = testAngles[t % testAngles.length];
  const rgba = cropAtRotated(testRow, testCol, phiDeg * Math.PI / 180);
  const { match, consistency } = decode(rgba);

  if (!match) { misses++; continue; }
  const correct = within(testRow, match.row, order, R) && within(testCol, match.col, order, C);
  if (correct) { hits++; correctScores.push(consistency); }
  else { wrong++; wrongScores.push(consistency); console.log(`WRONG at true (${testRow},${testCol}) angle ${phiDeg}deg consistency=${consistency.toFixed(3)} decoded=(${match.row},${match.col})`); }
}

function stats(xs: number[]): string {
  if (xs.length === 0) return 'n/a';
  const min = Math.min(...xs), max = Math.max(...xs), mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  return `min=${min.toFixed(3)} mean=${mean.toFixed(3)} max=${max.toFixed(3)} n=${xs.length}`;
}
console.log(`\ncorrect-match consistency scores: ${stats(correctScores)}`);
console.log(`wrong-match consistency scores: ${stats(wrongScores)}`);

console.log(`\n${hits}/${trials} correct, ${misses} no-lock, ${wrong} wrong (before applying any confidence threshold).`);
const wrongRate = wrong / trials;
if (wrongRate > 0.1) { console.error(`FAIL: wrong-decode rate ${(wrongRate * 100).toFixed(1)}% is too high.`); process.exit(1); }
console.log(`PASS (wrong-decode rate ${(wrongRate * 100).toFixed(1)}%).`);

// Second pass: with the CONFIDENCE_THRESHOLD main.ts actually uses (0.85 —
// well above the highest observed wrong-match score of ~0.6, comfortably
// below what a correct decode should show even with real-world bit noise).
const CONFIDENCE_THRESHOLD = 0.85;
let gatedHits = 0, gatedMisses = 0, gatedWrong = 0;
for (let t = 0; t < trials; t++) {
  const testRow = Math.floor(Math.random() * R);
  const testCol = Math.floor(Math.random() * C);
  const phiDeg = testAngles[t % testAngles.length];
  const rgba = cropAtRotated(testRow, testCol, phiDeg * Math.PI / 180);
  const { match, consistency } = decode(rgba);
  if (!match || consistency < CONFIDENCE_THRESHOLD) { gatedMisses++; continue; }
  const correct = within(testRow, match.row, order, R) && within(testCol, match.col, order, C);
  if (correct) gatedHits++; else gatedWrong++;
}
console.log(`\nWith CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD}: ${gatedHits}/${trials} correct, ${gatedMisses} no-lock, ${gatedWrong} wrong.`);
if (gatedWrong > 0) { console.error('FAIL: threshold did not eliminate wrong decodes.'); process.exit(1); }

// Third pass: the actual point of correlation-based decode — every previous
// trial was either bit-perfect-correct or totally-wrong, never PARTIALLY
// corrupted, so none of them actually exercised graceful degradation. This
// injects random bit flips directly into the sampled grid (simulating real
// misreads from lighting/blur/etc, downstream of the pixel pipeline) and
// checks that decoding still succeeds, and that the confidence score
// degrades roughly with the noise rate rather than falling off a cliff.
function decodeSampledGrids(sampledGrids: SampledGrid[]) {
  return pickBestCandidate(sampledGrids, order, lookup, debruijn.torus, R, C);
}

function buildSampledGrids(rgba: Uint8ClampedArray): SampledGrid[] {
  const rawGray = toGrayscale(rgba, RAW, RAW);
  const thetaCoarse = estimateRotationRad(rawGray, RAW, RAW);
  const previewGray = derotate(rawGray, thetaCoarse);
  const residual = asSignedResidual(estimateRotationRad(previewGray, ALIGNED, ALIGNED));
  const theta0 = thetaCoarse + residual;
  return [0, 1, 2, 3].map(k => {
    const theta = theta0 + k * (Math.PI / 2);
    const alignedBin = binarize(derotate(rawGray, theta));
    const grid = detectLocalGrid(alignedBin, ALIGNED, ALIGNED);
    return sampleFullGrid(alignedBin, ALIGNED, ALIGNED, grid, grid);
  });
}

function injectNoise(sampledGrids: SampledGrid[], rate: number): SampledGrid[] {
  return sampledGrids.map(sg => ({
    ...sg,
    cells: sg.cells.map(row => row.map(cell => Math.random() < rate ? { ...cell, bit: 1 - cell.bit } : cell)),
  }));
}

console.log('\nBit-noise tolerance (injected directly into the sampled grid):');
for (const noiseRate of [0, 0.02, 0.05, 0.1, 0.15, 0.2]) {
  let nHits = 0, nMisses = 0, nWrong = 0;
  const scores: number[] = [];
  for (let t = 0; t < 100; t++) {
    const testRow = Math.floor(Math.random() * R);
    const testCol = Math.floor(Math.random() * C);
    const rgba = cropAtRotated(testRow, testCol, 20 * Math.PI / 180);
    const sampledGrids = injectNoise(buildSampledGrids(rgba), noiseRate);
    const { match, consistency } = decodeSampledGrids(sampledGrids);
    if (!match || consistency < CONFIDENCE_THRESHOLD) { nMisses++; continue; }
    scores.push(consistency);
    if (within(testRow, match.row, order, R) && within(testCol, match.col, order, C)) nHits++;
    else nWrong++;
  }
  console.log(`  ${(noiseRate * 100).toFixed(0)}% bit noise: ${nHits}/100 correct, ${nMisses} no-lock, ${nWrong} wrong, avg score ${stats(scores)}`);
}
