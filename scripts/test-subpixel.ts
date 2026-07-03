// Validates src/cornerdetect.ts's refineJunctionSubPixel against exact
// synthetic ground truth (same 16-coloring construction as
// scripts/test-junctions.ts, plus rotation), checking that the refined
// position lands close to the TRUE lattice point — not just that the coarse
// detector found something nearby (that's test-junctions.ts's job).
//
// Usage: node scripts/test-subpixel.ts

import { computeJunctionField, detectJunctions, refineJunctionSubPixel } from '../src/cornerdetect.ts';

const CELL = 40;
const SIZE = CELL * 2;
const BIG = SIZE * 2; // extra margin so rotation doesn't clip the corner out of frame
const TRUE_CENTER = BIG / 2;

function render(tl: number, tr: number, bl: number, br: number): Float64Array {
  const gray = new Float64Array(BIG * BIG);
  const off = (BIG - SIZE) / 2;
  for (let y = 0; y < BIG; y++) {
    for (let x = 0; x < BIG; x++) {
      const ly = y - off, lx = x - off;
      let v = 255;
      if (ly >= 0 && ly < SIZE && lx >= 0 && lx < SIZE) {
        const top = ly < CELL, left = lx < CELL;
        v = (top ? (left ? tl : tr) : (left ? bl : br)) ? 0 : 255;
      }
      gray[y * BIG + x] = v;
    }
  }
  return gray;
}

function rotate(gray: Float64Array, theta: number): Float64Array {
  const out = new Float64Array(BIG * BIG);
  const c = Math.cos(theta), s = Math.sin(theta), cx = BIG / 2, cy = BIG / 2;
  for (let y = 0; y < BIG; y++) {
    for (let x = 0; x < BIG; x++) {
      const dx = x - cx, dy = y - cy;
      const sx = Math.round(dx * c - dy * s + cx), sy = Math.round(dx * s + dy * c + cy);
      out[y * BIG + x] = (sx >= 0 && sx < BIG && sy >= 0 && sy < BIG) ? gray[sy * BIG + sx] : 255;
    }
  }
  return out;
}

function preBlur(gray: Float64Array, radius: number): Float64Array {
  const out = new Float64Array(BIG * BIG);
  for (let y = 0; y < BIG; y++) {
    for (let x = 0; x < BIG; x++) {
      let sum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= BIG) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= BIG) continue;
          sum += gray[yy * BIG + xx]; count++;
        }
      }
      out[y * BIG + x] = sum / count;
    }
  }
  return out;
}

type Expected = 'none' | 'lcorner' | 'saddle';
function expectedType(tl: number, tr: number, bl: number, br: number): Expected {
  const allSame = tl === tr && tr === bl && bl === br;
  if (allSame) return 'none';
  const diag1Same = tl === br, diag2Same = tr === bl;
  if (diag1Same && diag2Same) return 'saddle';
  const rowsSame = tl === tr && bl === br, colsSame = tl === bl && tr === br;
  if (rowsSame || colsSame) return 'none';
  return 'lcorner';
}

const ANGLES = [0, 15, 30, 45, 60, 75];
const errorsByType: Record<'lcorner' | 'saddle', number[]> = { lcorner: [], saddle: [] };
let failures = 0;

for (const angleDeg of ANGLES) {
  for (let tl = 0; tl <= 1; tl++) for (let tr = 0; tr <= 1; tr++) for (let bl = 0; bl <= 1; bl++) for (let br = 0; br <= 1; br++) {
    const expected = expectedType(tl, tr, bl, br);
    if (expected === 'none') continue; // no discrete point to refine
    const gray = preBlur(rotate(render(tl, tr, bl, br), angleDeg * Math.PI / 180), 2);
    const field = computeJunctionField(gray, BIG, BIG);
    const junctions = detectJunctions(field);
    const coarse = junctions.find(j => (j.x - TRUE_CENTER) ** 2 + (j.y - TRUE_CENTER) ** 2 < 64);
    if (!coarse) { console.log(`FAIL (no coarse detection) angle=${angleDeg} tl${tl}tr${tr}bl${bl}br${br} expected=${expected}`); failures++; continue; }
    const refined = refineJunctionSubPixel(gray, BIG, BIG, coarse.x, coarse.y);
    const err = Math.hypot(refined.x - TRUE_CENTER, refined.y - TRUE_CENTER);
    errorsByType[expected].push(err);
    if (err > 1.5) { console.log(`FAIL (position off by ${err.toFixed(2)}px) angle=${angleDeg} tl${tl}tr${tr}bl${bl}br${br} expected=${expected} coarseErr=${Math.hypot(coarse.x - TRUE_CENTER, coarse.y - TRUE_CENTER).toFixed(2)}`); failures++; }
  }
}

function stats(xs: number[]): string {
  if (!xs.length) return 'n/a';
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  return `mean=${mean.toFixed(3)}px max=${Math.max(...xs).toFixed(3)}px n=${xs.length}`;
}
console.log(`\nlcorner refined-position error: ${stats(errorsByType.lcorner)}`);
console.log(`saddle  refined-position error: ${stats(errorsByType.saddle)}`);
console.log(`\n${failures === 0 ? 'PASS' : `FAIL (${failures} cases over 1.5px error)`}`);
if (failures > 0) process.exit(1);
