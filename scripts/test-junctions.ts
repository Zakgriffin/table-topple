// Validates src/cornerdetect.ts's 4-way junction classification against
// EXACT ground truth: constructs all 16 possible 2x2-cell colorings
// directly (no camera/pattern involved — the ground truth topology is known
// by construction), lightly blurs each to mimic real camera softening, and
// checks the classifier buckets each into the right one of the 4 types.
//
// Usage: node scripts/test-junctions.ts

import { computeJunctionField, detectJunctions } from '../src/cornerdetect.ts';

const CELL = 40;
const SIZE = CELL * 2;
const CENTER = CELL; // the junction sits exactly at the 2x2 block's center

function render(tl: number, tr: number, bl: number, br: number): Float64Array {
  const gray = new Float64Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const top = y < CELL, left = x < CELL;
      const cell = top ? (left ? tl : tr) : (left ? bl : br);
      gray[y * SIZE + x] = cell ? 0 : 255; // 1 = black, matching binarize()'s dark=1 convention
    }
  }
  return gray;
}

// Mild pre-blur so the synthetic edge isn't a razor-sharp step (a real
// camera capture never is) — independent of cornerdetect's own internal blur.
function preBlur(gray: Float64Array, radius: number): Float64Array {
  const out = new Float64Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      let sum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= SIZE) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= SIZE) continue;
          sum += gray[yy * SIZE + xx]; count++;
        }
      }
      out[y * SIZE + x] = sum / count;
    }
  }
  return out;
}

type Expected = 'none' | 'lcorner' | 'saddle';

function expectedType(tl: number, tr: number, bl: number, br: number): Expected {
  const allSame = tl === tr && tr === bl && bl === br;
  if (allSame) return 'none'; // flat
  const diag1Same = tl === br, diag2Same = tr === bl;
  if (diag1Same && diag2Same) return 'saddle'; // diagonals match, adjacents differ
  const rowsSame = tl === tr && bl === br;
  const colsSame = tl === bl && tr === br;
  if (rowsSame || colsSame) return 'none'; // straight edge, no 2D-localizable signal
  return 'lcorner'; // exactly one cell differs from the other three
}

let pass = 0, fail = 0;
for (let tl = 0; tl <= 1; tl++) for (let tr = 0; tr <= 1; tr++) for (let bl = 0; bl <= 1; bl++) for (let br = 0; br <= 1; br++) {
  const gray = preBlur(render(tl, tr, bl, br), 2);
  const field = computeJunctionField(gray, SIZE, SIZE);
  const junctions = detectJunctions(field);
  // L-corner peaks land a few px off the true lattice point (see
  // cornerdetect.ts's comment on this — unlike a saddle, a plain corner
  // lacks the 4-fold symmetry that keeps the peak exactly centered; that's
  // what sub-pixel refinement against a proper corner model, not this coarse
  // detector, is for), so the search radius here is generous on purpose.
  const near = junctions.find(j => (j.x - CENTER) ** 2 + (j.y - CENTER) ** 2 < 64);
  const expected = expectedType(tl, tr, bl, br);
  const got: Expected = near ? near.type : 'none';
  const ok = got === expected;
  if (ok) pass++; else fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'} tl=${tl} tr=${tr} bl=${bl} br=${br}  expected=${expected.padEnd(8)} got=${got}${near ? ` (strength=${near.strength.toFixed(3)})` : ''}`);
}
console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
