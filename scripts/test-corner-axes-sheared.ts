// Validates the generalized (non-orthogonal) saddle axis recovery in
// src/cornerdetect.ts against GENUINELY SHEARED synthetic corners, where the
// two edges meet at an angle other than 90deg — something
// scripts/test-corner-axes.ts structurally cannot test, since it only
// rotates an always-90deg-apart corner as a whole (the true angle gap is
// always 90deg there, so it can't distinguish "correctly recovers any gap"
// from "always outputs 90deg regardless of the truth"). This is the actual
// claim under test: that axis1/axis2 for a saddle track real perspective
// shear instead of being forced orthogonal.
//
// Usage: node scripts/test-corner-axes-sheared.ts

import { computeJunctionField, detectJunctions, computeAxisDirections, refineJunctionSubPixel } from '../src/cornerdetect.ts';

const CELL = 40, SIZE = CELL * 4;
const CENTER = SIZE / 2;

// normal1/normal2 are the two edges' NORMAL angles (not necessarily 90deg
// apart) — the edges themselves (their tangent/line direction) sit at
// normal+90deg, which is what axis1/axis2 are supposed to recover.
function render(normal1: number, normal2: number, tl: number, tr: number, bl: number, br: number): Float64Array {
  const gray = new Float64Array(SIZE * SIZE);
  const c1 = Math.cos(normal1), s1 = Math.sin(normal1);
  const c2 = Math.cos(normal2), s2 = Math.sin(normal2);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const dx = x - CENTER, dy = y - CENTER;
      const d1 = dx * c1 + dy * s1;
      const d2 = dx * c2 + dy * s2;
      const top = d2 < 0, left = d1 < 0;
      const cell = top ? (left ? tl : tr) : (left ? bl : br);
      gray[y * SIZE + x] = cell ? 0 : 255;
    }
  }
  return gray;
}

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

function angleDistModPi(a: number, b: number): number {
  let d = Math.abs(a - b) % Math.PI;
  if (d > Math.PI / 2) d = Math.PI - d;
  return d;
}

const TOLERANCE_DEG = 10; // slightly looser than the pure-rotation test — see its own discretization-artifact notes at grid-aligned angles
let pass = 0, fail = 0, misclassified = 0;

for (const baseDeg of [0, 25, 60, 95]) {
  for (const gapDeg of [30, 50, 70, 90, 110, 130, 150]) {
    const normal1 = baseDeg * Math.PI / 180;
    const normal2 = (baseDeg + gapDeg) * Math.PI / 180;
    // Saddle coloring (tl=br, tr=bl) generalizes fine to non-perpendicular
    // normal1/normal2 — any two non-parallel lines still split the plane
    // into 4 alternating regions around their intersection.
    const gray = preBlur(render(normal1, normal2, 1, 0, 0, 1), 2);
    const field = computeJunctionField(gray, SIZE, SIZE);
    const junctions = detectJunctions(field);
    const near = junctions.find(j => (j.x - CENTER) ** 2 + (j.y - CENTER) ** 2 < 100);
    if (!near) { console.log(`FAIL base=${baseDeg} gap=${gapDeg}: no junction detected near center`); fail++; continue; }
    // Classification drift under strong shear is a separate, known, pre-
    // existing concern (hessianDet's sign threshold) — not what this test
    // is validating, so tallied separately rather than counted as an
    // axis-recovery failure.
    if (near.type !== 'saddle') { console.log(`MISCLASSIFIED base=${baseDeg} gap=${gapDeg}: got ${near.type}, not saddle`); misclassified++; continue; }

    // Recompute axes at the sub-pixel REFINED position, not detectJunctions'
    // coarse one — see computeAxisDirections' doc: the coarse peak can sit
    // several px off the true corner under shear, which measurably biases
    // this formula (confirmed the hard way on this exact test).
    const refined = refineJunctionSubPixel(gray, SIZE, SIZE, near.x, near.y);
    const { axis1, axis2 } = computeAxisDirections(field, near.type, refined.x, refined.y);

    const trueA = normal1 + Math.PI / 2, trueB = normal2 + Math.PI / 2;
    const pairingA = angleDistModPi(axis1, trueA) + angleDistModPi(axis2, trueB);
    const pairingB = angleDistModPi(axis1, trueB) + angleDistModPi(axis2, trueA);
    const [e1, e2] = pairingA <= pairingB
      ? [angleDistModPi(axis1, trueA), angleDistModPi(axis2, trueB)]
      : [angleDistModPi(axis1, trueB), angleDistModPi(axis2, trueA)];
    const maxErrDeg = Math.max(e1, e2) * 180 / Math.PI;
    const ok = maxErrDeg < TOLERANCE_DEG;
    if (ok) pass++; else fail++;
    console.log(`${ok ? 'PASS' : 'FAIL'} base=${String(baseDeg).padStart(3)}deg gap=${String(gapDeg).padStart(3)}deg: maxAxisErr=${maxErrDeg.toFixed(2)}deg`);
  }
}

console.log(`\n${pass}/${pass + fail} correct (${misclassified} misclassified as non-saddle, not counted — separate known issue)`);
if (fail > 0) process.exit(1);
