// Validates the new Junction.axis1/axis2 fields (src/cornerdetect.ts): the
// structure tensor's eigenvector angles at a detected corner, which should
// approximate the corner's two local edge directions. Renders synthetic
// L-corner and saddle patterns at known rotations (edge directions are
// exactly {theta, theta+90deg} mod 180 by construction) and checks the
// detected axes match, since real image classification tests
// (scripts/test-junctions.ts) don't cover this — a corner can be classified
// correctly while its axis angles are wrong.
//
// Usage: node scripts/test-corner-axes.ts

import { computeJunctionField, detectJunctions } from '../src/cornerdetect.ts';

const CELL = 40, SIZE = CELL * 4;
const CENTER = SIZE / 2;

function render(theta: number, tl: number, tr: number, bl: number, br: number): Float64Array {
  const gray = new Float64Array(SIZE * SIZE);
  const cosT = Math.cos(theta), sinT = Math.sin(theta);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const dx = x - CENTER, dy = y - CENTER;
      // Rotate into the pattern's own canonical (unrotated) frame to decide
      // which quadrant this pixel falls in — equivalent to rendering the
      // corner pattern rotated by +theta in the image.
      const rx = dx * cosT + dy * sinT;
      const ry = -dx * sinT + dy * cosT;
      const top = ry < 0, left = rx < 0;
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

const TOLERANCE_DEG = 8;
let pass = 0, fail = 0;

for (const [name, tl, tr, bl, br] of [
  ['lcorner', 1, 0, 0, 0],
  ['saddle', 1, 0, 0, 1],
] as const) {
  for (let thetaDeg = 0; thetaDeg < 180; thetaDeg += 10) {
    const theta = thetaDeg * Math.PI / 180;
    const gray = preBlur(render(theta, tl, tr, bl, br), 2);
    const field = computeJunctionField(gray, SIZE, SIZE);
    const junctions = detectJunctions(field);
    const near = junctions.find(j => (j.x - CENTER) ** 2 + (j.y - CENTER) ** 2 < 100);
    if (!near) { console.log(`FAIL ${name} theta=${thetaDeg}deg: no junction detected near center`); fail++; continue; }

    const trueA = theta, trueB = theta + Math.PI / 2;
    const pairingA = angleDistModPi(near.axis1, trueA) + angleDistModPi(near.axis2, trueB);
    const pairingB = angleDistModPi(near.axis1, trueB) + angleDistModPi(near.axis2, trueA);
    const [e1, e2] = pairingA <= pairingB
      ? [angleDistModPi(near.axis1, trueA), angleDistModPi(near.axis2, trueB)]
      : [angleDistModPi(near.axis1, trueB), angleDistModPi(near.axis2, trueA)];
    const maxErrDeg = Math.max(e1, e2) * 180 / Math.PI;
    const ok = maxErrDeg < TOLERANCE_DEG;
    if (ok) pass++; else fail++;
    console.log(`${ok ? 'PASS' : 'FAIL'} ${name.padEnd(8)} theta=${String(thetaDeg).padStart(3)}deg: maxAxisErr=${maxErrDeg.toFixed(2)}deg`);
  }
}

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
