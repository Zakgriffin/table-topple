// Characterizes how the current decode pipeline (src/decode.ts, rotation +
// uniform scale only — see its module header) degrades under REAL
// perspective, using scripts/lib/synth-camera.ts's pinhole-projection
// synthetic captures rather than scripts/test-decode.ts's rotation-only
// crops. This is the baseline "before" measurement for Option B (corner/
// junction-mesh detection): it quantifies the tilt angle at which a single
// global pitch estimate stops being a good enough model, which is the actual
// problem corner detection needs to solve.
//
// Usage: node scripts/test-perspective.ts [order]

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus, buildLookupTable } from '../src/debruijn.ts';
import { detectGrid, sampleFullGrid, pickBestCandidate, toGrayscale, binarize, estimateRotationRad, asSignedResidual } from '../src/decode.ts';
import { captureHomography } from './lib/synth-camera.ts';
import type { CameraPose } from './lib/synth-camera.ts';

const order = parseInt(process.argv[2] ?? '4', 10);
const pngPath = `samples/order${order}.png`;

console.log(`Building order-${order} torus + lookup table...`);
const debruijn = generateTorus(order);
const lookup = buildLookupTable(debruijn);
const { R, C } = debruijn;

const png = PNG.sync.read(readFileSync(pngPath));
const cellPx = png.width / C;
console.log(`Loaded ${pngPath}: ${png.width}x${png.height}px, ${cellPx}px/cell`);

const RAW = 360;
const ALIGNED = 240;
const CONFIDENCE_THRESHOLD = 0.85;

// Manual nearest-neighbor derotation, mirroring the browser's canvas
// ctx.rotate(-theta) + drawImage (see src/decode.ts's module comment and
// scripts/test-decode.ts's identical copy of this function).
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

function decode(rgba: Uint8ClampedArray): { match: { row: number; col: number } | null; consistency: number } {
  const rawGray = toGrayscale(rgba, RAW, RAW);
  const thetaCoarse = estimateRotationRad(rawGray, RAW, RAW);
  const previewGray = derotate(rawGray, thetaCoarse);
  const residual = asSignedResidual(estimateRotationRad(previewGray, ALIGNED, ALIGNED));
  const theta0 = thetaCoarse + residual;

  const sampledGrids = [0, 1, 2, 3].map(k => {
    const theta = theta0 + k * (Math.PI / 2);
    const alignedBin = binarize(derotate(rawGray, theta));
    const grid = detectGrid(alignedBin, ALIGNED, ALIGNED);
    return sampleFullGrid(alignedBin, ALIGNED, ALIGNED, grid);
  });

  const { match, consistency } = pickBestCandidate(sampledGrids, order, lookup, debruijn.torus, R, C);
  return { match, consistency };
}

function within(target: number, start: number, span: number, mod: number): boolean {
  const rel = ((target - start) % mod + mod) % mod;
  return rel <= span || rel >= mod - 1;
}

// dist == focal keeps magnification 1:1 at the image center regardless of
// tilt (apparent pitch = focal/dist * cellPx), so results are comparable
// across tilt levels and to test-decode.ts's fixed cellPx-based crop.
const DIST = 300;
const FOCAL = 300;
const TRIALS_PER_TILT = 50;
const TILT_DEGREES = [0, 10, 20, 30, 40, 50, 60];

console.log(`\nPerspective tolerance sweep (${TRIALS_PER_TILT} trials/tilt, random azimuth+roll+target):`);
for (const tiltDeg of TILT_DEGREES) {
  let hits = 0, misses = 0, wrong = 0;
  const scores: number[] = [];
  for (let t = 0; t < TRIALS_PER_TILT; t++) {
    const testRow = Math.floor(Math.random() * R);
    const testCol = Math.floor(Math.random() * C);
    const pose: CameraPose = {
      targetX: testCol * cellPx + cellPx / 2,
      targetY: testRow * cellPx + cellPx / 2,
      dist: DIST, focal: FOCAL,
      tilt: tiltDeg * Math.PI / 180,
      azimuth: Math.random() * 2 * Math.PI,
      roll: Math.random() * 2 * Math.PI,
    };
    const rgba = captureHomography(png, pose, RAW, RAW);
    const { match, consistency } = decode(rgba);
    if (!match || consistency < CONFIDENCE_THRESHOLD) { misses++; continue; }
    scores.push(consistency);
    if (within(testRow, match.row, order, R) && within(testCol, match.col, order, C)) hits++;
    else wrong++;
  }
  const meanScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3) : 'n/a';
  console.log(`  tilt ${String(tiltDeg).padStart(2)}deg: ${hits}/${TRIALS_PER_TILT} correct, ${misses} no-lock, ${wrong} wrong, mean score ${meanScore}`);
}
