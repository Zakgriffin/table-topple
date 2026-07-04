// End-to-end validation of the homography-based mesh pipeline (corner
// detect -> refine -> estimateVanishingPoints -> buildMeshViaHomography ->
// sampleFromMesh -> pickBestCandidate) against real perspective tilt —
// directly comparable to scripts/test-mesh-decode.ts, which measured the
// same thing for the old BFS-walk mesh (0/8 correct at every tilt tested,
// blocked by the "coherent region drift" bug). This tells us whether the
// homography approach actually resolves that, not just whether its pieces
// are individually accurate in isolation.
//
// Usage: node scripts/test-homography-decode.ts [order]

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus, buildLookupTable } from '../src/debruijn.ts';
import { toGrayscale, binarize, detectGrid, estimateRotationRad, sampleFromMesh, pickBestCandidate } from '../src/decode.ts';
import { computeJunctionField, detectJunctions, refineJunctionSubPixel } from '../src/cornerdetect.ts';
import { estimateVanishingPoints } from '../src/vanishing.ts';
import { buildMeshViaHomography } from '../src/rectify.ts';
import { captureHomography } from './lib/synth-camera.ts';
import type { CameraPose } from './lib/synth-camera.ts';

const order = parseInt(process.argv[2] ?? '4', 10);
const debruijn = generateTorus(order);
const lookup = buildLookupTable(debruijn);
const { R, C } = debruijn;

const png = PNG.sync.read(readFileSync(`samples/order${order}.png`));
const cellPx = png.width / C;
console.log(`Loaded samples/order${order}.png: ${cellPx}px/cell, torus ${R}x${C}`);

const RAW = 300;
const DIST = 300, FOCAL = 300;
const CONFIDENCE_THRESHOLD = 0.85;
const TRIALS = 8;

const PATCH = 120;
function derotatePatch(gray: Float64Array, w: number, h: number, theta: number): Float64Array {
  const out = new Float64Array(PATCH * PATCH);
  const cosT = Math.cos(theta), sinT = Math.sin(theta), cx = w / 2, cy = h / 2;
  for (let ay = 0; ay < PATCH; ay++) {
    const relY = ay - PATCH / 2;
    for (let ax = 0; ax < PATCH; ax++) {
      const relX = ax - PATCH / 2;
      const sx = Math.round(relX * cosT - relY * sinT + cx);
      const sy = Math.round(relX * sinT + relY * cosT + cy);
      out[ay * PATCH + ax] = (sx >= 0 && sx < w && sy >= 0 && sy < h) ? gray[sy * w + sx] : 255;
    }
  }
  return out;
}

interface DecodeOutcome { match: { row: number; col: number } | null; consistency: number; }

function decodeViaHomographyMesh(pose: CameraPose): DecodeOutcome | 'novp' | 'noseed' | 'nogrid' {
  // supersampleN=4: raw per-pixel gradient angle (vanishing.ts) is much more
  // exposed to nearest-neighbor staircase aliasing than junction detection's
  // windowed structure tensor was — see src/vanishing.ts's docs and
  // scripts/test-vanishing.ts's discovery of this. A real camera antialiases
  // optically, so this matches real-world conditions rather than working
  // around a synthetic-only artifact.
  const rgba = captureHomography(png, pose, RAW, RAW, 4);
  const gray = toGrayscale(rgba, RAW, RAW);
  const bin = binarize(gray);

  const theta0 = estimateRotationRad(gray, RAW, RAW);
  const patchBin = binarize(derotatePatch(gray, RAW, RAW, theta0));
  const coarseGrid = detectGrid(patchBin, PATCH, PATCH);

  const apparentPitch = (coarseGrid.pitchX + coarseGrid.pitchY) / 2;
  const tensorRadius = Math.max(2, Math.round(apparentPitch / 4));
  const minDistance = Math.max(5, Math.round(tensorRadius * 2.5));

  const field = computeJunctionField(gray, RAW, RAW, 1, tensorRadius);
  const coarseJ = detectJunctions(field, 0.15, minDistance);
  if (coarseJ.length < 5) return 'nogrid';
  const junctions = coarseJ.map(j => {
    const r = refineJunctionSubPixel(gray, RAW, RAW, j.x, j.y);
    return { x: r.x, y: r.y, type: j.type };
  });

  const vp = estimateVanishingPoints(gray, RAW, RAW);
  if (!vp) return 'novp';

  const result = buildMeshViaHomography(
    junctions, RAW / 2, RAW / 2, coarseGrid.pitchX, coarseGrid.pitchY, -theta0,
    vp.vp1, vp.vp2,
  );
  if (!result) return 'noseed';

  const sg = sampleFromMesh(bin, RAW, RAW, result.mesh);
  if (sg.rows < order || sg.cols < order) return 'noseed';

  return pickBestCandidate([sg], order, lookup, debruijn.torus, R, C);
}

function within(target: number, start: number, span: number, mod: number): boolean {
  const rel = ((target - start) % mod + mod) % mod;
  return rel <= span || rel >= mod - 1;
}

console.log(`\nEnd-to-end homography-mesh decode vs perspective tilt (${TRIALS} trials/tilt):`);
for (const tiltDeg of [0, 10, 20, 30, 40, 50, 60]) {
  let hits = 0, misses = 0, wrong = 0, novp = 0, noseed = 0, nogrid = 0;
  const scores: number[] = [];
  for (let t = 0; t < TRIALS; t++) {
    const testRow = Math.floor(Math.random() * R);
    const testCol = Math.floor(Math.random() * C);
    const pose: CameraPose = {
      targetX: testCol * cellPx + cellPx / 2, targetY: testRow * cellPx + cellPx / 2,
      dist: DIST, focal: FOCAL, tilt: tiltDeg * Math.PI / 180,
      azimuth: Math.random() * 2 * Math.PI, roll: Math.random() * 2 * Math.PI,
    };
    const result = decodeViaHomographyMesh(pose);
    if (result === 'novp') { novp++; continue; }
    if (result === 'noseed') { noseed++; continue; }
    if (result === 'nogrid') { nogrid++; continue; }
    if (!result.match || result.consistency < CONFIDENCE_THRESHOLD) { misses++; continue; }
    scores.push(result.consistency);
    if (within(testRow, result.match.row, order, R) && within(testCol, result.match.col, order, C)) hits++;
    else wrong++;
  }
  const meanScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3) : 'n/a';
  console.log(
    `  tilt ${String(tiltDeg).padStart(2)}deg: ${hits}/${TRIALS} correct, ${misses} no-lock, ${wrong} wrong, ` +
    `novp=${novp} noseed=${noseed} nogrid=${nogrid}, mean score ${meanScore}`
  );
}
