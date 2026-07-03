// End-to-end validation of the full mesh-based decode path (corner detect ->
// refine -> mesh -> sampleFromMesh -> pickBestCandidate) against real
// perspective tilt — directly comparable to scripts/test-perspective.ts's
// sweep, which showed the old rotation+uniform-scale pipeline collapsing to
// 0/50 correct by 20deg tilt. This is the actual "does the decoded position
// come out right" test; scripts/test-mesh-perspective.ts only checked mesh
// geometry accuracy, not full bit-decoding.
//
// Usage: node scripts/test-mesh-decode.ts [order]

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus, buildLookupTable } from '../src/debruijn.ts';
import { toGrayscale, binarize, detectGrid, estimateRotationRad, sampleFromMesh, pickBestCandidate } from '../src/decode.ts';
import { computeJunctionField, detectJunctions, refineJunctionSubPixel } from '../src/cornerdetect.ts';
import { buildMesh } from '../src/mesh.ts';
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

// Same derotation-for-seed-pitch trick as scripts/test-mesh-perspective.ts —
// see that file for why detectGrid's pitch estimate needs derotating first.
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

function decodeViaMesh(pose: CameraPose): { match: { row: number; col: number } | null; consistency: number } | null {
  const rgba = captureHomography(png, pose, RAW, RAW);
  const gray = toGrayscale(rgba, RAW, RAW);
  const bin = binarize(gray);

  const theta0 = estimateRotationRad(gray, RAW, RAW);
  const patchBin = binarize(derotatePatch(gray, RAW, RAW, theta0));
  const coarseGrid = detectGrid(patchBin, PATCH, PATCH);

  const apparentPitch = (coarseGrid.pitchX + coarseGrid.pitchY) / 2;
  const tensorRadius = Math.max(1, Math.round(apparentPitch / 10));
  const minDistance = Math.max(2, Math.round(apparentPitch / 4));

  const field = computeJunctionField(gray, RAW, RAW, 1, tensorRadius);
  const coarseJ = detectJunctions(field, 0.15, minDistance);
  if (coarseJ.length < 5) return null;
  const junctions = coarseJ.map(j => {
    const r = refineJunctionSubPixel(gray, RAW, RAW, j.x, j.y);
    return { x: r.x, y: r.y, type: j.type };
  });

  const mesh = buildMesh(junctions, RAW / 2, RAW / 2, coarseGrid.pitchX, coarseGrid.pitchY, -theta0);
  if (mesh.nodes.length < order * order) return null;

  const sg = sampleFromMesh(bin, RAW, RAW, mesh);
  if (sg.rows < order || sg.cols < order) return null;

  return pickBestCandidate([sg], order, lookup, debruijn.torus, R, C);
}

function within(target: number, start: number, span: number, mod: number): boolean {
  const rel = ((target - start) % mod + mod) % mod;
  return rel <= span || rel >= mod - 1;
}

console.log(`\nEnd-to-end mesh decode vs perspective tilt (${TRIALS} trials/tilt):`);
for (const tiltDeg of [0, 10, 20, 30, 40, 50, 60]) {
  let hits = 0, misses = 0, wrong = 0;
  const scores: number[] = [];
  for (let t = 0; t < TRIALS; t++) {
    const testRow = Math.floor(Math.random() * R);
    const testCol = Math.floor(Math.random() * C);
    const pose: CameraPose = {
      targetX: testCol * cellPx + cellPx / 2, targetY: testRow * cellPx + cellPx / 2,
      dist: DIST, focal: FOCAL, tilt: tiltDeg * Math.PI / 180,
      azimuth: Math.random() * 2 * Math.PI, roll: Math.random() * 2 * Math.PI,
    };
    const result = decodeViaMesh(pose);
    if (!result || !result.match || result.consistency < CONFIDENCE_THRESHOLD) { misses++; continue; }
    scores.push(result.consistency);
    if (within(testRow, result.match.row, order, R) && within(testCol, result.match.col, order, C)) hits++;
    else wrong++;
  }
  const meanScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3) : 'n/a';
  console.log(`  tilt ${String(tiltDeg).padStart(2)}deg: ${hits}/${TRIALS} correct, ${misses} no-lock, ${wrong} wrong, mean score ${meanScore}`);
}
