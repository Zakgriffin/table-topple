// Validates src/lattice.ts (indexFamilyLines + buildLatticeCorrespondences)
// feeding src/homography.ts's EXISTING fitHomographyDLT, under a REAL
// pinhole camera pose (scripts/lib/synth-camera.ts — same ground-truth
// projection this repo's other perspective tests use), rather than a
// hand-built homography matrix.
//
// Row/col lines are constructed directly from the camera's exact projection
// of two points per world grid line (not rendered/detected from pixels —
// Level 1's detection accuracy is validated separately in
// test-hough-lines.ts; this isolates whether indexing+correspondence+DLT
// correctly turns a set of KNOWN-membership lines into a homography that
// extrapolates correctly to an UNSEEN lattice point).
//
// Usage: node scripts/test-lattice-homography.ts

import type { LineCandidate } from '../src/lines.ts';
import type { LineFamily } from '../src/vp.ts';
import { estimateVanishingPoint } from '../src/vp.ts';
import { indexFamilyLines, buildLatticeCorrespondences } from '../src/lattice.ts';
import { fitHomographyDLT, applyHomography } from '../src/homography.ts';
import { projectToImage, type CameraPose } from './lib/synth-camera.ts';

const W = 640, H = 480;
const PITCH = 30;
let pass = 0, fail = 0;
function check(name: string, ok: boolean, detail: string) {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${detail}`);
  if (ok) pass++; else fail++;
}

function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Builds a (theta, rho) LineCandidate from the camera's projection of two
// points on a world line, or null if either point isn't visible (behind the
// camera) — a real, if partial, grid line may legitimately not be usable.
function projectLine(pose: CameraPose, worldA: [number, number], worldB: [number, number], weight: number): LineCandidate | null {
  const pa = projectToImage(pose, W, H, worldA[0], worldA[1]);
  const pb = projectToImage(pose, W, H, worldB[0], worldB[1]);
  if (!pa || !pb) return null;
  const cx = W / 2, cy = H / 2;
  const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
  let theta = Math.atan2(dx, -dy); // normal = tangent rotated -90deg: (dy,-dx) direction... see below
  // Line direction is (dx,dy); its NORMAL is (-dy,dx) (rotate +90deg). Fold to [0,PI).
  let nx = -dy, ny = dx;
  theta = Math.atan2(ny, nx);
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  const a = Math.cos(theta), b = Math.sin(theta);
  const rho = a * (pa[0] - cx) + b * (pa[1] - cy);
  return { theta, rho, weight };
}

function fitAffineIndexMap(pairs: { rec: number; truth: number }[]): { m: number; k: number } {
  // Exact (noiseless) 2-point solve using the extremes — robust enough here
  // since indexFamilyLines' ordering is exact for this clean synthetic input.
  const sorted = pairs.slice().sort((a, b) => a.truth - b.truth);
  const first = sorted[0], last = sorted[sorted.length - 1];
  const m = (last.rec - first.rec) / (last.truth - first.truth);
  const k = first.rec - m * first.truth;
  return { m, k };
}

function runScenario(name: string, pose: CameraPose, noisePx: number, tolerancePx: number, rnd: () => number) {
  const rowTrueIdx = new Map<LineCandidate, number>();
  const colTrueIdx = new Map<LineCandidate, number>();
  const rowLines: LineCandidate[] = [];
  const colLines: LineCandidate[] = [];

  for (let i = -4; i <= 4; i++) {
    const line = projectLine(pose, [-300, i * PITCH], [300, i * PITCH], 1);
    if (!line) continue;
    if (noisePx > 0) line.rho += (rnd() - 0.5) * noisePx;
    rowTrueIdx.set(line, i);
    rowLines.push(line);
  }
  for (let j = -4; j <= 4; j++) {
    const line = projectLine(pose, [j * PITCH, -300], [j * PITCH, 300], 1);
    if (!line) continue;
    if (noisePx > 0) line.rho += (rnd() - 0.5) * noisePx;
    colTrueIdx.set(line, j);
    colLines.push(line);
  }
  if (rowLines.length < 4 || colLines.length < 4) { check(name, false, 'not enough visible lines generated'); return; }

  const vpRow = estimateVanishingPoint(rowLines, W, H);
  const vpCol = estimateVanishingPoint(colLines, W, H);
  const rowFamily: LineFamily = { vp: vpRow, lines: rowLines };
  const colFamily: LineFamily = { vp: vpCol, lines: colLines };

  const rowIndexed = indexFamilyLines(rowFamily, vpCol, W, H);
  const colIndexed = indexFamilyLines(colFamily, vpRow, W, H);

  const rowMap = fitAffineIndexMap(rowIndexed.map(r => ({ rec: r.index, truth: rowTrueIdx.get(r.line)! })));
  const colMap = fitAffineIndexMap(colIndexed.map(c => ({ rec: c.index, truth: colTrueIdx.get(c.line)! })));

  const correspondences = buildLatticeCorrespondences(rowIndexed, colIndexed, W, H);
  const H_fit = fitHomographyDLT(correspondences);
  if (!H_fit) { check(name, false, 'fitHomographyDLT returned null'); return; }

  // Held-out lattice point NOT among the lines used to fit — real
  // extrapolation, not just re-fitting the training points.
  const heldOutI = 6, heldOutJ = -6;
  const truePixel = projectToImage(pose, W, H, heldOutJ * PITCH, heldOutI * PITCH);
  if (!truePixel) { check(name, false, 'held-out point not visible under this pose (pick a milder pose)'); return; }
  const predRowIdx = Math.round(rowMap.m * heldOutI + rowMap.k);
  const predColIdx = Math.round(colMap.m * heldOutJ + colMap.k);
  const predicted = applyHomography(H_fit, predRowIdx, predColIdx);
  if (!predicted) { check(name, false, 'applyHomography returned null'); return; }
  const err = Math.hypot(predicted[0] - truePixel[0], predicted[1] - truePixel[1]);
  const ok = err < tolerancePx;
  check(name, ok, `rows=${rowLines.length} cols=${colLines.length} held-out err=${err.toFixed(2)}px (tol=${tolerancePx})`);
}

runScenario('moderate tilt, exact', {
  targetX: 0, targetY: 0, dist: 400, tilt: 0.4, azimuth: 0.5, roll: 0.3, focal: 500,
}, 0, 0.5, mulberry32(1));

runScenario('moderate tilt, realistic sub-pixel noise', {
  targetX: 0, targetY: 0, dist: 400, tilt: 0.4, azimuth: 0.5, roll: 0.3, focal: 500,
}, 1.5, 8, mulberry32(2));

runScenario('steep tilt, exact', {
  targetX: 0, targetY: 0, dist: 350, tilt: 0.9, azimuth: 1.1, roll: -0.4, focal: 500,
}, 0, 1, mulberry32(3));

runScenario('near-fronto-parallel (both VPs near-infinite), exact', {
  targetX: 0, targetY: 0, dist: 400, tilt: 0.03, azimuth: 0.2, roll: 0.1, focal: 500,
}, 0, 1, mulberry32(4));

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
