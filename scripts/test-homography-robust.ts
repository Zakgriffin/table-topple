// Validates fitHomographyRobust's outlier-rejection behavior in isolation:
// a handful of grossly-wrong correspondences (simulating a crossing built
// from a badly-mispositioned or wrong-family line, which real Hough
// detection occasionally produces) should get rejected, recovering a fit
// close to the clean baseline -- unlike plain fitHomographyDLT, which has no
// such protection and lets a few bad points drag the whole fit off.
//
// Found necessary via real end-to-end testing (scripts/test-lines-decode.ts):
// every individual pipeline stage tests as accurate against clean synthetic
// input, yet the full chain on REAL rendered/detected lines showed 5-20px of
// homography positional error that isolated per-stage tests never surfaced —
// a few weak/noisy correspondences having outsized, unweighted leverage in
// the final DLT fit was the most parsimonious explanation.
//
// Correspondence counts here match src/lattice.ts's buildLatticeCorrespondences
// at realistic scale (every row-line x col-line crossing -- a well-detected
// 10-15 cell grid gives 100-225 of them), NOT a small hand-picked point set:
// a first attempt at this test used a tiny (12-point) "wide-spread" set
// borrowed from test-homography.ts's unrelated small-vs-clustered comparison,
// and found that 2 gross outliers among only 12 points (8 DOF) can drag the
// INITIAL fit enough that some genuinely clean points end up with LARGER
// residuals than the actual outliers -- an honest limitation at that scale,
// but not the scale this function is actually used at (see Scenario 3 below,
// which documents that small-N case directly rather than hiding it).
//
// Usage: node scripts/test-homography-robust.ts

import { fitHomographyDLT, fitHomographyRobust, applyHomography } from '../src/homography.ts';
import type { Mat3, PointCorrespondence } from '../src/homography.ts';

let pass = 0, fail = 0;
function check(name: string, ok: boolean, detail: string) {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${detail}`);
  if (ok) pass++; else fail++;
}

// Same camera-derived random-homography generator as test-homography.ts,
// duplicated rather than imported to keep each test file independent.
function randomHomography(seed: number): Mat3 {
  let s = seed >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const tilt = rng() * 60 * Math.PI / 180;
  const azimuth = rng() * 2 * Math.PI;
  const roll = rng() * 2 * Math.PI;
  const dist = 250 + rng() * 100;
  const focal = 250 + rng() * 100;

  const camPos = [dist * Math.sin(tilt) * Math.cos(azimuth), dist * Math.sin(tilt) * Math.sin(azimuth), dist * Math.cos(tilt)];
  const forward = normalize3([-camPos[0], -camPos[1], -camPos[2]]);
  let right = normalize3(cross3(forward, [0, 1, 0]));
  let up = cross3(right, forward);
  const cosR = Math.cos(roll), sinR = Math.sin(roll);
  const right2 = [right[0] * cosR + up[0] * sinR, right[1] * cosR + up[1] * sinR, right[2] * cosR + up[2] * sinR];
  const up2 = [-right[0] * sinR + up[0] * cosR, -right[1] * sinR + up[1] * cosR, -right[2] * sinR + up[2] * cosR];
  right = right2; up = up2;

  const f = focal, rawW = 300, rawH = 300;
  const rowX = [right[0], right[1], -right[0] * camPos[0] - right[1] * camPos[1] - right[2] * camPos[2]];
  const rowY = [up[0], up[1], -up[0] * camPos[0] - up[1] * camPos[1] - up[2] * camPos[2]];
  const rowZ = [forward[0], forward[1], -forward[0] * camPos[0] - forward[1] * camPos[1] - forward[2] * camPos[2]];

  const row0 = rowX.map((v, i) => f * v + (rawW / 2) * rowZ[i]);
  const row1 = rowY.map((v, i) => f * v + (rawH / 2) * rowZ[i]);
  return new Float64Array([...row0, ...row1, ...rowZ]);
}
function normalize3(a: number[]): number[] { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; }
function cross3(a: number[], b: number[]): number[] { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

function heldOutError(H: Mat3, Hfit: Mat3, usedUV: Set<string>, span: number): number {
  let sum = 0, count = 0;
  for (let v = 0; v < span; v++) {
    for (let u = 0; u < span; u++) {
      if (usedUV.has(`${u},${v}`)) continue;
      const truePt = applyHomography(H, u, v), estPt = applyHomography(Hfit, u, v);
      if (!truePt || !estPt) continue;
      sum += Math.hypot(truePt[0] - estPt[0], truePt[1] - estPt[1]);
      count++;
    }
  }
  return count ? sum / count : NaN;
}

// Builds a FULL gridSize x gridSize correspondence grid (matching real
// buildLatticeCorrespondences scale -- every row x col crossing, not a
// hand-picked sparse set), lightly noised, with outlierFrac of them
// corrupted by a gross positional error in a random direction.
function buildGrid(H: Mat3, seed: number, gridSize: number, noisePx: number, outlierFrac: number, outlierPx: number): { pts: PointCorrespondence[]; usedUV: Set<string> } {
  let rs = (seed * 7919 + 12345) >>> 0;
  const rng = () => { rs = (rs * 1664525 + 1013904223) >>> 0; return rs / 4294967296; };
  const pts: PointCorrespondence[] = [];
  const usedUV = new Set<string>();
  for (let v = 0; v < gridSize; v++) {
    for (let u = 0; u < gridSize; u++) {
      const img = applyHomography(H, u, v);
      if (!img) continue;
      usedUV.add(`${u},${v}`);
      pts.push({ u, v, x: img[0] + (rng() - 0.5) * 2 * noisePx, y: img[1] + (rng() - 0.5) * 2 * noisePx });
    }
  }
  const nOutliers = Math.round(pts.length * outlierFrac);
  for (let i = 0; i < nOutliers; i++) {
    const idx = Math.floor(rng() * pts.length);
    const angle = rng() * 2 * Math.PI;
    pts[idx].x += Math.cos(angle) * outlierPx;
    pts[idx].y += Math.sin(angle) * outlierPx;
  }
  return { pts, usedUV };
}

console.log('fitHomographyRobust outlier-rejection validation:\n');

// --- Scenario 1: clean (no outliers), realistic grid scale -- robust must
// not regress an already-good fit.
{
  const dltErrs: number[] = [], robustErrs: number[] = [];
  for (let t = 0; t < 15; t++) {
    const H = randomHomography(1000 + t);
    const { pts, usedUV } = buildGrid(H, t, 12, 0.5, 0, 0);
    const HfitDLT = fitHomographyDLT(pts), HfitRobust = fitHomographyRobust(pts);
    if (!HfitDLT || !HfitRobust) continue;
    dltErrs.push(heldOutError(H, HfitDLT, usedUV, 16));
    robustErrs.push(heldOutError(H, HfitRobust, usedUV, 16));
  }
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const dltMean = mean(dltErrs), robustMean = mean(robustErrs);
  check('clean 12x12 grid: robust does not regress plain DLT', robustMean <= dltMean * 1.5, `DLT=${dltMean.toFixed(3)}px robust=${robustMean.toFixed(3)}px`);
}

// --- Scenario 2: realistic grid scale (100-225 correspondences, matching
// buildLatticeCorrespondences on a well-detected grid) with a small fraction
// of gross outliers -- this is the actual regime the fix targets.
for (const [gridSize, outlierFrac] of [[10, 0.02], [10, 0.05], [10, 0.1], [15, 0.05]] as const) {
  const dltErrs: number[] = [], robustErrs: number[] = [];
  const TRIALS = 15;
  for (let t = 0; t < TRIALS; t++) {
    const H = randomHomography(2000 + t);
    const { pts, usedUV } = buildGrid(H, t, gridSize, 0.5, outlierFrac, 40);
    const HfitDLT = fitHomographyDLT(pts), HfitRobust = fitHomographyRobust(pts);
    if (!HfitDLT || !HfitRobust) continue;
    dltErrs.push(heldOutError(H, HfitDLT, usedUV, gridSize + 4));
    robustErrs.push(heldOutError(H, HfitRobust, usedUV, gridSize + 4));
  }
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const dltMean = mean(dltErrs), robustMean = mean(robustErrs);
  check(
    `${gridSize}x${gridSize} grid (${gridSize * gridSize}pts), ${(outlierFrac * 100).toFixed(0)}% outliers (40px): robust recovers a clean-level fit`,
    robustMean < 1 && robustMean < dltMean * 0.2,
    `DLT mean heldOutErr=${dltMean.toFixed(2)}px, robust mean heldOutErr=${robustMean.toFixed(2)}px (n=${dltErrs.length}/${TRIALS})`
  );
}

// --- Scenario 3: documented limitation -- a SMALL, sparse correspondence
// set (12 points, 8 DOF) with a couple of gross outliers is genuinely hard:
// the contaminated initial fit can distort enough that some clean points end
// up with LARGER residuals than the actual outliers. Not the regime this
// function is used in (real correspondence counts are 100+), but recorded
// here as an honest characterization rather than silently assumed away.
{
  const uvs: [number, number][] = [[0, 0], [15, 0], [0, 15], [15, 15], [7, 7], [3, 12], [12, 3], [7, 0], [0, 7], [10, 4], [4, 10], [13, 13]];
  const dltErrs: number[] = [], robustErrs: number[] = [];
  for (let t = 0; t < 20; t++) {
    const H = randomHomography(3000 + t);
    let rs = (t * 7919) >>> 0;
    const rng = () => { rs = (rs * 1664525 + 1013904223) >>> 0; return rs / 4294967296; };
    const pts: PointCorrespondence[] = [];
    const usedUV = new Set<string>();
    for (const [u, v] of uvs) {
      const img = applyHomography(H, u, v);
      if (!img) continue;
      usedUV.add(`${u},${v}`);
      pts.push({ u, v, x: img[0] + (rng() - 0.5) * 1, y: img[1] + (rng() - 0.5) * 1 });
    }
    for (let i = 0; i < 2; i++) {
      const p = pts[pts.length - 1 - i];
      const angle = rng() * 2 * Math.PI;
      p.x += Math.cos(angle) * 40; p.y += Math.sin(angle) * 40;
    }
    const HfitDLT = fitHomographyDLT(pts), HfitRobust = fitHomographyRobust(pts);
    if (!HfitDLT || !HfitRobust) continue;
    dltErrs.push(heldOutError(H, HfitDLT, usedUV, 16));
    robustErrs.push(heldOutError(H, HfitRobust, usedUV, 16));
  }
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const dltMean = mean(dltErrs), robustMean = mean(robustErrs);
  // Not asserting robust is dramatically better here -- only that it never
  // makes things WORSE than plain DLT on average, since even a partial
  // improvement is worth having even in this admittedly hard small-N case.
  check('12-point sparse set, 2 gross outliers: robust at least as good as plain DLT (known-hard case)', robustMean <= dltMean, `DLT=${dltMean.toFixed(2)}px robust=${robustMean.toFixed(2)}px`);
}

// --- Scenario 4: too few correspondences to safely reject any (< 8) --
// robust must fall back to the plain fit rather than aggressively discarding
// points it can't afford to lose.
{
  const H = randomHomography(4000);
  const { pts } = buildGrid(H, 0, 3, 0, 0, 0); // 3x3 = 9... trim to 6 to force the <8 fallback path
  const fewPts = pts.slice(0, 6);
  const HfitDLT = fitHomographyDLT(fewPts);
  const HfitRobust = fitHomographyRobust(fewPts);
  const same = !!HfitDLT && !!HfitRobust && HfitDLT.every((v, i) => Math.abs(v - HfitRobust[i]) < 1e-9);
  check('fewer than 8 correspondences: robust falls back to plain DLT unchanged', same, `HfitDLT and HfitRobust ${same ? 'match exactly' : 'differ'}`);
}

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
