// Validates fitHomographyDLT in isolation: build a random homography,
// project a set of known lattice points through it to get correspondences,
// fit, and check the fitted H reproduces the same mapping (both on the
// fitting points and on held-out points, to catch overfitting to a
// degenerate/near-collinear point set). Also checks robustness to pixel
// noise on the correspondences, since real corner detections won't be exact.
//
// Usage: node scripts/test-homography.ts

import { fitHomographyDLT, applyHomography } from '../src/homography.ts';
import type { Mat3, PointCorrespondence } from '../src/homography.ts';

function randomHomography(seed: number): Mat3 {
  // A homography derived from a plausible camera-like transform (not
  // arbitrary 8 random numbers, which are usually numerically extreme) —
  // reuse the same spherical-camera parametrization idea as
  // scripts/lib/synth-camera.ts, but inline here to keep this test
  // independent of that module.
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

  // Maps world (X,Y,0) -> image via pinhole projection; expressed as a
  // homography acting on (X,Y,1) directly since Z=0 drops that term.
  // image = rawW/2 + f*(right.(P-cam))/(forward.(P-cam)), similarly y.
  // (P-cam) = (X - camPos.x, Y - camPos.y, -camPos.z), linear in (X,Y,1).
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

function project(H: Mat3, u: number, v: number): [number, number] | null {
  return applyHomography(H, u, v);
}

function runTrial(seed: number, noisePx: number, nFitPoints: number): { fitErr: number; heldOutErr: number } | null {
  const H = randomHomography(seed);
  let rs = (seed * 7919) >>> 0;
  const rng = () => { rs = (rs * 1664525 + 1013904223) >>> 0; return rs / 4294967296; };

  // Fit points: a small local patch (mimics the "local anchor patch"), e.g.
  // a roughly 4x4 to 5x5 region near lattice origin.
  const fitPts: PointCorrespondence[] = [];
  const usedUV = new Set<string>();
  const span = Math.ceil(Math.sqrt(nFitPoints)) + 1;
  outer: for (let v = 0; v < span; v++) {
    for (let u = 0; u < span; u++) {
      if (fitPts.length >= nFitPoints) break outer;
      const img = project(H, u, v);
      if (!img) continue;
      usedUV.add(`${u},${v}`);
      fitPts.push({ u, v, x: img[0] + (rng() - 0.5) * 2 * noisePx, y: img[1] + (rng() - 0.5) * 2 * noisePx });
    }
  }
  if (fitPts.length < 4) return null;

  const Hfit = fitHomographyDLT(fitPts);
  if (!Hfit) return null;

  let fitErrSum = 0;
  for (const p of fitPts) {
    const est = project(Hfit, p.u, p.v);
    if (!est) return null;
    fitErrSum += Math.hypot(est[0] - p.x, est[1] - p.y);
  }

  // Held-out points: further from the fit patch, up to a 16x16 lattice —
  // this is the real test, since the whole point is predicting corners the
  // fit never saw.
  let heldOutErrSum = 0, heldOutCount = 0;
  for (let v = 0; v < 16; v++) {
    for (let u = 0; u < 16; u++) {
      if (usedUV.has(`${u},${v}`)) continue;
      const trueImg = project(H, u, v);
      const estImg = project(Hfit, u, v);
      if (!trueImg || !estImg) continue;
      heldOutErrSum += Math.hypot(trueImg[0] - estImg[0], trueImg[1] - estImg[1]);
      heldOutCount++;
    }
  }
  return { fitErr: fitErrSum / fitPts.length, heldOutErr: heldOutCount ? heldOutErrSum / heldOutCount : NaN };
}

console.log('Homography DLT fit validation (synthetic camera-derived H):\n');
for (const noisePx of [0, 0.5, 1, 2]) {
  for (const nFitPoints of [4, 9, 16]) {
    const fitErrs: number[] = [], heldOutErrs: number[] = [];
    let failures = 0;
    const TRIALS = 30;
    for (let t = 0; t < TRIALS; t++) {
      const r = runTrial(1000 + t, noisePx, nFitPoints);
      if (!r) { failures++; continue; }
      fitErrs.push(r.fitErr);
      heldOutErrs.push(r.heldOutErr);
    }
    const mean = (a: number[]) => a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN;
    console.log(
      `noise=${noisePx}px nFitPts=${nFitPoints}: ` +
      `fitErr mean=${mean(fitErrs).toFixed(3)}px, heldOutErr mean=${mean(heldOutErrs).toFixed(3)}px, failures=${failures}/${TRIALS}`
    );
  }
}

console.log('\nControl: does a WIDER-spread fit patch fix the extrapolation blowup?');
function runWideTrial(seed: number, noisePx: number): { fitErr: number; heldOutErr: number } | null {
  const H = randomHomography(seed);
  let rs = (seed * 7919) >>> 0;
  const rng = () => { rs = (rs * 1664525 + 1013904223) >>> 0; return rs / 4294967296; };
  const fitPts: PointCorrespondence[] = [];
  const usedUV = new Set<string>();
  // Same point COUNT as the nFitPts=9 case above, but spread across the
  // whole 16x16 lattice instead of clustered near the origin.
  for (const [u, v] of [[0,0],[15,0],[0,15],[15,15],[7,7],[3,12],[12,3],[7,0],[0,7]]) {
    const img = project(H, u, v);
    if (!img) continue;
    usedUV.add(`${u},${v}`);
    fitPts.push({ u, v, x: img[0] + (rng() - 0.5) * 2 * noisePx, y: img[1] + (rng() - 0.5) * 2 * noisePx });
  }
  const Hfit = fitHomographyDLT(fitPts);
  if (!Hfit) return null;
  let fitErrSum = 0;
  for (const p of fitPts) { const est = project(Hfit, p.u, p.v); if (!est) return null; fitErrSum += Math.hypot(est[0]-p.x, est[1]-p.y); }
  let heldOutErrSum = 0, heldOutCount = 0;
  for (let v = 0; v < 16; v++) for (let u = 0; u < 16; u++) {
    if (usedUV.has(`${u},${v}`)) continue;
    const trueImg = project(H, u, v), estImg = project(Hfit, u, v);
    if (!trueImg || !estImg) continue;
    heldOutErrSum += Math.hypot(trueImg[0]-estImg[0], trueImg[1]-estImg[1]); heldOutCount++;
  }
  return { fitErr: fitErrSum / fitPts.length, heldOutErr: heldOutCount ? heldOutErrSum / heldOutCount : NaN };
}
for (const noisePx of [0, 0.5, 1, 2]) {
  const fitErrs: number[] = [], heldOutErrs: number[] = [];
  let failures = 0;
  for (let t = 0; t < 30; t++) {
    const r = runWideTrial(1000 + t, noisePx);
    if (!r) { failures++; continue; }
    fitErrs.push(r.fitErr); heldOutErrs.push(r.heldOutErr);
  }
  const mean = (a: number[]) => a.length ? a.reduce((s,v)=>s+v,0)/a.length : NaN;
  console.log(`noise=${noisePx}px wideSpread(9pts): fitErr mean=${mean(fitErrs).toFixed(3)}px, heldOutErr mean=${mean(heldOutErrs).toFixed(3)}px, failures=${failures}/30`);
}
