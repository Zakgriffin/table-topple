// EXPERIMENT, not a regression test: tests skipping 2D line extraction
// entirely and voting directly from raw per-pixel gradients into a 3D
// "Gaussian sphere" accumulator (Barnard 1983 / Tuytelaars et al.'s Cascaded
// Hough Transform, but collapsed from two stages into one).
//
// Current live pipeline: per-pixel gradients -> 2D (theta,rho) Hough
// accumulator -> discrete peak extraction -> each peak treated as a line ->
// src/orthogonalVp.ts's coarse+refine search over that small line list.
//
// This experiment: EVERY qualifying pixel's (position, gradient angle)
// already fully determines a (theta,rho) for the line through it --
// rho = x*cos(theta) + y*sin(theta), exactly src/lines.ts's own convention
// -- so it can be lifted directly to a 3D plane normal
// n = normalize([cos theta, sin theta, -rho/f]), the same formula
// src/orthogonalVp.ts already uses per LINE, applied per PIXEL instead. No
// 2D Hough stage, no discrete-peak-finding-in-a-sinusoidal-space step at
// all -- votes land straight into a Fibonacci-hemisphere accumulator.
//
// Prediction being tested: pixels on the same real line collapse to (near)
// the same point on the sphere (tight peak, one per visible line -- no
// different from 2D Hough in that respect), but an entire FAMILY of
// parallel lines' peaks all sit on one great circle (the one perpendicular
// to the family's vanishing direction), so a second small search recovers
// that direction. With two orthogonal families, the geometry has a strong
// visible signature: 4 coplanar points (the two anti-podal pole pairs),
// coplanar through the sphere's center.
//
// Usage: node scripts/experiments/spherical-gradient-hough.ts

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus } from '../../src/debruijn.ts';
import { toGrayscale } from '../../src/decode.ts';
import { boxBlur, buildLineAccumulator, findLinePeaksTiered } from '../../src/lines.ts';
import type { LineCandidate } from '../../src/lines.ts';
import { searchOrthogonalVPs } from '../../src/orthogonalVp.ts';
import type { Vec3 } from '../../src/orthogonalVp.ts';
import { captureHomography, buildCamera } from '../lib/synth-camera.ts';
import type { CameraPose } from '../lib/synth-camera.ts';

const order = 4;
const debruijn = generateTorus(order);
const png = PNG.sync.read(readFileSync(`samples/order${order}.png`));
const RAW = 300;
const RESCUE_THRESHOLD_FRACTION = 0.3;

// --- small pure-math helpers, duplicated rather than imported from
// src/orthogonalVp.ts's non-exported internals -- same convention already
// used by scripts/experiments/orthogonal-vp-search.ts. ---
const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize3 = (a: Vec3): Vec3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const scale3 = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const add3 = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

function angleBetweenDeg(a: Vec3, b: Vec3): number {
  return (Math.acos(Math.min(1, Math.abs(dot3(a, b)))) * 180) / Math.PI;
}

function trueAxes(pose: CameraPose): { row: Vec3; col: Vec3 } {
  const { right, up, forward } = buildCamera(pose);
  const row = normalize3([right[0], up[0], forward[0]]);
  const col = normalize3([right[1], up[1], forward[1]]);
  return { row, col };
}

// --- Stage 1 (the new part): every qualifying pixel becomes exactly one
// vote directly in 3D, with NO intermediate 2D accumulator. ---
interface Vote { n: Vec3; weight: number }

function computeGradientVotes(
  gray: Float64Array, w: number, h: number, focalPx: number,
  blurRadius = 1, minMag = 4, gradientRadius = 1,
): Vote[] {
  const blurred = boxBlur(gray, w, h, blurRadius);
  const cx = w / 2, cy = h / 2;
  const r = gradientRadius;
  const votes: Vote[] = [];
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const fx = blurred[i + r] - blurred[i - r];
      const fy = blurred[i + r * w] - blurred[i - r * w];
      const mag = Math.hypot(fx, fy);
      if (mag < minMag) continue;
      let theta = Math.atan2(fy, fx);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      const dx = x - cx, dy = y - cy;
      const rho = dx * Math.cos(theta) + dy * Math.sin(theta);
      const n = normalize3([Math.cos(theta), Math.sin(theta), -rho / focalPx]);
      votes.push({ n, weight: mag });
    }
  }
  return votes;
}

// Fibonacci-hemisphere bucket centers -- identical formula to
// src/orthogonalVp.ts's fibonacciHemisphere (duplicated, not imported: it's
// a private, unexported implementation detail there).
function fibonacciHemisphere(n: number): Vec3[] {
  const pts: Vec3[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / Math.max(1, n - 1)) * 2;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    const p: Vec3 = [Math.cos(theta) * radius, y, Math.sin(theta) * radius];
    pts.push(p[2] < 0 ? scale3(p, -1) : p);
  }
  return pts;
}

// Bins votes into the fixed bucket set via brute-force nearest-neighbor
// (max dot product == min angle) -- a Fibonacci sphere has no closed-form
// "which bucket is this" inverse the way a rectangular grid does. Votes are
// folded into the same z>=0 hemisphere as the buckets first (a line's
// normal has no inherent sign), or they'd never match their true bucket.
function buildSphericalAccumulator(votes: Vote[], buckets: Vec3[]): { weight: Float64Array; count: Int32Array } {
  const weight = new Float64Array(buckets.length);
  const count = new Int32Array(buckets.length);
  for (const { n, weight: w } of votes) {
    const folded = n[2] < 0 ? scale3(n, -1) : n;
    let best = 0, bestDot = -Infinity;
    for (let i = 0; i < buckets.length; i++) {
      const d = dot3(folded, buckets[i]);
      if (d > bestDot) { bestDot = d; best = i; }
    }
    weight[best] += w;
    count[best]++;
  }
  return { weight, count };
}

// Greedy sort-and-dedup peak extraction -- same shape as src/lines.ts's
// findPeakBins, just angular distance instead of (theta,rho) bin distance.
function findSpherePeaks(
  buckets: Vec3[], weight: Float64Array, count: Int32Array,
  thresholdFraction: number, dedupRadiusDeg: number, maxPeaks: number,
): Vote[] {
  let maxW = 0;
  for (const w of weight) if (w > maxW) maxW = w;
  const minW = maxW * thresholdFraction;
  const cosDedup = Math.cos((dedupRadiusDeg * Math.PI) / 180);

  const order = Array.from(weight.keys())
    .filter((i) => weight[i] >= minW && count[i] > 0)
    .sort((a, b) => weight[b] - weight[a]);

  const peaks: Vote[] = [];
  for (const i of order) {
    if (peaks.length >= maxPeaks) break;
    const n = buckets[i];
    if (peaks.some((p) => dot3(p.n, n) > cosDedup)) continue;
    peaks.push({ n, weight: weight[i] });
  }
  return peaks;
}

// --- Stage 2: same coarse Fibonacci-hemisphere sweep + coordinate-descent
// refinement as src/orthogonalVp.ts's searchOrthogonalVPs, duplicated here
// to operate directly on {n,weight} votes -- skips needing to round-trip
// peaks back through a fake (theta,rho) just to satisfy that function's
// LineCandidate-shaped input. ---
function perpBasis(D: Vec3): [Vec3, Vec3] {
  const ref: Vec3 = Math.abs(D[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const e1 = normalize3(cross3(ref, D));
  const e2 = cross3(D, e1);
  return [e1, e2];
}
const dirAtRoll = (e1: Vec3, e2: Vec3, psi: number): Vec3 =>
  normalize3(add3(scale3(e1, Math.cos(psi)), scale3(e2, Math.sin(psi))));
const rotateToward = (D: Vec3, tangent: Vec3, angle: number): Vec3 =>
  normalize3(add3(scale3(D, Math.cos(angle)), scale3(tangent, Math.sin(angle))));

function totalCostFromVotes(votes: Vote[], Drow: Vec3, Dcol: Vec3, deltaRad: number): number {
  const delta2 = deltaRad * deltaRad;
  let cost = 0;
  for (const { n, weight } of votes) {
    const rA = Math.asin(Math.min(1, Math.abs(dot3(n, Drow))));
    const rB = Math.asin(Math.min(1, Math.abs(dot3(n, Dcol))));
    cost += weight * Math.min(rA * rA, rB * rB, delta2);
  }
  return cost;
}

function searchOrthogonalFromVotes(
  votes: Vote[], deltaRad: number, nRowDirs = 150, nPsiSteps = 36, refineRounds = 24,
): { Drow: Vec3; Dcol: Vec3; cost: number } {
  const rowDirs = fibonacciHemisphere(nRowDirs);
  let bestCost = Infinity, bestDrow: Vec3 = rowDirs[0], bestPsi = 0;
  for (const Drow of rowDirs) {
    const [e1, e2] = perpBasis(Drow);
    for (let k = 0; k < nPsiSteps; k++) {
      const psi = (k / nPsiSteps) * Math.PI;
      const Dcol = dirAtRoll(e1, e2, psi);
      const cost = totalCostFromVotes(votes, Drow, Dcol, deltaRad);
      if (cost < bestCost) { bestCost = cost; bestDrow = Drow; bestPsi = psi; }
    }
  }
  let Drow = bestDrow, psi = bestPsi, cost = bestCost;
  let [e1, e2] = perpBasis(Drow);
  let Dcol = dirAtRoll(e1, e2, psi);
  let step = (Math.PI / nRowDirs) * 1.5;
  for (let round = 0; round < refineRounds; round++) {
    let improved = false;
    for (const tangent of [e1, scale3(e1, -1), e2, scale3(e2, -1)]) {
      const cand = rotateToward(Drow, tangent, step);
      const [ce1, ce2] = perpBasis(cand);
      const candDcol = dirAtRoll(ce1, ce2, psi);
      const c = totalCostFromVotes(votes, cand, candDcol, deltaRad);
      if (c < cost) { cost = c; Drow = cand; e1 = ce1; e2 = ce2; Dcol = candDcol; improved = true; }
    }
    for (const dpsi of [step, -step]) {
      const candDcol = dirAtRoll(e1, e2, psi + dpsi);
      const c = totalCostFromVotes(votes, Drow, candDcol, deltaRad);
      if (c < cost) { cost = c; psi += dpsi; Dcol = candDcol; improved = true; }
    }
    step *= improved ? 0.85 : 0.5;
  }
  return { Drow, Dcol, cost };
}

// --- baseline: current live pipeline (2D Hough peaks -> orthogonalVp.ts) ---
function runBaseline(gray: Float64Array, f: number): { Drow: Vec3; Dcol: Vec3; ms: number; nLines: number } {
  const t0 = performance.now();
  const field = buildLineAccumulator(gray, RAW, RAW, 240, 1.5);
  const { strong, weak } = findLinePeaksTiered(field, 0.15, 0.15 * RESCUE_THRESHOLD_FRACTION, 4, 3);
  const lines: LineCandidate[] = [...strong, ...weak];
  const res = searchOrthogonalVPs(lines, f);
  const ms = performance.now() - t0;
  return { Drow: res.Drow, Dcol: res.Dcol, ms, nLines: lines.length };
}

function runScenario(name: string, pose: CameraPose) {
  console.log(`\n=== ${name} (tilt=${((pose.tilt * 180) / Math.PI).toFixed(0)}deg) ===`);
  const rgba = captureHomography(png, pose, RAW, RAW, 4);
  const gray = toGrayscale(rgba, RAW, RAW);
  const f = pose.focal;
  const { row: trueRow, col: trueCol } = trueAxes(pose);

  const base = runBaseline(gray, f);
  const baseErrRow = Math.min(angleBetweenDeg(base.Drow, trueRow), angleBetweenDeg(base.Dcol, trueRow));
  const baseErrCol = Math.min(angleBetweenDeg(base.Drow, trueCol), angleBetweenDeg(base.Dcol, trueCol));
  console.log(`  baseline (2D Hough -> orthogonalVp, ${base.nLines} lines, ${base.ms.toFixed(0)}ms): row err=${baseErrRow.toFixed(2)}deg col err=${baseErrCol.toFixed(2)}deg`);

  const t0 = performance.now();
  const votes = computeGradientVotes(gray, RAW, RAW, f);
  const t1 = performance.now();

  for (const nBuckets of [1500, 6000, 20000]) {
    const buckets = fibonacciHemisphere(nBuckets);
    const t2 = performance.now();
    const { weight, count } = buildSphericalAccumulator(votes, buckets);
    const t3 = performance.now();
    const peaks = findSpherePeaks(buckets, weight, count, 0.12, 2, 400);
    const t4 = performance.now();
    const deltaRad = 6 / f;
    const res = searchOrthogonalFromVotes(peaks, deltaRad);
    const t5 = performance.now();

    const errRow = Math.min(angleBetweenDeg(res.Drow, trueRow), angleBetweenDeg(res.Dcol, trueRow));
    const errCol = Math.min(angleBetweenDeg(res.Drow, trueCol), angleBetweenDeg(res.Dcol, trueCol));
    console.log(`  spherical (${nBuckets} buckets), ${votes.length} votes -> ${peaks.length} peaks:`);
    console.log(`    timing: votes=${(t1 - t0).toFixed(0)}ms bin=${(t3 - t2).toFixed(0)}ms peaks=${(t4 - t3).toFixed(0)}ms search=${(t5 - t4).toFixed(0)}ms`);
    console.log(`    row err=${errRow.toFixed(2)}deg col err=${errCol.toFixed(2)}deg cost=${res.cost.toFixed(4)}`);
    console.log(`    top 6 peak weights: ${peaks.slice(0, 6).map((p) => p.weight.toFixed(0)).join(', ')}`);
  }
}

runScenario('moderate tilt', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.4, azimuth: 0.5, roll: 0.3 });
runScenario('steep tilt (known weak-family case)', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.9, azimuth: 1.0, roll: -0.2 });
runScenario('near-grazing (one axis toward infinity)', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 1.3, azimuth: 0, roll: 0 });
