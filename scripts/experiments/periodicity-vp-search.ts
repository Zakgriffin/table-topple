// EXPERIMENT, not wired into anything: tests whether the "meta-function"
// (see the conversation this implements -- l'_m = A - m*B is affine-linear
// in the true integer grid index m, making tan(theta_m) exactly Mobius in m)
// can be used as an ADDITIONAL scoring signal on top of src/orthogonalVp.ts's
// existing direction search, specifically targeting its one known failure:
// a genuinely sparse weak family can pick the wrong roll angle (psi) because
// truncated-squared-residual cost can't distinguish "a few real points,
// correctly placed" from "diffuse noise near some other angle" -- it only
// asks "how close", never "does this look like a periodic sequence".
//
// Requires two things nothing else in the pipeline needs: (1) a known real
// -world cell size (cellSize -- here taken from the test PNG exactly, in a
// real deployment this would be a calibrated physical constant, same
// category of assumption as focal length), and (2) the camera's PERPENDICULAR
// DISTANCE to the grid plane (d) -- a new search dimension, since orientation
// (Drow, Dcol) alone doesn't fix scale.
//
// Architecture (deliberately NOT a blind joint search over all parameters):
//   1. Run the existing searchOrthogonalVPs for a coarse (Drow, Dcol) --
//      cheap, already validated.
//   2. Using Drow (trust the presumably-stronger family) and the coarse
//      Dcol as a placeholder for spacing direction, sweep candidate distances
//      d and score each by how PERIODIC the row family's dewarped signal
//      looks (see periodicityScore) -- pick the best d.
//   3. Holding Drow and the now-known d fixed, sweep candidate roll angles
//      psi for Dcol, and score EACH by how periodic the (line-pool-filtered)
//      col-family dewarped signal looks at that psi -- pick the best psi.
// Step 3 is the one actually targeting the known bug: a wrong psi should
// show much weaker periodicity even when its raw truncated-cost looks
// deceptively similar to the right one.
//
// FINDINGS SO FAR (informal, small sample -- exploratory, not validated;
// does NOT look promising enough to pursue further right now):
//   - Found and fixed two real bugs along the way. (1) The original m=0
//     anchor was the grid plane's perpendicular foot from the camera, which
//     for any tilt>0 sits at a DIFFERENT, a-priori-unknown world offset than
//     where the visible lines actually are -- the sampled m-window ended up
//     centered on a region with no real data in it at all. Fixed by
//     anchoring at (0,0,dist) instead -- the point the camera is actually
//     centered on, which is always exactly known regardless of tilt. (2)
//     The axis-aligned gradient-quantization artifact (see the Hough-space
//     investigation earlier this session: theta piling up at exactly
//     0deg/90deg with essentially random rho) contaminates this approach
//     far more than the existing cost-based search, since dewarping relies
//     on precise theta alone -- fixed with a blunt exclusion band around
//     those two angles (a real cost: suppresses genuine near-axis-aligned
//     data too, not a fix of the underlying artifact).
//   - After both fixes: scale (d) recovery is genuinely good in 2 of 3
//     scenarios (1.7%-3.7% error) but still badly wrong in "moderate tilt"
//     (unexplained). The psi search shows only a modest improvement in the
//     target weak-family case (45.6deg -> 41.0deg error vs the ORIGINAL
//     pre-fix baseline -- nowhere near a real fix) and actively regresses
//     the two other scenarios.
//   - Root cause for the remaining failure looks structural, not a bug:
//     checked the dewarped m-values' fractional parts (distance to nearest
//     integer) at the EXACT true Drow/Dcol/d for moderate tilt, and they
//     scatter almost uniformly across [-0.5,+0.5] rather than clustering
//     around one consistent phase -- i.e. even ground-truth-perfect
//     orientation and distance doesn't recover clean integer indices from
//     raw peak theta alone. Likely explanation, and it connects directly to
//     the earlier "meta-function" result: realistic captures always sit far
//     from m* (the density peak) -- precisely what made the visible Hough
//     arc look reassuringly straight -- which means dtheta/dm is small
//     there, so its inverse (what dewarping actually divides by) is large,
//     amplifying Level 1's ordinary small sub-pixel theta noise into
//     potentially large errors in the recovered m. The same geometric
//     property that made the curve-shape claim check out visually is what
//     makes recovering PHASE from theta alone fragile.
//   - Bottom line: not pursuing further as-is. A fix would likely need to
//     use rho jointly with theta for dewarping (not theta alone), which is
//     a real redesign, not a tuning pass -- parking this here, same
//     treatment as dual-hough-vp.ts and em-vp-finder.ts.
//
// Usage: node scripts/experiments/periodicity-vp-search.ts

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus } from '../../src/debruijn.ts';
import { toGrayscale } from '../../src/decode.ts';
import { buildLineAccumulator, findLinePeaksTiered } from '../../src/lines.ts';
import type { LineCandidate } from '../../src/lines.ts';
import { searchOrthogonalVPs } from '../../src/orthogonalVp.ts';
import { captureHomography, buildCamera } from '../lib/synth-camera.ts';
import type { CameraPose } from '../lib/synth-camera.ts';

const order = 4;
const debruijn = generateTorus(order);
const { C } = debruijn;
const png = PNG.sync.read(readFileSync(`samples/order${order}.png`));
const cellPx = png.width / C; // the "known calibrated real-world cell size", in this synthetic test's own units
const RAW = 320;
const HOUGH_RHO_BIN_PX = 1.5;
const HOUGH_THETA_BINS = Math.round(360 / HOUGH_RHO_BIN_PX);
const RESCUE_THRESHOLD_FRACTION = 0.3;

type Vec3 = [number, number, number];
const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize3 = (a: Vec3): Vec3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const scale3 = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const add3 = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

function perpBasis(D: Vec3): [Vec3, Vec3] {
  const ref: Vec3 = Math.abs(D[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const e1 = normalize3(cross3(ref, D));
  const e2 = cross3(D, e1);
  return [e1, e2];
}
const dirAtRoll = (e1: Vec3, e2: Vec3, psi: number): Vec3 =>
  normalize3(add3(scale3(e1, Math.cos(psi)), scale3(e2, Math.sin(psi))));

function lineToNormal(line: LineCandidate, f: number): Vec3 {
  const a = Math.cos(line.theta), b = Math.sin(line.theta);
  return normalize3([a, b, -line.rho / f]);
}
function angResidual(n: Vec3, D: Vec3): number {
  return Math.asin(Math.min(1, Math.abs(dot3(n, D))));
}

// --- forward geometric model: predicts the (theta,rho) of the line at
// (possibly non-integer) index m in a family running along `Dalong`, whose
// consecutive members step by `cellSize` along `Dacross`, on a plane at
// perpendicular distance `d` along `Dnormal` -- all in the SAME
// camera-centered frame src/orthogonalVp.ts already works in (image plane
// at Z=f). Exact forward simulation (project two real 3D points), not the
// abstract affine-line algebra -- less error-prone to implement correctly. ---
function projectCam(P: Vec3, f: number): [number, number] | null {
  if (P[2] <= 1e-6) return null;
  return [(f * P[0]) / P[2], (f * P[1]) / P[2]];
}
function lineFromTwoPoints(p1: [number, number], p2: [number, number]): { theta: number; rho: number } {
  const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
  const nx = -dy, ny = dx;
  let theta = Math.atan2(ny, nx);
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  const a = Math.cos(theta), b = Math.sin(theta);
  return { theta, rho: a * p1[0] + b * p1[1] };
}
// `d` is the distance from camera to the point it's actually centered on,
// along the optical axis -- (0,0,d) in this camera-centered frame, exactly,
// by construction of how the pose aims (unlike the plane's perpendicular
// foot, which sits at a DIFFERENT, unknown-without-more-work offset once
// tilt>0, and was the actual bug in an earlier version of this: the sampled
// m-window ended up centered on a region with no real data in it at all).
// m=0 here is "whichever line passes through the image center," which is
// always a good phase-origin guess regardless of tilt.
function predictedLine(Dalong: Vec3, Dacross: Vec3, d: number, cellSize: number, f: number, m: number): { theta: number; rho: number } | null {
  const base = add3([0, 0, d], scale3(Dacross, m * cellSize));
  const T = Math.max(1, d * 0.3);
  const p1 = projectCam(add3(base, scale3(Dalong, -T)), f);
  const p2 = projectCam(add3(base, scale3(Dalong, T)), f);
  if (!p1 || !p2) return null;
  return lineFromTwoPoints(p1, p2);
}

// --- dewarp table: theta(m) sampled densely and inverted for lookup.
// Circular-seam-safe the same way the earlier Hough-space investigation
// needed to be (theta folded to [0,PI)). ---
interface DewarpTable { thetas: number[]; ms: number[]; }
function buildDewarpTable(Dalong: Vec3, Dacross: Vec3, d: number, cellSize: number, f: number, mMin: number, mMax: number, steps: number): DewarpTable | null {
  const raw: { theta: number; m: number }[] = [];
  for (let i = 0; i <= steps; i++) {
    const m = mMin + ((mMax - mMin) * i) / steps;
    const line = predictedLine(Dalong, Dacross, d, cellSize, f, m);
    if (line) raw.push({ theta: line.theta, m });
  }
  if (raw.length < 4) return null;
  const sorted = raw.slice().sort((a, b) => a.theta - b.theta);
  let biggestGap = 0, gapAfterIdx = -1;
  for (let i = 0; i < sorted.length; i++) {
    const next = i + 1 < sorted.length ? sorted[i + 1].theta : sorted[0].theta + Math.PI;
    const gap = next - sorted[i].theta;
    if (gap > biggestGap) { biggestGap = gap; gapAfterIdx = i; }
  }
  const unwrapped = sorted
    .map((p, i) => ({ theta: i <= gapAfterIdx ? p.theta + Math.PI : p.theta, m: p.m }))
    .sort((a, b) => a.theta - b.theta);
  return { thetas: unwrapped.map(p => p.theta), ms: unwrapped.map(p => p.m) };
}
function lookupM(table: DewarpTable, theta: number): number | null {
  const tMin = table.thetas[0], tMax = table.thetas[table.thetas.length - 1];
  let t = theta;
  if (t < tMin || t > tMax) t = theta + Math.PI;
  if (t < tMin || t > tMax) return null;
  for (let i = 0; i < table.thetas.length - 1; i++) {
    if (t >= table.thetas[i] && t <= table.thetas[i + 1]) {
      const denom = table.thetas[i + 1] - table.thetas[i] || 1;
      const frac = (t - table.thetas[i]) / denom;
      return table.ms[i] + frac * (table.ms[i + 1] - table.ms[i]);
    }
  }
  return null;
}

// --- bin dewarped line weight into a uniform m-histogram, then score its
// periodicity at period=1 (one grid cell) via a single-frequency (Goertzel-
// style) correlation, normalized to [0,1] by total weight so it's comparable
// across candidates with different amounts of assigned mass. ---
function buildHistogram(lines: LineCandidate[], ownDir: Vec3, otherDir: Vec3, table: DewarpTable, f: number, mMin: number, mMax: number, nBins: number): { hist: Float64Array; count: number } {
  const hist = new Float64Array(nBins);
  const binSize = (mMax - mMin) / nBins;
  let count = 0;
  for (const line of lines) {
    const n = lineToNormal(line, f);
    const rOwn = angResidual(n, ownDir), rOther = angResidual(n, otherDir);
    if (rOwn > rOther) continue;
    const m = lookupM(table, line.theta);
    if (m === null || m < mMin || m > mMax) continue;
    const bin = Math.min(nBins - 1, Math.max(0, Math.floor((m - mMin) / binSize)));
    hist[bin] += line.weight;
    count++;
  }
  return { hist, count };
}
// Raw phase-coherence ratio (|sum|/sum|.|) is trivially gameable: a
// candidate that classifies just 1-2 lines into "own family" scores ~1.0
// regardless of real periodicity (a single point is always "coherent" with
// itself). Multiply by sqrt(count) -- a weighted analog of the Rayleigh Z
// statistic from circular statistics, which properly discounts a high
// coherence ratio achieved from too little evidence to mean anything.
// Candidates below MIN_COUNT are disqualified outright.
const MIN_COUNT = 5;
function periodicityScore(hist: Float64Array, mBinSize: number, count: number): number {
  if (count < MIN_COUNT) return -1;
  let re = 0, im = 0, totalAbs = 0;
  const omega = 2 * Math.PI * mBinSize; // targets period = 1 (one grid cell) in m
  for (let i = 0; i < hist.length; i++) {
    re += hist[i] * Math.cos(omega * i);
    im -= hist[i] * Math.sin(omega * i);
    totalAbs += hist[i];
  }
  const R = totalAbs > 0 ? Math.hypot(re, im) / totalAbs : 0;
  return R * Math.sqrt(count);
}

const M_BIN = 0.1;
// The sampled m-range must scale with the candidate distance d -- a FIXED
// window (e.g. always +-40 cells) means small-d candidates get queried at
// physical offsets (m*cellSize) far larger than d itself, producing
// garbage/degenerate geometry (points behind the camera, wildly distorted
// tables) that has nothing to do with whether that d is really a bad fit.
// Same "visible FOV" formula used throughout this session's other
// experiments (dist/focal * RAW), just with d standing in for dist.
function mRangeForD(d: number, f: number, rawSize: number, cellSize: number): number {
  const fovSpan = (d / f) * rawSize * 1.2;
  return Math.max(3, fovSpan / cellSize);
}

function searchScale(lines: LineCandidate[], Drow: Vec3, DcolGuess: Vec3, f: number, cellSize: number, rawSize: number, dCandidates: number[]) {
  let best = { d: dCandidates[0], score: -1 };
  for (const d of dCandidates) {
    const mCap = mRangeForD(d, f, rawSize, cellSize);
    const nBins = Math.round((2 * mCap) / M_BIN);
    const table = buildDewarpTable(Drow, DcolGuess, d, cellSize, f, -mCap, mCap, 240);
    if (!table) continue;
    const { hist, count } = buildHistogram(lines, Drow, DcolGuess, table, f, -mCap, mCap, nBins);
    const score = periodicityScore(hist, M_BIN, count);
    if (score > best.score) best = { d, score };
  }
  return best;
}

function searchPsi(lines: LineCandidate[], Drow: Vec3, d: number, f: number, cellSize: number, rawSize: number, nPsiSteps: number) {
  const [e1, e2] = perpBasis(Drow);
  const mCap = mRangeForD(d, f, rawSize, cellSize);
  const nBins = Math.round((2 * mCap) / M_BIN);
  let best = { psi: 0, Dcol: e1, score: -1 };
  for (let k = 0; k < nPsiSteps; k++) {
    const psi = (k / nPsiSteps) * Math.PI;
    const Dcol = dirAtRoll(e1, e2, psi);
    const table = buildDewarpTable(Dcol, Drow, d, cellSize, f, -mCap, mCap, 240);
    if (!table) continue;
    const { hist, count } = buildHistogram(lines, Dcol, Drow, table, f, -mCap, mCap, nBins);
    const score = periodicityScore(hist, M_BIN, count);
    if (score > best.score) best = { psi, Dcol, score };
  }
  return best;
}

// --- ground truth (same method as scripts/experiments/orthogonal-vp-search.ts) ---
// d = distance from camera to the point it's centered on, along the optical
// axis -- exactly pose.dist (see predictedLine's comment), NOT the plane's
// perpendicular foot (pose.dist*cos(tilt)), which was the earlier bug.
function trueAxes(pose: CameraPose): { row: Vec3; col: Vec3; d: number } {
  const { right, up, forward } = buildCamera(pose);
  const row = normalize3([right[0], up[0], forward[0]]);
  const col = normalize3([right[1], up[1], forward[1]]);
  return { row, col, d: pose.dist };
}
function angleBetweenDeg(a: Vec3, b: Vec3): number {
  return (Math.acos(Math.min(1, Math.abs(dot3(a, b)))) * 180) / Math.PI;
}

function logSpace(min: number, max: number, n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(min * Math.pow(max / min, i / (n - 1)));
  return out;
}

function runScenario(name: string, pose: CameraPose) {
  console.log(`\n=== ${name} (tilt=${((pose.tilt * 180) / Math.PI).toFixed(0)}deg) ===`);
  const rgba = captureHomography(png, pose, RAW, RAW, 4);
  const gray = toGrayscale(rgba, RAW, RAW);
  const field = buildLineAccumulator(gray, RAW, RAW, HOUGH_THETA_BINS, HOUGH_RHO_BIN_PX);
  const { strong: peaks, weak: rescuePeaks } = findLinePeaksTiered(field, 0.15, 0.15 * RESCUE_THRESHOLD_FRACTION, 4, 3);
  // Exclude the axis-aligned gradient-quantization artifact (theta piling up
  // exactly at 0deg/90deg with essentially random rho -- see the Hough-space
  // investigation a few turns back): this approach dewarps by theta alone,
  // so it's far more sensitive to that contamination than the existing
  // truncated-residual cost, which at least also weighs rho consistency.
  // Real cost: if the true row or col direction genuinely sits near exactly
  // axis-aligned in-frame, this filter suppresses real data too -- a
  // heuristic patch, not a fix of the underlying artifact.
  const ARTIFACT_BAND_RAD = (2 * Math.PI) / 180;
  const lines = [...peaks, ...rescuePeaks].filter(l => {
    const nearZero = Math.min(l.theta, Math.PI - l.theta) < ARTIFACT_BAND_RAD;
    const nearNinety = Math.abs(l.theta - Math.PI / 2) < ARTIFACT_BAND_RAD;
    return !nearZero && !nearNinety;
  });
  const f = pose.focal;

  const { row: trueRow, col: trueCol, d: trueD } = trueAxes(pose);

  // baseline: existing coarse orthogonal search alone
  const coarse = searchOrthogonalVPs(lines, f);
  const coarseRowErr = angleBetweenDeg(coarse.Drow, trueRow);
  const coarseColErr = angleBetweenDeg(coarse.Dcol, trueCol);
  // match which of Drow/Dcol is "row" by proximity, same convention as before
  const rowIsDrow = angleBetweenDeg(coarse.Drow, trueRow) < angleBetweenDeg(coarse.Dcol, trueRow);
  const Drow0 = rowIsDrow ? coarse.Drow : coarse.Dcol;
  const Dcol0 = rowIsDrow ? coarse.Dcol : coarse.Drow;
  const baselineRowErr = angleBetweenDeg(Drow0, trueRow);
  const baselineColErr = angleBetweenDeg(Dcol0, trueCol);
  console.log(`  baseline (coarse orthogonal search only): row err=${baselineRowErr.toFixed(2)}deg col err=${baselineColErr.toFixed(2)}deg`);

  // stage 2: scale search using Drow0 (trusted) + Dcol0 as placeholder direction
  const dCandidates = logSpace(30, 2000, 80);
  const scaleResult = searchScale(lines, Drow0, Dcol0, f, cellPx, RAW, dCandidates);
  const dErrPct = (100 * Math.abs(scaleResult.d - trueD)) / trueD;
  console.log(`  scale search: d_hat=${scaleResult.d.toFixed(1)} (true=${trueD.toFixed(1)}, err=${dErrPct.toFixed(1)}%), periodicity score=${scaleResult.score.toFixed(3)}`);

  // stage 3: psi search using Drow0 + d_hat, scored by periodicity
  const psiResult = searchPsi(lines, Drow0, scaleResult.d, f, cellPx, RAW, 180);
  const periodicColErr = angleBetweenDeg(psiResult.Dcol, trueCol);
  console.log(`  periodicity-based psi search: col err=${periodicColErr.toFixed(2)}deg (score=${psiResult.score.toFixed(3)}), vs baseline col err=${baselineColErr.toFixed(2)}deg`);
  console.log(`  ${periodicColErr < baselineColErr ? 'IMPROVED' : periodicColErr > baselineColErr + 0.5 ? 'REGRESSED' : 'no meaningful change'}`);
}

runScenario('moderate tilt', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.4, azimuth: 0.5, roll: 0.3 });
runScenario('steep tilt (known weak-family case)', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.9, azimuth: 1.0, roll: -0.2 });
runScenario('near-grazing (one axis toward infinity)', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 1.3, azimuth: 0, roll: 0 });
