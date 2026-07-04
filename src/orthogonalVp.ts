// Level 2 (alternative): replaces src/vp.ts's splitIntoTwoFamilies (discrete
// peaks + exhaustive-pair RANSAC) with a constrained 3-parameter direct
// search that exploits a fact specific to this problem: the pattern is a
// RIGHT-ANGLE GRID, so its two vanishing directions are not independent --
// they are two ORTHOGONAL unit vectors in 3D. Ported from (and kept
// numerically identical to) scripts/experiments/orthogonal-vp-search.ts,
// where the math, validation methodology, and known limitations are
// documented in full. Summary of what's known:
//   - Matches the old pipeline on ordinary tilts, and clearly wins the
//     near-infinite-VP case (the old pixel-based approach can hit a
//     numerical NaN there; this approach has no such branch at all).
//   - Requires an assumed/calibrated focal length -- a new dependency
//     nothing else in this pipeline needs. Validated to degrade gracefully
//     (a few degrees of orientation error) even at +-20% focal-length
//     error, so a researched nominal spec is an acceptable starting point.
//   - Known weakness: a genuinely sparse weak family (e.g. steep tilt) can
//     still find the wrong roll angle for the second axis, because the
//     truncated-squared-error objective doesn't distinguish "many lines
//     moderately wrong" from "few lines very wrong" the way the old
//     pipeline's inlier-COUNTING RANSAC does. Not yet fixed.
//
// This module holds only the pure math (no camera/rendering/test-harness
// code, unlike the experiment script) so it can run in the browser.

import type { LineCandidate } from './lines.ts';
import type { VanishingPoint } from './vp.ts';

export type Vec3 = [number, number, number];
const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize3 = (a: Vec3): Vec3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const scale3 = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const add3 = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

// Plane normal of a Hough line's back-projection through the camera center,
// given an ASSUMED focal length f (same centered-pixel units as line.rho) --
// see the experiment script's header for the full derivation. line.rho is
// already center-relative (src/lines.ts's own convention), so no cx/cy
// offset is needed here, unlike src/vp.ts's toAbsoluteLine.
function lineToNormal(line: LineCandidate, f: number): Vec3 {
  const a = Math.cos(line.theta), b = Math.sin(line.theta);
  return normalize3([a, b, -line.rho / f]);
}

function angularResidual(n: Vec3, D: Vec3): number {
  return Math.asin(Math.min(1, Math.abs(dot3(n, D))));
}

// Fibonacci-sphere point set folded into the z>=0 hemisphere (antipodal
// fold: negate any point with z<0) -- valid because D and -D give the same
// residual to every line, so the real search space for the row direction is
// RP^2 (a hemisphere), not the full sphere.
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

interface Cells { nx: Float64Array; ny: Float64Array; nz: Float64Array; weight: Float64Array; }

function toCells(lines: LineCandidate[], f: number): Cells {
  const n = lines.length;
  const nx = new Float64Array(n), ny = new Float64Array(n), nz = new Float64Array(n), weight = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [x, y, z] = lineToNormal(lines[i], f);
    nx[i] = x; ny[i] = y; nz[i] = z; weight[i] = lines[i].weight;
  }
  return { nx, ny, nz, weight };
}

function totalCost(cells: Cells, Drow: Vec3, Dcol: Vec3, deltaRad: number): number {
  const { nx, ny, nz, weight } = cells;
  const delta2 = deltaRad * deltaRad;
  const [ax, ay, az] = Drow, [bx, by, bz] = Dcol;
  let cost = 0;
  for (let i = 0; i < weight.length; i++) {
    const dotA = Math.abs(nx[i] * ax + ny[i] * ay + nz[i] * az);
    const dotB = Math.abs(nx[i] * bx + ny[i] * by + nz[i] * bz);
    const rA = Math.asin(Math.min(1, dotA));
    const rB = Math.asin(Math.min(1, dotB));
    const r2 = Math.min(rA * rA, rB * rB, delta2);
    cost += weight[i] * r2;
  }
  return cost;
}

export interface OrthogonalVpResult { Drow: Vec3; Dcol: Vec3; Dnormal: Vec3; cost: number; }

// Coarse global search over the compact 3-parameter space (2 DOF for the
// row direction via Fibonacci-hemisphere sampling, 1 DOF roll angle for the
// orthogonal column direction), then local coordinate-descent refinement
// with a shrinking step size.
export function searchOrthogonalVPs(
  lines: LineCandidate[], focalPx: number,
  deltaPx = 6, nRowDirs = 150, nPsiSteps = 36, refineRounds = 24,
): OrthogonalVpResult {
  const cells = toCells(lines, focalPx);
  const deltaRad = deltaPx / focalPx;

  const rowDirs = fibonacciHemisphere(nRowDirs);
  let bestCost = Infinity, bestDrow: Vec3 = rowDirs[0], bestPsi = 0;
  for (const Drow of rowDirs) {
    const [e1, e2] = perpBasis(Drow);
    for (let k = 0; k < nPsiSteps; k++) {
      const psi = (k / nPsiSteps) * Math.PI;
      const Dcol = dirAtRoll(e1, e2, psi);
      const cost = totalCost(cells, Drow, Dcol, deltaRad);
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
      const c = totalCost(cells, cand, candDcol, deltaRad);
      if (c < cost) { cost = c; Drow = cand; e1 = ce1; e2 = ce2; Dcol = candDcol; improved = true; }
    }
    for (const dpsi of [step, -step]) {
      const candDcol = dirAtRoll(e1, e2, psi + dpsi);
      const c = totalCost(cells, Drow, candDcol, deltaRad);
      if (c < cost) { cost = c; psi += dpsi; Dcol = candDcol; improved = true; }
    }
    step *= improved ? 0.85 : 0.5;
  }

  return { Drow, Dcol, Dnormal: cross3(Drow, Dcol), cost };
}

// Assigns each line to whichever of the two converged directions it's more
// consistent with -- no threshold needed for correctness (indexFamilyLines
// downstream already rejects lines that don't fit its own gap-tolerant
// Mobius model), but a generous rejection gate keeps obviously-unrelated
// lines from diluting either family's index-recovery input.
export function assignLinesToFamilies(
  lines: LineCandidate[], Drow: Vec3, Dcol: Vec3, focalPx: number, rejectPx = 12,
): { familyA: LineCandidate[]; familyB: LineCandidate[]; unassigned: LineCandidate[] } {
  const rejectRad = rejectPx / focalPx;
  const familyA: LineCandidate[] = [], familyB: LineCandidate[] = [], unassigned: LineCandidate[] = [];
  for (const line of lines) {
    const n = lineToNormal(line, focalPx);
    const rA = angularResidual(n, Drow), rB = angularResidual(n, Dcol);
    if (rA >= rejectRad && rB >= rejectRad) { unassigned.push(line); continue; }
    (rA <= rB ? familyA : familyB).push(line);
  }
  return { familyA, familyB, unassigned };
}

// Converts a converged 3D direction back into the existing homogeneous
// VanishingPoint type (src/vp.ts) -- ABSOLUTE pixel coordinates, matching
// that module's own convention -- so every downstream stage (indexFamilyLines,
// buildLatticeCorrespondences, the debug overlays) needs zero changes to
// consume this approach's output instead of splitIntoTwoFamilies's.
export function directionToVanishingPoint(D: Vec3, focalPx: number, cx: number, cy: number): VanishingPoint {
  if (Math.abs(D[2]) < 1e-6) return { x: D[0], y: D[1], w: 0 };
  return { x: cx + (focalPx * D[0]) / D[2], y: cy + (focalPx * D[1]) / D[2], w: 1 };
}

// Nominal focal length (in RAW analysis-buffer pixel units) from an assumed
// diagonal field of view -- deliberately diagonal rather than horizontal or
// vertical, since that's the one number actually published with confidence
// (iPhone 13 ultra-wide / 0.5x: 13mm-equivalent, ~120deg diagonal FOV) and
// it sidesteps needing to know the exact aspect ratio getUserMedia's video
// stream actually delivers. NOT yet empirically calibrated against the real
// device -- validated only for how much a WRONG assumption costs (a few
// degrees of orientation error even at +-20%), not that this specific
// number is correct. See scripts/experiments/orthogonal-vp-search.ts's
// focal-length sensitivity sweep.
export function estimateFocalPxFromDiagonalFov(rawW: number, rawH: number, diagonalFovDeg = 120): number {
  const diagPx = Math.hypot(rawW, rawH);
  return diagPx / (2 * Math.tan((diagonalFovDeg * Math.PI) / 360));
}
