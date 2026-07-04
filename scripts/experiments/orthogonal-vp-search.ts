// EXPERIMENT, not a regression test: a structurally different alternative
// to src/vp.ts's splitIntoTwoFamilies (discrete peaks + exhaustive-pair
// RANSAC), dual-hough-vp.ts (a second flat accumulator -- known to dilute
// distant VPs), and em-vp-finder.ts (soft two-cluster EM -- needs an
// angle-sweep seed and has an unresolved distant-VP failure). Unlike all
// three, this assumes a KNOWN (or calibrated) camera focal length -- a real
// new dependency nothing else in this pipeline needs -- in exchange for
// using a fact specific to THIS problem that none of them exploit: the
// pattern is a RIGHT-ANGLE GRID, so its two vanishing directions are not two
// independent unknowns to search for separately. They are two ORTHOGONAL
// unit vectors in 3D. As a direct bonus, the answer this produces IS the
// camera's full 3D orientation relative to the grid (row axis, col axis,
// and their cross product as the grid normal) -- not two pixel coordinates
// that would still need a separate decomposition step to become a pose.
//
// --- Math ---
// Work in CENTERED pixel coordinates (matching LineCandidate.rho's own
// convention, see src/lines.ts) with the image plane at Z=f in a
// camera-centered frame (standard pinhole model): a centered pixel (x,y)
// corresponds to the 3D ray direction (x,y,f).
//
// A Hough line (theta,rho) is, in centered coords, the equation
// a*x + b*y = rho (a=cos theta, b=sin theta). Back-projecting it through the
// camera center -- lifting to the 3D points (x,y,f) that satisfy this, and
// asking what plane through the origin contains all of them -- gives plane
// normal N = (a, b, -rho/f): check N.(x,y,f) = a*x + b*y - rho = 0 exactly
// when the 2D line equation holds. Normalize to a unit vector n.
//
// A 3D direction D is "consistent with" that line (i.e. the line could be a
// member of the pencil converging toward D, whether D is a finite vanishing
// point (D_z != 0) or a genuine point at infinity in the image (D_z == 0,
// a direction parallel to the image plane) -- no special case needed, same
// theme as src/vp.ts's homogeneous VanishingPoint) exactly when n.D = 0:
// D lies IN the line's back-projected plane.
//
// Residual: n and D are both unit vectors, so n.D = cos(angle between
// them). The ideal is a 90-degree angle (D lying exactly in the plane), so
// asin(|n.D|) is the angle by which that's violated -- a single, bounded,
// uniformly-scaled angular residual with NO finite/near-infinite branching,
// unlike src/vp.ts's lineResidualPx (suspected, in em-vp-finder.ts's
// unresolved distant-VP failure, of biasing EM's responsibility comparison
// exactly because its two branches aren't on a comparable scale).
//
// Search space: D_row is 2 DOF (any unit direction; sign doesn't matter,
// since n.D and n.(-D) give the same residual, so the real search space is
// RP^2, a hemisphere). Given D_row, D_col is confined to the 1-parameter
// family of unit vectors perpendicular to it -- a roll angle psi around
// D_row. So the entire two-VP problem is exactly 3 real parameters, found
// by a coarse global search over that space followed by local coordinate-
// descent refinement -- never via peak extraction, RANSAC pairing, a second
// accumulator, or per-line soft cluster responsibilities.
//
// A weak family gets a real, structural advantage here that none of the
// other three approaches have: it is never searched for independently. Once
// D_row's direction is well-determined by whichever family has the
// stronger/cleaner evidence, D_col is already reduced to a single scalar
// (psi) -- the weak family doesn't need enough of its own peaks to mount an
// independent search, only enough weighted evidence to pick one angle out
// of a 1D sweep against a direction the strong family has already pinned
// down.
//
// Usage: node scripts/experiments/orthogonal-vp-search.ts

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus } from '../../src/debruijn.ts';
import { toGrayscale } from '../../src/decode.ts';
import { buildLineAccumulator, findLinePeaksTiered } from '../../src/lines.ts';
import type { LineCandidate, HoughField } from '../../src/lines.ts';
import { splitIntoTwoFamilies, vpIsFinite } from '../../src/vp.ts';
import type { VanishingPoint } from '../../src/vp.ts';
import { searchOrthogonalVPs as searchOrthogonalVPsImpl } from '../../src/orthogonalVp.ts';
import type { Vec3, OrthogonalVpResult } from '../../src/orthogonalVp.ts';
import { captureHomography, buildCamera } from '../lib/synth-camera.ts';
import type { CameraPose } from '../lib/synth-camera.ts';

const order = 4;
const debruijn = generateTorus(order);
const { C } = debruijn;
const png = PNG.sync.read(readFileSync(`samples/order${order}.png`));
const RAW = 300;
const HOUGH_RHO_BIN_PX = 1.5;
const HOUGH_THETA_BINS = Math.round(360 / HOUGH_RHO_BIN_PX);
const RESCUE_THRESHOLD_FRACTION = 0.3;

const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize3 = (a: Vec3): Vec3 => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// searchOrthogonalVPs itself now lives in src/orthogonalVp.ts (so the app's
// live pipeline and this validation harness are guaranteed to run the exact
// same code, not a copy that can drift) -- see that file for the math.
const searchOrthogonalVPs = (lines: LineCandidate[], f: number, deltaPx?: number, nRowDirs?: number, nPsiSteps?: number, refineRounds?: number): OrthogonalVpResult =>
  searchOrthogonalVPsImpl(lines, f, deltaPx, nRowDirs, nPsiSteps, refineRounds);

// --- ground truth: the row/col axes' true 3D directions in camera space,
// derived directly from the pose's own rotation basis (right,up,forward),
// NOT by projecting world points to pixels and intersecting -- that method
// (used in the previous two experiments) degenerates exactly at infinity,
// which is the one case this approach is specifically supposed to handle
// gracefully. A world DIRECTION (unlike a world POINT) only needs the
// rotation part of the camera basis, no translation -- so this is exact and
// well-defined even when the corresponding image-space VP is at infinity,
// and (correctly) independent of focal length: orientation is a physical
// fact about the pose, f only affects how it maps to pixels. ---
function trueAxes(pose: CameraPose): { row: Vec3; col: Vec3 } {
  const { right, up, forward } = buildCamera(pose);
  // world (1,0,0) ["row" direction, see runScenario's own line convention]
  // expressed in the camera's (right,up,forward) basis:
  const row = normalize3([right[0], up[0], forward[0]]);
  const col = normalize3([right[1], up[1], forward[1]]);
  return { row, col };
}

function angleBetweenDeg(a: Vec3, b: Vec3): number {
  return (Math.acos(Math.min(1, Math.abs(dot3(a, b)))) * 180) / Math.PI;
}

// Converts the CURRENT pipeline's VanishingPoint (absolute pixel coords,
// see vp.ts) into the same 3D-direction representation used here, so it can
// be compared against ground truth on equal footing.
function vpTo3D(vp: VanishingPoint, f: number, cx: number, cy: number): Vec3 {
  if (vpIsFinite(vp)) return normalize3([vp.x / vp.w - cx, vp.y / vp.w - cy, f]);
  return normalize3([vp.x, vp.y, 0]);
}

function extractActiveCells(field: HoughField): LineCandidate[] {
  const { thetaBins, rhoBins, rhoMin, rhoBinSize, acc } = field;
  const cells: LineCandidate[] = [];
  for (let tb = 0; tb < thetaBins; tb++) {
    const theta = (tb / thetaBins) * Math.PI;
    for (let rb = 0; rb < rhoBins; rb++) {
      const w = acc[tb * rhoBins + rb];
      if (w <= 0) continue;
      cells.push({ theta, rho: rhoMin + (rb + 0.5) * rhoBinSize, weight: w });
    }
  }
  return cells;
}

function runScenario(name: string, pose: CameraPose) {
  console.log(`\n=== ${name} (tilt=${((pose.tilt * 180) / Math.PI).toFixed(0)}deg) ===`);
  const rgba = captureHomography(png, pose, RAW, RAW, 4);
  const gray = toGrayscale(rgba, RAW, RAW);
  const field = buildLineAccumulator(gray, RAW, RAW, HOUGH_THETA_BINS, HOUGH_RHO_BIN_PX);
  const cx = RAW / 2, cy = RAW / 2;
  const f = pose.focal; // assumed = true for this scenario -- isolates search-algorithm correctness from focal-length-guess error (see the separate sensitivity sweep below)

  const { row: trueRow, col: trueCol } = trueAxes(pose);
  const trueNormal = cross3(trueRow, trueCol);

  const t0 = performance.now();
  const { strong: peaks, weak: rescuePeaks } = findLinePeaksTiered(field, 0.15, 0.15 * RESCUE_THRESHOLD_FRACTION, 4, 3);
  let currentErrRow = NaN, currentErrCol = NaN;
  try {
    const split = splitIntoTwoFamilies(peaks, RAW, RAW, 6, 60, rescuePeaks);
    const dA = vpTo3D(split.familyA.vp, f, cx, cy), dB = vpTo3D(split.familyB.vp, f, cx, cy);
    // match recovered families to row/col by proximity to ground truth, same as previous experiments
    if (angleBetweenDeg(dA, trueRow) < angleBetweenDeg(dB, trueRow)) {
      currentErrRow = angleBetweenDeg(dA, trueRow); currentErrCol = angleBetweenDeg(dB, trueCol);
    } else {
      currentErrRow = angleBetweenDeg(dB, trueRow); currentErrCol = angleBetweenDeg(dA, trueCol);
    }
  } catch { /* leave NaN */ }
  const t1 = performance.now();

  const filteredLines = [...peaks, ...rescuePeaks];
  const filteredResult = searchOrthogonalVPs(filteredLines, f);
  const t2 = performance.now();

  const allCells = extractActiveCells(field);
  const rawResult = searchOrthogonalVPs(allCells, f, 6, 60, 24, 24); // fewer coarse dirs -- O(n) cost per hypothesis is much larger here
  const t3 = performance.now();

  function report(label: string, res: OrthogonalVpResult, ms: number) {
    const errRow = Math.min(angleBetweenDeg(res.Drow, trueRow), angleBetweenDeg(res.Dcol, trueRow));
    const errCol = Math.min(angleBetweenDeg(res.Drow, trueCol), angleBetweenDeg(res.Dcol, trueCol));
    const errNormal = angleBetweenDeg(res.Dnormal, trueNormal);
    console.log(`  ${label} (${ms.toFixed(0)}ms): row err=${errRow.toFixed(2)}deg col err=${errCol.toFixed(2)}deg normal err=${errNormal.toFixed(2)}deg cost=${res.cost.toFixed(4)}`);
  }

  console.log(`  -- current pipeline (discrete peaks + RANSAC + rescue), ${(t1 - t0).toFixed(0)}ms, ${peaks.length} strong + ${rescuePeaks.length} rescue --`);
  console.log(`     row err=${currentErrRow.toFixed(2)}deg col err=${currentErrCol.toFixed(2)}deg`);
  report(`orthogonal search (${filteredLines.length} strong+rescue peaks)`, filteredResult, t2 - t1);
  report(`orthogonal search (${allCells.length} raw active cells, zero filtering)`, rawResult, t3 - t2);
}

function focalSensitivitySweep(pose: CameraPose) {
  console.log(`\n=== focal-length sensitivity sweep (tilt=${((pose.tilt * 180) / Math.PI).toFixed(0)}deg) ===`);
  const rgba = captureHomography(png, pose, RAW, RAW, 4);
  const gray = toGrayscale(rgba, RAW, RAW);
  const field = buildLineAccumulator(gray, RAW, RAW, HOUGH_THETA_BINS, HOUGH_RHO_BIN_PX);
  const { strong: peaks, weak: rescuePeaks } = findLinePeaksTiered(field, 0.15, 0.15 * RESCUE_THRESHOLD_FRACTION, 4, 3);
  const lines = [...peaks, ...rescuePeaks];

  const { row: trueRow, col: trueCol } = trueAxes(pose);
  const trueFocal = pose.focal;

  for (const pct of [-20, -15, -10, -5, 0, 5, 10, 15, 20]) {
    const assumedF = trueFocal * (1 + pct / 100);
    const res = searchOrthogonalVPs(lines, assumedF);
    const errRow = Math.min(angleBetweenDeg(res.Drow, trueRow), angleBetweenDeg(res.Dcol, trueRow));
    const errCol = Math.min(angleBetweenDeg(res.Drow, trueCol), angleBetweenDeg(res.Dcol, trueCol));
    console.log(`  assumed f ${pct >= 0 ? '+' : ''}${pct}% (${assumedF.toFixed(0)}px vs true ${trueFocal}px): row err=${errRow.toFixed(2)}deg col err=${errCol.toFixed(2)}deg`);
  }
}

runScenario('moderate tilt', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.4, azimuth: 0.5, roll: 0.3 });
runScenario('steep tilt (known weak-family case)', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.9, azimuth: 1.0, roll: -0.2 });
runScenario('near-grazing (one axis toward infinity)', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 1.3, azimuth: 0, roll: 0 });

focalSensitivitySweep({ targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.4, azimuth: 0.5, roll: 0.3 });
