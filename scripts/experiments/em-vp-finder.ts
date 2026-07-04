// EXPERIMENT, not a regression test: replaces discrete-peak extraction + NMS
// + RANSAC pairing (src/vp.ts's splitIntoTwoFamilies) with a continuous,
// SOFT two-cluster fit (Expectation-Maximization) directly on the raw Hough
// accumulator's weighted cells. No per-peak vote threshold, no NMS, no
// pairwise hypothesis search anywhere in the fitting loop -- every
// accumulator cell above Level 1's existing gradient-magnitude floor
// (buildLineAccumulator's minMag, shared with the current pipeline, not
// something this experiment adds on top) participates in every iteration,
// weighted continuously rather than included/excluded by a threshold.
//
// Motivated directly by the previous prototype (dual-hough-vp.ts): that one
// swapped discrete peaks for a discretized (vx,vy) accumulator, which
// genuinely won on the weak-family case it targeted, but hit a real,
// understood wall -- a flat Cartesian grid can't represent a VP at/near
// infinity gracefully, diluting a distant VP's vote mass across many bins.
// EM sidesteps this by never discretizing VP-space at all: each M-step
// reuses src/vp.ts's EXISTING estimateVanishingPoint (a continuous,
// homogeneous-coordinate eigenvector solve, no code changes needed there),
// which already treats infinity as an ordinary, non-special-cased result
// (w~0) -- there is no grid for a distant VP to be diluted across.
//
// Some starting seed is unavoidable in principle: splitting one blob of
// mixed evidence into two unknown clusters needs SOME symmetry-breaking
// nudge, since a perfectly neutral/symmetric start has no reason to ever
// split. But the seed here doesn't look anything like discrete-peak-and-
// RANSAC -- it's a small, FIXED set of generic starting angle pairs (not
// derived from this image's own content at all, just a coarse sweep of
// plausible grid orientations), each one started AT INFINITY in its
// direction (a safe, assumption-free homogeneous default -- no guess about
// scale or distance), refined the rest of the way by EM using the full
// continuous field. Restarts are picked by post-hoc fit quality, not by any
// threshold in the fitting process itself.
//
// FINDINGS SO FAR (informal, small sample -- exploratory, not validated):
//   - The EM machinery itself is correct: sanity-checked against a clean
//     synthetic two-pencil scenario (no real-image noise at all, the same
//     kind test-vp-split.ts uses) and it converges EXACTLY to both true VPs
//     from a deliberately wrong starting guess. Any failure below is a
//     real-content difficulty, not a bug in the fitting math.
//   - EM given literally EVERY raw accumulator cell (~20-25K of them, zero
//     filtering) fails badly in BOTH test scenarios, all 6 restarts
//     collapsing into just 2-3 repeating, badly-wrong basins nowhere near
//     either true VP. Diagnosis: with that many active cells, the vast
//     majority is real-image texture/noise unrelated to grid geometry, not
//     genuine line evidence -- unlike the clean synthetic sanity check
//     where EVERY cell was real signal. A weighted least-squares fit over a
//     small-signal/large-noise mixture gets dominated by the noise's own
//     incidental structure. True "zero threshold, work off the raw
//     accumulator directly" does NOT hold up on real content.
//   - EM given the SAME strong+rescue peak pool the current pipeline
//     already computes (a few dozen to ~100 lines, not thousands) -- still
//     no hard RANSAC pairing or family assignment, every line still
//     contributes continuously to BOTH clusters every iteration -- performs
//     MUCH better: in the steep-tilt weak-family scenario, it matched or
//     beat the current pipeline outright (row VP 16.5px vs 43.9px, col VP
//     19.7px vs 20.3px). This is a real, direct win in the exact case that
//     motivated this whole investigation.
//   - BUT it still fails on the moderate-tilt scenario's distant col VP
//     (created by a much larger dist/vp separation than the steep-tilt
//     case), even with the same filtered input that fixed the other
//     scenario -- so this is a DIFFERENT problem from the noise-volume one
//     above, not yet root-caused. Suspect: lineResidualPx uses two
//     different residual regimes (pixel distance for finite VPs, an
//     angle-times-diagonal proxy for near-infinite ones), and EM's
//     Gaussian-kernel responsibility comparison implicitly assumes both
//     clusters' residuals live on a comparable scale each iteration --
//     if one cluster is still near-infinite while the other has already
//     become finite, the two residual regimes may not be numerically
//     comparable, biasing responsibility unfairly. Not confirmed, just the
//     leading hypothesis; would need its own isolated test to verify.
//   - Bottom line: replacing the FAMILY-ASSIGNMENT step (RANSAC) with soft,
//     continuous EM while keeping SOME lightweight upstream filtering
//     (findLinePeaksTiered's existing strong+rescue pool, not a new
//     threshold) is a real, working improvement for the weak-family case.
//     Going further to eliminate peak-extraction entirely and feed EM the
//     raw accumulator directly does not currently work on real content.
//
// Usage: node scripts/experiments/em-vp-finder.ts

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus } from '../../src/debruijn.ts';
import { toGrayscale } from '../../src/decode.ts';
import { buildLineAccumulator, findLinePeaksTiered } from '../../src/lines.ts';
import type { LineCandidate, HoughField } from '../../src/lines.ts';
import { crossLines, splitIntoTwoFamilies, estimateVanishingPoint, lineResidualPx, vpToPoint, vpIsFinite } from '../../src/vp.ts';
import type { VanishingPoint } from '../../src/vp.ts';
import { captureHomography, projectToImage } from '../lib/synth-camera.ts';
import type { CameraPose } from '../lib/synth-camera.ts';

const order = 4;
const debruijn = generateTorus(order);
const { C } = debruijn;
const png = PNG.sync.read(readFileSync(`samples/order${order}.png`));
const cellPx = png.width / C;
const RAW = 300;
const HOUGH_RHO_BIN_PX = 1.5;
const HOUGH_THETA_BINS = Math.round(360 / HOUGH_RHO_BIN_PX);
const RESCUE_THRESHOLD_FRACTION = 0.3;

// --- ground truth: intersect two exactly-projected parallel world lines,
// independent of anything under test here (same approach as dual-hough-vp.ts). ---
function trueLineAbs(pose: CameraPose, worldA: [number, number], worldB: [number, number]): [number, number, number] | null {
  const pa = projectToImage(pose, RAW, RAW, worldA[0], worldA[1]);
  const pb = projectToImage(pose, RAW, RAW, worldB[0], worldB[1]);
  if (!pa || !pb) return null;
  const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
  const nx = -dy, ny = dx;
  let theta = Math.atan2(ny, nx);
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  const a = Math.cos(theta), b = Math.sin(theta);
  const c = -(a * pa[0] + b * pa[1]);
  return [a, b, c];
}
function trueVp(pose: CameraPose, direction: 'row' | 'col'): { x: number; y: number } | null {
  const fovSpan = (pose.dist / pose.focal) * RAW;
  const l1 = direction === 'row' ? trueLineAbs(pose, [-fovSpan, 0], [fovSpan, 0]) : trueLineAbs(pose, [0, -fovSpan], [0, fovSpan]);
  const l2 = direction === 'row' ? trueLineAbs(pose, [-fovSpan, 8 * cellPx], [fovSpan, 8 * cellPx]) : trueLineAbs(pose, [8 * cellPx, -fovSpan], [8 * cellPx, fovSpan]);
  if (!l1 || !l2) return null;
  const p = crossLines(l1, l2);
  if (Math.abs(p.w) < 1e-9) return null;
  return { x: p.x / p.w, y: p.y / p.w };
}

// --- Every active accumulator cell, weighted, with NO peak/NMS/threshold
// step beyond the accumulator's own existing gradient-magnitude floor. ---
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

// --- EM: soft two-cluster fit. No thresholds anywhere in this loop --
// every cell contributes to both clusters' fits every iteration, just with
// a continuously varying weight. ---
function emFit(cells: LineCandidate[], w: number, h: number, vpA0: VanishingPoint, vpB0: VanishingPoint, iterations = 20): { vpA: VanishingPoint; vpB: VanishingPoint } {
  let vpA = vpA0, vpB = vpB0;
  const diag = Math.hypot(w, h);
  // Deterministic annealing: start WIDE (near-uniform 50/50 responsibility
  // for most cells) so the fit isn't locked into a bad early guess, then
  // sharpen as the two VP estimates (hopefully) separate.
  let sigma = diag * 0.15;
  const sigmaFloor = diag * 0.01;
  for (let iter = 0; iter < iterations; iter++) {
    const respA = new Float64Array(cells.length);
    for (let i = 0; i < cells.length; i++) {
      const residA = lineResidualPx(cells[i], vpA, w, h);
      const residB = lineResidualPx(cells[i], vpB, w, h);
      const la = Math.exp(-(residA * residA) / (2 * sigma * sigma));
      const lb = Math.exp(-(residB * residB) / (2 * sigma * sigma));
      const total = la + lb;
      respA[i] = total > 0 ? la / total : 0.5;
    }
    const weightedA = cells.map((c, i) => ({ theta: c.theta, rho: c.rho, weight: c.weight * respA[i] }));
    const weightedB = cells.map((c, i) => ({ theta: c.theta, rho: c.rho, weight: c.weight * (1 - respA[i]) }));
    vpA = estimateVanishingPoint(weightedA, w, h);
    vpB = estimateVanishingPoint(weightedB, w, h);
    sigma = Math.max(sigmaFloor, sigma * 0.75);
  }
  return { vpA, vpB };
}

// Total weighted support for a converged (vpA,vpB) pair -- used ONLY to
// rank the handful of restarts against each other after EM has already
// converged, not inside the EM loop itself.
function totalSupport(cells: LineCandidate[], vpA: VanishingPoint, vpB: VanishingPoint, w: number, h: number, inlierPx: number): number {
  let support = 0;
  for (const c of cells) {
    const dA = lineResidualPx(c, vpA, w, h), dB = lineResidualPx(c, vpB, w, h);
    if (Math.min(dA, dB) < inlierPx) support += c.weight;
  }
  return support;
}

function findVpsViaEM(cells: LineCandidate[], w: number, h: number, trueRowVp: { x: number; y: number }, trueColVp: { x: number; y: number }, verbose: boolean): { vpA: VanishingPoint; vpB: VanishingPoint } {
  // A coarse, content-independent sweep of plausible grid orientations --
  // not derived from this image's own peaks/content at all.
  const candidateAnglesDeg = [0, 30, 60, 90, 120, 150];
  let best: { vpA: VanishingPoint; vpB: VanishingPoint; score: number } | null = null;
  for (const deg of candidateAnglesDeg) {
    const thetaA = (deg * Math.PI) / 180;
    const thetaB = (thetaA + Math.PI / 2) % Math.PI;
    // VP "at infinity" in the TANGENT direction of a family whose lines have
    // normal angle theta (tangent = normal rotated 90deg: (-sin,cos)) -- a
    // safe, scale-free starting guess; EM is free to pull it to a genuine
    // finite point if the data supports one.
    const vpA0: VanishingPoint = { x: -Math.sin(thetaA), y: Math.cos(thetaA), w: 0 };
    const vpB0: VanishingPoint = { x: -Math.sin(thetaB), y: Math.cos(thetaB), w: 0 };
    const { vpA, vpB } = emFit(cells, w, h, vpA0, vpB0);
    const score = totalSupport(cells, vpA, vpB, w, h, 6);
    if (verbose) {
      const pA = vpIsFinite(vpA) ? vpToPoint(vpA) : null, pB = vpIsFinite(vpB) ? vpToPoint(vpB) : null;
      const errA = pA ? Math.min(Math.hypot(pA.x - trueRowVp.x, pA.y - trueRowVp.y), Math.hypot(pA.x - trueColVp.x, pA.y - trueColVp.y)) : NaN;
      const errB = pB ? Math.min(Math.hypot(pB.x - trueRowVp.x, pB.y - trueRowVp.y), Math.hypot(pB.x - trueColVp.x, pB.y - trueColVp.y)) : NaN;
      console.log(
        `    restart deg=${deg}: score=${score.toFixed(0)} vpA=${pA ? `(${pA.x.toFixed(0)},${pA.y.toFixed(0)})` : 'inf'} nearestTrueErr=${errA.toFixed(0)}px, ` +
        `vpB=${pB ? `(${pB.x.toFixed(0)},${pB.y.toFixed(0)})` : 'inf'} nearestTrueErr=${errB.toFixed(0)}px`
      );
    }
    if (!best || score > best.score) best = { vpA, vpB, score };
  }
  return { vpA: best!.vpA, vpB: best!.vpB };
}

function runScenario(name: string, pose: CameraPose) {
  console.log(`\n=== ${name} (tilt=${((pose.tilt * 180) / Math.PI).toFixed(0)}deg) ===`);
  const rgba = captureHomography(png, pose, RAW, RAW, 4);
  const gray = toGrayscale(rgba, RAW, RAW);
  const field = buildLineAccumulator(gray, RAW, RAW, HOUGH_THETA_BINS, HOUGH_RHO_BIN_PX);
  const w = RAW, h = RAW;

  const trueRowVp = trueVp(pose, 'row');
  const trueColVp = trueVp(pose, 'col');
  if (!trueRowVp || !trueColVp) { console.log('  (skipped -- a true VP is at/near infinity, out of scope for this prototype)'); return; }
  console.log(`  ground truth: row VP=(${trueRowVp.x.toFixed(1)},${trueRowVp.y.toFixed(1)}) col VP=(${trueColVp.x.toFixed(1)},${trueColVp.y.toFixed(1)})`);

  // --- Current pipeline: discrete peaks + RANSAC pairing + rescue pool ---
  const t0 = performance.now();
  const { strong: peaks, weak: rescuePeaks } = findLinePeaksTiered(field, 0.15, 0.15 * RESCUE_THRESHOLD_FRACTION, 4, 3);
  let currentRowVp: VanishingPoint | null = null, currentColVp: VanishingPoint | null = null;
  try {
    const split = splitIntoTwoFamilies(peaks, w, h, 6, 60, rescuePeaks);
    const distTo = (vp: VanishingPoint, truth: { x: number; y: number }) => vpIsFinite(vp) ? Math.hypot(vpToPoint(vp).x - truth.x, vpToPoint(vp).y - truth.y) : Infinity;
    if (distTo(split.familyA.vp, trueRowVp) < distTo(split.familyB.vp, trueRowVp)) {
      currentRowVp = split.familyA.vp; currentColVp = split.familyB.vp;
    } else {
      currentRowVp = split.familyB.vp; currentColVp = split.familyA.vp;
    }
  } catch { /* leave null */ }
  const t1 = performance.now();

  // --- EM, take 1: every raw active cell (~20K), zero filtering ---
  const allCells = extractActiveCells(field);
  console.log(`  -- EM per-restart diagnostics (ALL ${allCells.length} active cells, zero filtering) --`);
  const { vpA, vpB } = findVpsViaEM(allCells, w, h, trueRowVp, trueColVp, true);
  const t2 = performance.now();

  // --- EM, take 2: same soft/thresholdless FITTING loop, but given the
  // strong+rescue peaks (already computed above) as its input population
  // instead of literally every raw bin -- tests whether EM's difficulty is
  // "no threshold at all lets noise volume swamp the fit" specifically,
  // keeping the same code path (extractActiveCells is NOT used here; peaks
  // and rescuePeaks are LineCandidate[] already). Still far more permissive
  // than the current pipeline's hard family-assignment step (NMS+RANSAC
  // decide FAMILY MEMBERSHIP; here they only decide which lines EM gets to
  // see at all -- every one of them still contributes continuously, with no
  // hard in/out family assignment, to BOTH clusters' fits every iteration.
  const filteredCells = [...peaks, ...rescuePeaks];
  console.log(`  -- EM per-restart diagnostics (${filteredCells.length} strong+rescue peaks only) --`);
  const { vpA: vpA2, vpB: vpB2 } = findVpsViaEM(filteredCells, w, h, trueRowVp, trueColVp, true);
  const t3 = performance.now();

  function report(label: string, est: { x: number; y: number } | null, truth: { x: number; y: number }) {
    if (!est) { console.log(`  ${label}: NOT FOUND`); return; }
    const err = Math.hypot(est.x - truth.x, est.y - truth.y);
    console.log(`  ${label}: (${est.x.toFixed(1)},${est.y.toFixed(1)}) err=${err.toFixed(1)}px`);
  }

  console.log(`  -- current pipeline (discrete peaks + RANSAC + rescue), ${(t1 - t0).toFixed(1)}ms, ${peaks.length} strong + ${rescuePeaks.length} rescue peaks --`);
  report('row VP', currentRowVp && vpIsFinite(currentRowVp) ? vpToPoint(currentRowVp) : null, trueRowVp);
  report('col VP', currentColVp && vpIsFinite(currentColVp) ? vpToPoint(currentColVp) : null, trueColVp);

  function reportPair(label: string, a: VanishingPoint, b: VanishingPoint) {
    const rowIsA = vpIsFinite(a) && (!vpIsFinite(b) || Math.hypot(vpToPoint(a).x - trueRowVp!.x, vpToPoint(a).y - trueRowVp!.y) < Math.hypot(vpToPoint(b).x - trueRowVp!.x, vpToPoint(b).y - trueRowVp!.y));
    const [row, col] = rowIsA ? [a, b] : [b, a];
    console.log(`  -- ${label} --`);
    report('row VP', vpIsFinite(row) ? vpToPoint(row) : null, trueRowVp!);
    report('col VP', vpIsFinite(col) ? vpToPoint(col) : null, trueColVp!);
  }
  reportPair(`EM take 1 (ALL ${allCells.length} active cells), ${(t2 - t1).toFixed(1)}ms`, vpA, vpB);
  reportPair(`EM take 2 (${filteredCells.length} strong+rescue peaks only), ${(t3 - t2).toFixed(1)}ms`, vpA2, vpB2);
}

runScenario('moderate tilt', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.4, azimuth: 0.5, roll: 0.3 });
runScenario('steep tilt (known weak-family case)', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.9, azimuth: 1.0, roll: -0.2 });
