// EXPERIMENT, not a regression test: prototypes finding vanishing points
// directly from the raw Hough accumulator field, instead of extracting
// discrete peaks first and RANSAC-pairing them (src/vp.ts's
// splitIntoTwoFamilies). Nothing here is wired into the live pipeline.
//
// The idea: for a fixed (theta,rho) accumulator cell with weight w, the set
// of candidate vanishing points (vx,vy) consistent with it is exactly the
// line { (vx,vy) : vx*cos(theta) + vy*sin(theta) = rho } -- a straight line
// in VP-space, one Hough duality step further than the original transform
// (which mapped raw edge pixels to sinusoidal curves in (theta,rho) space;
// this maps (theta,rho) accumulator cells to STRAIGHT lines in (vx,vy)
// space). Rasterizing every cell's weight into that VP-space line and
// finding where lines concur builds a SECOND accumulator whose peaks are
// the true vanishing points directly -- using every cell's raw weight, not
// just the subset that survived NMS + a fixed vote threshold as discrete
// peaks. This is a real, documented family of techniques (sometimes called
// dual-space or accumulator-based vanishing point detection), motivated
// here specifically because it uses continuous weight rather than a hard
// per-peak threshold, which is exactly what src/vp.ts's splitIntoTwoFamilies
// extraLines rescue mechanism has to work around today: a systematically
// weak family's real members can fail a fixed vote threshold entirely, but
// their weight still lands in the right place in a weight-summing
// accumulator regardless of whether any single cell alone would have
// cleared that threshold.
//
// Validates against ground-truth VPs (computed independently, by
// intersecting two exactly-projected parallel world lines -- not derived
// from anything this experiment is trying to validate) and against the
// CURRENT pipeline's discrete-peaks + RANSAC estimate, on the same raw
// accumulator field, across a normal-imbalance and a KNOWN-weak-family
// scenario (steep tilt, see scripts/test-hough-lines-real-content.ts, which
// measured a genuine 55%/29% detection-count split at ~52deg tilt).
//
// FINDINGS SO FAR (informal, small sample -- this is a prototype, not a
// validated result):
//   - Real, measured win in the scenario that motivated this: at steep tilt
//     (the known weak-family case), the dual accumulator found the WEAK
//     family's VP more accurately than the current pipeline (16.0px vs
//     43.9px error), while staying competitive on the strong family
//     (28.7px vs 20.3px). This directly supports the core motivation: using
//     every cell's raw weight, with no per-peak vote threshold at all,
//     helps exactly when one family is systematically under-detected.
//   - Real, understood failure mode: for a VP far outside the visible frame
//     (common when one axis is close to fronto-parallel), a naive flat
//     Cartesian (vx,vy) grid dilutes that VP's vote mass across many bins
//     instead of concentrating it -- confirmed directly by comparing a 7x7
//     neighborhood SUM at the true VP location (1.9M) against the tallest
//     SINGLE bin found elsewhere (103K): the true answer's mass is real but
//     smeared, so plain per-bin NMS picks a sharper-but-wrong peak instead.
//   - Tried box-summing before NMS as the fix (same idea as src/lines.ts's
//     own pre-gradient boxBlur) -- this DID recover the diluted true peak,
//     but also amplified a spurious crossing hot-spot elsewhere badly
//     enough to make the OTHER (previously correct) family's estimate
//     worse. This is a known, named problem in the vanishing-point-
//     detection literature: naive flat-grid dual accumulators need a
//     non-uniform parametrization (log-polar remapping, or projecting onto
//     a bounded "Gaussian sphere") to represent both nearby and very
//     distant VPs without either diluting the far ones or amplifying
//     coincidental crossings near the center -- a real, substantially
//     bigger implementation lift than this prototype attempts, not
//     something a quick blur radius tweak resolves.
//   - Current default here (smoothRadiusBins=0, i.e. no blur) keeps the
//     clean steep-tilt win and accepts the known distant-VP dilution
//     failure mode as an honest, explained limitation rather than papering
//     over it with a fix that traded one failure for a different one.
//
// Usage: node scripts/experiments/dual-hough-vp.ts

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus } from '../../src/debruijn.ts';
import { toGrayscale } from '../../src/decode.ts';
import { buildLineAccumulator, findLinePeaksTiered } from '../../src/lines.ts';
import type { HoughField } from '../../src/lines.ts';
import { crossLines, splitIntoTwoFamilies, vpToPoint, vpIsFinite } from '../../src/vp.ts';
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

// --- Ground truth: intersect two exactly-projected parallel world lines to
// get the true VP directly, independent of anything under test here. ---
function trueLineAbs(pose: CameraPose, worldA: [number, number], worldB: [number, number]): [number, number, number] | null {
  const pa = projectToImage(pose, RAW, RAW, worldA[0], worldA[1]);
  const pb = projectToImage(pose, RAW, RAW, worldB[0], worldB[1]);
  if (!pa || !pb) return null;
  const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
  let theta = Math.atan2(dx, -dy);
  const nx = -dy, ny = dx;
  theta = Math.atan2(ny, nx);
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  const a = Math.cos(theta), b = Math.sin(theta);
  const c = -(a * pa[0] + b * pa[1]);
  return [a, b, c];
}
function trueVp(pose: CameraPose, direction: 'row' | 'col'): { x: number; y: number } | null {
  // Span sized to the actual visible field of view (same as
  // test-hough-lines-real-content.ts) -- a larger span pushes endpoints
  // behind the camera and silently fails, which isn't "VP at infinity" at
  // all, just a bad test-point choice.
  const fovSpan = (pose.dist / pose.focal) * RAW;
  const l1 = direction === 'row' ? trueLineAbs(pose, [-fovSpan, 0], [fovSpan, 0]) : trueLineAbs(pose, [0, -fovSpan], [0, fovSpan]);
  const l2 = direction === 'row' ? trueLineAbs(pose, [-fovSpan, 8 * cellPx], [fovSpan, 8 * cellPx]) : trueLineAbs(pose, [8 * cellPx, -fovSpan], [8 * cellPx, fovSpan]);
  if (!l1 || !l2) return null;
  const p = crossLines(l1, l2);
  if (Math.abs(p.w) < 1e-9) return null; // true VP at infinity -- out of scope for this prototype, see header
  return { x: p.x / p.w, y: p.y / p.w };
}

// --- The experiment: a second accumulator over (vx,vy), center-relative to
// match src/lines.ts's rho convention directly. ---
interface VpField { acc: Float64Array; width: number; height: number; minX: number; minY: number; binSize: number; }

function buildVpAccumulator(field: HoughField, minX: number, maxX: number, minY: number, maxY: number, binSize: number): VpField {
  const width = Math.ceil((maxX - minX) / binSize);
  const height = Math.ceil((maxY - minY) / binSize);
  const acc = new Float64Array(width * height);
  const { thetaBins, rhoBins, rhoMin, rhoBinSize } = field;
  for (let tb = 0; tb < thetaBins; tb++) {
    const theta = (tb / thetaBins) * Math.PI;
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    for (let rb = 0; rb < rhoBins; rb++) {
      const w = field.acc[tb * rhoBins + rb];
      if (w <= 0) continue;
      const rho = rhoMin + (rb + 0.5) * rhoBinSize;
      // Line in VP-space: vx*cosT + vy*sinT = rho. Step along whichever axis
      // the line is more "horizontal" with respect to, same trick used to
      // rasterize any-angle lines without gaps.
      if (Math.abs(sinT) > Math.abs(cosT)) {
        for (let xi = 0; xi < width; xi++) {
          const vx = minX + (xi + 0.5) * binSize;
          const vy = (rho - vx * cosT) / sinT;
          const yi = Math.floor((vy - minY) / binSize);
          if (yi < 0 || yi >= height) continue;
          acc[yi * width + xi] += w;
        }
      } else {
        for (let yi = 0; yi < height; yi++) {
          const vy = minY + (yi + 0.5) * binSize;
          const vx = (rho - vy * sinT) / cosT;
          const xi = Math.floor((vx - minX) / binSize);
          if (xi < 0 || xi >= width) continue;
          acc[yi * width + xi] += w;
        }
      }
    }
  }
  return { acc, width, height, minX, minY, binSize };
}

// Box-sum smoothing before peak-finding: voting a full LINE per source cell
// (rather than a single refined point) spreads real mass smoothly across a
// broad ridge here, rather than concentrating it in one sharp bin the way
// src/lines.ts's own (already NMS-refined) peaks do -- confirmed directly by
// comparing raw per-bin values against a summed neighborhood at the TRUE VP
// location (1.9M summed vs 103K in the single tallest bin nearby), so a
// narrower but taller spurious spike elsewhere can out-rank the true,
// smoothly-spread answer under plain per-bin NMS. Box-summing over a window
// first (same principle as src/lines.ts's own boxBlur before gradient
// computation) converts "spread-out real mass" back into a genuine
// single-bin peak before NMS ever runs.
function boxSum(acc: Float64Array, width: number, height: number, radius: number): Float64Array {
  const out = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= width) continue;
          sum += acc[yy * width + xx];
        }
      }
      out[y * width + x] = sum;
    }
  }
  return out;
}

// Local-max NMS followed by greedy top-K selection with a minimum
// separation, run on the box-summed field.
function findVpPeaks(vpField: VpField, cx: number, cy: number, k: number, smoothRadiusBins: number, nmsRadiusBins: number, minSeparationPx: number): { x: number; y: number; weight: number }[] {
  const { width, height, minX, minY, binSize } = vpField;
  const acc = boxSum(vpField.acc, width, height, smoothRadiusBins);
  const candidates: { xi: number; yi: number; v: number }[] = [];
  for (let yi = 0; yi < height; yi++) {
    for (let xi = 0; xi < width; xi++) {
      const v = acc[yi * width + xi];
      if (v <= 0) continue;
      let isPeak = true;
      for (let dy = -nmsRadiusBins; dy <= nmsRadiusBins && isPeak; dy++) {
        const yy = yi + dy;
        if (yy < 0 || yy >= height) continue;
        for (let dx = -nmsRadiusBins; dx <= nmsRadiusBins; dx++) {
          if (dx === 0 && dy === 0) continue;
          const xx = xi + dx;
          if (xx < 0 || xx >= width) continue;
          if (acc[yy * width + xx] > v) { isPeak = false; break; }
        }
      }
      if (isPeak) candidates.push({ xi, yi, v });
    }
  }
  candidates.sort((a, b) => b.v - a.v);
  const kept: { xi: number; yi: number; v: number }[] = [];
  for (const c of candidates) {
    if (kept.length >= k) break;
    const cxPx = minX + (c.xi + 0.5) * binSize, cyPx = minY + (c.yi + 0.5) * binSize;
    if (kept.some(k2 => {
      const kx = minX + (k2.xi + 0.5) * binSize, ky = minY + (k2.yi + 0.5) * binSize;
      return Math.hypot(kx - cxPx, ky - cyPx) < minSeparationPx;
    })) continue;
    kept.push(c);
  }
  return kept.map(c => ({ x: cx + minX + (c.xi + 0.5) * binSize, y: cy + minY + (c.yi + 0.5) * binSize, weight: c.v }));
}

function runScenario(name: string, pose: CameraPose) {
  console.log(`\n=== ${name} (tilt=${((pose.tilt * 180) / Math.PI).toFixed(0)}deg) ===`);
  const rgba = captureHomography(png, pose, RAW, RAW, 4);
  const gray = toGrayscale(rgba, RAW, RAW);
  const field = buildLineAccumulator(gray, RAW, RAW, HOUGH_THETA_BINS, HOUGH_RHO_BIN_PX);
  const cx = RAW / 2, cy = RAW / 2;

  const trueRowVp = trueVp(pose, 'row');
  const trueColVp = trueVp(pose, 'col');
  if (!trueRowVp || !trueColVp) { console.log('  (skipped -- a true VP is at/near infinity, out of scope for this prototype)'); return; }
  console.log(`  ground truth: row VP=(${trueRowVp.x.toFixed(1)},${trueRowVp.y.toFixed(1)}) col VP=(${trueColVp.x.toFixed(1)},${trueColVp.y.toFixed(1)})`);

  // --- Current pipeline: discrete peaks + RANSAC pairing ---
  const t0 = performance.now();
  const { strong: peaks, weak: rescuePeaks } = findLinePeaksTiered(field, 0.15, 0.15 * RESCUE_THRESHOLD_FRACTION, 4, 3);
  let currentRowVp: VanishingPoint | null = null, currentColVp: VanishingPoint | null = null;
  try {
    const split = splitIntoTwoFamilies(peaks, RAW, RAW, 6, 60, rescuePeaks);
    // Match recovered families to row/col by proximity to ground truth.
    const distTo = (vp: VanishingPoint, truth: { x: number; y: number }) => vpIsFinite(vp) ? Math.hypot(vpToPoint(vp).x - truth.x, vpToPoint(vp).y - truth.y) : Infinity;
    if (distTo(split.familyA.vp, trueRowVp) < distTo(split.familyB.vp, trueRowVp)) {
      currentRowVp = split.familyA.vp; currentColVp = split.familyB.vp;
    } else {
      currentRowVp = split.familyB.vp; currentColVp = split.familyA.vp;
    }
  } catch { /* leave null */ }
  const t1 = performance.now();

  // --- Dual accumulator: vote every raw cell's weight directly ---
  const span = RAW * 7;
  const vpField = buildVpAccumulator(field, -span, span, -span, span, 3);
  // smoothRadiusBins=0 (no box-blur) is the current default -- see the
  // module header's "Findings" section for why blurring, tried as a fix for
  // diluted distant-VP peaks, was reverted: it can amplify a spurious
  // crossing hot-spot elsewhere badly enough to make the OTHER family worse,
  // a net regression in the one trial run here. Left toggleable (just pass
  // a nonzero radius) since it's a real, recorded negative finding, not
  // something to silently discard.
  const vpPeaks = findVpPeaks(vpField, cx, cy, 2, 0, 15, RAW * 0.5);
  const t2 = performance.now();

  function report(label: string, est: { x: number; y: number } | null, truth: { x: number; y: number }) {
    if (!est) { console.log(`  ${label}: NOT FOUND`); return; }
    const err = Math.hypot(est.x - truth.x, est.y - truth.y);
    console.log(`  ${label}: (${est.x.toFixed(1)},${est.y.toFixed(1)}) err=${err.toFixed(1)}px`);
  }

  console.log(`  -- current pipeline (discrete peaks + RANSAC), ${(t1 - t0).toFixed(1)}ms --`);
  report('row VP', currentRowVp && vpIsFinite(currentRowVp) ? vpToPoint(currentRowVp) : null, trueRowVp);
  report('col VP', currentColVp && vpIsFinite(currentColVp) ? vpToPoint(currentColVp) : null, trueColVp);

  console.log(`  -- dual accumulator (direct weight voting), ${(t2 - t1).toFixed(1)}ms --`);
  const dualByRow = vpPeaks.slice().sort((a, b) => Math.hypot(a.x - trueRowVp.x, a.y - trueRowVp.y) - Math.hypot(b.x - trueRowVp.x, b.y - trueRowVp.y));
  report('row VP (closest of 2 found peaks)', dualByRow[0] ?? null, trueRowVp);
  const dualByCol = vpPeaks.slice().sort((a, b) => Math.hypot(a.x - trueColVp.x, a.y - trueColVp.y) - Math.hypot(b.x - trueColVp.x, b.y - trueColVp.y));
  report('col VP (closest of 2 found peaks)', dualByCol[0] ?? null, trueColVp);
  console.log(`  raw peaks found: ${peaks.length} strong + ${rescuePeaks.length} rescue candidates; VP accumulator peaks: ${vpPeaks.map(p => `(${p.x.toFixed(0)},${p.y.toFixed(0)},w=${p.weight.toFixed(0)})`).join(' ')}`);

  // Diagnostic: how much weight actually landed AT the true VP's own bin,
  // vs. the strongest peak found elsewhere -- distinguishes "the true VP's
  // vote mass is diluted/spread thin" from "something is just broken".
  function weightAt(truth: { x: number; y: number }): number {
    const xi = Math.floor((truth.x - cx - vpField.minX) / vpField.binSize);
    const yi = Math.floor((truth.y - cy - vpField.minY) / vpField.binSize);
    if (xi < 0 || xi >= vpField.width || yi < 0 || yi >= vpField.height) return NaN;
    // Sum a small neighborhood, not just the single bin, since sub-bin
    // position noise means the true peak isn't necessarily in exactly one bin.
    let sum = 0;
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
      const yy = yi + dy, xx = xi + dx;
      if (yy < 0 || yy >= vpField.height || xx < 0 || xx >= vpField.width) continue;
      sum += vpField.acc[yy * vpField.width + xx];
    }
    return sum;
  }
  console.log(`  weight AT true row VP's own bin (7x7 nbhd): ${weightAt(trueRowVp).toFixed(0)}  (found peak weight: ${vpPeaks[0]?.weight.toFixed(0) ?? 'n/a'})`);
  console.log(`  weight AT true col VP's own bin (7x7 nbhd): ${weightAt(trueColVp).toFixed(0)}  (found peak weight: ${vpPeaks[1]?.weight.toFixed(0) ?? 'n/a'})`);
}

runScenario('moderate tilt', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.4, azimuth: 0.5, roll: 0.3 });
runScenario('steep tilt (known weak-family case)', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.9, azimuth: 1.0, roll: -0.2 });
