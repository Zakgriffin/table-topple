// Validates fitHomographyFromVPAndPatch — the "option (a)" construction
// that fixes H's two direction columns from vanishing points and only fits
// two scalars (per-axis scale) from a small local patch, instead of 8
// unknowns from the patch alone (see scripts/test-homography.ts, which
// showed the plain-DLT-from-clustered-patch approach extrapolating to
// 10-90px error under mild noise).
//
// Two parts:
//   1. Ground-truth VP: isolates the fitHomographyFromVPAndPatch math itself
//      from vanishing.ts's estimation quality, using exact VPs from
//      synth-camera's closed-form vanishingPointForDirection.
//   2. Estimated VP: feeds through src/vanishing.ts's actual gradient-based
//      estimate (on a real rendered pattern) to see how much its known
//      imperfection actually costs downstream, once combined with the local
//      patch.
//
// Usage: node scripts/test-homography-vp.ts [order]

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { toGrayscale } from '../src/decode.ts';
import { estimateVanishingPoints } from '../src/vanishing.ts';
import { fitHomographyFromVPAndPatch, applyHomography } from '../src/homography.ts';
import type { PointCorrespondence } from '../src/homography.ts';
import { captureHomography, projectToImage, vanishingPointForDirection } from './lib/synth-camera.ts';
import type { CameraPose } from './lib/synth-camera.ts';

const RAW = 300, DIST = 300, FOCAL = 300, CELL = 20;

function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

// Builds a small clustered local patch (mimicking a short-range BFS seed)
// around lattice origin (0,0), in WORLD pixel space via u*CELL, v*CELL —
// same "world units are pixels" convention scripts/test-build-mesh-exact.ts
// uses, since this test only cares about geometry, not pattern content.
function buildPatch(pose: CameraPose, noisePx: number, rng: () => number): PointCorrespondence[] {
  const pts: PointCorrespondence[] = [];
  for (let v = -1; v <= 1; v++) {
    for (let u = -1; u <= 1; u++) {
      const img = projectToImage(pose, RAW, RAW, u * CELL, v * CELL);
      if (!img) continue;
      pts.push({ u, v, x: img[0] + (rng() - 0.5) * 2 * noisePx, y: img[1] + (rng() - 0.5) * 2 * noisePx });
    }
  }
  return pts;
}

function heldOutError(pose: CameraPose, H: Float64Array, usedUV: Set<string>): number {
  let sum = 0, count = 0;
  for (let v = -8; v <= 8; v++) {
    for (let u = -8; u <= 8; u++) {
      if (usedUV.has(`${u},${v}`)) continue;
      const trueImg = projectToImage(pose, RAW, RAW, u * CELL, v * CELL);
      const estImg = applyHomography(H, u, v);
      if (!trueImg || !estImg) continue;
      sum += Math.hypot(trueImg[0] - estImg[0], trueImg[1] - estImg[1]);
      count++;
    }
  }
  return count ? sum / count : NaN;
}

console.log('Part 1: fitHomographyFromVPAndPatch with GROUND-TRUTH VPs\n');
for (const tiltDeg of [0, 10, 20, 30, 45, 60]) {
  for (const noisePx of [0, 0.5, 1, 2]) {
    const errs: number[] = [];
    let failures = 0;
    const TRIALS = 20;
    for (let t = 0; t < TRIALS; t++) {
      const rng = makeRng(tiltDeg * 1000 + noisePx * 100 + t);
      const pose: CameraPose = {
        targetX: 0, targetY: 0, dist: DIST, focal: FOCAL,
        tilt: tiltDeg * Math.PI / 180, azimuth: rng() * 2 * Math.PI, roll: rng() * 2 * Math.PI,
      };
      const patch = buildPatch(pose, noisePx, rng);
      const origin = patch.find(p => p.u === 0 && p.v === 0);
      if (!origin || patch.length < 4) { failures++; continue; }
      const vpRow = vanishingPointForDirection(pose, RAW, RAW, 1, 0);
      const vpCol = vanishingPointForDirection(pose, RAW, RAW, 0, 1);
      const H = fitHomographyFromVPAndPatch(vpRow, vpCol, origin, patch);
      if (!H) { failures++; continue; }
      const usedUV = new Set(patch.map(p => `${p.u},${p.v}`));
      const err = heldOutError(pose, H, usedUV);
      if (!Number.isNaN(err)) errs.push(err);
    }
    const mean = errs.length ? errs.reduce((s, v) => s + v, 0) / errs.length : NaN;
    console.log(`tilt=${String(tiltDeg).padStart(2)}deg noise=${noisePx}px: heldOutErr mean=${mean.toFixed(3)}px, failures=${failures}/${TRIALS}`);
  }
}

console.log('\nPart 2: fitHomographyFromVPAndPatch with ESTIMATED VPs (real gradient field)\n');
const order = parseInt(process.argv[2] ?? '4', 10);
const png = PNG.sync.read(readFileSync(`samples/order${order}.png`));
for (const tiltDeg of [20, 30, 45, 60]) {
  const errs: number[] = [];
  let vpFailures = 0, fitFailures = 0;
  const TRIALS = 12;
  for (let t = 0; t < TRIALS; t++) {
    const rng = makeRng(tiltDeg * 5000 + t);
    const pose: CameraPose = {
      targetX: png.width / 2, targetY: png.height / 2, dist: DIST, focal: FOCAL,
      tilt: tiltDeg * Math.PI / 180, azimuth: rng() * 2 * Math.PI, roll: rng() * 2 * Math.PI,
    };
    const rgba = captureHomography(png, pose, RAW, RAW, 4);
    const gray = toGrayscale(rgba, RAW, RAW);
    const est = estimateVanishingPoints(gray, RAW, RAW);
    if (!est) { vpFailures++; continue; }

    // Real patch positions (0.5px noise, plausible for sub-pixel-refined
    // corner detection) around a seed near image center.
    const patch = buildPatch(pose, 0.5, rng);
    const origin = patch.find(p => p.u === 0 && p.v === 0);
    if (!origin || patch.length < 4) { fitFailures++; continue; }

    // Ground truth used only to pick which of est.vp1/vp2 is "row" vs "col"
    // — matching what the real pipeline would do by checking which is more
    // consistent with the patch's own local basis, but that logic doesn't
    // exist yet, so this test takes the best of both pairings as an
    // upper-bound stand-in.
    const gtRow = vanishingPointForDirection(pose, RAW, RAW, 1, 0);
    const H_a = fitHomographyFromVPAndPatch(est.vp1, est.vp2, origin, patch);
    const H_b = fitHomographyFromVPAndPatch(est.vp2, est.vp1, origin, patch);
    const usedUV = new Set(patch.map(p => `${p.u},${p.v}`));
    const errA = H_a ? heldOutError(pose, H_a, usedUV) : NaN;
    const errB = H_b ? heldOutError(pose, H_b, usedUV) : NaN;
    const best = Math.min(...[errA, errB].filter(e => !Number.isNaN(e)));
    if (Number.isFinite(best)) errs.push(best); else fitFailures++;
  }
  const mean = errs.length ? errs.reduce((s, v) => s + v, 0) / errs.length : NaN;
  const median = errs.length ? [...errs].sort((a, b) => a - b)[Math.floor(errs.length / 2)] : NaN;
  console.log(`tilt=${String(tiltDeg).padStart(2)}deg: heldOutErr mean=${mean.toFixed(1)}px median=${median.toFixed(1)}px, vpFailures=${vpFailures}, fitFailures=${fitFailures} (n=${TRIALS})`);
}
