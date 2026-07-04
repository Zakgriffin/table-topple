// Validates estimateVanishingPoints against exact closed-form ground truth
// (scripts/lib/synth-camera.ts's vanishingPointForDirection) across a real
// perspective tilt sweep, including near-fronto-parallel (tilt=0, where both
// VPs are genuinely at infinity) through steep tilt (where both are finite
// and visibly non-symmetric — the case the old single-averaged-angle
// approach couldn't represent at all).
//
// Usage: node scripts/test-vanishing.ts [order]

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { toGrayscale } from '../src/decode.ts';
import { estimateVanishingPoints } from '../src/vanishing.ts';
import { captureHomography, vanishingPointForDirection } from './lib/synth-camera.ts';
import type { CameraPose, VPGroundTruth } from './lib/synth-camera.ts';

const order = parseInt(process.argv[2] ?? '4', 10);
const png = PNG.sync.read(readFileSync(`samples/order${order}.png`));
console.log(`Loaded samples/order${order}.png`);

const RAW = 300, DIST = 300, FOCAL = 300;
const TRIALS = 8;

function angleDegDiffModPi(a: number, b: number): number {
  let d = (Math.abs(a - b) * 180 / Math.PI) % 180;
  if (d > 90) d = 180 - d;
  return d;
}

// vanishingPointForDirection only reports finite:false when the world
// direction is EXACTLY parallel to the image plane (fwdDot below 1e-9),
// which essentially never happens with continuous random azimuth/roll. But
// a VP 5000px away from a 300px capture is just as unrecoverable from pixel
// data as one at true infinity — so for comparison purposes, reclassify
// "far enough away that only its direction is meaningful" using a
// scale-relative cutoff, matching the kind of practical threshold any
// pixel-based estimator (including ours) has to use internally too.
const FAR_AWAY_MULT = 20;
function effectiveGt(gt: VPGroundTruth): { finite: boolean; x: number; y: number; angle: number } {
  if (!gt.finite) return gt;
  const dist = Math.hypot(gt.x - RAW / 2, gt.y - RAW / 2);
  if (dist > FAR_AWAY_MULT * RAW) {
    return { finite: false, x: 0, y: 0, angle: Math.atan2(gt.y - RAW / 2, gt.x - RAW / 2) };
  }
  return gt;
}

console.log(`\nVanishing point estimation vs perspective tilt (${TRIALS} trials/tilt):`);
for (const tiltDeg of [0, 5, 10, 20, 30, 40, 50, 60]) {
  const pixelErrors: number[] = [];
  const angleErrors: number[] = [];
  let mismatches = 0, noResult = 0;
  const inlierFracs: number[] = [];

  for (let t = 0; t < TRIALS; t++) {
    const testCol = png.width / 2 + (Math.random() - 0.5) * png.width * 0.3;
    const testRow = png.height / 2 + (Math.random() - 0.5) * png.height * 0.3;
    const pose: CameraPose = {
      targetX: testCol, targetY: testRow,
      dist: DIST, focal: FOCAL, tilt: tiltDeg * Math.PI / 180,
      azimuth: Math.random() * 2 * Math.PI, roll: Math.random() * 2 * Math.PI,
    };
    // supersampleN=4: raw per-pixel gradient angle (unlike junction
    // detection's windowed structure tensor) has no spatial averaging to
    // wash out nearest-neighbor staircase aliasing from a single-sample
    // synthetic render — that aliasing was confirmed via debug script to
    // fully explain an earlier spurious 100%-mismatch result (estimated
    // angles landing on exact 0/45/90deg regardless of the true pose). A
    // real camera photo antialiases optically, so this is a test-harness
    // fidelity fix, not a workaround for a real-world failure mode.
    const rgba = captureHomography(png, pose, RAW, RAW, 4);
    const gray = toGrayscale(rgba, RAW, RAW);
    const est = estimateVanishingPoints(gray, RAW, RAW);
    if (!est) { noResult++; continue; }

    const gtRow = effectiveGt(vanishingPointForDirection(pose, RAW, RAW, 1, 0));
    const gtCol = effectiveGt(vanishingPointForDirection(pose, RAW, RAW, 0, 1));
    const pairingA: [typeof est.vp1, typeof gtRow][] = [[est.vp1, gtRow], [est.vp2, gtCol]];
    const pairingB: [typeof est.vp1, typeof gtRow][] = [[est.vp1, gtCol], [est.vp2, gtRow]];
    const scoreOf = (pairing: typeof pairingA) => pairing.reduce((s, [estVp, gt]) => {
      if (estVp.finite !== gt.finite) return s + 1000;
      return s + (gt.finite ? Math.hypot(estVp.x - gt.x, estVp.y - gt.y) : angleDegDiffModPi(estVp.angle, gt.angle));
    }, 0);
    const pairs = scoreOf(pairingA) <= scoreOf(pairingB) ? pairingA : pairingB;
    for (const [estVp, gt] of pairs) {
      if (estVp.finite !== gt.finite) { mismatches++; continue; }
      if (gt.finite) pixelErrors.push(Math.hypot(estVp.x - gt.x, estVp.y - gt.y));
      else angleErrors.push(angleDegDiffModPi(estVp.angle, gt.angle));
    }
    inlierFracs.push(est.vp1.inliers / Math.max(1, est.vp1.total));
    inlierFracs.push(est.vp2.inliers / Math.max(1, est.vp2.total));
  }

  const mean = (a: number[]) => a.length ? (a.reduce((s, v) => s + v, 0) / a.length) : NaN;
  const median = (a: number[]) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  console.log(
    `tilt ${String(tiltDeg).padStart(2)}deg: ` +
    `pxErr mean/median=${mean(pixelErrors).toFixed(1)}/${median(pixelErrors).toFixed(1)} (n=${pixelErrors.length}), ` +
    `angErr mean/median=${mean(angleErrors).toFixed(2)}/${median(angleErrors).toFixed(2)}deg (n=${angleErrors.length}), ` +
    `mismatches=${mismatches}, noResult=${noResult}, meanInlierFrac=${mean(inlierFracs).toFixed(2)}`
  );
}
