// Validates Level 1 (buildLineAccumulator + findLinePeaks) against REAL
// rendered pattern content sampled through a real camera pose -- not the
// clean synthetic stripe/grid patterns test-hough-lines.ts already covers,
// which have perfectly uniform edge contrast by construction and can't
// reveal a real-content-specific accuracy problem.
//
// Reports row-direction and column-direction line accuracy SEPARATELY
// (not just combined), motivated directly by live-device testing: the user
// observed real captures sometimes detecting almost all of one line family
// and almost none of the other, especially near fronto-parallel. A
// dedicated investigation (see src/vp.ts's splitIntoTwoFamilies extraLines
// comment) ruled out both a content-based flip-rate asymmetry (measured
// exactly 50.0% in both directions) and a Hough theta=0/PI wrap-seam bug
// (a controlled synthetic sweep across roll angles showed no imbalance at
// any angle) -- pointing at a real-camera-capture artifact (lighting,
// focus, sensor/ISP directional sharpening) rather than a fixable detection
// bug. This test is the ongoing regression check for that conclusion: if a
// future change ever DOES introduce a real directional accuracy bug, it
// should show up here even though the two hypotheses above didn't pan out.
//
// Usage: node scripts/test-hough-lines-real-content.ts

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus } from '../src/debruijn.ts';
import { toGrayscale } from '../src/decode.ts';
import { buildLineAccumulator, findLinePeaks } from '../src/lines.ts';
import type { LineCandidate } from '../src/lines.ts';
import { captureHomography, projectToImage } from './lib/synth-camera.ts';
import type { CameraPose } from './lib/synth-camera.ts';

let pass = 0, fail = 0;
function check(name: string, ok: boolean, detail: string) {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${detail}`);
  if (ok) pass++; else fail++;
}

const order = 4;
const debruijn = generateTorus(order);
const { C } = debruijn;
const png = PNG.sync.read(readFileSync(`samples/order${order}.png`));
const cellPx = png.width / C;
const RAW = 300;
const HOUGH_RHO_BIN_PX = 1.5;
const HOUGH_THETA_BINS = Math.round(360 / HOUGH_RHO_BIN_PX);

// Ground-truth (theta,rho) for the world line through two points, projected
// through the known pose -- same convention src/lines.ts uses (normal angle
// folded to [0,PI), rho relative to image center).
function trueLine(pose: CameraPose, worldA: [number, number], worldB: [number, number]): { theta: number; rho: number } | null {
  const pa = projectToImage(pose, RAW, RAW, worldA[0], worldA[1]);
  const pb = projectToImage(pose, RAW, RAW, worldB[0], worldB[1]);
  if (!pa || !pb) return null;
  const cx = RAW / 2, cy = RAW / 2;
  const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
  let theta = Math.atan2(dx, -dy);
  const nx = -dy, ny = dx;
  theta = Math.atan2(ny, nx);
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  const a = Math.cos(theta), b = Math.sin(theta);
  const rho = a * (pa[0] - cx) + b * (pa[1] - cy);
  return { theta, rho };
}

function angularDistRad(t1: number, t2: number): number {
  const d = Math.abs(t1 - t2) % Math.PI;
  return Math.min(d, Math.PI - d);
}

interface FamilyScore { n: number; meanRhoErr: number; meanThetaErrDeg: number; maxRhoErr: number; }

function scoreFamily(trueLines: { theta: number; rho: number }[], peaks: LineCandidate[]): FamilyScore {
  let sumRho = 0, sumTheta = 0, n = 0, maxRho = 0;
  for (const t of trueLines) {
    let best: LineCandidate | null = null, bestDist = Infinity;
    for (const p of peaks) {
      const dTheta = angularDistRad(p.theta, t.theta);
      if (dTheta > (5 * Math.PI) / 180) continue; // only consider plausibly-matching orientation
      const dRho = Math.abs(p.rho - t.rho);
      const dist = dRho + dTheta * 100; // rho dominates; theta breaks ties among near-parallel peaks
      if (dist < bestDist) { bestDist = dist; best = p; }
    }
    if (!best) continue; // this true line simply wasn't detected -- not every line needs to be found
    const dRho = Math.abs(best.rho - t.rho);
    if (dRho > 10) continue; // no plausible match nearby -- don't count a garbage residual
    sumRho += dRho;
    sumTheta += (angularDistRad(best.theta, t.theta) * 180) / Math.PI;
    n++;
    maxRho = Math.max(maxRho, dRho);
  }
  return { n, meanRhoErr: n ? sumRho / n : NaN, meanThetaErrDeg: n ? sumTheta / n : NaN, maxRhoErr: maxRho };
}

function runScenario(name: string, pose: CameraPose) {
  const rgba = captureHomography(png, pose, RAW, RAW, 4);
  const gray = toGrayscale(rgba, RAW, RAW);
  const field = buildLineAccumulator(gray, RAW, RAW, HOUGH_THETA_BINS, HOUGH_RHO_BIN_PX);
  const peaks = findLinePeaks(field, 0.15, 4, 3);

  // Span sized to roughly match the actual visible field of view at this
  // pose (dist/focal * RAW world units around the target), not an arbitrary
  // large number -- a line whose endpoints fall outside that span is prone
  // to landing behind the camera and being discarded for a reason that has
  // nothing to do with Level 1's real accuracy.
  const fovSpan = (pose.dist / pose.focal) * RAW;
  const rowTrue: { theta: number; rho: number }[] = [];
  const colTrue: { theta: number; rho: number }[] = [];
  for (let k = -Math.round(fovSpan / cellPx); k <= Math.round(fovSpan / cellPx); k++) {
    const y = k * cellPx;
    const t = trueLine(pose, [-fovSpan, y], [fovSpan, y]);
    if (t) rowTrue.push(t);
    const x = k * cellPx;
    const c = trueLine(pose, [x, -fovSpan], [x, fovSpan]);
    if (c) colTrue.push(c);
  }

  const rowScore = scoreFamily(rowTrue, peaks);
  const colScore = scoreFamily(colTrue, peaks);
  console.log(
    `${name}: row matched=${rowScore.n}/${rowTrue.length} meanRhoErr=${rowScore.meanRhoErr.toFixed(2)}px ` +
    `meanThetaErr=${rowScore.meanThetaErrDeg.toFixed(2)}deg maxRhoErr=${rowScore.maxRhoErr.toFixed(2)}px`
  );
  console.log(
    `${' '.repeat(name.length)}  col matched=${colScore.n}/${colTrue.length} meanRhoErr=${colScore.meanRhoErr.toFixed(2)}px ` +
    `meanThetaErr=${colScore.meanThetaErrDeg.toFixed(2)}deg maxRhoErr=${colScore.maxRhoErr.toFixed(2)}px`
  );

  // Threshold is realistic for REAL content, not test-hough-lines.ts's clean
  // synthetic stripes (<1px there): a real cell boundary's local contrast
  // depends on the specific neighboring bits, unlike a stripe's uniformly
  // strong edge, and captureHomography's antialiasing softens it further —
  // ~4-5px mean error is what real content actually achieves here, a
  // genuine (if modest) precision limit worth tracking, not a bug to chase.
  check(`${name}: row-direction lines detected reasonably accurately`, rowScore.n >= 5 && rowScore.meanRhoErr < 7, `n=${rowScore.n} meanRhoErr=${rowScore.meanRhoErr.toFixed(2)}px`);
  check(`${name}: col-direction lines detected reasonably accurately`, colScore.n >= 5 && colScore.meanRhoErr < 7, `n=${colScore.n} meanRhoErr=${colScore.meanRhoErr.toFixed(2)}px`);
  check(
    `${name}: no directional ACCURACY bias between row and col families`,
    Math.abs(rowScore.meanRhoErr - colScore.meanRhoErr) < 2,
    `row=${rowScore.meanRhoErr.toFixed(2)}px col=${colScore.meanRhoErr.toFixed(2)}px`
  );
  // Detection-COUNT bias is a different story: at steep tilt, genuine
  // perspective foreshortening can legitimately compress one family's far
  // side enough to matter (expected geometry, not a bug) -- so this is
  // reported for every scenario but only asserted at gentler tilts, where a
  // real imbalance would actually be suspicious. Reconciles with the
  // separate finding (see module header) that pure in-plane ROLL at tilt=0
  // shows no imbalance at all -- it's TILT/perspective specifically that can
  // cause this, not a Hough detection flaw.
  const matchRateDiff = Math.abs(rowScore.n / rowTrue.length - colScore.n / colTrue.length);
  const isGentle = pose.tilt < 0.6;
  check(
    `${name}: detection-count balance between families${isGentle ? '' : ' (steep tilt -- reported, not asserted; foreshortening can legitimately imbalance this)'}`,
    !isGentle || matchRateDiff < 0.25,
    `row matched ${((rowScore.n / rowTrue.length) * 100).toFixed(0)}%, col matched ${((colScore.n / colTrue.length) * 100).toFixed(0)}%`
  );
}

runScenario('moderate tilt', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.4, azimuth: 0.5, roll: 0.3 });
runScenario('near fronto-parallel', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.05, azimuth: 0.2, roll: 0.1 });
runScenario('steep tilt', { targetX: 0, targetY: 0, dist: 300, focal: 300, tilt: 0.9, azimuth: 1.0, roll: -0.2 });

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
