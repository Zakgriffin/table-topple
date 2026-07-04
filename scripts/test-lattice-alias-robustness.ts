// Validates the fix for a real bug found via live-device testing: a
// genuinely gap-free family of lines was being mis-indexed as if lines were
// missing (2x, 3x, ... step aliasing), splitting real cells into phantom
// half-cells in the rectified-grid overlay. Root cause: from relative line
// positions ALONE, "no gaps" and "a uniform pattern of missing lines" are
// mathematically indistinguishable (same Mobius shape, different assumed
// step multiplier) — under realistic pixel noise, competing candidate fits
// from different seed windows aren't exactly tied, so one can win on raw
// inlier count before the span-based tie-break (for the noiseless case) ever
// gets a chance to apply. Confirmed via a dedicated repro: 103/200 trials
// (51.5%) aliased on a completely gap-free family under just 1.5px noise.
//
// Fix: recoverIndicesFromTransversal's estimateLocalSpacing measures the
// true local pitch directly from this family's OWN real neighboring
// detections (minimum of a few nearby adjacent-line gaps), and rejects any
// candidate model whose implied per-step pixel spacing is off by more than a
// tolerance from it. This is genuinely new information a single 4-point seed
// window can't provide on its own, which is why it can actually settle the
// ambiguity rather than just re-deriving it from the same data. An earlier
// version of this fix took the reference spacing from an external, GLOBAL
// hint (a coarse apparent-pitch estimate from a totally different
// subsystem) — replaced because a single global number is systematically
// wrong under perspective for windows far from wherever that estimate was
// taken; measuring locally from the family's own data has no such issue.
//
// Usage: node scripts/test-lattice-alias-robustness.ts

import type { LineCandidate } from '../src/lines.ts';
import type { LineFamily } from '../src/vp.ts';
import { estimateVanishingPoint } from '../src/vp.ts';
import { indexFamilyLines } from '../src/lattice.ts';
import { projectToImage, type CameraPose } from './lib/synth-camera.ts';

const W = 640, H = 480, PITCH = 30;
let pass = 0, fail = 0;
function check(name: string, ok: boolean, detail: string) {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${detail}`);
  if (ok) pass++; else fail++;
}

function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function projectLine(pose: CameraPose, worldA: [number, number], worldB: [number, number], rnd: () => number, noisePx: number): LineCandidate | null {
  const pa = projectToImage(pose, W, H, worldA[0], worldA[1]);
  const pb = projectToImage(pose, W, H, worldB[0], worldB[1]);
  if (!pa || !pb) return null;
  const cx = W / 2, cy = H / 2;
  const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
  let nx = -dy, ny = dx;
  let theta = Math.atan2(ny, nx);
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  // Realistic noise on BOTH theta and rho (sub-bin Hough peak jitter), not
  // just rho alone — matches how a real detected peak's position error
  // actually looks.
  theta += (rnd() - 0.5) * noisePx * 0.002;
  const a = Math.cos(theta), b = Math.sin(theta);
  const rho = a * (pa[0] - cx) + b * (pa[1] - cy) + (rnd() - 0.5) * noisePx;
  return { theta, rho, weight: 1 };
}

const pose: CameraPose = { targetX: 0, targetY: 0, dist: 400, tilt: 0.4, azimuth: 0.5, roll: 0.3, focal: 500 };

function buildOtherVp(rnd: () => number) {
  return estimateVanishingPoint(
    Array.from({ length: 10 }, (_, j) => projectLine(pose, [(j - 5) * PITCH, -300], [(j - 5) * PITCH, 300], rnd, 0)!),
    W, H,
  );
}

// --- Scenario 1: gap-free family under realistic noise, many trials ---
// A statistical bar, not a single deterministic check, since this is
// fundamentally about robustness under noise — the pre-fix rate was 51.5%
// aliased; requiring 90%+ correct here is a strong regression guard without
// demanding literally 100% on inherently noisy data.
{
  const TRIALS = 100;
  let correct = 0;
  for (let trial = 0; trial < TRIALS; trial++) {
    const rnd = mulberry32(trial * 97 + 13);
    const trueIdx = new Map<LineCandidate, number>();
    const lines: LineCandidate[] = [];
    for (let i = -12; i <= 12; i++) {
      const line = projectLine(pose, [-300, i * PITCH], [300, i * PITCH], rnd, 1.5);
      if (!line) continue;
      trueIdx.set(line, i);
      lines.push(line);
    }
    const family: LineFamily = { vp: estimateVanishingPoint(lines, W, H), lines };
    const indexed = indexFamilyLines(family, buildOtherVp(rnd), W, H, 4);
    if (indexed.length < lines.length - 3) continue; // too many dropped — not a "correct" trial either way
    const pts = indexed.map(x => ({ rec: x.index, truth: trueIdx.get(x.line)! })).sort((a, b) => a.truth - b.truth);
    const first = pts[0], last = pts[pts.length - 1];
    const m = (last.rec - first.rec) / (last.truth - first.truth);
    if (Math.abs(Math.abs(m) - 1) < 0.05) correct++;
  }
  check('gap-free family, 1.5px noise, 100 trials', correct >= 90, `${correct}/${TRIALS} correctly recovered step=1 (no aliasing)`);
}

// --- Scenario 2: genuine gaps PLUS noise, relying on LOCAL inference alone ---
// Makes sure the scale-check fix doesn't over-correct and start rejecting
// real, necessary multi-step gaps — estimateLocalSpacing constrains the
// PER-INDEX spacing, not the per-DETECTED-LINE spacing, so a correctly
// gap-aware model should still measure ~1x the family's own locally-measured
// pitch even where real gaps exist nearby.
{
  const rnd = mulberry32(42);
  const dropped = new Set([-8, -3, 1, 5]);
  const trueIdx = new Map<LineCandidate, number>();
  const lines: LineCandidate[] = [];
  for (let i = -12; i <= 12; i++) {
    if (dropped.has(i)) continue;
    const line = projectLine(pose, [-300, i * PITCH], [300, i * PITCH], rnd, 1.5);
    if (!line) continue;
    trueIdx.set(line, i);
    lines.push(line);
  }
  const family: LineFamily = { vp: estimateVanishingPoint(lines, W, H), lines };
  const indexed = indexFamilyLines(family, buildOtherVp(rnd), W, H, 4);
  const pts = indexed.map(x => ({ rec: x.index, truth: trueIdx.get(x.line)! })).sort((a, b) => a.truth - b.truth);
  const first = pts[0], last = pts[pts.length - 1];
  const m = pts.length >= 2 ? (last.rec - first.rec) / (last.truth - first.truth) : NaN;
  let maxAffineErr = 0;
  const k = first.rec - m * first.truth;
  for (const p of pts) maxAffineErr = Math.max(maxAffineErr, Math.abs((m * p.truth + k) - p.rec));
  const ok = indexed.length >= lines.length - 2 && Math.abs(Math.abs(m) - 1) < 0.05 && maxAffineErr < 1.01;
  check('genuine gaps + noise, local spacing inference', ok, `recovered ${indexed.length}/${lines.length}, m=${m.toFixed(3)}, maxAffineErr=${maxAffineErr.toFixed(2)}`);
}

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
