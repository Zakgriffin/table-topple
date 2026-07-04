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
// Fix: indexFamilyLines/recoverIndicesFromTransversal now accept an optional
// expectedSpacingPx (the same coarse apparent-pitch estimate src/main.ts
// already computes for Hough bin-sizing, from a totally different method —
// autocorrelation, not line detection) and reject any candidate model whose
// implied per-step pixel spacing (the Mobius transform's local derivative)
// is off by more than a tolerance from it. This is genuinely new information
// the line positions alone can't provide, which is why it can actually
// settle the ambiguity rather than just re-deriving it from the same data.
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

// An independent (imperfect — +-15%, like a real autocorrelation-based
// apparentPitch estimate) pixel-spacing hint, derived from two adjacent TRUE
// lines near the middle of the range.
function makeExpectedSpacingPx(rnd: () => number): number {
  const pa = projectToImage(pose, W, H, -300, 0)!, pb = projectToImage(pose, W, H, 300, 0)!;
  const pc = projectToImage(pose, W, H, -300, PITCH)!, pd = projectToImage(pose, W, H, 300, PITCH)!;
  const trueSpacing = Math.hypot((pc[0] + pd[0]) / 2 - (pa[0] + pb[0]) / 2, (pc[1] + pd[1]) / 2 - (pa[1] + pb[1]) / 2);
  return trueSpacing * (1 + (rnd() - 0.5) * 0.3);
}

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
  const rndPitch = mulberry32(777);
  const expectedSpacingPx = makeExpectedSpacingPx(rndPitch);
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
    const indexed = indexFamilyLines(family, buildOtherVp(rnd), W, H, 4, expectedSpacingPx);
    if (indexed.length < lines.length - 3) continue; // too many dropped — not a "correct" trial either way
    const pts = indexed.map(x => ({ rec: x.index, truth: trueIdx.get(x.line)! })).sort((a, b) => a.truth - b.truth);
    const first = pts[0], last = pts[pts.length - 1];
    const m = (last.rec - first.rec) / (last.truth - first.truth);
    if (Math.abs(Math.abs(m) - 1) < 0.05) correct++;
  }
  check('gap-free family, 1.5px noise, 100 trials', correct >= 90, `${correct}/${TRIALS} correctly recovered step=1 (no aliasing)`);
}

// --- Scenario 2: genuine gaps PLUS noise PLUS the spacing hint together ---
// Makes sure the scale-check fix doesn't over-correct and start rejecting
// real, necessary multi-step gaps — the hint constrains the PER-INDEX
// spacing, not the per-DETECTED-LINE spacing, so a correctly gap-aware model
// should still measure ~1x expectedSpacingPx even where real gaps exist.
{
  const rnd = mulberry32(42);
  const expectedSpacingPx = makeExpectedSpacingPx(mulberry32(999));
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
  const indexed = indexFamilyLines(family, buildOtherVp(rnd), W, H, 4, expectedSpacingPx);
  const pts = indexed.map(x => ({ rec: x.index, truth: trueIdx.get(x.line)! })).sort((a, b) => a.truth - b.truth);
  const first = pts[0], last = pts[pts.length - 1];
  const m = pts.length >= 2 ? (last.rec - first.rec) / (last.truth - first.truth) : NaN;
  let maxAffineErr = 0;
  const k = first.rec - m * first.truth;
  for (const p of pts) maxAffineErr = Math.max(maxAffineErr, Math.abs((m * p.truth + k) - p.rec));
  const ok = indexed.length >= lines.length - 2 && Math.abs(Math.abs(m) - 1) < 0.05 && maxAffineErr < 1.01;
  check('genuine gaps + noise + spacing hint', ok, `recovered ${indexed.length}/${lines.length}, m=${m.toFixed(3)}, maxAffineErr=${maxAffineErr.toFixed(2)}`);
}

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
