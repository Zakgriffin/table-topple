// Validates src/vp.ts's estimateVanishingPoint: given a set of lines
// (theta, rho, weight) constructed to pass through a KNOWN point (or share a
// KNOWN direction, for the at-infinity case), does the homogeneous
// least-squares solve recover it? This is a pure math test of the
// SVD/Jacobi-eigenvector step in isolation from Level 1's pixel-level
// detection (already validated in scripts/test-hough-lines.ts) — lines are
// constructed directly here, not rendered from an image.
//
// Usage: node scripts/test-vp.ts

import type { LineCandidate } from '../src/lines.ts';
import { estimateVanishingPoint, vpIsFinite, vpToPoint } from '../src/vp.ts';

const W = 640, H = 480;
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

// Builds a line through point (px,py) at normal angle theta, in the
// (theta, rho-relative-to-center) representation Level 1 produces.
function lineThrough(px: number, py: number, theta: number, weight = 1): LineCandidate {
  const cx = W / 2, cy = H / 2;
  const a = Math.cos(theta), b = Math.sin(theta);
  const rho = a * (px - cx) + b * (py - cy);
  return { theta, rho, weight };
}

// --- Scenario 1: finite VP, exact lines (no noise) ---
{
  const trueX = 900, trueY = 200; // outside the image — a realistic vanishing point location
  const lines: LineCandidate[] = [];
  const rnd = mulberry32(1);
  for (let i = 0; i < 8; i++) lines.push(lineThrough(trueX, trueY, rnd() * Math.PI));
  const vp = estimateVanishingPoint(lines, W, H);
  const ok1 = vpIsFinite(vp);
  const p = ok1 ? vpToPoint(vp) : { x: NaN, y: NaN };
  const err = Math.hypot(p.x - trueX, p.y - trueY);
  const ok = ok1 && err < 1;
  check('finite exact', ok, `finite=${ok1} recovered=(${p.x.toFixed(1)},${p.y.toFixed(1)}) err=${err.toFixed(3)}px`);
}

// --- Scenario 2: finite VP, with per-line angular noise (simulates imperfect Level-1 peaks) ---
{
  const trueX = -300, trueY = 600;
  const rnd = mulberry32(2);
  const lines: LineCandidate[] = [];
  for (let i = 0; i < 12; i++) {
    const theta = rnd() * Math.PI;
    const line = lineThrough(trueX, trueY, theta);
    // Perturb rho by a small amount (simulates sub-bin quantization noise),
    // proportional to nothing in particular — a fixed few-pixel jitter.
    line.rho += (rnd() - 0.5) * 3;
    lines.push(line);
  }
  const vp = estimateVanishingPoint(lines, W, H);
  const ok1 = vpIsFinite(vp);
  const p = ok1 ? vpToPoint(vp) : { x: NaN, y: NaN };
  const err = Math.hypot(p.x - trueX, p.y - trueY);
  const ok = ok1 && err < 25;
  check('finite noisy', ok, `finite=${ok1} recovered=(${p.x.toFixed(1)},${p.y.toFixed(1)}) err=${err.toFixed(2)}px`);
}

// --- Scenario 3: at-infinity VP (truly parallel lines), exact ---
{
  const trueDirDeg = 35;
  const trueDir = trueDirDeg * Math.PI / 180;
  // Parallel lines all sharing direction trueDir: their NORMAL is
  // trueDir + PI/2 (constant), only rho differs between them.
  const theta = (trueDir + Math.PI / 2) % Math.PI;
  const rnd = mulberry32(3);
  const lines: LineCandidate[] = [];
  for (let i = 0; i < 8; i++) lines.push({ theta, rho: (rnd() - 0.5) * 400, weight: 1 });
  const vp = estimateVanishingPoint(lines, W, H);
  const ok1 = !vpIsFinite(vp, 1e-3);
  const dirLen = Math.hypot(vp.x, vp.y);
  const gotDeg = (Math.atan2(vp.y, vp.x) * 180 / Math.PI + 360) % 180;
  let dirErr = Math.abs(gotDeg - trueDirDeg) % 180;
  if (dirErr > 90) dirErr = 180 - dirErr;
  const ok = ok1 && dirErr < 1;
  check('at-infinity exact', ok, `finite=${!ok1} |w|=${Math.abs(vp.w).toExponential(2)} dir=${gotDeg.toFixed(2)}deg (true=${trueDirDeg}deg) dirErr=${dirErr.toFixed(3)}deg`);
}

// --- Scenario 4: near-infinity VP (very distant but technically finite — the realistic low-tilt case) ---
{
  const trueDirDeg = 70;
  const trueDir = trueDirDeg * Math.PI / 180;
  const dist = 200000; // pixels — absurdly far, simulating near-zero tilt
  const trueX = W / 2 + dist * Math.cos(trueDir), trueY = H / 2 + dist * Math.sin(trueDir);
  const rnd = mulberry32(4);
  const lines: LineCandidate[] = [];
  for (let i = 0; i < 10; i++) lines.push(lineThrough(trueX, trueY, rnd() * Math.PI));
  const vp = estimateVanishingPoint(lines, W, H);
  const gotDeg = (Math.atan2(vp.y, vp.x) * 180 / Math.PI + 360) % 180;
  let dirErr = Math.abs(gotDeg - trueDirDeg) % 180;
  if (dirErr > 90) dirErr = 180 - dirErr;
  const ok = dirErr < 1;
  check('near-infinity', ok, `finite=${vpIsFinite(vp)} dir=${gotDeg.toFixed(2)}deg (true=${trueDirDeg}deg) dirErr=${dirErr.toFixed(3)}deg`);
}

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
