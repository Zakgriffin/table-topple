// Validates src/lines.ts's Level-1 gradient-oriented Hough line detector:
// given a synthetic image containing straight edges at KNOWN (theta, rho),
// does buildLineAccumulator + findLinePeaks recover them?
//
// Three scenarios, increasing complexity:
//   1. a single straight edge at a known angle/offset
//   2. a set of evenly-spaced parallel stripes (tests peak SEPARATION along rho)
//   3. a full rotated grid (both line families present at once, tests that
//      unrelated directions don't smear into each other's peaks)
//
// Usage: node scripts/test-hough-lines.ts

import { buildLineAccumulator, findLinePeaks } from '../src/lines.ts';

const SIZE = 200;
const CENTER = SIZE / 2;

// Same discipline as scripts/test-corner-axes.ts: this renderer draws hard
// 0/255 edges with no anti-aliasing, which produces staircase aliasing at
// angles not aligned to the pixel grid — spurious extra edge orientations
// that a real (optically-blurred) camera image would never show. A pre-blur
// simulates that missing anti-aliasing; without it, confirmed via
// scripts/_debug-hough.ts (deleted) that a 15deg grid produces 30+ spurious
// off-family peaks that all disappear once blurred.
function preBlur(gray: Float64Array, radius: number): Float64Array {
  const out = new Float64Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      let sum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= SIZE) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= SIZE) continue;
          sum += gray[yy * SIZE + xx]; count++;
        }
      }
      out[y * SIZE + x] = sum / count;
    }
  }
  return out;
}

function angleDistModPi(a: number, b: number): number {
  let d = Math.abs(a - b) % Math.PI;
  if (d > Math.PI / 2) d = Math.PI - d;
  return d;
}

// A line with normal angle theta and offset rho (relative to image center)
// is the set of points {p : (p - center) . (cos theta, sin theta) = rho}.
function renderEdge(gray: Float64Array, theta: number, rho: number) {
  const c = Math.cos(theta), s = Math.sin(theta);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const dx = x - CENTER, dy = y - CENTER;
      const d = dx * c + dy * s - rho;
      gray[y * SIZE + x] = d < 0 ? 0 : 255;
    }
  }
}

function renderStripes(gray: Float64Array, theta: number, pitch: number, phase: number) {
  const c = Math.cos(theta), s = Math.sin(theta);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const dx = x - CENTER, dy = y - CENTER;
      const d = dx * c + dy * s - phase;
      const cell = Math.floor(d / pitch);
      gray[y * SIZE + x] = (cell & 1) === 0 ? 255 : 0;
    }
  }
}

function renderGrid(gray: Float64Array, theta: number, pitch: number) {
  const c = Math.cos(theta), s = Math.sin(theta);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const dx = x - CENTER, dy = y - CENTER;
      const rx = dx * c + dy * s;
      const ry = -dx * s + dy * c;
      const cx = Math.floor(rx / pitch), cy = Math.floor(ry / pitch);
      gray[y * SIZE + x] = ((cx + cy) & 1) === 0 ? 255 : 0;
    }
  }
}

let pass = 0, fail = 0;
function check(name: string, ok: boolean, detail: string) {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${detail}`);
  if (ok) pass++; else fail++;
}

// --- Scenario 1: single edge ---
for (const thetaDeg of [0, 30, 45, 60, 90, 120, 150]) {
  for (const rho of [-40, 0, 35]) {
    const theta = thetaDeg * Math.PI / 180;
    const gray = new Float64Array(SIZE * SIZE);
    renderEdge(gray, theta, rho);
    const field = buildLineAccumulator(gray, SIZE, SIZE);
    const peaks = findLinePeaks(field);
    if (peaks.length === 0) { check(`edge theta=${thetaDeg} rho=${rho}`, false, 'no peaks found'); continue; }
    const best = peaks.reduce((a, b) => a.weight > b.weight ? a : b);
    const thetaErrDeg = angleDistModPi(best.theta, theta) * 180 / Math.PI;
    const rhoErr = Math.abs(best.rho - rho);
    const ok = thetaErrDeg < 3 && rhoErr < 3;
    check(`edge theta=${String(thetaDeg).padStart(3)} rho=${String(rho).padStart(3)}`, ok,
      `got theta=${(best.theta * 180 / Math.PI).toFixed(1)} rho=${best.rho.toFixed(1)} (thetaErr=${thetaErrDeg.toFixed(2)}deg rhoErr=${rhoErr.toFixed(2)}px)`);
  }
}

// --- Scenario 2: parallel stripes (peak separation along rho) ---
for (const thetaDeg of [0, 25, 90]) {
  for (const pitch of [16, 24]) {
    const theta = thetaDeg * Math.PI / 180;
    const raw = new Float64Array(SIZE * SIZE);
    renderStripes(raw, theta, pitch, 0);
    const gray = preBlur(raw, 2);
    const field = buildLineAccumulator(gray, SIZE, SIZE, 180, 2, 1, 4);
    const peaks = findLinePeaks(field, 0.3);
    // Every peak should be near theta (mod PI). Band parity alternates every
    // `pitch` units of d, so there's exactly one boundary edge per pitch
    // (not per pitch/2 — a boundary is a boundary regardless of which side
    // is light vs dark), giving ~SIZE/pitch boundaries for an axis-aligned
    // band (confirmed against the actual counts observed here).
    const badAngle = peaks.filter(p => angleDistModPi(p.theta, theta) * 180 / Math.PI > 5);
    const expectedCount = SIZE / pitch;
    const ok = badAngle.length === 0 && peaks.length >= expectedCount * 0.6;
    check(`stripes theta=${thetaDeg} pitch=${pitch}`, ok,
      `${peaks.length} peaks, ${badAngle.length} off-angle (expected roughly >=${Math.round(expectedCount * 0.6)} peaks near theta=${thetaDeg}deg)`);
  }
}

// --- Scenario 3: rotated grid (two orthogonal families present at once) ---
for (const thetaDeg of [0, 15, 40, 65]) {
  const theta = thetaDeg * Math.PI / 180;
  const pitch = 20;
  const raw = new Float64Array(SIZE * SIZE);
  renderGrid(raw, theta, pitch);
  const gray = preBlur(raw, 2);
  const field = buildLineAccumulator(gray, SIZE, SIZE, 180, 2, 1, 4);
  const peaks = findLinePeaks(field, 0.3);
  const famA = theta, famB = theta + Math.PI / 2;
  const misfit = peaks.filter(p => Math.min(angleDistModPi(p.theta, famA), angleDistModPi(p.theta, famB)) * 180 / Math.PI > 5);
  const ok = peaks.length >= 6 && misfit.length === 0;
  check(`grid theta=${thetaDeg}`, ok, `${peaks.length} peaks, ${misfit.length} not matching either family (theta=${thetaDeg} or +90)`);
}

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
