// Validates src/vp.ts's splitIntoTwoFamilies: given a MIXED, shuffled set of
// lines from two different pencils (each pencil = lines through one shared
// vanishing point, exactly what a real family of converging grid lines looks
// like under perspective — NOT a set of parallel lines, which is why this
// couldn't be tested by clustering on angle alone), does it recover the
// correct partition and both VPs without being told the split up front?
//
// Usage: node scripts/test-vp-split.ts

import type { LineCandidate } from '../src/lines.ts';
import { estimateVanishingPoint, splitIntoTwoFamilies, vpIsFinite, vpToPoint } from '../src/vp.ts';

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

function pencilThroughFinite(px: number, py: number, count: number, rnd: () => number, tag: string): (LineCandidate & { tag: string })[] {
  const cx = W / 2, cy = H / 2;
  const out: (LineCandidate & { tag: string })[] = [];
  for (let i = 0; i < count; i++) {
    const theta = rnd() * Math.PI;
    const a = Math.cos(theta), b = Math.sin(theta);
    let rho = a * (px - cx) + b * (py - cy);
    rho += (rnd() - 0.5) * 2; // small realistic jitter, like sub-bin Hough noise
    out.push({ theta, rho, weight: 1 + rnd(), tag });
  }
  return out;
}

function pencilParallel(dirDeg: number, count: number, rnd: () => number, tag: string): (LineCandidate & { tag: string })[] {
  const theta = ((dirDeg + 90) * Math.PI / 180) % Math.PI;
  const out: (LineCandidate & { tag: string })[] = [];
  for (let i = 0; i < count; i++) {
    out.push({ theta, rho: (rnd() - 0.5) * 300, weight: 1 + rnd(), tag });
  }
  return out;
}

function shuffle<T>(arr: T[], rnd: () => number): T[] {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function evaluate(
  name: string,
  linesTagged: (LineCandidate & { tag: string })[],
  trueA: { finite: true; x: number; y: number } | { finite: false; dirDeg: number },
  trueB: { finite: true; x: number; y: number } | { finite: false; dirDeg: number },
  rnd: () => number,
) {
  const shuffled = shuffle(linesTagged, rnd);
  const lines: LineCandidate[] = shuffled.map(({ theta, rho, weight }) => ({ theta, rho, weight }));
  const { familyA, familyB, unassigned } = splitIntoTwoFamilies(lines, W, H);

  // Figure out which recovered family corresponds to which tag by majority
  // vote of the ORIGINAL tags among its member lines (recover tag via index
  // matching against the shuffled array's original tag).
  const tagOf = new Map<LineCandidate, string>();
  shuffled.forEach((l, i) => tagOf.set(lines[i], l.tag));
  function majorityTag(fam: LineCandidate[]): string {
    const counts: Record<string, number> = {};
    for (const l of fam) { const t = tagOf.get(l)!; counts[t] = (counts[t] || 0) + 1; }
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? '?';
  }
  const tagA = majorityTag(familyA.lines), tagB = majorityTag(familyB.lines);
  const purityA = familyA.lines.filter(l => tagOf.get(l) === tagA).length / Math.max(1, familyA.lines.length);
  const purityB = familyB.lines.filter(l => tagOf.get(l) === tagB).length / Math.max(1, familyB.lines.length);
  const noCrossTalk = tagA !== tagB && purityA === 1 && purityB === 1 && unassigned.length === 0;

  const [recA, trueForA] = tagA === 'A' ? [familyA, trueA] : [familyB, trueA];
  const [recB, trueForB] = tagA === 'A' ? [familyB, trueB] : [familyA, trueB];

  function vpError(vp: ReturnType<typeof estimateVanishingPoint>, truth: typeof trueA): { ok: boolean; detail: string } {
    if (truth.finite) {
      if (!vpIsFinite(vp)) return { ok: false, detail: 'expected finite, got infinite' };
      const p = vpToPoint(vp);
      const err = Math.hypot(p.x - truth.x, p.y - truth.y);
      return { ok: err < 15, detail: `err=${err.toFixed(1)}px` };
    } else {
      const gotDeg = (Math.atan2(vp.y, vp.x) * 180 / Math.PI + 360) % 180;
      let dirErr = Math.abs(gotDeg - truth.dirDeg) % 180;
      if (dirErr > 90) dirErr = 180 - dirErr;
      return { ok: dirErr < 2, detail: `dirErr=${dirErr.toFixed(2)}deg` };
    }
  }
  const eA = vpError(recA.vp, trueForA);
  const eB = vpError(recB.vp, trueForB);
  const ok = noCrossTalk && eA.ok && eB.ok;
  check(name, ok,
    `split=${familyA.lines.length}/${familyB.lines.length}/${unassigned.length} purity=${purityA.toFixed(2)}/${purityB.toFixed(2)} vpA:${eA.detail} vpB:${eB.detail}`);
}

// --- Scenario 1: two well-separated finite VPs (typical moderate tilt) ---
{
  const rnd = mulberry32(10);
  const lines = [...pencilThroughFinite(900, 150, 9, rnd, 'A'), ...pencilThroughFinite(-500, 900, 9, rnd, 'B')];
  evaluate('two finite VPs', lines, { finite: true, x: 900, y: 150 }, { finite: true, x: -500, y: 900 }, rnd);
}

// --- Scenario 2: one finite, one at-infinity (typical: rows converge, cols near-vertical) ---
{
  const rnd = mulberry32(11);
  const lines = [...pencilThroughFinite(1200, 400, 10, rnd, 'A'), ...pencilParallel(5, 10, rnd, 'B')];
  evaluate('finite + at-infinity', lines, { finite: true, x: 1200, y: 400 }, { finite: false, dirDeg: 5 }, rnd);
}

// --- Scenario 3: both at-infinity (near-fronto-parallel view, low tilt both axes) ---
{
  const rnd = mulberry32(12);
  const lines = [...pencilParallel(2, 9, rnd, 'A'), ...pencilParallel(91, 9, rnd, 'B')];
  evaluate('both at-infinity', lines, { finite: false, dirDeg: 2 }, { finite: false, dirDeg: 91 }, rnd);
}

// --- Scenario 4: stress case — two finite VPs that are close together in direction ---
// (families whose pencils are less angularly distinct — expect this to be
// harder; included to find where the technique actually breaks, not just to
// confirm the easy cases.)
{
  const rnd = mulberry32(13);
  const lines = [...pencilThroughFinite(1500, 300, 10, rnd, 'A'), ...pencilThroughFinite(1500, 500, 10, rnd, 'B')];
  evaluate('close finite VPs (stress)', lines, { finite: true, x: 1500, y: 300 }, { finite: true, x: 1500, y: 500 }, rnd);
}

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
