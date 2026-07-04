// Validates splitIntoTwoFamilies' extraLines rescue mechanism: when one
// family is systematically weak (most of its real members never clear Level
// 1's single global vote threshold, confirmed via live-device testing to
// happen for real cameras even when a controlled synthetic capture at any
// roll angle shows no such asymmetry -- pointing at a real-capture artifact
// like lighting/focus/sensor sharpening, not a detection bug), a SECOND,
// lower-threshold peak pool (extraLines) can rescue that family's missing
// members by checking them against the VP the STRONG pool already
// established, without ever letting the weak/noisy pool influence which VP
// gets proposed in the first place.
//
// Usage: node scripts/test-vp-split-rescue.ts

import type { LineCandidate } from '../src/lines.ts';
import { estimateVanishingPoint, splitIntoTwoFamilies, vpToPoint } from '../src/vp.ts';

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

function pencilThroughFinite(px: number, py: number, count: number, rnd: () => number): LineCandidate[] {
  const cx = W / 2, cy = H / 2;
  const out: LineCandidate[] = [];
  for (let i = 0; i < count; i++) {
    const theta = rnd() * Math.PI;
    const a = Math.cos(theta), b = Math.sin(theta);
    let rho = a * (px - cx) + b * (py - cy);
    rho += (rnd() - 0.5) * 2;
    out.push({ theta, rho, weight: 1 + rnd() });
  }
  return out;
}

// --- Scenario: family A strong (10 members all clear Level 1), family B
// weak -- only 3 of its 10 true members clear the (simulated) main
// threshold, the other 7 only clear a much lower one (the extraLines pool).
{
  const rnd = mulberry32(20);
  const trueA = { x: 900, y: 150 };
  const trueB = { x: -500, y: 900 };
  const allA = pencilThroughFinite(trueA.x, trueA.y, 10, rnd);
  const allB = pencilThroughFinite(trueB.x, trueB.y, 10, rnd);
  const strongB = allB.slice(0, 3), weakB = allB.slice(3);

  const mainPool = [...allA, ...strongB];
  const extraPool = weakB;

  function evalSplit(withRescue: boolean) {
    const { familyA, familyB } = withRescue
      ? splitIntoTwoFamilies(mainPool, W, H, 6, 60, extraPool)
      : splitIntoTwoFamilies(mainPool, W, H);
    // Whichever recovered family is closer to trueB's direction is "the weak one" here.
    const distToB = (vp: ReturnType<typeof estimateVanishingPoint>) => {
      const p = vpToPoint(vp);
      return Math.hypot(p.x - trueB.x, p.y - trueB.y);
    };
    const weakFam = distToB(familyA.vp) < distToB(familyB.vp) ? familyA : familyB;
    const vpErr = distToB(weakFam.vp);
    return { size: weakFam.lines.length, vpErr };
  }

  const without = evalSplit(false);
  check(
    'baseline (no rescue): weak family stays sparse',
    without.size <= 4,
    `weak family recovered ${without.size}/10 true members (only 3 were in the main pool)`
  );

  const withRescue = evalSplit(true);
  check(
    'with rescue: weak family recovers most of its true members',
    withRescue.size >= 9,
    `weak family recovered ${withRescue.size}/10 true members, vpErr=${withRescue.vpErr.toFixed(1)}px`
  );

  check(
    'rescue improves (or at least does not worsen) the weak family VP estimate',
    withRescue.vpErr <= without.vpErr + 5, // small slack for noise; the point is it shouldn't get WORSE
    `without=${without.vpErr.toFixed(1)}px with=${withRescue.vpErr.toFixed(1)}px`
  );
}

// --- Scenario: extraLines that don't belong to EITHER family should be
// silently dropped, not injected into a family or counted as "unassigned"
// noise that downstream code has to separately filter.
{
  const rnd = mulberry32(21);
  const trueA = { x: 900, y: 150 };
  const trueB = { x: -500, y: 900 };
  const mainPool = [...pencilThroughFinite(trueA.x, trueA.y, 10, rnd), ...pencilThroughFinite(trueB.x, trueB.y, 10, rnd)];
  // Junk lines with essentially random theta/rho -- shouldn't match either VP.
  const junk: LineCandidate[] = Array.from({ length: 5 }, () => ({ theta: rnd() * Math.PI, rho: (rnd() - 0.5) * 2000, weight: 1 }));
  const { familyA, familyB } = splitIntoTwoFamilies(mainPool, W, H, 6, 60, junk);
  const totalRecovered = familyA.lines.length + familyB.lines.length;
  check(
    'junk extraLines matching neither VP are dropped, not force-assigned',
    totalRecovered <= 22, // 20 true + at most a couple junk lines coincidentally close to a VP by chance
    `familyA=${familyA.lines.length} familyB=${familyB.lines.length} (20 true lines + 5 junk offered)`
  );
}

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
