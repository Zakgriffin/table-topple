import { activeCamera } from '../camera/store.ts';
import { Camera } from '../camera/model.ts';
import { createLsdChainResidency, LsdRectangle, LsdSettings, runLsdChain } from '../pipeline/lsdSegments.ts';
import { globalState } from '../state.ts';

// ── Dev harness: does the LSD chain give the same answer at every toggle? ──
//
// Not part of any pipeline. Call it from the devtools console on a real
// capture -- main.ts re-exports every module onto globalThis:
//
//   await verifyLsdChain()
//
// This is the test that matters for pipelineGPU/fieldResidency.ts, and it
// exists because the per-stage verify harnesses CANNOT catch what it catches.
// growRegionsVerify/collectRegionsVerify/lsdFitVerify each compare one stage's
// CPU output against its GPU output through a path that asks the residency for
// the CPU side of everything -- by construction they never exercise two GPU
// stages handing a buffer to each other, which is precisely where a residency
// bug lives. A stage reading the wrong side, a buffer destroyed while a
// neighbour still holds it, a failure path that publishes half its output and
// then trips the single-assignment invariant: all of that is invisible to the
// per-stage harnesses and immediately visible here.
//
// So this runs the REAL entry point (createLsdChainResidency + runLsdChain, the
// same two calls pipeline/poseCompute.ts makes) once per legal toggle
// configuration and diffs every configuration against the all-CPU one.
//
// KNOWN AND EXPECTED, measured 2026-08-03 at 512x384 so you don't re-chase it:
// the three configurations with gradient=GPU AND fit=GPU report a
// maxNfaLog10Delta of ~1.4 while every other configuration reports 0 or ~1.7e-5.
// It is EXACTLY ONE rectangle out of 2133 (index 684, a 17-member region,
// 8.6x1.96px), and its geometry, member list and accept decision are all
// identical -- only the NFA score moves, -5.92 -> -7.30.
//
// The cause is that fx/fy computed in f32 on the device differ in the last bit
// from the same values computed in f64 and then rounded to f32, which is enough
// to flip one boundary pixel's alignment test inside the shader's count. On a
// 17-member region a single pixel is ~6% of the evidence, so k changing by 1
// moves log10(NFA) by over a decade -- the size of the number says nothing
// about the size of the cause. The CPU fitter absorbs the same perturbation
// because it counts in f64.
//
// There is no principled epsilon fix, for the same reason growRegionsVerify's
// borderlinePairs has none: the test is a threshold on a continuous quantity
// and something has to land nearest it. Unlike the grower's case this one is
// BOUNDED -- it perturbs one region's own n/k and cannot cascade -- and it
// changed no accept decision. Watch it, don't fix it; if acceptFlips ever
// becomes nonzero that is a different and real problem.
//
// What "agrees" means, and why the numbers are deltas rather than assertions:
// the grower is the one stage whose GPU output is not bit-identical to its CPU
// output (f32-vs-f64 on the edge predicate -- see growRegions.ts's header and
// growRegionsVerify's borderlinePairs), so a capture with neighbour pairs
// sitting on the tolerance boundary can legitimately split or merge a component
// and shift every count downstream. Judge accordingly: with borderlinePairs 0,
// anything nonzero below is a bug; with borderlinePairs nonzero, expect small
// deltas confined to configurations that differ in useGPUGrowRegions, and treat
// a delta between two configurations that share a grower setting as a bug
// regardless.

// The toggles this sweeps, in the order they appear in a config label.
interface ChainToggles {
  gradient: boolean;
  grow: boolean;
  collect: boolean;
  fit: boolean;
}

export interface ChainConfigReport {
  config: string; // e.g. "gradient=GPU grow=GPU collect=CPU fit=GPU"
  n: number; // rectangles returned
  accepted: number;
  members: number; // total rawMembers across all rectangles
  // Deltas against the all-CPU baseline. Every one of these should be 0 unless
  // the grower legitimately diverged -- see the header.
  dN: number;
  dAccepted: number;
  dMembers: number;
  // Largest geometric disagreement over rectangles that pair up by index.
  // Nonzero-but-tiny is expected wherever a GPU stage did the arithmetic in
  // f32; the 1e-6 class is what lsdFitVerify already measured.
  maxCenterDelta: number; // px
  maxLengthDelta: number; // px
  maxNfaLog10Delta: number;
  // What this configuration actually cost the bus, straight off the
  // residency's own ledger. This is the readout the toggle UI is meant to grow
  // (see the GPU parallelization plan's phase 3) -- it is free here because
  // the residency records a transfer at the point it decides to make one.
  crossings: number;
  bytes: number;
  medianMs: number;
  error: string | null; // a throw is itself a result worth reporting, not a reason to abandon the sweep
}

export interface LsdChainVerifyReport {
  reps: number;
  baseline: string;
  configs: ChainConfigReport[];
  worstDN: number;
  worstDAccepted: number;
  worstDMembers: number;
  errors: number;
  focusWaitMs: number; // how long the sweep sat waiting for the page -- see awaitPageFocus
  focusedThroughout: boolean; // false invalidates every medianMs below, nothing else
}

// Blocks until the page is actually focused and visible, then lets it settle.
//
// This is not politeness, it is a correctness gate on the timing column. A
// backgrounded or unfocused tab throttles rAF and timers, so a measurement
// taken across a focus transition is WRONG rather than merely noisy -- a
// full-image pass in this project once measured 1.08ms unfocused against
// 0.19ms focused, and the bogus number briefly made a GPU port look
// unnecessary. Anything driven from the dev bridge is especially exposed,
// because accepting the command in another window is itself a focus change.
export async function awaitPageFocus(timeoutMs = 120000): Promise<number> {
  const t0 = performance.now();
  while (!document.hasFocus() || document.visibilityState !== 'visible') {
    await new Promise((r) => setTimeout(r, 100));
    if (performance.now() - t0 > timeoutMs) throw new Error('awaitPageFocus: page never regained focus');
  }
  await new Promise((r) => setTimeout(r, 600));
  return Math.round(performance.now() - t0);
}

function label(t: ChainToggles): string {
  const s = (b: boolean) => (b ? 'GPU' : 'CPU');
  return `gradient=${s(t.gradient)} grow=${s(t.grow)} collect=${s(t.collect)} fit=${s(t.fit)}`;
}

// useGPUCollectRegions only means anything with useGPUGrowRegions on -- stage 3b
// is dispatched from inside growRegionsCCLGPU, so with a CPU grower the flag is
// dead and sweeping it would just run the same configuration twice.
function legalConfigs(): ChainToggles[] {
  const out: ChainToggles[] = [];
  for (const gradient of [false, true]) {
    for (const grow of [false, true]) {
      for (const collect of grow ? [false, true] : [false]) {
        for (const fit of [false, true]) out.push({ gradient, grow, collect, fit });
      }
    }
  }
  return out;
}

function summarize(rects: LsdRectangle[]): { n: number; accepted: number; members: number } {
  let accepted = 0, members = 0;
  for (const r of rects) {
    if (r.accepted) accepted++;
    members += r.rawMembers.length;
  }
  return { n: rects.length, accepted, members };
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s.length === 0 ? 0 : s[Math.floor(s.length / 2)];
}

export async function verifyLsdChain(camera?: Camera | null, reps = 3): Promise<LsdChainVerifyReport | string> {
  camera = camera ?? activeCamera() ?? null;
  if (!camera) return 'no active camera';
  const gray = camera.lastNoisedPreviewGray;
  if (!gray) return 'no capture yet -- run a capture first';
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const s = camera.settings;

  const settings: LsdSettings = {
    toleranceDeg: s.lsdToleranceDeg,
    rhoNoiseThreshold: s.lsdRhoNoiseThreshold,
    rhoHighThreshold: s.lsdRhoHighThreshold,
    cclSteps: s.lsdCclSteps,
    minRegionSize: s.lsdMinRegionSize,
    nfaEpsilon: s.lsdNfaEpsilon,
    nfaTestExponent: s.lsdNfaTestExponent,
    maxRetries: s.lsdMaxRetries,
    retryToleranceFactor: s.lsdRetryToleranceFactor,
    retryShrinkFraction: s.lsdRetryShrinkFraction,
  };

  const saved = {
    gradient: globalState.useGPUGradient,
    grow: globalState.useGPUGrowRegions,
    collect: globalState.useGPUCollectRegions,
    fit: globalState.useGPULsdFit,
  };

  // One run of the real chain under one toggle configuration, returning the
  // rectangles AND the residency's ledger. The residency is created and
  // destroyed inside, exactly as production does it, so a leak or a
  // double-destroy shows up here too.
  const runOnce = async (t: ChainToggles) => {
    globalState.useGPUGradient = t.gradient;
    globalState.useGPUGrowRegions = t.grow;
    globalState.useGPUCollectRegions = t.collect;
    globalState.useGPULsdFit = t.fit;
    const res = await createLsdChainResidency(gray, w, h);
    try {
      const t0 = performance.now();
      const rects = await runLsdChain(res, w, h, settings);
      const ms = performance.now() - t0;
      return { rects, ms, summary: res.summary() };
    } finally {
      res.destroy();
    }
  };

  try {
    const focusWaitMs = await awaitPageFocus();
    const configs = legalConfigs();
    const baseline = configs[0]; // all-CPU, and the source of truth every stage is verified against
    const baseRects = (await runOnce(baseline)).rects;
    const base = summarize(baseRects);

    // Interleaved (config-inner, rep-outer) rather than reps-inner, so a JIT
    // tier change or thermal drift shows up as spread across every
    // configuration instead of as a fake win for whichever one ran last.
    const times = new Map<string, number[]>();
    const results = new Map<string, { rects: LsdRectangle[]; crossings: number; bytes: number } | string>();
    for (let rep = 0; rep < reps; rep++) {
      for (const t of configs) {
        const key = label(t);
        try {
          const { rects, ms, summary } = await runOnce(t);
          times.set(key, [...(times.get(key) ?? []), ms]);
          results.set(key, { rects, crossings: summary.crossings, bytes: summary.bytes });
        } catch (e) {
          // Keep sweeping: which OTHER configurations also fail is the most
          // useful thing to know about a residency bug.
          results.set(key, e instanceof Error ? e.message : String(e));
        }
      }
    }

    const reports: ChainConfigReport[] = [];
    for (const t of configs) {
      const key = label(t);
      const got = results.get(key);
      if (typeof got === 'string' || got === undefined) {
        reports.push({
          config: key, n: 0, accepted: 0, members: 0, dN: 0, dAccepted: 0, dMembers: 0,
          maxCenterDelta: 0, maxLengthDelta: 0, maxNfaLog10Delta: 0,
          crossings: 0, bytes: 0, medianMs: 0, error: got ?? 'never ran',
        });
        continue;
      }
      const sum = summarize(got.rects);
      let maxCenterDelta = 0, maxLengthDelta = 0, maxNfaLog10Delta = 0;
      const pairs = Math.min(got.rects.length, baseRects.length);
      for (let i = 0; i < pairs; i++) {
        const a = got.rects[i], b = baseRects[i];
        maxCenterDelta = Math.max(maxCenterDelta, Math.hypot(a.cx - b.cx, a.cy - b.cy));
        maxLengthDelta = Math.max(maxLengthDelta, Math.abs(a.length - b.length));
        maxNfaLog10Delta = Math.max(maxNfaLog10Delta, Math.abs(a.nfaLog10 - b.nfaLog10));
      }
      reports.push({
        config: key, ...sum,
        dN: sum.n - base.n, dAccepted: sum.accepted - base.accepted, dMembers: sum.members - base.members,
        maxCenterDelta, maxLengthDelta, maxNfaLog10Delta,
        crossings: got.crossings, bytes: got.bytes,
        medianMs: median(times.get(key) ?? []), error: null,
      });
    }

    return {
      reps,
      baseline: label(baseline),
      configs: reports,
      worstDN: Math.max(...reports.map((r) => Math.abs(r.dN))),
      worstDAccepted: Math.max(...reports.map((r) => Math.abs(r.dAccepted))),
      worstDMembers: Math.max(...reports.map((r) => Math.abs(r.dMembers))),
      errors: reports.filter((r) => r.error !== null).length,
      focusWaitMs,
      // Checked at the END, not just the start: focus can be lost mid-sweep,
      // and if it was, every medianMs above is garbage even though every
      // correctness number remains valid (they are deterministic).
      focusedThroughout: document.hasFocus() && document.visibilityState === 'visible',
    };
  } finally {
    // Restoring these matters more than it looks: they are the live production
    // toggles, and leaving the sweep's last configuration behind would silently
    // reconfigure the app for every frame after the harness returns.
    globalState.useGPUGradient = saved.gradient;
    globalState.useGPUGrowRegions = saved.grow;
    globalState.useGPUCollectRegions = saved.collect;
    globalState.useGPULsdFit = saved.fit;
  }
}
