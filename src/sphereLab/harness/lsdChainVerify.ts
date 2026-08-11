import { runLsdChainCPU, runLsdChainGPU } from '../../pose/stages/lsd/chain.ts';
import { type TransferSummary } from '../../pose/gpu/device.ts';
import type { LsdRectangle, LsdSettings } from '../../pose/stages/lsd/types.ts';
import { type Backend } from '../../pose/backend.ts';
import { type InputProvenance, provenance } from './input.ts';
import type { HarnessInput } from './input.ts';

// ── Dev harness: does the LSD chain give the same answer at every toggle? ──
//
// Not part of any pipeline. Call it from the devtools console on a real
// capture -- main.ts re-exports every module onto globalThis:
//
//   await verifyLsdChain(cameraInput())
//   await verifyLsdChain(await fixtureInput('default'))
//
// This is the test that matters for pose/gpu/fieldResidency.ts, and it
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
// same two calls pose/poseCompute.ts makes) in both configurations and diffs
// production against the CPU reference.
//
// ── It used to sweep 12 configurations, and that is not a loss (2026-08-05) ──
//
// Four per-stage GPU toggles gave 12 legal combinations, and what the sweep
// specifically tested was the RESIDENCY'S TRANSFER DECISIONS -- put stage 2 on
// the GPU and stage 3 on the CPU, and check the labeling still arrives where the
// CPU collect expects it. That is why its columns are crossings and bytes next
// to the geometry deltas.
//
// Perf TODO item 6 collapsed those toggles into globalState.forceCPU, and with
// them went the mixed configurations: fieldResidency stopped being a
// transfer-decision engine and became a plain named-slot arena. So 11 of the 12
// are now UNREACHABLE STATES rather than untested ones -- there is no longer a
// way to run the chain half on each side, and therefore no such plumbing to get
// wrong.
//
// What survives is the differential, and it matters MORE than it did: production
// no longer runs the CPU implementations at all, so this is the only thing
// standing between them and silent rot. If they drift, two things break at once
// -- the no-WebGPU fallback ships broken, and the reference every future port is
// verified against is wrong.
//
// STATE ITS BLIND SPOT rather than trusting a green run: a CPU-vs-GPU diff
// cannot see an error the two paths SHARE. It catches porting mistakes, not
// wrong shared formulas -- which is exactly how the level-line component order
// survived two sessions. This is a regression guard on the reference
// implementations, not a correctness proof of the pipeline.
//
// KNOWN AND EXPECTED, measured 2026-08-03 at 512x384 so you don't re-chase it:
// production reports a maxNfaLog10Delta of ~1.4 (back then, the configurations
// with gradient=GPU AND fit=GPU did; that is now simply "production").
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
// anything nonzero below is a bug; with borderlinePairs nonzero, small deltas
// are legitimate, since the two configurations differ in the grower by
// construction.

// The two configurations this compares. What used to be here was a
// ChainToggles record and a 12-configuration sweep over four per-stage GPU
// toggles; those toggles collapsed into globalState.forceCPU (2026-08-05, perf
// TODO item 6), and 11 of the 12 configurations became unreachable states
// rather than untested ones. See verifyLsdChain's header.
type ChainConfig = 'reference (forceCPU)' | 'production (GPU)';

interface ChainConfigReport {
  config: ChainConfig;
  n: number; // rectangles returned
  accepted: number;
  members: number; // total member pixels across all grown regions
  // Deltas against the CPU reference. Every one of these should be 0 unless the
  // grower legitimately diverged -- see the header.
  dN: number;
  dAccepted: number;
  dMembers: number;
  // Largest geometric disagreement over rectangles that pair up by index.
  // Nonzero-but-tiny is expected wherever a GPU stage did the arithmetic in
  // f32; the 1e-6 class is what lsdFitVerify already measured.
  maxCenterDelta: number; // px
  maxLengthDelta: number; // px
  maxNfaLog10Delta: number;
  // What this configuration actually cost the bus, off the chain's own ledger
  // -- free here, because a crossing is counted where it is made. This is now
  // the production configuration exactly: the sweep no longer asks for anything
  // the pose path does not, so the GPU row is what a real reconstruction pays.
  crossings: number;
  bytes: number;
  medianMs: number;
  error: string | null; // a throw is itself a result worth reporting, not a reason to abandon the sweep
}

// Every report in this directory carries where it came from. See
// harness/input.ts's InputProvenance: six of these recorded nothing at all
// about their input, which made their deltas exactly as un-re-derivable as the
// timing numbers the config-pinning work exists to replace.
interface LsdChainVerifyReport extends InputProvenance {
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


const CONFIGS: ChainConfig[] = ['reference (forceCPU)', 'production (GPU)'];

// `members` is passed in rather than summed off the rectangles, and the reason
// outlived the mechanism it was written about: rectangles never carried member
// lists on the GPU path, and now they carry them on neither. Summing off rects
// would report 0 for the production row and a real total for the reference one
// -- this harness's most alarming-looking column showing a large dMembers on
// exactly the configuration it exists to clear, a false positive manufactured
// by the instrument. It comes from each backend's own region set instead, where
// it is the same quantity on both sides.
function summarize(rects: LsdRectangle[], members: number): { n: number; accepted: number; members: number } {
  let accepted = 0;
  for (const r of rects) if (r.accepted) accepted++;
  return { n: rects.length, accepted, members };
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s.length === 0 ? 0 : s[Math.floor(s.length / 2)];
}

export async function verifyLsdChain(input: HarnessInput, reps = 3): Promise<LsdChainVerifyReport> {
  const { gray, w, h, settings: s } = input;

  const settings: LsdSettings = {
    toleranceDeg: s.lsdToleranceDeg,
    rhoNoiseThreshold: s.lsdRhoNoiseThreshold,
    rhoHighThreshold: s.lsdRhoHighThreshold,
    cclSteps: s.lsdCclSteps,
    minRegionSize: s.lsdMinRegionSize,
    nfaEpsilon: s.lsdNfaEpsilon,
    nfaTestExponent: s.lsdNfaTestExponent,
  };

  // One run of the real chain under one backend, returning the rectangles AND
  // the residency's ledger. The residency is created and destroyed inside,
  // exactly as production does it, so a leak or a double-destroy shows up here
  // too.
  //
  // The backend is an ARGUMENT now. This harness used to save
  // globalState.forceCPU, write it before each run, and restore it in a finally
  // -- which meant the differential could only be expressed by mutating the
  // running app's configuration, and a throw between the two runs left the app
  // on whatever the last run had set. The sweep is "same settings, two
  // backends", and that is now literally what the code says.
  const runOnce = async (t: ChainConfig) => {
    const backend: Backend = t === 'reference (forceCPU)' ? 'cpu' : 'gpu';
    const t0 = performance.now();
    const gpu = backend === 'gpu' ? await runLsdChainGPU(gray, w, h, settings) : null;
    const cpu = gpu ? null : runLsdChainCPU(gray, w, h, settings);
    const ms = performance.now() - t0;
    const rects = gpu ? gpu.rects : cpu!.rects;
    const summary: TransferSummary = gpu ? gpu.transfers : { crossings: 0, bytes: 0, entries: [] };
    // The member total costs this sweep NOTHING on either backend now. It used
    // to ask the chain for `wantMembers`, which pulled the whole region CSR
    // down -- four readbacks charged to the production row's crossings column,
    // so the configuration being measured was not quite the configuration that
    // ships. `memberCount` is the CSR's own total, already read as part of the
    // two-word counts readback the fit stage takes regardless.
    const members = gpu
      ? gpu.memberCount
      : cpu!.regions.reduce((sum, r) => sum + r.members.length, 0);
    return { rects, ms, summary, members };
  };

  const focusWaitMs = await awaitPageFocus();
  const configs = CONFIGS;
  const baseline = configs[0]; // the CPU reference, and the source of truth the GPU path is verified against
  const baseRun = await runOnce(baseline);
  const baseRects = baseRun.rects;
  const base = summarize(baseRects, baseRun.members);

  // Interleaved (config-inner, rep-outer) rather than reps-inner, so a JIT
  // tier change or thermal drift shows up as spread across every
  // configuration instead of as a fake win for whichever one ran last.
  const times = new Map<string, number[]>();
  const results = new Map<string, { rects: LsdRectangle[]; crossings: number; bytes: number; members: number } | string>();
  for (let rep = 0; rep < reps; rep++) {
    for (const t of configs) {
      const key = t;
      try {
        const { rects, ms, summary, members } = await runOnce(t);
        times.set(key, [...(times.get(key) ?? []), ms]);
        results.set(key, { rects, crossings: summary.crossings, bytes: summary.bytes, members });
      } catch (e) {
        // Keep sweeping: which OTHER configurations also fail is the most
        // useful thing to know about a residency bug.
        results.set(key, e instanceof Error ? e.message : String(e));
      }
    }
  }

  const reports: ChainConfigReport[] = [];
  for (const t of configs) {
    const key = t;
    const got = results.get(key);
    if (typeof got === 'string' || got === undefined) {
      reports.push({
        config: key, n: 0, accepted: 0, members: 0, dN: 0, dAccepted: 0, dMembers: 0,
        maxCenterDelta: 0, maxLengthDelta: 0, maxNfaLog10Delta: 0,
        crossings: 0, bytes: 0, medianMs: 0, error: got ?? 'never ran',
      });
      continue;
    }
    const sum = summarize(got.rects, got.members);
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
    ...provenance(input),
    reps,
    baseline,
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
}
