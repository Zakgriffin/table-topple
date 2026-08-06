import { Camera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { captureDistortedGrayscale } from '../pipeline/capture.ts';
import { Backend } from '../pipeline/backend.ts';
import { computePoseFromCapture, PoseComputeState } from '../pipeline/poseCompute.ts';
import {
  ProfileSpan, checkNesting, getRoots, profilerDevToolsMirror, profilerPutRoots, profilerSetDevToolsMirror,
  profilerTakeRoots, spanDurationMs,
} from '../profiling/profiler.ts';
import {
  AllocationSample, allocationProbeResult, getGPUDevice, setAllocationProbe,
  setTransferProbe, TransferSample, transferLedger, transferLedgerReset,
} from './device.ts';
import { GpuTimelineResult, gpuTimelineArm, gpuTimelineDisarm, gpuTimelineResolve } from './gpuTimeline.ts';
import { awaitPageFocus } from './lsdChainVerify.ts';

// ── Dev harness: what does ONE WHOLE RECONSTRUCTION cost? ─────────────────
//
//   await timeReconstruction()          // 9 timed reps after an adaptive warm-up
//   await timeReconstruction(null, 9)
//
// This is the instrument the 2026-08-05 perf plan is blocked on, and it exists
// because NOTHING else answers the question. verifyLsdChain covers stages 1-4
// only (~10ms of a ~159ms reconstruction), and the 158.9ms breakdown on record
// came from a PROFILED run, which inflates every GPU-dispatch span and so is
// good for structure and proportion but not for wall clock. Before this there
// was no way to answer "did that change make a reconstruction faster" except by
// hand-timing.
//
// ── What it measures, and what it deliberately does not ──
//
// The timed unit is `computePoseFromCapture` -- gray in, pose out. That is the
// real product path and the thing every item on the perf list is trying to
// shrink. Two things sit outside it, both on purpose:
//
//   capture+preprocess is measured SEPARATELY (and only for simulated cameras,
//   which are the only ones that pay it -- a physical camera reads
//   lastRealCaptureGray and pays ~0). Folding it into the headline number would
//   make every measurement on this machine a measurement of the dev loop rather
//   than of the pipeline.
//
//   the display tail is not measured at all. It is deferred through the
//   visuals mailbox and is no longer awaited on the pose path, so including it
//   would re-couple exactly what 97bf24b decoupled.
//
// ── Why it runs on a DETACHED state object ──
//
// computePoseFromCapture mutates its argument in place -- lastRecoveredAxes,
// lastPositionDecode, lastDecodeGrid and friends. A real Camera structurally
// satisfies PoseComputeState, so passing one would work and would ALSO
// overwrite whatever the app is currently displaying, and leave the last rep's
// state behind afterwards. So each rep gets a fresh, null-initialized state
// carrying only the camera's `settings` and `aspect`. This is the same shape
// mobileCapture.ts builds for on-device pose recovery, which is the existing
// proof that the pipeline runs standalone against one.
//
// ── The `ok` column is not decoration ──
//
// A reconstruction that BAILS is fast. If gridPeriodPhase finds no period or
// the quadric fit comes back degenerate, whole stages return early and the
// wall clock drops -- so a timing harness with no correctness column will
// happily report a large speedup for a change that broke the pipeline. `ok`,
// `votes` and `consistency` are here to make that impossible to miss, and a
// run where ok is false should be treated as having no timing result at all.
//
// HOW TO READ `consistency`, because it is easy to misjudge: it is the LOCAL
// consistency of a CORRECT patch -- the fraction of sampled cells whose decoded
// bit matches the torus at the winning registration. **The failure floor is
// ~50%**, i.e. chance, meaning the winner means nothing. Values in the 60s are a
// clear, good local patch, NOT a warning sign.
//
// It also doubles as a free ACCURACY signal rather than just a bail check,
// which is the more useful property here: it rises as the POSE improves,
// because a better pose skews the sampling lattice less, so cell centres land
// nearer the middle of their cells and fewer bits flip. So if a PERF change ever
// moves this number, that is a real pose regression hiding inside a speedup.

interface TransferGroup {
  what: string;
  kind: TransferSample['kind'];
  dir: 'up' | 'down';
  count: number;
  bytes: number;
  ms: number;
  // Probe-mode attribution, averaged over this group's samples. See device.ts.
  // byteMs = ms - bareFenceMs is the byte-proportional half; fenceMs is the
  // fixed round trip that a 4-byte readback would have paid just as much.
  fenceMs: number | null;
  byteMs: number | null;
  queuedAheadMs: number | null;
}

// ── Reading the votes stage out of the span tree ──────────────────────────
//
// The perf TODO's "split the votes stage into sub-timings" item. What it is
// after is a subtraction with no answer today: `votes` is ~11.7ms, gpuTimeline
// says ~4.7ms of that is device execution and the transfer ledger says ~0.3ms is
// bytes, leaving ~7ms attributed to nothing at all.
//
// This reports it off the profiler's own span tree rather than from a parallel
// instrument. The tree was always the right structure and was briefly thought
// unreachable from here, because turning it on also turned on per-module GPU
// timestamp queries costing a mapAsync each -- those queries are now deleted
// outright (see gpuTimeline.ts) and spans record unconditionally, so there is
// no configuration to arrange at all.
//
// ── SELF TIME is the measurement, and it is why a tree beats a flat list ──
//
// selfMs = a span's own duration minus the sum of its children's. So the
// unattributed time appears AS A ROW -- the self time of `votes stage` and of
// `composites (2x2 gradient field)` -- localized to the function it is actually
// in, rather than as one global remainder computed by subtracting a sum. Nothing
// has to be kept disjoint by hand, and there is no invariant to violate: adding
// a child span later just moves time out of its parent's self row.
//
// ── WHAT THIS IS BLIND TO, which decides how to read it ──
//
// A span that encloses an `await` measures WALL TIME, so it absorbs whatever the
// event loop did while suspended: the fence (wanted), but also a rAF render that
// landed mid-fence (not wanted, and a named suspect for the 7ms). No span can
// separate those two.
//
// The asymmetry that rescues it: spans split in two, and the SYNCHRONOUS ones
// contain no await at all, so on a single-threaded event loop nothing can be
// interleaved into them. A large number there is real executing JS -- or a GC
// pause, which is not a confound here but the object-churn hypothesis itself.
// That half is what makes the hypothesis testable; the awaiting half is
// meaningful only in aggregate.
//
// Which spans are which is declared at the span (ProfileSpan.sync), not by a
// list of names here. This module used to carry that list, covering four spans
// defined in four other modules, and it could not be right about one of them.
interface SpanRow {
  label: string;
  medianMs: number; // median over reps of this label's SELF time
  count: number;    // spans carrying this label per reconstruction
  depth: number;    // indent in the tree, so the table reads as structure
  // Straight off ProfileSpan.sync, declared at the span itself -- see its
  // comment. True means no await inside, so the number is host CPU.
  sync: boolean;
}

// Labels belonging to OTHER operations -- the display tail, the capture path,
// a whole competing reconstruction. None of them can legitimately appear while
// this harness is running, because it calls computePoseFromCapture directly and
// nothing else should be recording.
//
// This check exists because profiler.ts keeps ONE shared span stack and its
// header says what that costs: a second operation running concurrently gets its
// spans REPARENTED under whatever the first has open. The pose path suspends at
// every readback, so if an auto-capture fires during one of those awaits, its
// entire tree lands inside `votes stage` and gets subtracted from its self time
// -- silently, and in the direction that makes the answer look attributed. That
// is precisely the shape of failure worth spending ten lines to detect rather
// than trusting a run to be quiet.
const FOREIGN_SPANS = [
  'axesReconstruction', 'capture+preprocess', 'projectBins', 'poleMarkers+overlays',
  'idle (waiting for next frame)', 'ingest (decode+preprocess)', 'image decode',
];

interface VotesBreakdown {
  rows: SpanRow[]; // tree order (pre-order, by start time), not sorted by cost
  syncMs: number;  // sum of the sync rows -- real host CPU work
  asyncMs: number; // sum of the awaiting rows -- fence + kernel + rAF, mixed; a sum only
  // Foreign span labels seen in any rep. NON-EMPTY MEANS THE BREAKDOWN IS VOID:
  // another operation was recording into the same span stack and its cost is
  // inside these rows. Turn auto-capture off and re-run.
  foreign: string[];
  // ── Two IMPOSSIBLE-VALUE checks, reported rather than rendered ──
  //
  // `stageDeltaMs` used to live here: `votes stage` span duration minus the
  // stage's own votesMs, asserting the two had not drifted apart. It is gone
  // because they are now the SAME NUMBER -- poseCompute returns
  // spanDurationMs(stageSpan) -- so there is no longer anything to drift.
  //
  // These two replace it, and unlike it they catch faults that are live:
  //
  // Labels whose SELF TIME came out negative. A span cannot be shorter than its
  // children, so any row here means the tree's structure is wrong for that row
  // -- the known cause being a Promise.all's two concurrent awaits, which the
  // single span stack records as parent/child even though their durations
  // OVERLAP. Such a row's percentage is meaningless and it is suppressed from
  // the table rather than printed as a small negative number that reads like a
  // rounding artifact.
  negativeSelf: string[];
  // Parent/child pairs where the child outlasts the parent -- profiler.ts's
  // checkNesting, run over the votes subtree. Same root cause as above and the
  // same consequence: a non-empty list means the DECOMPOSITION is not to be
  // trusted, even where individual durations are.
  nestingViolations: string[];
  // performance.now() is coarsened to ~100us without cross-origin isolation and
  // ~5us with it. The sync spans are expected to be in the hundreds of
  // microseconds, so a non-isolated run does not merely add noise -- it biases
  // every one of them DOWN, the same way it made the allocation probe report a
  // real 0.60ms as 0.00ms. A run with this false has no small-span result.
  crossOriginIsolated: boolean;
}

// ── Every bus crossing that happens DURING a reconstruction ───────────────
//
// Built from the transfer ledger in the TIMED reps (probe off), so the ms column
// is a clean median over N rather than the probe rep's tripled wall clock.
//
// This is the table that answers "is the pipeline actually readback-free yet",
// and the answer it gives is no: the single-fence target is perf-TODO item 9 and
// is not started. What landed was the removal of the HABITUAL crossings -- the
// display-only ones deferred into the visuals drain, the region CSR made opt-in,
// `collect:regionCount` replaced by indirect dispatch. Every crossing listed
// here is load-bearing: the host needs the value to size a dispatch, terminate a
// loop, or finish a computation it has not ported.
//
// ── STALLS, not readbacks, and the difference is the whole point ──
//
// A readback is not a fence. `regions:counts` and `lsdFit:rects` sit in one
// Promise.all after one submit, as do tally's two -- they wait CONCURRENTLY, so
// their costs OVERLAP rather than sum, and adding the ms column double-counts
// them. That correction has had to be made by hand repeatedly in the perf notes
// ("12 readbacks are ~8 distinct stalls"), so it is computed here instead:
// crossings whose [start, start+ms] windows overlap are one stall, and a stall
// costs its wall time, not the sum of its members.
interface StallGroup { members: string[]; ms: number }

interface CrossingRow {
  what: string;
  kind: TransferSample['kind'];
  dir: 'up' | 'down';
  count: number;    // occurrences per reconstruction
  bytes: number;    // per reconstruction
  medianMs: number; // median over reps of this label's per-frame total
  // EVERY stall group this label appears in, not just one. It is a list because
  // a label can legitimately occur more than once per reconstruction in
  // DIFFERENT stalls -- `grow:converged` is read once per batch of CCL rounds,
  // so it lands in two sequential stalls. A scalar here reported only the last,
  // which made the table say "7 stalls" while printing six distinct numbers and
  // silently hid a 4.23ms stall behind a duplicate label.
  // Empty for kinds that never fence (uploads, converts).
  stalls: number[];
}

interface CrossingReport {
  rows: CrossingRow[]; // ledger order, i.e. pipeline order
  readbacks: number;
  stalls: number;      // distinct stall groups -- concurrent readbacks count ONCE
  // Sum over stall groups of each group's WALL time. The honest "how much of the
  // reconstruction is spent blocked on the device" figure, and strictly less than
  // summing the ms column.
  stallMs: number;
}

interface ReconstructionTimingReport {
  reps: number;
  w: number;
  h: number;
  cameraKind: 'simulated' | 'physical';
  // WHICH CONFIGURATION PRODUCED THESE NUMBERS. Recorded rather than assumed,
  // because the alternative is what voided every timing this project took before
  // 2026-08-06: the backend was an ambient global at measurement time, so a
  // report could not say what it had measured and the numbers were not
  // re-derivable afterwards. A measurement that does not carry its inputs is not
  // a measurement.
  backend: Backend;

  // ── the headline ──
  //
  // TWO statistics, because they answer different questions and the second is
  // the one to compare code changes with.
  //
  // poseMedianMs is what a reconstruction typically costs -- the user-facing
  // number, and the one to quote for "how fast is the pipeline".
  //
  // poseMinMs is the least-contaminated sample: no rAF frame landed mid-rep, no
  // GC, no scheduler interference. Measured spread here is 15-38% while the
  // MINIMUM is stable across runs, which makes the min the sensitive instrument
  // for "did that change help" even though it flatters the pipeline. The spread
  // is not noise to average away -- 16 fences per rep are 16 browser-scheduled
  // mapAsync callbacks, and whether a rep straddles a frame boundary is exactly
  // the rAF-alignment effect the perf plan lists as item 11.
  poseMedianMs: number;
  poseMinMs: number;
  poseMsAll: number[];
  // computePoseFromCapture's own per-stage breakdown, median over reps.
  stageMedianMs: { votes: number; fit: number; pose: number; distance: number; decode: number };
  captureMedianMs: number | null; // simulated only; NOT included in poseMedianMs

  // ── the fence/byte accounting, from the probe-OFF reps ──
  fences: number; // readbacks per reconstruction -- the count the single-fence plan is trying to drive to 1
  transferBytes: number;
  // Total time inside the upload/readback/convert helpers, probe off.
  //
  // READ THIS AS AN UPPER BOUND, NOT AS "TRANSFER COST". A readback blocks until
  // everything already queued has executed, so this number INCLUDES the GPU
  // compute the CPU happened to be waiting on -- measured at ~16ms of ~33ms on
  // the first real run, i.e. the majority of it was the pipeline working, not
  // the bus. Only the probe table below separates the two, and the honest
  // summary of that split is fenceMs + byteMs, not this.
  transferMs: number;
  transferSharePct: number;

  // ── attribution, from ONE probe-ON rep ──
  // Its wall clock is meaningless (probing triples every readback); only the
  // fenceMs/byteMs split is. Sorted by ms descending -- the actionable order.
  probe: TransferGroup[] | null;
  probeNote: string;

  // ── every bus crossing during a reconstruction, from the CLEAN reps ──
  // Answers "is the pipeline readback-free yet" (no -- that is item 9) and
  // separates readbacks from STALLS, which is not the same count.
  crossings: CrossingReport;

  // ── where the HOST time inside `votes` goes, from EVERY timed rep ──
  //
  // Unlike the three probes below, this rides along in the timed reps, because
  // recording it costs nothing this run can resolve: spans are a
  // performance.now() pair and an object each, ~14 per reconstruction, and the
  // DevTools mirror (the expensive half) is forced off for the duration. So
  // every row is a median over N reps on the same footing as the stage timings
  // it decomposes -- which are read off these very spans.
  //
  // That matters more than it sounds: the standing lesson from the kernel table
  // is that a single sample moves -- 4.65ms was one rep and the ranking it
  // produced changed three times under measurement fixes. A one-rep breakdown
  // compared against a nine-rep median would be comparing two different runs.
  votesBreakdown: VotesBreakdown;

  // ── where the GPU compute actually goes, from its own rep ──
  // The only instrument that can see INSIDE `votes`. Null when the device lacks
  // the optional 'timestamp-query' feature -- which is a different statement
  // from "no kernels ran", so it must not be reported as zeros.
  gpuTimeline: GpuTimelineResult | null;

  // ── what buffer churn costs, from its own rep ──
  // The question perf-TODO item 5 rests on. Null only if there is no device.
  // Read createMs+destroyMs against poseMedianMs: that ratio is the entire case
  // for or against a persistent arena, and it has never been measured.
  allocation: AllocationSample | null;

  // ── correctness, so a "fast" bail cannot masquerade as a win ──
  ok: boolean;
  votes: number;
  consistency: number | null;
  distance: number | null;

  // ── did this run actually reach steady state ──
  warmupMs: number[]; // untimed warm-up reps, in order -- should be visibly descending then flat
  warmedUp: boolean;  // false means WARMUP_MAX hit while still improving; the reps below are still warming
  // Interquartile range over median, as a percent -- a ROBUST spread, so one
  // scheduler hiccup does not define the whole run's credibility the way a
  // max-minus-min does. The harness's own noise floor, measured rather than
  // assumed: do not believe a median-to-median change smaller than this.
  spreadPct: number;

  focusWaitMs: number;
  focusedThroughout: boolean;
  // True if EITHER profiler flag was set before this run. Not a validity
  // problem for spans (this run turns those on itself), but worth reporting:
  // whoever had timestamps on was reading inflated GPU-dispatch spans, and their
  // previous numbers are not comparable to this run's.
  profilerWasOn: boolean;
}

// Interquartile range. Nearest-rank quartiles rather than interpolated -- with
// 9 reps the difference is noise, and this way the number is always one real
// sample minus another.
function iqr(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  if (s.length < 4) return s.length ? s[s.length - 1] - s[0] : 0;
  return s[Math.floor(s.length * 0.75)] - s[Math.floor(s.length * 0.25)];
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s.length === 0 ? 0 : s[Math.floor(s.length / 2)];
}

// One fresh state per rep. Nothing carries over between reps, which matters:
// a state reused across reps would keep the previous pose alive and let a rep
// that should have failed silently inherit the last good answer.
// The pose exactly as production runs it, INCLUDING deferring the decode grid's
// readback -- and then dropping it on the floor.
//
// Passing false instead would have been the tempting "measure everything"
// choice, and it would have made this harness the only caller still paying for a
// readback the real pose path stopped paying for: the fence count would report
// 12 where production does 10, and the median would carry ~0.45MB this pipeline
// no longer moves. What is being timed is gray in / pose out, and the grid is
// not part of the pose.
//
// Released rather than resolved, since nothing here paints: releasing frees the
// device buffers without performing the transfer, which is the whole point. A
// rep that leaked instead would hold ~0.45MB per rep across the sweep.
async function poseOnce(state: PoseComputeState, gray: Float64Array, w: number, h: number, backend: Backend) {
  try {
    return await computePoseFromCapture(state, gray, w, h, backend, true);
  } finally {
    state.pendingDecodeGrid?.release();
    state.pendingDecodeGrid = null;
  }
}

function freshState(camera: Camera): PoseComputeState {
  return {
    aspect: camera.aspect,
    settings: camera.settings,
    lastVoteComposites: null, lastVotes: null, lastQuadricPair: null, lastGridPeriodPhase: null,
    lastRecoveredAxes: null, lastDecodeGrid: null, lastDecodeRotated: null, lastDecodeCorrectness: null,
    lastPositionDecode: null, lastChainTransfers: null, pendingDecodeGrid: null,
  };
}

// One rep's span tree, flattened to (label, selfMs, depth) in pre-order.
//
// Sorted by start time at each level so the output reads top-to-bottom in the
// order the work happened -- the same choice formatFlamechart makes, and the
// reason the rows do not need a cost sort: this table is read as a decomposition,
// where position carries meaning, not as a ranking.
//
// Only the subtree under `votes stage` is walked. The other stages have no child
// spans worth decomposing yet, and including them would put rows in the table
// whose parents are not shown.
function flattenSelfTimes(
  spans: readonly ProfileSpan[], depth: number,
  out: { label: string; selfMs: number; depth: number; sync: boolean }[],
): void {
  // Every child counts now. There used to be a filter here dropping spans of
  // kind 'gpu' -- synthetic placements of GPU kernel durations on the host
  // timeline, which would have double-counted device time against host time.
  // That mechanism is deleted (device time lives in gpuTimeline, on its own
  // clock, in its own table), so the filter has nothing left to exclude and its
  // absence cannot silently corrupt the arithmetic.
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    let childMs = 0;
    for (const c of s.children) childMs += spanDurationMs(c);
    out.push({ label: s.name, selfMs: spanDurationMs(s) - childMs, depth, sync: s.sync });
    flattenSelfTimes(s.children, depth + 1, out);
  }
}

function findSpan(spans: readonly ProfileSpan[], name: string): ProfileSpan | null {
  for (const s of spans) {
    if (s.name === name) return s;
    const hit = findSpan(s.children, name);
    if (hit) return hit;
  }
  return null;
}

function collectForeign(spans: readonly ProfileSpan[], into: Set<string>): void {
  for (const s of spans) {
    if (FOREIGN_SPANS.some((f) => s.name.startsWith(f))) into.add(s.name);
    collectForeign(s.children, into);
  }
}

// Readbacks whose wait windows overlap were queued behind the same submit and
// resolved together, so they are ONE stall costing its wall time. Sweep in start
// order, extending the current group while the next sample starts before the
// group ends.
//
// Uploads and converts are excluded: neither fences (uploads are
// mappedAtCreation-synchronous, converts are pure CPU), so grouping them by
// overlap would invent stalls that do not exist.
function stallGroups(samples: readonly TransferSample[]): StallGroup[] {
  const reads = samples.filter((s) => s.kind === 'readback').sort((a, b) => a.startMs - b.startMs);
  const groups: StallGroup[] = [];
  let start = 0, end = -Infinity, members: string[] = [];
  for (const s of reads) {
    if (s.startMs > end) {
      if (members.length) groups.push({ members, ms: end - start });
      start = s.startMs; end = s.startMs + s.ms; members = [s.what];
    } else {
      end = Math.max(end, s.startMs + s.ms);
      members.push(s.what);
    }
  }
  if (members.length) groups.push({ members, ms: end - start });
  return groups;
}

function groupLedger(samples: readonly TransferSample[]): TransferGroup[] {
  const byKey = new Map<string, TransferGroup>();
  for (const s of samples) {
    const key = `${s.what}|${s.kind}|${s.dir}`;
    let g = byKey.get(key);
    if (!g) {
      g = {
        what: s.what, kind: s.kind, dir: s.dir, count: 0, bytes: 0, ms: 0,
        fenceMs: null, byteMs: null, queuedAheadMs: null,
      };
      byKey.set(key, g);
    }
    g.count++;
    g.bytes += s.bytes;
    g.ms += s.ms;
    if (s.bareFenceMs !== null) {
      g.fenceMs = (g.fenceMs ?? 0) + s.bareFenceMs;
      // Can go slightly negative on a small readback whose byte cost is below
      // the fence's own jitter. Left unclamped on purpose -- a negative here is
      // the honest reading "indistinguishable from a bare fence", and clamping
      // it to 0 would disguise noise as a real, if tiny, byte cost.
      g.byteMs = (g.byteMs ?? 0) + (s.ms - s.bareFenceMs);
    }
    if (s.queueDrainMs !== null && s.bareFenceMs !== null) {
      g.queuedAheadMs = (g.queuedAheadMs ?? 0) + (s.queueDrainMs - s.bareFenceMs);
    }
  }
  return [...byKey.values()].sort((a, b) => b.ms - a.ms);
}

export async function timeReconstruction(
  camera?: Camera | null, reps = 9, backend: Backend = 'gpu',
): Promise<ReconstructionTimingReport | string> {
  camera = camera ?? activeCamera() ?? null;
  if (!camera) return 'no active camera';
  const cap = camera.lastAxesCaptureGray;
  if (!cap) return 'no capture yet -- run a capture first';
  const { gray, w, h } = cap;

  // There is one profiler flag left and it gates the DevTools mirror only, so
  // "configure the profiler for a timing run" has shrunk to turning that off.
  // Spans record either way -- they are where both the stage timings and the
  // votes breakdown come from, so switching them off would leave this harness
  // with nothing to report.
  //
  // Saved and restored because a caller who had the mirror on gets it back, and
  // reported in the result because their PREVIOUS numbers were taken with a
  // performance.measure per span and are not comparable to these.
  const profilerWas = profilerDevToolsMirror();
  // Swapped aside, not cleared. Whoever ran a profiled capture before calling
  // this still has their tree to print afterwards, and their DevTools recording
  // keeps its measures -- the property applyProfilerToggle's comment protects.
  const rootsWas = profilerTakeRoots();
  // The mirror off for the duration: a performance.measure per span across ~25
  // reps is main-thread work inside the window being timed, and nobody is
  // watching a DevTools track during a harness run. Spans themselves are not
  // optional any more and are not turned off here -- they are where the stage
  // timings come from.
  profilerSetDevToolsMirror(false);

  try {
    const focusWaitMs = await awaitPageFocus();

    // ── Warm-up, until it stops getting faster ──────────────────────────────
    //
    // ONE warm-up rep was not enough and the first two real runs of this harness
    // proved it: identical code and an identical capture measured 39.4ms and
    // then 30.1ms, a 24% swing that dwarfs the +-2-3ms noise floor recorded for
    // verifyLsdChain. That is not variance to average away, it is a TREND --
    // shader compilation (~88ms for the fused decode alone) plus V8 tiering,
    // which keeps improving for several iterations, not one.
    //
    // A fixed larger count would work but has to be guessed. This instead runs
    // until the improvement stops, which is well defined here because warm-up is
    // MONOTONE: tiering only ever makes it faster, so the running MINIMUM is a
    // clean progress signal in a way the median or mean is not (either can move
    // either direction on noise alone). Stop once a whole batch fails to beat
    // the best by more than WARMUP_IMPROVE, i.e. the floor has stopped falling.
    //
    // The min is used ONLY to decide when to stop. The reported number is still
    // the median of the timed reps that follow, because a minimum is a
    // best-case and this is supposed to report a typical reconstruction.
    const WARMUP_BATCH = 3, WARMUP_MAX = 18, WARMUP_IMPROVE = 0.02;
    const warmupMs: number[] = [];
    let best = Infinity, warmedUp = false;
    while (warmupMs.length < WARMUP_MAX) {
      let improved = false;
      for (let i = 0; i < WARMUP_BATCH && warmupMs.length < WARMUP_MAX; i++) {
        const t = performance.now();
        await poseOnce(freshState(camera), gray, w, h, backend);
        const ms = performance.now() - t;
        warmupMs.push(ms);
        if (ms < best * (1 - WARMUP_IMPROVE)) { best = Math.min(best, ms); improved = true; }
        else best = Math.min(best, ms);
      }
      if (!improved) { warmedUp = true; break; }
    }

    const poseMsAll: number[] = [];
    const stages = { votes: [] as number[], fit: [] as number[], pose: [] as number[], distance: [] as number[], decode: [] as number[] };
    const captureMs: number[] = [];
    let last: PoseComputeState | null = null;
    let fences = 0, transferBytes = 0, transferMs = 0;

    // Capture is timed in its OWN loop, BEFORE the pose reps, and its output is
    // discarded. Interleaving the two (which this did first) was a real
    // methodological bug, not a style choice: captureDistortedGrayscale renders
    // the RT and runs ~36ms of supersampled CPU filtering, which evicts cache
    // and leaves the GPU queue in a different state than the pose stage would
    // otherwise see. The symptom was unmissable once warm-up was fixed -- the
    // warm-up tail (no capture between reps) settled around 20ms while the timed
    // reps (capture between reps) sat at 27.7ms, measuring the same code on the
    // same input. Whichever of those is "right", they cannot both be, and the
    // one contaminated by an unrelated 36ms of work is the wrong one.
    if (isSimulated(camera)) {
      for (let rep = 0; rep < reps; rep++) {
        const t0 = performance.now();
        captureDistortedGrayscale(camera);
        captureMs.push(performance.now() - t0);
      }
    }

    // Per-rep self times, keyed by label. Accumulated per rep rather than summed
    // across them so every row is a MEDIAN on the same footing as the stage
    // timings it decomposes.
    const spanMs = new Map<string, number[]>();
    const spanMeta = new Map<string, { depth: number; count: number; sync: boolean }>();
    const foreign = new Set<string>();
    // Impossible values, collected across reps. Sets rather than counts because
    // WHICH label is broken is what tells a reader whether the row they care
    // about is affected -- a bare count would make every row suspect.
    const negativeSelf = new Set<string>();
    const nestingViolations = new Set<string>();
    // Per-label crossing cost from the CLEAN reps, and the stall grouping. Kept
    // per rep so both get a median instead of resting on whichever rep the probe
    // happened to run.
    const crossMs = new Map<string, number[]>();
    const crossMeta = new Map<string, { kind: TransferSample['kind']; dir: 'up' | 'down'; count: number; bytes: number; repSeen: number }>();
    const stallCounts: number[] = [];
    const stallMsAll: number[] = [];
    let stallOf = new Map<string, number[]>();

    for (let rep = 0; rep < reps; rep++) {
      const state = freshState(camera);
      transferLedgerReset();
      // Take-and-discard, so `getRoots()` after the rep is exactly this rep's
      // tree rather than every tree since the run began (including the warm-up's
      // and the capture loop's). Taken rather than profilerReset()'d because
      // reset also clears the User Timing buffer -- see profilerTakeRoots.
      profilerTakeRoots();
      const t0 = performance.now();
      const timing = await poseOnce(state, gray, w, h, backend);
      poseMsAll.push(performance.now() - t0);

      // Walked from the `votes stage` span rather than from the roots, so the
      // table is exactly the stage being decomposed. Absent only if spans somehow
      // did not record, in which case the rows stay empty rather than reporting
      // zeros as though they were measurements.
      const roots = getRoots();
      collectForeign(roots, foreign);
      const stage = findSpan(roots, 'votes stage');
      if (stage) {
        const flat: { label: string; selfMs: number; depth: number; sync: boolean }[] = [];
        flattenSelfTimes([stage], 0, flat);
        // Summed per label within a rep before being pushed, so a label
        // appearing twice in one reconstruction contributes ONE sample of its
        // per-frame total. Pushing each occurrence separately would make the
        // median a median over occurrences, which for a span that runs twice
        // reports half of what the frame paid.
        const perRep = new Map<string, { ms: number; depth: number; count: number; sync: boolean }>();
        for (const f of flat) {
          const e = perRep.get(f.label);
          // Two spans sharing a label must agree on `sync` for the total to
          // mean anything, so a disagreement collapses to the safe answer
          // rather than to whichever ran last.
          if (e) { e.ms += f.selfMs; e.count++; e.sync = e.sync && f.sync; }
          else perRep.set(f.label, { ms: f.selfMs, depth: f.depth, count: 1, sync: f.sync });
        }
        for (const [label, v] of perRep) {
          // Recorded and then EXCLUDED, rather than pushed and rendered. A
          // negative self time is not a small measurement, it is proof the tree
          // is wrong for this label -- so letting it into the median would put a
          // fabricated number in the table and let it drag `asyncMs` down with
          // it. The label survives in negativeSelf so the report can say which
          // row is missing and why.
          if (v.ms < 0) { negativeSelf.add(label); continue; }
          if (!spanMs.has(label)) spanMs.set(label, []);
          spanMs.get(label)!.push(v.ms);
          spanMeta.set(label, { depth: v.depth, count: v.count, sync: v.sync });
        }
        for (const v of checkNesting([stage])) {
          nestingViolations.add(`${v.child} (${v.childMs.toFixed(2)}ms) inside ${v.parent} (${v.parentMs.toFixed(2)}ms)`);
        }
      }

      const led = transferLedger();
      // Counted per rep and kept from the LAST one rather than summed: these
      // are "per reconstruction" quantities, and every rep runs the identical
      // computation, so summing would just multiply by reps.
      fences = led.filter((s) => s.kind === 'readback').length;
      transferBytes = led.reduce((a, s) => a + (s.kind === 'convert' ? 0 : s.bytes), 0);
      transferMs = led.reduce((a, s) => a + s.ms, 0);

      // Per-label totals for THIS rep, then pushed as one sample each, so a
      // label occurring twice a frame contributes its per-frame total rather
      // than two half-sized samples to the median.
      {
        const perRep = new Map<string, number>();
        for (const s of led) {
          perRep.set(s.what, (perRep.get(s.what) ?? 0) + s.ms);
          // OVERWRITTEN per rep, not accumulated across them: count and bytes are
          // "per reconstruction" quantities and every rep does identical work, so
          // accumulating would report nine reconstructions' worth.
          const m = crossMeta.get(s.what);
          if (m && m.repSeen === rep) { m.count++; m.bytes += s.bytes; }
          else crossMeta.set(s.what, { kind: s.kind, dir: s.dir, count: 1, bytes: s.bytes, repSeen: rep });
        }
        for (const [what, ms] of perRep) {
          if (!crossMs.has(what)) crossMs.set(what, []);
          crossMs.get(what)!.push(ms);
        }
      }
      const groups = stallGroups(led);
      stallCounts.push(groups.length);
      stallMsAll.push(groups.reduce((a, g) => a + g.ms, 0));
      // Overwritten each rep rather than accumulated: the grouping is structural
      // (which readbacks share a submit) and identical every rep, so the last
      // one is representative and cheaper than reconciling nine copies.
      stallOf = new Map();
      groups.forEach((g, i) => {
        // Every DISTINCT label in the group, appended to whatever that label
        // already has. Built by accumulation rather than by Map-from-pairs
        // because the pair form silently kept the last write per key, which is
        // exactly how one of two stalls went missing from the table.
        for (const m of new Set(g.members)) stallOf.set(m, [...(stallOf.get(m) ?? []), i]);
      });

      stages.votes.push(timing.votesMs);
      stages.fit.push(timing.fitMs);
      stages.pose.push(timing.poseMs);
      stages.distance.push(timing.distanceMs);
      stages.decode.push(timing.decodeMs);
      last = state;
    }

    // ── The discriminating experiment: ONE rep with probe mode on ──
    // Separate phase, never interleaved with the timed reps, because probing
    // adds two extra readbacks per crossing and would poison every number
    // above if it ran alongside them.
    let probe: TransferGroup[] | null = null;
    try {
      setTransferProbe(true);
      transferLedgerReset();
      await poseOnce(freshState(camera), gray, w, h, backend);
      probe = groupLedger(transferLedger());
    } finally {
      setTransferProbe(false);
      transferLedgerReset();
    }

    // ── And ONE more rep, for what a buffer costs to exist ──
    // Its own rep for the same reason as the transfer probe, though for a
    // weaker reason: the patches add only a performance.now() pair per call, so
    // this would not visibly distort the median -- but there is no reason to put
    // measurement overhead in the headline number when a spare rep is free.
    // Separate from the transfer probe because that one triples every readback,
    // which would inflate the allocation count with staging buffers that only
    // exist because it is running.
    let allocation: AllocationSample | null = null;
    const probeDevice = await getGPUDevice();
    if (probeDevice) {
      try {
        setAllocationProbe(probeDevice, true);
        await poseOnce(freshState(camera), gray, w, h, backend);
      } finally {
        setAllocationProbe(probeDevice, false);
        allocation = allocationProbeResult();
      }
    }

    // ── And one more, for per-pass GPU kernel time ──
    // Its own rep for the strongest reason of the three: a timestamp query set
    // is bound into every compute pass descriptor, so arming it changes what the
    // passes ARE. Timing a rep that is measuring itself would be the same
    // mistake the per-module timestamps make (see gpuTimeline.ts's header on
    // the 9.3ms-around-a-0.07ms-kernel case), just at a different scale.
    let gpuTimeline: GpuTimelineResult | null = null;
    if (probeDevice && gpuTimelineArm(probeDevice)) {
      try {
        await poseOnce(freshState(camera), gray, w, h, backend);
        gpuTimeline = await gpuTimelineResolve();
      } finally {
        gpuTimelineDisarm();
      }
    }

    const poseMedianMs = median(poseMsAll);
    const pd = last?.lastPositionDecode ?? null;

    // Map insertion order is the pre-order tree walk, so no sort is needed to
    // make the table read as structure.
    const spanRows: SpanRow[] = [...spanMs.entries()].map(([label, xs]) => ({
      label,
      medianMs: median(xs),
      count: spanMeta.get(label)?.count ?? 0,
      depth: spanMeta.get(label)?.depth ?? 0,
      sync: spanMeta.get(label)?.sync ?? false,
    }));
    const sumWhere = (want: boolean) => spanRows.reduce((a, r) => a + (r.sync === want ? r.medianMs : 0), 0);

    const crossRows: CrossingRow[] = [...crossMs.entries()].map(([what, xs]) => {
      const m = crossMeta.get(what)!;
      return {
        what, kind: m.kind, dir: m.dir, count: m.count, bytes: m.bytes,
        medianMs: median(xs), stalls: m.kind === 'readback' ? (stallOf.get(what) ?? []) : [],
      };
    });

    return {
      reps, w, h,
      cameraKind: isSimulated(camera) ? 'simulated' : 'physical',
      backend,
      poseMedianMs,
      poseMsAll,
      stageMedianMs: {
        votes: median(stages.votes), fit: median(stages.fit), pose: median(stages.pose),
        distance: median(stages.distance), decode: median(stages.decode),
      },
      captureMedianMs: captureMs.length ? median(captureMs) : null,
      fences,
      transferBytes,
      transferMs,
      transferSharePct: poseMedianMs > 0 ? (transferMs / poseMedianMs) * 100 : 0,
      probe,
      probeNote: 'probe rep only: read fenceMs/byteMs, IGNORE its ms (probing triples every readback)',
      crossings: {
        rows: crossRows,
        readbacks: fences,
        stalls: stallCounts.length ? median(stallCounts) : 0,
        stallMs: stallMsAll.length ? median(stallMsAll) : 0,
      },
      votesBreakdown: {
        rows: spanRows,
        syncMs: sumWhere(true),
        asyncMs: sumWhere(false),
        foreign: [...foreign],
        negativeSelf: [...negativeSelf],
        nestingViolations: [...nestingViolations],
        crossOriginIsolated: globalThis.crossOriginIsolated === true,
      },
      allocation,
      gpuTimeline,
      ok: !!pd,
      votes: last?.lastVotes?.length ?? 0,
      consistency: pd ? pd.consistency : null,
      distance: last?.lastRecoveredAxes?.distance ?? null,
      warmupMs,
      warmedUp,
      profilerWasOn: profilerWas,
      poseMinMs: Math.min(...poseMsAll),
      spreadPct: poseMedianMs > 0 ? (iqr(poseMsAll) / poseMedianMs) * 100 : 0,
      focusWaitMs,
      // Checked at the END as well as the start: focus can be lost mid-run, and
      // if it was, every millisecond above is garbage. Same gate verifyLsdChain
      // uses, and for the same reason -- an unfocused tab throttles timers, so
      // the numbers are WRONG rather than merely noisy.
      focusedThroughout: document.hasFocus() && document.visibilityState === 'visible',
    };
  } finally {
    // Restore, or the harness silently reconfigures the app it just measured.
    profilerSetDevToolsMirror(profilerWas);
    profilerPutRoots(rootsWas);
    setTransferProbe(false);
  }
}

// Console-friendly rendering -- the report object is the source of truth, but
// the probe table is the part worth reading as a table.
export function formatReconstructionTiming(r: ReconstructionTimingReport | string): string {
  if (typeof r === 'string') return r;
  const lines: string[] = [];
  const warn: string[] = [];
  if (!r.ok) warn.push('!! ok=false -- the pipeline did not produce a pose. TIMING IS MEANINGLESS.');
  if (!r.focusedThroughout) warn.push('!! focus was lost mid-run -- every ms below is invalid.');
  // Not "forced off" any more: this run turns SPANS ON deliberately (that is
  // where the votes breakdown comes from) and only forces GPU timestamps off.
  // Both flags are restored afterwards, and so is the caller's own span tree.
  if (r.profilerWasOn) warn.push('note: the DevTools span mirror was on before this run; it was forced off for the duration, then restored.');
  if (!r.warmedUp) warn.push(`!! never reached steady state in ${r.warmupMs.length} warm-up reps -- still speeding up, treat as an upper bound.`);
  if (r.spreadPct > 12) warn.push(`!! IQR spread ${r.spreadPct.toFixed(0)}% across reps -- this run cannot resolve a change smaller than that.`);
  lines.push(...warn);
  lines.push(`reconstruction: ${r.poseMedianMs.toFixed(1)}ms median / ${r.poseMinMs.toFixed(1)}ms min of ${r.reps}  (${r.w}x${r.h}, ${r.cameraKind}, backend=${r.backend})`);
  lines.push(`  compare changes on the MIN; quote the MEDIAN.`);
  lines.push(`  reps: ${r.poseMsAll.map((m) => m.toFixed(1)).join(', ')}   (IQR spread ${r.spreadPct.toFixed(0)}%)`);
  lines.push(`  warm-up (${r.warmupMs.length} reps, ${r.warmedUp ? 'settled' : 'NOT settled'}): ${r.warmupMs.map((m) => m.toFixed(1)).join(', ')}`);
  const s = r.stageMedianMs;
  lines.push(`  stages: votes ${s.votes.toFixed(1)}  fit ${s.fit.toFixed(1)}  pose ${s.pose.toFixed(1)}  distance ${s.distance.toFixed(1)}  decode ${s.decode.toFixed(1)}`);
  if (r.captureMedianMs !== null) lines.push(`  capture+preprocess: ${r.captureMedianMs.toFixed(1)}ms (SIMULATED ONLY, not in the median above)`);
  lines.push(`  fences: ${r.fences}   bytes: ${(r.transferBytes / 1048576).toFixed(2)}MB   in transfer helpers: ${r.transferMs.toFixed(1)}ms (${r.transferSharePct.toFixed(0)}% -- UPPER BOUND, includes GPU compute waited on)`);
  if (r.probe) {
    let f = 0, b = 0, q = 0;
    for (const g of r.probe) { f += g.fenceMs ?? 0; b += g.byteMs ?? (g.kind === 'readback' ? 0 : g.ms); q += g.queuedAheadMs ?? 0; }
    lines.push(`  attributed: ${q.toFixed(1)}ms GPU compute queued ahead | ${f.toFixed(1)}ms fence latency | ${b.toFixed(1)}ms byte-proportional`);
  }
  {
    const c = r.crossings;
    lines.push(`  crossings DURING the reconstruction: ${c.readbacks} readbacks in ${c.stalls} STALLS`
      + `  --  ${c.stallMs.toFixed(1)}ms blocked (${r.poseMedianMs > 0 ? ((c.stallMs / r.poseMedianMs) * 100).toFixed(0) : '?'}% of the median)`);
    lines.push(`  ${'what'.padEnd(26)} ${'kind'.padEnd(9)} ${'n'.padStart(2)} ${'MB'.padStart(7)} ${'ms'.padStart(7)}  stall`);
    for (const row of c.rows) {
      // The stall column is what stops the ms column being summed. Two rows
      // sharing a number waited concurrently on one submit, so their cost
      // overlaps -- the honest total is the stallMs above, never this column added up.
      lines.push(`  ${row.what.padEnd(26)} ${row.kind.padEnd(9)} ${String(row.count).padStart(2)}`
        + ` ${(row.bytes / 1048576).toFixed(2).padStart(7)} ${row.medianMs.toFixed(2).padStart(7)}`
        + `  ${row.stalls.length ? row.stalls.map((n) => '#' + n).join(',') : '-- (never fences)'}`);
    }
    lines.push(`  do NOT sum the ms column -- rows sharing a stall number overlap; use the ${c.stallMs.toFixed(1)}ms above.`);
  }
  // Printed BEFORE the kernel table on purpose: the kernels are a known ~28% and
  // this is the instrument aimed at the majority that is not.
  {
    const vb = r.votesBreakdown;
    const s = r.stageMedianMs.votes;
    if (vb.rows.length === 0) {
      lines.push(`  votes ${s.toFixed(1)}ms -- NO SPAN BREAKDOWN (the 'votes stage' span did not record)`);
    } else {
      lines.push(`  votes ${s.toFixed(1)}ms, by span SELF time (median over reps; indent = nesting):`);
      for (const row of vb.rows) {
        const pct = s > 0 ? (row.medianMs / s) * 100 : 0;
        // A parent's self time is the time inside it that no child accounts for
        // -- which for `votes stage` and `composites` IS the unattributed time
        // the whole exercise is about, so it is labelled rather than left to be
        // inferred from the indentation.
        const kind = row.sync ? 'SYNC (host CPU)' : 'awaits (fence+kernel+rAF)';
        const tail = row.count > 1 ? `  x${row.count}` : '';
        lines.push(`    ${'  '.repeat(row.depth)}${(row.label + ' [self]').padEnd(34 - row.depth * 2)}`
          + ` ${row.medianMs.toFixed(2).padStart(6)}ms ${pct.toFixed(0).padStart(3)}%  ${kind}${tail}`);
      }
      lines.push(`    sync total ${vb.syncMs.toFixed(2)}ms (real host CPU) | awaiting total ${vb.asyncMs.toFixed(2)}ms (mixed, read as a sum only)`);
      if (vb.negativeSelf.length) {
        lines.push(`    !! ${vb.negativeSelf.length} row(s) had NEGATIVE self time and were suppressed: ${vb.negativeSelf.join(', ')}`);
        lines.push(`       A span cannot be shorter than its children. Concurrent awaits in one Promise.all are`);
        lines.push(`       recorded as parent/child by the single span stack, so their durations overlap. Read`);
        lines.push(`       those readbacks as ONE stall (see the crossings table), not as separate rows.`);
      }
      if (vb.nestingViolations.length) {
        lines.push(`    !! nesting is INVALID -- the decomposition above does not partition the stage:`);
        for (const v of vb.nestingViolations) lines.push(`       ${v}`);
      }
    }
    if (vb.foreign.length) {
      lines.push(`    !! BREAKDOWN VOID -- another operation recorded into the same span stack: ${vb.foreign.join(', ')}`);
      lines.push(`       Its cost is inside the rows above. Turn auto-capture off and re-run.`);
    }
    // Loud, because a false here silently biases every sync row DOWNWARD rather
    // than adding noise -- the failure that made the allocation probe read
    // 0.00ms for a real 0.60ms. The rows are not "approximate" without it, they
    // are wrong in a known direction.
    if (!vb.crossOriginIsolated) {
      lines.push(`    !! crossOriginIsolated === false -- performance.now() is coarsened to ~100us, so every`);
      lines.push(`       sub-millisecond row above is biased LOW. Hard-reload in a new tab (HMR will not apply COOP/COEP).`);
    }
  }
  if (r.gpuTimeline) {
    const t = r.gpuTimeline;
    // Stated against the median rather than left as a bare total, because the
    // useful question is what fraction of a reconstruction is device execution
    // -- and the answer is expected to be LESS than 100%, since the host does
    // not wait for every kernel.
    const pct = r.poseMedianMs > 0 ? (t.totalMs / r.poseMedianMs) * 100 : 0;
    lines.push(`  GPU kernels: ${t.totalMs.toFixed(2)}ms across ${t.passes} passes (${pct.toFixed(0)}% of the median)${t.overflowed ? '  !! PASS CAP HIT, undercounted' : ''}`);
    for (const e of t.entries) {
      lines.push(`    ${e.label.padEnd(20)} x${String(e.count).padStart(3)}  ${e.totalMs.toFixed(2).padStart(7)}ms   max ${e.maxMs.toFixed(3)}ms`);
    }
  } else {
    lines.push(`  GPU kernels: unavailable (no 'timestamp-query' feature on this device)`);
  }
  if (r.allocation) {
    const a = r.allocation;
    const churnMs = a.createMs + a.destroyMs;
    // The verdict is printed rather than left to be worked out, because the
    // whole point of this row is a go/no-go on perf-TODO item 5 and "1.2ms" on
    // its own does not say against what.
    const pct = r.poseMedianMs > 0 ? (churnMs / r.poseMedianMs) * 100 : 0;
    lines.push(`  buffers: ${a.creates} created (${(a.bytesAllocated / 1048576).toFixed(1)}MB), ${a.destroys} destroyed`
      + `  --  ${churnMs.toFixed(2)}ms total (${pct.toFixed(1)}% of the median), ${a.repeatedShapes} repeat shapes`);
    if (a.top.length) {
      lines.push(`  biggest repeat allocations (what an arena would take first):`);
      for (const t of a.top) {
        lines.push(`    ${(t.bytes / 1048576).toFixed(2).padStart(7)}MB x${String(t.count).padStart(3)} = ${((t.bytes * t.count) / 1048576).toFixed(2).padStart(7)}MB   usage 0x${t.usage.toString(16)}`);
      }
    }
  }
  lines.push(`  pose: votes ${r.votes}, consistency ${r.consistency !== null ? (r.consistency * 100).toFixed(1) + '%' : 'n/a'}, distance ${r.distance !== null ? r.distance.toFixed(2) : 'n/a'}`);
  if (r.probe) {
    lines.push(`  --- probe (${r.probeNote}) ---`);
    lines.push(`  ${'what'.padEnd(26)} ${'kind'.padEnd(9)} ${'n'.padStart(3)} ${'MB'.padStart(7)} ${'fence'.padStart(8)} ${'bytes'.padStart(8)} ${'queued'.padStart(8)}`);
    for (const g of r.probe) {
      lines.push(`  ${g.what.padEnd(26)} ${g.kind.padEnd(9)} ${String(g.count).padStart(3)} ${(g.bytes / 1048576).toFixed(2).padStart(7)} ${(g.fenceMs !== null ? g.fenceMs.toFixed(2) : '-').padStart(8)} ${(g.byteMs !== null ? g.byteMs.toFixed(2) : g.ms.toFixed(2)).padStart(8)} ${(g.queuedAheadMs !== null ? g.queuedAheadMs.toFixed(2) : '-').padStart(8)}`);
    }
  }
  return lines.join('\n');
}
