import { type Backend } from '../../pose/backend.ts';
import { computePoseFromCapture, type PoseResult } from '../../pose/poseCompute.ts';
import { NO_INTERMEDIATES } from '../../pose/intermediates.ts';
import type { HarnessInput } from './input.ts';
import {
  type PathNode, type TreeNode, criticalPath, joinRecords, profilerBeginSession,
} from '../profiling/profiler.ts';
import { POSE_STAGE_TABLE } from '../../pose/timing/stages.ts';
import {
  type AllocationSample, allocationProbeResult, getGPUDevice, setAllocationProbe,
  setTransferProbe, type TransferSample, transferLedger, transferLedgerReset,
} from '../../pose/gpu/device.ts';
import { type GpuTimelineResult, gpuTimelineArm, gpuTimelineDisarm, gpuTimelineResolve } from '../../pose/gpu/gpuTimeline.ts';
import { awaitPageFocus } from './lsdChainVerify.ts';

// ── Dev harness: what does ONE WHOLE RECONSTRUCTION cost? ─────────────────
//
//   await timeReconstruction(cameraInput())            // 9 reps after an adaptive warm-up
//   await timeReconstruction(await fixtureInput('default'), 9, 'cpu')
//
// The input is a REQUIRED argument. It used to default to activeCamera(), which
// is exactly how a report ended up unable to say what it had measured -- see
// harness/input.ts and harness/cameraInput.ts.
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
// ── Reps cannot contaminate each other, and it is no longer a precaution ──
//
// computePoseFromCapture used to MUTATE its argument, so this harness built a
// fresh null-initialized state per rep -- otherwise it would have overwritten
// whatever the app was displaying and left the last rep's results behind. The
// pipeline returns its result now, so each rep's answer is its own object and
// there is nothing to detach from. `input` goes straight in as the PoseInput.
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

// ── Reading the reconstruction out of the joined span tree ────────────────
//
// The perf TODO's "split the votes stage into sub-timings" item, grown to cover
// the whole pose. What it is after is a subtraction that had no answer: `votes`
// is ~11.7ms, gpuTimeline says ~4.7ms of that is device execution and the
// transfer ledger says ~0.3ms is bytes, leaving ~7ms attributed to nothing.
//
// Rooted at `pose.run` rather than at the votes stage. That is only possible
// because the library declares its own root now -- structure used to come from
// the call stack, so everything after the votes stage was a sibling under
// whatever the CALLER had open, and a table rooted above it would have shown
// rows whose parents were not in the tree.
//
// ── SELF TIME is the measurement, and the UNION is what makes it sound ──
//
// selfMs = a span's own duration minus the time covered by its children, where
// covered is the UNION of their intervals rather than the SUM of their
// durations. So the unattributed time appears AS A ROW, localized to the
// function it is actually in, rather than as one global remainder.
//
// The union is not a refinement, it is the correctness condition. Two children
// awaited concurrently in one Promise.all overlap; summing them double-counts
// and could push a parent's self time NEGATIVE, which this module used to
// detect and suppress. Contained intervals cannot out-cover their container, so
// the row is non-negative by construction. See profiler.ts's TreeNode.selfMs.
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
// Which spans are which is declared in the STAGE TABLE (StageNode.sync), not by
// a list of names here. This module used to carry that list, covering four
// spans defined in four other modules, and it could not be right about one of
// them.
interface SpanRow {
  label: string;
  medianMs: number; // median over reps of this label's SELF time
  count: number;    // spans carrying this label per reconstruction
  depth: number;    // indent in the tree, so the table reads as structure
  // Straight off StageNode.sync, declared once in the stage table -- see its
  // comment. True means no await inside, so the number is host CPU.
  sync: boolean;
}

interface SpanBreakdown {
  rows: SpanRow[]; // tree order (pre-order, by start time), not sorted by cost
  syncMs: number;  // sum of the sync rows -- real host CPU work
  asyncMs: number; // sum of the awaiting rows -- fence + kernel + rAF, mixed; a sum only
  // ── What used to be here, and why three of the four checks are GONE ──
  //
  // `foreign`, `negativeSelf` and `nestingViolations` all existed to report
  // ways the SPAN STACK could lie, and the stack is gone (see profiler.ts):
  //
  //   - `foreign` listed labels belonging to other operations, because a
  //     concurrent capture's spans got REPARENTED into `votes stage` and were
  //     subtracted from its self time. Records join by DECLARED id now, so a
  //     foreign record cannot land inside a pose stage at all. It is reported
  //     below as context, not as grounds for voiding the table.
  //   - `negativeSelf` caught self times below zero, whose known cause was a
  //     Promise.all's two concurrent awaits recorded as parent/child. Children
  //     are subtracted as a UNION of intervals now, which cannot exceed a
  //     parent that contains them.
  //   - `nestingViolations` ran checkNesting over the subtree. A child is only
  //     attached when its interval is contained in its parent's, so the
  //     violation it looked for is unrepresentable.
  //
  // `stageDeltaMs` went earlier and for a different reason: `votes stage`'s
  // duration and the stage's own votesMs are the SAME NUMBER now, read off one
  // record, so there is nothing left to drift.
  //
  // Two REAL checks remain. Both can still fire, and neither is about the
  // recorder:
  //
  // Records from outside the pose table -- another subsystem recording while
  // the harness ran. Harmless to the arithmetic, but it means the machine was
  // not quiet, which is worth knowing before believing a median.
  foreignIds: string[];
  // Records whose declared parent OVERLAPPED them without containing them.
  // Not a recorder artifact: it means a stage straddled its own parent's
  // boundary, so either the declaration in pose/timing/stages.ts is wrong or a
  // span is unbalanced. See joinRecords.
  straddled: string[];
  // performance.now() is coarsened to ~100us without cross-origin isolation and
  // ~5us with it. The sync spans are expected to be in the hundreds of
  // microseconds, so a non-isolated run does not merely add noise -- it biases
  // every one of them DOWN, the same way it made the allocation probe report a
  // real 0.60ms as 0.00ms. A run with this false has no small-span result.
  crossOriginIsolated: boolean;
}

// ── The DEPENDENT chain, which the self-time table structurally cannot show ──
//
// The span breakdown above decomposes cost; this one answers what to do about
// it. A stage OFF the critical path can be made free without the reconstruction
// getting any faster, and the self-time table gives no way to tell which stages
// those are -- it ranks by cost, and cost is not leverage.
//
// Two readings, and they are different findings:
//
//   sharePct near 100 means the level is a serial chain. There is no slack to
//   exploit and no reordering to do; the only lever is making a stage cheaper.
//   That is the expected answer for `pose.run` today and it is worth having
//   MEASURED rather than assumed, because it is the premise every "overlap
//   these two stages" idea rests on.
//
//   waitMs on an edge is a stall the self-time table files under the enclosing
//   stage. `fit.dispatch` ends at submit, `fit.finish` begins after the readback
//   resolves; the fence between them belongs to neither span and shows up in
//   `pose.fit`'s self time with nothing naming it. Here it is an edge with a
//   producer's name on it, which is the form the perf TODO's readback-stall
//   finding has always wanted and never had.
interface CriticalRow {
  id: string;
  label: string;
  depth: number;    // level in the chain, so the row reads as nesting
  medianMs: number; // median over reps of this stage's DURATION (not self time)
  waitMs: number;   // median over reps of start - readyAt
  boundBy: string | null;
}

interface CriticalPathReport {
  rows: CriticalRow[];
  // Median over reps of head-start to tail-end at the TOP level.
  spanMs: number;
  sharePct: number; // spanMs against the reconstruction median
  // Every waitMs on the chain, summed. These do not double-count: a wait at one
  // level sits BETWEEN two siblings, a wait a level down sits INSIDE one of
  // them, and neither interval is inside the other. So this is the honest "time
  // on the critical path spent waiting on a dependency rather than executing".
  waitTotalMs: number;
  // The chain's SHAPE changed between reps. The rows below are one rep's, so a
  // true here means they are not representative -- and it is a finding rather
  // than a defect: it means two stages are close enough in end time that which
  // one binds is decided by jitter, i.e. there is real slack somewhere.
  varied: boolean;
  // See CriticalPath.unsatisfied -- a declared input that had not finished when
  // its consumer started.
  unsatisfied: string[];
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
  // What was measured, by name -- 'fixture:default' or 'camera:<id>'. The
  // report's whole claim to being re-derivable rests on this plus `backend`
  // plus the fixture's own pinned config; see harness/input.ts.
  input: string;
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

  // ── where the HOST time in a whole reconstruction goes, EVERY timed rep ──
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
  spanBreakdown: SpanBreakdown;

  // ── which of those rows are actually in SERIES, every timed rep ──
  // Same records, joined the other way: `inputs` instead of `within`. This is
  // what makes a breakdown actionable rather than merely informative.
  criticalPath: CriticalPathReport;

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
// Asks for NOTHING, which is now literally free rather than nearly free. This
// used to pass deferDecodeGrid=true and then release the handle in a finally --
// deferring a readback and immediately throwing it away, because that was the
// only way to keep it off the measured path. An empty request does not build
// the handle, does not read the grid back, and destroys the residency in
// computePoseFromCapture's own finally, so there is nothing left here to clean
// up. See pose/intermediates.ts, capability (1).
async function poseOnce(input: HarnessInput, gray: Float64Array, w: number, h: number, backend: Backend) {
  return computePoseFromCapture(input, gray, w, h, backend, NO_INTERMEDIATES);
}

// One rep's joined tree, flattened to (label, selfMs, depth) in pre-order.
//
// Sorted by start time at each level so the output reads top-to-bottom in the
// order the work happened -- the same choice formatSpanTree makes, and the
// reason the rows do not need a cost sort: this table is read as a
// decomposition, where position carries meaning, not as a ranking.
//
// The whole `pose.run` tree is walked now, not just the votes subtree. That is
// new, and it is the point of the library declaring its own root: before, the
// stages after `votes stage` were siblings under whatever the caller happened
// to have open, so a table rooted anywhere would have shown rows whose parents
// were missing. There is a real root to walk from now.
function flattenSelfTimes(
  nodes: readonly TreeNode[], depth: number,
  out: { label: string; selfMs: number; depth: number; sync: boolean }[],
): void {
  for (const n of nodes) {
    out.push({ label: n.node.label, selfMs: n.selfMs, depth, sync: n.node.sync === true });
    flattenSelfTimes(n.children, depth + 1, out);
  }
}

// One rep's critical path, flattened depth-first in execution order. Same shape
// as flattenSelfTimes and for the same reason: position carries the structure,
// so the rows need no sort.
function flattenPath(
  chain: readonly PathNode[], depth: number,
  out: { id: string; label: string; depth: number; durationMs: number; waitMs: number; boundBy: string | null }[],
): void {
  for (const p of chain) {
    out.push({
      id: p.node.rec.id, label: p.node.node.label, depth,
      durationMs: p.node.durationMs, waitMs: p.waitMs, boundBy: p.boundBy,
    });
    flattenPath(p.inner, depth + 1, out);
  }
}

function findNode(nodes: readonly TreeNode[], id: string): TreeNode | null {
  for (const n of nodes) {
    if (n.rec.id === id) return n;
    const hit = findNode(n.children, id);
    if (hit) return hit;
  }
  return null;
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
  input: HarnessInput, reps = 9, backend: Backend = 'gpu',
): Promise<ReconstructionTimingReport> {
  const { gray, w, h } = input;

  // One session, and it owns the whole protocol: swap the caller's tree aside,
  // force the DevTools mirror off for the duration, hand back a fresh tree per
  // rep, and restore both at the end. This used to be four profiler primitives
  // and a try/finally written out here, which meant this harness had to know
  // that the tree and the mirror are module-level variables over there. See
  // profiler.ts's ProfilerSession.
  //
  // Spans themselves are NOT turned off -- they are where the stage timings and
  // the votes breakdown come from.
  const session = profilerBeginSession();

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
        await poseOnce(input, gray, w, h, backend);
        const ms = performance.now() - t;
        warmupMs.push(ms);
        if (ms < best * (1 - WARMUP_IMPROVE)) { best = Math.min(best, ms); improved = true; }
        else best = Math.min(best, ms);
      }
      if (!improved) { warmedUp = true; break; }
    }

    const poseMsAll: number[] = [];
    const stages = { votes: [] as number[], fit: [] as number[], pose: [] as number[], distance: [] as number[], decode: [] as number[] };
    let last: PoseResult | null = null;
    let fences = 0, transferBytes = 0, transferMs = 0;

    // The capture+preprocess loop that used to sit here is GONE with the switch
    // to a HarnessInput. It timed captureDistortedGrayscale, which only a
    // SIMULATED camera runs -- a physical capture reads a buffer and pays ~0 --
    // and a fixture is a physical capture by construction, so there is nothing
    // left for it to measure. It was never part of the headline number anyway
    // (its own comment: folding it in "would make every measurement on this
    // machine a measurement of the dev loop rather than of the pipeline").
    //
    // Worth keeping its one hard-won finding, because it is about METHOD and
    // applies to whatever measures that stage next: it had to run in its OWN
    // loop, before the pose reps, never interleaved. Interleaving put ~36ms of
    // supersampled CPU filtering between reps, which evicted cache and left the
    // GPU queue in a state the pose stage would not otherwise see -- the same
    // code on the same input measured 20ms with a clean loop and 27.7ms
    // interleaved.

    // Per-rep self times, keyed by label. Accumulated per rep rather than summed
    // across them so every row is a MEDIAN on the same footing as the stage
    // timings it decomposes.
    const spanMs = new Map<string, number[]>();
    const spanMeta = new Map<string, { depth: number; count: number; sync: boolean }>();
    // Collected across reps. Sets rather than counts because WHICH id is
    // involved is what tells a reader whether the row they care about is
    // affected -- a bare count would make every row suspect.
    const foreignIds = new Set<string>();
    const straddled = new Set<string>();
    // The critical path, per rep. Durations and waits keyed by stage id so every
    // number is a median on the same footing as the self-time rows beside it;
    // the SHAPE is taken from the last rep, with `cpShapes` there to say whether
    // that was a safe thing to do.
    const cpSpanMs: number[] = [];
    const cpDurMs = new Map<string, number[]>();
    const cpWaitMs = new Map<string, number[]>();
    const cpUnsatisfied = new Set<string>();
    const cpShapes = new Set<string>();
    let cpRows: CriticalRow[] = [];
    // Per-label crossing cost from the CLEAN reps, and the stall grouping. Kept
    // per rep so both get a median instead of resting on whichever rep the probe
    // happened to run.
    const crossMs = new Map<string, number[]>();
    const crossMeta = new Map<string, { kind: TransferSample['kind']; dir: 'up' | 'down'; count: number; bytes: number; repSeen: number }>();
    const stallCounts: number[] = [];
    const stallMsAll: number[] = [];
    let stallOf = new Map<string, number[]>();

    // Discard everything the warm-up accumulated, so rep 0's tree is rep 0's.
    session.takeRepRecords();

    for (let rep = 0; rep < reps; rep++) {
      transferLedgerReset();
      const t0 = performance.now();
      const pose = await poseOnce(input, gray, w, h, backend);
      const timing = pose.timing;
      poseMsAll.push(performance.now() - t0);

      // Taken, not read: this rep's records come out and the next rep starts on
      // a fresh set, so no rep can see another's. (The warm-up's records are
      // discarded by the take just before the loop.)
      //
      // Joined against the LIBRARY's table alone, not the app's merged one.
      // That is deliberate and it is what makes `foreignIds` mean something:
      // this harness calls computePoseFromCapture directly, so any record that
      // fails to join belongs to something else that was running -- an
      // auto-capture, a preview projection -- and the run was not quiet.
      const recs = session.takeRepRecords();
      const join = joinRecords(recs, POSE_STAGE_TABLE);
      for (const u of join.unknown) foreignIds.add(u.id);
      for (const o of join.orphans) straddled.add(o.id);
      const stage = findNode(join.roots, 'pose.run');
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
          // No negative-self filter here any more. There used to be one, and it
          // was load-bearing: self times were parent-minus-SUM-of-children, so
          // a Promise.all's overlapping awaits could make a row negative and it
          // had to be suppressed before it dragged `asyncMs` down. Children are
          // subtracted as a union of contained intervals now, so a negative is
          // not merely unlikely, it is unrepresentable -- and a filter for an
          // impossible value is a filter nobody can tell is broken.
          if (!spanMs.has(label)) spanMs.set(label, []);
          spanMs.get(label)!.push(v.ms);
          spanMeta.set(label, { depth: v.depth, count: v.count, sync: v.sync });
        }

        // The same joined tree walked along `inputs` rather than `within`. Free
        // here -- no extra reps, no extra recording, just a second reading of
        // records already taken, which is the argument for the DAG being one
        // instrument with two views rather than two instruments.
        const cp = criticalPath(stage);
        cpSpanMs.push(cp.spanMs);
        for (const u of cp.unsatisfied) cpUnsatisfied.add(u);
        const cpFlat: { id: string; label: string; depth: number; durationMs: number; waitMs: number; boundBy: string | null }[] = [];
        flattenPath(cp.chain, 0, cpFlat);
        cpShapes.add(cpFlat.map((f) => `${f.depth}:${f.id}`).join('>'));
        // Overwritten per rep rather than accumulated, matching the stall
        // grouping above: the chain is structural and expected to be identical
        // every rep, and `cpShapes` is what turns that expectation into a
        // reported fact instead of an assumption.
        cpRows = cpFlat.map((f) => ({
          id: f.id, label: f.label, depth: f.depth, medianMs: 0, waitMs: 0, boundBy: f.boundBy,
        }));
        for (const f of cpFlat) {
          if (!cpDurMs.has(f.id)) { cpDurMs.set(f.id, []); cpWaitMs.set(f.id, []); }
          cpDurMs.get(f.id)!.push(f.durationMs);
          cpWaitMs.get(f.id)!.push(f.waitMs);
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
      last = pose;
    }

    // ── The discriminating experiment: ONE rep with probe mode on ──
    // Separate phase, never interleaved with the timed reps, because probing
    // adds two extra readbacks per crossing and would poison every number
    // above if it ran alongside them.
    let probe: TransferGroup[] | null = null;
    try {
      setTransferProbe(true);
      transferLedgerReset();
      await poseOnce(input, gray, w, h, backend);
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
        await poseOnce(input, gray, w, h, backend);
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
        await poseOnce(input, gray, w, h, backend);
        gpuTimeline = await gpuTimelineResolve();
      } finally {
        gpuTimelineDisarm();
      }
    }

    const poseMedianMs = median(poseMsAll);
    const pd = last?.positionDecode ?? null;

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

    // Medians filled in against the shape taken from the last rep. A row whose
    // id was on the chain in only SOME reps still gets a median -- over the reps
    // it appeared in -- which is the right answer for a row that is on the shape
    // being printed, and `varied` is what tells the reader to distrust it.
    for (const row of cpRows) {
      row.medianMs = median(cpDurMs.get(row.id) ?? []);
      row.waitMs = median(cpWaitMs.get(row.id) ?? []);
    }
    const cpSpan = median(cpSpanMs);

    const crossRows: CrossingRow[] = [...crossMs.entries()].map(([what, xs]) => {
      const m = crossMeta.get(what)!;
      return {
        what, kind: m.kind, dir: m.dir, count: m.count, bytes: m.bytes,
        medianMs: median(xs), stalls: m.kind === 'readback' ? (stallOf.get(what) ?? []) : [],
      };
    });

    return {
      reps, w, h,
      input: input.label,
      backend,
      poseMedianMs,
      poseMsAll,
      stageMedianMs: {
        votes: median(stages.votes), fit: median(stages.fit), pose: median(stages.pose),
        distance: median(stages.distance), decode: median(stages.decode),
      },
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
      spanBreakdown: {
        rows: spanRows,
        syncMs: sumWhere(true),
        asyncMs: sumWhere(false),
        foreignIds: [...foreignIds],
        straddled: [...straddled],
        crossOriginIsolated: globalThis.crossOriginIsolated === true,
      },
      criticalPath: {
        rows: cpRows,
        spanMs: cpSpan,
        sharePct: poseMedianMs > 0 ? (cpSpan / poseMedianMs) * 100 : 0,
        waitTotalMs: cpRows.reduce((a, r) => a + r.waitMs, 0),
        varied: cpShapes.size > 1,
        unsatisfied: [...cpUnsatisfied],
      },
      allocation,
      gpuTimeline,
      ok: !!pd,
      votes: last?.votes.length ?? 0,
      consistency: pd ? pd.consistency : null,
      distance: last?.recoveredAxes?.distance ?? null,
      warmupMs,
      warmedUp,
      profilerWasOn: session.mirrorWasOn,
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
    session.end();
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
  lines.push(`reconstruction: ${r.poseMedianMs.toFixed(1)}ms median / ${r.poseMinMs.toFixed(1)}ms min of ${r.reps}  (${r.w}x${r.h}, ${r.input}, backend=${r.backend})`);
  lines.push(`  compare changes on the MIN; quote the MEDIAN.`);
  lines.push(`  reps: ${r.poseMsAll.map((m) => m.toFixed(1)).join(', ')}   (IQR spread ${r.spreadPct.toFixed(0)}%)`);
  lines.push(`  warm-up (${r.warmupMs.length} reps, ${r.warmedUp ? 'settled' : 'NOT settled'}): ${r.warmupMs.map((m) => m.toFixed(1)).join(', ')}`);
  const s = r.stageMedianMs;
  lines.push(`  stages: votes ${s.votes.toFixed(1)}  fit ${s.fit.toFixed(1)}  pose ${s.pose.toFixed(1)}  distance ${s.distance.toFixed(1)}  decode ${s.decode.toFixed(1)}`);
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
    const vb = r.spanBreakdown;
    // Percentages are against the WHOLE reconstruction now, not against the
    // votes stage: the tree is rooted at `pose.run`, so a row's share of the
    // votes stage would be meaningless for the rows outside it.
    const s = r.poseMedianMs;
    if (vb.rows.length === 0) {
      lines.push(`  NO SPAN BREAKDOWN -- the 'pose.run' record did not join (see profiler.ts)`);
    } else {
      lines.push(`  reconstruction by span SELF time (median over reps; indent = declared nesting):`);
      for (const row of vb.rows) {
        const pct = s > 0 ? (row.medianMs / s) * 100 : 0;
        // A parent's self time is the time inside it that no child accounts for
        // -- which for `pose.votes` and `pose.composites` IS the unattributed
        // time the whole exercise is about, so it is labelled rather than left
        // to be inferred from the indentation.
        const kind = row.sync ? 'SYNC (host CPU)' : 'awaits (fence+kernel+rAF)';
        const tail = row.count > 1 ? `  x${row.count}` : '';
        lines.push(`    ${'  '.repeat(row.depth)}${(row.label + ' [self]').padEnd(34 - row.depth * 2)}`
          + ` ${row.medianMs.toFixed(2).padStart(6)}ms ${pct.toFixed(0).padStart(3)}%  ${kind}${tail}`);
      }
      lines.push(`    sync total ${vb.syncMs.toFixed(2)}ms (real host CPU) | awaiting total ${vb.asyncMs.toFixed(2)}ms (mixed, read as a sum only)`);
      if (vb.straddled.length) {
        lines.push(`    !! ${vb.straddled.length} stage(s) STRADDLED their declared parent: ${vb.straddled.join(', ')}`);
        lines.push(`       Not a recorder artifact -- either pose/timing/stages.ts declares the wrong \`within\``);
        lines.push(`       for these, or a span is unbalanced. Their time is missing from the rows above.`);
      }
    }
    if (vb.foreignIds.length) {
      lines.push(`    note: ${vb.foreignIds.length} id(s) recorded from outside the pose table: ${vb.foreignIds.join(', ')}`);
      lines.push(`          They did NOT join, so no cost of theirs is inside the rows above -- but the machine`);
      lines.push(`          was not quiet, so treat the wall-clock median with more suspicion than usual.`);
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
  // Straight after the decomposition, because it is the same rows read for
  // leverage rather than for cost: what to make faster, not where time went.
  {
    const cp = r.criticalPath;
    if (cp.rows.length === 0) {
      lines.push(`  NO CRITICAL PATH -- no stage on the chain declared an input that recorded (see pose/timing/stages.ts)`);
    } else {
      lines.push(`  critical path (the DEPENDENT chain, joined on \`inputs\` not \`within\`):`
        + ` ${cp.spanMs.toFixed(2)}ms = ${cp.sharePct.toFixed(0)}% of the median,`
        + ` ${cp.waitTotalMs.toFixed(2)}ms of it WAITING`);
      for (const row of cp.rows) {
        // The wait column is on the EDGE into this row, so it is blank for a
        // head. Printed with the producer's id because "waited 4.2ms" without
        // naming what it waited on is the form the old reports already had, in
        // the enclosing stage's self time, and it was not actionable.
        const wait = row.boundBy === null
          ? ''
          : `  wait ${row.waitMs.toFixed(2)}ms on ${row.boundBy}`;
        lines.push(`    ${'  '.repeat(row.depth)}${row.label.padEnd(36 - row.depth * 2)}`
          + ` ${row.medianMs.toFixed(2).padStart(6)}ms${wait}`);
      }
      // Spelled out rather than left to be inferred from a percentage, because
      // the two readings lead to opposite work.
      lines.push(cp.sharePct >= 90
        ? `    ${cp.sharePct.toFixed(0)}% in series: there is no slack to overlap here. The only lever is making a stage cheaper.`
        : `    ${cp.sharePct.toFixed(0)}% in series: ${(100 - cp.sharePct).toFixed(0)}% of the median is NOT on the chain -- either genuine slack or unmeasured work.`);
      if (cp.varied) {
        lines.push(`    !! the chain's SHAPE differed between reps -- the rows above are one rep's. Two stages`);
        lines.push(`       end close enough together that jitter decides which one binds, so there IS slack there.`);
      }
      if (cp.unsatisfied.length) {
        lines.push(`    !! ${cp.unsatisfied.length} edge(s) whose input had not FINISHED when the consumer started:`);
        lines.push(`       ${cp.unsatisfied.join(', ')}. Either \`inputs\` is wrong in pose/timing/stages.ts,`);
        lines.push(`       or those stages genuinely overlap and the edge is not a dependency.`);
      }
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
