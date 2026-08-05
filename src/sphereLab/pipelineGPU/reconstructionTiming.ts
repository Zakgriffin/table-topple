import { Camera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { captureDistortedGrayscale } from '../pipeline/capture.ts';
import { computePoseFromCapture, PoseComputeState } from '../pipeline/poseCompute.ts';
import { profilerEnabled, profilerSetEnabled } from '../profiling/profiler.ts';
import {
  setTransferProbe, TransferSample, transferLedger, transferLedgerReset,
} from './device.ts';
import { awaitPageFocus } from './lsdChainVerify.ts';

// ── Dev harness: what does ONE WHOLE RECONSTRUCTION cost? ─────────────────
//
//   await timeReconstruction()          // 5 reps, default
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

export interface TransferGroup {
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

export interface ReconstructionTimingReport {
  reps: number;
  w: number;
  h: number;
  cameraKind: 'simulated' | 'physical';

  // ── the headline ──
  poseMedianMs: number;
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

  // ── correctness, so a "fast" bail cannot masquerade as a win ──
  ok: boolean;
  votes: number;
  consistency: number | null;
  distance: number | null;

  focusWaitMs: number;
  focusedThroughout: boolean;
  profilerWasOn: boolean; // if true, this run is invalid -- see the restore below
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s.length === 0 ? 0 : s[Math.floor(s.length / 2)];
}

// One fresh state per rep. Nothing carries over between reps, which matters:
// a state reused across reps would keep the previous pose alive and let a rep
// that should have failed silently inherit the last good answer.
function freshState(camera: Camera): PoseComputeState {
  return {
    aspect: camera.aspect,
    settings: camera.settings,
    lastVoteComposites: null, lastVotes: null, lastQuadricPair: null, lastGridPeriodPhase: null,
    lastRecoveredAxes: null, lastDecodeGrid: null, lastDecodeRotated: null, lastDecodeCorrectness: null,
    lastPositionDecode: null, lastChainTransfers: null,
  };
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
  camera?: Camera | null, reps = 5,
): Promise<ReconstructionTimingReport | string> {
  camera = camera ?? activeCamera() ?? null;
  if (!camera) return 'no active camera';
  const cap = camera.lastAxesCaptureGray;
  if (!cap) return 'no capture yet -- run a capture first';
  const { gray, w, h } = cap;

  // Profiler OFF for the duration. It switches on timestamp queries whose
  // resolve costs its own mapAsync, so a profiled run and a harness run are not
  // comparable -- and leaving it on would silently make every number here worse
  // than the pipeline actually is. Reported in the result either way, because a
  // caller who had it on deserves to know their previous numbers were profiled.
  const profilerWasOn = profilerEnabled();
  profilerSetEnabled(false);

  try {
    const focusWaitMs = await awaitPageFocus();

    // One untimed warm-up: first-call shader compilation is on the order of
    // 88ms for the fused decode alone (measured, see the plan file), which
    // would dominate rep 0 and drag a mean without ever touching the median.
    // Timing the median of N would survive it; reporting poseMsAll honestly
    // would not, and the array is there to be read.
    await computePoseFromCapture(freshState(camera), gray, w, h);

    const poseMsAll: number[] = [];
    const stages = { votes: [] as number[], fit: [] as number[], pose: [] as number[], distance: [] as number[], decode: [] as number[] };
    const captureMs: number[] = [];
    let last: PoseComputeState | null = null;
    let fences = 0, transferBytes = 0, transferMs = 0;

    for (let rep = 0; rep < reps; rep++) {
      // Capture is measured but its output is DISCARDED -- the pose reps all
      // run on the same cached `gray` so that every rep is the same
      // computation. Re-capturing per rep would re-render the scene and make
      // each rep a slightly different input, which is the wrong tradeoff for a
      // timing harness even though it is closer to what the app does.
      if (isSimulated(camera)) {
        const t0 = performance.now();
        captureDistortedGrayscale(camera);
        captureMs.push(performance.now() - t0);
      }

      const state = freshState(camera);
      transferLedgerReset();
      const t0 = performance.now();
      const timing = await computePoseFromCapture(state, gray, w, h);
      poseMsAll.push(performance.now() - t0);

      const led = transferLedger();
      // Counted per rep and kept from the LAST one rather than summed: these
      // are "per reconstruction" quantities, and every rep runs the identical
      // computation, so summing would just multiply by reps.
      fences = led.filter((s) => s.kind === 'readback').length;
      transferBytes = led.reduce((a, s) => a + (s.kind === 'convert' ? 0 : s.bytes), 0);
      transferMs = led.reduce((a, s) => a + s.ms, 0);

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
      await computePoseFromCapture(freshState(camera), gray, w, h);
      probe = groupLedger(transferLedger());
    } finally {
      setTransferProbe(false);
      transferLedgerReset();
    }

    const poseMedianMs = median(poseMsAll);
    const pd = last?.lastPositionDecode ?? null;

    return {
      reps, w, h,
      cameraKind: isSimulated(camera) ? 'simulated' : 'physical',
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
      ok: !!pd,
      votes: last?.lastVotes?.length ?? 0,
      consistency: pd ? pd.consistency : null,
      distance: last?.lastRecoveredAxes?.distance ?? null,
      focusWaitMs,
      // Checked at the END as well as the start: focus can be lost mid-run, and
      // if it was, every millisecond above is garbage. Same gate verifyLsdChain
      // uses, and for the same reason -- an unfocused tab throttles timers, so
      // the numbers are WRONG rather than merely noisy.
      focusedThroughout: document.hasFocus() && document.visibilityState === 'visible',
      profilerWasOn,
    };
  } finally {
    // Restore, or the harness silently reconfigures the app it just measured.
    profilerSetEnabled(profilerWasOn);
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
  if (r.profilerWasOn) warn.push('note: profiler was on before this run; it was forced off and restored.');
  lines.push(...warn);
  lines.push(`reconstruction: ${r.poseMedianMs.toFixed(1)}ms median of ${r.reps}  (${r.w}x${r.h}, ${r.cameraKind})`);
  lines.push(`  reps: ${r.poseMsAll.map((m) => m.toFixed(1)).join(', ')}`);
  const s = r.stageMedianMs;
  lines.push(`  stages: votes ${s.votes.toFixed(1)}  fit ${s.fit.toFixed(1)}  pose ${s.pose.toFixed(1)}  distance ${s.distance.toFixed(1)}  decode ${s.decode.toFixed(1)}`);
  if (r.captureMedianMs !== null) lines.push(`  capture+preprocess: ${r.captureMedianMs.toFixed(1)}ms (SIMULATED ONLY, not in the median above)`);
  lines.push(`  fences: ${r.fences}   bytes: ${(r.transferBytes / 1048576).toFixed(2)}MB   in transfer helpers: ${r.transferMs.toFixed(1)}ms (${r.transferSharePct.toFixed(0)}% -- UPPER BOUND, includes GPU compute waited on)`);
  if (r.probe) {
    let f = 0, b = 0, q = 0;
    for (const g of r.probe) { f += g.fenceMs ?? 0; b += g.byteMs ?? (g.kind === 'readback' ? 0 : g.ms); q += g.queuedAheadMs ?? 0; }
    lines.push(`  attributed: ${q.toFixed(1)}ms GPU compute queued ahead | ${f.toFixed(1)}ms fence latency | ${b.toFixed(1)}ms byte-proportional`);
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
