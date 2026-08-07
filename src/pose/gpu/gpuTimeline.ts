import { createTimestampQuerySet, supportsTimestampQuery } from './device.ts';

// ── Per-pass GPU kernel timing for a whole reconstruction ────────────────
//
// The instrument perf-TODO item 7b is blocked on, and the reason it did not
// already exist: `votes` is 10.8ms of a 15.0ms reconstruction and NOTHING can
// see inside it. The transfer ledger measures the bus, the allocation probe
// measures buffer churn, and the CPU spans measure how long the host waited --
// but a readback blocks until everything queued has executed, so "GPU compute
// queued ahead" is a single 10.6ms lump with no attribution at all.
//
// ── THE one instrument for device time, and why the other one is gone ──
//
// device.ts used to carry a second implementation (`resolveTimestamps`) that
// four modules drove per-module: each allocated its own query set, resolved it,
// and awaited its own mapAsync -- so measuring cost a submit and a fence PER
// MODULE, and the number that came back was inflated by the very stall it
// added. fitPlanes reported `GPU dispatch (ATA reduction) 9.3ms` around a
// kernel that took 0.07ms. Two orders of magnitude, not a small error.
//
// It is deleted, and its two live users (fitPlanes, decodeTally) write into
// this timeline instead -- the other two had no callers at all and went with
// it. So there is now ONE query set for the whole pipeline, every pass writing
// into its own slot, ONE resolveQuerySet and ONE readback at the end. Measuring
// costs one fence total rather than one per module, and that fence lands after
// every kernel it is measuring rather than in the middle of them.
//
// The coverage that buys: `fit:ATA` and `tally:orient` now appear in the same
// table as the LSD chain, on the same clock, in the same run. Previously the
// two halves of the pipeline were measured by different instruments and could
// not be put in one ranking at all.
//
// ── Aggregation by label is the point, not a convenience ──
//
// growRegions runs ~16 rounds of hook+compress, so a flat list would be 32 rows
// saying `hook`. What the item actually asks is "where do the 10.8ms go", and
// the answer wants `hook x16, 3.2ms total` -- so slots are labelled, repeated
// labels are summed, and the count comes back alongside. A per-round breakdown
// would be a different question (does round 12 cost what round 2 does), and if
// it is ever wanted the raw slots are still there to expose.
//
// ── THE RESOLUTION LIMIT, which is not optional reading ──
//
// Chrome QUANTIZES timestamp query results, as an anti-fingerprinting measure.
// On this machine the quantum is ~65.5us: the first real run reported 0.066,
// 0.131, 0.197, 0.262 and 0.655ms and nothing in between -- every value an exact
// multiple of 65.5us. So:
//
//   - a pass reported at 0.066ms took SOMEWHERE IN (0, 0.13]ms. It is one
//     quantum, which is the instrument saying "too small to measure", not
//     "66 microseconds".
//   - an aggregate over many passes is much better than any single one, because
//     the quantization error is roughly zero-mean across repeats. `grow:compress
//     x16 = 0.85ms` is a usable number; any ONE of those 16 is not.
//   - a total is trustworthy, a single small row is not. Rank by totalMs and
//     ignore the difference between two rows that are both a quantum or two.
//
// Do not chase a 0.066ms row. Do not report one as a measurement.
//
// ── What this is NOT ──
//
// Not the in-repo profiler. profiler.ts's spans are host wall-clock and stay
// useful for structure; this measures device execution. They disagree BY
// CONSTRUCTION under a pipelined queue -- a pass whose CPU span is 0.1ms can be
// 4ms of GPU work that the host simply did not wait for -- and the disagreement
// is the finding, not an error in either.
//
// That is also why the two are never merged into one number. An earlier design
// laid GPU durations into the span tree at synthetic positions; it made the
// flamechart readable and the arithmetic wrong, since device time and host time
// overlap and subtracting one from the other double-counts. Read them as two
// tables side by side, which is what formatReconstructionTiming prints.

interface Armed {
  device: GPUDevice;
  querySet: GPUQuerySet;
  labels: string[]; // one per PAIR, index i covers slots 2i and 2i+1
  pairCap: number;
  overflowed: boolean;
}

let armed: Armed | null = null;


// `pairCap` is generous rather than computed, because the pass count is
// data-dependent (growRegions runs however many rounds it takes to converge) and
// a query set cannot grow once a command encoder is using it. 256 pairs against
// the ~50 passes a reconstruction actually runs; a timestamp query set is a few
// KB, so the slack costs nothing worth measuring.
//
// Returns false when the device never got the 'timestamp-query' feature, which
// is optional and not universally present -- callers should report "unavailable"
// rather than reporting zeros as though they were measurements.
export function gpuTimelineArm(device: GPUDevice, pairCap = 256): boolean {
  gpuTimelineDisarm();
  if (!supportsTimestampQuery(device)) return false;
  armed = { device, querySet: createTimestampQuerySet(device, pairCap), labels: [], pairCap, overflowed: false };
  return true;
}

export function gpuTimelineDisarm(): void {
  armed?.querySet.destroy();
  armed = null;
}

// Claims a slot pair and returns the compute-pass descriptor that writes into
// it, or undefined when nothing is armed -- so a call site reads
// `encoder.beginComputePass(gpuTimelineSlot('collect:scatter'))` and costs
// exactly nothing when the timeline is off. That is the property that lets these
// live on the production path rather than behind a branch at every pass.
export function gpuTimelineSlot(label: string): GPUComputePassDescriptor | undefined {
  if (!armed) return undefined;
  const i = armed.labels.length;
  // Silently dropping passes past the cap would under-report a total that reads
  // as a finding, so the overflow is recorded and surfaced in the result.
  if (i >= armed.pairCap) { armed.overflowed = true; return undefined; }
  armed.labels.push(label);
  return { timestampWrites: { querySet: armed.querySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 } };
}

interface GpuTimelineEntry {
  label: string;
  count: number;   // how many passes carried this label
  totalMs: number;
  maxMs: number;   // the slowest single pass under this label
}

export interface GpuTimelineResult {
  entries: GpuTimelineEntry[]; // descending by totalMs -- the actionable order
  passes: number;
  totalMs: number;
  overflowed: boolean;
}

// ONE resolve, ONE readback, for every pass since arming. Call it after the work
// being measured has been submitted; it does not need the queue to be idle,
// because mapAsync already waits for the resolve it just enqueued.
export async function gpuTimelineResolve(): Promise<GpuTimelineResult | null> {
  if (!armed) return null;
  const { device, querySet, labels, overflowed } = armed;
  if (labels.length === 0) return { entries: [], passes: 0, totalMs: 0, overflowed };

  const count = labels.length * 2;
  const resolveBuf = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.resolveQuerySet(querySet, 0, count, resolveBuf, 0);
  encoder.copyBufferToBuffer(resolveBuf, 0, staging, 0, count * 8);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const raw = new BigInt64Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  resolveBuf.destroy();

  const byLabel = new Map<string, GpuTimelineEntry>();
  let totalMs = 0;
  for (let i = 0; i < labels.length; i++) {
    // ns -> ms. A pass that never executed (an indirect dispatch of zero
    // workgroups, say) reports begin == end rather than garbage, so it lands as
    // a legitimate 0 and still counts as a pass -- which is the honest reading:
    // it was encoded and dispatched, it just had nothing to do.
    const ms = Number(raw[i * 2 + 1] - raw[i * 2]) / 1e6;
    totalMs += ms;
    const e = byLabel.get(labels[i]);
    if (e) { e.count++; e.totalMs += ms; e.maxMs = Math.max(e.maxMs, ms); }
    else byLabel.set(labels[i], { label: labels[i], count: 1, totalMs: ms, maxMs: ms });
  }
  return {
    entries: [...byLabel.values()].sort((a, b) => b.totalMs - a.totalMs),
    passes: labels.length,
    totalMs,
    overflowed,
  };
}
