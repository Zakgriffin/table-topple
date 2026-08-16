import { type Span, spanIngest } from './profiler.ts';
import type { GpuFrameTiming } from '../../pose/pose.ts';

// ── THE CLOCK BOUNDARY ────────────────────────────────────────────────────
//
// **This file is the only place in the project that knows there is more than
// one clock.** Everything upstream of it forwards raw foreign timestamps;
// everything downstream reads host-clock milliseconds and needs no special
// case. That is the entire reason it exists as its own file rather than as a
// few lines inside the paths that happen to cross a boundary.
//
// Two boundaries live here, and they are the same problem twice:
//
//   GPU  -- a device counter, nanoseconds, unknown offset to the host clock.
//   PEER -- the phone's `nowMs()`, epoch milliseconds, offset by NTP skew.
//
// Both know exactly how long something took and only approximately when.
//
// ── THE PROBLEM ──
//
// A GPU timestamp counter and `performance.now()` are different clocks with no
// defined relationship, and WebGPU exposes no API to correlate them. So:
//
//   DURATIONS are trustworthy      -- both counters tick real time.
//   ABSOLUTE POSITIONS are not     -- the offset between them is unknown.
//
// ── THE RESOLUTION: one offset, applied once ──
//
// The submit is bracketed on the host (`submittedAt`, `resolvedAt`), and the
// whole GPU block provably lies inside that window: a submit precedes execution,
// and the map cannot resolve before the work completes. Anchoring the frame's
// first GPU timestamp to `submittedAt` therefore places the block at the
// EARLIEST it could have run.
//
// No scaling, only an offset. Both counters tick real time, and even 100 ppm of
// relative drift is ~1.3 us across a 13 ms reconstruction -- far under the noise
// these numbers already carry. Fitting a scale would be modelling precision we
// do not have.
//
// ── WHAT THIS GUARANTEES, AND WHAT IT DOES NOT ──
//
// Guaranteed: every pass's DURATION is exact to the counter's tick, and the
// ORDER and RELATIVE SPACING of passes within the frame are exact.
//
// Not guaranteed: the absolute placement. Anchoring at `submittedAt` asserts the
// GPU began the instant the queue received the work, which is a LOWER BOUND, not
// a measurement. Some of the window is queue latency before the first pass and
// some is fence and map overhead after the last, and nothing available to us can
// say how it splits.
//
// So the leftover is not hidden -- it gets a row of its own. See UNATTRIBUTED.

/** The id every GPU pass span is filed under, prefixed with its stage. */
const GPU_PREFIX = 'gpu:';

/**
 * The part of the submit window that no pass accounts for.
 *
 * Queue latency before the first pass, plus fence and map overhead after the
 * last. For pose -- one submit, one fence -- this IS the cost of crossing the
 * bus, which makes it one of the more interesting rows rather than a rounding
 * error.
 *
 * It is `clock: 'host'` because BOTH its endpoints are host-measured, even
 * though its start is derived from a GPU total. Giving it a row is what keeps
 * the decomposition honest: the GPU rows plus this row account for the whole
 * window, with nothing swept into the parent's self time.
 */
export const GPU_UNATTRIBUTED_ID = 'pose.gpu.unattributed';

/**
 * Translate one frame's device timings onto the host clock and file them under
 * the span that submitted them.
 *
 * `parentId` is the host span the passes belong to, declared per-occurrence --
 * a GPU span's owner is a fact about the call, and the join accepts it without
 * the ids needing to exist in any stage table.
 */
export function ingestGpuFrame(gpu: GpuFrameTiming, parentId: string): void {
  if (gpu.passes.length === 0) return;

  // ns -> ms, applied to offsets that are already relative to the frame's first
  // timestamp. The anchor is the submit.
  const at = (ns: number): number => gpu.submittedAt + ns * 1e-6;

  let lastEndNs = 0;
  for (const p of gpu.passes) {
    const endNs = p.startNs + p.ns;
    if (endNs > lastEndNs) lastEndNs = endNs;
    const span: Span = {
      id: `${GPU_PREFIX}${p.stage}`,
      start: at(p.startNs),
      end: at(endNs),
      clock: 'gpu',
      // The pass's position in the submit, kept as an attribute rather than in
      // the id: grow's stages repeat up to 32 times a frame, and folding the
      // occurrence into the id would fragment them into 32 unaggregatable rows.
      attrs: { index: p.index },
      within: parentId,
    };
    spanIngest(span);
  }

  // Clamped at the window's end. The anchor is a lower bound, so a GPU block
  // whose measured length exceeds the host window means the queue latency this
  // row is meant to hold was real and the anchor consumed it -- the honest
  // report is then zero unattributed, never a negative span.
  const gpuEnd = Math.min(at(lastEndNs), gpu.resolvedAt);
  spanIngest({
    id: GPU_UNATTRIBUTED_ID,
    start: gpuEnd,
    end: gpu.resolvedAt,
    clock: 'host',
    attrs: null,
    within: parentId,
  });
}

// ── THE PEER BOUNDARY: the phone ──────────────────────────────────────────
//
// The phone stamps each frame on ITS clock, as epoch milliseconds
// (`performance.timeOrigin + performance.now()`, see clock.ts). The store holds
// desktop `performance.now()` values, which are milliseconds since THIS page's
// timeOrigin. So an epoch stamp is ~1.7e12 where the store holds ~1e5, and
// putting one in untranslated would join nothing and would break the DevTools
// mirror outright.
//
// The translation is exact arithmetic and inexact physics: subtracting our own
// timeOrigin converts the units perfectly, and leaves behind precisely the
// cross-device skew that no arithmetic can remove.
const fromPeerEpoch = (epochMs: number): number => epochMs - performance.timeOrigin;

/**
 * The phone's stamps for one delivered frame, all on the PHONE's clock.
 * `receivedAt` is the odd one out and is deliberately not here -- see below.
 */
export interface LinkStamps {
  /** The desktop asked for a frame (relayed; phone clock). */
  readonly sentAt: number;
  /** The phone had the video frame on a canvas. */
  readonly pulledAt: number;
  /** The phone finished encoding it to JPEG. */
  readonly encodedAt: number;
}

/**
 * File the phone-side half of a frame's journey as `peer` spans.
 *
 * ── WHY THERE IS NO `link.transit` SPAN, THOUGH §7 ASKED FOR ONE ──
 *
 * Transit would be `receivedAt` (DESKTOP clock) minus `encodedAt` (PHONE
 * clock), and on this pair of machines that is **about -38 ms** -- a negative
 * network time. It is not a duration at all; it is a measurement of the clock
 * SKEW between the two devices, and it was already documented as such in
 * capture.ts before any of this existed.
 *
 * A span cannot hold it. `end < start` breaks containment and would make a
 * union of children exceed its parent, which is the exact class of defect the
 * flat store was built to eliminate -- and unlike the GPU anchor, there is no
 * bound to fall back on, because the sign itself is wrong.
 *
 * So the honest report is no row. Recovering real transit needs the offset
 * solved for first, and until something does that, an omitted row beats a
 * confidently negative one. The raw stamps all survive on
 * `camera.lastCaptureTiming`, which is where the IMU work reads them.
 *
 * The two spans below are safe precisely because both of their endpoints are on
 * the PHONE's clock, so the skew cancels: what they measure is real regardless
 * of how far apart the two machines think they are.
 */
export function ingestLinkSpans(s: LinkStamps): void {
  const span = (id: string, from: number, to: number): void => {
    spanIngest({
      id, start: fromPeerEpoch(from), end: fromPeerEpoch(to),
      clock: 'peer', attrs: null, within: null,
    });
  };
  span('link.pull', s.sentAt, s.pulledAt);
  span('link.encode', s.pulledAt, s.encodedAt);
}
