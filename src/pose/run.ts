import { canTimestamp } from '../gpu/device.ts';
import { type Board, buildBoard, uploadBoard } from './board.ts';
import { type Buffers, type PoolPlan, bufferBytes, createBuffers, planPool } from './buffers.ts';
import type { Dims } from './pipeline.ts';
import {
  type CollectSettings, type GppSettings, type GrowSettings, type LayoutSettings,
  type GpuFrameTiming, type JoinSettings, type LineSettings, type LsdFitSettings,
  type PassTimer, type PassTiming, type PoseResult, type VoteSettings,
  decodePose, encodeCollect, encodeDecodeBuild, encodeDecodeLayout, encodeDecodeTally,
  encodeFinish, encodeFit, encodeGpp, encodeGradient, encodeGrow, encodeJoin, encodeLines,
  encodeLsdFit, encodeVotes, makeCtx,
} from './pose.ts';

// ── The whole pipeline, as one call ───────────────────────────────────────
//
// A grayscale image goes up, 128 bytes come back, and nothing crosses the bus in
// between. This file is where that sentence is actually true: it holds the
// device lifecycle, the one upload and the one readback, so pose.ts keeps its
// own invariant -- no stage allocates, submits, awaits or reads back.
//
// ── WHY THE READBACK IS HERE AND NOT IN pose.ts ──
//
// Rule 1 is "one upload, one readback", not "no readback". Keeping the single
// fence in its own file makes the count checkable by reading one screen rather
// than by grepping fourteen encode functions -- and it is the number the whole
// design was chosen to protect.
//
// ── INSPECTION KEEPS THAT NUMBER AT ONE ──
//
// The display path (§22) wants buffers the pose path has no use for: fx/fy for
// the gradient rasters, the region CSR for the LSD views, `triad`/`layout` for
// the recovered axes. They ride in the SAME staging buffer as the pose, copied
// in the SAME encoder, behind the SAME fence -- so inspecting costs bytes and
// never a submit, a fence or an await.
//
// That is what makes it safe to publish the pose and its intermediates
// together, rather than deferring the display reads to a second round trip the
// way the old pipeline had to. There is nothing to defer: the pose is not known
// until this map resolves either.

/** The pose block, and the one part of the staging buffer that is never optional. */
const POSE_BYTES = 128;

// ── GPU pass timing ───────────────────────────────────────────────────────
//
// Two queries per pass, resolved into a buffer, copied into THE SAME staging
// buffer inside THE SAME encoder, before THE SAME submit. That is the whole
// design, and it is the argument that made inspection cheap reused verbatim:
// timestamps cost bytes, never a round trip. So timing is on whenever the
// device can do it -- there is no per-frame toggle because there is nothing a
// toggle would save.
//
// ── WHY A CAPACITY CONSTANT AND NOT A DERIVED COUNT ──
//
// The query set is allocated once per context, so its size is needed before any
// pass is encoded -- and the encoded pass count is NOT `plan.stages.length`.
// grow's hook/compress/gate are re-encoded once per convergence round, all 32 of
// them every frame (rounds past the fixpoint dispatch zero workgroups but are
// still real to ENCODE), so the true count is several times the number of
// declared stages.
//
// The alternative to a constant is a function that mirrors the encode structure
// to predict its own pass count, which is precisely the hand-mirrored second
// declaration this project keeps finding as a defect. So instead: one generous
// capacity, a THROW in `pass()` if a frame exceeds it, and a test that pins the
// actual count -- adding passes moves a number in a test rather than silently
// truncating a report.
//
// 512 passes is 8 KiB of query set and 8 KiB of staging tail, against a real
// count in the low hundreds.
export const MAX_TIMED_PASSES = 512;

/** 2 queries x 8 bytes (u64 nanoseconds) per pass. */
const TIMESTAMP_BYTES_PER_PASS = 16;

// Every mapped region starts 8-aligned, so a host-side view of any width can be
// taken over it directly. Buffer sizes are all multiples of 4 already; this is
// about where the NEXT region begins, not about the copy itself.
const align8 = (n: number): number => (n + 7) & ~7;

export interface PoseSettings {
  grow: GrowSettings;
  collect: CollectSettings;
  lsdFit: LsdFitSettings;
  lines: LineSettings;
  votes: VoteSettings;
  join: JoinSettings;
  gpp: GppSettings;
  layout: LayoutSettings;
}

/**
 * Everything that outlives a frame: the buffers, the plan, and the board.
 *
 * Allocated once per (device, dims) and reused, which is what makes every buffer
 * start each frame holding the PREVIOUS frame's bytes -- the hazard §16 exists
 * for, and the reason the tests share one buffer set rather than taking a fresh
 * one each time.
 */
export interface PoseContext {
  device: GPUDevice;
  dims: Dims;
  plan: PoolPlan;
  bufs: Buffers;
  /**
   * The board this context decodes against. Kept because `runPose` needs its
   * `order` -- the De Bruijn window size is a property of the board, and having
   * it ALSO in `PoseSettings` made it a second declaration that could disagree
   * with the pattern actually uploaded.
   */
  board: Board;
  staging: GPUBuffer;
  /**
   * Declared-inspectable buffer -> its byte size. The per-frame request is
   * looked up here, so asking for something the plan did not promise is a throw
   * rather than a slot holding some other stage's bytes.
   */
  inspectable: Map<string, number>;
  /**
   * The timestamp machinery, or null on a device without `timestamp-query`.
   *
   * Null is a normal state, not a degraded one: the pipeline runs identically
   * and `PoseFrame.gpu` is simply absent. Nothing downstream may require it.
   */
  timing: {
    querySet: GPUQuerySet;
    /** resolveQuerySet's destination. Cannot be the staging buffer: that one is
     *  MAP_READ, and QUERY_RESOLVE cannot be combined with it. */
    resolve: GPUBuffer;
  } | null;
}

/** One frame's output: the pose, and whatever was asked to ride back with it. */
export interface PoseFrame {
  pose: PoseResult;
  /**
   * Raw bytes per inspected buffer, by name. Empty unless this frame asked.
   *
   * BYTES, not typed arrays: which of f32/u32/i32 a buffer is, and what its
   * fields mean, is the caller's business. A pipeline that handed back
   * `Float32Array` would be taking a position on how its own intermediates are
   * displayed, which is the coupling §22 keeps out.
   */
  inspected: Record<string, ArrayBuffer>;
  /**
   * Per-pass device time, in encode order. **Absent** when the device lacks
   * `timestamp-query` -- not empty, absent, so a consumer cannot mistake "this
   * machine cannot be timed" for "every pass took zero".
   *
   * Raw nanoseconds and a stage id, and deliberately nothing else. Turning these
   * into a timeline needs the host clock they must be anchored against, which
   * this library does not have and does not ask for (§5): a GPU counter and
   * performance.now() have no defined relationship, so translating them is the
   * caller's job, done once at its own boundary.
   */
  gpu?: GpuFrameTiming;
}

export interface PoseOptions {
  /**
   * Share one allocation between buffers whose live ranges do not overlap
   * (§18). Not a second code path -- the same interval colouring, over a
   * degenerate liveness table when off.
   *
   * OFF by default. It saves ~4 MiB at 480x640 against a budget nothing is
   * hitting, and it is worth turning on when resolution goes up: at 1080p an
   * n*4B array is 7.91 MiB, so 15 arrays is 119 MiB against 71 for 9.
   *
   * **Running BOTH and comparing the poses is a free self-check**, and it is the
   * only check on this that exists -- any aliasing or clear-scheduling bug shows
   * up as a difference. `tests/poseStages.test.ts` does it on one frame; the
   * sweep does it across the pose range with `--alias`.
   */
  alias?: boolean;
  /**
   * Buffers this context may be asked to read back after a frame -- the whole
   * catalogue, not the per-frame selection. See PlanOptions.inspect for what it
   * does to the pooling, and `runPose` for the per-frame half.
   *
   * A CATALOGUE rather than a per-frame list because it sizes the staging
   * buffer, and re-sizing that is an allocation. Pose Viewer declares every
   * buffer any overlay can draw, once, and then asks per capture for whichever
   * toggles are actually on -- so flipping a toggle costs bytes on the next
   * frame and never a reallocation.
   */
  inspect?: readonly string[];
}

export function createPoseContext(
  device: GPUDevice, dims: Dims, board: Board, opts: PoseOptions = {},
): PoseContext {
  const inspect = opts.inspect ?? [];
  const plan = planPool(dims, { alias: opts.alias ?? false, inspect });
  const bufs = createBuffers(device, plan);
  // Flattened and uploaded once, then dropped: the device copy is the one that
  // matters from here, and `dims` is checked against the board on the way past.
  uploadBoard(device, bufs, buildBoard(board, dims));

  const inspectable = new Map(inspect.map((name) => [name, bufferBytes(dims, name)]));
  // Sized for the whole catalogue at once, even though a frame that asks for
  // nothing maps only the first 128 bytes. Worst-case sizing is what buys "no
  // frame allocates"; the cost is host-visible bytes in the one app whose
  // purpose is looking at them.
  const inspectBytes = [...inspectable.values()].reduce((a, b) => a + align8(b), POSE_BYTES);

  // Read the DEVICE, not the adapter: a device only has the features it was
  // created with, so an adapter that offers timestamps says nothing about a
  // device that never asked.
  const timed = canTimestamp(device);
  // The timestamp tail is sized for CAPACITY rather than for the frame's actual
  // pass count, for the same reason the inspect region is sized for the whole
  // catalogue: this buffer is allocated once and a frame must never resize it.
  const stagingBytes = inspectBytes
    + (timed ? MAX_TIMED_PASSES * TIMESTAMP_BYTES_PER_PASS : 0);

  return {
    device, dims, plan, bufs, board, inspectable,
    timing: timed
      ? {
        querySet: device.createQuerySet({
          type: 'timestamp', count: MAX_TIMED_PASSES * 2, label: 'pose pass timing',
        }),
        resolve: device.createBuffer({
          size: MAX_TIMED_PASSES * TIMESTAMP_BYTES_PER_PASS,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
          label: 'pose timestamp resolve',
        }),
      }
      : null,
    // One staging buffer, kept, because it is mapped and unmapped every frame
    // and creating it per call is an allocation on the one path that is supposed
    // to have none.
    staging: device.createBuffer({
      size: stagingBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'pose readback',
    }),
  };
}

export function destroyPoseContext(ctx: PoseContext): void {
  ctx.staging.destroy();
  ctx.timing?.querySet.destroy();
  ctx.timing?.resolve.destroy();
  for (const b of new Set(Object.values(ctx.bufs))) b.destroy();
}

/**
 * One frame: image in, pose out.
 *
 * `gray` is one f32 per pixel, 0..255, row-major. A Float32Array rather than the
 * Float64Array the old entry point took -- that narrowing loop was measured as
 * part of the byte-proportional cost per reconstruction and there is no reason
 * to pay it (§4).
 */
export async function runPose(
  ctx: PoseContext, gray: Float32Array, s: PoseSettings, inspect: readonly string[] = [],
): Promise<PoseFrame> {
  const { device, plan, bufs, dims } = ctx;
  if (gray.length !== dims.w * dims.h) {
    throw new Error(`gray is ${gray.length} samples, expected ${dims.w * dims.h}`);
  }

  // Packed fresh each frame, from the requested subset only, so a frame that
  // asks for one small buffer maps one small buffer -- a fixed offset per
  // catalogue entry would make every map reach as far as the highest-numbered
  // request. Deduped rather than rejected: asking twice is wasteful, not wrong.
  const regions: { name: string; offset: number; bytes: number }[] = [];
  let used = POSE_BYTES;
  for (const name of new Set(inspect)) {
    const bytes = ctx.inspectable.get(name);
    if (bytes === undefined) {
      throw new Error(
        `runPose was asked to inspect '${name}', which this context did not declare. ` +
        `Declared: [${[...ctx.inspectable.keys()].join(', ') || 'none'}]. The declaration is what ` +
        `sizes the staging buffer and (under alias) keeps the slot exclusive, so it cannot be widened per frame.`);
    }
    regions.push({ name, offset: used, bytes });
    used += align8(bytes);
  }

  // THE ONE UPLOAD.
  device.queue.writeBuffer(bufs.gray!, 0, gray);

  // `ids` accumulates one entry per pass as they are encoded, so its length
  // after encoding IS the pass count -- the library never predicts it.
  const timer: PassTimer | undefined = ctx.timing
    ? { querySet: ctx.timing.querySet, capacity: MAX_TIMED_PASSES, ids: [] }
    : undefined;

  const c = makeCtx(device, plan, bufs, dims, timer);
  encodeGradient(c);
  encodeGrow(c, s.grow);
  encodeCollect(c, s.collect);
  encodeLsdFit(c, s.lsdFit);
  encodeLines(c, s.lines);
  encodeVotes(c, s.votes);
  encodeJoin(c, s.join);
  encodeFit(c);
  encodeGpp(c, s.gpp);
  encodeDecodeLayout(c, s.layout);
  encodeDecodeBuild(c);
  encodeDecodeTally(c, { order: ctx.board.order });
  encodeFinish(c);
  // THE ONE READBACK, copied inside the same encoder so there is one submit too.
  // The inspected buffers join it here, in that same encoder: more bytes across
  // the bus, still one submit and still one fence.
  c.enc.copyBufferToBuffer(bufs.pose!, 0, ctx.staging, 0, POSE_BYTES);
  for (const r of regions) c.enc.copyBufferToBuffer(bufs[r.name]!, 0, ctx.staging, r.offset, r.bytes);

  // The timestamps join the same encoder, after the last pass and before the
  // same submit: resolve the queries this frame actually wrote, then copy them
  // into the staging tail. Still one submit, one fence, one map.
  //
  // Resolved through a separate buffer because QUERY_RESOLVE and MAP_READ cannot
  // be combined in one usage -- the extra hop is device-local and costs no round
  // trip, which is the only cost that would matter here.
  const tsOffset = align8(used);
  const tsBytes = timer ? timer.ids.length * TIMESTAMP_BYTES_PER_PASS : 0;
  if (timer && tsBytes > 0) {
    c.enc.resolveQuerySet(timer.querySet, 0, timer.ids.length * 2, ctx.timing!.resolve, 0);
    c.enc.copyBufferToBuffer(ctx.timing!.resolve, 0, ctx.staging, tsOffset, tsBytes);
  }
  // Stamped as tightly around the crossing as possible: everything before this
  // line is upload and encode, which a caller can bracket for itself.
  const submittedAt = performance.now();
  device.queue.submit([c.enc.finish()]);

  // Only the prefix actually written -- `used` is POSE_BYTES when nothing was
  // asked for, which is byte for byte the frame this function ran before
  // inspection existed.
  const mapBytes = tsBytes > 0 ? tsOffset + tsBytes : used;
  await ctx.staging.mapAsync(GPUMapMode.READ, 0, mapBytes);
  // The map cannot resolve before the submitted work completes, so this bounds
  // the GPU block from above exactly as submittedAt bounds it from below.
  const resolvedAt = performance.now();
  // Copied OUT of the mapped range: it stops being valid at unmap.
  const mapped = ctx.staging.getMappedRange(0, mapBytes);
  const block = mapped.slice(0, POSE_BYTES);
  const inspected: Record<string, ArrayBuffer> = {};
  for (const r of regions) inspected[r.name] = mapped.slice(r.offset, r.offset + r.bytes);
  const gpu: GpuFrameTiming | undefined = timer && tsBytes > 0
    ? {
      passes: readTimings(mapped.slice(tsOffset, tsOffset + tsBytes), timer.ids),
      submittedAt,
      resolvedAt,
    }
    : undefined;
  ctx.staging.unmap();
  return gpu ? { pose: decodePose(block), inspected, gpu } : { pose: decodePose(block), inspected };
}

/**
 * The resolved query pairs, as one duration per pass.
 *
 * The values are u64 nanoseconds on the GPU's own counter, so they are read as
 * BigInt and SUBTRACTED before narrowing -- the raw counter can be large enough
 * to lose nanosecond resolution as a double, while a difference never is.
 *
 * Zeros are passed through rather than filtered. A pass that legitimately took
 * no measurable time and a pass whose query was never written are both worth
 * seeing, and the caller can tell them apart by whether the count matches.
 */
function readTimings(bytes: ArrayBuffer, ids: readonly string[]): PassTiming[] {
  const raw = new BigUint64Array(bytes);
  // The frame's first timestamp, which every offset below is relative to. It is
  // the FIRST PASS'S begin rather than the minimum over all of them: passes
  // execute in encode order, so pass 0 begins first, and taking a minimum would
  // quietly paper over an out-of-order read instead of exposing it as a
  // negative offset.
  const origin = raw[0] ?? 0n;
  return ids.map((stage, index) => ({
    stage,
    ns: Number(raw[index * 2 + 1]! - raw[index * 2]!),
    startNs: Number(raw[index * 2]! - origin),
    index,
  }));
}
