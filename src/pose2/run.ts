import { type BoardData, buildBoard, uploadBoard } from './board.ts';
import { type Buffers, type PoolPlan, bufferBytes, createBuffers, planPool } from './buffers.ts';
import type { Dims } from './pipeline.ts';
import {
  type CollectSettings, type GppSettings, type GrowSettings, type LayoutSettings,
  type LineSettings, type LsdFitSettings, type Pose2Result, type VoteSettings,
  decodePose, encodeCollect, encodeDecodeBuild, encodeDecodeLayout, encodeDecodeTally,
  encodeFinish, encodeFit, encodeGpp, encodeGradient, encodeGrow, encodeLines,
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

// Every mapped region starts 8-aligned, so a host-side view of any width can be
// taken over it directly. Buffer sizes are all multiples of 4 already; this is
// about where the NEXT region begins, not about the copy itself.
const align8 = (n: number): number => (n + 7) & ~7;

export interface Pose2Settings {
  grow: GrowSettings;
  collect: CollectSettings;
  lsdFit: LsdFitSettings;
  lines: LineSettings;
  votes: VoteSettings;
  gpp: GppSettings;
  layout: LayoutSettings;
  /** The De Bruijn window size. 5, and the board is built around it. */
  order: number;
}

/**
 * Everything that outlives a frame: the buffers, the plan, and the board.
 *
 * Allocated once per (device, dims) and reused, which is what makes every buffer
 * start each frame holding the PREVIOUS frame's bytes -- the hazard §16 exists
 * for, and the reason the tests share one buffer set rather than taking a fresh
 * one each time.
 */
export interface Pose2Context {
  device: GPUDevice;
  dims: Dims;
  plan: PoolPlan;
  bufs: Buffers;
  board: BoardData;
  staging: GPUBuffer;
  /**
   * Declared-inspectable buffer -> its byte size. The per-frame request is
   * looked up here, so asking for something the plan did not promise is a throw
   * rather than a slot holding some other stage's bytes.
   */
  inspectable: Map<string, number>;
}

/** One frame's output: the pose, and whatever was asked to ride back with it. */
export interface Pose2Frame {
  pose: Pose2Result;
  /**
   * Raw bytes per inspected buffer, by name. Empty unless this frame asked.
   *
   * BYTES, not typed arrays: which of f32/u32/i32 a buffer is, and what its
   * fields mean, is the caller's business. A pipeline that handed back
   * `Float32Array` would be taking a position on how its own intermediates are
   * displayed, which is the coupling §22 keeps out.
   */
  inspected: Record<string, ArrayBuffer>;
}

export interface Pose2Options {
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
   * up as a difference. `tests/pose2Stages.test.ts` does it on one frame; the
   * sweep does it across the pose range with `--alias`.
   */
  alias?: boolean;
  /**
   * Buffers this context may be asked to read back after a frame -- the whole
   * catalogue, not the per-frame selection. See PlanOptions.inspect for what it
   * does to the pooling, and `runPose2` for the per-frame half.
   *
   * A CATALOGUE rather than a per-frame list because it sizes the staging
   * buffer, and re-sizing that is an allocation. Sphere Lab declares every
   * buffer any overlay can draw, once, and then asks per capture for whichever
   * toggles are actually on -- so flipping a toggle costs bytes on the next
   * frame and never a reallocation.
   */
  inspect?: readonly string[];
}

export function createPose2Context(
  device: GPUDevice, dims: Dims, opts: Pose2Options = {},
): Pose2Context {
  const inspect = opts.inspect ?? [];
  const plan = planPool(dims, { alias: opts.alias ?? false, inspect });
  const bufs = createBuffers(device, plan);
  const board = buildBoard(dims);
  uploadBoard(device, bufs, board);

  const inspectable = new Map(inspect.map((name) => [name, bufferBytes(dims, name)]));
  // Sized for the whole catalogue at once, even though a frame that asks for
  // nothing maps only the first 128 bytes. Worst-case sizing is what buys "no
  // frame allocates"; the cost is host-visible bytes in the one app whose
  // purpose is looking at them.
  const stagingBytes = [...inspectable.values()].reduce((a, b) => a + align8(b), POSE_BYTES);

  return {
    device, dims, plan, bufs, board, inspectable,
    // One staging buffer, kept, because it is mapped and unmapped every frame
    // and creating it per call is an allocation on the one path that is supposed
    // to have none.
    staging: device.createBuffer({
      size: stagingBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'pose readback',
    }),
  };
}

export function destroyPose2Context(ctx: Pose2Context): void {
  ctx.staging.destroy();
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
export async function runPose2(
  ctx: Pose2Context, gray: Float32Array, s: Pose2Settings, inspect: readonly string[] = [],
): Promise<Pose2Frame> {
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
        `runPose2 was asked to inspect '${name}', which this context did not declare. ` +
        `Declared: [${[...ctx.inspectable.keys()].join(', ') || 'none'}]. The declaration is what ` +
        `sizes the staging buffer and (under alias) keeps the slot exclusive, so it cannot be widened per frame.`);
    }
    regions.push({ name, offset: used, bytes });
    used += align8(bytes);
  }

  // THE ONE UPLOAD.
  device.queue.writeBuffer(bufs.gray!, 0, gray);

  const c = makeCtx(device, plan, bufs, dims);
  encodeGradient(c);
  encodeGrow(c, s.grow);
  encodeCollect(c, s.collect);
  encodeLsdFit(c, s.lsdFit);
  encodeLines(c, s.lines);
  encodeVotes(c, s.votes);
  encodeFit(c);
  encodeGpp(c, s.gpp);
  encodeDecodeLayout(c, s.layout);
  encodeDecodeBuild(c);
  encodeDecodeTally(c, { order: s.order });
  encodeFinish(c);
  // THE ONE READBACK, copied inside the same encoder so there is one submit too.
  // The inspected buffers join it here, in that same encoder: more bytes across
  // the bus, still one submit and still one fence.
  c.enc.copyBufferToBuffer(bufs.pose!, 0, ctx.staging, 0, POSE_BYTES);
  for (const r of regions) c.enc.copyBufferToBuffer(bufs[r.name]!, 0, ctx.staging, r.offset, r.bytes);
  device.queue.submit([c.enc.finish()]);

  // Only the prefix actually written -- `used` is POSE_BYTES when nothing was
  // asked for, which is byte for byte the frame this function ran before
  // inspection existed.
  await ctx.staging.mapAsync(GPUMapMode.READ, 0, used);
  // Copied OUT of the mapped range: it stops being valid at unmap.
  const mapped = ctx.staging.getMappedRange(0, used);
  const block = mapped.slice(0, POSE_BYTES);
  const inspected: Record<string, ArrayBuffer> = {};
  for (const r of regions) inspected[r.name] = mapped.slice(r.offset, r.offset + r.bytes);
  ctx.staging.unmap();
  return { pose: decodePose(block), inspected };
}
