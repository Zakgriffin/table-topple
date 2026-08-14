import { type BoardData, buildBoard, uploadBoard } from './board.ts';
import { type Buffers, type PoolPlan, createBuffers, planPool } from './buffers.ts';
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
}

export function createPose2Context(
  device: GPUDevice, dims: Dims, opts: Pose2Options = {},
): Pose2Context {
  const plan = planPool(dims, { alias: opts.alias ?? false });
  const bufs = createBuffers(device, plan);
  const board = buildBoard(dims);
  uploadBoard(device, bufs, board);
  return {
    device, dims, plan, bufs, board,
    // One staging buffer, kept, because it is mapped and unmapped every frame
    // and creating it per call is an allocation on the one path that is supposed
    // to have none.
    staging: device.createBuffer({
      size: 128, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'pose readback',
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
  ctx: Pose2Context, gray: Float32Array, s: Pose2Settings,
): Promise<Pose2Result> {
  const { device, plan, bufs, dims } = ctx;
  if (gray.length !== dims.w * dims.h) {
    throw new Error(`gray is ${gray.length} samples, expected ${dims.w * dims.h}`);
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
  c.enc.copyBufferToBuffer(bufs.pose!, 0, ctx.staging, 0, 128);
  device.queue.submit([c.enc.finish()]);

  await ctx.staging.mapAsync(GPUMapMode.READ);
  // Copied OUT of the mapped range: it stops being valid at unmap.
  const block = ctx.staging.getMappedRange().slice(0);
  ctx.staging.unmap();
  return decodePose(block);
}
