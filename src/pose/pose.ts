import {
  COLLECT_FINALIZE_WGSL, COLLECT_MARKKEPT_WGSL, COLLECT_REGIONMETA_WGSL,
  COLLECT_SCATTER_WGSL, COLLECT_TALLY_WGSL, FIT_ATA_WGSL, FIT_EIGEN_WGSL,
  GPP_CLASSIFY_WGSL, GPP_COMPACT_WGSL, GPP_DISTINCT_WGSL, GPP_EXTENT_WGSL,
  GPP_PEAKS_WGSL, GPP_POLISH_WGSL, GPP_SWEEP_WGSL,
  DECODE_BINTHRESH_PARTIALS_WGSL, DECODE_BINTHRESH_REDUCE_WGSL, DECODE_LAYOUT_WGSL,
  DECODE_BUILD_WGSL, DECODE_TALLY_WGSL, DECODE_ARGMAX_WGSL, DECODE_CORRECTNESS_WGSL,
  FINISH_WGSL,
  GRADIENT_WGSL, GROW_WGSL, JOIN_ANCHOR_WGSL, JOIN_ATTACH_WGSL, JOIN_FLATTEN_WGSL,
  JOIN_INIT_WGSL, JOIN_REDUCE_WGSL, JOIN_REFIT_WGSL,
  LINES_EMIT_WGSL, LINES_FLAG_WGSL, LSD_FIT_WGSL,
  SCAN_WGSL, VOTES_WGSL,
} from './pose.wgsl.ts';
import { assertBinds, type Buffers, type PoolPlan } from './buffers.ts';
import {
  BUFFERS, GPP_HEADROOM_CAP, GPP_N_MIN, SCAN_PER, SCAN_THREADS, type Dims,
  gppCandidateCount, scanBlocks,
} from './pipeline.ts';

// ── The flat pose pipeline ────────────────────────────────────────────────
//
// One encode function per stage, all writing into one encoder, one submit, one
// readback. "Flat" means no directory tree, no residency, no arena, no backend
// dispatch and no abstraction layers -- NOT one continuous function body. The
// function boundaries are what make the file both readable and testable, and
// they cost nothing at runtime.
//
// Every stage follows the same shape and it is enforced rather than
// conventional:
//
//   - it takes (ctx), reads buffers by name off ctx.bufs, and encodes passes
//   - it does not allocate, submit, await, or read back
//   - it goes through `pass()`, which asserts the names it binds against
//     pipeline.ts and applies that stage's scheduled clears
//
// A stage that reaches for ctx.bufs directly and builds its own bind group
// bypasses both of those. Don't.

const WG2D = 8; // matches @workgroup_size(8, 8) in every 2D kernel here
export const groups2D = (w: number, h: number): [number, number] =>
  [Math.ceil(w / WG2D), Math.ceil(h / WG2D)];

/**
 * How many grow rounds get encoded.
 *
 * The zero-crossing rule costs us something specific here: with no readback the
 * host cannot ask "did it converge yet" between batches, so the round count is
 * fixed at encode time. Measured 9 at 480x640; the O(log n) bound is ~19; the
 * true worst case for pointer jumping is far higher and cannot be encoded.
 *
 * 32 is a practical margin, not a guarantee. Rounds past the fixpoint dispatch
 * ZERO workgroups (gate zeroes `args`), so they are nearly free to run -- but
 * still real to ENCODE, which is what bounds this. Non-convergence is reported
 * as a status bit rather than silently producing a half-grown labelling.
 */
export const GROW_ROUNDS = 32;

export interface Ctx {
  device: GPUDevice;
  plan: PoolPlan;
  bufs: Buffers;
  enc: GPUCommandEncoder;
  dims: Dims;
  /**
   * Bind groups, memoized per stage for the lifetime of this ctx.
   *
   * Not a micro-optimization. `encodeGrow` encodes 3 passes x 32 rounds, and
   * building a fresh bind group for each meant ~97 per reconstruction over an
   * unchanging buffer set. Two such runs against two buffer sets crashed Dawn
   * outright -- the tests went from passing individually to killing the process
   * when run back to back. Memoizing collapses grow to 2 bind groups.
   *
   * Safe because a ctx's buffers never change: it is created per encoder, and
   * `bufs` is captured at construction.
   */
  bindGroups?: Map<string, GPUBindGroup[]>;
  /**
   * GPU pass timing, when the device can do it. Absent means untimed, which is
   * the only behaviour on a device without `timestamp-query` -- and asking for a
   * timestamp there is a validation error, not a no-op, so this is a real gate
   * rather than a preference.
   */
  timing?: PassTimer;
}

/**
 * One frame's timestamp accounting, filled in as passes are encoded.
 *
 * ── THE LIBRARY TAKES NO POSITION ON WHAT THESE MEAN ──
 *
 * `src/pose` gets no profiler, no spans and no timing module (the design's §5).
 * It owns `pass()`, which is the single `beginComputePass` in the pipeline, so
 * it is the only thing that can say WHICH pass inside the submit took the time.
 * Everything else -- upload, encode, the fence, the map -- is a host-clock
 * quantity the caller can bracket from outside, and does. So this forwards raw
 * nanoseconds and stops.
 */
export interface PassTimer {
  /** Two queries per pass: one at the beginning, one at the end. */
  readonly querySet: GPUQuerySet;
  /** Capacity in PASSES. The query set holds twice this many queries. */
  readonly capacity: number;
  /**
   * The stage id of each encoded pass, in encode order -- so `ids.length` is
   * the pass count and `ids[i]` names query pair `2i`/`2i+1`.
   *
   * Ids REPEAT, and that is not a defect to fix here: grow's hook/compress/gate
   * run once per convergence round, so a frame holds up to 32 passes under each
   * of those three ids. Which occurrence is which is the `index`, and what to do
   * about the repetition (aggregate, or not) is the consumer's decision.
   */
  readonly ids: string[];
}

/** One encoded pass's measured device time. */
export interface PassTiming {
  /** The pipeline stage id. Not unique within a frame -- see PassTimer.ids. */
  readonly stage: string;
  /** Elapsed device time in nanoseconds, on the GPU's own counter. */
  readonly ns: number;
  /**
   * Nanoseconds from the frame's FIRST timestamp to this pass's beginning.
   *
   * Relative, not absolute, and deliberately: the raw counter runs to ~1.4e15,
   * which is inside a double's exact-integer range but not by much, and nothing
   * outside this frame can use the absolute value anyway -- the GPU counter has
   * no defined relationship to any host clock. A consumer places the block by
   * anchoring `startNs = 0` to the host timestamp of the submit and laying the
   * rest out from there, which is exactly what these offsets are for.
   */
  readonly startNs: number;
  /** Position in the submit, 0-based, so a consumer need not re-derive encode order. */
  readonly index: number;
}

/**
 * One frame's device timing, or absent entirely.
 *
 * One block rather than three optional fields, so "this frame was timed" is a
 * single check and a consumer cannot end up holding passes without the host
 * stamps needed to place them.
 */
export interface GpuFrameTiming {
  readonly passes: readonly PassTiming[];
  /**
   * `performance.now()` immediately before and after the submit/fence.
   *
   * These are STAMPS, not a span -- the same role a phone link's sentAt /
   * pulledAt play for a relayed frame. They exist because they are the one
   * host-clock pair a caller CANNOT take for itself: the submit and the map
   * both happen inside `runPose`, so an outside bracket measures upload and
   * encode too and would anchor the GPU block earlier than it could have run.
   *
   * The library still takes no position on what they mean -- it does not build
   * a span, subtract them, or name the difference.
   */
  readonly submittedAt: number;
  readonly resolvedAt: number;
}

/**
 * A ctx over a fresh encoder against an existing buffer set.
 *
 * `timing` is optional and omitted by every caller that is not a whole frame:
 * the stage tests encode one stage at a time and have nothing to time.
 */
export function makeCtx(
  device: GPUDevice, plan: PoolPlan, bufs: Buffers, dims: Dims, timing?: PassTimer,
): Ctx {
  return { device, plan, bufs, enc: device.createCommandEncoder(), dims, bindGroups: new Map(), timing };
}

interface Program {
  layouts: GPUBindGroupLayout[];
  pipeline: GPUComputePipeline;
}

interface Programs {
  gradient: Program;
  growInit: Program; growHook: Program; growCompress: Program; growGate: Program;
  scanBlocks: Program; scanAdd: Program;
  collectTally: Program; collectMarkKept: Program; collectRegionMeta: Program;
  collectScatter: Program; collectFinalize: Program;
  lsdFit: Program;
  linesFlag: Program; linesEmit: Program; votesCast: Program;
  joinInit: Program; joinRefit: Program; joinAnchor: Program;
  joinAttach: Program; joinFlatten: Program; joinReduce: Program;
  fitAta: Program; fitEigen: Program;
  gppClassify: Program; gppCompact: Program; gppExtent: Program;
  gppSweep: Program; gppPeaks: Program; gppDistinct: Program; gppPolish: Program;
  decodeBinThreshPartials: Program; decodeBinThreshReduce: Program; decodeLayout: Program;
  decodeBuild: Program; decodeTally: readonly Program[]; decodeArgmax: Program; decodeCorrectness: Program;
  finish: Program;
}

const RO = 'read-only-storage' as const;
const RW = 'storage' as const;
const UNI = 'uniform' as const;

/** A single-entry-point compute program over one explicit bind-group layout. */
function simpleProgram(
  device: GPUDevice, label: string, code: string, types: readonly Binding[],
  entryPoint = 'main',
): Program {
  const layout = layoutOf(device, label, types);
  return {
    layouts: [layout],
    pipeline: device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: device.createShaderModule({ code, label }), entryPoint },
    }),
  };
}

const programCache = new WeakMap<GPUDevice, Programs>();

/** decode.tally's four orientations, over one module and one explicit layout. */
function tallyPrograms(device: GPUDevice): readonly Program[] {
  const layout = layoutOf(device, 'decode.tally', [UNI, RO, RO, RO, RO, RW, RW]);
  const module = device.createShaderModule({ code: DECODE_TALLY_WGSL, label: 'decode.tally' });
  return [0, 1, 2, 3].map((o) => ({
    layouts: [layout],
    pipeline: device.createComputePipeline({
      label: `decode.tally.o${o}`,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: `o${o}` },
    }),
  }));
}

type Binding = 'uniform' | 'storage' | 'read-only-storage';

function layoutOf(device: GPUDevice, label: string, types: readonly Binding[]): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label,
    entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })),
  });
}

function programs(device: GPUDevice): Programs {
  const cached = programCache.get(device);
  if (cached) return cached;

  const gradLayout = layoutOf(device, 'gradient', ['uniform', 'read-only-storage', 'storage', 'storage']);
  const gradModule = device.createShaderModule({ code: GRADIENT_WGSL, label: 'gradient' });
  const gradient: Program = {
    layouts: [gradLayout],
    pipeline: device.createComputePipeline({
      label: 'gradient',
      layout: device.createPipelineLayout({ bindGroupLayouts: [gradLayout] }),
      compute: { module: gradModule, entryPoint: 'main' },
    }),
  };

  // ONE shared explicit layout across all four grow entry points -- see
  // GROW_WGSL's header for why `layout: 'auto'` cannot work here.
  const growLayout = layoutOf(device, 'grow', [
    'uniform', 'read-only-storage', 'read-only-storage',
    'storage', 'storage', 'storage', 'storage', 'storage',
  ]);
  const argsLayout = layoutOf(device, 'grow.args', ['storage']);
  const growModule = device.createShaderModule({ code: GROW_WGSL, label: 'grow' });
  const growPipeline = (entryPoint: string, withArgs: boolean): Program => ({
    layouts: withArgs ? [growLayout, argsLayout] : [growLayout],
    pipeline: device.createComputePipeline({
      label: `grow.${entryPoint}`,
      layout: device.createPipelineLayout({
        bindGroupLayouts: withArgs ? [growLayout, argsLayout] : [growLayout],
      }),
      compute: { module: growModule, entryPoint },
    }),
  });

  // Both scan entry points live in one module, so they share module-scope
  // bindings -- but `add` never touches `sums`, so they cannot share a LAYOUT
  // without add binding a buffer it has no use for. Two explicit layouts over
  // one module: each pipeline declares exactly what its entry point uses.
  const scanModule = device.createShaderModule({ code: SCAN_WGSL, label: 'scan' });
  const scanLayout = layoutOf(device, 'scan.blocks', [UNI, RO, RW, RW]);
  const addLayout = layoutOf(device, 'scan.add', [UNI, RO, RW]);
  const scanPipeline = (entryPoint: string, layout: GPUBindGroupLayout): Program => ({
    layouts: [layout],
    pipeline: device.createComputePipeline({
      label: `scan.${entryPoint}`,
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: scanModule, entryPoint },
    }),
  });

  const built: Programs = {
    gradient,
    growInit: growPipeline('init', false),
    growHook: growPipeline('hook', false),
    growCompress: growPipeline('compress', false),
    // gate is the only entry point that references group 1, so it is the only
    // one whose pipeline layout mentions it.
    growGate: growPipeline('gate', true),

    scanBlocks: scanPipeline('blocks', scanLayout),
    scanAdd: scanPipeline('add', addLayout),

    // Each collect layout is exactly its stage's `binds` in pipeline.ts, in
    // order. assertBinds is what keeps the two from drifting apart.
    collectTally: simpleProgram(device, 'collect.tally', COLLECT_TALLY_WGSL,
      [UNI, RO, RO, RO, RW, RW]),
    collectMarkKept: simpleProgram(device, 'collect.markKept', COLLECT_MARKKEPT_WGSL,
      [UNI, RO, RO, RW]),
    collectRegionMeta: simpleProgram(device, 'collect.regionMeta', COLLECT_REGIONMETA_WGSL,
      [UNI, RO, RO, RO, RW, RW, RW]),
    collectScatter: simpleProgram(device, 'collect.scatter', COLLECT_SCATTER_WGSL,
      [UNI, RO, RO, RO, RW, RW]),
    collectFinalize: simpleProgram(device, 'collect.finalize', COLLECT_FINALIZE_WGSL,
      [UNI, RO, RO, RO, RO, RO, RW, RW]),

    lsdFit: simpleProgram(device, 'lsdFit', LSD_FIT_WGSL,
      [UNI, RO, RO, RO, RO, RO, RO, RO, RW]),

    linesFlag: simpleProgram(device, 'lines.flag', LINES_FLAG_WGSL,
      [UNI, RO, RO, RW]),
    linesEmit: simpleProgram(device, 'lines.emit', LINES_EMIT_WGSL,
      [UNI, RO, RO, RO, RW, RO, RW, RW]),
    votesCast: simpleProgram(device, 'votes.cast', VOTES_WGSL,
      [UNI, RO, RO, RW, RW]),

    joinInit: simpleProgram(device, 'join.init', JOIN_INIT_WGSL,
      [UNI, RW]),
    joinRefit: simpleProgram(device, 'join.refit', JOIN_REFIT_WGSL,
      [UNI, RO, RO, RO, RW, RW]),
    joinAnchor: simpleProgram(device, 'join.anchor', JOIN_ANCHOR_WGSL,
      [UNI, RO, RO, RO, RW]),
    joinAttach: simpleProgram(device, 'join.attach', JOIN_ATTACH_WGSL,
      [UNI, RO, RO, RO, RO, RW]),
    joinFlatten: simpleProgram(device, 'join.flatten', JOIN_FLATTEN_WGSL,
      [UNI, RO, RO, RW, RW]),
    joinReduce: simpleProgram(device, 'join.reduce', JOIN_REDUCE_WGSL,
      [UNI, RO, RO, RO, RO, RW, RW, RW]),

    fitAta: simpleProgram(device, 'fit.ata', FIT_ATA_WGSL,
      [UNI, RO, RO, RO, RW]),
    fitEigen: simpleProgram(device, 'fit.eigen', FIT_EIGEN_WGSL,
      [RO, RW, RW, RW]),

    gppClassify: simpleProgram(device, 'gpp.classify', GPP_CLASSIFY_WGSL,
      [UNI, RO, RO, RO, RO, RW, RW]),
    gppCompact: simpleProgram(device, 'gpp.compact', GPP_COMPACT_WGSL,
      [UNI, RO, RO, RO, RW, RW]),
    gppExtent: simpleProgram(device, 'gpp.extent', GPP_EXTENT_WGSL,
      [UNI, RO, RO, RO, RW, RW]),
    gppSweep: simpleProgram(device, 'gpp.sweep', GPP_SWEEP_WGSL,
      [UNI, RO, RO, RO, RO, RW]),
    gppPeaks: simpleProgram(device, 'gpp.peaks', GPP_PEAKS_WGSL,
      [UNI, RO, RW, RW]),
    gppDistinct: simpleProgram(device, 'gpp.distinct', GPP_DISTINCT_WGSL,
      [UNI, RO, RO, RO, RO, RW]),
    gppPolish: simpleProgram(device, 'gpp.polish', GPP_POLISH_WGSL,
      [UNI, RO, RO, RO, RO, RO, RO, RW]),

    decodeBinThreshPartials: simpleProgram(device, 'decode.binThreshPartials',
      DECODE_BINTHRESH_PARTIALS_WGSL, [UNI, RO, RW]),
    decodeBinThreshReduce: simpleProgram(device, 'decode.binThreshReduce',
      DECODE_BINTHRESH_REDUCE_WGSL, [UNI, RO, RW]),
    // EXACTLY EIGHT storage buffers -- the WebGPU baseline. See pipeline.ts.
    decodeLayout: simpleProgram(device, 'decode.layout', DECODE_LAYOUT_WGSL,
      [UNI, RO, RO, RO, RO, RW, RW, RW, RW]),
    // No uniform at all: everything this pass needs is device-side now, and the
    // block it reads is the one decode.layout just wrote.
    decodeBuild: simpleProgram(device, 'decode.build', DECODE_BUILD_WGSL, [RO, RO, RW]),
    // Four entry points, one module, one shared layout -- the orientation is a
    // compile-time constant in each. Same shape as grow's four.
    decodeTally: tallyPrograms(device),
    decodeArgmax: simpleProgram(device, 'decode.argmax', DECODE_ARGMAX_WGSL,
      [UNI, RO, RO, RO, RW, RW]),
    decodeCorrectness: simpleProgram(device, 'decode.correctness', DECODE_CORRECTNESS_WGSL,
      [UNI, RO, RO, RO, RW]),
    finish: simpleProgram(device, 'finish', FINISH_WGSL, [UNI, RO, RO, RO, RO, RO, RO, RW, RW]),
  };
  programCache.set(device, built);
  return built;
}

type Dispatch =
  | { kind: 'direct'; x: number; y?: number }
  // `offset` is bytes into the args buffer. Only decode's tally uses it: one
  // buffer holds four triples, one per orientation, because they are written by
  // a single pass and a buffer per orientation would be four clears and four
  // declaration entries for twelve u32.
  | { kind: 'indirect'; args: string; offset?: number };

/**
 * Encode one pass.
 *
 * Does three things beyond the obvious, all of which exist because skipping
 * them fails silently rather than loudly:
 *
 *  1. Applies this stage's scheduled clears first. The schedule is DERIVED from
 *     liveness, so with pooling a clear lands immediately before the buffer's
 *     first bind -- a frame-start clear would be writing over whoever still owns
 *     that slot.
 *  2. Calls assertBinds, so the names bound here and the names pipeline.ts
 *     declares cannot drift. Drift is what would let two co-bound buffers share
 *     a pool slot.
 *  3. Takes bind groups as an array of arrays, because a stage with an indirect
 *     args buffer keeps it in its own group -- see GROW_WGSL.
 */
/**
 * The compute-pass descriptor for one pass, with timestamp writes when this
 * frame is being timed.
 *
 * Claims the next pass slot as a side effect, so it must be called exactly once
 * per pass and at the point the pass is actually begun -- the index it hands out
 * IS the encode order the caller reads back.
 */
function timingFor(ctx: Ctx, stageId: string): GPUComputePassDescriptor {
  const t = ctx.timing;
  if (!t) return { label: stageId };
  const index = t.ids.length;
  if (index >= t.capacity) {
    // A throw, not a silent stop. Running past the query set would otherwise
    // report a frame whose last passes are missing while looking complete, and
    // a timing report that quietly omits its tail is worse than no report.
    throw new Error(
      `pass: this frame encoded more than ${t.capacity} passes, which is the timestamp ` +
      `query set's capacity (MAX_TIMED_PASSES in run.ts). Raise it to match the pipeline.`);
  }
  t.ids.push(stageId);
  return {
    label: stageId,
    timestampWrites: {
      querySet: t.querySet,
      beginningOfPassWriteIndex: index * 2,
      endOfPassWriteIndex: index * 2 + 1,
    },
  };
}

export function pass(
  ctx: Ctx, stageId: string, program: Program,
  bindGroups: readonly (readonly string[])[], dispatch: Dispatch,
): void {
  const { device, plan, bufs, enc } = ctx;

  const stageIndex = plan.stages.findIndex((s) => s.id === stageId);
  if (stageIndex < 0) throw new Error(`pass: no stage '${stageId}' in the pipeline declaration`);
  for (const name of plan.clears.get(stageIndex) ?? []) enc.clearBuffer(bufs[name]!);

  assertBinds(plan, stageId, bindGroups.flat());

  let groups = ctx.bindGroups?.get(stageId);
  if (!groups) {
    groups = bindGroups.map((names, i) => device.createBindGroup({
      label: `${stageId}[${i}]`,
      layout: program.layouts[i]!,
      entries: names.map((name, binding) => ({ binding, resource: { buffer: bufs[name]! } })),
    }));
    ctx.bindGroups?.set(stageId, groups);
  }

  // Timestamps ride on the pass descriptor, so they cost no extra encoding and
  // nothing on the main thread -- unlike performance.measure, which is real work
  // inside the window being measured. The pair is written by the DEVICE at the
  // pass boundaries.
  //
  // Recorded here rather than at each of the fourteen encode functions for the
  // reason `assertBinds` lives here: this is the one chokepoint every pass goes
  // through, so a new stage is timed by construction and cannot be forgotten.
  const cp = enc.beginComputePass(timingFor(ctx, stageId));
  cp.setPipeline(program.pipeline);
  groups.forEach((g, i) => cp.setBindGroup(i, g));
  if (dispatch.kind === 'indirect') {
    cp.dispatchWorkgroupsIndirect(bufs[dispatch.args]!, dispatch.offset ?? 0);
  } else {
    cp.dispatchWorkgroups(dispatch.x, dispatch.y ?? 1);
  }
  cp.end();
}

/** Writes a uniform block. Uniforms are dedicated buffers, never pooled. */
function writeUniform(ctx: Ctx, name: string, fill: (dv: DataView) => void): void {
  const bytes = BUFFERS[name]!.bytes(ctx.dims);
  const buf = new ArrayBuffer(bytes);
  fill(new DataView(buf));
  ctx.device.queue.writeBuffer(ctx.bufs[name]!, 0, buf);
}

// ── S1 gradient ───────────────────────────────────────────────────────────

export function encodeGradient(ctx: Ctx): void {
  const { w, h } = ctx.dims;
  writeUniform(ctx, 'gradientUni', (dv) => {
    dv.setUint32(0, w, true);
    dv.setUint32(4, h, true);
  });
  const [gx, gy] = groups2D(w, h);
  pass(ctx, 'gradient', programs(ctx.device).gradient,
    [['gradientUni', 'gray', 'fx', 'fy']], { kind: 'direct', x: gx, y: gy });
}

// ── S2 grow ───────────────────────────────────────────────────────────────

export interface GrowSettings {
  /** Gradient magnitude below which a pixel never participates. */
  rhoLow: number;
  /** Level-line angular tolerance, degrees. */
  toleranceDeg: number;
  /**
   * How many hook+compress rounds to encode. Omit for GROW_ROUNDS, which is the
   * only value the pipeline should ship with.
   *
   * A DEBUG SCRUBBER, and the one setting here that is meant to produce a WRONG
   * answer: capping the rounds below the fixpoint leaves the labelling
   * half-grown, so the overlay can watch components coalesce. That truncation
   * is not silent -- `grow.gate` leaves `growArgs` non-zero, which surfaces as
   * the `growNotConverged` status bit, and `growRounds` reports what was
   * actually spent. Connected components are a fixpoint, so anything at or
   * above convergence is the same answer.
   */
  rounds?: number;
}

const GROW_GROUP0 = ['growUni', 'fx', 'fy', 'ux', 'uy', 'label', 'next', 'changed'] as const;

export function encodeGrowInit(ctx: Ctx, s: GrowSettings): void {
  const { w, h } = ctx.dims;
  writeUniform(ctx, 'growUni', (dv) => {
    dv.setUint32(0, w, true);
    dv.setUint32(4, h, true);
    // Squared, so the shader's eligibility test costs no sqrt. The negative
    // case has to survive: a negative threshold means "everything is eligible",
    // and squaring it would flip that into "almost nothing is".
    dv.setFloat32(16, s.rhoLow >= 0 ? s.rhoLow * s.rhoLow : -Infinity, true);
    dv.setFloat32(20, Math.cos((s.toleranceDeg * Math.PI) / 180), true);
  });

  const [gx, gy] = groups2D(w, h);
  // WRITTEN, not cleared, and this is the sharpest instance of the whole
  // zero-initialization family. A converged run leaves args at [0,0,0], so a
  // second reconstruction that inherited it would dispatch zero workgroups for
  // every round and emit init's singleton labelling -- one region per eligible
  // pixel, silently, at full speed, with no validation error anywhere.
  ctx.device.queue.writeBuffer(ctx.bufs.growArgs!, 0, new Uint32Array([gx, gy, 1, 0]));

  // init dispatches DIRECTLY. It must run over the whole image exactly once
  // whatever `args` says -- gating it on the early-out would be gating it on the
  // previous reconstruction.
  pass(ctx, 'grow.init', programs(ctx.device).growInit,
    [GROW_GROUP0], { kind: 'direct', x: gx, y: gy });
}

/**
 * One round: hook, compress, gate. Unconditional and identical every time,
 * which is the point of the early-out -- a round encoded after the fixpoint
 * dispatches zero workgroups for hook and compress and runs a single-thread
 * gate that re-zeroes an already-zero `args`.
 */
export function encodeGrowRound(ctx: Ctx): void {
  const p = programs(ctx.device);
  pass(ctx, 'grow.hook', p.growHook, [GROW_GROUP0], { kind: 'indirect', args: 'growArgs' });
  pass(ctx, 'grow.compress', p.growCompress, [GROW_GROUP0], { kind: 'indirect', args: 'growArgs' });
  pass(ctx, 'grow.gate', p.growGate, [GROW_GROUP0, ['growArgs']], { kind: 'direct', x: 1 });
}

// The round count comes off the SETTINGS, not a third parameter. It used to be
// both, and nothing ever passed the parameter -- which is exactly how the app's
// `lsdCclSteps` slider spent months bound, persisted and pushed over the dev
// bridge while reaching nothing at all.
export function encodeGrow(ctx: Ctx, s: GrowSettings): void {
  encodeGrowInit(ctx, s);
  const rounds = s.rounds ?? GROW_ROUNDS;
  for (let i = 0; i < rounds; i++) encodeGrowRound(ctx);
}

// ── The prefix scan ───────────────────────────────────────────────────────

/**
 * Encode one exclusive prefix scan over `count` vec2<u32> elements.
 *
 * Three passes and no recursion: scan each block, scan the block sums in a
 * single workgroup, add the block offsets back. `total` receives the grand sum,
 * which is why nothing downstream copies two scalars into a pair.
 *
 * The stage ids are passed in rather than derived, because liveness is keyed on
 * them -- the three scans in this pipeline are the same code over different
 * buffers, and pipeline.ts has to be able to tell them apart.
 */
export function encodeScan(ctx: Ctx, s: {
  ids: { blocks: string; spine: string; add: string };
  uni: string; spineUni: string;
  src: string; dst: string; sums: string; offs: string; total: string;
  count: number;
}): void {
  const p = programs(ctx.device);
  const blocks = scanBlocks(s.count);
  // One workgroup for the spine, so each of its threads takes however many
  // block sums it takes to cover them all. planPool has already asserted this
  // stays sane.
  const spinePer = Math.max(1, Math.ceil(blocks / SCAN_THREADS));

  writeUniform(ctx, s.uni, (dv) => {
    dv.setUint32(0, s.count, true);
    dv.setUint32(4, SCAN_PER, true);
  });
  writeUniform(ctx, s.spineUni, (dv) => {
    dv.setUint32(0, blocks, true);
    dv.setUint32(4, spinePer, true);
  });

  pass(ctx, s.ids.blocks, p.scanBlocks, [[s.uni, s.src, s.dst, s.sums]],
    { kind: 'direct', x: blocks });
  // ONE workgroup, and the whole two-level design rests on that being enough.
  pass(ctx, s.ids.spine, p.scanBlocks, [[s.spineUni, s.sums, s.offs, s.total]],
    { kind: 'direct', x: 1 });
  // `src` here is the block OFFSETS, not the original input -- the same binding
  // slot carrying a different array is what lets one layout serve both passes.
  pass(ctx, s.ids.add, p.scanAdd, [[s.uni, s.offs, s.dst]],
    { kind: 'direct', x: Math.ceil(s.count / SCAN_THREADS) });
}

// ── S3 collect ────────────────────────────────────────────────────────────

export interface CollectSettings {
  /** Gradient magnitude at least one member must clear, or the region is dropped. */
  rhoHigh: number;
  /** Regions smaller than this are dropped. */
  minRegionSize: number;
}

const WG1D = 64;
const groups1D = (count: number): number => Math.max(1, Math.ceil(count / WG1D));

export function encodeCollect(ctx: Ctx, s: CollectSettings): void {
  const { w, h, maxRegions } = ctx.dims;
  const n = w * h;
  const p = programs(ctx.device);

  writeUniform(ctx, 'collectUni', (dv) => {
    dv.setUint32(0, w, true);
    dv.setUint32(4, h, true);
    // Squared, matching grow's rhoLow, so the shader's test costs no sqrt.
    dv.setFloat32(8, s.rhoHigh >= 0 ? s.rhoHigh * s.rhoHigh : -Infinity, true);
    dv.setUint32(12, s.minRegionSize, true);
    dv.setUint32(16, maxRegions, true);
  });

  pass(ctx, 'collect.tally', p.collectTally,
    [['collectUni', 'label', 'fx', 'fy', 'labelSurvives', 'labelCounts']],
    { kind: 'direct', x: groups1D(n) });
  pass(ctx, 'collect.markKept', p.collectMarkKept,
    [['collectUni', 'labelSurvives', 'labelCounts', 'kept']],
    { kind: 'direct', x: groups1D(n) });

  encodeScan(ctx, {
    ids: { blocks: 'kept.scan', spine: 'kept.spine', add: 'kept.add' },
    uni: 'keptScanUni', spineUni: 'keptSpineUni',
    src: 'kept', dst: 'keptScan', sums: 'keptSums', offs: 'keptOffs', total: 'counts',
    count: n,
  });

  pass(ctx, 'collect.regionMeta', p.collectRegionMeta,
    [['collectUni', 'kept', 'keptScan', 'counts', 'regionOffsets', 'regionSizes', 'regionArgs']],
    { kind: 'direct', x: groups1D(n) });
  pass(ctx, 'collect.scatter', p.collectScatter,
    [['collectUni', 'label', 'kept', 'keptScan', 'cursor', 'members']],
    { kind: 'direct', x: groups1D(n) });
  // Indirect: the region count only exists on the device.
  pass(ctx, 'collect.finalize', p.collectFinalize,
    [['collectUni', 'counts', 'fx', 'fy', 'regionOffsets', 'regionSizes', 'members', 'meanDirs']],
    { kind: 'indirect', args: 'regionArgs' });
}

// ── S4 lsdFit ─────────────────────────────────────────────────────────────

export interface LsdFitSettings {
  /**
   * Gradient magnitude a footprint pixel must clear before its direction counts
   * toward the aligned tally. Weak pixels still count toward the footprint.
   */
  rho: number;
  /** Alignment tolerance, degrees. Also the NFA null model's own parameter. */
  toleranceDeg: number;
  /** Number of tests in the NFA = max(w, h) ^ this. */
  nfaTestExponent: number;
  /** Accept when the expected false-alarm count falls below this. */
  nfaEpsilon: number;
}

/**
 * Rectangle fit + NFA validation, one thread per region.
 *
 * The two log terms are derived here rather than in the shader because they are
 * pure functions of the settings and the image size -- constants for the frame,
 * and computing them per region would be per-region transcendentals for an
 * answer that never varies. Nothing here reads the device.
 */
export function encodeLsdFit(ctx: Ctx, s: LsdFitSettings): void {
  const { w, h, maxRegions } = ctx.dims;

  writeUniform(ctx, 'lsdFitUni', (dv) => {
    dv.setUint32(0, w, true);
    dv.setUint32(4, h, true);
    dv.setUint32(8, maxRegions, true);
    // Squared, matching grow's rhoLow and collect's rhoHigh, so the footprint's
    // hot loop rejects a weak pixel with no sqrt. The negative case has to
    // survive: negative means "every pixel is strong enough", and squaring it
    // would flip that into "almost none are".
    dv.setFloat32(16, s.rho >= 0 ? s.rho * s.rho : -Infinity, true);
    dv.setFloat32(20, (s.toleranceDeg * Math.PI) / 180, true);
    dv.setFloat32(24, s.nfaTestExponent * Math.log(Math.max(w, h)), true);
    dv.setFloat32(28, Math.log(s.nfaEpsilon), true);
  });

  // The same args triple collect's finalize ran off -- both are workgroup_size
  // (64) over regions, so one device-written count serves both and the host
  // still never learns how many regions there are.
  pass(ctx, 'lsdFit', programs(ctx.device).lsdFit,
    [['lsdFitUni', 'fx', 'fy', 'regionOffsets', 'regionSizes', 'members', 'meanDirs', 'counts', 'rects']],
    { kind: 'indirect', args: 'regionArgs' });
}

// ── S5 lines + votes ──────────────────────────────────────────────────────

export interface LineSettings {
  /**
   * A rectangle shorter than this is not usable as a line segment, however
   * confident the fitter is that it is real.
   *
   * This is the vote stage's own filter, and it is deliberately NOT applied
   * inside lsdFit -- NFA acceptance is that stage's verdict about noise, and the
   * two criteria only happen to land on the same object. The mechanical half of
   * that argument (which is the half that actually decided it) is in
   * LINES_FLAG_WGSL's header.
   */
  minLengthPx: number;
}

/**
 * Accepted rectangles -> a dense array of line segments, plus the indirect args
 * every per-line pass downstream runs off.
 *
 * Three of the five passes are the shared vec2 scan (§7), over maxRegions rather
 * than over n -- 16 blocks at maxRegions 16,384, so it is the cheapest of the
 * pipeline's three uses of that primitive.
 */
export function encodeLines(ctx: Ctx, s: LineSettings): void {
  const { maxRegions, maxLines } = ctx.dims;
  const p = programs(ctx.device);

  writeUniform(ctx, 'linesUni', (dv) => {
    dv.setUint32(0, maxRegions, true);
    dv.setUint32(4, maxLines, true);
    dv.setFloat32(8, s.minLengthPx, true);
  });

  // DIRECT over maxRegions, not indirect over the region count: the flag array
  // is a scan input and has to be written in full. See pipeline.ts.
  pass(ctx, 'lines.flag', p.linesFlag,
    [['linesUni', 'rects', 'counts', 'lineFlag']],
    { kind: 'direct', x: groups1D(maxRegions) });

  encodeScan(ctx, {
    ids: { blocks: 'line.scan', spine: 'line.spine', add: 'line.add' },
    uni: 'lineScanUni', spineUni: 'lineSpineUni',
    src: 'lineFlag', dst: 'lineScan', sums: 'lineSums', offs: 'lineOffs', total: 'lineCount',
    count: maxRegions,
  });

  // Also over maxRegions: the compaction reads a flag per REGION and writes a
  // segment per LINE, so its thread index is a region index.
  pass(ctx, 'lines.emit', p.linesEmit,
    [['linesUni', 'rects', 'lineFlag', 'lineScan', 'lines', 'lineCount', 'lineArgs', 'status']],
    { kind: 'direct', x: groups1D(maxRegions) });
}

export interface VoteSettings {
  /**
   * VERTICAL field of view, radians -- three.js's camera convention, and what
   * getAnalysisVFovRad returns. The horizontal FOV the config carries is the
   * caller's business; the aspect comes from `dims`.
   */
  vFovRad: number;
}

/**
 * One vote per line: the normal of the plane the segment sweeps out through the
 * camera centre, weighted by the segment's projected arc length.
 *
 * There is no orientation parameter. The rays are camera-space, which is what
 * the host version's always-identity MATH_QUAT argument amounted to, and the
 * orientation is precisely what the next stage derives from these votes.
 */
export function encodeVotes(ctx: Ctx, s: VoteSettings): void {
  const { w, h, maxLines } = ctx.dims;

  writeUniform(ctx, 'votesUni', (dv) => {
    dv.setUint32(0, w, true);
    dv.setUint32(4, h, true);
    dv.setUint32(8, maxLines, true);
    dv.setFloat32(16, Math.tan(s.vFovRad / 2), true);
    dv.setFloat32(20, w / h, true);
  });

  pass(ctx, 'votes.cast', programs(ctx.device).votesCast,
    [['votesUni', 'lines', 'lineCount', 'votes', 'maxWeight']],
    { kind: 'indirect', args: 'lineArgs' });
}

// ── S5c join ──────────────────────────────────────────────────────────────

export interface JoinSettings {
  /** VERTICAL field of view, radians -- the same quantity VoteSettings takes. */
  vFovRad: number;
  /**
   * Position noise on a detected segment ENDPOINT, in pixels. Not the pixel
   * noise of the image: an LSD endpoint is the extreme of a least-squares fit
   * over a whole region, so it is much better than one sample.
   *
   * With the corridor gate this is measured in the same units it is used in --
   * the corridor's half-width is kSigma * endpointNoisePx * sqrt(2) pixels,
   * straight off the error-propagation bound in JOIN_GATE_WGSL. It used to be
   * converted to radians first and compared against a squared angle.
   *
   * 0.15 FOR MONTHS, AND THAT WAS AN UNDERESTIMATE. The corridor sweep put the
   * best half-width at ~2.1px, which under 0.15 would have meant a 10-sigma
   * confidence level -- not a confidence level at all, but a model saying the
   * noise is about 3x what it was declared to be. Re-encoded as 0.5 at 3 sigma:
   * identical arithmetic, knobs that mean what they say. The angular gate could
   * not have surfaced this, because there this number was scaled by fov and
   * divided by an arc before anyone saw it in pixels.
   */
  endpointNoisePx: number;
  /**
   * How many sigma of LATERAL disagreement still counts as the same line: the
   * corridor's half-width is this many times the endpoint-noise bound.
   *
   * A confidence level rather than a tuning constant -- and **zero disables
   * joining entirely**, since no pair can then pass the gate and every line
   * becomes its own singleton composite. That is a degenerate configuration of
   * one code path rather than a second path, the same relationship
   * `PoseOptions.alias: false` has to the pooling: it makes "join off" testable
   * as an EXACT reproduction of the pre-join pose rather than as a branch
   * nothing exercises.
   */
  kSigma: number;
  /**
   * How far a segment's front may travel, as a multiple of the segment's own
   * length. Two segments may join when their fronts meet, i.e. when the gap
   * between them is at most `reachFrac * (Li + Lj)`.
   *
   * THE SCALE-FREE CORE OF THE GATE, and 1.0 is the principled value rather
   * than a tuned one: at exactly one own-length of travel, a front's lateral
   * uncertainty equals endpointNoisePx * sqrt(2) regardless of how long the
   * segment is. See JOIN_GATE_WGSL for the derivation.
   *
   * MEASURED AT 4, NOT AT THE 1 THE DERIVATION SUGGESTS. The argument fixes the
   * FORM of the constraint -- reach proportional to own length -- but not the
   * constant, and the sweep is unambiguous: anchor-exact runs 290 at reach 1,
   * 306 at reach 4, and back to 289 with the reach bound effectively removed
   * (reach 1000). So the bound is genuinely load-bearing, just looser than the
   * one-sigma-of-one-own-length reading of it. Do not "correct" this to 1 on the
   * strength of the derivation; the derivation does not determine it.
   *
   * REPLACED `maxAngleDeg`, which was the scale gate in disguise and denominated
   * in degrees while the danger is denominated in cells.
   */
  reachFrac: number;
  /**
   * How far, in pixels, a member endpoint may sit off the finished composite
   * before that member is dropped from it.
   *
   * Applied AFTER the composite is provisionally built, so it catches what no
   * pairwise test can: a cluster that is individually plausible pair by pair
   * and collectively a diagonal.
   */
  maxResidualPx: number;
  /**
   * How many merge rounds to encode. Each is anchor -> attach -> flatten ->
   * refit, and each recomputes the geometry the next one gates on -- so a
   * cluster's REACH grows only as fast as the length it has actually assembled.
   * Two 20px fragments 30px apart become a 70px composite, and a 70px composite
   * legitimately reaches further than either fragment did.
   *
   * **1 reproduces the one-shot star clustering exactly**, which is what makes
   * the iteration testable as a superset of the thing it generalises rather than
   * as a rewrite. Overshooting is safe: a converged round is a no-op, costing
   * four dispatches and changing nothing.
   */
  rounds: number;
  /**
   * Compare |dot| rather than dot, so two segments join across a gradient
   * POLARITY flip. See JOIN_GATE_WGSL: correct for a board of black and white
   * cells, wrong for lines drawn as strokes.
   */
  polarityAbs: boolean;
}

/**
 * Collinear segments -> composite lines, by star clustering in vote space.
 *
 * Six passes, three of which are the shared vec2 scan. The two O(N^2) passes
 * are DIRECT over maxLines: `joinFlag` is a scan input and must be total over
 * the scanned range, and `anchorFlag`'s tail has to be zeroed by the pass that
 * writes its interior.
 */
export function encodeJoin(ctx: Ctx, s: JoinSettings): void {
  const { w, h, maxLines } = ctx.dims;
  const p = programs(ctx.device);

  // The corridor's half-width, IN PIXELS -- no projection anywhere. A front that
  // travels its own segment's length arrives with lateral error
  // endpointNoisePx*sqrt(2) whatever that length is (JOIN_GATE_WGSL derives it),
  // so this is that bound at a confidence level. kSigma 0 gives 0, which no
  // perpendicular distance can be under, which is the exact-off switch.
  const tol = s.kSigma * s.endpointNoisePx * Math.SQRT2;

  writeUniform(ctx, 'joinUni', (dv) => {
    dv.setUint32(0, maxLines, true);
    dv.setUint32(4, w, true);
    dv.setUint32(8, h, true);
    dv.setFloat32(16, tol, true);
    dv.setFloat32(20, Math.tan(s.vFovRad / 2), true);
    dv.setFloat32(24, w / h, true);
    dv.setFloat32(28, s.reachFrac, true);
    dv.setFloat32(32, s.maxResidualPx, true);
    dv.setFloat32(36, s.polarityAbs ? 1 : 0, true);
  });

  // ── THE ROUND LOOP ───────────────────────────────────────────────────────
  //
  //   init -> refit -> R x [ anchor -> attach -> flatten -> refit ]
  //
  // `refit` sits at BOTH ends deliberately: it has to precede the first anchor
  // (which gates on cluster geometry) and follow the last flatten (or the
  // compaction would publish the second-to-last round's composites). The extra
  // pass is one O(N^2) dispatch, ~0.12ms.
  //
  // A CONVERGED ROUND IS A NO-OP, so overshooting `rounds` costs time and
  // nothing else: with no pair left inside the corridor every cluster is its own
  // anchor, parent is the identity, flatten rewrites the pointers it already
  // had, and refit recomputes the same geometry. That is the same early-out
  // shape grow's rounds have, minus the indirect-args trick -- worth having here
  // too if the round count ever grows past a handful.
  pass(ctx, 'join.init', p.joinInit,
    [['joinUni', 'anchorOf']],
    { kind: 'direct', x: groups1D(maxLines) });
  pass(ctx, 'join.refit', p.joinRefit,
    [['joinUni', 'lines', 'lineCount', 'anchorOf', 'clusterLine', 'clusterVote']],
    { kind: 'direct', x: groups1D(maxLines) });
  for (let r = 0; r < Math.max(1, s.rounds); r++) {
    pass(ctx, 'join.anchor', p.joinAnchor,
      [['joinUni', 'clusterVote', 'clusterLine', 'lineCount', 'anchorFlag']],
      { kind: 'direct', x: groups1D(maxLines) });
    pass(ctx, 'join.attach', p.joinAttach,
      [['joinUni', 'clusterVote', 'clusterLine', 'lineCount', 'anchorFlag', 'parent']],
      { kind: 'direct', x: groups1D(maxLines) });
    pass(ctx, 'join.flatten', p.joinFlatten,
      [['joinUni', 'lineCount', 'parent', 'anchorOf', 'joinFlag']],
      { kind: 'direct', x: groups1D(maxLines) });
    pass(ctx, 'join.refit', p.joinRefit,
      [['joinUni', 'lines', 'lineCount', 'anchorOf', 'clusterLine', 'clusterVote']],
      { kind: 'direct', x: groups1D(maxLines) });
  }

  encodeScan(ctx, {
    ids: { blocks: 'join.scan', spine: 'join.spine', add: 'join.add' },
    uni: 'joinScanUni', spineUni: 'joinSpineUni',
    src: 'joinFlag', dst: 'joinScan', sums: 'joinSums', offs: 'joinOffs', total: 'joinCount',
    count: maxLines,
  });

  pass(ctx, 'join.reduce', p.joinReduce,
    [['joinUni', 'joinFlag', 'joinScan', 'clusterLine', 'clusterVote', 'compLines', 'compVotes', 'compMaxWeight']],
    { kind: 'direct', x: groups1D(maxLines) });
}

// ── S6 fit ────────────────────────────────────────────────────────────────

/**
 * All the votes -> the floor's row/column axes and its normal. The end of Act I.
 *
 * Two passes, and BOTH dispatch one workgroup directly. There is nothing to
 * parameterize: the vote count lives on the device and the shaders read it
 * there, the matrix sizes are fixed at 6x6 and 3x3 whatever the vote count is,
 * and every threshold in here is a numerical constant rather than a setting. So
 * this stage takes no settings object -- the only encode function in the
 * pipeline that does not.
 *
 * `ataPartials` and the second reduction pass are deliberately absent; the
 * reasoning is in FIT_ATA_WGSL's header and is the one place this stage departs
 * from both the old pipeline and the declaration it inherited.
 */
export function encodeFit(ctx: Ctx): void {
  const p = programs(ctx.device);
  writeUniform(ctx, 'fitUni', (dv) => {
    dv.setUint32(0, ctx.dims.maxLines, true);
  });

  pass(ctx, 'fit.ata', p.fitAta,
    [['fitUni', 'compVotes', 'joinCount', 'compMaxWeight', 'ata']],
    { kind: 'direct', x: 1 });
  pass(ctx, 'fit.eigen', p.fitEigen,
    [['ata', 'triad', 'status', 'pose']],
    { kind: 'direct', x: 1 });
}

// ── S7 grid period + phase ────────────────────────────────────────────────

export interface GppSettings {
  /** VERTICAL field of view, radians -- the same convention votes.cast takes. */
  vFovRad: number;
  /** World units per pattern cell. GRID_STEP. `height = cellPitch / period`. */
  cellPitch: number;
  /**
   * Grazing cutoff for the cell-centre reprojection, as a cosine against the
   * floor normal. Rays flatter than this sample the floor at a glancing angle
   * where a cell is sub-pixel.
   */
  minGrazingCos: number;
}

/**
 * The period search's candidate range, and why both ends are physical.
 *
 * A detected line IS a grid line, so the two extreme lines of a family sit an
 * INTEGER number of periods apart and the only possible periods are
 * `spread / n`. That is what removes the seed, the bracket width and the sample
 * count a seeded search would have to be told.
 *
 * `nMin = 3`: fewer than three spacings is not credibly periodic, and decode
 * needs ORDER = 5 cells anyway.
 *
 * The upper end is the looser of two INDEPENDENT bounds, so a shortfall in one
 * cannot hide the true period from the other:
 *
 *  - BOARD: n is exactly how many cells of a family the detected lines span, and
 *    you cannot span more than the board has.
 *  - HEIGHT: assume the camera never gets further from the floor than
 *    MAX_CAMERA_HEIGHT_IN_BOARDS times the board's linear size. Since
 *    P = cellPitch / distance, a maximum distance is a MINIMUM period, and
 *    cellPitch cancels: P >= 1 / (H * N), i.e. n <= spread * H * N.
 *
 * They agree at exactly the assumed maximum height and the height bound is
 * looser below it, which is where the headroom comes from -- physically argued
 * rather than a tuning fudge. Axis-fit error and detection noise near the
 * extremes can push the APPARENT count slightly past the board's hard limit, and
 * the board bound alone leaves nowhere to go.
 *
 * HEADROOM_CAP bounds how much of that is bought, because `spread` is ANGULAR
 * rather than metric -- the gnomonic extent of the field of view is
 * ~2*tan(halfVFov) regardless of height, so it sits near 1 for a phone camera
 * and near 3 for a very wide lens.
 *
 * ── nMax IS DEVICE-SIDE, WHICH DECIDES HOW THE SWEEP DISPATCHES ──
 *
 * `nFromHeight` depends on `spread`, and `spread` only exists on the device. So
 * the host cannot size the sweep. Rather than write a fourth indirect-args
 * triple, the sweep dispatches over the HARD cap and each candidate past the
 * frame's own nMax scores itself zero -- a few hundred no-op workgroups against
 * an args buffer, a write, and a hazard.
 */
const MAX_CAMERA_HEIGHT_IN_BOARDS = 1.5;
/** Resultant floor for a peak to be a real periodicity peak rather than noise.
 *  A ~10x margin, not a decision boundary -- true and sub-multiple peaks sit at
 *  0.7-1.0 of the best, noise and spurious-long below 0.2. Distinctness makes
 *  the actual decision among the survivors. */
const GPP_SIGNIFICANCE = 0.5;
/** Compute budget for the distinctness test, which is what separates the true
 *  period from its sub-multiples. */
const GPP_TOP_K = 6;
/** The cell-centre patch is (2M+1)^2. */
const GPP_PATCH_M = 6;

function writeGppUni(ctx: Ctx, s: GppSettings): void {
  const { w, h, maxLines, torusR, torusC } = ctx.dims;
  const boardCells = Math.max(torusR, torusC);
  writeUniform(ctx, 'gppUni', (dv) => {
    dv.setUint32(0, w, true);
    dv.setUint32(4, h, true);
    dv.setUint32(8, maxLines, true);
    dv.setUint32(12, GPP_N_MIN, true);
    dv.setFloat32(16, Math.tan(s.vFovRad / 2), true);
    dv.setFloat32(20, w / h, true);
    dv.setFloat32(24, s.cellPitch, true);
    dv.setUint32(28, boardCells, true);
    dv.setUint32(32, GPP_HEADROOM_CAP * boardCells, true);
    dv.setFloat32(36, MAX_CAMERA_HEIGHT_IN_BOARDS * boardCells, true);
    dv.setUint32(40, gppCandidateCount(ctx.dims), true);
    dv.setFloat32(44, GPP_SIGNIFICANCE, true);
    dv.setUint32(48, GPP_TOP_K, true);
    dv.setUint32(52, GPP_PATCH_M, true);
    dv.setFloat32(56, s.minGrazingCos, true);
  });
}

/**
 * Lines -> two families of rectified 1D points, and how far each family reaches.
 *
 * The first half of Act II. `gpp.classify` is DIRECT over maxLines rather than
 * indirect off `lineArgs`, because `family` is a scan input -- see
 * GPP_CLASSIFY_WGSL, and pipeline.ts at the buffer.
 */
export function encodeGppSamples(ctx: Ctx, s: GppSettings): void {
  const { maxLines } = ctx.dims;
  const p = programs(ctx.device);
  writeGppUni(ctx, s);

  pass(ctx, 'gpp.classify', p.gppClassify,
    // The COMPOSITES. Same layouts, so the shader's own binding names are
    // untouched -- see pipeline.ts's note on the rebind.
    [['gppUni', 'compLines', 'joinCount', 'compVotes', 'triad', 'samples', 'family']],
    { kind: 'direct', x: groups1D(maxLines) });

  encodeScan(ctx, {
    ids: { blocks: 'family.scan', spine: 'family.spine', add: 'family.add' },
    uni: 'familyScanUni', spineUni: 'familySpineUni',
    src: 'family', dst: 'familyScan', sums: 'familySums', offs: 'familyOffs',
    total: 'familyCounts',
    count: maxLines,
  });

  pass(ctx, 'gpp.compact', p.gppCompact,
    [['gppUni', 'samples', 'family', 'familyScan', 'rowSamples', 'colSamples']],
    { kind: 'direct', x: groups1D(maxLines) });

  // ONE workgroup. `gpp.extentInit` and the atomic min/max it was initializing
  // are both gone -- GPP_EXTENT_WGSL's header has the argument, which is the
  // same one that deleted `ataPartials`.
  pass(ctx, 'gpp.extent', p.gppExtent,
    [['gppUni', 'rowSamples', 'colSamples', 'familyCounts', 'extent', 'status']],
    { kind: 'direct', x: 1 });
}

/**
 * The period search: score every physically possible period, pick the real
 * peaks, break the harmonic tie on image content, refine the winner.
 *
 * Four passes, and only the first is parallel over anything interesting. That
 * is not a compromise -- the harmonic disambiguation is six independent patch
 * samples and the refinement is inherently serial, so the shape of the problem
 * is one wide pass followed by three narrow ones. GPP_POLISH_WGSL's header has
 * the cost, measured against the alternative rather than assumed.
 *
 * `gpp.sweep` dispatches over the HARD candidate cap rather than the frame's own
 * nMax, because nMax depends on `spread` and `spread` only exists on the device.
 * Candidates past it score zero. The alternative is a fourth indirect-args
 * triple for a few hundred workgroups that read nothing.
 */
export function encodeGppSearch(ctx: Ctx, s: GppSettings): void {
  const p = programs(ctx.device);
  writeGppUni(ctx, s);

  // One workgroup per candidate.
  pass(ctx, 'gpp.sweep', p.gppSweep,
    [['gppUni', 'rowSamples', 'colSamples', 'familyCounts', 'extent', 'scores']],
    { kind: 'direct', x: gppCandidateCount(ctx.dims) });

  pass(ctx, 'gpp.peaks', p.gppPeaks,
    [['gppUni', 'scores', 'topK', 'status']], { kind: 'direct', x: 1 });

  // One thread per candidate, and the workgroup is 8 wide so every slot of
  // `distinctness` is written whether or not a candidate occupies it.
  pass(ctx, 'gpp.distinct', p.gppDistinct,
    [['gppUni', 'gray', 'topK', 'extent', 'triad', 'distinctness']],
    { kind: 'direct', x: 1 });

  pass(ctx, 'gpp.polish', p.gppPolish,
    [['gppUni', 'rowSamples', 'colSamples', 'familyCounts', 'extent', 'topK',
      'distinctness', 'gppResult']],
    { kind: 'direct', x: 1 });
}

/** Act II end to end: lines in, period / phases / height out. */
export function encodeGpp(ctx: Ctx, s: GppSettings): void {
  encodeGppSamples(ctx, s);
  encodeGppSearch(ctx, s);
}

// ── S8 decode layout ──────────────────────────────────────────────────────

export interface LayoutSettings {
  /** VERTICAL field of view, radians -- the same convention every other stage
   *  here takes. */
  vFovRad: number;
  /** World units per pattern cell. GRID_STEP. */
  cellPitch: number;
  /**
   * Grazing cutoff, as a cosine against the floor normal. It no longer gates any
   * BOUNDS computation -- the hull replaced the ray cast that needed it -- but it
   * is still the per-cell test decode.build applies, and the hull makes that test
   * MORE load-bearing rather than less: measured over 24 grazing poses, 9,811
   * hull cells are rejected by it that the screen-bounds test would have
   * accepted. Those are cells past the horizon, which project through a negated
   * depth onto a plausible-looking pixel.
   */
  minGrazingCos: number;
}

function writeLayoutUni(ctx: Ctx, s: LayoutSettings): void {
  const { w, h, torusR, torusC, maxCells } = ctx.dims;
  writeUniform(ctx, 'layoutUni', (dv) => {
    dv.setUint32(0, w, true);
    dv.setUint32(4, h, true);
    dv.setUint32(8, torusR, true);
    dv.setUint32(12, torusC, true);
    dv.setUint32(16, maxCells, true);
    dv.setFloat32(32, s.cellPitch, true);
    dv.setFloat32(36, Math.tan(s.vFovRad / 2), true);
    dv.setFloat32(40, w / h, true);
    dv.setFloat32(44, s.minGrazingCos, true);
  });
}

/**
 * Where the sampling lattice is: which floor cells to read, and what counts as
 * white.
 *
 * Three passes, and the interesting one is the third. The first two are the
 * image's mean grey level, which used to be a host loop over `gray`. The third
 * turns the recovered triad, period and phases into a lattice -- bounded by the
 * DETECTED LINES (§12), which is what deleted the 49x49 ray grid, its
 * minGrazingCos gate, `uvBounds` and its init pass.
 *
 * It also writes five of decode's six indirect-args triples. The sixth is
 * decode.argmax's, because its extent depends on the winning orientation.
 */
export function encodeDecodeLayout(ctx: Ctx, s: LayoutSettings): void {
  const { w, h } = ctx.dims;
  const p = programs(ctx.device);
  writeLayoutUni(ctx, s);

  // One workgroup per 256 pixels, which is exactly `binThreshPartials`' length.
  pass(ctx, 'decode.binThreshPartials', p.decodeBinThreshPartials,
    [['layoutUni', 'gray', 'binThreshPartials']],
    { kind: 'direct', x: Math.ceil((w * h) / 256) });

  pass(ctx, 'decode.binThreshReduce', p.decodeBinThreshReduce,
    [['layoutUni', 'binThreshPartials', 'binThreshold']], { kind: 'direct', x: 1 });

  pass(ctx, 'decode.layout', p.decodeLayout,
    [['layoutUni', 'triad', 'gppResult', 'extent', 'binThreshold', 'layout',
      'buildArgs', 'tallyArgs', 'status']],
    { kind: 'direct', x: 1 });
}

/**
 * The lattice, sampled: one thread per cell, two bits out.
 *
 * Indirect off `buildArgs`, so a frame with no lattice runs no threads -- which
 * is the whole of the failure propagation here. `packed` needs no clear: every
 * cell inside (rows, cols) is written on every path, and nothing downstream
 * reads outside that.
 */
export function encodeDecodeBuild(ctx: Ctx): void {
  pass(ctx, 'decode.build', programs(ctx.device).decodeBuild,
    [['layout', 'gray', 'packed']], { kind: 'indirect', args: 'buildArgs' });
}

/**
 * Act III's back half: every window votes, the winner is picked, and the grid is
 * scored against what the pattern says should be there.
 *
 * ── THE UNIFORMS CARRY ONLY HOST CONSTANTS ──
 *
 * `rows` and `cols` are computed on the device by decode.layout, so every pass
 * here reads them off the `layout` block rather than out of a uniform. Writing
 * them from the host would be exactly the readback this pipeline exists to
 * delete -- and it is the reason §12 called decode's layout the most
 * structurally disruptive change in the plan, showing up three stages after the
 * change itself.
 *
 * `board` is the printed pattern's own dimensions, which ARE host constants --
 * see board.ts, and open decision 5b.
 */
export function encodeDecodeTally(ctx: Ctx, board: { order: number }): void {
  const { torusR, torusC, hashSlots } = ctx.dims;
  const p = programs(ctx.device);

  writeUniform(ctx, 'tallyUni', (dv) => {
    dv.setUint32(0, board.order, true);
    dv.setUint32(4, torusR, true);
    dv.setUint32(8, torusC, true);
    dv.setUint32(12, hashSlots, true);
  });

  // One dispatch per orientation, each off its own triple in `tallyArgs`.
  // Accumulation is atomic, so the order between them does not matter.
  for (let o = 0; o < 4; o++) {
    pass(ctx, `decode.tally.o${o}`, p.decodeTally[o]!,
      [['tallyUni', 'layout', 'packed', 'hashKeys', 'hashValues', 'hist', 'totalWindows']],
      { kind: 'indirect', args: 'tallyArgs', offset: o * 12 });
  }

  writeUniform(ctx, 'argmaxUni', (dv) => {
    dv.setUint32(0, 4 * torusR * torusC, true);
    dv.setUint32(4, torusR, true);
    dv.setUint32(8, torusC, true);
    dv.setUint32(12, WG2D, true);
  });
  // ONE workgroup, dispatched directly -- it always runs, which is what makes it
  // the pass that zeroes `correctnessArgs` when there is no winner.
  pass(ctx, 'decode.argmax', p.decodeArgmax,
    [['argmaxUni', 'layout', 'hist', 'totalWindows', 'result', 'correctnessArgs']],
    { kind: 'direct', x: 1 });

  writeUniform(ctx, 'correctUni', (dv) => {
    dv.setUint32(0, torusR, true);
    dv.setUint32(4, torusC, true);
  });
  pass(ctx, 'decode.correctness', p.decodeCorrectness,
    [['correctUni', 'layout', 'packed', 'torus', 'result']],
    { kind: 'indirect', args: 'correctnessArgs' });
}

// ── S13 finish ────────────────────────────────────────────────────────────

/**
 * The anchor -> a world pose, and the 128 bytes that are the only readback.
 *
 * One thread. It also does the REPORTING half of the status word -- deriving the
 * bits that are readable off buffers it already binds, rather than having every
 * kernel test a status it does not otherwise care about. The PROPAGATION half is
 * separate and happens where each failure occurs, by zeroing indirect args.
 */
export function encodeFinish(ctx: Ctx): void {
  const { torusR, torusC, maxRegions, maxLines } = ctx.dims;
  writeUniform(ctx, 'finishUni', (dv) => {
    dv.setUint32(0, torusR, true);
    dv.setUint32(4, torusC, true);
    dv.setUint32(8, maxRegions, true);
    dv.setUint32(12, maxLines, true);
  });
  pass(ctx, 'finish', programs(ctx.device).finish,
    [['finishUni', 'layout', 'result', 'counts', 'lineCount', 'growArgs', 'joinCount', 'status', 'pose']],
    { kind: 'direct', x: 1 });
}

/**
 * §15's status word, by name.
 *
 * The bits are set in WGSL, tabulated in §15 and read here -- three places, and
 * this is the only one a caller can reach. Named rather than left as literals at
 * each call site: a host testing `status & 32` is a number that means nothing at
 * the point of reading and cannot be grepped back to the pass that sets it.
 *
 * FOUR KINDS, and §15's point is that they must stay distinguishable. `ordinary`
 * means the frame does not contain a decodable board, which is an outcome and
 * not a fault; `cap` means raise a constant; `budget` means the round count was
 * too small; `gridOverflow` is diagnostic and the frame decoded correctly
 * anyway. A caller that collapses these into "failed" makes a real capacity
 * problem look like an empty frame.
 */
export const POSE_STATUS = {
  /** budget -- grow did not converge inside the encoded round count. */
  growNotConverged: 1 << 0,
  /** cap -- more line-support regions than maxRegions. */
  regionOverflow: 1 << 1,
  /** ordinary */
  noRegions: 1 << 2,
  /** cap -- more accepted segments than maxLines. */
  lineOverflow: 1 << 3,
  /** ordinary */
  noVotes: 1 << 4,
  /** ordinary -- an all-zero scatter matrix, so there is no triad. */
  fitDegenerate: 1 << 5,
  /** ordinary */
  gppNoSamples: 1 << 6,
  /** ordinary */
  gppNoCandidates: 1 << 7,
  /** ordinary -- fewer than 4 valid rays, so there is no lattice. */
  layoutInvalid: 1 << 8,
  /** DIAGNOSTIC, never a failure -- the hull exceeded one board period and was
   *  clamped, and the frame decoded correctly anyway (§12). */
  gridOverflow: 1 << 9,
  /** ordinary */
  decodeNoAnchor: 1 << 10,
} as const;

/**
 * The 128-byte `layout` block, decoded -- the lattice description `decode.build`
 * and `finish` both read off the device.
 *
 * Host-readable because it is the display's only source for the recovered TRIAD
 * and the camera's distance from the floor: the pose block carries a quaternion
 * and a position, not the axes those are expressed against.
 *
 * The offsets are hand-derived from DECODE_LAYOUT_WGSL's `Layout`, and that is
 * exactly the trap that shader's header records -- the three axes are `vec4`
 * with only `xyz` used SO THAT these offsets are the obvious ones. As `vec3`
 * they would be 12 bytes with align 16, and every scalar after them would pack
 * four bytes earlier than a reader written from the field order expects: not a
 * wrong value, a DIFFERENT FIELD, for every read from the first scalar on.
 */
export interface PoseLayout {
  Drow: { x: number; y: number; z: number };
  Dcol: { x: number; y: number; z: number };
  normal: { x: number; y: number; z: number };
  distance: number; tanHalf: number; aspect: number; minGrazingCos: number;
  uPhase: number; vPhase: number; cellPitch: number; binThreshold: number;
  rows: number; cols: number; imageW: number; imageH: number;
  kMinU: number; kMinV: number; zeroI: number; zeroJ: number;
  /** 0 when decode.layout found no usable lattice. Everything above is then
   *  meaningless -- including the triad, which it copies rather than derives. */
  valid: number;
}

export function decodeLayout(bytes: ArrayBuffer): PoseLayout {
  const f = new Float32Array(bytes);
  const u = new Uint32Array(bytes);
  const i = new Int32Array(bytes);
  const v3 = (at: number) => ({ x: f[at]!, y: f[at + 1]!, z: f[at + 2]! });
  return {
    Drow: v3(0), Dcol: v3(4), normal: v3(8),
    distance: f[12]!, tanHalf: f[13]!, aspect: f[14]!, minGrazingCos: f[15]!,
    uPhase: f[16]!, vPhase: f[17]!, cellPitch: f[18]!, binThreshold: f[19]!,
    rows: u[20]!, cols: u[21]!, imageW: u[22]!, imageH: u[23]!,
    kMinU: i[24]!, kMinV: i[25]!, zeroI: u[26]!, zeroJ: u[27]!, valid: u[28]!,
  };
}

/** The 128 bytes, decoded. The one place a host reads anything off this pipeline. */
export interface PoseResult {
  status: number;
  ok: boolean;
  position: { x: number; y: number; z: number };
  quaternion: { x: number; y: number; z: number; w: number };
  consistency: number;
  orientation: number;
  boardRow: number;
  boardCol: number;
  votes: number;
  totalWindows: number;
  correct: number;
  wrong: number;
  regionCount: number;
  memberCount: number;
  lineCount: number;
  gridRows: number;
  gridCols: number;
  growRounds: number;
  /** Composite lines after the join. See FINISH_WGSL for how to read it. */
  compositeCount: number;
  period: number;
  height: number;
  /**
   * THE FIT'S OWN MISFIT, both in [0, 1] with 0 perfect. Written by `fit.eigen`
   * itself, not by `finish`, and present on EVERY frame -- including one whose
   * decode failed, which is the case they exist for.
   *
   * The triad is orthonormal however badly the fit went, so these are the only
   * signal that distinguishes a real board from noise that happened to produce
   * axes. Read them BEFORE believing an orientation.
   *
   * `fitResidual` -- do the vote normals lie on a common quadric at all.
   * `fitPlanarity` -- is that quadric a genuine PAIR OF PLANES, which is what
   * the triad extraction assumes. The second is the load-bearing one: a small
   * residual against an ellipsoid still yields a confident triad pointing
   * nowhere. See FIT_EIGEN_WGSL for the derivation of both.
   */
  fitResidual: number;
  fitPlanarity: number;
  /**
   * How much evidence the SECOND recovered axis has, in [0, 1] -- and this one
   * is 1-is-good, unlike the two above. Near 0 means the fit saw only ONE family
   * of parallel lines and the "pair of planes" is a doubled plane, which
   * `fitPlanarity` cannot see because a doubled plane is a legitimate pair (it
   * reports ~1e-17 on exactly that input). Check this before trusting a low
   * planarity.
   */
  fitAxisSupport: number;
}

export function decodePose(bytes: ArrayBuffer): PoseResult {
  const u = new Uint32Array(bytes);
  const f = new Float32Array(bytes);
  return {
    status: u[0]!, ok: u[1] === 1,
    position: { x: f[2]!, y: f[3]!, z: f[4]! },
    quaternion: { x: f[5]!, y: f[6]!, z: f[7]!, w: f[8]! },
    consistency: f[9]!,
    orientation: u[10]!, boardRow: u[11]!, boardCol: u[12]!,
    votes: u[13]!, totalWindows: u[14]!, correct: u[15]!, wrong: u[16]!,
    regionCount: u[17]!, memberCount: u[18]!, lineCount: u[19]!,
    gridRows: u[20]!, gridCols: u[21]!, growRounds: u[22]!, compositeCount: u[25]!,
    period: f[23]!, height: f[24]!,
    fitResidual: f[26]!, fitPlanarity: f[27]!, fitAxisSupport: f[28]!,
  };
}
