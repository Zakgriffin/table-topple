import { type Arena, type Slice, createArena } from '../../gpu/arena.ts';
import {
  type TransferEntry, type TransferSummary, getGPUDevice, readSliceU32, recordTransfer, writeSlice,
} from '../../gpu/device.ts';
import { encodeGradient2x2 } from '../gradient/gradient2x2.gpu.ts';
import { type RegionSetGPU, encodeCollectRegions } from './collectRegions.gpu.ts';
import { encodeGrowInit, encodeGrowRound } from './growRegions.gpu.ts';
// readRegionMembers is here for the HARNESS BRIDGE at the bottom of this file
// only -- the production chain does not read the region CSR at all any more.
import { type LsdFitResult, encodeLsdFit, readLsdFitResults, readRegionMembers } from './lsdFit.gpu.ts';
import { spanEnd } from '../../../sphereLab/profiling/profiler.ts';
import { poseSpan } from '../../timing/stages.ts';
import { type GradientField } from '../../results.ts';
import { type Backend } from '../../backend.ts';
import { computeGradient2x2Field } from '../gradient/gradientField.ts';
import { collectRegionsFromLabels, growRegionsCCL } from './regions.cpu.ts';
import { fitRegionsCPU, nfaLog10ToLineScore, nfaLogTerms } from './rectangles.cpu.ts';
import { type GrownRegion, type LsdRectangle, type LsdSettings } from './types.ts';

// ── LSD (Line Segment Detector, von Gioi/Jakubowicz/Morel/Randall 2010) ───
// ── from scratch ────────────────────────────────────────────────────────
//
// A genuinely traditional reimplementation, and the PRODUCTION composite-line
// source: pose/stages/votes/votes.ts's computeGradient2x2Composites turns each
// accepted rectangle straight into one line.
//
// Pipeline: 2x2 gradient -> hysteresis-gated, directed-angle CONNECTED-COMPONENT
// region growing -> hysteresis survival + CSR collect -> magnitude-weighted PCA
// rectangle fit per region -> NFA statistical validation.
//
// This file forms segments ONLY, and NOTHING bridges them afterwards. Bridging
// genuinely disjoint segments (across a gradient dropout, or across a level-line
// POLARITY flip) is therefore currently UNSOLVED rather than solved elsewhere.
// Every pixel comparison is strictly between 8-NEIGHBORS, so the class of error
// the old long-strided design had -- a jump landing on a different parallel
// ridge that merely shares a direction -- is unreachable rather than guarded.
//
// ── TWO STRAIGHT-LINE CHAINS, not one broker ──
//
// There used to be a FieldResidency between every pair of stages, deciding per
// field whether a transfer was needed, because each stage dispatched to CPU or
// GPU on its OWN toggle. Those toggles collapsed into a single `backend` a while
// ago and nothing revisited the broker -- so 449 lines were arbitrating a
// configuration matrix that no longer existed. There is no production
// configuration that produces a mixed chain.
//
// So: `runLsdChainCPU` is typed arrays start to finish, `runLsdChainGPU` is
// slices start to finish, and neither knows the other exists. A GPU validation
// failure falls the WHOLE chain back to CPU rather than one stage, which is the
// deliberate loss of a capability nothing was using.

// The arena is per DEVICE and lives across reconstructions -- that is the point
// of it, and re-creating one per run would reinstate exactly the per-frame
// allocation churn it exists to remove. `reset()` at the top of each run frees
// the previous run's slices and bumps the generation, so anything still holding
// one (a display drain that never ran, say) throws instead of silently reading
// another run's bytes.
const arenaCache = new WeakMap<GPUDevice, Arena>();
export function poseArena(device: GPUDevice): Arena {
  let a = arenaCache.get(device);
  if (!a) {
    a = createArena(device, { initialBytes: 1 << 24 }); // 16MB; resizes itself from the high-water mark
    arenaCache.set(device, a);
  }
  return a;
}

// What a GPU chain leaves behind. Every field is a live arena slice, valid until
// the next `arena.reset()` -- which is the whole ownership story now: the caller
// holds these, reads whichever it wants, and frees them all at once by starting
// the next run. Nothing here needs destroying and nothing can be destroyed
// twice.
export interface LsdChainGPU {
  rects: LsdRectangle[];
  arena: Arena;
  n: number;
  // Every crossing this run actually made, counted where it is made. The
  // residency used to own this and could only see the transfers it brokered.
  transfers: TransferSummary;
  gray: Slice;
  fx: Slice;
  fy: Slice;
  label: Slice;
  regionId: Slice;
  regions: RegionSetGPU;
  regionCount: number;
  memberCount: number;
}

// How many rounds get encoded into ONE command submission before the
// convergence flag is read back. The flag lives on the GPU, so checking it every
// round would mean a GPU->CPU sync per round.
//
// SIZED TO THE OBSERVED ROUND COUNT, not to the hard cap, and the trade changed
// when the early-out landed. Overshoot used to cost real work -- up to
// (ROUNDS_PER_BATCH - 1) full-image rounds past the fixpoint -- so the batch was
// kept small at 8 and a converging run paid two fences. Now `gate` zeroes the
// dispatch args at the fixpoint, so an overshot round is a zero-workgroup
// dispatch plus a one-thread gate: nearly free to RUN, but still real to ENCODE.
// So the batch wants to be just big enough to cover convergence in one pass and
// no bigger.
//
// 16 against a measured 9 rounds at 480x640 (fixtures/default, CPU reference).
// The theoretical bound is O(log n) ~ 19 for a 307200-pixel image, since
// pointer-jumping halves the label-chain depth every round -- so 16 is a
// practical margin, not a guarantee. Exceeding it is not a correctness problem;
// it costs one more batch and one more fence, which is exactly what every run
// paid before.
//
// The hard cap (w + h + 64) is still the loop's real bound and is ~1184 here.
// Encoding it outright would remove the last fence entirely and is the obvious
// next question -- 1184 rounds is ~3550 encoded passes, so it is a question
// about ENCODE cost, and it should be answered with a measurement rather than
// this comment.
const ROUNDS_PER_BATCH = 16;

// Stages 1-4 on GPU: gray in, rectangles out, every intermediate left on device.
export async function runLsdChainGPU(
  gray: Float64Array, w: number, h: number, settings: LsdSettings,
): Promise<LsdChainGPU | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const arena = poseArena(device);
  arena.reset();

  // ONE validation scope around the whole chain, popped once at the end. It used
  // to be one per stage, each with its own `await popErrorScope()` in the middle
  // of what should be a single encode. Validation failures are reported
  // per-device, not per-pass, so one scope catches exactly what four caught
  // between them -- and reports it in one place, where the fallback lives.
  device.pushErrorScope('validation');
  const n = w * h;
  const alloc = arena.alloc;
  const entries: TransferEntry[] = [];
  const crossed = (what: string, direction: 'up' | 'down', bytes: number) => {
    entries.push({ what, direction, bytes });
  };

  // ── gray up, and the narrowing that goes with it ──
  const graySlice = alloc(n * 4, 'chain.gray');
  {
    const t0 = performance.now();
    const narrowed = new Float32Array(gray);
    recordTransfer({
      what: 'gray (f64→f32 narrow)', kind: 'convert', dir: 'up', bytes: n * 4,
      ms: performance.now() - t0, startMs: t0, bareFenceMs: null, queueDrainMs: null,
    }, 'lsd.gradient');
    writeSlice(arena, graySlice, narrowed, 'gray', 'lsd.gradient');
    crossed('gray', 'up', n * 4);
  }

  const gradSpan = poseSpan('lsd.gradient', { backend: 'gpu' });
  const growSpan = poseSpan('lsd.grow', { backend: 'gpu' });
  let fx: Slice, fy: Slice, growLabel: Slice;
  {
    const enc = device.createCommandEncoder();
    const grad = encodeGradient2x2(arena, alloc, enc, { gray: graySlice, w, h });
    fx = grad.fx; fy = grad.fy;
    const st = encodeGrowInit(arena, alloc, enc, {
      fx, fy, w, h, toleranceDeg: settings.toleranceDeg, rhoLow: settings.rhoNoiseThreshold,
    });
    growLabel = st.label;

    // Same bound the CPU path uses -- hook alone propagates one pixel per round,
    // so the longest possible chain caps it; compression makes the real count
    // logarithmic, so this is never reached in practice.
    const hardCap = w + h + 64;
    const cap = settings.cclSteps > 0 ? Math.min(settings.cclSteps, hardCap) : hardCap;

    // The first batch rides along in the same encoder as init and the gradient,
    // so a converged-in-one-batch capture costs exactly one submit for stages
    // 1-3 rather than three.
    let roundsEncoded = 0;
    let encoder: GPUCommandEncoder | null = enc;
    while (roundsEncoded < cap) {
      const e = encoder ?? device.createCommandEncoder();
      encoder = null;
      const batch = Math.min(ROUNDS_PER_BATCH, cap - roundsEncoded);
      for (let r = 0; r < batch; r++) encodeGrowRound(arena, e, st);
      device.queue.submit([e.finish()]);
      roundsEncoded += batch;
      // The ONLY fence in the round loop, and with a batch that covers the
      // observed round count it is paid once rather than per batch. What it
      // reads is the dispatch args themselves: `gate` zeroes them at the
      // fixpoint, so x === 0 IS convergence -- there is no second flag to keep
      // in sync, and a real image cannot make ceil(w/8) legitimately zero.
      const a = await readSliceU32(arena, st.args, 16, 'grow:converged', 'lsd.grow');
      crossed('grow:converged', 'down', 16);
      if (a[0] === 0) break;
    }
    // A cap-length run that never converged still submitted everything above; if
    // the loop never ran at all (cap <= 0) the gradient still has to reach the
    // device, so the encoder is flushed here.
    if (encoder) device.queue.submit([encoder.finish()]);
  }
  spanEnd(gradSpan);

  spanEnd(growSpan);

  // ── collect + fit, one encoder, one submit, nothing between them ──
  //
  // Two DECLARED stages sharing one encoder, which is the whole argument for
  // un-fusing them: the labeling still never crosses the bus, and a harness can
  // now name `label` as a stage output.
  const { logNTests, logEpsilon } = nfaLogTerms(w, h, settings);
  const enc2 = device.createCommandEncoder();
  const collectSpan = poseSpan('lsd.collect', { backend: 'gpu' });
  const { regionId, regions } = encodeCollectRegions(arena, alloc, enc2, {
    label: growLabel, fx, fy, n, rhoHigh: settings.rhoHighThreshold, minRegionSize: settings.minRegionSize,
  });
  spanEnd(collectSpan);

  const fitSpan = poseSpan('lsd.fit', { backend: 'gpu' });
  const out = encodeLsdFit(arena, alloc, enc2, {
    fx, fy, regions, w, h,
    rho: settings.rhoNoiseThreshold, toleranceDeg: settings.toleranceDeg, logNTests, logEpsilon,
  });
  device.queue.submit([enc2.finish()]);

  const { results, regionCount, memberCount } = await readLsdFitResults(arena, out, regions, 'lsd.fit');
  crossed('regions:counts', 'down', 8);
  crossed('lsdFit:rects', 'down', regionCount * 40);

  // Checked BEFORE `results` is believed. WebGPU reports validation failures
  // asynchronously: an invalid bind group does not throw, it makes every command
  // that uses it a silent no-op, so the readback would be all zeros and this
  // would return a full set of plausible-looking rectangles at the origin. That
  // is the whole reason the error scope exists.
  const err = await device.popErrorScope();
  if (err) {
    console.error('runLsdChainGPU: WebGPU validation error, falling back to CPU --', err.message);
    spanEnd(fitSpan);
    return null;
  }

  // ONE RECT PER REGION, in region order -- that alignment is what every
  // member-drawing caller joins on now (see LsdRectangle). The chain used to
  // read the whole region CSR back HERE, under a `wantMembers` flag, purely so
  // each rect could carry its own member list; the display path then read the
  // same CSR a second time to draw the raw-region view. Both reads are the
  // caller's now, and there is only one of them.
  const rects: LsdRectangle[] = new Array(results.length);
  for (let i = 0; i < results.length; i++) {
    const g = results[i];
    rects[i] = {
      cx: g.cx, cy: g.cy, theta: g.theta, length: g.length, width: g.width,
      accepted: g.accepted, nfaLog10: g.nfaLog10,
      lineScore: nfaLog10ToLineScore(g.nfaLog10, logEpsilon),
    };
  }
  spanEnd(fitSpan);

  let bytes = 0;
  for (const e of entries) bytes += e.bytes;
  return {
    rects, arena, n, transfers: { crossings: entries.length, bytes, entries },
    gray: graySlice, fx, fy, label: growLabel, regionId, regions, regionCount, memberCount,
  };
}

// Fully-CPU path, and the source of truth every GPU stage is verified against.
// Typed arrays throughout -- no residency, no slices, nothing to own.
export function computeLsdRectangles(field: GradientField, settings: LsdSettings): LsdRectangle[] {
  const { w, h, fx, fy } = field;
  const { label } = growRegionsCCL(
    fx, fy, w, h, settings.toleranceDeg, settings.rhoNoiseThreshold, settings.cclSteps,
  );
  const { regions } = collectRegionsFromLabels(
    label, fx, fy, settings.rhoHighThreshold, w * h, settings.minRegionSize,
  );
  return fitRegionsCPU(regions, fx, fy, w, h, settings);
}

// What the CPU chain leaves behind, shaped so a caller can ask for the same
// things the GPU chain hands back without branching on which ran.
export interface LsdChainCPU {
  rects: LsdRectangle[];
  fx: Float64Array;
  fy: Float64Array;
  // `lsd.grow`'s output, carried for the same reason `LsdChainGPU` carries its
  // slice: it is a stage output, and it became one on this side only when the
  // collect moved out of `growRegionsCCL`.
  label: Int32Array;
  regionId: Int32Array;
  regions: { members: Int32Array; meanUx: number; meanUy: number }[];
}

export function runLsdChainCPU(
  gray: Float64Array, w: number, h: number, settings: LsdSettings,
): LsdChainCPU {
  const gradSpan = poseSpan('lsd.gradient', { backend: 'cpu' });
  const field = computeGradient2x2Field(gray, w, h);
  spanEnd(gradSpan);

  // Two spans, because there are two stages -- the round loop hands over a
  // labeling and the collector turns it into regions, exactly as
  // `runLsdChainGPU` encodes them. Until Step 3 split `growRegionsCCL` the
  // collect ran inside the grow call, so `lsd.collect` never recorded on a CPU
  // run and the critical-path walk saw a silently absent input.
  const growSpan = poseSpan('lsd.grow', { backend: 'cpu' });
  const { label } = growRegionsCCL(
    field.fx, field.fy, w, h, settings.toleranceDeg, settings.rhoNoiseThreshold, settings.cclSteps,
  );
  spanEnd(growSpan);

  const collectSpan = poseSpan('lsd.collect', { backend: 'cpu' });
  const { regionId, regions } = collectRegionsFromLabels(
    label, field.fx, field.fy, settings.rhoHighThreshold, w * h, settings.minRegionSize,
  );
  spanEnd(collectSpan);

  // Around the fitter, so the span means "fit finished" on both backends and
  // `lsd.fit` aggregates across its `backend` attr -- which was the point of
  // collapsing the two ids into one.
  const fitSpan = poseSpan('lsd.fit', { backend: 'cpu' });
  const rects = fitRegionsCPU(regions, field.fx, field.fy, w, h, settings);
  spanEnd(fitSpan);

  return { rects, fx: field.fx, fy: field.fy, label, regionId, regions };
}

// ── Stage-isolation bridges, for the verify harnesses only ───────────────
//
// The production chain has no seam a harness can reach into: it runs gray to
// rectangles with everything on device, which is the point. These take CPU
// inputs, run ONE stage on GPU, and bring the answer back -- which is what a
// differential check actually wants, and it is now stated in the harness's own
// terms rather than as a `collectOnGPU` boolean threaded three calls deep into a
// kernel.
//
// Nothing in the pose path may call these: they exist to pay for readbacks the
// production path spent years removing.

// Grow on GPU, then collect on whichever side the caller names. `collectOnGPU`
// isolates the collect stage: the labeling comes from the same GPU round loop
// both ways, so the ONLY difference is which collector turned it into regions.
export async function growCollectGPUToCPU(
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  toleranceDeg: number, rhoLow: number, rhoHigh: number, maxRounds: number, minRegionSize: number,
  collectOnGPU: boolean,
): Promise<{ regionId: Int32Array; regions: GrownRegion[]; roundsRun: number; converged: boolean } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const arena = poseArena(device);
  arena.reset();
  device.pushErrorScope('validation');
  const n = w * h;
  const alloc = arena.alloc;

  const fxS = alloc(n * 4, 'verify.fx');
  const fyS = alloc(n * 4, 'verify.fy');
  writeSlice(arena, fxS, new Float32Array(fx), 'verify:fx');
  writeSlice(arena, fyS, new Float32Array(fy), 'verify:fy');

  const enc = device.createCommandEncoder();
  const st = encodeGrowInit(arena, alloc, enc, { fx: fxS, fy: fyS, w, h, toleranceDeg, rhoLow });
  const hardCap = w + h + 64;
  const cap = maxRounds > 0 ? Math.min(maxRounds, hardCap) : hardCap;
  let roundsEncoded = 0, roundsRun = 0, converged = false;
  let encoder: GPUCommandEncoder | null = enc;
  while (roundsEncoded < cap) {
    const e = encoder ?? device.createCommandEncoder();
    encoder = null;
    const batch = Math.min(ROUNDS_PER_BATCH, cap - roundsEncoded);
    for (let r = 0; r < batch; r++) encodeGrowRound(arena, e, st);
    device.queue.submit([e.finish()]);
    roundsEncoded += batch;
    const a = await readSliceU32(arena, st.args, 16, 'grow:converged', 'lsd.grow');
    // `activeRounds` counts rounds that changed a label, so the reported count
    // is those plus the one quiet round that proved the fixpoint -- the same
    // convention `growRegionsCCL` returns. It used to be reported as rounds
    // ENCODED, which the batching rounded UP to a multiple of ROUNDS_PER_BATCH;
    // this is the true number, and now that a batch overshoots on purpose the
    // encoded count would have been meaningless.
    if (a[0] === 0) { converged = true; roundsRun = a[3] + 1; break; }
    roundsRun = a[3];
  }
  if (encoder) device.queue.submit([encoder.finish()]);

  let regionId: Int32Array, regions: GrownRegion[];
  if (collectOnGPU) {
    const enc2 = device.createCommandEncoder();
    const collected = encodeCollectRegions(arena, alloc, enc2, {
      label: st.label, fx: fxS, fy: fyS, n, rhoHigh, minRegionSize,
    });
    device.queue.submit([enc2.finish()]);
    const counts = await readSliceU32(arena, collected.regions.counts, 8, 'regions:counts');
    const idRaw = await readSliceU32(arena, collected.regionId, n * 4, 'regionId');
    regionId = new Int32Array(idRaw.buffer.slice(0));
    regions = (await readRegionMembers(arena, collected.regions, counts[0], counts[1])) as GrownRegion[];
  } else {
    const labRaw = await readSliceU32(arena, st.label, n * 4, 'label');
    const cpu = collectRegionsFromLabels(
      new Int32Array(labRaw.buffer.slice(0)), fx, fy, rhoHigh, n, minRegionSize,
    );
    regionId = cpu.regionId;
    regions = cpu.regions;
  }

  const err = await device.popErrorScope();
  if (err) {
    console.error('growCollectGPUToCPU: WebGPU validation error --', err.message);
    return null;
  }
  return { regionId, regions, roundsRun, converged };
}

// Stage 4 in isolation: CPU-built regions in, GPU fit results out.
export async function fitRegionsGPUFromCPU(
  fx: Float64Array, fy: Float64Array, regions: GrownRegion[], w: number, h: number,
  settings: LsdSettings,
): Promise<LsdFitResult[] | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const arena = poseArena(device);
  arena.reset();
  device.pushErrorScope('validation');
  const n = w * h;
  const alloc = arena.alloc;
  const { logNTests, logEpsilon } = nfaLogTerms(w, h, settings);

  const fxS = alloc(n * 4, 'verify.fx');
  const fyS = alloc(n * 4, 'verify.fy');
  writeSlice(arena, fxS, new Float32Array(fx), 'verify:fx');
  writeSlice(arena, fyS, new Float32Array(fy), 'verify:fy');

  // The CSR the collect stage would have built on device, built here from host
  // arrays instead. Padded to one element each because a zero-byte binding is
  // not legal and an empty region set is an ordinary input.
  const regionCount = regions.length;
  const offsets = new Uint32Array(Math.max(regionCount, 1));
  const sizes = new Uint32Array(Math.max(regionCount, 1));
  let total = 0;
  for (let i = 0; i < regionCount; i++) {
    offsets[i] = total; sizes[i] = regions[i].members.length; total += sizes[i];
  }
  const members = new Uint32Array(Math.max(total, 1));
  const meanDirs = new Float32Array(Math.max(regionCount * 2, 2));
  for (let i = 0; i < regionCount; i++) {
    members.set(regions[i].members, offsets[i]);
    meanDirs[i * 2] = regions[i].meanUx; meanDirs[i * 2 + 1] = regions[i].meanUy;
  }

  const put = (data: Uint32Array | Float32Array, label: string): Slice => {
    const s = alloc(data.byteLength, label);
    writeSlice(arena, s, data, label);
    return s;
  };
  const set: RegionSetGPU = {
    offsets: put(offsets, 'verify:offsets'),
    sizes: put(sizes, 'verify:sizes'),
    members: put(members, 'verify:members'),
    meanDirs: put(meanDirs, 'verify:meanDirs'),
    counts: put(new Uint32Array([regionCount, total]), 'verify:counts'),
    dispatchArgs: put(new Uint32Array([Math.ceil(regionCount / 64), 1, 1]), 'verify:dispatchArgs'),
    maxRegions: Math.max(regionCount, 1),
  };

  const enc = device.createCommandEncoder();
  const out = encodeLsdFit(arena, alloc, enc, {
    fx: fxS, fy: fyS, regions: set, w, h,
    rho: settings.rhoNoiseThreshold, toleranceDeg: settings.toleranceDeg, logNTests, logEpsilon,
  });
  device.queue.submit([enc.finish()]);
  const { results } = await readLsdFitResults(arena, out, set, 'lsd.fit');

  const err = await device.popErrorScope();
  if (err) {
    console.error('fitRegionsGPUFromCPU: WebGPU validation error --', err.message);
    return null;
  }
  return results;
}

// For callers that already hold a CPU gradient field and only want rectangles --
// the phone's sendDebugInfo pass, which serializes them as flat numbers and
// never needs the regions they were fitted from.
export function computeLsdRectanglesFromField(
  field: GradientField, settings: LsdSettings, backend: Backend,
): LsdRectangle[] {
  // `backend` is accepted and ignored: stage 1 has already happened on CPU here,
  // and running stages 2-4 on GPU would mean uploading fx/fy for one caller that
  // is off the pose path entirely. It stays in the signature because the two
  // call sites thread a backend everywhere and dropping it there reads as a
  // decision rather than an omission.
  void backend;
  return computeLsdRectangles(field, settings);
}
