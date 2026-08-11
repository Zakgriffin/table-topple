import { type Alloc, type Arena, type Slice, allocZeroed, bind, sliceRange } from '../../gpu/arena.ts';
import { dispatchCount } from '../../gpu/device.ts';
import { GROW_REGIONS_WGSL } from './growRegions.wgsl.ts';
import { gpuTimelineSlot } from '../../gpu/gpuTimeline.ts';

interface GrowPipelines {
  bindGroupLayout: GPUBindGroupLayout;
  // Group 1, bound only by `gate`. See the WGSL's own note: `args` must not be
  // in hook's or compress's usage scope, because those two dispatch indirectly
  // off it.
  argsLayout: GPUBindGroupLayout;
  init: GPUComputePipeline;
  hook: GPUComputePipeline;
  compress: GPUComputePipeline;
  gate: GPUComputePipeline;
}

const pipelineCache = new WeakMap<GPUDevice, GrowPipelines>();
function getPipelines(device: GPUDevice): GrowPipelines {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: GROW_REGIONS_WGSL, label: 'growRegions' });
    // An EXPLICIT layout, not `layout: 'auto'`, and this is load-bearing rather
    // than stylistic. 'auto' derives each pipeline's layout from the bindings
    // that entry point actually references, and none of these four references
    // all nine: init never touches `changed`, hook never touches fx/fy/changed,
    // compress never touches fx/fy/ux/uy, and `gate` touches only `changed` and
    // `args`. Binding one buffer set across four
    // auto layouts therefore fails validation on every createBindGroup -- and
    // WebGPU reports that ASYNCHRONOUSLY, so the invalid bind groups just poison
    // their encoders and every submit silently becomes a no-op. The symptom is
    // not an exception, it is a labeling of all zeros that looks like one
    // image-sized region. One shared layout makes the buffer set valid for all
    // three by construction.
    const storage = (i: number, ro: boolean): GPUBindGroupLayoutEntry => ({
      binding: i, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: ro ? 'read-only-storage' : 'storage' },
    });
    const bindGroupLayout = device.createBindGroupLayout({
      label: 'growRegions',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        storage(1, true), storage(2, true), // fx, fy
        storage(3, false), storage(4, false), // ux, uy (the normalized level line)
        storage(5, false), storage(6, false), // label, next
        storage(7, false), // changed (atomic)
      ],
    });
    const argsLayout = device.createBindGroupLayout({
      label: 'growRegions.args',
      entries: [storage(0, false)],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    // `gate` is the only entry point that references group 1, so it is the only
    // one whose pipeline layout mentions it.
    const gateLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout, argsLayout] });
    p = {
      bindGroupLayout, argsLayout,
      init: device.createComputePipeline({ layout, compute: { module, entryPoint: 'init' }, label: 'growRegions.init' }),
      hook: device.createComputePipeline({ layout, compute: { module, entryPoint: 'hook' }, label: 'growRegions.hook' }),
      compress: device.createComputePipeline({ layout, compute: { module, entryPoint: 'compress' }, label: 'growRegions.compress' }),
      gate: device.createComputePipeline({ layout: gateLayout, compute: { module, entryPoint: 'gate' }, label: 'growRegions.gate' }),
    };
    pipelineCache.set(device, p);
  }
  return p;
}

// GPU-resident counterpart to pose/stages/lsd/regions.cpu.ts's growRegionsCCL --
// the ROUND LOOP only. Hysteresis survival and the collect/relabel pass are a
// SEPARATE STAGE now (collectRegions.gpu.ts); they used to be called from inside
// this function behind a `collectOnGPU` boolean, which meant `label` was not a
// stage output on either backend and no harness could address the seam without
// that flag being threaded down to it.
//
// NOT bit-identical to the CPU path, and cannot be. The edge predicate
// `cos(theta_i - theta_j) >= cosTol` is evaluated here in f32 and there in f64,
// so a neighbour pair sitting very near exactly the tolerance can fall on
// opposite sides in the two paths. Unlike lsdFit's boundary-pixel problem
// (bounded: a miscounted pixel perturbs one region's own n and k), a single
// flipped EDGE here merges or splits whole components, and the difference
// cascades. Compare the two statistically -- region count, size distribution,
// how many regions match -- not exactly. See growRegionsCCL's own header for why
// the result is nonetheless well-defined: it is the transitive closure of a
// fixed relation, so each path converges to the exact connected components OF
// ITS OWN graph.
//
// ── The loop is the CALLER's ──
//
// This stage encodes passes. It does not submit, does not await, and does not
// read the convergence flag -- because the flag is a READBACK, and a stage that
// reads anything back cannot be composed into someone else's encoder. The
// caller batches rounds, submits, and decides when to stop (see
// stages/lsd/chain.ts).
//
// An earlier version of this comment predicted the early-out would arrive
// "without this file changing at all." That was WRONG and is worth keeping as a
// correction: moving the stopping decision onto the device needed a fourth entry
// point, a ninth binding and indirect dispatch, all of them here. What the
// caller-owned loop actually bought was that the change stayed INSIDE the round
// -- `encodeGrowRound` still encodes one round and the chain still decides how
// many to encode. The seam held; the stage was not free.

export function sizeOfGrow(w: number, h: number): number {
  // ux, uy, label, next -- four n-element arrays. `changed` is 4 bytes, `args`
  // is 16 and the uniform is 32; all are rounded to a block by the arena anyway.
  return w * h * 4 * 4 + 4 + 16 + 32;
}

export interface GrowState {
  label: Slice;
  // [x, y, z, activeRounds]. The host reads all four: x === 0 means converged,
  // and activeRounds is how many rounds actually changed a label. This is the
  // ONLY thing a caller reads back from the round loop now -- `changed` moved
  // below the line with the rest of the scratch when `gate` took over clearing
  // it, because nothing outside the stage has a reason to look at it.
  args: Slice;
  // Everything below is scratch the rounds need and nothing outside does. Held
  // rather than returned piecemeal so the round encoder takes one argument.
  changed: Slice;
  bindGroup: GPUBindGroup;
  argsBindGroup: GPUBindGroup;
  gx: number;
  gy: number;
  pipelines: GrowPipelines;
}

// Allocates, builds the bind group, and encodes the seeding pass. The caller
// submits when it likes -- including in the same submission as the first batch
// of rounds.
export function encodeGrowInit(
  arena: Arena, alloc: Alloc, enc: GPUCommandEncoder,
  inp: { fx: Slice; fy: Slice; w: number; h: number; toleranceDeg: number; rhoLow: number },
): GrowState {
  const { fx, fy, w, h, toleranceDeg, rhoLow } = inp;
  const device = arena.device;
  const pipelines = getPipelines(device);
  const n = w * h;

  const ux = alloc(n * 4, 'grow.ux');
  const uy = alloc(n * 4, 'grow.uy');
  const label = alloc(n * 4, 'grow.label');
  const next = alloc(n * 4, 'grow.next');
  // ZEROED, and it is not optional. `compress` only ever stores 1 and `gate`
  // only ever stores 0, so nothing writes this before the first round reads it
  // -- under the arena that first read would be the PREVIOUS reconstruction's
  // flag. Inheriting a 1 costs one wasted round; inheriting a 0 makes `gate`
  // converge before a single round has run and the labeling stays at init's
  // singletons, i.e. one region per eligible pixel.
  const changed = allocZeroed(arena, alloc, enc, 4, 'grow.changed');

  const gx = dispatchCount(w), gy = dispatchCount(h);
  // The SAME failure class, and the more dangerous half of it: a converged run
  // leaves this at [0,0,0], so a second reconstruction that inherited it would
  // dispatch zero workgroups for every round and emit init's singleton labeling
  // -- silently, at full speed, with a plausible-looking region count. Written
  // rather than cleared, because the correct initial state is not zero.
  const args = alloc(16, 'grow.args');
  device.queue.writeBuffer(args.buffer, args.offset, new Uint32Array([gx, gy, 1, 0]));

  const uni = alloc(32, 'grow.uniforms');
  {
    const buf = new ArrayBuffer(32);
    const dv = new DataView(buf);
    dv.setUint32(0, w, true); dv.setUint32(4, h, true);
    // Squared, matching levelLine.ts's eligibilityThresholdSq (negative case
    // included) -- the shader's init tests squared magnitude so an ineligible
    // pixel costs no sqrt.
    dv.setFloat32(16, rhoLow >= 0 ? rhoLow * rhoLow : -Infinity, true);
    dv.setFloat32(20, Math.cos((toleranceDeg * Math.PI) / 180), true);
    device.queue.writeBuffer(uni.buffer, uni.offset, buf);
  }

  // ONE bind group for all four entry points -- the shared explicit layout is
  // what makes that legal.
  const bindGroup = device.createBindGroup({
    layout: pipelines.bindGroupLayout,
    entries: [
      { binding: 0, resource: bind(uni, arena) },
      { binding: 1, resource: bind(fx, arena) },
      { binding: 2, resource: bind(fy, arena) },
      { binding: 3, resource: bind(ux, arena) },
      { binding: 4, resource: bind(uy, arena) },
      { binding: 5, resource: bind(label, arena) },
      { binding: 6, resource: bind(next, arena) },
      { binding: 7, resource: bind(changed, arena) },
    ],
  });
  const argsBindGroup = device.createBindGroup({
    layout: pipelines.argsLayout,
    entries: [{ binding: 0, resource: bind(args, arena) }],
  });

  const st: GrowState = { label, changed, args, bindGroup, argsBindGroup, gx, gy, pipelines };
  // init dispatches DIRECTLY. It must run over the whole image exactly once
  // whatever `args` says, and it is what establishes the state the first round
  // reads -- gating it on the early-out would be gating it on the previous
  // reconstruction.
  runPass(arena, enc, st, 'init', 'direct');
  return st;
}

function runPass(
  arena: Arena, enc: GPUCommandEncoder, st: GrowState,
  which: 'init' | 'hook' | 'compress' | 'gate', dispatch: 'direct' | 'indirect' | 'single',
): void {
  // A SEPARATE compute pass per dispatch, not several dispatches in one pass:
  // compress reads next[l] for an arbitrary l that some other thread wrote
  // during hook, and WebGPU only guarantees the storage-buffer memory barrier
  // BETWEEN passes. `gate` needs the same treatment for the same reason -- it
  // reads the flag compress wrote.
  const pass = enc.beginComputePass(gpuTimelineSlot(`grow:${which}`));
  pass.setPipeline(st.pipelines[which]);
  pass.setBindGroup(0, st.bindGroup);
  if (which === 'gate') pass.setBindGroup(1, st.argsBindGroup);
  if (dispatch === 'indirect') {
    // Through sliceRange rather than st.args.buffer, so the generation check
    // still runs -- reaching for `.buffer` is the one way past the arena's guard.
    const a = sliceRange(st.args, arena);
    pass.dispatchWorkgroupsIndirect(a.buffer, a.offset);
  } else if (dispatch === 'single') {
    pass.dispatchWorkgroups(1);
  } else {
    pass.dispatchWorkgroups(st.gx, st.gy);
  }
  pass.end();
}

// One round: hook, compress, gate. UNCONDITIONAL and identical every time --
// which is the point of the early-out. A round encoded after the fixpoint
// dispatches zero workgroups for hook and compress and runs a single-thread
// gate that re-zeroes an already-zero `args`, so the caller can encode a whole
// batch without knowing where convergence falls.
//
// The `clearChanged` parameter this used to take is gone. It had to be set on
// the LAST round of a batch so the flag meant "did the final round change
// anything" rather than "did any round in the batch" -- which made convergence a
// property of where the caller drew its batch boundaries. `gate` clears the flag
// per round instead, so the fixpoint is detected on the round it happens.
//
// The underlying argument is unchanged and is what makes one quiet round
// sufficient: the round operator is deterministic in `label` alone, so a round
// that changes no label is a FIXPOINT -- every later round would recompute the
// identical `next` and the identical `label`.
export function encodeGrowRound(arena: Arena, enc: GPUCommandEncoder, st: GrowState): void {
  runPass(arena, enc, st, 'hook', 'indirect');
  runPass(arena, enc, st, 'compress', 'indirect');
  runPass(arena, enc, st, 'gate', 'single');
}
