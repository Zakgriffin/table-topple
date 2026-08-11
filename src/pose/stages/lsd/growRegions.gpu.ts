import { type Alloc, type Arena, type Slice, bind, sliceRange } from '../../gpu/arena.ts';
import { dispatchCount } from '../../gpu/device.ts';
import { GROW_REGIONS_WGSL } from './growRegions.wgsl.ts';
import { gpuTimelineSlot } from '../../gpu/gpuTimeline.ts';

interface GrowPipelines {
  bindGroupLayout: GPUBindGroupLayout;
  init: GPUComputePipeline;
  hook: GPUComputePipeline;
  compress: GPUComputePipeline;
}

const pipelineCache = new WeakMap<GPUDevice, GrowPipelines>();
function getPipelines(device: GPUDevice): GrowPipelines {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: GROW_REGIONS_WGSL, label: 'growRegions' });
    // An EXPLICIT layout, not `layout: 'auto'`, and this is load-bearing rather
    // than stylistic. 'auto' derives each pipeline's layout from the bindings
    // that entry point actually references, and none of these three references
    // all eight: init never touches `changed`, hook never touches fx/fy/changed,
    // compress never touches fx/fy/ux/uy. Binding one buffer set across three
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
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    p = {
      bindGroupLayout,
      init: device.createComputePipeline({ layout, compute: { module, entryPoint: 'init' }, label: 'growRegions.init' }),
      hook: device.createComputePipeline({ layout, compute: { module, entryPoint: 'hook' }, label: 'growRegions.hook' }),
      compress: device.createComputePipeline({ layout, compute: { module, entryPoint: 'compress' }, label: 'growRegions.compress' }),
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
// stages/lsd/chain.ts), which is also what will let the round budget become an
// indirect-dispatch early-out without this file changing at all.

export function sizeOfGrow(w: number, h: number): number {
  // ux, uy, label, next -- four n-element arrays. `changed` is 4 bytes and the
  // uniform is 32; both are rounded to a block by the arena anyway.
  return w * h * 4 * 4 + 4 + 32;
}

export interface GrowState {
  label: Slice;
  changed: Slice;
  // Everything below is scratch the rounds need and nothing outside does. Held
  // rather than returned piecemeal so the round encoder takes one argument.
  bindGroup: GPUBindGroup;
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
  const changed = alloc(4, 'grow.changed');

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

  // ONE bind group for all three entry points -- the shared explicit layout is
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

  const st: GrowState = { label, changed, bindGroup, gx: dispatchCount(w), gy: dispatchCount(h), pipelines };
  runPass(enc, st, 'init');
  return st;
}

function runPass(enc: GPUCommandEncoder, st: GrowState, which: 'init' | 'hook' | 'compress'): void {
  // A SEPARATE compute pass per dispatch, not several dispatches in one pass:
  // compress reads next[l] for an arbitrary l that some other thread wrote
  // during hook, and WebGPU only guarantees the storage-buffer memory barrier
  // BETWEEN passes.
  const pass = enc.beginComputePass(gpuTimelineSlot(`grow:${which}`));
  pass.setPipeline(st.pipelines[which]);
  pass.setBindGroup(0, st.bindGroup);
  pass.dispatchWorkgroups(st.gx, st.gy);
  pass.end();
}

// One hook+compress round. `clearChanged` must be set on the LAST round of a
// batch, not the first, so what the caller reads back is "did the final round
// change anything" rather than "did any round in the batch". That is the
// difference between detecting convergence one batch late and detecting it
// immediately: the round operator is deterministic in `label` alone, so a round
// that changes no label is a FIXPOINT -- every later round recomputes the
// identical `next` and the identical `label`. A quiet final round therefore
// proves convergence outright, even though earlier rounds in the same batch were
// still doing work.
export function encodeGrowRound(
  arena: Arena, enc: GPUCommandEncoder, st: GrowState, clearChanged: boolean,
): void {
  if (clearChanged) {
    // Through sliceRange, not off st.changed directly: reaching for
    // `slice.buffer` here would skip the generation check, which is the one way
    // a stale slice gets past the arena's only guard.
    const { buffer, offset, size } = sliceRange(st.changed, arena);
    enc.clearBuffer(buffer, offset, size);
  }
  runPass(enc, st, 'hook');
  runPass(enc, st, 'compress');
}
