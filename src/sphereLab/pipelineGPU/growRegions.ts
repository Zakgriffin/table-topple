import { collectRegionsFromLabels, GrownRegion } from '../pipeline/lsdSegments.ts';
import {
  createStorageBuffer, dispatchCount, readUint32, uploadUniform,
} from './device.ts';
import { GROW_REGIONS_WGSL } from './growRegions.wgsl.ts';
import { collectRegionsGPU } from './collectRegions.ts';
import { FieldResidency } from './fieldResidency.ts';
import { globalState } from '../state.ts';

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
    // all eight: init never touches `changed`, hook never touches mag/theta/
    // changed, compress never touches mag/theta/cos/sin. Binding one buffer set
    // across three auto layouts therefore fails validation on every
    // createBindGroup -- and WebGPU reports that ASYNCHRONOUSLY, so the invalid
    // bind groups just poison their encoders and every submit silently becomes
    // a no-op. The symptom is not an exception, it's a labeling of all zeros
    // that looks like one image-sized region. One shared layout makes the
    // buffer set valid for all three by construction.
    const storage = (i: number, ro: boolean): GPUBindGroupLayoutEntry => ({
      binding: i, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: ro ? 'read-only-storage' : 'storage' },
    });
    const bindGroupLayout = device.createBindGroupLayout({
      label: 'growRegions',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        storage(1, true), storage(2, true), // mag, theta
        storage(3, false), storage(4, false), // cosT, sinT
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

// How many hook+compress rounds get encoded into ONE command submission before
// the convergence flag is read back. The flag lives on the GPU, so checking it
// every round would mean a GPU->CPU sync per round -- and at ~log(L) rounds
// that latency dominates the actual work, which is only two cheap per-pixel
// passes. Batching amortizes it: the cost of overshooting is at most
// (ROUNDS_PER_BATCH - 1) no-op rounds, each of which is far cheaper than one
// extra readback stall.
const ROUNDS_PER_BATCH = 8;

// GPU-resident counterpart to pipeline/lsdSegments.ts's growRegionsCCL, for
// the ROUND LOOP (stage 2+3's fixpoint iteration). Hysteresis survival and the
// collect/relabel pass follow it, on GPU when globalState.useGPUCollectRegions
// is set (pipelineGPU/collectRegions.ts, consuming label/mag/theta where they
// already sit) and on CPU otherwise.
//
// NOT bit-identical to the CPU path, and can't be. The edge predicate is
// `cos(theta_i - theta_j) >= cosTol`, evaluated here in f32 and there in f64 --
// so a neighbour pair sitting very near exactly tau can fall on opposite sides
// in the two paths. Unlike lsdFit's boundary-pixel problem (bounded: a
// miscounted pixel perturbs one region's own n and k), a single flipped EDGE
// here merges or splits whole components, and the difference cascades. Compare
// the two statistically -- region count, size distribution, how many regions
// match -- not exactly. See growRegionsCCL's own header for why the result is
// nonetheless well-defined: it's the transitive closure of a fixed relation, so
// each path converges to the exact connected components OF ITS OWN graph.
//
// Takes mag/theta and returns label/regionId/regions THROUGH THE RESIDENCY
// (pipelineGPU/fieldResidency.ts) rather than as CPU arrays, which is what lets
// this stage sit next to the fitter without either of them landing the field on
// CPU in between. `res.gpu('fx')` returns the upstream stage's own buffer when
// there is one and uploads only when there isn't, so the duplicate mag/theta
// upload this module and lsdFit.ts each used to do is now structurally
// impossible rather than merely avoided.
//
// Returns null if the residency has no device or a dispatch failed validation;
// caller falls back to the CPU version, which stays the source of truth. On
// that path nothing has been published, so the fallback is free to publish its
// own results into the same residency.
export async function growRegionsCCLGPU(
  res: FieldResidency, w: number, h: number,
  toleranceDeg: number, rhoLow: number, rhoHigh: number, maxRounds: number, minRegionSize: number,
): Promise<{ roundsRun: number; converged: boolean } | null> {
  const device = res.device;
  if (!device) return null;
  // Everything from pipeline creation to the last submit runs inside a
  // validation error scope. WebGPU reports validation failures asynchronously
  // -- an invalid bind group or pipeline does not throw, it just makes every
  // command that uses it a silent no-op -- so without this, a layout mistake
  // surfaces as PLAUSIBLE-LOOKING GARBAGE (see getPipelines above for the one
  // that actually happened) rather than as a failure. Popping the scope turns
  // that class of bug into an honest fall back to the CPU grower.
  device.pushErrorScope('validation');
  const pipelines = getPipelines(device);

  const n = w * h;
  const magBuf = res.gpu('fx');
  const thetaBuf = res.gpu('fy');
  const cosBuf = createStorageBuffer(device, n * 4);
  const sinBuf = createStorageBuffer(device, n * 4);
  const labelBuf = createStorageBuffer(device, n * 4);
  const nextBuf = createStorageBuffer(device, n * 4);
  const changedBuf = createStorageBuffer(device, 4);

  const uni = new ArrayBuffer(32);
  const dv = new DataView(uni);
  dv.setUint32(0, w, true); dv.setUint32(4, h, true);
  // Squared, matching lsdSegments.ts's eligibilityThresholdSq (negative case
  // included) -- the shader's init tests squared magnitude so an ineligible
  // pixel costs no sqrt.
  dv.setFloat32(16, rhoLow >= 0 ? rhoLow * rhoLow : -Infinity, true);
  dv.setFloat32(20, Math.cos((toleranceDeg * Math.PI) / 180), true);
  const uniBuf = uploadUniform(device, uni);

  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: uniBuf } },
    { binding: 1, resource: { buffer: magBuf } },
    { binding: 2, resource: { buffer: thetaBuf } },
    { binding: 3, resource: { buffer: cosBuf } },
    { binding: 4, resource: { buffer: sinBuf } },
    { binding: 5, resource: { buffer: labelBuf } },
    { binding: 6, resource: { buffer: nextBuf } },
    { binding: 7, resource: { buffer: changedBuf } },
  ];
  // ONE bind group for all three entry points -- the shared explicit layout is
  // what makes that legal.
  const bg = device.createBindGroup({ layout: pipelines.bindGroupLayout, entries });
  const gx = dispatchCount(w), gy = dispatchCount(h);
  const runPass = (encoder: GPUCommandEncoder, which: 'init' | 'hook' | 'compress') => {
    // A SEPARATE compute pass per dispatch, not several dispatches in one pass:
    // compress reads next[l] for an arbitrary l that some other thread wrote
    // during hook, and WebGPU only guarantees the storage-buffer memory barrier
    // BETWEEN passes.
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines[which]);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
  };

  {
    const encoder = device.createCommandEncoder();
    runPass(encoder, 'init');
    device.queue.submit([encoder.finish()]);
  }

  // Same bound the CPU path uses -- hook alone propagates one pixel per round,
  // so the longest possible chain caps it; compression makes the real count
  // logarithmic, so this is never reached in practice.
  const hardCap = w + h + 64;
  const cap = maxRounds > 0 ? Math.min(maxRounds, hardCap) : hardCap;
  let roundsRun = 0, converged = false;

  while (roundsRun < cap) {
    const batch = Math.min(ROUNDS_PER_BATCH, cap - roundsRun);
    const encoder = device.createCommandEncoder();
    for (let r = 0; r < batch; r++) {
      // Clear the flag before the batch's LAST round, not before the batch, so
      // what comes back is "did the final round change anything" rather than
      // "did any round in the batch". That's the difference between detecting
      // convergence one batch late and detecting it immediately: the round
      // operator is deterministic in `label` alone, so a round that changes no
      // label is a FIXPOINT -- every later round recomputes the identical
      // `next` and the identical `label`. A quiet final round therefore proves
      // convergence outright, even though earlier rounds in the same batch were
      // still doing work.
      if (r === batch - 1) encoder.clearBuffer(changedBuf);
      runPass(encoder, 'hook');
      runPass(encoder, 'compress');
    }
    device.queue.submit([encoder.finish()]);
    roundsRun += batch;
    const flag = await readUint32(device, changedBuf, 4);
    // Still batch-granular in the sense that we always RUN a whole batch --
    // convergence at round 7 of 8 costs the 8th round, which is what proves it.
    // So roundsRun stays an upper bound on the rounds that did real work, and
    // is not comparable to the CPU path's own count.
    if (flag[0] === 0) { converged = true; break; }
  }

  // Popped BEFORE anything is published, and that ordering is load-bearing. On
  // a validation failure the caller falls back to the CPU grower, which
  // publishes ITS regionId/regions into this same residency -- so every failure
  // path here has to leave the residency exactly as it found it, or the
  // single-assignment invariant fires on the fallback's own provide() and turns
  // a recoverable GPU fault into a thrown exception. Note mag/theta are NOT in
  // the destroy list on either path: the residency owns them, and this stage is
  // only borrowing.
  const scratch = [cosBuf, sinBuf, nextBuf, changedBuf, uniBuf];
  const err = await device.popErrorScope();
  if (err) {
    console.error('growRegionsCCLGPU: WebGPU validation error, falling back to CPU --', err.message);
    for (const b of [...scratch, labelBuf]) b.destroy();
    return null;
  }

  // label is stage 3's output and stage 3b's only input. Handing it over rather
  // than reading it back is the whole point of the port: with collect also on
  // GPU it never crosses the bus, and with collect on CPU the residency does
  // the readback in one place and records it.
  res.provideGPU('label', labelBuf);

  // Hysteresis + collect. Two routes to the same result:
  //
  // GPU -- collectRegionsGPU consumes label/mag/theta WHERE THEY ALREADY ARE
  // and publishes the region CSR device-side, so neither the labeling nor the
  // members cross the bus. See collectRegions.wgsl.ts for why its region
  // numbering and member ordering are exact rather than merely equivalent.
  //
  // CPU -- pulls the labels down through the residency and runs
  // collectRegionsFromLabels, shared verbatim with the pure-CPU path so the two
  // can never drift.
  let collected = false;
  if (globalState.useGPUCollectRegions) {
    collected = await collectRegionsGPU(res, rhoHigh, minRegionSize);
  }
  if (!collected) {
    const { regionId, regions } = collectRegionsFromLabels(
      await res.cpuI32('label'), await res.cpuF64('fx'), await res.cpuF64('fy'),
      rhoHigh, n, minRegionSize,
    );
    res.provideCPU('regionId', regionId);
    res.provideRegionsCPU(regions);
  }

  for (const b of scratch) b.destroy();
  return { roundsRun, converged };
}

// The old all-on-CPU signature, preserved for the verify harnesses
// (growRegionsVerify.ts, collectRegionsVerify.ts). They compare CPU arrays
// element by element, so they genuinely want every readback -- which is exactly
// what asking the residency for the CPU side of everything expresses. Nothing
// in the production path should use this.
export async function growRegionsCCLGPUToCPU(
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  toleranceDeg: number, rhoLow: number, rhoHigh: number, maxRounds: number, minRegionSize: number,
): Promise<{ regionId: Int32Array; regions: GrownRegion[]; roundsRun: number; converged: boolean } | null> {
  const res = await FieldResidency.create(w * h, true);
  try {
    res.provideCPU('fx', fx);
    res.provideCPU('fy', fy);
    const grown = await growRegionsCCLGPU(res, w, h, toleranceDeg, rhoLow, rhoHigh, maxRounds, minRegionSize);
    if (!grown) return null;
    return { regionId: await res.cpuI32('regionId'), regions: await res.regionsCPU(), ...grown };
  } finally {
    res.destroy();
  }
}
