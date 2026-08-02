import { collectRegionsFromLabels, GrownRegion } from '../pipeline/lsdSegments.ts';
import {
  createStorageBuffer, dispatchCount, getGPUDevice, readUint32, uploadFloat32, uploadUniform,
} from './device.ts';
import { GROW_REGIONS_WGSL } from './growRegions.wgsl.ts';
import { collectRegionsGPU } from './collectRegions.ts';
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
// Returns null if WebGPU isn't available; caller falls back to the CPU version,
// which stays the source of truth.
export async function growRegionsCCLGPU(
  mag: Float64Array, theta: Float64Array, w: number, h: number,
  toleranceDeg: number, rhoLow: number, rhoHigh: number, maxRounds: number, minRegionSize: number,
): Promise<{ regionId: Int32Array; regions: GrownRegion[]; roundsRun: number; converged: boolean } | null> {
  const device = await getGPUDevice();
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
  const magBuf = uploadFloat32(device, new Float32Array(mag));
  const thetaBuf = uploadFloat32(device, new Float32Array(theta));
  const cosBuf = createStorageBuffer(device, n * 4);
  const sinBuf = createStorageBuffer(device, n * 4);
  const labelBuf = createStorageBuffer(device, n * 4);
  const nextBuf = createStorageBuffer(device, n * 4);
  const changedBuf = createStorageBuffer(device, 4);

  const uni = new ArrayBuffer(32);
  const dv = new DataView(uni);
  dv.setUint32(0, w, true); dv.setUint32(4, h, true);
  dv.setFloat32(16, rhoLow, true);
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

  // Hysteresis + collect. Two routes to the same result:
  //
  // GPU -- collectRegionsGPU consumes label/mag/theta WHERE THEY ALREADY ARE,
  // so the labeling never crosses the bus. This is the step that makes stages
  // 1-4 closable into one resident run; see collectRegions.wgsl.ts for why its
  // region numbering and member ordering are exact rather than merely
  // equivalent.
  //
  // CPU -- reads the labels back and runs collectRegionsFromLabels, shared
  // verbatim with the pure-CPU path so the two can never drift.
  let collected: { regionId: Int32Array; regions: GrownRegion[] } | null = null;
  if (globalState.useGPUCollectRegions) {
    collected = await collectRegionsGPU(device, labelBuf, magBuf, thetaBuf, n, rhoHigh, minRegionSize);
  }
  if (!collected) {
    const labels32 = await readUint32(device, labelBuf, n * 4);
    const label = new Int32Array(labels32.buffer.slice(0));
    collected = collectRegionsFromLabels(label, mag, theta, rhoHigh, n, minRegionSize);
  }

  for (const b of [magBuf, thetaBuf, cosBuf, sinBuf, labelBuf, nextBuf, changedBuf, uniBuf]) b.destroy();

  const err = await device.popErrorScope();
  if (err) {
    console.error('growRegionsCCLGPU: WebGPU validation error, falling back to CPU --', err.message);
    return null;
  }

  return { regionId: collected.regionId, regions: collected.regions, roundsRun, converged };
}
