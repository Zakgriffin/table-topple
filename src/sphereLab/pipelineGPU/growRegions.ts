import { collectRegionsFromLabels, GrownRegion } from '../pipeline/lsdSegments.ts';
import {
  createStorageBuffer, dispatchCount, getGPUDevice, readUint32, uploadFloat32, uploadUniform,
} from './device.ts';
import { GROW_REGIONS_WGSL } from './growRegions.wgsl.ts';

const pipelineCache = new WeakMap<GPUDevice, Record<string, GPUComputePipeline>>();
function getPipelines(device: GPUDevice): Record<string, GPUComputePipeline> {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: GROW_REGIONS_WGSL, label: 'growRegions' });
    p = {
      init: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'init' }, label: 'growRegions.init' }),
      hook: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'hook' }, label: 'growRegions.hook' }),
      compress: device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'compress' }, label: 'growRegions.compress' }),
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
// the ROUND LOOP only (stage 2+3's fixpoint iteration). Hysteresis survival and
// the collect/relabel pass stay on CPU: both are inherently serial, both run
// exactly once after the last round rather than per round, and the label buffer
// has to come back across for them regardless.
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
  toleranceDeg: number, rhoLow: number, rhoHigh: number, maxRounds: number,
): Promise<{ regionId: Int32Array; regions: GrownRegion[]; roundsRun: number; converged: boolean } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
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
  // 'auto' layout produces a DISTINCT bind group layout per pipeline even when
  // the declared bindings are identical, so each entry point needs its own
  // bind group over the same buffers.
  const bg = {
    init: device.createBindGroup({ layout: pipelines.init.getBindGroupLayout(0), entries }),
    hook: device.createBindGroup({ layout: pipelines.hook.getBindGroupLayout(0), entries }),
    compress: device.createBindGroup({ layout: pipelines.compress.getBindGroupLayout(0), entries }),
  };
  const gx = dispatchCount(w), gy = dispatchCount(h);
  const runPass = (encoder: GPUCommandEncoder, which: 'init' | 'hook' | 'compress') => {
    // A SEPARATE compute pass per dispatch, not several dispatches in one pass:
    // compress reads next[l] for an arbitrary l that some other thread wrote
    // during hook, and WebGPU only guarantees the storage-buffer memory barrier
    // BETWEEN passes.
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipelines[which]);
    pass.setBindGroup(0, bg[which]);
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
  const zero = new Uint32Array([0]);

  while (roundsRun < cap) {
    const batch = Math.min(ROUNDS_PER_BATCH, cap - roundsRun);
    device.queue.writeBuffer(changedBuf, 0, zero);
    const encoder = device.createCommandEncoder();
    for (let r = 0; r < batch; r++) { runPass(encoder, 'hook'); runPass(encoder, 'compress'); }
    device.queue.submit([encoder.finish()]);
    roundsRun += batch;
    const flag = await readUint32(device, changedBuf, 4);
    // The flag is per-BATCH, so convergence is only detectable at batch
    // granularity: a batch that converges on its 3rd round still reports
    // "changed" and costs one extra (entirely no-op) batch to confirm. roundsRun
    // is therefore an upper bound on the rounds that did real work, which is why
    // it is not compared against the CPU path's own count.
    if (flag[0] === 0) { converged = true; break; }
  }

  const labels32 = await readUint32(device, labelBuf, n * 4);
  const label = new Int32Array(labels32.buffer.slice(0));

  for (const b of [magBuf, thetaBuf, cosBuf, sinBuf, labelBuf, nextBuf, changedBuf, uniBuf]) b.destroy();

  // Hysteresis + collect, shared verbatim with the CPU path so the two can
  // never drift in how a finished labeling becomes regions.
  const { regionId, regions } = collectRegionsFromLabels(label, mag, theta, rhoHigh, n);
  return { regionId, regions, roundsRun, converged };
}
