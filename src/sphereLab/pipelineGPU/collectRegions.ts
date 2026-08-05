import {
  COLLECT_FINALIZE_WGSL, COLLECT_HISTOGRAM_WGSL, COLLECT_MARK_KEPT_WGSL,
  COLLECT_REGION_META_WGSL, COLLECT_SCATTER_WGSL, COLLECT_SURVIVE_WGSL,
} from './collectRegions.wgsl.ts';
import { createStorageBuffer, readUint32, uploadUniform } from './device.ts';
import { FieldResidency } from './fieldResidency.ts';
import { encodeExclusiveScan } from './prefixSum.ts';

const WG = 256;
const up = (n: number, w = WG) => Math.ceil(n / w);

interface Stage { layout: GPUBindGroupLayout; pipeline: GPUComputePipeline }
interface Pipelines {
  survive: Stage; histogram: Stage; markKept: Stage;
  regionMeta: Stage; scatter: Stage; finalize: Stage;
}

const RO = 'read-only-storage' as const, RW = 'storage' as const, UNI = 'uniform' as const;
const cache = new WeakMap<GPUDevice, Pipelines>();

function makeStage(device: GPUDevice, code: string, entryPoint: string, types: GPUBufferBindingType[]): Stage {
  const layout = device.createBindGroupLayout({
    label: `collectRegions.${entryPoint}`,
    entries: types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })),
  });
  return {
    layout,
    pipeline: device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: device.createShaderModule({ code, label: entryPoint }), entryPoint },
      label: `collectRegions.${entryPoint}`,
    }),
  };
}

function getPipelines(device: GPUDevice): Pipelines {
  let p = cache.get(device);
  if (!p) {
    p = {
      survive: makeStage(device, COLLECT_SURVIVE_WGSL, 'survive', [UNI, RO, RO, RO, RW]),
      histogram: makeStage(device, COLLECT_HISTOGRAM_WGSL, 'histogram', [UNI, RO, RO, RW]),
      markKept: makeStage(device, COLLECT_MARK_KEPT_WGSL, 'markKept', [UNI, RO, RO, RW, RW]),
      regionMeta: makeStage(device, COLLECT_REGION_META_WGSL, 'regionMeta', [UNI, RO, RO, RO, RO, RW, RW]),
      scatter: makeStage(device, COLLECT_SCATTER_WGSL, 'scatter', [UNI, RO, RO, RO, RO, RW, RW, RW]),
      finalize: makeStage(device, COLLECT_FINALIZE_WGSL, 'finalize', [UNI, RO, RO, RO, RW, RW, RO]),
    };
    cache.set(device, p);
  }
  return p;
}

// GPU-resident counterpart to pipeline/lsdSegments.ts's collectRegionsFromLabels.
//
// Reads label/mag/theta out of the residency (already on device when stage 3
// ran there) and PUBLISHES regionId plus the region CSR back into it, still on
// device. Same region numbering and same member ordering as the CPU version --
// see collectRegions.wgsl.ts for why both are exact rather than merely
// equivalent.
//
// Two submissions, not one, and the split is forced: the region count only
// exists after the first scan, and it is what sizes the `finalize` dispatch.
// Those two counts are the ONLY things this function still reads back, 8 bytes
// total. Everything else it produces is left where the fitter can consume it:
// the ~800KB regionId array is pure debug-overlay data that the production path
// never looks at, and the CSR is stage 4's direct input. Both used to come down
// unconditionally, which is most of why useGPUCollectRegions measured as a loss.
//
// Returns false if WebGPU is unavailable or a dispatch failed validation, in
// which case NOTHING has been published and the caller is free to run the CPU
// collect into the same residency.
export async function collectRegionsGPU(
  res: FieldResidency, rhoHigh: number, minRegionSize: number,
): Promise<boolean> {
  const device = res.device;
  if (!device) return false;
  const n = res.n;
  const labelBuf = res.gpu('label'), fxBuf = res.gpu('fx'), fyBuf = res.gpu('fy');

  device.pushErrorScope('validation');
  const p = getPipelines(device);

  const uni = (() => {
    const b = new ArrayBuffer(16);
    const dv = new DataView(b);
    // Squared here, once, rather than in the shader per pixel. Matches
    // lsdSegments.ts's eligibilityThresholdSq, including the negative case.
    dv.setUint32(0, n, true);
    dv.setFloat32(4, rhoHigh >= 0 ? rhoHigh * rhoHigh : -Infinity, true);
    dv.setUint32(8, minRegionSize, true);
    return uploadUniform(device, b);
  })();

  // Every per-LABEL array is size n, because labels are pixel indices. That is
  // ~6 arrays x 4 bytes x n; at 196608 pixels it is under 5MB total, which is
  // the price of never hashing or compacting the label space.
  const labelSurvives = createStorageBuffer(device, n * 4);
  const counts = createStorageBuffer(device, n * 4);
  const keptFlag = createStorageBuffer(device, n * 4);
  const keptCount = createStorageBuffer(device, n * 4);
  const regionIndex = createStorageBuffer(device, n * 4);
  const memberOffset = createStorageBuffer(device, n * 4);
  const cursor = createStorageBuffer(device, n * 4);
  const totalRegions = createStorageBuffer(device, 4);
  const totalMembers = createStorageBuffer(device, 4);

  // Split by DESTINATION, not by lifetime: `scratch` dies with this call,
  // `handoff` is what gets published into the residency on success (and has to
  // be destroyed by hand on every failure path, since nothing else has taken
  // ownership of it yet).
  const members = createStorageBuffer(device, n * 4);
  const regionId = createStorageBuffer(device, n * 4);
  const regionOffsets = createStorageBuffer(device, n * 4);
  const regionSizes = createStorageBuffer(device, n * 4);
  const scratch = [
    uni, labelSurvives, counts, keptFlag, keptCount, regionIndex, memberOffset,
    cursor, totalRegions, totalMembers,
  ];
  const handoff = [members, regionId, regionOffsets, regionSizes];

  const run = (encoder: GPUCommandEncoder, stage: Stage, resources: GPUBuffer[], groups: number) => {
    const bindGroup = device.createBindGroup({
      layout: stage.layout,
      entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const cp = encoder.beginComputePass();
    cp.setPipeline(stage.pipeline);
    cp.setBindGroup(0, bindGroup);
    cp.dispatchWorkgroups(groups);
    cp.end();
  };

  const encoder = device.createCommandEncoder();
  run(encoder, p.survive, [uni, labelBuf, fxBuf, fyBuf, labelSurvives], up(n));
  run(encoder, p.histogram, [uni, labelBuf, labelSurvives, counts], up(n));
  run(encoder, p.markKept, [uni, labelSurvives, counts, keptFlag, keptCount], up(n));
  const temps = [
    ...encodeExclusiveScan(device, encoder, keptFlag, regionIndex, totalRegions, n),
    ...encodeExclusiveScan(device, encoder, keptCount, memberOffset, totalMembers, n),
  ];
  run(encoder, p.regionMeta, [uni, keptFlag, regionIndex, memberOffset, counts, regionOffsets, regionSizes], up(n));
  run(encoder, p.scatter, [uni, labelBuf, keptFlag, regionIndex, memberOffset, cursor, members, regionId], up(n));
  device.queue.submit([encoder.finish()]);

  const [rTot, mTot] = await Promise.all([
    readUint32(device, totalRegions, 4, 'collect:regionCount'),
    readUint32(device, totalMembers, 4, 'collect:memberCount'),
  ]);
  const regionCount = rTot[0], memberCount = mTot[0];

  // Sized max(regionCount, 1) so RegionSetGPU is a complete, bindable set even
  // for an empty capture -- a zero-length storage buffer is not a legal binding,
  // and making the consumer branch on emptiness instead would push this special
  // case into every downstream stage.
  // vec2<f32> per region -- a direction, not an angle, since nothing downstream
  // wants the angle and the fitter would only take cos/sin of it again.
  const meanDirs = createStorageBuffer(device, Math.max(regionCount, 1) * 8);
  handoff.push(meanDirs);
  if (regionCount > 0) {
    const fUni = (() => {
      const b = new ArrayBuffer(16);
      new DataView(b).setUint32(0, regionCount, true);
      return uploadUniform(device, b);
    })();
    scratch.push(fUni);
    const enc2 = device.createCommandEncoder();
    run(enc2, p.finalize, [fUni, fxBuf, regionOffsets, regionSizes, members, meanDirs, fyBuf], up(regionCount, 64));
    device.queue.submit([enc2.finish()]);
  }

  for (const b of [...scratch, ...temps]) b.destroy();
  const err = await device.popErrorScope();
  if (err) {
    console.error('collectRegionsGPU: WebGPU validation error, falling back to CPU --', err.message);
    for (const b of handoff) b.destroy();
    return false;
  }

  res.provideGPU('regionId', regionId);
  res.provideRegionsGPU({ offsets: regionOffsets, sizes: regionSizes, members, meanDirs, regionCount, memberCount });
  return true;
}
