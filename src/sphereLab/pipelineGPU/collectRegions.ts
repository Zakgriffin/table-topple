import { GrownRegion } from '../pipeline/lsdSegments.ts';
import {
  COLLECT_FINALIZE_WGSL, COLLECT_HISTOGRAM_WGSL, COLLECT_MARK_KEPT_WGSL,
  COLLECT_REGION_META_WGSL, COLLECT_SCATTER_WGSL, COLLECT_SURVIVE_WGSL,
} from './collectRegions.wgsl.ts';
import { createStorageBuffer, readFloat32, readUint32, uploadUniform } from './device.ts';
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
      survive: makeStage(device, COLLECT_SURVIVE_WGSL, 'survive', [UNI, RO, RO, RW]),
      histogram: makeStage(device, COLLECT_HISTOGRAM_WGSL, 'histogram', [UNI, RO, RO, RW]),
      markKept: makeStage(device, COLLECT_MARK_KEPT_WGSL, 'markKept', [UNI, RO, RO, RW, RW]),
      regionMeta: makeStage(device, COLLECT_REGION_META_WGSL, 'regionMeta', [UNI, RO, RO, RO, RO, RW, RW]),
      scatter: makeStage(device, COLLECT_SCATTER_WGSL, 'scatter', [UNI, RO, RO, RO, RO, RW, RW, RW]),
      finalize: makeStage(device, COLLECT_FINALIZE_WGSL, 'finalize', [UNI, RO, RO, RO, RW, RW]),
    };
    cache.set(device, p);
  }
  return p;
}

// GPU-resident counterpart to pipeline/lsdSegments.ts's collectRegionsFromLabels.
//
// Takes buffers the caller owns (label/mag/theta already on device) and returns
// the same regionId + regions the CPU version does, with the SAME region
// numbering and the SAME member ordering -- see collectRegions.wgsl.ts for why
// both are exact rather than merely equivalent.
//
// Two submissions, not one, and the split is forced: the region count only
// exists after the first scan, and it is what sizes the `finalize` dispatch and
// every readback. Everything before that point is encoded together.
//
// Returns null if WebGPU is unavailable or a dispatch failed validation.
export async function collectRegionsGPU(
  device: GPUDevice,
  labelBuf: GPUBuffer, magBuf: GPUBuffer, thetaBuf: GPUBuffer,
  n: number, rhoHigh: number, minRegionSize: number,
): Promise<{ regionId: Int32Array; regions: GrownRegion[] } | null> {
  device.pushErrorScope('validation');
  const p = getPipelines(device);

  const uni = (() => {
    const b = new ArrayBuffer(16);
    const dv = new DataView(b);
    dv.setUint32(0, n, true); dv.setFloat32(4, rhoHigh, true); dv.setUint32(8, minRegionSize, true);
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
  const members = createStorageBuffer(device, n * 4);
  const regionId = createStorageBuffer(device, n * 4);
  const regionOffsets = createStorageBuffer(device, n * 4);
  const regionSizes = createStorageBuffer(device, n * 4);
  const totalRegions = createStorageBuffer(device, 4);
  const totalMembers = createStorageBuffer(device, 4);
  const owned = [
    uni, labelSurvives, counts, keptFlag, keptCount, regionIndex, memberOffset,
    cursor, members, regionId, regionOffsets, regionSizes, totalRegions, totalMembers,
  ];

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
  run(encoder, p.survive, [uni, labelBuf, magBuf, labelSurvives], up(n));
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
    readUint32(device, totalRegions, 4),
    readUint32(device, totalMembers, 4),
  ]);
  const regionCount = rTot[0], memberCount = mTot[0];

  let meanAngles: Float32Array<ArrayBufferLike> = new Float32Array(0);
  if (regionCount > 0) {
    const fUni = (() => {
      const b = new ArrayBuffer(16);
      new DataView(b).setUint32(0, regionCount, true);
      return uploadUniform(device, b);
    })();
    owned.push(fUni);
    const meanBuf = createStorageBuffer(device, regionCount * 4);
    owned.push(meanBuf);
    const enc2 = device.createCommandEncoder();
    run(enc2, p.finalize, [fUni, thetaBuf, regionOffsets, regionSizes, members, meanBuf], up(regionCount, 64));
    device.queue.submit([enc2.finish()]);
    meanAngles = await readFloat32(device, meanBuf, regionCount * 4);
  }

  const [ridRaw, memRaw, offRaw, sizeRaw] = await Promise.all([
    readUint32(device, regionId, n * 4),
    memberCount > 0 ? readUint32(device, members, memberCount * 4) : Promise.resolve(new Uint32Array(0)),
    regionCount > 0 ? readUint32(device, regionOffsets, regionCount * 4) : Promise.resolve(new Uint32Array(0)),
    regionCount > 0 ? readUint32(device, regionSizes, regionCount * 4) : Promise.resolve(new Uint32Array(0)),
  ]);

  for (const b of [...owned, ...temps]) b.destroy();
  const err = await device.popErrorScope();
  if (err) {
    console.error('collectRegionsGPU: WebGPU validation error, falling back to CPU --', err.message);
    return null;
  }

  const rid = new Int32Array(ridRaw.buffer.slice(0));
  const regions: GrownRegion[] = [];
  for (let r = 0; r < regionCount; r++) {
    const off = offRaw[r], size = sizeRaw[r];
    regions.push({
      // subarray + from, not a view over memRaw.buffer: the readback's buffer can
      // carry a byteOffset, and a raw Int32Array view would silently read from
      // the wrong place. This also copies, which is required -- memRaw is
      // transient and GrownRegion.members outlives it.
      members: Int32Array.from(memRaw.subarray(off, off + size)),
      meanAngle: meanAngles[r],
    });
  }
  return { regionId: rid, regions };
}
