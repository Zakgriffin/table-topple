import {
  COLLECT_FINALIZE_WGSL, COLLECT_HISTOGRAM_WGSL, COLLECT_MARK_KEPT_WGSL,
  COLLECT_REGION_META_WGSL, COLLECT_SCATTER_WGSL, COLLECT_SURVIVE_WGSL,
} from './collectRegions.wgsl.ts';
import { createStorageBuffer, uploadUniform } from '../../gpu/device.ts';
import { FieldResidency, maxRegionCount } from '../../gpu/fieldResidency.ts';
import { gpuTimelineSlot } from '../../gpu/gpuTimeline.ts';
import { encodeExclusiveScan } from '../../gpu/prefixSum.ts';

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
      regionMeta: makeStage(device, COLLECT_REGION_META_WGSL, 'regionMeta', [UNI, RO, RO, RO, RO, RW, RW, RO, RW]),
      scatter: makeStage(device, COLLECT_SCATTER_WGSL, 'scatter', [UNI, RO, RO, RO, RO, RW, RW, RW]),
      finalize: makeStage(device, COLLECT_FINALIZE_WGSL, 'finalize', [RO, RO, RO, RO, RW, RW, RO]),
    };
    cache.set(device, p);
  }
  return p;
}

// GPU-resident counterpart to pose/stages/lsd/regions.cpu.ts's collectRegionsFromLabels.
//
// Reads label/mag/theta out of the residency (already on device when stage 3
// ran there) and PUBLISHES regionId plus the region CSR back into it, still on
// device. Same region numbering and same member ordering as the CPU version --
// see collectRegions.wgsl.ts for why both are exact rather than merely
// equivalent.
//
// ONE submission, and NOTHING crosses the bus -- not even the 8 bytes of region
// and member count it used to wait for.
//
// It used to be two submissions, and the split was forced: the region count only
// exists after the first scan, and it was what sized the `finalize` dispatch. So
// the host read it back, which is a submit boundary plus a browser-scheduled
// fence to learn one integer. `finalize` now dispatches INDIRECTLY off a
// workgroup count regionMeta writes on device (see collectRegions.wgsl.ts), and
// the counts are published as a BUFFER rather than as numbers -- see
// RegionSetGPU.knownCounts for who is allowed to want them as numbers, and what
// it costs them.
//
// Everything else it produces is left where the fitter can consume it:
// the ~800KB regionId array is pure debug-overlay data that the production path
// never looks at, and the CSR is stage 4's direct input. Both used to come down
// unconditionally, which is most of why the GPU collect once measured as a loss.
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
    // levelLine.ts's eligibilityThresholdSq, including the negative case.
    dv.setUint32(0, n, true);
    dv.setFloat32(4, rhoHigh >= 0 ? rhoHigh * rhoHigh : -Infinity, true);
    dv.setUint32(8, minRegionSize, true);
    return uploadUniform(device, b);
  })();

  // Every per-LABEL array is size n, because labels are pixel indices. That is
  // ~6 arrays x 4 bytes x n; at 196608 pixels it is under 5MB total, which is
  // the price of never hashing or compacting the label space.
  const labelSurvives = createStorageBuffer(device, n * 4);
  const labelCounts = createStorageBuffer(device, n * 4);
  const keptFlag = createStorageBuffer(device, n * 4);
  const keptCount = createStorageBuffer(device, n * 4);
  const regionIndex = createStorageBuffer(device, n * 4);
  const memberOffset = createStorageBuffer(device, n * 4);
  const cursor = createStorageBuffer(device, n * 4);
  const totalRegions = createStorageBuffer(device, 4);
  const totalMembers = createStorageBuffer(device, 4);
  // The two counts, side by side, where a shader can read them -- finalize's
  // bounds check and lsdFit's both index this instead of a host-filled uniform.
  const counts = createStorageBuffer(device, 8);
  // (ceil(regionCount/64), 1, 1). INDIRECT is what makes it dispatchable; the
  // rest is so regionMeta can write it and the harness can read it.
  const dispatchArgs = createStorageBuffer(device, 12, GPUBufferUsage.INDIRECT);

  // Split by DESTINATION, not by lifetime: `scratch` dies with this call,
  // `handoff` is what gets published into the residency on success (and has to
  // be destroyed by hand on every failure path, since nothing else has taken
  // ownership of it yet).
  const members = createStorageBuffer(device, n * 4);
  const regionId = createStorageBuffer(device, n * 4);
  const regionOffsets = createStorageBuffer(device, n * 4);
  const regionSizes = createStorageBuffer(device, n * 4);
  // Sized at the PROVABLE bound rather than at the region count, because the
  // whole point of this rewrite is that the region count is never read back. A
  // kept region holds at least minRegionSize pixels, so there are at most
  // n/minRegionSize of them -- exact, not a guess, and no overflow case to
  // handle. At the default minRegionSize of 2 this is n/2 * 8B, i.e. the same
  // 1.2MB as any other n-sized field.
  const maxRegions = maxRegionCount(n, minRegionSize);
  const meanDirs = createStorageBuffer(device, maxRegions * 8);
  const scratch = [
    uni, labelSurvives, labelCounts, keptFlag, keptCount, regionIndex, memberOffset,
    cursor, totalRegions, totalMembers,
  ];
  const handoff = [members, regionId, regionOffsets, regionSizes, meanDirs, counts, dispatchArgs];

  const run = (encoder: GPUCommandEncoder, stage: Stage, label: string, resources: GPUBuffer[], groups: number) => {
    const bindGroup = device.createBindGroup({
      layout: stage.layout,
      entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const cp = encoder.beginComputePass(gpuTimelineSlot(label));
    cp.setPipeline(stage.pipeline);
    cp.setBindGroup(0, bindGroup);
    cp.dispatchWorkgroups(groups);
    cp.end();
  };

  const runIndirect = (encoder: GPUCommandEncoder, stage: Stage, label: string, resources: GPUBuffer[], args: GPUBuffer) => {
    const bindGroup = device.createBindGroup({
      layout: stage.layout,
      entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const cp = encoder.beginComputePass(gpuTimelineSlot(label));
    cp.setPipeline(stage.pipeline);
    cp.setBindGroup(0, bindGroup);
    cp.dispatchWorkgroupsIndirect(args, 0);
    cp.end();
  };

  const encoder = device.createCommandEncoder();
  run(encoder, p.survive, 'collect:survive', [uni, labelBuf, fxBuf, fyBuf, labelSurvives], up(n));
  run(encoder, p.histogram, 'collect:histogram', [uni, labelBuf, labelSurvives, labelCounts], up(n));
  run(encoder, p.markKept, 'collect:markKept', [uni, labelSurvives, labelCounts, keptFlag, keptCount], up(n));
  const temps = [
    ...encodeExclusiveScan(device, encoder, keptFlag, regionIndex, totalRegions, n),
    ...encodeExclusiveScan(device, encoder, keptCount, memberOffset, totalMembers, n),
  ];
  run(encoder, p.regionMeta, 'collect:regionMeta', [uni, keptFlag, regionIndex, memberOffset, labelCounts, regionOffsets, regionSizes, totalRegions, dispatchArgs], up(n));
  run(encoder, p.scatter, 'collect:scatter', [uni, labelBuf, keptFlag, regionIndex, memberOffset, cursor, members, regionId], up(n));
  // The two scans wrote their totals into separate 4-byte buffers; put them
  // side by side so one binding serves finalize's bounds check here and
  // lsdFit's downstream. copyBufferToBuffer, not a pass -- there is nothing to
  // compute, and a pass would cost what plan item 12 measured passes to cost.
  encoder.copyBufferToBuffer(totalRegions, 0, counts, 0, 4);
  encoder.copyBufferToBuffer(totalMembers, 0, counts, 4, 4);
  // INDIRECT, and this is the line the whole item is about. It used to be
  // `up(regionCount, 64)` -- a JS number, which meant the host had to read the
  // region count back, which meant a submit boundary and a fence between the
  // scatter above and this. Now the workgroup count comes off dispatchArgs,
  // which regionMeta wrote two passes ago and the host never sees.
  //
  // The `if (regionCount > 0)` guard this replaced is gone too: regionMeta
  // writes 0 workgroups for an empty capture, and a zero-workgroup indirect
  // dispatch is legal and does nothing.
  runIndirect(encoder, p.finalize, 'collect:finalize', [counts, fxBuf, regionOffsets, regionSizes, members, meanDirs, fyBuf], dispatchArgs);
  device.queue.submit([encoder.finish()]);

  for (const b of [...scratch, ...temps]) b.destroy();
  // Awaited, but NOT a fence on any data: popErrorScope resolves off the
  // validation queue, not off pipeline completion, so this does not wait for
  // the submit above to execute. Nothing else in this function awaits anything
  // now -- the whole collect is fire-and-forget.
  const err = await device.popErrorScope();
  if (err) {
    console.error('collectRegionsGPU: WebGPU validation error, falling back to CPU --', err.message);
    for (const b of handoff) b.destroy();
    return false;
  }

  res.provideGPU('regionId', regionId);
  res.provideRegionsGPU({
    offsets: regionOffsets, sizes: regionSizes, members, meanDirs, counts, dispatchArgs, maxRegions,
    // Null, and that is the whole point: nobody fenced to learn them. The CPU
    // collect fills these in because it genuinely knows them for free.
    knownCounts: null,
  });
  return true;
}
