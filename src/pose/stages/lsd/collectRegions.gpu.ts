import {
  COLLECT_FINALIZE_WGSL, COLLECT_HISTOGRAM_WGSL, COLLECT_MARK_KEPT_WGSL,
  COLLECT_REGION_META_WGSL, COLLECT_SCATTER_WGSL, COLLECT_SURVIVE_WGSL,
} from './collectRegions.wgsl.ts';
import { type Alloc, type Arena, type Slice, allocZeroed, bind, sliceRange } from '../../gpu/arena.ts';
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

// The provable ceiling on regions for an n-pixel field: a kept region holds at
// least minRegionSize pixels, so there can be at most n/minRegionSize of them.
// Exact rather than a policy guess, which is why buffers sized with it need no
// overflow handling. Guarded at 1 because minRegionSize is a user-facing setting
// and 0 would be a division by zero rather than an infinite bound.
//
// Lives here now rather than in the deleted fieldResidency.ts: it is this
// stage's own bound, and every buffer sized by it is one this stage writes.
export function maxRegionCount(n: number, minRegionSize: number): number {
  return Math.ceil(n / Math.max(1, minRegionSize));
}

// The region set in its GPU form: a CSR over region members. `offsets`/`sizes`
// are per-region and `members` is the concatenation; because regions are laid
// out contiguously, offsets[i] + sizes[i] == offsets[i+1].
export interface RegionSetGPU {
  offsets: Slice;
  sizes: Slice;
  members: Slice;
  meanDirs: Slice; // vec2<f32> per region -- a direction, never an angle
  // [regionCount, memberCount], on DEVICE. Bound by every consumer that needs a
  // bounds check, so none of them needs the host to have read it.
  counts: Slice;
  // (ceil(regionCount/64), 1, 1) -- the workgroup count for a per-region
  // dispatch at workgroup_size(64), which finalize and lsdFit's fitRegion both
  // are. Written on device by the regionMeta pass.
  dispatchArgs: Slice;
  // The provable ceiling, carried because it is what any per-region buffer must
  // be sized at once nobody reads the actual count back.
  maxRegions: number;
}

// GPU-resident counterpart to pose/stages/lsd/regions.cpu.ts's
// collectRegionsFromLabels.
//
// Same region numbering and same member ordering as the CPU version -- see
// collectRegions.wgsl.ts for why both are exact rather than merely equivalent.
//
// ── The model implementation ──
//
// Six passes plus two scans, all encoded, NOTHING read back -- not even the 8
// bytes of region and member count. It used to be two submissions, and the split
// was forced: the region count only exists after the first scan, and it was what
// sized the `finalize` dispatch, so the host read it back, which is a submit
// boundary plus a browser-scheduled fence to learn one integer. `finalize`
// dispatches INDIRECTLY off a workgroup count regionMeta writes on device, and
// the counts are published as a BUFFER rather than as numbers.
//
// This was already encode-only before the restructure; all that changed is that
// it takes slices and an encoder instead of a FieldResidency and its own.
export function sizeOfCollect(n: number, minRegionSize: number): number {
  // 7 scratch arrays of n + 4 handoff arrays of n + meanDirs at the bound, plus
  // the small ones. Approximate on purpose: it is a hint for the arena's initial
  // capacity, and the high-water mark is what actually sizes it.
  return n * 4 * 11 + maxRegionCount(n, minRegionSize) * 8 + 4096;
}

export function encodeCollectRegions(
  arena: Arena, alloc: Alloc, enc: GPUCommandEncoder,
  inp: { label: Slice; fx: Slice; fy: Slice; n: number; rhoHigh: number; minRegionSize: number },
): { regionId: Slice; regions: RegionSetGPU } {
  const { label, fx, fy, n, rhoHigh, minRegionSize } = inp;
  const device = arena.device;
  const p = getPipelines(device);

  const uni = alloc(16, 'collect.uniforms');
  {
    const b = new ArrayBuffer(16);
    const dv = new DataView(b);
    // Squared here, once, rather than in the shader per pixel. Matches
    // levelLine.ts's eligibilityThresholdSq, including the negative case.
    dv.setUint32(0, n, true);
    dv.setFloat32(4, rhoHigh >= 0 ? rhoHigh * rhoHigh : -Infinity, true);
    dv.setUint32(8, minRegionSize, true);
    device.queue.writeBuffer(uni.buffer, uni.offset, b);
  }

  // Every per-LABEL array is size n, because labels are pixel indices. That is
  // ~6 arrays x 4 bytes x n; at 196608 pixels it is under 5MB total, which is
  // the price of never hashing or compacting the label space.
  const labelSurvives = alloc(n * 4, 'collect.labelSurvives');
  // ZEROED: histogram does atomicAdd into this. See allocZeroed's header --
  // createBuffer used to guarantee zeroes, an arena slice is last frame's bytes,
  // and a missed clear here reads as a region count that grows every frame.
  const labelCounts = allocZeroed(arena, alloc, enc, n * 4, 'collect.labelCounts');
  const keptFlag = alloc(n * 4, 'collect.keptFlag');
  const keptCount = alloc(n * 4, 'collect.keptCount');
  const regionIndex = alloc(n * 4, 'collect.regionIndex');
  const memberOffset = alloc(n * 4, 'collect.memberOffset');
  // ZEROED: scatter's atomicAdd uses this as a per-label write cursor.
  const cursor = allocZeroed(arena, alloc, enc, n * 4, 'collect.cursor');
  const totalRegions = alloc(4, 'collect.totalRegions');
  const totalMembers = alloc(4, 'collect.totalMembers');
  // The two counts, side by side, where a shader can read them -- finalize's
  // bounds check and lsdFit's both index this instead of a host-filled uniform.
  const counts = alloc(8, 'collect.counts');
  // (ceil(regionCount/64), 1, 1), written by regionMeta and dispatched off.
  const dispatchArgs = alloc(12, 'collect.dispatchArgs');

  const members = alloc(n * 4, 'collect.members');
  const regionId = alloc(n * 4, 'collect.regionId');
  const regionOffsets = alloc(n * 4, 'collect.offsets');
  const regionSizes = alloc(n * 4, 'collect.sizes');
  // Sized at the PROVABLE bound rather than at the region count, because the
  // whole point of this design is that the region count is never read back. At
  // the default minRegionSize of 2 this is n/2 * 8B, i.e. the same ~1.2MB as any
  // other n-sized field.
  const maxRegions = maxRegionCount(n, minRegionSize);
  const meanDirs = alloc(maxRegions * 8, 'collect.meanDirs');

  const run = (stage: Stage, label2: string, resources: Slice[], groups: number) => {
    const bindGroup = device.createBindGroup({
      layout: stage.layout,
      entries: resources.map((s, binding) => ({ binding, resource: bind(s, arena) })),
    });
    const cp = enc.beginComputePass(gpuTimelineSlot(label2));
    cp.setPipeline(stage.pipeline);
    cp.setBindGroup(0, bindGroup);
    cp.dispatchWorkgroups(groups);
    cp.end();
  };

  run(p.survive, 'collect:survive', [uni, label, fx, fy, labelSurvives], up(n));
  run(p.histogram, 'collect:histogram', [uni, label, labelSurvives, labelCounts], up(n));
  run(p.markKept, 'collect:markKept', [uni, labelSurvives, labelCounts, keptFlag, keptCount], up(n));
  encodeExclusiveScan(arena, alloc, enc, keptFlag, regionIndex, totalRegions, n);
  encodeExclusiveScan(arena, alloc, enc, keptCount, memberOffset, totalMembers, n);
  run(p.regionMeta, 'collect:regionMeta', [uni, keptFlag, regionIndex, memberOffset, labelCounts, regionOffsets, regionSizes, totalRegions, dispatchArgs], up(n));
  run(p.scatter, 'collect:scatter', [uni, label, keptFlag, regionIndex, memberOffset, cursor, members, regionId], up(n));

  // The two scans wrote their totals into separate 4-byte slices; put them side
  // by side so one binding serves finalize's bounds check here and lsdFit's
  // downstream. copyBufferToBuffer, not a pass -- there is nothing to compute.
  {
    const tr = sliceRange(totalRegions, arena), tm = sliceRange(totalMembers, arena);
    const c = sliceRange(counts, arena);
    enc.copyBufferToBuffer(tr.buffer, tr.offset, c.buffer, c.offset, 4);
    enc.copyBufferToBuffer(tm.buffer, tm.offset, c.buffer, c.offset + 4, 4);
  }

  // INDIRECT: the workgroup count comes off dispatchArgs, which regionMeta wrote
  // two passes ago and the host never sees. regionMeta writes 0 workgroups for
  // an empty capture, and a zero-workgroup indirect dispatch is legal and does
  // nothing -- which is why there is no `if (regionCount > 0)` guard.
  {
    const bindGroup = device.createBindGroup({
      layout: p.finalize.layout,
      entries: [counts, fx, regionOffsets, regionSizes, members, meanDirs, fy]
        .map((s, binding) => ({ binding, resource: bind(s, arena) })),
    });
    const args = sliceRange(dispatchArgs, arena);
    const cp = enc.beginComputePass(gpuTimelineSlot('collect:finalize'));
    cp.setPipeline(p.finalize.pipeline);
    cp.setBindGroup(0, bindGroup);
    cp.dispatchWorkgroupsIndirect(args.buffer, args.offset);
    cp.end();
  }

  return {
    regionId,
    regions: { offsets: regionOffsets, sizes: regionSizes, members, meanDirs, counts, dispatchArgs, maxRegions },
  };
}
