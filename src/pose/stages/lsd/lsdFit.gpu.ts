import { type Alloc, type Arena, type Slice, bind, sliceRange } from '../../gpu/arena.ts';
import { readSliceF32, readSliceU32 } from '../../gpu/device.ts';
import { gpuTimelineSlot } from '../../gpu/gpuTimeline.ts';
import { LSD_FIT_WGSL } from './lsdFit.wgsl.ts';
import { type RegionSetGPU } from './collectRegions.gpu.ts';

const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();
function getPipeline(device: GPUDevice): GPUComputePipeline {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: LSD_FIT_WGSL, label: 'lsdFit' });
    p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: 'lsdFit' });
    pipelineCache.set(device, p);
  }
  return p;
}

export interface LsdFitResult {
  cx: number; cy: number; theta: number; length: number; width: number;
  nfaLog10: number; accepted: boolean;
  // The kernel's own NFA counts -- pixels inside the rectangle footprint, and
  // how many of those were aligned. Diagnostic only; see lsdFitVerify.ts.
  n: number; k: number;
}

// GPU-resident counterpart to pose/stages/lsd/rectangles.cpu.ts's fitRectangle +
// countRectanglePixels + logBinomialTail (stage 4 + stage 5's first pass, no
// retry -- see lsdFit.wgsl.ts's header for why the retry loop stays CPU-only).
// One region's members never overlap another's (the collect stage's disjoint-
// partition guarantee), so this is a plain one-thread-per-region parallel map.
//
// VERIFIED against the CPU path (harness/lsdFitVerify.ts, on a real 2931-region
// capture): zero disagreements on n, k, or accept/reject, and a max nfaLog10
// delta of 7.7e-6 -- pure f32-vs-f64 rounding. Geometry agrees to ~4e-5.
// Getting there required one real fix on BOTH sides: countRectanglePixels'
// inclusion test carries a BOUNDARY_EPS, because a rectangle's extent is defined
// by its own extreme members, so those members land exactly ON the boundary and
// were being included or dropped by whichever way the last ulp rounded.

// How many rectangles the readback copies OPTIMISTICALLY, before anyone knows
// how many there are.
//
// A copyBufferToBuffer size is fixed at ENCODE time, so to copy exactly
// regionCount rectangles the host must already know regionCount -- which is the
// fence the indirect-dispatch design removes. The alternatives are to copy the
// PROVABLE bound (n/minRegionSize = 155520 rectangles = 6.2MB at this
// resolution, against the ~0.2MB actually used) or to copy a cap.
//
// A cap, but NOT a correctness policy: `counts` comes back in the same stall, so
// the host always learns the true count, and if it exceeds this it reads the
// remainder in a second pass. Overflow is a PERFORMANCE CLIFF on a rare frame,
// never a truncated answer.
//
// 8192 against ~5200 observed at 480x648 -- 1.6x headroom, 328KB. Deliberately
// NOT generous: overshoot is paid in bytes on EVERY frame while overflow is paid
// in one extra fence on frames that never happen.
const RECT_COPY_CAP = 8192;
const RECT_STRIDE_BYTES = 10 * 4;

export function sizeOfLsdFit(maxRegions: number): number {
  return maxRegions * RECT_STRIDE_BYTES + 256;
}

// ENCODE ONLY. The host never learns the region count here, by design: the
// dispatch is INDIRECT off the same triple collect's finalize used (both are
// workgroup_size(64) over regions, so one device-written count serves both). An
// empty capture falls out correctly -- regionMeta writes 0 workgroups, the
// dispatch does nothing, and the readback below reports 0.
export function encodeLsdFit(
  arena: Arena, alloc: Alloc, enc: GPUCommandEncoder,
  inp: {
    fx: Slice; fy: Slice; regions: RegionSetGPU; w: number; h: number;
    rho: number; toleranceDeg: number; logNTests: number; logEpsilon: number;
  },
): Slice {
  const { fx, fy, regions, w, h, rho, toleranceDeg, logNTests, logEpsilon } = inp;
  const device = arena.device;
  const pipeline = getPipeline(device);

  // The PROVABLE bound, not a cap: this one is only allocated, never copied, so
  // its size costs address space rather than bandwidth.
  const out = alloc(sizeOfLsdFit(regions.maxRegions), 'lsdFit.out');

  const uni = alloc(32, 'lsdFit.uniforms');
  {
    const buf = new ArrayBuffer(32);
    const dv = new DataView(buf);
    dv.setUint32(0, w, true); dv.setUint32(4, h, true);
    // Squared, matching levelLine.ts's eligibilityThresholdSq (negative case
    // included), so the shader's per-pixel test needs no sqrt.
    dv.setFloat32(16, rho >= 0 ? rho * rho : -Infinity, true);
    dv.setFloat32(20, (toleranceDeg * Math.PI) / 180, true);
    dv.setFloat32(24, logNTests, true); dv.setFloat32(28, logEpsilon, true);
    device.queue.writeBuffer(uni.buffer, uni.offset, buf);
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [uni, fx, fy, regions.offsets, regions.members, regions.meanDirs, out, regions.sizes, regions.counts]
      .map((s, binding) => ({ binding, resource: bind(s, arena) })),
  });

  const args = sliceRange(regions.dispatchArgs, arena);
  const pass = enc.beginComputePass(gpuTimelineSlot('lsdFit:fitRegion'));
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroupsIndirect(args.buffer, args.offset);
  pass.end();

  return out;
}

// The readback half, called by the chain AFTER it submits. Kept separate from
// the encode so the stage itself stays composable: everything above can go into
// any encoder, and only this pays a fence.
//
// ONE stall for both: the counts and the optimistic rectangle slice are read off
// the same submit, concurrently, so this costs one fence rather than the
// count-then-rectangles pair the design started with.
export async function readLsdFitResults(
  arena: Arena, out: Slice, regions: RegionSetGPU, owner?: string,
): Promise<{ results: LsdFitResult[]; regionCount: number; memberCount: number }> {
  const capped = Math.min(RECT_COPY_CAP, regions.maxRegions);
  const [counts, head] = await Promise.all([
    readSliceU32(arena, regions.counts, 8, 'regions:counts', owner),
    readSliceF32(arena, out, capped * RECT_STRIDE_BYTES, 'lsdFit:rects', owner),
  ]);
  const regionCount = counts[0], memberCount = counts[1];

  // The cliff, and it is a cliff rather than a bug: everything past the cap is
  // read in a second pass, and it reports itself so RECT_COPY_CAP can be raised
  // rather than silently eaten.
  let raw = head;
  if (regionCount > capped) {
    console.warn(`readLsdFitResults: ${regionCount} regions exceeded RECT_COPY_CAP ${RECT_COPY_CAP} -- second readback taken, consider raising it`);
    raw = await readSliceF32(arena, out, regionCount * RECT_STRIDE_BYTES, 'lsdFit:rectsOverflow', owner);
  }

  const results: LsdFitResult[] = new Array(regionCount);
  for (let i = 0; i < regionCount; i++) {
    const o = i * 10;
    results[i] = {
      cx: raw[o], cy: raw[o + 1], theta: raw[o + 2], length: raw[o + 3], width: raw[o + 4],
      nfaLog10: raw[o + 5], accepted: raw[o + 6] !== 0, n: raw[o + 8], k: raw[o + 9],
    };
  }
  return { results, regionCount, memberCount };
}

// The region CSR brought down to host arrays. OFF the pose path -- only the
// display callers and the harnesses want members (see chain.ts's `wantMembers`).
export async function readRegionMembers(
  arena: Arena, regions: RegionSetGPU, regionCount: number, memberCount: number, owner?: string,
): Promise<{ members: Int32Array; meanUx: number; meanUy: number }[]> {
  if (regionCount === 0) return [];
  const [offRaw, sizeRaw, memRaw, meanRaw] = await Promise.all([
    readSliceU32(arena, regions.offsets, regionCount * 4, 'regions:offsets', owner),
    readSliceU32(arena, regions.sizes, regionCount * 4, 'regions:sizes', owner),
    memberCount > 0
      ? readSliceU32(arena, regions.members, memberCount * 4, 'regions:members', owner)
      : Promise.resolve(new Uint32Array(0)),
    readSliceF32(arena, regions.meanDirs, regionCount * 8, 'regions:meanDirs', owner),
  ]);
  const out: { members: Int32Array; meanUx: number; meanUy: number }[] = [];
  for (let r = 0; r < regionCount; r++) {
    out.push({
      // subarray + from, not a view onto memRaw.buffer: a readback's buffer can
      // carry a byteOffset, and a raw Int32Array view would silently read from
      // the wrong place. The copy is required anyway -- memRaw is transient.
      members: Int32Array.from(memRaw.subarray(offRaw[r], offRaw[r] + sizeRaw[r])),
      meanUx: meanRaw[r * 2], meanUy: meanRaw[r * 2 + 1],
    });
  }
  return out;
}
