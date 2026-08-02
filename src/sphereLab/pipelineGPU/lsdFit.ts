import { createStorageBuffer, getGPUDevice, readFloat32, uploadFloat32, uploadUint32, uploadUniform } from './device.ts';
import { LSD_FIT_WGSL } from './lsdFit.wgsl.ts';

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

const WORKGROUP_SIZE_1D = 64;

export interface LsdFitResult {
  cx: number; cy: number; theta: number; length: number; width: number;
  nfaLog10: number; accepted: boolean;
  // The kernel's own NFA counts -- pixels inside the rectangle footprint, and
  // how many of those were aligned. Diagnostic only (nothing in the production
  // path reads them); see lsdFitVerify.ts for why they're worth carrying.
  n: number; k: number;
}

// GPU-resident counterpart to pipeline/lsdSegments.ts's fitRectangle +
// countRectanglePixels + logBinomialTail (stage 4 + stage 5's first pass,
// no retry -- see lsdFit.wgsl.ts's own header for why the retry loop stays
// CPU-only). One region's members never overlap another's (stage 3's own
// disjoint-partition guarantee), so this is a plain one-thread-per-region
// parallel map, no cross-region synchronization needed at all.
//
// VERIFIED against the CPU path (pipelineGPU/lsdFitVerify.ts, on a real
// 2931-region capture): zero disagreements on n, k, or accept/reject, and a
// max nfaLog10 delta of 7.7e-6 -- pure f32-vs-f64 rounding. Geometry agrees to
// ~4e-5. The mod-π/directed parity problem that originally pinned this is gone
// (both sides are directed again, see levelLineAngle in lsdSegments.ts).
//
// Getting there required one real fix on BOTH sides: countRectanglePixels'
// inclusion test now carries a BOUNDARY_EPS, because a rectangle's extent is
// defined by its own extreme members, so those members land exactly ON the
// boundary and were being included or dropped by whichever way the last ulp
// rounded. That disagreed across arithmetic on 180/2931 regions for n and
// 277 for k, moving nfaLog10 by up to 4.7 decades.
//
// STILL DEFAULT-OFF, but for a PERFORMANCE reason now, not a correctness one:
// warm-cache timing on that same capture was CPU 3.5ms vs GPU 8ms. This
// function takes mag/theta as CPU Float64Arrays and builds the CSR member
// arrays on CPU, so switching it on today ADDS a round trip (gradient GPU ->
// readback -> grow CPU -> upload -> dispatch -> readback) rather than removing
// one. It inverts once stage 2+3 (growRegionsCCL) moves to GPU and the
// labeling can be handed over device-side without ever landing on CPU -- that
// is the point at which to flip the default.
//
// Returns null if WebGPU isn't available; caller falls back to the CPU
// version, which stays the source of truth. Returns [] (not null) for
// regions.length === 0 -- a valid empty-input result, not a GPU failure.
export async function fitAndTestRegionsGPU(
  mag: Float64Array, theta: Float64Array, w: number, h: number,
  regions: readonly { members: Int32Array; meanAngle: number }[],
  rho: number, toleranceDeg: number, logNTests: number, logEpsilon: number,
): Promise<LsdFitResult[] | null> {
  if (regions.length === 0) return [];
  const device = await getGPUDevice();
  if (!device) return null;
  const pipeline = getPipeline(device);

  const regionCount = regions.length;
  const memberOffsets = new Uint32Array(regionCount + 1);
  let total = 0;
  for (let i = 0; i < regionCount; i++) { memberOffsets[i] = total; total += regions[i].members.length; }
  memberOffsets[regionCount] = total;
  const memberIndices = new Uint32Array(total);
  const meanAngles = new Float32Array(regionCount);
  for (let i = 0; i < regionCount; i++) {
    memberIndices.set(regions[i].members, memberOffsets[i]);
    meanAngles[i] = regions[i].meanAngle;
  }

  const magBuf = uploadFloat32(device, new Float32Array(mag));
  const thetaBuf = uploadFloat32(device, new Float32Array(theta));
  const offsetsBuf = uploadUint32(device, memberOffsets);
  const indicesBuf = uploadUint32(device, memberIndices);
  const meanAnglesBuf = uploadFloat32(device, meanAngles);
  const outBuf = createStorageBuffer(device, regionCount * 10 * 4);

  const toleranceRad = (toleranceDeg * Math.PI) / 180;
  const uniformData = new ArrayBuffer(32);
  const dv = new DataView(uniformData);
  dv.setUint32(0, w, true); dv.setUint32(4, h, true); dv.setUint32(8, regionCount, true); dv.setUint32(12, 0, true);
  dv.setFloat32(16, rho, true); dv.setFloat32(20, toleranceRad, true);
  dv.setFloat32(24, logNTests, true); dv.setFloat32(28, logEpsilon, true);
  const uniformBuf = uploadUniform(device, uniformData);

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: magBuf } },
      { binding: 2, resource: { buffer: thetaBuf } },
      { binding: 3, resource: { buffer: offsetsBuf } },
      { binding: 4, resource: { buffer: indicesBuf } },
      { binding: 5, resource: { buffer: meanAnglesBuf } },
      { binding: 6, resource: { buffer: outBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(regionCount / WORKGROUP_SIZE_1D));
  pass.end();
  device.queue.submit([encoder.finish()]);

  const raw = await readFloat32(device, outBuf, regionCount * 10 * 4);
  for (const b of [magBuf, thetaBuf, offsetsBuf, indicesBuf, meanAnglesBuf, outBuf, uniformBuf]) b.destroy();

  const results: LsdFitResult[] = new Array(regionCount);
  for (let i = 0; i < regionCount; i++) {
    const o = i * 10;
    results[i] = {
      cx: raw[o], cy: raw[o + 1], theta: raw[o + 2], length: raw[o + 3], width: raw[o + 4],
      nfaLog10: raw[o + 5], accepted: raw[o + 6] !== 0, n: raw[o + 8], k: raw[o + 9],
    };
  }
  return results;
}
