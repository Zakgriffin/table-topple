import { activeCamera } from '../camera/store.ts';
import { Camera } from '../camera/model.ts';
import { computeGradient2x2Field } from '../pipeline/gradientField.ts';
import { computeMagTheta, GrownRegion } from '../pipeline/lsdSegments.ts';
import { globalState } from '../state.ts';
import { growRegionsCCLGPU } from './growRegions.ts';

// ── Dev harness: does the GPU region collector match the CPU one? ────────
//
// Run from the devtools console on a real capture:
//
//   await verifyCollectRegions()
//
// Runs growRegionsCCLGPU TWICE against the same mag/theta -- once with
// useGPUCollectRegions off, once on -- so the labeling is produced by the same
// GPU round loop both times and the ONLY difference is which collector turned
// it into regions. That isolates this stage from the f32/f64 edge-predicate
// question growRegionsVerify already covers.
//
// Unlike the grower, this comparison SHOULD be exact, and the report is written
// to assert that rather than to characterise a distribution:
//   - region count identical (both filter by rhoHigh then minRegionSize)
//   - region numbering identical (ascending label order, which the GPU gets from
//     the scan visiting labels in index order and the CPU from sorting Map keys)
//   - member sets AND member order identical (the GPU sorts each CSR row, since
//     atomicAdd hands out slots in arrival order)
//   - regionId identical for every pixel
// The one thing that legitimately differs is meanAngle: the GPU sums cos/sin in
// f32. Reported as a delta rather than an equality.
export interface CollectRegionsVerifyReport {
  cpuRegions: number;
  gpuRegions: number;
  countMatches: boolean;
  memberSetMismatches: number;   // regions whose member SETS differ
  memberOrderMismatches: number; // regions with the same set but a different order
  regionIdMismatches: number;    // pixels assigned to different region ids
  maxMeanAngleDelta: number;     // radians; f32 vs f64 summation only
  sizeHistogramMatches: boolean;
  cpuMs: number;
  gpuMs: number;
}

function sizeSig(regions: GrownRegion[]): string {
  return regions.map((r) => r.members.length).join(',');
}

export async function verifyCollectRegions(camera?: Camera | null): Promise<CollectRegionsVerifyReport | string> {
  camera = camera ?? activeCamera() ?? null;
  if (!camera) return 'no active camera';
  const gray = camera.lastNoisedPreviewGray;
  if (!gray) return 'no capture yet -- run a capture first';
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const s = camera.settings;

  const field = computeGradient2x2Field(gray, w, h);
  const { mag, theta } = computeMagTheta(field);
  const args = [
    w, h, s.lsdToleranceDeg, s.lsdRhoNoiseThreshold, s.lsdRhoHighThreshold, s.lsdCclSteps, s.lsdMinRegionSize,
  ] as const;

  const saved = globalState.useGPUCollectRegions;
  try {
    globalState.useGPUCollectRegions = false;
    const t0 = performance.now();
    const cpu = await growRegionsCCLGPU(mag, theta, ...args);
    const cpuMs = performance.now() - t0;

    globalState.useGPUCollectRegions = true;
    const t1 = performance.now();
    const gpu = await growRegionsCCLGPU(mag, theta, ...args);
    const gpuMs = performance.now() - t1;

    if (!cpu || !gpu) return 'grower returned null (WebGPU unavailable, or a validation error -- check the console)';

    let memberSetMismatches = 0, memberOrderMismatches = 0, maxMeanAngleDelta = 0;
    const shared = Math.min(cpu.regions.length, gpu.regions.length);
    for (let r = 0; r < shared; r++) {
      const a = cpu.regions[r].members, b = gpu.regions[r].members;
      if (a.length !== b.length) { memberSetMismatches++; continue; }
      let sameOrder = true;
      for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) { sameOrder = false; break; }
      if (!sameOrder) {
        // Same length but different sequence: distinguish "different pixels" from
        // "same pixels, different order", since only the first is a real bug.
        const sa = Array.from(a).sort((x, y) => x - y).join(',');
        const sb = Array.from(b).sort((x, y) => x - y).join(',');
        if (sa === sb) memberOrderMismatches++; else memberSetMismatches++;
      }
      let d = Math.abs(cpu.regions[r].meanAngle - gpu.regions[r].meanAngle) % (2 * Math.PI);
      if (d > Math.PI) d = 2 * Math.PI - d;
      maxMeanAngleDelta = Math.max(maxMeanAngleDelta, d);
    }

    let regionIdMismatches = 0;
    for (let i = 0; i < cpu.regionId.length; i++) if (cpu.regionId[i] !== gpu.regionId[i]) regionIdMismatches++;

    return {
      cpuRegions: cpu.regions.length,
      gpuRegions: gpu.regions.length,
      countMatches: cpu.regions.length === gpu.regions.length,
      memberSetMismatches, memberOrderMismatches, regionIdMismatches, maxMeanAngleDelta,
      sizeHistogramMatches: sizeSig(cpu.regions) === sizeSig(gpu.regions),
      cpuMs, gpuMs,
    };
  } finally {
    globalState.useGPUCollectRegions = saved;
  }
}
