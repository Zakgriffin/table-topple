import { activeCamera } from '../camera/store.ts';
import { Camera } from '../camera/model.ts';
import { computeGradient2x2Field } from '../pipeline/gradientField.ts';
import { GrownRegion } from '../pipeline/lsdSegments.ts';
import { globalState } from '../state.ts';
import { growRegionsCCLGPUToCPU } from './growRegions.ts';

// ── Dev harness: does the GPU region collector match the CPU one? ────────
//
// Run from the devtools console on a real capture:
//
//   await verifyCollectRegions()
//
// Runs growRegionsCCLGPUToCPU TWICE against the same mag/theta -- once with
// the collect on CPU, once on GPU -- so the labeling is produced by the same
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
  const { fx, fy } = field;
  const args = [
    w, h, s.lsdToleranceDeg, s.lsdRhoNoiseThreshold, s.lsdRhoHighThreshold, s.lsdCclSteps, s.lsdMinRegionSize,
  ] as const;

  {
    // Explicit per-call argument, not a globalState flip around the calls. Both
    // runs use the GPU GROWER and differ only in the collect, which is what
    // makes this a stage-3b check: the grower is not bit-identical to CPU, so
    // moving it too would swamp the comparison with grower divergence. That is
    // also why the global forceCPU could not replace this.
    const t0 = performance.now();
    const cpu = await growRegionsCCLGPUToCPU(fx, fy, ...args, false);
    const cpuMs = performance.now() - t0;

    const t1 = performance.now();
    const gpu = await growRegionsCCLGPUToCPU(fx, fy, ...args, true);
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
      // Angle between the two mean DIRECTIONS -- atan2(cross, dot), which stays
      // accurate near zero where acos(dot) would not. Comparable to the
      // pre-vector-space meanAngle delta.
      const cr = cpu.regions[r], gr = gpu.regions[r];
      const d = Math.abs(Math.atan2(
        cr.meanUx * gr.meanUy - cr.meanUy * gr.meanUx,
        cr.meanUx * gr.meanUx + cr.meanUy * gr.meanUy,
      ));
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
  }
}
