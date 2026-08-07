import {
  buildCorrectnessArray, buildDecodeSampleGrid, decodeGridCellUV, decodeGridLayout,
  rotateGrid, rotatedZeroIndex, tallyPositionVotes,
} from '../../pose/stages/decode/decodeGrid.ts';
import { getAnalysisVFovRad } from '../math/geometry.ts';
import { buildAndTallyDecodeGPU } from '../../pose/stages/decode/decodeGridBuild.gpu.ts';
import type { Backend } from '../../pose/backend.ts';
import type { HarnessInput } from './input.ts';
import { runPoseOn } from './runPose.ts';
import { wants } from '../../pose/intermediates.ts';

// ── Dev harness: does the fused GPU decode match the CPU pair? ───────────
//
// Not part of any pipeline. Run it from the devtools console on a real capture:
//
//   await verifyDecodeGridBuild(cameraInput())
//   await verifyDecodeGridBuild(await fixtureInput('default'), 'cpu')
//
// Compares buildDecodeSampleGrid + tallyPositionVotes against
// buildAndTallyDecodeGPU, cell by cell and then on the decisions that follow.
//
// The bits are the thing to watch, and they are NOT guaranteed identical. The
// grid build is f64 on CPU and f32 on GPU, so a lattice point whose reprojected
// pixel lands within a rounding step of a cell boundary can sample a DIFFERENT
// pixel on the two paths -- and near a stripe edge that pixel's bit may differ.
// One flipped bit invalidates every ORDER x ORDER window containing it, so a
// handful of bit diffs is survivable while a systematic offset is not. That is
// why the report separates `bitDiffs` (tolerable, expect a few) from
// `winnerAgrees` (must be true) and `validDiffs` (a validity disagreement is a
// geometry/threshold problem, not a rounding one).
interface DecodeGridBuildVerifyReport {
  dims: { rows: number; cols: number };
  cells: number;
  // Per-cell agreement between the two grids.
  validDiffs: number;   // cells one path calls valid and the other doesn't
  bitDiffs: number;     // cells both call valid but whose sampled bit differs
  maxPxDelta: number;   // largest |px_cpu - px_gpu| over valid cells (f32 vs f64 reprojection)
  maxPyDelta: number;
  maxUvDelta: number;   // largest |u|,|v| delta -- should be ~0, it's pure lattice arithmetic
  // The decisions that actually matter.
  winnerCpu: unknown;
  winnerGpu: unknown;
  winnerAgrees: boolean;
  // consistency, computed each path's own way: CPU from its correctness array,
  // GPU from its on-device reduction. A mismatch here with matching bits would
  // mean the correctness kernel disagrees with the CPU loop about the torus.
  consistencyCpu: number;
  consistencyGpu: number;
  correctCountCpu: number;
  correctCountGpu: number;
  // The reference cell's u/v, recomputed arithmetically by the fused path vs
  // read off the CPU grid. This is the claim that lets it skip the readback.
  refUvMatches: boolean;
  cpuMs: number;
  gpuMs: number;
}

// Runs a whole reconstruction first: the fused decode consumes recovered axes
// and a grid period/phase, so there is nothing to check without them. It used
// to read those off whatever the app had last displayed -- see runPose.ts.
//
// `camera` below is that detached run's state, not a Camera. It only ever
// needed the PoseCameraLike fields (aspect, two settings, the last* results),
// which a PoseComputeState carries by construction.
export async function verifyDecodeGridBuild(
  input: HarnessInput, backend: Backend = 'gpu',
): Promise<DecodeGridBuildVerifyReport | string> {
  const { gray, w, h } = input;
  // Asks for the decode grid explicitly: nothing brings it down unless a
  // caller says so now, and this harness compares against it.
  const camera = await runPoseOn(input, backend, wants('decodeGrid'));
  const vFovRad = getAnalysisVFovRad(camera);

  const layout = decodeGridLayout(camera, gray, vFovRad);
  if (!layout) return 'reconstruction produced no decode layout (no recovered axes or no grid period/phase) on this input';

  const cpuStart = performance.now();
  const cpuGrid = buildDecodeSampleGrid(camera, gray, w, h, vFovRad);
  if (!cpuGrid) return 'CPU builder returned null';
  const winnerCpu = tallyPositionVotes(cpuGrid);
  const cpuMs = performance.now() - cpuStart;
  if (!winnerCpu) return 'CPU tally found no winner -- nothing to compare';

  const gpuStart = performance.now();
  const fused = await buildAndTallyDecodeGPU(layout, gray, w, h);
  const gpuMs = performance.now() - gpuStart;
  if (!fused) return 'fused GPU decode returned null (WebGPU unavailable, or a validation error -- check the console)';

  try {
    const gpuGrid = await fused.readGrid();
    let validDiffs = 0, bitDiffs = 0, maxPxDelta = 0, maxPyDelta = 0, maxUvDelta = 0;
    for (let i = 0; i < cpuGrid.rows; i++) {
      for (let j = 0; j < cpuGrid.cols; j++) {
        const a = cpuGrid.points[i][j], b = gpuGrid.points[i][j];
        maxUvDelta = Math.max(maxUvDelta, Math.abs(a.u - b.u), Math.abs(a.v - b.v));
        if (a.valid !== b.valid) { validDiffs++; continue; }
        if (!a.valid) continue;
        maxPxDelta = Math.max(maxPxDelta, Math.abs(a.px - b.px));
        maxPyDelta = Math.max(maxPyDelta, Math.abs(a.py - b.py));
        if (a.bit !== b.bit) bitDiffs++;
      }
    }

    const rotCpu = rotateGrid(cpuGrid, winnerCpu.orientation);
    const cpuCorrect = buildCorrectnessArray(rotCpu, winnerCpu.anchorRow, winnerCpu.anchorCol);
    const denomCpu = cpuCorrect.correctCount + cpuCorrect.wrongCount;
    const denomGpu = fused.correctCount + fused.wrongCount;

    // The fused path's own arithmetic reference point, against the CPU grid's.
    const ref = decodeGridCellUV(layout, layout.zeroI, layout.zeroJ);
    const [rzI, rzJ] = rotatedZeroIndex(layout.rows, layout.cols, layout.zeroI, layout.zeroJ, winnerCpu.orientation);
    const cpuRef = rotCpu.points[rotCpu.zeroI][rotCpu.zeroJ];
    const refUvMatches = Math.abs(ref.u - cpuRef.u) < 1e-12 && Math.abs(ref.v - cpuRef.v) < 1e-12
      && rzI === rotCpu.zeroI && rzJ === rotCpu.zeroJ;

    return {
      dims: { rows: cpuGrid.rows, cols: cpuGrid.cols },
      cells: cpuGrid.rows * cpuGrid.cols,
      validDiffs, bitDiffs, maxPxDelta, maxPyDelta, maxUvDelta,
      winnerCpu, winnerGpu: fused.winner,
      winnerAgrees: JSON.stringify(winnerCpu) === JSON.stringify(fused.winner),
      consistencyCpu: denomCpu > 0 ? cpuCorrect.correctCount / denomCpu : 0,
      consistencyGpu: denomGpu > 0 ? fused.correctCount / denomGpu : 0,
      correctCountCpu: cpuCorrect.correctCount,
      correctCountGpu: fused.correctCount,
      refUvMatches,
      cpuMs, gpuMs,
    };
  } finally {
    fused.release();
  }
}
