import * as THREE from 'three';
import { GRID_STEP, MATH_QUAT } from '../constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../math/geometry.ts';
import { FieldResidency } from '../pipelineGPU/fieldResidency.ts';
import { fitPairOfPlanesGPU } from '../pipelineGPU/fitPlanes.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { globalState } from '../state.ts';
import { DecodeCellDebug, DecodeSampleGrid, GradientField, PositionDecodeResult, RecoveredAxes, Vote } from '../types.ts';
import { CompositeLine } from './bucketFillJoin.ts';
import { runPositionDecode } from './decodeGrid.ts';
import { computeGridPeriodPhase, GridPeriodPhaseResult } from './gridPeriodPhase.ts';
import { createLsdChainResidency } from './lsdSegments.ts';
import { fourFoldResidual } from './orientationLM.ts';
import {
  computeGradient2x2Composites, computePixelVotes2x2, computeSegmentVotes, fitPairOfPlanes, refineOrientationIRLS,
  LsdCompositeSettings,
} from './votes.ts';

// ── Shared pure pose-recovery orchestrator ────────────────────────────────
//
// Extracts the pure prefix of axesReconstruction.ts's recomputeStages (every
// stage through runPositionDecode) into one function that operates on a
// bare data object instead of a real Camera -- no THREE-scene objects, GPU
// textures, or DOM required, so this is exactly what a phone-side caller
// (mobileCapture.ts, see this session's on-device-pose-recovery plan) can
// run standalone. computeProjectedBinsAndMarginalsAuto/paintProjectedTexture
// are deliberately NOT included -- confirmed not on the critical path to a
// pose (distance is already finalized by gridPeriodPhase before that stage
// would run); they exist only to feed Projected-Cam/World-floor-decal
// DISPLAY, so axesReconstruction.ts keeps calling them separately, only for
// desktop display purposes. Likewise the pole-marker/gizmo/readout tail
// (axesReconstruction.ts's own overlaySpan) stays there, driven off
// lastQuadricPair/lastRecoveredAxes/lastPositionDecode after this returns.
export interface PoseComputeState {
  aspect: number;
  settings: LsdCompositeSettings & {
    horizFovDeg: number; weightSharpenPower: number; gridPeriodPhaseGapLowerBound: number; minGrazingCos: number;
    useWorldVoteOrientation: boolean; worldVoteRefineSteps: number;
  };
  lastVoteComposites: { root: number; line: CompositeLine }[] | null;
  lastVotes: Vote[] | null;
  // Raw fit result (Drow/Dcol/Dnormal only, no distance) BEFORE period-search
  // gating -- axesReconstruction.ts's pole markers render off
  // rowDirRecovered/colDirRecovered even when gridPeriodPhase fails and
  // lastRecoveredAxes ends up null. Needed as its own field so that behavior
  // survives the refactor instead of silently regressing -- see
  // camera/model.ts's CameraBase.lastQuadricPair, added for the same reason.
  lastQuadricPair: { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null;
  lastGridPeriodPhase: GridPeriodPhaseResult | null;
  lastRecoveredAxes: RecoveredAxes | null;
  lastDecodeGrid: DecodeSampleGrid | null;
  lastDecodeRotated: DecodeSampleGrid | null;
  lastDecodeCorrectness: (DecodeCellDebug | null)[][] | null;
  lastPositionDecode: PositionDecodeResult | null;
}

export interface PoseComputeTiming {
  votesMs: number; fitMs: number; poseMs: number; distanceMs: number; decodeMs: number;
  // How many refineOrientationIRLS iterations actually ran before converging
  // (or hitting worldVoteRefineSteps) -- null when useWorldVoteOrientation is
  // off, since fitPairOfPlanes has no iteration concept to report.
  worldVoteIterations: number | null;
}

// The residency's fx/fy as the GradientField shape the CPU-only consumers
// still want. Costs a readback if stage 1 ran on GPU, which is exactly why
// only computePixelVotes2x2 calls it -- everything else on the pose path is
// either device-agnostic or already inside the chain.
async function gradientFieldCPU(res: FieldResidency, w: number, h: number): Promise<GradientField> {
  return { fx: await res.cpuF64('fx'), fy: await res.cpuF64('fy'), w, h, r: 1 };
}

// Stages 1-6 -- gradient, the LSD chain, the join walk, and vote casting --
// under ONE FieldResidency, which is the whole reason they are grouped into a
// function here. It spans exactly as far as it is useful: gray goes in at the
// top, votes come out at the bottom, and every per-pixel intermediate in
// between is destroyed on the way out. Everything after this point in
// computePoseFromCapture works on votes and poses, not on pixels, so extending
// the residency past it would only hold GPU memory for no reader.
async function computeCompositesAndVotes(
  state: PoseComputeState, gray: Float64Array, w: number, h: number, vFovRad: number,
): Promise<{ voteComposites: { root: number; line: CompositeLine }[]; votes: Vote[] }> {
  const res = await createLsdChainResidency(gray, w, h);
  try {
    // Composite lines (bucket-fill -> join walk -> merge groups -> one line
    // per group, over the 2x2 gradient field) computed exactly once here and
    // shared by every downstream consumer that needs them -- vote casting
    // below and row/col family classification in computeGridPeriodPhase
    // further down (and, for a real desktop camera, the "color composite
    // lines by row/col family" debug overlay, which reads
    // state.lastVoteComposites back off the camera afterward).
    const compositesSpan = spanStart('composites (2x2 gradient field)');
    const voteComposites = await computeGradient2x2Composites(state.settings, res, w, h);
    spanEnd(compositesSpan);

    const votesSpan = spanStart('votes (segments)');
    // useWorldVoteOrientation swaps the vote SOURCE (one per pixel, straight
    // off the same gradient field/composites already computed above, instead
    // of one per composite line) -- voteComposites/gridPeriodPhase downstream
    // are untouched either way, see this session's chat. It is also the one
    // consumer that forces stage 1 back onto the CPU, hence the explicit
    // gradientFieldCPU rather than a field carried down from above.
    const votes = state.settings.useWorldVoteOrientation
      ? computePixelVotes2x2(await gradientFieldCPU(res, w, h), w, h, MATH_QUAT, vFovRad, state.aspect)
      : computeSegmentVotes(voteComposites, w, h, MATH_QUAT, vFovRad, state.aspect);
    spanEnd(votesSpan);

    return { voteComposites, votes };
  } finally {
    res.destroy();
  }
}

// Mutates `state` in place exactly like recomputeStages used to mutate
// `camera` -- same field names, same functions, same order:
// the LSD chain (stages 1-4, see runLsdChain) -> computeGradient2x2Composites ->
// computeSegmentVotes -> fitPairOfPlanes[GPU] -> handedness assembly ->
// computeGridPeriodPhase -> assemble RecoveredAxes -> runPositionDecode.
// No `useGPU` parameter -- every GPU/CPU branch point reads the shared
// globalState singleton exactly as it does today; the phone gets its own
// independent globalState module instance (separate JS realm, not shared
// memory with the desktop), kept in sync by settingsSync (see
// mobileCapture.ts), so this needs no parameter threading at all.
export async function computePoseFromCapture(
  state: PoseComputeState, gray: Float64Array, w: number, h: number,
): Promise<PoseComputeTiming> {
  const vFovRad = getAnalysisVFovRad(state);
  const t0 = performance.now();

  const { voteComposites, votes } = await computeCompositesAndVotes(state, gray, w, h, vFovRad);
  state.lastVoteComposites = voteComposites;
  state.lastVotes = votes;
  const t1 = performance.now();

  const fitSpan = spanStart('fit (fitPairOfPlanes)');
  let quadricPair: { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null;
  let worldVoteIterations: number | null = null;
  if (state.settings.useWorldVoteOrientation) {
    // CPU-only for now, no GPU port yet -- see this session's chat.
    const irlsSpan = spanStart('fitPairOfPlanes (IRLS, CPU)');
    const refined = refineOrientationIRLS(votes, state.settings.weightSharpenPower, state.settings.worldVoteRefineSteps);
    quadricPair = refined;
    worldVoteIterations = refined?.iterations ?? null;
    if (refined) {
      // Rewrite each vote's OWN weight to its final combined (magnitude x
      // residual-agreement, against the converged poles) trust -- so the
      // rings overlay (updateGradientCirclesDebug, keyed off
      // camera.lastVotes) visibly shows corner/noise votes faded out by
      // however many iterations actually ran, not just their raw input
      // gradient magnitude. Purely a display rewrite: the fit itself is
      // already finished, nothing re-reads these weights for math.
      let maxW = 0;
      for (const v of votes) if (v.weight > maxW) maxW = v.weight;
      for (const v of votes) {
        const magnitudeWeight = maxW > 0 ? Math.pow(v.weight / maxW, state.settings.weightSharpenPower) : 0;
        const residual = Math.abs(fourFoldResidual(v.n, refined.Drow, refined.Dcol));
        v.weight = magnitudeWeight * THREE.MathUtils.clamp(1 - residual, 0, 1);
      }
    }
    spanEnd(irlsSpan);
  } else {
    // Same fallback pattern as every other GPU sub-pipeline: fitPairOfPlanes
    // stays the source of truth, the GPU version is verified against it.
    const fitOnlySpan = spanStart(globalState.useGPUFit ? 'fitPairOfPlanes (GPU)' : 'fitPairOfPlanes (CPU)');
    quadricPair = globalState.useGPUFit
      ? (await fitPairOfPlanesGPU(votes, state.settings.weightSharpenPower))
        ?? fitPairOfPlanes(votes, state.settings.weightSharpenPower)
      : fitPairOfPlanes(votes, state.settings.weightSharpenPower);
    spanEnd(fitOnlySpan);
  }
  spanEnd(fitSpan);
  const t2 = performance.now();

  const poseAssemblySpan = spanStart('poseAssembly');
  let rowDirRecovered: THREE.Vector3 | null = null, colDirRecovered: THREE.Vector3 | null = null;
  if (quadricPair) {
    const normalForHandedness = quadricPair.Dnormal.clone();
    if (cornerDir(0, 0, MATH_QUAT, vFovRad, state.aspect).dot(normalForHandedness) > 0) normalForHandedness.negate();
    rowDirRecovered = quadricPair.Drow.clone();
    colDirRecovered = quadricPair.Dcol.clone();
    const handedness = rowDirRecovered.clone().cross(colDirRecovered).dot(normalForHandedness);
    if (handedness > 0) colDirRecovered.negate();
  }
  // Captured AFTER handedness correction (matching what axesReconstruction.ts's
  // pole markers actually rendered pre-refactor off their own local
  // rowDirRecovered/colDirRecovered vars) but BEFORE gridPeriodPhase's
  // period-search gating -- so pole markers still have something to render
  // off even when gridPeriodPhase fails below and lastRecoveredAxes ends up
  // null. Deliberately NOT the raw pre-handedness-correction quadricPair.
  state.lastQuadricPair = (quadricPair && rowDirRecovered && colDirRecovered)
    ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: quadricPair.Dnormal }
    : null;
  spanEnd(poseAssemblySpan);
  const t3 = performance.now();

  // Grid period/phase is the SOLE source of state.lastRecoveredAxes.distance
  // -- runs unconditionally whenever a quadric pair was found (real distance
  // depends on it).
  const gppSpan = spanStart('gridPeriodPhase (distance source)');
  const gpp = rowDirRecovered && colDirRecovered && quadricPair
    ? await computeGridPeriodPhase(
        voteComposites, gray, w, h, MATH_QUAT, vFovRad, state.aspect,
        rowDirRecovered, colDirRecovered, quadricPair.Dnormal, GRID_STEP,
        state.settings.minGrazingCos,
      )
    : null;
  state.lastGridPeriodPhase = gpp;
  spanEnd(gppSpan);
  const t4 = performance.now();

  state.lastRecoveredAxes = rowDirRecovered && colDirRecovered && quadricPair && gpp
    ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: quadricPair.Dnormal, distance: gpp.height ?? 1 }
    : null;

  const decodeSpan = spanStart('positionDecode');
  await runPositionDecode(state, gray, w, h, vFovRad);
  spanEnd(decodeSpan);
  const t5 = performance.now();

  return { votesMs: t1 - t0, fitMs: t2 - t1, poseMs: t3 - t2, distanceMs: t4 - t3, decodeMs: t5 - t4, worldVoteIterations };
}
