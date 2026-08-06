import * as THREE from 'three';
import { GRID_STEP, MATH_QUAT } from '../constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../math/geometry.ts';
import { FieldResidency, type TransferSummary } from '../pipelineGPU/fieldResidency.ts';
import { fitPairOfPlanesGPU } from '../pipelineGPU/fitPlanes.ts';
import { spanDurationMs, spanEnd, spanStart } from '../profiling/profiler.ts';
import { type CompositeLine, type DecodeCellDebug, type DecodeSampleGrid, type PositionDecodeResult, type RecoveredAxes, type Vote } from '../types.ts';
import { type Backend } from './backend.ts';
import { type PendingDecodeGrid, runPositionDecode } from './decodeGrid.ts';
import { computeGridPeriodPhase, type GridPeriodPhaseResult } from './gridPeriodPhase.ts';
import { createLsdChainResidency } from './lsdSegments.ts';
import { computeGradient2x2Composites, computeSegmentVotes, fitPairOfPlanes, type LsdCompositeSettings } from './votes.ts';

// ── Shared pure pose-recovery orchestrator ────────────────────────────────
//
// Extracts the pure prefix of axesReconstruction.ts's recomputeStages (every
// stage through runPositionDecode) into one function that operates on a
// bare data object instead of a real Camera -- no THREE-scene objects, GPU
// textures, or DOM required, so this is exactly what a phone-side caller
// (mobileCapture.ts, see this session's on-device-pose-recovery plan) can
// run standalone. computeProjectedBinsAuto/paintProjectedTexture
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
    horizFovDeg: number; gridPeriodPhaseGapLowerBound: number; minGrazingCos: number;
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
  // Set only when the caller asked to defer the decode grid's readback (see
  // computePoseFromCapture's `deferDecodeGrid`). Whoever set it owns resolving
  // or releasing it; left unresolved it holds ~0.45MB of GPU buffers.
  pendingDecodeGrid: PendingDecodeGrid | null;
  // What the LSD chain's toggle configuration actually cost the bus on the
  // last frame, straight off the residency's own ledger. Recorded rather than
  // derived because "how many crossings does this configuration imply" is not
  // answerable by reading the toggles: it depends on which stages fell back to
  // CPU, and on which optional readbacks a consumer happened to ask for.
  lastChainTransfers: TransferSummary | null;
}

export interface PoseComputeTiming {
  votesMs: number; fitMs: number; poseMs: number; distanceMs: number; decodeMs: number;
}

// Stages 1-4 plus vote casting, over a FieldResidency the CALLER now owns.
//
// The residency used to be created and destroyed right here, on the reasoning
// that "everything after this point works on votes and poses, not on pixels, so
// extending it would only hold GPU memory for no reader." That was WRONG about
// one reader: the fused decode wants `gray`, and with the residency already
// gone it re-uploaded all 1.19MB of it, narrowing f64->f32 again on the way. So
// the lifetime is hoisted into computePoseFromCapture and the buffer is handed
// down to runPositionDecode.
//
// The rest of the reasoning stands -- the per-pixel intermediates really do have
// no reader past this point -- but "no reader" turned out to be a claim worth
// checking against the transfer ledger rather than asserting.
async function computeCompositesAndVotes(
  state: PoseComputeState, res: FieldResidency, gray: Float64Array, w: number, h: number, vFovRad: number,
  backend: Backend,
): Promise<{ voteComposites: { root: number; line: CompositeLine }[]; votes: Vote[] }> {
  {
    // Composite lines (one per accepted LSD rectangle, over the 2x2 gradient
    // field) computed exactly once here and
    // shared by every downstream consumer that needs them -- vote casting
    // below and row/col family classification in computeGridPeriodPhase
    // further down (and, for a real desktop camera, the "color composite
    // lines by row/col family" debug overlay, which reads
    // state.lastVoteComposites back off the camera afterward).
    const compositesSpan = spanStart('composites (2x2 gradient field)');
    const voteComposites = await computeGradient2x2Composites(state.settings, res, w, h, backend);
    spanEnd(compositesSpan);

    // Contains no await, so this span's duration is host CPU and is marked
    // sync unconditionally. That used to be a branch: the retired per-pixel
    // "world votes" source awaited a gradient readback here, making its
    // duration wall time containing a fence.
    const votesSpan = spanStart('votes (segments)', true);
    const votes = computeSegmentVotes(voteComposites, w, h, MATH_QUAT, vFovRad, state.aspect);
    spanEnd(votesSpan);

    return { voteComposites, votes };
  }
}

// Mutates `state` in place exactly like recomputeStages used to mutate
// `camera` -- same field names, same functions, same order:
// the LSD chain (stages 1-4, see runLsdChain) -> computeGradient2x2Composites ->
// computeSegmentVotes -> fitPairOfPlanes[GPU] -> handedness assembly ->
// computeGridPeriodPhase -> assemble RecoveredAxes -> runPositionDecode.
//
// `backend` is now a PARAMETER at every branch point rather than a
// globalState.forceCPU read inside each one. It used to be argued that no
// parameter was needed because the phone gets its own globalState module
// instance kept in sync by settingsSync -- true, but it made the choice ambient,
// so a pose, a timing, or a verify delta did not carry the configuration that
// produced it and could not be re-derived. See pipeline/backend.ts.
//
// `deferDecodeGrid` moves the decode grid's 0.45MB readback off the pose path
// and onto state.pendingDecodeGrid for the caller to drain. It is a PARAMETER
// rather than a globalState read on purpose: this function has three callers and
// only one of them can honour it. axesReconstruction has the visual mailbox and
// passes true; mobileCapture reads state.lastDecodeGrid
// synchronously the moment this returns (buildDebugPayload) and has no drain at
// all; reconstructionTiming releases without resolving, because it is measuring
// the pose path and the readback is no longer on it. A globalState flag would
// have silently deferred on the phone, where nothing would ever resolve it.
export async function computePoseFromCapture(
  state: PoseComputeState, gray: Float64Array, w: number, h: number, backend: Backend,
  deferDecodeGrid = false,
): Promise<PoseComputeTiming> {
  const vFovRad = getAnalysisVFovRad(state);

  // IS `votesMs` -- see the return statement. It is also what makes the
  // subtree's self times readable as a decomposition of a stage rather than as
  // free-floating durations: reconstructionTiming subtracts the children from
  // this span to get the part of the votes stage that is in no child span at
  // all.
  const stageSpan = spanStart('votes stage');

  // Owned here rather than inside computeCompositesAndVotes so `gray` stays on
  // the device long enough for the fused decode at the bottom to reuse it --
  // see that function's own comment. Destroyed in the finally, which is also
  // what makes it safe to hand a raw GPUBuffer to runPositionDecode: nothing
  // outlives this call.
  const residencySpan = spanStart('residency create (gray up)');
  const res = await createLsdChainResidency(gray, w, h, backend);
  spanEnd(residencySpan);
  try {
    const { voteComposites, votes } = await computeCompositesAndVotes(state, res, gray, w, h, vFovRad, backend);
    state.lastVoteComposites = voteComposites;
    state.lastVotes = votes;
    spanEnd(stageSpan);

    // Same fallback pattern as every other GPU sub-pipeline: fitPairOfPlanes
    // stays the source of truth, the GPU version is verified against it.
    //
    // ONE span, not two. There used to be an outer 'fit (fitPairOfPlanes)'
    // wrapping this one, because the fit had two alternative implementations
    // (this, and the retired IRLS refinement) and `fitMs` had to cover either.
    // With one implementation left the outer span covered exactly the same
    // interval as this one, i.e. a guaranteed-zero self time in the harness'
    // span tree. `fitMs` reads this span directly now.
    const fitSpan = spanStart(backend === 'cpu' ? 'fitPairOfPlanes (CPU)' : 'fitPairOfPlanes (GPU)');
    const quadricPair: { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null =
      backend === 'gpu'
        ? (await fitPairOfPlanesGPU(votes)) ?? fitPairOfPlanes(votes)
        : fitPairOfPlanes(votes);
    spanEnd(fitSpan);

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

    // Grid period/phase is the SOLE source of state.lastRecoveredAxes.distance
    // -- runs unconditionally whenever a quadric pair was found (real distance
    // depends on it).
    const gppSpan = spanStart('gridPeriodPhase (distance source)');
    const gpp = rowDirRecovered && colDirRecovered && quadricPair
      ? await computeGridPeriodPhase(
          voteComposites, gray, w, h, MATH_QUAT, vFovRad, state.aspect,
          rowDirRecovered, colDirRecovered, quadricPair.Dnormal, GRID_STEP,
          state.settings.minGrazingCos, backend,
        )
      : null;
    state.lastGridPeriodPhase = gpp;
    spanEnd(gppSpan);

    state.lastRecoveredAxes = rowDirRecovered && colDirRecovered && quadricPair && gpp
      ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: quadricPair.Dnormal, distance: gpp.height ?? 1 }
      : null;

    const decodeSpan = spanStart('positionDecode');
    // Hand down the chain's own gray buffer when there IS one -- the residency has
    // no device on an all-CPU chain, and `gray` is only device-resident if some
    // stage put it there. Null just means decode uploads its own, as before.
    const sharedGray = res.device && res.hasGPU('gray') ? res.gpu('gray') : null;
    await runPositionDecode(state, gray, w, h, vFovRad, sharedGray, backend, deferDecodeGrid);
    spanEnd(decodeSpan);

    // Read off the span objects held above, not off a parallel set of
    // performance.now() marks. Those marks existed until the profiler stopped
    // being gated: there were six of them bracketing exactly these five spans,
    // and keeping the two in step was a hand-maintained invariant that
    // reconstructionTiming had to assert on (`stageDeltaMs`) to catch drift.
    // One source, so there is nothing left to drift.
    //
    // Immune to the span tree's reparenting hazard (see profiler.ts's header):
    // that corrupts a span's PLACE in the tree, never its own start/end, and
    // these are read straight from the objects.
    return {
      votesMs: spanDurationMs(stageSpan),
      fitMs: spanDurationMs(fitSpan),
      poseMs: spanDurationMs(poseAssemblySpan),
      distanceMs: spanDurationMs(gppSpan),
      decodeMs: spanDurationMs(decodeSpan),
    };
  } finally {
    // Read BEFORE destroy, and in the finally rather than the happy path, so a
    // configuration that threw still reports the traffic it managed to incur.
    state.lastChainTransfers = res.summary();
    res.destroy();
  }
}
