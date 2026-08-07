import * as THREE from 'three';
import { GRID_STEP, MATH_QUAT } from '../sphereLab/constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../sphereLab/math/geometry.ts';
import { FieldResidency, type TransferSummary } from './gpu/fieldResidency.ts';
import { fitPairOfPlanesGPU } from './stages/votes/fitPlanes.gpu.ts';
import { spanDurationMs, spanEnd, spanStart } from '../sphereLab/profiling/profiler.ts';
import { type CompositeLine, type DecodeCellDebug, type DecodeSampleGrid, type PositionDecodeResult, type RecoveredAxes, type Vote } from '../sphereLab/types.ts';
import { type Backend } from './backend.ts';
import { type PendingDecodeGrid, runPositionDecode } from './stages/decode/decodeGrid.ts';
import { computeGridPeriodPhase, type GridPeriodPhaseResult } from './stages/period/gridPeriodPhase.ts';
import { type Intermediates, type IntermediatesRequest, NO_INTERMEDIATES, type PendingIntermediates } from './intermediates.ts';
import { createLsdChainResidency } from './stages/lsd/chain.ts';
import type { GrownRegion, LsdRectangle } from './stages/lsd/types.ts';
import { computeGradient2x2Composites, computeSegmentVotes, fitPairOfPlanes, type LsdCompositeSettings } from './stages/votes/votes.ts';

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
  // Set exactly when the caller asked for intermediates (see
  // pose/intermediates.ts). Whoever holds it owns resolving or releasing
  // it; left unresolved it holds the chain's device buffers until device loss.
  // Null for the empty request, which is what makes "asking for nothing costs
  // nothing" literally true: there is no handle and nothing to drain.
  pendingIntermediates: PendingIntermediates | null;
  // Filled by that handle's resolve(). Null until then, and null forever for a
  // caller that asked for nothing -- deliberately not an empty object, so
  // "nobody asked" and "asked and got nothing back" are distinguishable.
  intermediates: Intermediates | null;
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
  backend: Backend, wantMembers: boolean,
): Promise<{ voteComposites: { root: number; line: CompositeLine }[]; votes: Vote[]; rects: LsdRectangle[] }> {
  {
    // Composite lines (one per accepted LSD rectangle, over the 2x2 gradient
    // field) computed exactly once here and
    // shared by every downstream consumer that needs them -- vote casting
    // below and row/col family classification in computeGridPeriodPhase
    // further down (and, for a real desktop camera, the "color composite
    // lines by row/col family" debug overlay, which reads
    // state.lastVoteComposites back off the camera afterward).
    const compositesSpan = spanStart('composites (2x2 gradient field)');
    const { composites: voteComposites, rects } = await computeGradient2x2Composites(state.settings, res, w, h, backend, wantMembers);
    spanEnd(compositesSpan);

    // Contains no await, so this span's duration is host CPU and is marked
    // sync unconditionally. That used to be a branch: the retired per-pixel
    // "world votes" source awaited a gradient readback here, making its
    // duration wall time containing a fence.
    const votesSpan = spanStart('votes (segments)', true);
    const votes = computeSegmentVotes(voteComposites, w, h, MATH_QUAT, vFovRad, state.aspect);
    spanEnd(votesSpan);

    return { voteComposites, votes, rects };
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
// produced it and could not be re-derived. See pose/backend.ts.
//
// `want` is what the CALLER asks to be handed back, as a set of names -- see
// pose/intermediates.ts for the whole argument. It replaces the single
// `deferDecodeGrid` boolean, whose own comment was the tell that deferral was a
// per-call-site convention: "this function has three callers and only one of
// them can honour it."
//
// The empty request is the fast path AND the default, and it is exact: no
// handle is created, the residency is destroyed in the finally exactly as
// before, and the decode grid is never brought down at all. That last part is a
// real behaviour change from `deferDecodeGrid=false`, which always read the
// 0.45MB grid back. It is not the view-conditional SKIP that was tried and
// rejected here (see runPositionDecode) -- that had the pipeline guessing what
// display might want, leaving lastDecodeGrid null behind a caller's back. This
// is the caller SAYING so up front, which is the difference between a guess and
// a contract: axesReconstruction and mobileCapture both ask for 'decodeGrid',
// so nothing either of them reads has changed.
export async function computePoseFromCapture(
  state: PoseComputeState, gray: Float64Array, w: number, h: number, backend: Backend,
  want: IntermediatesRequest = NO_INTERMEDIATES,
): Promise<PoseComputeTiming> {
  // A previous run's handle can still be here if its owner never drained it.
  // Released BEFORE this run overwrites the pointer, since nothing else holds
  // it and its buffers would otherwise live until device loss. This used to sit
  // inside runPositionDecode, which could only ever release the decode grid's
  // half of it.
  state.pendingIntermediates?.release();
  state.pendingIntermediates = null;
  state.intermediates = null;

  // Both are filled inside the try and read in the finally, so the residency
  // reaches the handle even on the error unwind rather than leaking.
  let pendingGrid: PendingDecodeGrid | null = null;
  let rects: LsdRectangle[] | null = null;
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
    const composited = await computeCompositesAndVotes(state, res, gray, w, h, vFovRad, backend, want.has('rects'));
    const { voteComposites, votes } = composited;
    rects = composited.rects;
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
    pendingGrid = await runPositionDecode(
      state, gray, w, h, vFovRad, sharedGray, backend, want.has('decodeGrid'),
    );
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
    //
    // Frozen HERE, at the end of the pose, even when a handle carries the
    // residency onward: this number is what the POSE PATH cost, and that is the
    // question the readout answers. Transfers a later drain incurs on the
    // caller's behalf are that caller's cost, not the pipeline's, and they are
    // visible as their own profiler spans.
    state.lastChainTransfers = res.summary();
    if (want.size === 0) {
      // The empty request, unchanged from before any of this existed. No
      // handle, no readback, nothing for anyone to remember to drain.
      pendingGrid?.release();
      res.destroy();
    } else {
      state.pendingIntermediates = makePendingIntermediates(state, res, want, pendingGrid, rects);
    }
  }
}

// Ownership of the residency (and of the decode grid's own handle, when there
// is one) moves in here. Both are destroyed exactly once, by whichever of
// resolve/release runs first -- see PendingIntermediates on why both have to be
// safe twice and in either order.
function makePendingIntermediates(
  state: PoseComputeState, res: FieldResidency, want: IntermediatesRequest,
  pendingGrid: PendingDecodeGrid | null, rects: LsdRectangle[] | null,
): PendingIntermediates {
  let spent = false;
  const destroy = () => { pendingGrid?.release(); res.destroy(); };
  return {
    async resolve() {
      if (spent) return;
      spent = true;
      const drainSpan = spanStart('intermediates drain');
      try {
        const out: Intermediates = {};
        // Straight off the residency's own accessors, which already know which
        // side each field is on and transfer only if the sides differ. A field
        // the chain happened to leave on the CPU costs nothing to hand over.
        if (want.has('fx')) out.fx = await res.cpuF64('fx');
        if (want.has('fy')) out.fy = await res.cpuF64('fy');
        if (want.has('regionId')) out.regionId = await res.cpuI32('regionId');
        if (want.has('regions')) out.regions = await res.regionsCPU() as GrownRegion[];
        // Not from the residency: stage 4's rectangles are a host-side array
        // the chain returns and then had nowhere to put. Captured during the
        // run and simply held until now.
        if (want.has('rects') && rects) out.rects = rects;
        state.intermediates = out;
        // Populates lastDecodeGrid/lastDecodeRotated/lastDecodeCorrectness, and
        // releases its own device buffers on the way. Last, so a throw reading
        // a field above still leaves the grid's memory to the destroy below.
        await pendingGrid?.resolve();
      } finally {
        spanEnd(drainSpan);
        destroy();
      }
    },
    release() {
      if (spent) return;
      spent = true;
      destroy();
    },
  };
}
