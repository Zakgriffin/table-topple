import * as THREE from 'three';
import { GRID_STEP, MATH_QUAT } from '../sphereLab/constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../sphereLab/math/geometry.ts';
import { type TransferSummary, readSliceF32, readSliceU32 } from './gpu/device.ts';
import { readRegionMembers } from './stages/lsd/lsdFit.gpu.ts';
import { fitPairOfPlanesGPU } from './stages/votes/fitPlanes.gpu.ts';
import { spanDurationMs, spanEnd } from '../sphereLab/profiling/profiler.ts';
import { poseSpan } from './timing/stages.ts';
import { type CompositeLine, type PositionDecodeResult, type RecoveredAxes, type Vote } from './results.ts';
import { type Backend } from './backend.ts';
import { type PendingDecodeGrid, runPositionDecode } from './stages/decode/decodeGrid.ts';
import { computeGridPeriodPhase, type GridPeriodPhaseResult } from './stages/period/gridPeriodPhase.ts';
import { type Intermediates, type IntermediatesRequest, NO_INTERMEDIATES, type PendingIntermediates } from './intermediates.ts';
import { type LsdChainCPU, type LsdChainGPU } from './stages/lsd/chain.ts';
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
// (axesReconstruction.ts's own overlaySpan) stays there, driven off the
// quadricPair/recoveredAxes/positionDecode this hands back.

// ── What the pipeline READS ───────────────────────────────────────────────
//
// The camera parameters a reconstruction runs under, and nothing else. A real
// `Camera` satisfies this structurally, as does a HarnessInput -- which is the
// whole point of it being this small: a caller does not have to build a
// pipeline-shaped object to run the pipeline.
//
// It was called `PoseComputeState` and carried the fourteen result fields as
// well, because the pipeline wrote its output BY MUTATING it. That is the
// tangle this step exists to undo: with inputs and outputs in one type, the
// library's output type was a subset of the UI's camera model, "is this field
// an input or a result?" had no answer, and every field existed and was
// written on every run whether anyone wanted it or not.
export interface PoseInput {
  aspect: number;
  settings: LsdCompositeSettings & {
    horizFovDeg: number; gridPeriodPhaseGapLowerBound: number; minGrazingCos: number;
  };
}

// The five stage spans, read straight off the record objects. These do NOT sum
// to a reconstruction: the gray upload (`pose.residency`) sits ahead of all of
// them and is deliberately outside `votesMs` -- the decode stage reuses that
// same residency, so charging its cost to votes was a misattribution. It has
// its own row in the span breakdown and heads the critical path.
export interface PoseComputeTiming {
  votesMs: number; fitMs: number; poseMs: number; distanceMs: number; decodeMs: number;
}

// ── What the pipeline RETURNS ─────────────────────────────────────────────
//
// One object, built fresh per run, with no `last` in any name -- these are not
// "the most recent" anything, they are THIS call's answer. A caller that keeps
// them around is the one for whom they become "last", and that is its own
// bookkeeping rather than the library's (see axesReconstruction.ts's
// applyPoseResult, the single site where they land on a Camera).
//
// The staleness hazard that shaped the old design is gone by construction
// rather than by guard. Mutating in place meant a repaint could catch a camera
// halfway through a reconstruction and paint new axes against an old decode --
// a picture that never existed -- so the fields had to be nulled in a specific
// order inside the stages and a mutual-exclusion flag had to keep repaints out
// of the window entirely. An atomic swap of one returned object cannot be
// observed halfway.
export interface PoseResult {
  // Not nullable, unlike everything below them: a run that got as far as
  // returning cast its votes, even if the fit that consumes them degenerated.
  // The old state object had them nullable because it existed BEFORE the run
  // as well as after.
  voteComposites: { root: number; line: CompositeLine }[];
  votes: Vote[];
  // Raw fit result (Drow/Dcol/Dnormal only, no distance) BEFORE period-search
  // gating -- axesReconstruction.ts's pole markers render off it even when
  // gridPeriodPhase fails and `recoveredAxes` ends up null. Its own field so
  // that behavior survives instead of silently regressing.
  quadricPair: { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null;
  gridPeriodPhase: GridPeriodPhaseResult | null;
  recoveredAxes: RecoveredAxes | null;
  positionDecode: PositionDecodeResult | null;
  // What the LSD chain's toggle configuration actually cost the bus on this
  // run, straight off the residency's own ledger. Recorded rather than
  // derived because "how many crossings does this configuration imply" is not
  // answerable by reading the toggles: it depends on which stages fell back to
  // CPU, and on which optional readbacks a consumer happened to ask for.
  chainTransfers: TransferSummary | null;
  timing: PoseComputeTiming;
  // Set exactly when the caller asked for intermediates (see
  // pose/intermediates.ts), and null otherwise -- which is what makes "asking
  // for nothing costs nothing" literally true: there is no handle and nothing
  // to drain. Whoever holds it owns resolving or releasing it; left unresolved
  // it holds the chain's device buffers until device loss.
  //
  // The drained payload is the resolve() call's RETURN VALUE, not a field
  // here. A result object that filled itself in later would be exactly the
  // mutable shared state this type replaced, one level down.
  pending: PendingIntermediates | null;
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
  input: PoseInput, gray: Float64Array, w: number, h: number, vFovRad: number,
  backend: Backend, wantMembers: boolean,
): Promise<{
  voteComposites: { root: number; line: CompositeLine }[]; votes: Vote[];
  rects: LsdRectangle[]; chain: LsdChainGPU | null; cpu: LsdChainCPU | null;
}> {
  // Composite lines (one per accepted LSD rectangle, over the 2x2 gradient
  // field) computed exactly once here and shared by every downstream consumer
  // that needs them -- vote casting below and row/col family classification in
  // computeGridPeriodPhase further down (and, for a real desktop camera, the
  // "color composite lines by row/col family" debug overlay, which reads
  // voteComposites back off the camera afterward).
  const compositesSpan = poseSpan('pose.composites');
  const { composites: voteComposites, rects, chain, cpu } =
    await computeGradient2x2Composites(input.settings, gray, w, h, backend, wantMembers);
  spanEnd(compositesSpan);

  // Contains no await, so this stage's duration is host CPU and is declared
  // `sync` in the stage table. That used to be a branch: the retired per-pixel
  // "world votes" source awaited a gradient readback here, making its duration
  // wall time containing a fence.
  const votesSpan = poseSpan('votes.segments');
  const votes = computeSegmentVotes(voteComposites, w, h, MATH_QUAT, vFovRad, input.aspect);
  spanEnd(votesSpan);

  return { voteComposites, votes, rects, chain, cpu };
}

// One reconstruction, start to finish, RETURNED. The order is what
// recomputeStages used to run inline: the LSD chain (stages 1-4, see
// runLsdChain) -> computeGradient2x2Composites -> computeSegmentVotes ->
// fitPairOfPlanes[GPU] -> handedness assembly -> computeGridPeriodPhase ->
// assemble RecoveredAxes -> runPositionDecode.
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
// display might want, leaving a null grid behind a caller's back. This is the
// caller SAYING so up front, which is the difference between a guess and a
// contract: axesReconstruction and mobileCapture both ask for 'decodeGrid', so
// nothing either of them reads has changed.
//
// RELEASING A PREVIOUS RUN'S HANDLE IS THE CALLER'S JOB NOW, and it has to be:
// this function used to find the stale handle on the state object it was about
// to overwrite and release it there. With nothing shared between two calls
// there is no stale pointer here to find, so a caller that keeps handles
// around (axesReconstruction, which parks one on the Camera for a frame)
// releases the one it is replacing. That is a real transfer of responsibility
// rather than a deletion -- the alternative is the pipeline reaching into its
// caller's storage, which is what this whole step removes.
export async function computePoseFromCapture(
  input: PoseInput, gray: Float64Array, w: number, h: number, backend: Backend,
  want: IntermediatesRequest = NO_INTERMEDIATES,
): Promise<PoseResult> {
  // Filled inside the try and read in the finally, so the chain's slices reach
  // the handle even on the error unwind.
  let pendingGrid: PendingDecodeGrid | null = null;
  let rects: LsdRectangle[] | null = null;
  let chain: LsdChainGPU | null = null;
  let cpuChain: LsdChainCPU | null = null;
  const vFovRad = getAnalysisVFovRad(input);

  // The root of this reconstruction, and every other span below is declared to
  // live inside it (see pose/timing/stages.ts). It did not exist while
  // structure came from the call stack, because the stack supplied a root for
  // free -- whatever the caller happened to have open. Now that structure is
  // declared, the library has to state its own top: without this, `pose.fit`
  // and friends would each be separate roots and no report could say what one
  // reconstruction cost without adding them up and hoping.
  const runSpan = poseSpan('pose.run', { backend, want: want.size });

  // Owned here rather than inside computeCompositesAndVotes so `gray` stays on
  // the device long enough for the fused decode at the bottom to reuse it --
  // see that function's own comment. Destroyed in the finally, which is also
  // what makes it safe to hand a raw GPUBuffer to runPositionDecode: nothing
  // outlives this call.
  // Assembled in the try and DECORATED in the finally -- see there.
  let result: PoseResult | null = null;

  // IS `votesMs` -- see the return statement. It is also what makes the
  // subtree's self times readable as a decomposition of a stage rather than as
  // free-floating durations: reconstructionTiming subtracts the children from
  // this span to get the part of the votes stage that is in no child span at
  // all.
  //
  // There used to be a `pose.residency` span ahead of this one, for the gray
  // upload. It is gone with the residency: `createLsdChainResidency` only ever
  // recorded a CPU array into a map, and the 1.19MB crossing actually happened
  // later, at the first `res.gpu('gray')` inside the gradient stage -- which the
  // stage table's own comment had already noticed. The upload is now written
  // where it happens, inside runLsdChainGPU, and charged to `lsd.gradient`.
  const stageSpan = poseSpan('pose.votes');
  try {
    const composited = await computeCompositesAndVotes(input, gray, w, h, vFovRad, backend, want.has('rects'));
    const { voteComposites, votes } = composited;
    rects = composited.rects;
    chain = composited.chain;
    cpuChain = composited.cpu;
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
    const fitSpan = poseSpan('pose.fit', { backend });
    const quadricPair: { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null =
      backend === 'gpu'
        ? (await fitPairOfPlanesGPU(votes)) ?? fitPairOfPlanes(votes)
        : fitPairOfPlanes(votes);
    spanEnd(fitSpan);

    const poseAssemblySpan = poseSpan('pose.assembly');
    let rowDirRecovered: THREE.Vector3 | null = null, colDirRecovered: THREE.Vector3 | null = null;
    if (quadricPair) {
      const normalForHandedness = quadricPair.Dnormal.clone();
      if (cornerDir(0, 0, MATH_QUAT, vFovRad, input.aspect).dot(normalForHandedness) > 0) normalForHandedness.negate();
      rowDirRecovered = quadricPair.Drow.clone();
      colDirRecovered = quadricPair.Dcol.clone();
      const handedness = rowDirRecovered.clone().cross(colDirRecovered).dot(normalForHandedness);
      if (handedness > 0) colDirRecovered.negate();
    }
    // Captured AFTER handedness correction (matching what axesReconstruction.ts's
    // pole markers actually rendered pre-refactor off their own local
    // rowDirRecovered/colDirRecovered vars) but BEFORE gridPeriodPhase's
    // period-search gating -- so pole markers still have something to render
    // off even when gridPeriodPhase fails below and recoveredAxes ends up null.
    // Deliberately NOT the raw pre-handedness-correction quadricPair.
    const correctedPair = (quadricPair && rowDirRecovered && colDirRecovered)
      ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: quadricPair.Dnormal }
      : null;
    spanEnd(poseAssemblySpan);

    // Grid period/phase is the SOLE source of recoveredAxes.distance -- runs
    // unconditionally whenever a quadric pair was found (real distance depends
    // on it).
    const gppSpan = poseSpan('pose.distance');
    const gpp = rowDirRecovered && colDirRecovered && quadricPair
      ? await computeGridPeriodPhase(
          voteComposites, gray, w, h, MATH_QUAT, vFovRad, input.aspect,
          rowDirRecovered, colDirRecovered, quadricPair.Dnormal, GRID_STEP,
          input.settings.minGrazingCos, backend,
        )
      : null;
    spanEnd(gppSpan);

    const recoveredAxes = rowDirRecovered && colDirRecovered && quadricPair && gpp
      ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: quadricPair.Dnormal, distance: gpp.height ?? 1 }
      : null;

    const decodeSpan = poseSpan('pose.decode');
    // Hand down the chain's own gray buffer when there IS one -- the residency has
    // no device on an all-CPU chain, and `gray` is only device-resident if some
    // stage put it there. Null just means decode uploads its own, as before.
    const sharedGray = chain ? { arena: chain.arena, slice: chain.gray } : null;
    const decoded = await runPositionDecode(
      { aspect: input.aspect, settings: input.settings, recoveredAxes, gridPeriodPhase: gpp },
      gray, w, h, vFovRad, sharedGray, backend, want.has('decodeGrid'),
    );
    pendingGrid = decoded.pendingGrid;
    spanEnd(decodeSpan);

    // Read off the span objects held above, not off a parallel set of
    // performance.now() marks. Those marks existed until the profiler stopped
    // being gated: there were six of them bracketing exactly these five spans,
    // and keeping the two in step was a hand-maintained invariant that
    // reconstructionTiming had to assert on (`stageDeltaMs`) to catch drift.
    // One source, so there is nothing left to drift.
    //
    // Read straight off the record objects, which is the only place a
    // duration ever lived -- and now the only place structure could have come
    // from either, since the recorder keeps no stack (see profiler.ts). These
    // numbers used to carry a caveat: a span's PLACE in the tree could be
    // corrupted by a concurrent operation even though its own start/end never
    // was. The caveat is retired along with the stack.
    result = {
      voteComposites, votes,
      quadricPair: correctedPair,
      gridPeriodPhase: gpp,
      recoveredAxes,
      positionDecode: decoded.decode,
      // Both are filled by the finally below, which runs before this object
      // reaches the caller. Placeholders rather than optional fields so the
      // type stays honest about what a caller receives.
      chainTransfers: null,
      pending: null,
      timing: {
        votesMs: spanDurationMs(stageSpan),
        fitMs: spanDurationMs(fitSpan),
        poseMs: spanDurationMs(poseAssemblySpan),
        distanceMs: spanDurationMs(gppSpan),
        decodeMs: spanDurationMs(decodeSpan),
      },
    };
    return result;
  } finally {
    // In the finally rather than the happy path, so a configuration that threw
    // still reports the traffic it managed to incur.
    //
    // Frozen HERE, at the end of the pose, even when a handle carries the
    // chain's slices onward: this number is what the POSE PATH cost, and that is
    // the question the readout answers. Transfers a later drain incurs on the
    // caller's behalf are that caller's cost, and they are visible as their own
    // profiler records.
    const transfers = chain ? chain.transfers : { crossings: 0, bytes: 0, entries: [] };

    // NOTHING TO DESTROY on either branch, and that is the point of the arena.
    // This used to be the one question that decided the residency's fate --
    // whether to destroy its buffers here or hand ownership to a drain -- and
    // getting it wrong either leaked device memory or freed buffers a display
    // pass was about to read. The chain's slices are freed by the NEXT run's
    // `arena.reset()`, so the only thing still worth handling is the decode
    // grid's own handle, which owns standalone buffers.
    if (result && want.size > 0) {
      result.pending = makePendingIntermediates(chain, cpuChain, want, pendingGrid, rects);
    } else {
      pendingGrid?.release();
    }

    // Decorating the object the `return` above already evaluated, which works
    // because it is a reference and the caller has not seen it yet.
    if (result) result.chainTransfers = transfers;

    // Closed LAST, in the finally, so a run that threw still contributes an
    // interval that contains its own children -- otherwise every span opened
    // before the throw becomes an orphan of a parent that never closed, and
    // the one report most worth reading is the one from the failure.
    spanEnd(runSpan);
  }
}

// Only the decode grid's handle moves in here now. The chain's own slices are
// BORROWED, not owned: they stay valid until the next `arena.reset()`, and a
// drain that runs too late throws a StaleSliceError rather than reading another
// run's bytes. That is the whole of what used to be an ownership-transfer
// invariant with a test attached.
//
// A CPU run hands its arrays over directly -- there is nothing to read back and
// nothing to transfer, which is what "asking costs nothing" means on that path.
function makePendingIntermediates(
  chain: LsdChainGPU | null, cpu: LsdChainCPU | null, want: IntermediatesRequest,
  pendingGrid: PendingDecodeGrid | null, rects: LsdRectangle[] | null,
): PendingIntermediates {
  let spent = false;
  // Held so a second resolve() re-hands the SAME arrays rather than draining
  // twice. Starts as the empty set, which is also what a resolve() after a
  // release() gets -- see PendingIntermediates.
  let drained: Intermediates = {};
  const destroy = () => { pendingGrid?.release(); };
  return {
    async resolve() {
      if (spent) return drained;
      spent = true;
      const drainSpan = poseSpan('pose.drain');
      try {
        const out: Intermediates = {};
        if (cpu) {
          // Already host arrays. No readback, no widening, no transfer.
          if (want.has('fx')) out.fx = cpu.fx;
          if (want.has('fy')) out.fy = cpu.fy;
          if (want.has('regionId')) out.regionId = cpu.regionId;
          if (want.has('regions')) out.regions = cpu.regions as GrownRegion[];
        } else if (chain) {
          const { arena } = chain;
          // Widened here rather than by a residency accessor. f32 is what the
          // device holds; Float64Array is what every CPU consumer expects.
          if (want.has('fx')) out.fx = new Float64Array(await readSliceF32(arena, chain.fx, chain.n * 4, 'fx', 'pose.drain'));
          if (want.has('fy')) out.fy = new Float64Array(await readSliceF32(arena, chain.fy, chain.n * 4, 'fy', 'pose.drain'));
          if (want.has('regionId')) {
            const raw = await readSliceU32(arena, chain.regionId, chain.n * 4, 'regionId', 'pose.drain');
            // BIT REINTERPRETATION, not a numeric convert: that is what makes
            // the -1 sentinel survive the trip (collectRegions' scatter writes a
            // real -1).
            out.regionId = new Int32Array(raw.buffer.slice(0));
          }
          if (want.has('regions')) {
            out.regions = (await readRegionMembers(
              arena, chain.regions, chain.regionCount, chain.memberCount, 'pose.drain',
            )) as GrownRegion[];
          }
        }
        // Stage 4's rectangles are a host-side array the chain already returned
        // and then had nowhere to put. Captured during the run and held.
        if (want.has('rects') && rects) out.rects = rects;
        // Last, so a throw reading a field above still leaves the grid's memory
        // to the destroy below. Folded into the SAME object as everything else:
        // the decode grid used to be the one requestable thing that landed on
        // the caller's state instead of here (see IntermediateName).
        const grid = await pendingGrid?.resolve();
        if (grid) {
          out.decodeGrid = grid.grid;
          out.decodeRotated = grid.rotated;
          out.decodeCorrectness = grid.correctness;
        }
        drained = out;
        return drained;
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
