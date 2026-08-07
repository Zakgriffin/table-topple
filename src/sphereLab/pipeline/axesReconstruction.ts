import * as THREE from 'three';
import { type Camera } from '../camera/model.ts';
import { activeCamera, isPhysical, isSimulated } from '../camera/store.ts';
import { COL_DIR, ROW_DIR, SPHERE_RADIUS } from '../constants.ts';
import { angleBetweenDegV } from '../math/geometry.ts';
import { updatePositionReadoutText } from '../overlays/projectedCamOverlays.ts';
import { applyRecoveredFloorOverlay, updateRecoveredCamGizmo, updateRecoveredFloorOutline } from '../overlays/recoveredOverlays.ts';
import { updateGradientCirclesDebug } from '../overlays/sphereOverlays.ts';
import { globalState } from '../state.ts';
import { axesReadout, captureAxesBtn, lsdChainTransfers } from '../ui/dom.ts';
import { backendFromForceCPU } from './backend.ts';
import { captureDistortedGrayscale } from './capture.ts';
import { computeProjectedBinsAuto, paintProjectedTexture, type ProjectedSampleResult } from './decodeGrid.ts';
import { flipRowsF64 } from './distortion.ts';
import { refreshModeVisualizations } from './modeRefresh.ts';
import { type IntermediateName, type IntermediatesRequest, wants } from './intermediates.ts';
import { computePoseFromCapture } from './poseCompute.ts';
import { type ProfileSpan, spanEnd, spanStart } from '../profiling/profiler.ts';

// Shared pole-marker/gizmo/floor-overlay/readout tail -- called after EITHER
// a real local reconstruction (recomputeStages below) or an already-computed
// pose arriving from a device-compute phone (pipeline/capture.ts's
// ingestRemotePose), so both paths render identically off the same
// lastQuadricPair/lastRecoveredAxes/lastPositionDecode fields instead of
// duplicating this logic -- see this session's on-device-pose-recovery
// plan. `extraReadoutLine`, if given, is appended as the readout's last
// line (recomputeStages passes its per-stage timing breakdown; a
// device-compute pose has no local timing to report, so ingestRemotePose
// passes nothing).
export function applyPoseVisualizations(camera: Camera, isActive: boolean, extraReadoutLine?: string) {
  let orientationErrorLine: string | null = null;
  // Pole markers read camera.lastQuadricPair (added specifically so this
  // survives even when gridPeriodPhase fails and lastRecoveredAxes ends up
  // null -- see camera/model.ts's own comment on the field) instead of
  // locally recomputing rowDirRecovered/colDirRecovered.
  const rowDirRecovered = camera.lastQuadricPair?.Drow ?? null;
  const colDirRecovered = camera.lastQuadricPair?.Dcol ?? null;
  if (camera.lastPositionDecode && rowDirRecovered && colDirRecovered) {
    const { recoveredCamQuat } = camera.lastPositionDecode;
    const rowDirWorld = rowDirRecovered.clone().applyQuaternion(recoveredCamQuat);
    const colDirWorld = colDirRecovered.clone().applyQuaternion(recoveredCamQuat);
    // Decode's own 4-way disambiguation (tallyPositionVotes, see
    // decodeGrid.ts) can legitimately swap which of Drow/Dcol maps to
    // the world ROW vs COL axis (and negate either) -- fitPairOfPlanes
    // only ever recovers the row/col PLANE PAIR up to that ambiguity,
    // by construction (it's a property of the quadric fit, not a bug).
    // axisErr is UNDIRECTED (angle to the nearer of +axis/-axis) since
    // both ends of an axis already get their own pole marker -- a
    // clean 180-degree flip isn't actually wrong, just a labeling
    // choice for which end is which. Picking whichever of the two
    // (row->ROW,col->COL) / (row->COL,col->ROW) pairings has the lower
    // TOTAL undirected error is legitimate here (unlike the old
    // pre-decode version of this check used to be) because
    // rowDirWorld/colDirWorld are genuinely in world space now --
    // decode has already resolved which pairing is physically correct,
    // this just detects which one it was.
    const axisErr = (v: THREE.Vector3, axis: THREE.Vector3) => Math.min(angleBetweenDegV(v, axis), angleBetweenDegV(v, axis.clone().negate()));
    const errUnswapped = axisErr(rowDirWorld, ROW_DIR) + axisErr(colDirWorld, COL_DIR);
    const errSwapped = axisErr(rowDirWorld, COL_DIR) + axisErr(colDirWorld, ROW_DIR);
    const swapped = errSwapped < errUnswapped;
    // Red pole markers always track whichever recovered vector ended
    // up closest to the world ROW axis, blue always tracks whichever
    // is closest to COL -- a fixed rowDirWorld->red assignment would
    // sometimes put red poles next to the blue ground-truth poles
    // whenever swapped is true.
    const redDirWorld = swapped ? colDirWorld : rowDirWorld;
    const blueDirWorld = swapped ? rowDirWorld : colDirWorld;
    camera.recoveredRowPoleA.position.copy(redDirWorld).multiplyScalar(SPHERE_RADIUS);
    camera.recoveredRowPoleB.position.copy(redDirWorld).multiplyScalar(-SPHERE_RADIUS);
    camera.recoveredColPoleA.position.copy(blueDirWorld).multiplyScalar(SPHERE_RADIUS);
    camera.recoveredColPoleB.position.copy(blueDirWorld).multiplyScalar(-SPHERE_RADIUS);

    if (isSimulated(camera)) {
      const rowErr = axisErr(redDirWorld, ROW_DIR);
      const colErr = axisErr(blueDirWorld, COL_DIR);
      orientationErrorLine = `row err ${rowErr.toFixed(2)}°  col err ${colErr.toFixed(2)}°  [post-decode${swapped ? ', swapped' : ''}]`;
    }
  }
  updateRecoveredCamGizmo(camera);
  applyRecoveredFloorOverlay(camera);
  // Drawn unconditionally alongside the fill above -- guards on
  // lastRecoveredAxes/lastPositionDecode only (both present in EITHER
  // compute mode), unlike applyRecoveredFloorOverlay which still requires
  // real pixel data (lastProjectedBins).
  updateRecoveredFloorOutline(camera);

  if (isActive) {
    const haveGroundTruth = isSimulated(camera);
    const lines = [`${camera.lastVotes.length} votes  (${camera.lastVotes.length} fed to fit)`];
    if (rowDirRecovered && colDirRecovered) {
      if (orientationErrorLine) lines.push(orientationErrorLine);
    } else {
      lines.push(`degenerate fit`);
    }
    const gpp = camera.lastGridPeriodPhase;
    if (camera.lastRecoveredAxes && gpp) {
      const trueDist = isSimulated(camera) ? camera.camPos.y : NaN;
      const dist = camera.lastRecoveredAxes.distance;
      if (haveGroundTruth) {
        const err = (Math.abs(dist - trueDist) / trueDist) * 100;
        lines.push(`distance ${dist.toFixed(2)} (${err.toFixed(1)}% err)  true ${trueDist.toFixed(2)}  period ${gpp.period.toFixed(4)}  [gridPeriodPhase]`);
      } else {
        lines.push(`distance ${dist.toFixed(2)}  period ${gpp.period.toFixed(4)}  [gridPeriodPhase]`);
      }
    } else if (camera.lastQuadricPair) {
      lines.push(`distance: no period found (gridPeriodPhase)`);
    }
    if (camera.lastPositionDecode) {
      lines.push(`decoded torus (row,col): (${camera.lastPositionDecode.row}, ${camera.lastPositionDecode.col})  consistency ${(camera.lastPositionDecode.consistency * 100).toFixed(1)}%  camPos (${camera.lastPositionDecode.camPos.x.toFixed(2)}, ${camera.lastPositionDecode.camPos.y.toFixed(2)}, ${camera.lastPositionDecode.camPos.z.toFixed(2)})`);
    }
    if (extraReadoutLine) lines.push(extraReadoutLine);
    axesReadout.textContent = lines.join('\n');
    updatePositionReadoutText(camera);
  }
}

// ── The LSD chain's bus-traffic readout ──────────────────────────────────
//
// What the chain actually cost the bus on the last frame. It used to answer a
// question the nine per-stage toggles raised -- savings along the chain are
// superadditive, so flipping one checkbox changed the traffic of stages you did
// not touch -- and with one switch left it answers a simpler one: is the chain
// still at ONE crossing (gray up, nothing down), or has something started
// pulling data back down? Sourced from FieldResidency's own ledger, which
// records a transfer at the moment it decides to make one, so the number cannot
// drift from the behaviour the way a hand-maintained estimate would.
//
// Worth reading alongside the measurement that motivated it, and against it:
// crossings stopped predicting wall clock once the grower moved to GPU (the top
// four configurations sit within 10.1-10.8ms regardless of whether they move
// 0.92MB or 2.42MB). So this readout shows what the configuration COSTS THE
// BUS, which is a real and stable quantity -- not what it costs in time.
export function updateChainTransfersReadout(camera: Camera | undefined) {
  // Two different empty states, deliberately not collapsed into one: on the
  // Global tab there may well be a captured camera sitting right there, and
  // saying "no capture yet" would be flatly untrue.
  if (!camera) {
    lsdChainTransfers.textContent = 'LSD chain: select a camera to see its bus traffic.';
    return;
  }
  const s = camera.lastChainTransfers;
  if (!s) {
    lsdChainTransfers.textContent = 'LSD chain: no capture yet.';
    return;
  }
  const mb = s.bytes / 1048576;
  const size = mb >= 0.01 ? `${mb.toFixed(2)} MB` : `${s.bytes} B`;
  const head = `LSD chain: ${s.crossings} crossing${s.crossings === 1 ? '' : 's'}, ${size}`;
  if (s.entries.length === 0) {
    // Genuinely zero, not missing: an all-CPU chain never touches the bus.
    lsdChainTransfers.textContent = `${head} -- nothing crosses (all-CPU chain).`;
    return;
  }
  const detail = s.entries
    .map((e) => `${e.what}${e.direction === 'up' ? '↑' : '↓'}`)
    .join(' ');
  lsdChainTransfers.textContent = `${head}  [${detail}]`;
}

// ── Axes/position reconstruction (the big orchestrator) ──────────────────
//
// Split into two entry points so every settings slider -- not just "capture
// now" -- can recompute everything downstream of it (see this session's
// chat): runAxesReconstruction does the expensive capture (stage 1: a real
// GPU render+readback for simulated, or picking up whatever the phone last
// sent for physical) and stores it into camera.lastAxesCaptureGray;
// recomputeFromLastCapture does everything downstream of that (gradient
// field through pole markers/overlays), reading the cached capture instead
// of taking a fresh one. Both share recomputeStages for that downstream
// work so the guard/busy-UI/span bookkeeping isn't duplicated in a way that
// could drift out of sync.

// ── The display tail, and the mailbox that defers it ─────────────────────
//
// Everything below the pose itself: the projected bins + texture paint, the
// pole markers/floor overlay, and the mode-specific overlays. Split out of
// recomputeStages so it runs deferred, through a one-slot mailbox drained by
// animate(), rather than inline inside the pose window -- that ~20ms of
// display GPU work used to serialize against the same device queue the pose
// stages were using. It costs one animation frame (~16ms) of overlay lag.
//
// The reason a boolean is enough where a work queue looks like it's needed:
// this function reads NOTHING but camera state, and that state is already
// settled by the time it's asked to run. It is idempotent -- running it twice
// paints the same thing, and running it once after three captures paints the
// third, not the first. So the mailbox holds no payload and coalescing is
// free rather than a feature that had to be built.
//
// `isActive` is re-evaluated HERE rather than inherited from whenever the
// reconstruction started, since deferral makes "was this the active camera
// 160ms ago" the wrong question -- what the readouts and mode overlays want
// is which camera is on screen at PAINT time.
async function runVisualTail(camera: Camera): Promise<void> {
  const isActive = camera === activeCamera();

  // FIRST, before anything reads lastDecodeGrid/lastDecodeRotated/
  // lastDecodeCorrectness or camera.intermediates. computePoseFromCapture left
  // those null and parked the readback here (see pipeline/intermediates.ts) so
  // the pose did not have to wait 0.45MB for display data; this is the moment
  // they get filled. updateGradientCirclesDebug, applyPoseVisualizations and
  // every mode overlay below are downstream of it.
  //
  // This is the ONE thing in this function that is not idempotent-by-reading-
  // settled-state: it consumes a handle. resolve() is itself idempotent, so a
  // second drain over the same capture is still safe -- it just finds the fields
  // already populated and the handle already spent.
  const pending = camera.pendingIntermediates;
  if (pending) {
    camera.pendingIntermediates = null;
    await pending.resolve();
  }

  // Painting projectedPreviewTex is a real GPU texture upload -- worth
  // skipping unless this camera's Projected-Cam view is what's actually
  // on screen right now. The numeric half (bins) stays unconditional --
  // camera.lastProjectedBins feeds the World-view floor overlay's decal
  // map for every camera, not just the displayed one. The RGBA half
  // also has to run whenever the World-view floor overlay is on, though
  // -- that overlay (see overlays/recoveredOverlays.ts) reuses
  // projectedPreviewTex as its decal map, so skipping the paint here
  // left it sitting at its all-zero (alpha 0, invisible) initial
  // contents for any camera that never happened to be viewed in
  // Projected-Cam mode first.
  const showProjected = (isActive && globalState.mode === 'projected')
    || (globalState.mode === 'world' && camera.settings.showRecoveredFloor);

  // The readout describes the toggle configuration that produced the frame
  // being painted right now, which is why it sits in the tail rather than
  // next to the chain that recorded it -- deferred, those are different
  // moments, and this is the one the user is looking at.
  if (isActive) updateChainTransfersReadout(camera);
  updateGradientCirclesDebug(camera);

  const projectSpan = spanStart('projectBins (display + decode-marginals bins)');
  const projectStart = performance.now();
  // Captured outside the `if` (stays null when there's no recovered axes to
  // project) so it can be handed to refreshModeVisualizations below instead
  // of that call recomputing the exact same (possibly GPU) result a second
  // time -- see modeRefresh.ts's own comment on precomputedProjection.
  let projResult: ProjectedSampleResult = null;
  if (camera.lastRecoveredAxes) {
    projResult = await computeProjectedBinsAuto(camera, backendFromForceCPU(globalState.forceCPU));
    if (showProjected) paintProjectedTexture(camera, projResult);
  }
  const projectMs = performance.now() - projectStart;
  spanEnd(projectSpan);

  const overlaySpan = spanStart('poleMarkers+overlays');
  // applyPoseVisualizations MUST stay downstream of the projection above, not
  // move back up next to the pose: applyRecoveredFloorOverlay builds the floor
  // quad's geometry out of lastProjectedBins' own u/v extent (see
  // overlays/recoveredOverlays.ts), so hoisting it would size this frame's
  // floor from the PREVIOUS frame's bins. That coupling is the reason the
  // whole tail defers as one unit instead of only its expensive half.
  const t = camera.lastPoseTiming;
  let timingLine: string | undefined;
  if (t) {
    timingLine = `votes ${t.votesMs.toFixed(0)}ms  fit ${t.fitMs.toFixed(0)}ms  pose ${t.poseMs.toFixed(0)}ms  distance ${t.distanceMs.toFixed(0)}ms  project ${projectMs.toFixed(0)}ms  decode ${t.decodeMs.toFixed(0)}ms`;
  }
  applyPoseVisualizations(camera, isActive, timingLine);
  spanEnd(overlaySpan);

  if (isActive) {
    // Everything mode-specific (Through-Cam's contamination/top-gradient/
    // bucket-fill/join/hover overlays and its grid-period/phase plot,
    // Projected-Cam's texture, World's recovered-floor decal) -- see
    // pipeline/modeRefresh.ts. Only meaningful for whichever camera is
    // actually on screen. projResult (computed above, possibly null) is
    // handed over so this doesn't pay for a second (possibly GPU)
    // re-projection of data it already has in hand. NOT part of
    // applyPoseVisualizations -- device-compute mode (ingestRemotePose)
    // deliberately never calls this, since it would re-derive
    // lastProjectedBins from zero real pixel data and undo the "no image ->
    // no fill" guard applyRecoveredFloorOverlay just applied.
    await refreshModeVisualizations(camera, globalState.mode, { value: projResult });
  }
}

// Posts to the mailbox.
function markVisualsDirty(camera: Camera): void {
  camera.visualsDirty = true;
}

// Drains the mailbox for one camera. Called from animate() every tick, for
// every camera -- cheap to the point of free when nothing is dirty, which is
// most ticks.
//
// Two things have to be true to START: something is pending, and this camera
// is not mid-reconstruction. The second is the staleness guard, and it is the
// important one: computePoseFromCapture mutates lastRecoveredAxes,
// lastPositionDecode and friends IN PLACE as it goes, so painting from a
// camera that is halfway through one would mix two frames -- new axes against
// an old decode -- and produce a picture that never existed.
//
// That exclusion is MUTUAL: runAxesReconstruction and recomputeFromLastCapture
// both decline to start while visualsDraining is set. Three reasons, in
// ascending order of how much they matter:
//
//   1. It costs nothing anyone can feel. The window it blocks is the same
//      window that was already blocked before deferral existed -- the tail
//      used to run INSIDE axesCapturing, so "reconstruction + tail" was one
//      uninterruptible stretch either way. All this does is split the flag
//      that covers it in two.
//   2. Overlapping wouldn't have bought much. The tail's cost is display GPU
//      work on the SAME device queue the pose stages use, so running it
//      alongside the next reconstruction trades a serial wait for queue
//      contention. The win here was never parallelism -- it is that the tail
//      stops being AWAITED on the pose path, and it gets that either way by
//      running in the idle gap between captures (~340ms of one, at the default
//      500ms interval against a ~159ms reconstruction).
//   3. It keeps the profiler honest, which is the reason it is unconditional
//      rather than a measurement-only mode. profiling/profiler.ts keeps ONE
//      module-level span stack and openly assumes a single profiled operation
//      is in flight. A drain that outlived the start of the next capture would
//      have its inner spans (projectBins' own GPU/CPU children, and
//      projectSamples' upload/dispatch/finish) reparented under whatever that
//      capture happened to have open -- they are opened after awaits, so
//      forcing just the two outer spans to be roots would not have covered it.
//      Nothing corrupts (spanEnd splices by identity), but a flamechart that
//      files display work under the pose pipeline is worse than no flamechart:
//      it is the exact measurement this deferral exists to move.
//
// Still per-camera, and so is the profiler's assumption: two cameras
// reconstructing at once would break the span stack regardless of this, since
// axesCapturing is per-camera too. Not a new problem and not one to solve here
// -- there is exactly one camera today (see main.ts's animate loop).
//
// Placement in animate() matters just as much as the guard: this must run
// BEFORE the auto-capture trigger. runAxesReconstruction sets axesCapturing
// synchronously, so on the one tick where a reconstruction has just finished
// and the next is about to start, whichever runs first wins the tick -- and if
// the capture wins every time (interval shorter than a reconstruction) the
// drain never runs at all.
export function drainVisuals(camera: Camera): void {
  if (!camera.visualsDirty || camera.visualsDraining || camera.axesCapturing) return;
  camera.visualsDirty = false;
  camera.visualsDraining = true;
  runVisualTail(camera)
    .catch((e) => console.error('[visuals] deferred refresh failed:', e))
    .finally(() => {
      camera.visualsDraining = false;
      // UNREACHABLE while the mutual exclusion above holds -- nothing can set
      // axesCapturing between this drain starting and finishing. Kept as the
      // invariant's own tripwire rather than deleted as dead code: if either
      // capture-side guard is ever relaxed, this is what stops a half-repainted
      // frame from being the LAST thing painted, by re-arming for a clean pass
      // over one settled state. Cheap enough that proving it unnecessary is not
      // worth as much as it costing nothing to be wrong about.
      if (camera.axesCapturing) camera.visualsDirty = true;
    });
}

// Assumes camera.lastAxesCaptureGray is already populated and
// camera.axesCapturing is already true -- callers (runAxesReconstruction,
// recomputeFromLastCapture) own the guard/busy-UI/RAF wrapper and the
// capture step itself.
// What the DISPLAY half needs handed back from this run, declared here because
// this is the only caller that has a drain to resolve it in and the only one
// that knows which overlays are on.
//
// 'decodeGrid' is unconditional: lastDecodeGrid/lastDecodeRotated/
// lastDecodeCorrectness feed the Projected-Cam view and the pose readout, and
// the drain always runs, so there is nothing to gate on.
//
// fx/fy are asked for only when something draws them. THIS FUNCTION IS THE
// OTHER HALF OF A CONTRACT: overlays/pipelineField.ts returns null when they
// were not requested and its consumers then draw nothing, so a toggle listed
// in one place and not the other fails as a blank overlay rather than as a
// silent recomputation. The list mirrors preview.ts's overlaysNeedGray, which
// exists for the same class of mistake one stage earlier -- if you add a
// toggle whose overlay reads pipelineField(), it belongs in BOTH.
//
// Every toggle here also has to invalidate: turning an overlay on cannot paint
// from a run that was never asked to produce its input, so the handlers call
// recomputeFromLastCapture rather than the overlay's own updater. That is a
// real cost change -- flipping one of these re-runs the pose stages instead of
// recomputing one field on the CPU -- and it is the trade the whole step is
// making: one pipeline, asked, instead of a second one, duplicated.
function displayIntermediates(camera: Camera): IntermediatesRequest {
  const s = camera.settings;
  const want: IntermediateName[] = ['decodeGrid'];
  if (s.showTrueContamination || s.showReconstructedContamination
    || s.showTopGradient
    || s.showGradientArrow || s.showLevelLineArrow) {
    want.push('fx', 'fy');
  }
  return wants(...want);
}

async function recomputeStages(camera: Camera) {
  const { gray, w, h } = camera.lastAxesCaptureGray!;

  // Every stage from the 2x2-gradient composite lines through
  // runPositionDecode now lives in pipeline/poseCompute.ts's
  // computePoseFromCapture -- a pure function operating on the same field
  // names (a real Camera structurally satisfies its PoseComputeState), so
  // it can also run standalone on the phone (see this session's
  // on-device-pose-recovery plan). Mutates camera.lastVoteComposites/
  // lastVotes/lastQuadricPair/lastGridPeriodPhase/lastRecoveredAxes/
  // lastDecodeGrid/lastDecodeRotated/lastDecodeCorrectness/lastPositionDecode
  // in place, exactly like this function's own inline stages used to.
  // computeProjectedBinsAuto/paintProjectedTexture (in
  // runVisualTail) are deliberately NOT part of that shared prefix --
  // confirmed not on the critical path to a pose (distance is already
  // finalized by gridPeriodPhase before that stage would run); they exist
  // only to feed Projected-Cam/World-floor-decal DISPLAY.
  camera.lastPoseTiming = await computePoseFromCapture(
    camera, gray, w, h, backendFromForceCPU(globalState.forceCPU), displayIntermediates(camera),
  );
  camera.axesComputed = !!camera.lastQuadricPair;

  // The pose is final here; everything past this point is display. Deferred,
  // that display work stops being awaited inside the reconstruction's own
  // window, where it serialized ~20ms of GPU work against the same device
  // queue the pose stages were using.
  markVisualsDirty(camera);
}

export function runAxesReconstruction(camera: Camera) {
  if (camera.axesCapturing) return; // don't stack overlapping captures
  if (camera.visualsDraining) return; // ...or start one underneath a repaint, see drainVisuals
  camera.axesCapturing = true;
  const isActive = camera === activeCamera();
  const prevLabel = captureAxesBtn.textContent;
  if (isActive) {
    captureAxesBtn.disabled = true;
    captureAxesBtn.textContent = '⏳ computing...';
    axesReadout.textContent = 'computing...';
  }
  requestAnimationFrame(async () => {
    let rootSpan: ProfileSpan | null = null;
    try {
      rootSpan = spanStart('axesReconstruction');
      if (isPhysical(camera) && !camera.lastRealCaptureGray) {
        if (isActive) axesReadout.textContent = 'waiting for a real capture -- take a photo on the phone page';
        return;
      }
      const captureSpan = spanStart('capture+preprocess');
      // rawGray is top-down now, from either source (pipeline/capture.ts's
      // captureDistortedGrayscale/ingestRealCapture both flip once at their
      // own source, see their own comments) -- top-down is this whole
      // pipeline's one dominant convention, so it feeds lastAxesCaptureGray
      // (-> computePoseFromCapture) directly, no flip needed. The ONE
      // remaining flip in this whole pipeline lives right here instead:
      // lastNoisedPreviewGray (and everything painted from it --
      // distortedPreviewData and the other flipY=false preview/overlay
      // textures in camera/factory.ts) is the one consumer that genuinely
      // needs bottom-up, so it gets flipped on the way OUT of the dominant
      // top-down convention, not the math on the way in.
      const { gray: rawGray, w, h } = isPhysical(camera)
        ? { gray: camera.lastRealCaptureGray!, w: camera.lastRealCaptureW, h: camera.lastRealCaptureH }
        : captureDistortedGrayscale(camera);
      camera.lastNoisedPreviewGray = flipRowsF64(rawGray, w, h);
      spanEnd(captureSpan);
      camera.lastAxesCaptureGray = { gray: rawGray, w, h };

      await recomputeStages(camera);
    } finally {
      spanEnd(rootSpan);
      if (isActive) {
        captureAxesBtn.disabled = false;
        captureAxesBtn.textContent = prevLabel;
      }
      camera.axesCapturing = false;
      // See PhysicalCamera.idleSpan's own comment -- this brackets exactly
      // the round trip ingestRealCapture is on the other end of closing.
      if (isPhysical(camera)) camera.idleSpan = spanStart('idle (waiting for next frame)');
    }
  });
}

// Re-runs everything downstream of the capture stage (gradient field
// through pole markers/overlays), reusing whatever runAxesReconstruction
// last stored into camera.lastAxesCaptureGray instead of taking a fresh
// capture -- the recompute half of "every setting recomputes everything
// downstream of it" for settings that don't invalidate the capture itself
// (LSD tuning, gridPeriodPhaseGapLowerBound, minGrazingCos, forceCPU). Guarded by the same axesCapturing
// flag as runAxesReconstruction (reused, not a separate flag) so a rapid
// slider drag naturally self-throttles: a call that arrives while one is
// already in flight just no-ops, same as an overlapping "capture now"
// click today.
export function recomputeFromLastCapture(camera: Camera) {
  if (camera.axesCapturing) return;
  if (camera.visualsDraining) return; // see drainVisuals -- same self-throttle, one stage later
  if (!camera.lastAxesCaptureGray) return; // nothing captured yet
  camera.axesCapturing = true;
  const isActive = camera === activeCamera();
  const prevLabel = captureAxesBtn.textContent;
  if (isActive) {
    captureAxesBtn.disabled = true;
    captureAxesBtn.textContent = '⏳ computing...';
    axesReadout.textContent = 'computing...';
  }
  requestAnimationFrame(async () => {
    let rootSpan: ProfileSpan | null = null;
    try {
      rootSpan = spanStart('axesReconstruction (recompute)');
      await recomputeStages(camera);
    } finally {
      spanEnd(rootSpan);
      if (isActive) {
        captureAxesBtn.disabled = false;
        captureAxesBtn.textContent = prevLabel;
      }
      camera.axesCapturing = false;
    }
  });
}
