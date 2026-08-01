import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { activeCamera, isPhysical, isSimulated } from '../camera/store.ts';
import { COL_DIR, ROW_DIR, SPHERE_RADIUS } from '../constants.ts';
import { angleBetweenDegV } from '../math/geometry.ts';
import { updatePositionReadoutText } from '../overlays/projectedCamOverlays.ts';
import { applyRecoveredFloorOverlay, updateRecoveredCamGizmo, updateRecoveredFloorOutline } from '../overlays/recoveredOverlays.ts';
import { updateGradientCirclesDebug } from '../overlays/sphereOverlays.ts';
import { globalState } from '../state.ts';
import { axesReadout, captureAxesBtn } from '../ui/dom.ts';
import { captureDistortedGrayscale } from './capture.ts';
import { computeProjectedBinsAndMarginalsAuto, paintProjectedTexture, ProjectedSampleResult } from './decodeGrid.ts';
import { flipRowsF64 } from './distortion.ts';
import { refreshModeVisualizations } from './modeRefresh.ts';
import { computePoseFromCapture } from './poseCompute.ts';
import { ProfileSpan, spanEnd, spanStart } from '../profiling/profiler.ts';

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

// Assumes camera.lastAxesCaptureGray is already populated and
// camera.axesCapturing is already true -- callers (runAxesReconstruction,
// recomputeFromLastCapture) own the guard/busy-UI/RAF wrapper and the
// capture step itself.
async function recomputeStages(camera: Camera, isActive: boolean) {
  const { gray, w, h } = camera.lastAxesCaptureGray!;

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

  // Every stage from the 2x2-gradient composite lines through
  // runPositionDecode now lives in pipeline/poseCompute.ts's
  // computePoseFromCapture -- a pure function operating on the same field
  // names (a real Camera structurally satisfies its PoseComputeState), so
  // it can also run standalone on the phone (see this session's
  // on-device-pose-recovery plan). Mutates camera.lastVoteComposites/
  // lastVotes/lastQuadricPair/lastGridPeriodPhase/lastRecoveredAxes/
  // lastDecodeGrid/lastDecodeRotated/lastDecodeCorrectness/lastPositionDecode
  // in place, exactly like this function's own inline stages used to.
  // computeProjectedBinsAndMarginalsAuto/paintProjectedTexture (below) are
  // deliberately NOT part of that shared prefix -- confirmed not on the
  // critical path to a pose (distance is already finalized by
  // gridPeriodPhase before that stage would run); they exist only to feed
  // Projected-Cam/World-floor-decal DISPLAY, so this keeps calling them
  // separately, right here, only for desktop display purposes.
  const timing = await computePoseFromCapture(camera, gray, w, h);
  camera.axesComputed = !!camera.lastQuadricPair;
  updateGradientCirclesDebug(camera);

  const projectSpan = spanStart('projectBins (display + decode-marginals bins)');
  const projectStart = performance.now();
  // Captured outside the `if` (stays null when there's no recovered axes to
  // project) so it can be handed to refreshModeVisualizations below instead
  // of that call recomputing the exact same (possibly GPU) result a second
  // time -- see modeRefresh.ts's own comment on precomputedProjection.
  let projResult: ProjectedSampleResult = null;
  if (camera.lastRecoveredAxes) {
    projResult = await computeProjectedBinsAndMarginalsAuto(camera);
    if (showProjected) paintProjectedTexture(camera, projResult);
  }
  const projectMs = performance.now() - projectStart;
  spanEnd(projectSpan);

  const overlaySpan = spanStart('poleMarkers+overlays');
  const irlsSuffix = timing.worldVoteIterations !== null ? `  irls ${timing.worldVoteIterations} iter` : '';
  const timingLine = `votes ${timing.votesMs.toFixed(0)}ms  fit ${timing.fitMs.toFixed(0)}ms  pose ${timing.poseMs.toFixed(0)}ms  distance ${timing.distanceMs.toFixed(0)}ms  project ${projectMs.toFixed(0)}ms  decode ${timing.decodeMs.toFixed(0)}ms${irlsSuffix}`;
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

export function runAxesReconstruction(camera: Camera) {
  if (camera.axesCapturing) return; // don't stack overlapping captures
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

      await recomputeStages(camera, isActive);
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
// (LSD/join-walk tuning, weightSharpenPower, gridPeriodPhaseGapLowerBound,
// minGrazingCos, the useGPU* toggles). Guarded by the same axesCapturing
// flag as runAxesReconstruction (reused, not a separate flag) so a rapid
// slider drag naturally self-throttles: a call that arrives while one is
// already in flight just no-ops, same as an overlapping "capture now"
// click today.
export function recomputeFromLastCapture(camera: Camera) {
  if (camera.axesCapturing) return;
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
      await recomputeStages(camera, isActive);
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
