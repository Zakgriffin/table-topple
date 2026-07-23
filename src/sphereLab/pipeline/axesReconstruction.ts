import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { activeCamera, isPhysical, isSimulated } from '../camera/store.ts';
import { COL_DIR, GRID_STEP, MATH_QUAT, ROW_DIR, SPHERE_RADIUS } from '../constants.ts';
import { angleBetweenDegV, cornerDir } from '../math/geometry.ts';
import { updateContaminationOverlays } from '../overlays/contaminationOverlays.ts';
import { drawGridPeriodPhasePlot } from '../overlays/gridPeriodPhaseOverlays.ts';
import { updatePositionReadoutText } from '../overlays/projectedCamOverlays.ts';
import { applyRecoveredFloorOverlay, updateRecoveredCamGizmo } from '../overlays/recoveredOverlays.ts';
import { updateGradientCirclesDebug } from '../overlays/sphereOverlays.ts';
import { globalState } from '../state.ts';
import { axesReadout, captureAxesBtn } from '../ui/dom.ts';
import { captureDistortedGrayscale, getAnalysisVFovRad } from './capture.ts';
import { computeProjectedBinsAndMarginals, computeProjectedBinsAndMarginalsGPU, paintProjectedTexture, runPositionDecode } from './decodeGrid.ts';
import { flipRowsF64 } from './distortion.ts';
import { computeGridPeriodPhase } from './gridPeriodPhase.ts';
import { computeSegmentVotes, fitPairOfPlanes } from './votes.ts';
import { fitPairOfPlanesGPU } from '../pipelineGPU/fitPlanes.ts';
import { ProfileSpan, spanEnd, spanStart } from '../profiling/profiler.ts';

// Falls back to CPU per-call if the GPU one returns null (WebGPU
// unavailable) -- same pattern as every other GPU sub-pipeline in this file.
async function projectBins(camera: Camera) {
  const s = spanStart(globalState.useGPUProject ? 'projectBins (GPU stage 1 + CPU bucket)' : 'projectBins (CPU)');
  const result = globalState.useGPUProject
    ? (await computeProjectedBinsAndMarginalsGPU(camera)) ?? computeProjectedBinsAndMarginals(camera)
    : computeProjectedBinsAndMarginals(camera);
  spanEnd(s);
  return result;
}

// ── Axes/position reconstruction (the big orchestrator) ──────────────────

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
      const t0 = performance.now();
      rootSpan = spanStart('axesReconstruction');
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
      if (isPhysical(camera) && !camera.lastRealCaptureGray) {
        if (isActive) axesReadout.textContent = 'waiting for a real capture -- take a photo on the phone page';
        return;
      }
      const captureSpan = spanStart('capture+preprocess');
      const { gray: rawGray, w, h } = isPhysical(camera)
        ? { gray: camera.lastRealCaptureGray!, w: camera.lastRealCaptureW, h: camera.lastRealCaptureH }
        : captureDistortedGrayscale(camera);
      camera.lastNoisedPreviewGray = rawGray;
      const gray = flipRowsF64(rawGray, w, h);
      const vFovRad = getAnalysisVFovRad(camera);
      spanEnd(captureSpan);
      // computeSegmentVotes is always the vote source now (see this
      // session's chat) -- one vote per bucket-fill line segment instead
      // of one per pixel. computeWorldVotes/computeWorldVotesGPU (the old
      // per-pixel CPU/GPU path this used to fall back to) are left defined
      // in pipeline/votes.ts / pipelineGPU/voteGeneration.ts, unreferenced
      // here, in case that comparison is wanted again later.
      const votesSpan = spanStart('votes (segments)');
      const votes = computeSegmentVotes(camera.settings, gray, w, h, camera.settings.simGradRadius, camera.settings.coherenceRadius, MATH_QUAT, vFovRad, camera.aspect);
      spanEnd(votesSpan);
      camera.lastVotes = votes;
      updateGradientCirclesDebug(camera);
      const t1 = performance.now();

      const fitSpan = spanStart('fit (fitPairOfPlanes)');
      // No band-select step anymore -- fitPairOfPlanes runs on every vote
      // (see this session's chat: the old top-N%-by-magnitude cutoff,
      // circleSamplePercentMin/Max, is gone entirely, not just defaulted).
      // votesInMagnitudeBand/votesInMagnitudeBandGPU are left defined in
      // pipeline/votes.ts / pipelineGPU/voteBandSelect.ts, unreferenced
      // here, in case a cutoff is wanted again later.
      const fitVotes = votes;
      // Same fallback pattern as the other GPU sub-pipelines: fitPairOfPlanes
      // stays the source of truth, the GPU version is verified against it.
      const fitOnlySpan = spanStart(globalState.useGPUFit ? 'fitPairOfPlanes (GPU)' : 'fitPairOfPlanes (CPU)');
      const quadricPair = globalState.useGPUFit
        ? (await fitPairOfPlanesGPU(fitVotes, camera.settings.weightSharpenPower))
          ?? fitPairOfPlanes(fitVotes, camera.settings.weightSharpenPower)
        : fitPairOfPlanes(fitVotes, camera.settings.weightSharpenPower);
      spanEnd(fitOnlySpan);
      spanEnd(fitSpan);
      const t2 = performance.now();

      camera.axesComputed = !!quadricPair;

      const poseAssemblySpan = spanStart('poseAssembly');
      let rowDirRecovered: THREE.Vector3 | null = null, colDirRecovered: THREE.Vector3 | null = null;
      if (quadricPair) {
        const normalForHandedness = quadricPair.Dnormal.clone();
        if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalForHandedness) > 0) normalForHandedness.negate();
        rowDirRecovered = quadricPair.Drow.clone();
        colDirRecovered = quadricPair.Dcol.clone();
        const handedness = rowDirRecovered.clone().cross(colDirRecovered).dot(normalForHandedness);
        if (handedness > 0) colDirRecovered.negate();
      }
      spanEnd(poseAssemblySpan);
      const t3 = performance.now();

      // Grid period/phase (pipeline/gridPeriodPhase.ts) is now the SOLE
      // source of camera.lastRecoveredAxes.distance -- see this session's
      // chat for why the old marginals/autocorrelation-based spacing
      // estimate (computeProjectedBinsAndMarginals's colPeriod/rowPeriod,
      // further refined via measurePeriodDistance's own re-bucket-and-
      // remeasure pass) was disconnected: composite-line-derived period
      // and height come out of a single narrowly-bracketed search, need no
      // placeholder-then-rescale dance, and don't need a second refine
      // pass. computeProjectedBinsAndMarginals/measurePeriodDistance are
      // left defined in pipeline/decodeGrid.ts, unreferenced here, in case
      // this needs revisiting. This is no longer gated behind
      // showGridPeriodPhaseDebug -- that toggle now only controls whether
      // the debug PLOT/overlay draws, not whether this runs (real distance
      // depends on it either way).
      const gppSpan = spanStart('gridPeriodPhase (distance source)');
      const gpp = rowDirRecovered && colDirRecovered && quadricPair
        ? computeGridPeriodPhase(
            camera.settings, gray, w, h, MATH_QUAT, vFovRad, camera.aspect,
            rowDirRecovered, colDirRecovered, quadricPair.Dnormal, GRID_STEP,
          )
        : null;
      camera.lastGridPeriodPhase = gpp;
      spanEnd(gppSpan);
      const t4 = performance.now();

      camera.lastRecoveredAxes = rowDirRecovered && colDirRecovered && quadricPair && gpp
        ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: quadricPair.Dnormal, distance: gpp.height ?? 1 }
        : null;

      const projectSpan = spanStart('projectBins (display + decode-marginals bins)');
      if (camera.lastRecoveredAxes) {
        const projResult = await projectBins(camera);
        if (showProjected) paintProjectedTexture(camera, projResult);
      }
      spanEnd(projectSpan);
      const t5 = performance.now();

      const decodeSpan = spanStart('positionDecode');
      await runPositionDecode(camera, gray, w, h, vFovRad);
      spanEnd(decodeSpan);
      const t6 = performance.now();

      const overlaySpan = spanStart('poleMarkers+overlays');
      let orientationErrorLine: string | null = null;
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
      if (isActive && globalState.mode === 'through') updateContaminationOverlays(camera);
      spanEnd(overlaySpan);

      if (isActive) {
        const haveGroundTruth = isSimulated(camera);
        const lines = [`${votes.length} votes  (${fitVotes.length} fed to fit)`];
        if (rowDirRecovered && colDirRecovered) {
          if (orientationErrorLine) lines.push(orientationErrorLine);
        } else {
          lines.push(`degenerate fit`);
        }
        if (camera.lastRecoveredAxes && gpp) {
          const trueDist = isSimulated(camera) ? camera.camPos.y : NaN;
          const dist = camera.lastRecoveredAxes.distance;
          if (haveGroundTruth) {
            const err = (Math.abs(dist - trueDist) / trueDist) * 100;
            lines.push(`distance ${dist.toFixed(2)} (${err.toFixed(1)}% err)  true ${trueDist.toFixed(2)}  period ${gpp.period.toFixed(4)}  [gridPeriodPhase]`);
          } else {
            lines.push(`distance ${dist.toFixed(2)}  period ${gpp.period.toFixed(4)}  [gridPeriodPhase]`);
          }
        } else if (quadricPair) {
          lines.push(`distance: no period found (gridPeriodPhase)`);
        }
        if (camera.lastPositionDecode) {
          lines.push(`decoded torus (row,col): (${camera.lastPositionDecode.row}, ${camera.lastPositionDecode.col})  consistency ${(camera.lastPositionDecode.consistency * 100).toFixed(1)}%  camPos (${camera.lastPositionDecode.camPos.x.toFixed(2)}, ${camera.lastPositionDecode.camPos.y.toFixed(2)}, ${camera.lastPositionDecode.camPos.z.toFixed(2)})`);
        }
        lines.push(`votes ${(t1 - t0).toFixed(0)}ms  fit ${(t2 - t1).toFixed(0)}ms  pose ${(t3 - t2).toFixed(0)}ms  distance ${(t4 - t3).toFixed(0)}ms  project ${(t5 - t4).toFixed(0)}ms  decode ${(t6 - t5).toFixed(0)}ms`);
        axesReadout.textContent = lines.join('\n');
        updatePositionReadoutText(camera);
        drawGridPeriodPhasePlot(camera);
      }
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

