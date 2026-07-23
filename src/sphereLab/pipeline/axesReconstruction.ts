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
import { computeProjectedBinsAndMarginals, computeProjectedBinsAndMarginalsGPU, measurePeriodDistance, measurePeriodDistanceGPU, paintProjectedTexture, runPositionDecode } from './decodeGrid.ts';
import { flipRowsF64 } from './distortion.ts';
import { computeGridPeriodPhase } from './gridPeriodPhase.ts';
import { computeSegmentVotes, computeWorldVotes, fitPairOfPlanes, votesInMagnitudeBand } from './votes.ts';
import { computeWorldVotesGPU } from '../pipelineGPU/voteGeneration.ts';
import { fitPairOfPlanesGPU } from '../pipelineGPU/fitPlanes.ts';
import { votesInMagnitudeBandGPU } from '../pipelineGPU/voteBandSelect.ts';
import { ProfileSpan, spanEnd, spanStart } from '../profiling/profiler.ts';

// Falls back to CPU per-call if the GPU one returns null (WebGPU
// unavailable) -- same pattern as every other GPU sub-pipeline in this
// file, just factored into one helper since computeProjectedBinsAndMarginals
// and measurePeriodDistance each have their own GPU/CPU pair called from
// multiple sites below.
async function projectBins(camera: Camera) {
  const s = spanStart(globalState.useGPUProject ? 'projectBins (GPU stage 1 + CPU bucket)' : 'projectBins (CPU)');
  const result = globalState.useGPUProject
    ? (await computeProjectedBinsAndMarginalsGPU(camera)) ?? computeProjectedBinsAndMarginals(camera)
    : computeProjectedBinsAndMarginals(camera);
  spanEnd(s);
  return result;
}
async function measureSpacing(camera: Camera, currentDistance: number, extentU: number, extentV: number) {
  const s = spanStart(globalState.useGPUProject ? 'measureSpacing (GPU stage 1 + CPU bucket)' : 'measureSpacing (CPU)');
  const result = globalState.useGPUProject
    ? (await measurePeriodDistanceGPU(camera, currentDistance, extentU, extentV)) ?? measurePeriodDistance(camera, currentDistance, extentU, extentV)
    : measurePeriodDistance(camera, currentDistance, extentU, extentV);
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
      // skipping at every one of the intermediate buildProjectedTexture
      // call sites below unless this camera's Projected-Cam view is what's
      // actually on screen right now. The numeric half (bins/marginals)
      // stays unconditional -- it feeds the spacing refinement that the
      // always-on World view's recovered-pose overlay depends on for every
      // camera, not just the displayed one. The RGBA half also has to run
      // whenever the World-view floor overlay is on, though -- that overlay
      // (see overlays/recoveredOverlays.ts) reuses projectedPreviewTex as
      // its decal map, so skipping the paint here left it sitting at its
      // all-zero (alpha 0, invisible) initial contents for any camera that
      // never happened to be viewed in Projected-Cam mode first.
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
      // Falls back to the CPU path if the GPU one returns null (WebGPU
      // unavailable, or the device request failed) -- see computeWorldVotesGPU's
      // own comment. computeWorldVotes stays the source of truth either way;
      // the GPU version is verified against it, not the other way around.
      //
      // useSegmentVotes is a separate, CPU-only, experimental swap of the
      // vote SOURCE itself (see computeSegmentVotes's own comment) -- one
      // vote per bucket-fill line segment instead of one per pixel. It
      // bypasses the GPU vote path entirely (this is a comparison/accuracy
      // experiment, not a perf-sensitive path) but still hands the same
      // Vote[] shape to everything below, so band-select/fit/LM/decode/
      // projected-cam all keep working completely unmodified.
      const votesSpan = spanStart(camera.settings.useSegmentVotes ? 'votes (segments)' : globalState.useGPUVotes ? 'votes (GPU)' : 'votes (CPU)');
      const votes = camera.settings.useSegmentVotes
        ? computeSegmentVotes(camera.settings, gray, w, h, camera.settings.simGradRadius, camera.settings.coherenceRadius, MATH_QUAT, vFovRad, camera.aspect)
        : globalState.useGPUVotes
          ? (await computeWorldVotesGPU(camera.settings, gray, w, h, camera.settings.simGradRadius, camera.settings.coherenceRadius, MATH_QUAT, vFovRad, camera.aspect))
            ?? computeWorldVotes(camera.settings, gray, w, h, camera.settings.simGradRadius, camera.settings.coherenceRadius, MATH_QUAT, vFovRad, camera.aspect)
          : computeWorldVotes(camera.settings, gray, w, h, camera.settings.simGradRadius, camera.settings.coherenceRadius, MATH_QUAT, vFovRad, camera.aspect);
      spanEnd(votesSpan);
      camera.lastVotes = votes;
      updateGradientCirclesDebug(camera);
      const t1 = performance.now();

      const fitSpan = spanStart('fit (band-select + fitPairOfPlanes)');
      const bandSpan = spanStart(globalState.useGPUFit ? 'votesInMagnitudeBand (GPU)' : 'votesInMagnitudeBand (CPU sort)');
      // Same fallback pattern as the other GPU sub-pipelines: votesInMagnitudeBand
      // stays the source of truth, the GPU version is verified against it.
      const fitVotes = globalState.useGPUFit
        ? (await votesInMagnitudeBandGPU(votes, camera.settings.circleSamplePercentMin, camera.settings.circleSamplePercentMax))
          ?? votesInMagnitudeBand(votes, camera.settings.circleSamplePercentMin, camera.settings.circleSamplePercentMax)
        : votesInMagnitudeBand(votes, camera.settings.circleSamplePercentMin, camera.settings.circleSamplePercentMax);
      spanEnd(bandSpan);
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

      const t2b = performance.now();

      camera.axesComputed = !!quadricPair;

      const poseAssemblySpan = spanStart('poseAssembly+roughSpacing');
      let rowDirRecovered: THREE.Vector3 | null = null, colDirRecovered: THREE.Vector3 | null = null;
      if (quadricPair) {
        const normalForHandedness = quadricPair.Dnormal.clone();
        if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalForHandedness) > 0) normalForHandedness.negate();
        rowDirRecovered = quadricPair.Drow.clone();
        colDirRecovered = quadricPair.Dcol.clone();
        const handedness = rowDirRecovered.clone().cross(colDirRecovered).dot(normalForHandedness);
        if (handedness > 0) colDirRecovered.negate();
      }

      const PLACEHOLDER_DISTANCE = 1;
      camera.lastRecoveredAxes = rowDirRecovered && colDirRecovered && quadricPair
        ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: quadricPair.Dnormal, distance: PLACEHOLDER_DISTANCE }
        : null;
      if (camera.lastRecoveredAxes) {
        const projResult = await projectBins(camera);
        if (showProjected) paintProjectedTexture(camera, projResult);
      }

      const marginals = camera.lastMarginals, bins = camera.lastProjectedBins;
      const spacing = camera.lastRecoveredAxes && marginals && bins && marginals.colPeriod !== null && marginals.rowPeriod !== null
        ? {
          distanceU: PLACEHOLDER_DISTANCE * (GRID_STEP / (marginals.colPeriod * bins.binWidthU)),
          distanceV: PLACEHOLDER_DISTANCE * (GRID_STEP / (marginals.rowPeriod * bins.binWidthV)),
        }
        : null;
      spanEnd(poseAssemblySpan);
      const t3 = performance.now();

      const refineSpan = spanStart('spacing refine (measurePeriodDistance + re-project)');
      let refinedSpacing: { distanceU: number; distanceV: number } | null = null;
      if (camera.lastRecoveredAxes && spacing) {
        camera.lastRecoveredAxes.distance = (spacing.distanceU + spacing.distanceV) / 2;

        const roughBins = camera.lastProjectedBins;
        if (roughBins) {
          const rescale = camera.lastRecoveredAxes.distance / PLACEHOLDER_DISTANCE;
          const trueExtentU = (roughBins.maxU - roughBins.minU) * rescale;
          const trueExtentV = (roughBins.maxV - roughBins.minV) * rescale;
          const roughDistance = camera.lastRecoveredAxes.distance;
          const measured = await measureSpacing(camera, roughDistance, trueExtentU, trueExtentV);
          if (measured) {
            refinedSpacing = measured;
            camera.lastRecoveredAxes.distance = (measured.distanceU + measured.distanceV) / 2;
          }
        }

        const projResult2 = await projectBins(camera);
        if (showProjected) paintProjectedTexture(camera, projResult2);
      } else {
        camera.lastRecoveredAxes = null;
      }
      spanEnd(refineSpan);
      const t3b = performance.now();
      const decodeSpan = spanStart('positionDecode');
      await runPositionDecode(camera, gray, w, h, vFovRad);
      spanEnd(decodeSpan);

      // Grid period/phase debug pipeline (pipeline/gridPeriodPhase.ts) --
      // opt-in (its own toggle, gated separately from the main vote/fit
      // path) since it recomputes the bucket-fill/join-walk/composite-line
      // steps a second time internally rather than threading identity
      // through the existing anonymous Vote[] used above.
      camera.lastGridPeriodPhase = camera.settings.showGridPeriodPhaseDebug && camera.lastRecoveredAxes
        ? computeGridPeriodPhase(
            camera.settings, gray, w, h, MATH_QUAT, vFovRad, camera.aspect,
            camera.lastRecoveredAxes.Drow, camera.lastRecoveredAxes.Dcol, camera.lastRecoveredAxes.Dnormal,
            GRID_STEP,
          )
        : null;

      const overlaySpan = spanStart('poleMarkers+overlays');
      if (camera.lastPositionDecode && rowDirRecovered && colDirRecovered) {
        const { recoveredCamQuat } = camera.lastPositionDecode;
        const rowDirWorld = rowDirRecovered.clone().applyQuaternion(recoveredCamQuat);
        const colDirWorld = colDirRecovered.clone().applyQuaternion(recoveredCamQuat);
        camera.recoveredRowPoleA.position.copy(rowDirWorld).multiplyScalar(SPHERE_RADIUS);
        camera.recoveredRowPoleB.position.copy(rowDirWorld).multiplyScalar(-SPHERE_RADIUS);
        camera.recoveredColPoleA.position.copy(colDirWorld).multiplyScalar(SPHERE_RADIUS);
        camera.recoveredColPoleB.position.copy(colDirWorld).multiplyScalar(-SPHERE_RADIUS);
      }
      updateRecoveredCamGizmo(camera);
      applyRecoveredFloorOverlay(camera);
      if (isActive && globalState.mode === 'through') updateContaminationOverlays(camera);
      spanEnd(overlaySpan);
      const t4 = performance.now();

      if (isActive) {
        const haveGroundTruth = isSimulated(camera);
        const lines = [`${votes.length} votes  (${fitVotes.length} fed to fit)`];
        if (rowDirRecovered && colDirRecovered) {
          if (haveGroundTruth) {
            const errUnswapped = angleBetweenDegV(rowDirRecovered, ROW_DIR) + angleBetweenDegV(colDirRecovered, COL_DIR);
            const errSwapped = angleBetweenDegV(rowDirRecovered, COL_DIR) + angleBetweenDegV(colDirRecovered, ROW_DIR);
            const rowErr = errSwapped < errUnswapped ? angleBetweenDegV(rowDirRecovered, COL_DIR) : angleBetweenDegV(rowDirRecovered, ROW_DIR);
            const colErr = errSwapped < errUnswapped ? angleBetweenDegV(colDirRecovered, ROW_DIR) : angleBetweenDegV(colDirRecovered, COL_DIR);
            lines.push(`row err ${rowErr.toFixed(2)}°  col err ${colErr.toFixed(2)}°  [Phase 1, vote-based${errSwapped < errUnswapped ? ', swapped' : ''}]`);
          }
        } else {
          lines.push(`degenerate fit`);
        }
        if (spacing) {
          const trueDist = isSimulated(camera) ? camera.camPos.y : NaN;
          const distU = spacing.distanceU, distV = spacing.distanceV;
          if (haveGroundTruth) {
            const errU = (Math.abs(distU - trueDist) / trueDist) * 100;
            const errV = (Math.abs(distV - trueDist) / trueDist) * 100;
            lines.push(`dist U ${distU.toFixed(2)} (${errU.toFixed(1)}% err)  dist V ${distV.toFixed(2)} (${errV.toFixed(1)}% err)  true ${trueDist.toFixed(2)}  [rough, rtSize buckets]`);
          } else {
            lines.push(`dist U ${distU.toFixed(2)}  dist V ${distV.toFixed(2)}  [rough, rtSize buckets]`);
          }
          if (refinedSpacing) {
            const rDistU = refinedSpacing.distanceU, rDistV = refinedSpacing.distanceV;
            if (haveGroundTruth) {
              const rErrU = (Math.abs(rDistU - trueDist) / trueDist) * 100;
              const rErrV = (Math.abs(rDistV - trueDist) / trueDist) * 100;
              lines.push(`dist U ${rDistU.toFixed(2)} (${rErrU.toFixed(1)}% err)  dist V ${rDistV.toFixed(2)} (${rErrV.toFixed(1)}% err)  [refined, adaptive buckets]`);
            } else {
              lines.push(`dist U ${rDistU.toFixed(2)}  dist V ${rDistV.toFixed(2)}  [refined, adaptive buckets]`);
            }
          }
        } else if (quadricPair) {
          lines.push(`spacing: no period found`);
        }
        if (camera.lastPositionDecode) {
          lines.push(`decoded torus (row,col): (${camera.lastPositionDecode.row}, ${camera.lastPositionDecode.col})  consistency ${(camera.lastPositionDecode.consistency * 100).toFixed(1)}%  camPos (${camera.lastPositionDecode.camPos.x.toFixed(2)}, ${camera.lastPositionDecode.camPos.y.toFixed(2)}, ${camera.lastPositionDecode.camPos.z.toFixed(2)})`);
        }
        lines.push(`votes ${(t1 - t0).toFixed(0)}ms  fit ${(t2 - t1).toFixed(0)}ms  spacing ${(t3 - t2b).toFixed(0)}ms  refine ${(t3b - t3).toFixed(0)}ms  decode ${(t4 - t3b).toFixed(0)}ms`);
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

