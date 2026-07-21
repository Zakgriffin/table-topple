import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { activeCamera, isPhysical, isSimulated } from '../camera/store.ts';
import { COL_DIR, GRID_STEP, MATH_QUAT, ROW_DIR, SPHERE_RADIUS } from '../constants.ts';
import { angleBetweenDegV, cornerDir } from '../math/geometry.ts';
import { updateContaminationOverlays } from '../overlays/contaminationOverlays.ts';
import { updatePositionReadoutText } from '../overlays/projectedCamOverlays.ts';
import { applyRecoveredFloorOverlay, updateRecoveredCamGizmo } from '../overlays/recoveredOverlays.ts';
import { updateGradientCirclesDebug } from '../overlays/sphereOverlays.ts';
import { C, R, torus } from '../scene/floor.ts';
import { globalState } from '../state.ts';
import { PositionFit } from '../types.ts';
import { axesReadout, captureAxesBtn } from '../ui/dom.ts';
import { captureDistortedGrayscale, getAnalysisVFovRad } from './capture.ts';
import { computeProjectedBinsAndMarginals, measurePeriodDistance, paintProjectedTexture, runPositionDecode, solveRecoveredCamQuat } from './decodeGrid.ts';
import { flipRowsF64 } from './distortion.ts';
import { refineOrientationLM } from './orientationLM.ts';
import { computePhotometricSamples, refineOrientationAndPositionLM } from './positionLM.ts';
import { computeWorldVotes, fitPairOfPlanes, votesInMagnitudeBand } from './votes.ts';
import { computeWorldVotesGPU } from '../pipelineGPU/voteGeneration.ts';
import { fitPairOfPlanesGPU } from '../pipelineGPU/fitPlanes.ts';
import { refineOrientationAndPositionLMGPU } from '../pipelineGPU/positionLM.ts';
import { ProfileSpan, spanEnd, spanStart } from '../profiling/profiler.ts';

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
      const votesSpan = spanStart(globalState.useGPUVotes ? 'votes (GPU)' : 'votes (CPU)');
      const votes = globalState.useGPUVotes
        ? (await computeWorldVotesGPU(camera.settings, gray, w, h, camera.settings.simGradRadius, camera.settings.coherenceRadius, MATH_QUAT, vFovRad, camera.aspect))
          ?? computeWorldVotes(camera.settings, gray, w, h, camera.settings.simGradRadius, camera.settings.coherenceRadius, MATH_QUAT, vFovRad, camera.aspect)
        : computeWorldVotes(camera.settings, gray, w, h, camera.settings.simGradRadius, camera.settings.coherenceRadius, MATH_QUAT, vFovRad, camera.aspect);
      spanEnd(votesSpan);
      camera.lastVotes = votes;
      updateGradientCirclesDebug(camera);
      const t1 = performance.now();

      const fitSpan = spanStart('fit (band-select + fitPairOfPlanes)');
      const fitVotes = votesInMagnitudeBand(votes, camera.settings.circleSamplePercentMin, camera.settings.circleSamplePercentMax);
      // Same fallback pattern as the other GPU sub-pipelines: fitPairOfPlanes
      // stays the source of truth, the GPU version is verified against it.
      const quadricPair = globalState.useGPUFit
        ? (await fitPairOfPlanesGPU(fitVotes, camera.settings.weightSharpenPower))
          ?? fitPairOfPlanes(fitVotes, camera.settings.weightSharpenPower)
        : fitPairOfPlanes(fitVotes, camera.settings.weightSharpenPower);
      spanEnd(fitSpan);
      const t2 = performance.now();

      const lmSpan = spanStart('orientationLM (Phase 1)');
      const refinedFit = quadricPair && camera.settings.orientationLM ? refineOrientationLM(fitVotes, quadricPair) : null;
      const orientationFit = refinedFit ?? quadricPair;
      spanEnd(lmSpan);
      const t2b = performance.now();

      camera.axesComputed = !!quadricPair;

      const poseAssemblySpan = spanStart('poseAssembly+roughSpacing');
      let rowDirRecovered: THREE.Vector3 | null = null, colDirRecovered: THREE.Vector3 | null = null;
      if (orientationFit) {
        const normalForHandedness = orientationFit.Dnormal.clone();
        if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalForHandedness) > 0) normalForHandedness.negate();
        rowDirRecovered = orientationFit.Drow.clone();
        colDirRecovered = orientationFit.Dcol.clone();
        const handedness = rowDirRecovered.clone().cross(colDirRecovered).dot(normalForHandedness);
        if (handedness > 0) colDirRecovered.negate();
      }

      const PLACEHOLDER_DISTANCE = 1;
      camera.lastRecoveredAxes = rowDirRecovered && colDirRecovered && orientationFit
        ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: orientationFit.Dnormal, distance: PLACEHOLDER_DISTANCE }
        : null;
      if (camera.lastRecoveredAxes) {
        const projResult = computeProjectedBinsAndMarginals(camera);
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
      let finalSpacing: { distanceU: number; distanceV: number } | null = null;
      if (camera.lastRecoveredAxes && spacing) {
        camera.lastRecoveredAxes.distance = (spacing.distanceU + spacing.distanceV) / 2;

        const roughBins = camera.lastProjectedBins;
        if (roughBins) {
          const rescale = camera.lastRecoveredAxes.distance / PLACEHOLDER_DISTANCE;
          const trueExtentU = (roughBins.maxU - roughBins.minU) * rescale;
          const trueExtentV = (roughBins.maxV - roughBins.minV) * rescale;
          const roughDistance = camera.lastRecoveredAxes.distance;
          const measured = measurePeriodDistance(camera, roughDistance, trueExtentU, trueExtentV);
          if (measured) {
            refinedSpacing = measured;
            camera.lastRecoveredAxes.distance = (measured.distanceU + measured.distanceV) / 2;
          }
        }

        const projResult2 = computeProjectedBinsAndMarginals(camera);
        if (showProjected) paintProjectedTexture(camera, projResult2);
      } else {
        camera.lastRecoveredAxes = null;
      }
      spanEnd(refineSpan);
      const t3b = performance.now();
      const decodeSpan = spanStart('positionDecode');
      runPositionDecode(camera, gray, w, h, vFovRad);
      spanEnd(decodeSpan);

      let lastPositionLMResult: (PositionFit & { iterations: number; initialCost: number; finalCost: number }) | null = null;
      if (camera.settings.positionLM && camera.lastRecoveredAxes && camera.lastPositionDecode && camera.lastNoisedPreviewGray) {
        const p3Span = spanStart('positionLM (Phase 3)');
        const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
        const normalForInit = Dnormal.clone();
        if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalForInit) > 0) normalForInit.negate();
        const normalForInitWorld = normalForInit.clone().applyQuaternion(camera.lastPositionDecode.recoveredCamQuat);
        const initialWorldX0 = camera.lastPositionDecode.camPos.x + normalForInitWorld.x * -distance;
        const initialWorldZ0 = camera.lastPositionDecode.camPos.z + normalForInitWorld.z * -distance;
        const photoSampleSpan = spanStart('computePhotometricSamples');
        const photoSamples = computePhotometricSamples(camera.lastNoisedPreviewGray, w, h, 4);
        spanEnd(photoSampleSpan);
        // Same fallback pattern as the GPU vote path above: computeWorldVotes/
        // refineOrientationAndPositionLM stay the source of truth, the GPU
        // version is verified against them, not the other way around.
        lastPositionLMResult = globalState.useGPUPositionLM
          ? (await refineOrientationAndPositionLMGPU(
              photoSamples, w, h, { Drow, Dcol, Dnormal }, distance, initialWorldX0, initialWorldZ0, MATH_QUAT, vFovRad, camera.aspect, torus, R, C,
            ))
            ?? refineOrientationAndPositionLM(photoSamples, w, h, { Drow, Dcol, Dnormal }, distance, initialWorldX0, initialWorldZ0, MATH_QUAT, vFovRad, camera.aspect)
          : refineOrientationAndPositionLM(photoSamples, w, h, { Drow, Dcol, Dnormal }, distance, initialWorldX0, initialWorldZ0, MATH_QUAT, vFovRad, camera.aspect);
        camera.lastRecoveredAxes.Drow = lastPositionLMResult.Drow;
        camera.lastRecoveredAxes.Dcol = lastPositionLMResult.Dcol;
        camera.lastRecoveredAxes.Dnormal = lastPositionLMResult.Dnormal;
        const refinedNormal = lastPositionLMResult.Dnormal.clone();
        if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(refinedNormal) > 0) refinedNormal.negate();
        if (camera.lastDecodeRotated) {
          const anchorRow = ((camera.lastPositionDecode.row - camera.lastDecodeRotated.zeroI) % R + R) % R;
          const anchorCol = ((camera.lastPositionDecode.col - camera.lastDecodeRotated.zeroJ) % C + C) % C;
          const refinedQuat = solveRecoveredCamQuat(
            camera.lastDecodeRotated, anchorRow, anchorCol, camera.lastRecoveredAxes.Drow, camera.lastRecoveredAxes.Dcol, refinedNormal, distance,
          );
          if (refinedQuat) camera.lastPositionDecode.recoveredCamQuat = refinedQuat;
        }
        const refinedNormalWorld = refinedNormal.clone().applyQuaternion(camera.lastPositionDecode.recoveredCamQuat);
        camera.lastPositionDecode.camPos.x = lastPositionLMResult.worldX0 + refinedNormalWorld.x * distance;
        camera.lastPositionDecode.camPos.z = lastPositionLMResult.worldZ0 + refinedNormalWorld.z * distance;
        const projResult3 = computeProjectedBinsAndMarginals(camera);
        if (showProjected) paintProjectedTexture(camera, projResult3);

        const postPhase3Bins = camera.lastProjectedBins;
        if (postPhase3Bins) {
          const extentU = postPhase3Bins.maxU - postPhase3Bins.minU;
          const extentV = postPhase3Bins.maxV - postPhase3Bins.minV;
          const currentDistance = camera.lastRecoveredAxes.distance;
          const measured = measurePeriodDistance(camera, currentDistance, extentU, extentV);
          if (measured) {
            finalSpacing = measured;
            camera.lastRecoveredAxes.distance = (measured.distanceU + measured.distanceV) / 2;
            camera.lastPositionDecode.camPos.x = lastPositionLMResult.worldX0 + refinedNormalWorld.x * camera.lastRecoveredAxes.distance;
            camera.lastPositionDecode.camPos.z = lastPositionLMResult.worldZ0 + refinedNormalWorld.z * camera.lastRecoveredAxes.distance;
            const projResult4 = computeProjectedBinsAndMarginals(camera);
            if (showProjected) paintProjectedTexture(camera, projResult4);
          }
        }
        spanEnd(p3Span);
      }
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
        if (refinedFit) {
          lines.push(`LM: ${refinedFit.iterations} iters, cost ${refinedFit.initialCost.toExponential(2)} -> ${refinedFit.finalCost.toExponential(2)}`);
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
          if (finalSpacing) {
            const fDistU = finalSpacing.distanceU, fDistV = finalSpacing.distanceV;
            if (haveGroundTruth) {
              const fErrU = (Math.abs(fDistU - trueDist) / trueDist) * 100;
              const fErrV = (Math.abs(fDistV - trueDist) / trueDist) * 100;
              lines.push(`dist U ${fDistU.toFixed(2)} (${fErrU.toFixed(1)}% err)  dist V ${fDistV.toFixed(2)} (${fErrV.toFixed(1)}% err)  [final, post-Phase-3 orientation]`);
            } else {
              lines.push(`dist U ${fDistU.toFixed(2)}  dist V ${fDistV.toFixed(2)}  [final, post-Phase-3 orientation]`);
            }
          }
        } else if (quadricPair) {
          lines.push(`spacing: no period found`);
        }
        if (camera.lastPositionDecode) {
          lines.push(`decoded torus (row,col): (${camera.lastPositionDecode.row}, ${camera.lastPositionDecode.col})  consistency ${(camera.lastPositionDecode.consistency * 100).toFixed(1)}%  camPos (${camera.lastPositionDecode.camPos.x.toFixed(2)}, ${camera.lastPositionDecode.camPos.y.toFixed(2)}, ${camera.lastPositionDecode.camPos.z.toFixed(2)})`);
        }
        if (lastPositionLMResult) {
          lines.push(`photoLM: ${lastPositionLMResult.iterations} iters, cost ${lastPositionLMResult.initialCost.toExponential(2)} -> ${lastPositionLMResult.finalCost.toExponential(2)}`);
          if (camera.lastRecoveredAxes && haveGroundTruth) {
            const { Drow: finalDrow, Dcol: finalDcol } = camera.lastRecoveredAxes;
            const errUnswapped = angleBetweenDegV(finalDrow, ROW_DIR) + angleBetweenDegV(finalDcol, COL_DIR);
            const errSwapped = angleBetweenDegV(finalDrow, COL_DIR) + angleBetweenDegV(finalDcol, ROW_DIR);
            const finalRowErr = errSwapped < errUnswapped ? angleBetweenDegV(finalDrow, COL_DIR) : angleBetweenDegV(finalDrow, ROW_DIR);
            const finalColErr = errSwapped < errUnswapped ? angleBetweenDegV(finalDcol, ROW_DIR) : angleBetweenDegV(finalDcol, COL_DIR);
            lines.push(`row err ${finalRowErr.toFixed(2)}°  col err ${finalColErr.toFixed(2)}°  [Phase 3, final${errSwapped < errUnswapped ? ', swapped' : ''}]`);
          }
        }
        lines.push(`votes ${(t1 - t0).toFixed(0)}ms  fit ${(t2 - t1).toFixed(0)}ms  LM ${(t2b - t2).toFixed(0)}ms  spacing ${(t3 - t2b).toFixed(0)}ms  refine ${(t3b - t3).toFixed(0)}ms  decode ${(t4 - t3b).toFixed(0)}ms`);
        axesReadout.textContent = lines.join('\n');
        updatePositionReadoutText(camera);
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

