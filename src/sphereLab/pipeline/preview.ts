import { Camera } from '../camera/model.ts';
import { isPhysical } from '../camera/store.ts';
import { toGrayscale } from '../../decode.ts';
import { renderer } from '../scene/renderer.ts';
import { addGaussianNoise, applyAntialiasFilter, downsampleBoxAverage, flipRowsF64, separableBoxBlur } from './distortion.ts';
import { computeGradient2x2Field, fillGrayscalePreview, paintVectorFieldAsColor } from './gradientField.ts';

// Shared tail for both capture sources: given a final analysis-resolution
// grayscale, paints whichever of the direction/scalar field views is
// currently selected (no-op for 'raw'/'antialiased'/'downsampled'/'noised',
// handled by each capture path's own display branch before this runs).
function paintFieldViewFromGray(camera: Camera, gray: Float64Array) {
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const settings = camera.settings;
  if (settings.fieldView === 'gradient2x2' || settings.fieldView === 'gradient2x2Directed') {
    const field = computeGradient2x2Field(gray, w, h);
    // Axial vs directed hue is purely a PAINTING choice -- same field either
    // way. 'gradient2x2' folds theta into [0, PI) so a black-to-white edge
    // and the white-to-black edge facing it share one hue;
    // 'gradient2x2Directed' does not, so those two land exactly opposite on
    // the hue wheel. See paintVectorFieldAsColor's own comment for why that
    // distinction is worth being able to SEE -- pipeline/lsdSegments.ts's
    // segment growing is directed now, so two edges that look identical in
    // the axial view are genuinely different lines to it.
    paintVectorFieldAsColor(field, camera.distortedPreviewData, settings.fieldView === 'gradient2x2Directed');
    camera.distortedPreviewTex.needsUpdate = true;
  }
}

// Every overlay that reads camera.lastNoisedPreviewGray directly, rather than
// the painted field-view colors updateDistortedPreview's own hideField branch
// gates. If an overlay is on but is NOT listed here, updateDistortedPreview's
// early-returns skip recomputing the gray entirely and the overlay silently
// reads a STALE buffer -- possibly one sized for a previous rtSize.w/h, which
// then gets indexed at the CURRENT dimensions, reading/writing at
// systematically wrong offsets (the diagonal "streaking" artifact). It reads
// as "the overlay is broken", not "the gray is stale", so keep this in sync:
// ANY new toggle whose overlay touches lastNoisedPreviewGray belongs here.
//
// This used to list only four flags, which was survivable only because every
// one of these overlays was ALSO hard-gated to the gradient2x2 field view (the
// view that always recomputes the gray anyway). Now that they render over any
// view, the omissions became reachable -- see updateLsdOverlay's own comment.
function overlaysNeedGray(settings: Camera['settings']): boolean {
  return settings.showTrueContamination || settings.showReconstructedContamination
    || settings.showTopGradient
    || settings.showLsdSegments || settings.showLsdRejected || settings.showLsdRawRegions
    || settings.showGradientArrow || settings.showLevelLineArrow;
}

export function updateDistortedPreview(camera: Camera) {
  const settings = camera.settings;
  if (settings.hideField) {
    for (let i = 0; i < camera.distortedPreviewData.length; i += 4) {
      camera.distortedPreviewData[i] = 0; camera.distortedPreviewData[i + 1] = 0; camera.distortedPreviewData[i + 2] = 0; camera.distortedPreviewData[i + 3] = 255;
    }
    camera.distortedPreviewTex.needsUpdate = true;
  }
  const needGrayForOverlay = overlaysNeedGray(settings);
  if (settings.hideField && !needGrayForOverlay) return;

  if (isPhysical(camera)) {
    if (!camera.lastRealCaptureGray) {
      // No real image ever reached the desktop for this cycle -- e.g. this
      // camera is in device-compute mode (see this session's
      // on-device-pose-recovery plan and pipeline/capture.ts's
      // ingestRemotePose, which never populates lastRealCaptureGray) --
      // blank the buffer rather than leaving whatever pixels a PRIOR
      // desktop-compute capture on this same camera last painted here
      // visible after a mode switch. There's no persistent "visible" flag
      // to toggle for this raw quad-blit view (see scene/quadRenderers.ts's
      // renderPreviewViewport, called fresh every frame from main.ts), so
      // blanking the source data is the equivalent fix.
      camera.distortedPreviewData.fill(0);
      camera.distortedPreviewTex.needsUpdate = true;
      // lastNoisedPreviewGray must go stale-free the SAME cycle -- every
      // reader (gradientHighlightOverlays.ts, contaminationOverlays.ts,
      // lsdOverlay.ts, decodeGrid.ts's projectSamplesCPU,
      // pipelineGPU/projectSamples.ts) already null-guards it, but leaving
      // a PRIOR real capture's gray buffer sitting here would let any of
      // them read it against the CURRENT camera.rtSize/aspect -- which can
      // legitimately differ now (a device-compute phone's own resolution
      // slider moved, or sendCapturedImage got toggled off, between one
      // real capture and the next) -- a resolution mismatch that silently
      // reads garbage/out-of-bounds instead of erroring. See this
      // session's chat: an explicit ask to guarantee capture resolution
      // matches every downstream computation exactly.
      camera.lastNoisedPreviewGray = null;
      return;
    }
    // camera.lastRealCaptureGray is top-down now (this pipeline's one
    // dominant convention, see pipeline/capture.ts's own comments) -- flip
    // once here, the same reason runAxesReconstruction's own single flip
    // exists: distortedPreviewData (flipY=false) and every other reader of
    // lastNoisedPreviewGray (contaminationOverlays.ts, lsdOverlay.ts, etc.)
    // still expect bottom-up.
    const bottomUp = flipRowsF64(camera.lastRealCaptureGray, camera.lastRealCaptureW, camera.lastRealCaptureH);
    if (!settings.hideField) {
      if (settings.fieldView === 'raw' || settings.fieldView === 'antialiased' || settings.fieldView === 'downsampled' || settings.fieldView === 'noised') {
        fillGrayscalePreview(bottomUp, camera.distortedPreviewData);
        camera.distortedPreviewTex.needsUpdate = true;
      } else {
        paintFieldViewFromGray(camera, bottomUp);
      }
    }
    camera.lastNoisedPreviewGray = bottomUp;
    return;
  }

  // camera narrowed to SimulatedCamera by the isPhysical() early-return
  // above, but `settings` was captured before that -- re-derive it so
  // TypeScript (and the simNoise/simBlur/captureSupersample accesses below)
  // see the narrower SimulatedCameraSettings type.
  const simSettings = camera.settings;
  const { w: cw, h: ch } = camera.captureRTSize;
  const rawRGBA = new Uint8Array(cw * ch * 4);
  renderer.readRenderTargetPixels(camera.camRT, 0, 0, cw, ch, rawRGBA);
  const hiResGray = toGrayscale(rawRGBA, cw, ch);

  if (!settings.hideField && settings.fieldView === 'raw') {
    const raw = downsampleBoxAverage(hiResGray, cw, ch, simSettings.captureSupersample, camera.rtSize.w, camera.rtSize.h);
    fillGrayscalePreview(raw, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
  }

  const antialiased = applyAntialiasFilter(hiResGray, cw, ch, simSettings.captureSupersample);

  if (!settings.hideField && settings.fieldView === 'antialiased') {
    const aaDisplay = downsampleBoxAverage(antialiased, cw, ch, simSettings.captureSupersample, camera.rtSize.w, camera.rtSize.h);
    fillGrayscalePreview(aaDisplay, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
  }

  const hiResBlurred = separableBoxBlur(antialiased, cw, ch, Math.round(simSettings.simBlur * simSettings.captureSupersample));
  const downsampled = downsampleBoxAverage(hiResBlurred, cw, ch, simSettings.captureSupersample, camera.rtSize.w, camera.rtSize.h);

  if (!settings.hideField && settings.fieldView === 'downsampled') {
    fillGrayscalePreview(downsampled, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
  }

  const noised = downsampled;
  addGaussianNoise(noised, simSettings.simNoise);
  if (!settings.hideField) {
    if (settings.fieldView === 'noised') {
      fillGrayscalePreview(noised, camera.distortedPreviewData);
      camera.distortedPreviewTex.needsUpdate = true;
    } else {
      paintFieldViewFromGray(camera, noised);
    }
  }
  camera.lastNoisedPreviewGray = noised;
}

export const PREVIEW_UPDATE_INTERVAL_MS = 100; // ~10fps
