import { Camera } from '../camera/model.ts';
import { isPhysical } from '../camera/store.ts';
import { toGrayscale } from '../../decode.ts';
import { renderer } from '../scene/renderer.ts';
import { addGaussianNoise, applyAntialiasFilter, downsampleBoxAverage, separableBoxBlur } from './distortion.ts';
import { computeEffectiveGradientField, computeGradient2x2Field, computeGradientAgreementField, computeGradientField, computeTriangleFold, fillGrayscalePreview, paintScalarFieldAsGray, paintVectorFieldAsColor } from './gradientField.ts';
import { computeLocalJacobianField, paintJacobianFieldAsColor } from './localJacobian.ts';
import { computeWalkedGradientField } from './tangentWalk.ts';

// Shared tail for both capture sources: given a final analysis-resolution
// grayscale, paints whichever of the 4 direction/scalar field views is
// currently selected.
export function paintFieldViewFromGray(camera: Camera, gray: Float64Array) {
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const settings = camera.settings;
  if (settings.fieldView === 'triangleFold') {
    const folded = computeTriangleFold(gray);
    fillGrayscalePreview(folded, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  } else if (settings.fieldView === 'gradient') {
    const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
    camera.lastDisplayedVectorField = field;
    paintVectorFieldAsColor(field, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  } else if (settings.fieldView === 'gradient2x2') {
    const field = computeGradient2x2Field(gray, w, h);
    camera.lastDisplayedVectorField = field;
    paintVectorFieldAsColor(field, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  } else if (settings.fieldView === 'jacobian') {
    const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
    // Stashed for the hover overlay's spoked walk (see hoverDebugOverlays.ts)
    // to sample the same underlying gradient field the Jacobian was built
    // from -- lastJacobianField's eigenvectors seed the walk's two axes.
    camera.lastDisplayedVectorField = field;
    const jac = computeLocalJacobianField(field, Math.round(settings.simGradRadius));
    camera.lastJacobianField = jac;
    paintJacobianFieldAsColor(jac, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  } else if (settings.fieldView === 'walked') {
    const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
    const agreement = computeGradientAgreementField(field, Math.round(settings.coherenceRadius));
    const effective = computeEffectiveGradientField(field, agreement);
    camera.lastEffectiveField = effective;
    const walked = computeWalkedGradientField(settings, effective);
    camera.lastDisplayedVectorField = walked;
    paintVectorFieldAsColor(walked, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  } else if (settings.fieldView === 'agreement') {
    const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
    const agreement = computeGradientAgreementField(field, Math.round(settings.coherenceRadius));
    paintScalarFieldAsGray(agreement, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  } else if (settings.fieldView === 'effective') {
    const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
    const agreement = computeGradientAgreementField(field, Math.round(settings.coherenceRadius));
    const effective = computeEffectiveGradientField(field, agreement);
    camera.lastEffectiveField = effective;
    camera.lastDisplayedVectorField = effective;
    paintVectorFieldAsColor(effective, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  }
}

// The tangent-walk-path hover overlay (overlays/hoverDebugOverlays.ts) always
// wants the EFFECTIVE field specifically, regardless of which fieldView the
// user currently has selected for the main preview -- but paintFieldViewFromGray
// above only ever computes it as a side effect of the 'walked'/'effective'
// branches, and is itself skipped entirely whenever hideField is on. Called
// at the end of both capture paths below so showTangentWalkPath gets a fresh
// effective field either way; a no-op if paintFieldViewFromGray already set
// one this pass (fieldView was 'walked'/'effective' and hideField was off).
function ensureEffectiveField(camera: Camera, gray: Float64Array) {
  if (camera.lastEffectiveField || !camera.settings.showTangentWalkPath) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const field = computeGradientField(gray, w, h, Math.round(camera.settings.simGradRadius));
  const agreement = computeGradientAgreementField(field, Math.round(camera.settings.coherenceRadius));
  camera.lastEffectiveField = computeEffectiveGradientField(field, agreement);
}

export function updateDistortedPreview(camera: Camera) {
  camera.lastDisplayedVectorField = null;
  camera.lastEffectiveField = null;
  camera.lastJacobianField = null;
  const settings = camera.settings;
  if (settings.hideField) {
    for (let i = 0; i < camera.distortedPreviewData.length; i += 4) {
      camera.distortedPreviewData[i] = 0; camera.distortedPreviewData[i + 1] = 0; camera.distortedPreviewData[i + 2] = 0; camera.distortedPreviewData[i + 3] = 255;
    }
    camera.distortedPreviewTex.needsUpdate = true;
  }
  // Anything that reads camera.lastNoisedPreviewGray directly (not just the
  // painted field-view colors this function's own hideField branches gate)
  // needs to be listed here, or hideField skips recomputing it entirely --
  // the STALE gray buffer (e.g. from before a viewport resize, when it was
  // a different width/height) then gets reused at the CURRENT rtSize.w/h by
  // whichever overlay reads it, reading/writing at systematically wrong
  // offsets -- the diagonal "streaking" artifact. Bucket-fill and top-
  // gradient (overlays/bucketFillOverlay.ts, overlays/
  // gradientHighlightOverlays.ts) were both missing here; showBucketFillJoin/
  // Composite/MergeMarkers don't need their own entry since their
  // availability gating already forces them off whenever
  // showBucketFillSegments is off. showTangentWalkPath needs lastEffectiveField
  // specifically, not lastNoisedPreviewGray -- see ensureEffectiveField above.
  const needGrayForOverlay = settings.showTrueContamination || settings.showReconstructedContamination
    || settings.showBucketFillSegments || settings.showTopGradient || settings.showTangentWalkPath;
  if (settings.hideField && !needGrayForOverlay) return;

  if (isPhysical(camera)) {
    if (!camera.lastRealCaptureGray) return;
    if (!settings.hideField) {
      if (settings.fieldView === 'raw' || settings.fieldView === 'antialiased' || settings.fieldView === 'downsampled' || settings.fieldView === 'noised') {
        fillGrayscalePreview(camera.lastRealCaptureGray, camera.distortedPreviewData);
        camera.distortedPreviewTex.needsUpdate = true;
      } else {
        paintFieldViewFromGray(camera, camera.lastRealCaptureGray);
      }
    }
    camera.lastNoisedPreviewGray = camera.lastRealCaptureGray;
    ensureEffectiveField(camera, camera.lastRealCaptureGray);
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
  ensureEffectiveField(camera, noised);
}

export const PREVIEW_UPDATE_INTERVAL_MS = 100; // ~10fps
