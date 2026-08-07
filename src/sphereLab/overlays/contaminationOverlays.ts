import { type Camera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { COL_DIR, MATH_QUAT, ROW_DIR } from '../constants.ts';
import { getAnalysisVFovRad } from '../pipeline/capture.ts';
import { RECON_CONTAM_COLOR, TRUE_CONTAM_COLOR, computeContaminationAlpha, paintContaminationOverlay } from '../pipeline/contamination.ts';
import { computeGradientMagnitudeField } from '../pipeline/fieldPaint.ts';
import { pipelineField } from './pipelineField.ts';
import { toggleReconContamBtn, toggleTrueContamBtn } from '../ui/dom.ts';

// Recomputes whichever contamination overlay(s) are actually toggled on.
export function updateContaminationOverlays(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showTrueContamination && !settings.showReconstructedContamination) return;
  // Not gated on fieldView -- it draws over whatever view is painted, which is
  // often easier to read than the hue field it used to be locked to.
  //
  // The field is the POSE RUN's now, not a third independent recomputation off
  // lastNoisedPreviewGray. Null means nobody asked for it; see
  // updateTopGradientOverlay on why that draws nothing rather than falling back.
  const field = pipelineField(camera);
  if (!field) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const vFovRad = getAnalysisVFovRad(camera);
  // The alpha WEIGHT for both overlays: each pixel's own raw gradient
  // magnitude, frame-normalized. This used to be a spatially-aggregated
  // "agreement" field (a double-angle fold plus two separable blurs, windowed
  // by a `coherence radius` slider) with raw magnitude offered alongside it as
  // a third, cyan A/B overlay. That whole concept is gone -- the aggregation
  // never earned its keep and the A/B answered its own question, so the two
  // overlays that remain are the ones the cyan one was the control for.
  const magnitude = computeGradientMagnitudeField(field);

  if (settings.showTrueContamination && isSimulated(camera)) {
    const alpha = computeContaminationAlpha(field, magnitude, ROW_DIR, COL_DIR, camera.camQuat, vFovRad, camera.aspect);
    paintContaminationOverlay(alpha, TRUE_CONTAM_COLOR, camera.trueContamData, w, h);
    camera.trueContamTex.needsUpdate = true;
  } else if (settings.showTrueContamination) {
    // No ground truth for a physical camera -- nothing to compare against.
    camera.trueContamData.fill(0);
    camera.trueContamTex.needsUpdate = true;
  }
  if (settings.showReconstructedContamination) {
    if (camera.lastRecoveredAxes) {
      const alpha = computeContaminationAlpha(field, magnitude, camera.lastRecoveredAxes.Drow, camera.lastRecoveredAxes.Dcol, MATH_QUAT, vFovRad, camera.aspect);
      paintContaminationOverlay(alpha, RECON_CONTAM_COLOR, camera.reconContamData, w, h);
      toggleReconContamBtn.textContent = 'reconstructed contamination overlay (orange)';
    } else {
      camera.reconContamData.fill(0);
      toggleReconContamBtn.textContent = 'reconstructed contamination overlay (orange) — run "capture now" first';
    }
    camera.reconContamTex.needsUpdate = true;
  }
}


// Kept as a no-op hook rather than deleted: the contamination overlays used to
// be force-disabled on any field view other than gradient2x2, which silently
// turned the user's own toggles OFF on a view switch. They render over every
// view now (see updateContaminationOverlays above), so there is nothing to
// gate -- but every caller site (ui/cameraPanel.ts's fieldView radio group and
// its per-camera sync) still wants a single named place to hang any future
// availability rule, and dropping the function would scatter that decision.
export function updateContaminationAvailability() {
  const cam = activeCamera(); if (!cam) return;
  toggleTrueContamBtn.disabled = false;
  toggleReconContamBtn.disabled = false;
}
