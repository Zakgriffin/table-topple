import { Camera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { COL_DIR, MATH_QUAT, ROW_DIR } from '../constants.ts';
import { getAnalysisVFovRad } from '../pipeline/capture.ts';
import { RECON_CONTAM_COLOR, TRUE_CONTAM_COLOR, computeContaminationAlpha, paintContaminationOverlay } from '../pipeline/contamination.ts';
import { computeGradient2x2Field, computeGradientAgreementField } from '../pipeline/gradientField.ts';
import { toggleReconContamBtn, toggleTrueContamBtn } from '../ui/dom.ts';

// Recomputes whichever contamination overlay(s) are actually toggled on.
export function updateContaminationOverlays(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showTrueContamination && !settings.showReconstructedContamination) return;
  // Deliberately NOT gated on fieldView -- this overlay derives its own field
  // from lastNoisedPreviewGray below, so it never depended on gradient2x2
  // being the painted view, only on the gray being fresh (which
  // preview.ts's overlaysNeedGray now guarantees for these toggles on every
  // view). Drawing contamination over the raw/noised image is often easier to
  // read than over the hue field it used to be locked to.
  if (!camera.lastNoisedPreviewGray) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const lum = camera.lastNoisedPreviewGray;
  const vFovRad = getAnalysisVFovRad(camera);
  const field = computeGradient2x2Field(lum, w, h);
  const agreement = computeGradientAgreementField(field, Math.round(settings.coherenceRadius));

  if (settings.showTrueContamination && isSimulated(camera)) {
    const alpha = computeContaminationAlpha(field, agreement, ROW_DIR, COL_DIR, camera.camQuat, vFovRad, camera.aspect);
    paintContaminationOverlay(alpha, TRUE_CONTAM_COLOR, camera.trueContamData);
    camera.trueContamTex.needsUpdate = true;
  } else if (settings.showTrueContamination) {
    // No ground truth for a physical camera -- nothing to compare against.
    camera.trueContamData.fill(0);
    camera.trueContamTex.needsUpdate = true;
  }
  if (settings.showReconstructedContamination) {
    if (camera.lastRecoveredAxes) {
      const alpha = computeContaminationAlpha(field, agreement, camera.lastRecoveredAxes.Drow, camera.lastRecoveredAxes.Dcol, MATH_QUAT, vFovRad, camera.aspect);
      paintContaminationOverlay(alpha, RECON_CONTAM_COLOR, camera.reconContamData);
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
