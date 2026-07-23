import { Camera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { COL_DIR, MATH_QUAT, ROW_DIR } from '../constants.ts';
import { getAnalysisVFovRad } from '../pipeline/capture.ts';
import { RECON_CONTAM_COLOR, TRUE_CONTAM_COLOR, computeContaminationAlpha, paintContaminationOverlay } from '../pipeline/contamination.ts';
import { computeEffectiveGradientField, computeGradientAgreementField, computeGradientField } from '../pipeline/gradientField.ts';
import { computeWalkedGradientField } from '../pipeline/tangentWalk.ts';
import { toggleReconContamBtn, toggleTrueContamBtn } from '../ui/dom.ts';

// Recomputes whichever contamination overlay(s) are actually toggled on.
export function updateContaminationOverlays(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showTrueContamination && !settings.showReconstructedContamination) return;
  if (settings.fieldView !== 'gradient' && settings.fieldView !== 'walked') return;
  if (!camera.lastNoisedPreviewGray) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const lum = camera.lastNoisedPreviewGray;
  const vFovRad = getAnalysisVFovRad(camera);
  const rawField = computeGradientField(lum, w, h, Math.round(settings.simGradRadius));
  const agreement = computeGradientAgreementField(rawField, Math.round(settings.coherenceRadius));
  const field = settings.fieldView === 'gradient' ? rawField
    : computeWalkedGradientField(settings, computeEffectiveGradientField(rawField, agreement));

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


export function updateContaminationAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.fieldView === 'gradient' || cam.settings.fieldView === 'walked';
  toggleTrueContamBtn.disabled = !relevant;
  toggleReconContamBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showTrueContamination = false;
    cam.settings.showReconstructedContamination = false;
    toggleTrueContamBtn.classList.remove('active');
    toggleReconContamBtn.classList.remove('active');
    cam.trueContamData.fill(0); cam.trueContamTex.needsUpdate = true;
    cam.reconContamData.fill(0); cam.reconContamTex.needsUpdate = true;
  }
}
