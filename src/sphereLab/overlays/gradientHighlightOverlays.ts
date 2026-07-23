import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { computeTopGradientAlpha, paintTopGradientOverlay, TOP_GRADIENT_COLOR } from '../pipeline/gradientHighlight.ts';
import { computeEffectiveGradientField, computeGradient2x2Field, computeGradientAgreementField, computeGradientField } from '../pipeline/gradientField.ts';
import { computeWalkedGradientField } from '../pipeline/tangentWalk.ts';
import { FieldView } from '../types.ts';
import { toggleTopGradientBtn } from '../ui/dom.ts';

// Every FieldView that's an actual GradientField (has a direction, not just
// a scalar).
const VECTOR_FIELD_VIEWS: readonly FieldView[] = ['gradient', 'gradient2x2', 'walked'];

// Recomputes the top-gradient overlay if it's actually toggled on.
export function updateTopGradientOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showTopGradient) return;
  if (!VECTOR_FIELD_VIEWS.includes(settings.fieldView)) return;
  if (!camera.lastNoisedPreviewGray) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const lum = camera.lastNoisedPreviewGray;

  let field;
  if (settings.fieldView === 'gradient') {
    field = computeGradientField(lum, w, h, Math.round(settings.simGradRadius));
  } else if (settings.fieldView === 'gradient2x2') {
    field = computeGradient2x2Field(lum, w, h);
  } else {
    // Only 'walked' can reach here -- VECTOR_FIELD_VIEWS already filtered
    // out anything else at the top of this function.
    const rawField = computeGradientField(lum, w, h, Math.round(settings.simGradRadius));
    const agreement = computeGradientAgreementField(rawField, Math.round(settings.coherenceRadius));
    const effectiveField = computeEffectiveGradientField(rawField, agreement);
    field = computeWalkedGradientField(settings, effectiveField);
  }

  const alpha = computeTopGradientAlpha(field, 0, 100);
  paintTopGradientOverlay(alpha, TOP_GRADIENT_COLOR, camera.topGradientData);
  camera.topGradientTex.needsUpdate = true;
}

export function updateTopGradientAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = VECTOR_FIELD_VIEWS.includes(cam.settings.fieldView);
  toggleTopGradientBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showTopGradient = false;
    toggleTopGradientBtn.classList.remove('active');
    cam.topGradientData.fill(0);
    cam.topGradientTex.needsUpdate = true;
  }
}
