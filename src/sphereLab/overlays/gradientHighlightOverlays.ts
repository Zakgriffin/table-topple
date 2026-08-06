import { type Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { computeGradient2x2Field } from '../pipeline/gradientField.ts';
import { paintTopGradientOverlay, TOP_GRADIENT_COLOR } from '../pipeline/gradientHighlight.ts';
import { toggleTopGradientBtn } from '../ui/dom.ts';

// Recomputes the top-gradient overlay if it's actually toggled on.
export function updateTopGradientOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showTopGradient) return;
  // Not gated on fieldView -- derives its own field from lastNoisedPreviewGray
  // below, so gradient2x2 was never a real dependency. See
  // pipeline/preview.ts's overlaysNeedGray.
  if (!camera.lastNoisedPreviewGray) return;
  const { w, h } = camera.rtSize;
  const field = computeGradient2x2Field(camera.lastNoisedPreviewGray, w, h);
  paintTopGradientOverlay(TOP_GRADIENT_COLOR, field, camera.topGradientData);
  camera.topGradientTex.needsUpdate = true;
}

// No-op hook -- see contaminationOverlays.ts's updateContaminationAvailability
// for why this stays a named function now that there's no field view to gate on.
export function updateTopGradientAvailability() {
  const cam = activeCamera(); if (!cam) return;
  toggleTopGradientBtn.disabled = false;
}
