import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { paintTopGradientOverlay, TOP_GRADIENT_COLOR } from '../pipeline/gradientHighlight.ts';
import { toggleTopGradientBtn } from '../ui/dom.ts';

// Recomputes the top-gradient overlay if it's actually toggled on.
export function updateTopGradientOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showTopGradient) return;
  if (settings.fieldView !== 'gradient2x2') return;
  if (!camera.lastNoisedPreviewGray) return;
  paintTopGradientOverlay(TOP_GRADIENT_COLOR, camera.topGradientData);
  camera.topGradientTex.needsUpdate = true;
}

export function updateTopGradientAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.fieldView === 'gradient2x2';
  toggleTopGradientBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showTopGradient = false;
    toggleTopGradientBtn.classList.remove('active');
    cam.topGradientData.fill(0);
    cam.topGradientTex.needsUpdate = true;
  }
}
