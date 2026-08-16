import { type Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { paintTopGradientOverlay, TOP_GRADIENT_COLOR } from '../pipeline/gradientHighlight.ts';
import { pipelineField } from './pipelineField.ts';
import { toggleTopGradientBtn } from '../ui/dom.ts';

// Repaints the top-gradient overlay if it's actually toggled on.
//
// Reads the POSE RUN's field now instead of recomputing one from
// lastNoisedPreviewGray. Null means nobody asked for it -- which, with the
// toggle on, means the request in axesReconstruction.ts and this consumer
// disagree. Draws nothing rather than falling back to a recomputation: a
// fallback would silently reinstate the duplicated stage, and would do it
// exactly where it is hardest to notice.
export function updateTopGradientOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showTopGradient) return;
  const field = pipelineField(camera);
  if (!field) return;
  paintTopGradientOverlay(TOP_GRADIENT_COLOR, field, camera.topGradientData);
  camera.topGradientTex.needsUpdate = true;
}

// No-op hook -- see contaminationOverlays.ts's updateContaminationAvailability
// for why this stays a named function now that there's no field view to gate on.
export function updateTopGradientAvailability() {
  const cam = activeCamera(); if (!cam) return;
  toggleTopGradientBtn.disabled = false;
}
