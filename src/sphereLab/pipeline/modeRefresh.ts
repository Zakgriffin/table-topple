import { Camera } from '../camera/model.ts';
import { Mode } from '../types.ts';
import { updateBucketFillOverlay } from '../overlays/bucketFillOverlay.ts';
import { updateBucketFillJoinOverlay } from '../overlays/bucketFillJoinOverlay.ts';
import { updateContaminationOverlays } from '../overlays/contaminationOverlays.ts';
import { updateTopGradientOverlay } from '../overlays/gradientHighlightOverlays.ts';
import { drawGridPeriodPhasePlot } from '../overlays/gridPeriodPhaseOverlays.ts';
import { lastHoverClientX, lastHoverClientY, updateHoverOverlays } from '../overlays/hoverDebugOverlays.ts';
import { buildProjectedTexture } from './decodeGrid.ts';
import { updateDistortedPreview } from './preview.ts';

// Single source of truth for "make whichever visualizations `mode` actually
// shows reflect this camera's current captured/decoded state" -- called from
// every place that's supposed to leave those visualizations non-stale: the
// recapture button (via runAxesReconstruction's own tail), setMode (switching
// into a view), and the settings-change-driven captureDirty throttle in
// main.ts's animate(). Before this, each of those three triggers hand-rolled
// its own partial list and they'd drifted out of sync with each other and
// with what each mode's animate() branch actually renders -- see this
// session's chat for the audit. Doesn't render the underlying data (that's
// runAxesReconstruction's job); this only repaints/redraws FROM whatever
// data already exists on `camera`, so it's cheap enough to call on every
// trigger without its own throttle.
export function refreshModeVisualizations(camera: Camera, mode: Mode) {
  updateDistortedPreview(camera); // feeds the PIP in world/through/inside, and Through-Cam's own preview
  if (mode === 'through') {
    updateContaminationOverlays(camera);
    updateTopGradientOverlay(camera);
    updateBucketFillOverlay(camera);
    updateBucketFillJoinOverlay(camera); // needs lastBucketFillSegments, hence after updateBucketFillOverlay
    updateHoverOverlays(lastHoverClientX, lastHoverClientY);
    // The grid period/phase SVG plot lives in the Through-Cam toggle panel
    // (#contamToggles, sphere-lab.html), not Projected-Cam's, despite what
    // it's plotting -- self-gates on lastGridPeriodPhase existing internally.
    drawGridPeriodPhasePlot(camera);
  } else if (mode === 'projected') {
    buildProjectedTexture(camera);
  } else if (mode === 'world') {
    // The recovered-floor decal reuses projectedPreviewTex -- only worth
    // repainting if that overlay is actually the reason it'd be visible.
    if (camera.settings.showRecoveredFloor) buildProjectedTexture(camera);
  }
}
