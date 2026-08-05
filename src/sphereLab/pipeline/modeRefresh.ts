import { Camera } from '../camera/model.ts';
import { Mode } from '../types.ts';
import { updateContaminationOverlays } from '../overlays/contaminationOverlays.ts';
import { updateTopGradientOverlay } from '../overlays/gradientHighlightOverlays.ts';
import { drawGridPeriodPhasePlot } from '../overlays/gridPeriodPhaseOverlays.ts';
import { lastHoverClientX, lastHoverClientY, updateHoverOverlays } from '../overlays/hoverDebugOverlays.ts';
import { updateLsdOverlay } from '../overlays/lsdOverlay.ts';
import { buildProjectedTexture, ProjectedSampleResult } from './decodeGrid.ts';
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
// data already exists on `camera`.
//
// Async (and every caller now threads that through -- see this session's
// chat) because the projected-texture repaint can go through a real
// (possibly GPU) re-projection when `precomputedProjection` isn't provided
// -- buildProjectedTexture used to silently always recompute on CPU only,
// bypassing globalState.forceCPU's projection branch entirely. `precomputedProjection` lets
// axesReconstruction.ts's recomputeStages -- which, for the active camera,
// has USUALLY already computed and even painted this exact result a few
// lines earlier -- hand it over directly instead of paying for a second
// (possibly GPU) round trip; the other two callers (a plain mode switch, or
// the throttled preview tick) have no fresher result available and leave it
// unset, so this recomputes for them, same as before.
export async function refreshModeVisualizations(
  camera: Camera, mode: Mode, precomputedProjection?: { value: ProjectedSampleResult },
): Promise<void> {
  updateDistortedPreview(camera); // feeds the PIP in world/through/inside, and Through-Cam's own preview
  if (mode === 'through') {
    updateContaminationOverlays(camera);
    updateTopGradientOverlay(camera);
    await updateLsdOverlay(camera);
    updateHoverOverlays(lastHoverClientX, lastHoverClientY);
    // The grid period/phase SVG plot lives in the Through-Cam toggle panel
    // (#contamToggles, sphere-lab.html), not Projected-Cam's, despite what
    // it's plotting -- self-gates on lastGridPeriodPhase existing internally.
    drawGridPeriodPhasePlot(camera);
  } else if (mode === 'projected') {
    await buildProjectedTexture(camera, precomputedProjection);
  } else if (mode === 'world') {
    // The recovered-floor decal reuses projectedPreviewTex -- only worth
    // repainting if that overlay is actually the reason it'd be visible.
    if (camera.settings.showRecoveredFloor) await buildProjectedTexture(camera, precomputedProjection);
  }
}
