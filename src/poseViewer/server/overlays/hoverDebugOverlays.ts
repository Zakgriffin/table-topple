import { type Camera } from '../../shared/camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { flipDy, pipelineField } from './pipelineField.ts';
import { recomputeFromLastCapture } from '../pipeline/axesReconstruction.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { globalState } from '../../shared/state.ts';
import { canvas, gradientArrowGroup, levelLineArrowGroup, lsdCompositeGroup, throughCamCanvas, toggleColorByFamilyBtn, toggleGradientArrowBtn, toggleHideFieldBtn, toggleLevelLineArrowBtn, toggleCompositeLinesBtn, toggleLsdRawRegionsBtn, toggleLsdRejectedBtn, toggleLsdSegmentsBtn, toggleReconContamBtn, toggleTopGradientBtn, toggleLineSegmentsBtn, toggleSampleLatticeBtn, toggleTrueCardinalOrientationBtn, toggleTrueContamBtn } from '../ui/dom.ts';
import { computeThroughRect } from '../ui/layout.ts';
import { persistConfig } from '../../shared/config.ts';
import { updateContaminationOverlays } from './contaminationOverlays.ts';
import { repaintLsdRawRegionsHighlight } from './lsdOverlay.ts';
import { drawLinesSvg, drawableLines } from './lines.ts';
import { drawOneArrow } from './svgUtil.ts';

// Clears the gradient/level-line arrow groups only -- NOT lsdRectanglesGroup/
// lsdCompositeGroup, even though all four now share the same <svg> (see
// ui/dom.ts's lsdSvgOverlay) -- those two are independently toggled/redrawn
// by overlays/lsdOverlay.ts and drawCompositeLines below, so clearing them
// here on every hover move (this function's own call sites: updateHoverOverlays,
// onHoverPointerLeave, updateGradientArrowAvailability) would wipe overlays
// this function has no business touching.
export function clearArrowOverlays() {
  while (gradientArrowGroup.firstChild) gradientArrowGroup.removeChild(gradientArrowGroup.firstChild);
  while (levelLineArrowGroup.firstChild) levelLineArrowGroup.removeChild(levelLineArrowGroup.firstChild);
}





// Draws the pose's own `composites` -- the detected segments the pipeline
// actually voted from, as SVG lines in lsdCompositeGroup. Drawing that exact
// array rather than an independently-recomputed copy is the point: a second
// detector in an overlay is the failure this whole display path is built to
// avoid, and a past version of this file had precisely that mismatch.
//
// (A second copy of this paragraph sat here, describing the same function twice
// and citing files deleted with the old pipeline. Both copies also claimed the
// endpoints came from a ROW-FLIPPED gray, which the arithmetic below has never
// matched -- see the next block.)
//
// ── THE COORDINATES ARE TOP-DOWN, whatever the paragraph above once said ──
//
// An earlier version of this comment claimed the endpoints came from a
// row-flipped gray. They do not, and the arithmetic below never matched that
// claim: `rasterY = fieldH - 1 - fy` followed by measuring UP from the rect's
// bottom is the composition that maps a TOP-DOWN fy=0 to the TOP of the rect.
// src/pose's `lines` are in the pipeline's own top-down pixel space, which is
// the dominant convention everywhere except the preview textures, so this is
// correct as written -- and it was correct-looking either way round, which is
// why the stale comment survived. See overlays/pipelineField.ts.
//
// ── THE LINE OVERLAY MOVED TO overlays/lines.ts ──────────────────────────
//
// `familyRanks` and `drawCompositeLines` lived here, and between them they
// owned the Through-Cam half of a picture the Projected-Cam and World views
// each drew their own way. They are one model now, with one colour function and
// one style table, so the three views differ ONLY in how they place a line.
// This file keeps the projection, which is genuinely its own: image space,
// through the letterboxed Through-Cam rect.
//
// ── THE COORDINATES ARE TOP-DOWN ─────────────────────────────────────────
//
// An earlier version of this comment claimed the endpoints came from a
// row-flipped gray. They do not, and the arithmetic below never matched that
// claim: `rasterY = fieldH - 1 - fy` followed by measuring UP from the rect's
// bottom is the composition that maps a TOP-DOWN fy=0 to the TOP of the rect.
// src/pose's lines are in the pipeline's own top-down pixel space, which is the
// dominant convention everywhere except the preview textures, so this is correct
// as written -- and it was correct-looking either way round, which is why the
// stale comment survived. See overlays/pipelineField.ts.
function drawThroughCamLines(camera: Camera) {
  const rect = computeThroughRect(camera);
  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  drawLinesSvg(lsdCompositeGroup, camera, drawableLines(camera), (fx, fy) => {
    const rasterY = fieldH - 1 - fy;
    return {
      x: rect.x + (fx + 0.5) * (rect.w / fieldW),
      y: rect.y + rect.h - (rasterY + 0.5) * (rect.h / fieldH),
    };
  });
}

// Single per-hover entry point -- operates on the ACTIVE camera, since only
// its Through-Cam view is ever on screen. Also doubles as the redraw entry
// point for PERSISTENT content (LSD rectangles, vote-family lines) below --
// they don't depend on cursor position, but share this canvas/clear cycle
// with the cursor-driven content further down, so anything that recomputes
// them re-invokes this function (with the last-known cursor position) to
// get them back on screen without waiting for the next pointermove.
export function updateHoverOverlays(clientX: number, clientY: number) {
  const camera = activeCamera();
  if (!camera) { clearArrowOverlays(); return; }
  const settings = camera.settings;
  const arrowsOn = settings.showGradientArrow || settings.showLevelLineArrow;

  clearArrowOverlays();

  if (globalState.mode !== 'through') return;

  drawThroughCamLines(camera);

  // Field-pixel-under-cursor, computed UNCONDITIONALLY (not gated behind
  // arrowsOn) -- tracked on the camera itself so other hover-driven repaints
  // (this function's own raw-regions highlight call just below) can read it
  // without needing their own copy of this rect-containment/field-index
  // math. See camera.lastHoverFieldIndex's own comment.
  const rect = computeThroughRect(camera);
  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  let fieldIndex: number | null = null;
  if (clientX >= rect.x && clientX < rect.x + rect.w && clientY >= rect.y && clientY < rect.y + rect.h) {
    const nx = (clientX - rect.x) / rect.w, ny = (clientY - rect.y) / rect.h;
    const fieldCol = Math.min(fieldW - 1, Math.max(0, Math.floor(nx * fieldW)));
    // TOP-DOWN, matching the pipeline's own convention: `ny` already counts
    // down from the top of the rect, so this is the direct mapping and the
    // 1-ny it replaces was the flip. THIS IS THE ONLY PLACE the pointer is
    // converted -- camera.lastHoverFieldIndex is a pipeline index from here on,
    // and every consumer of it (this function's arrows,
    // repaintLsdRawRegionsHighlight, drawEdgeConnectivityPreview) indexes the
    // pipeline's own fx/fy/regionId with it directly. See
    // overlays/pipelineField.ts.
    const fieldRow = Math.min(fieldH - 1, Math.max(0, Math.floor(ny * fieldH)));
    fieldIndex = fieldRow * fieldW + fieldCol;
  }
  camera.lastHoverFieldIndex = fieldIndex;
  repaintLsdRawRegionsHighlight(camera);

  if (!arrowsOn || fieldIndex === null) return; // cursor outside -- markers (if any) stay drawn, just no hover-specific content below
  const i = fieldIndex;
  const fieldCol = i % fieldW, fieldRow = (i / fieldW) | 0;

  // The POSE RUN's field, not a fresh one. This site was the worst of the four
  // display recomputations by a wide margin: it rebuilt ~307k pixels of
  // gradient on every single pointermove in order to read ONE of them.
  //
  // `i` is already a pipeline index (see above), so the field is indexed
  // directly. fy still has to be negated into the display's bottom-up
  // convention, because the arrow drawing below negates again to reach screen
  // space -- a named function rather than inline arithmetic, since an
  // un-negated fy is a working arrow pointing the wrong way.
  const arrowField = pipelineField(camera);
  if (arrowsOn && arrowField) {
    const { fx, fy } = arrowField;
    const gx = fx[i], gy = flipDy(fy[i]);
    const mag = Math.hypot(gx, gy);
    if (mag > 0) {
      const px = rect.x + (fieldCol + 0.5) * (rect.w / fieldW);
      const py = rect.y + (fieldRow + 0.5) * (rect.h / fieldH);
      let hueTheta = Math.atan2(gy, gx);
      if (hueTheta < 0) hueTheta += Math.PI;
      if (hueTheta >= Math.PI) hueTheta -= Math.PI;
      const [rr, gg, bb] = hsvToRgb((hueTheta / Math.PI) * 360, 1, 1);
      const color = `rgb(${rr},${gg},${bb})`;

      if (settings.showGradientArrow) {
        const theta = Math.atan2(gy, gx);
        drawOneArrow(gradientArrowGroup, px, py, Math.cos(theta) * mag, -Math.sin(theta) * mag, color, settings.gradientArrowScale);
      }
      if (settings.showLevelLineArrow) {
        const theta = Math.atan2(gx, -gy);
        drawOneArrow(levelLineArrowGroup, px, py, Math.cos(theta) * mag, -Math.sin(theta) * mag, color, settings.gradientArrowScale);
      }
    }
  }
}
export let lastHoverClientX = -1, lastHoverClientY = -1;
function onHoverPointerMove(e: PointerEvent) {
  lastHoverClientX = e.clientX; lastHoverClientY = e.clientY;
  updateHoverOverlays(e.clientX, e.clientY);
}
function onHoverPointerLeave() {
  lastHoverClientX = -1; lastHoverClientY = -1;
  clearArrowOverlays();
  // Also reset the raw-regions highlight back to "every region" -- without
  // this, the last-hovered region would stay isolated forever once the
  // cursor actually leaves the canvas (updateHoverOverlays, which would
  // otherwise naturally null this out via its own rect-containment check,
  // is deliberately NOT called here, see this function's own minimal scope).
  const camera = activeCamera();
  if (camera) { camera.lastHoverFieldIndex = null; repaintLsdRawRegionsHighlight(camera); }
}
canvas.addEventListener('pointermove', onHoverPointerMove);
canvas.addEventListener('pointerleave', onHoverPointerLeave);
// throughCamCanvas is NOT pointer-events:none (see pose-viewer-server.html's own
// comment -- it needs to be the real right-click target for "Save Image
// As"), so while it's on top in Through-Cam mode, events land on it
// instead of passing through to canvas#gl -- needs its own copy of the
// same listeners, or hover arrows/composite-line redraws would silently
// stop working the instant it's visible.
throughCamCanvas.addEventListener('pointermove', onHoverPointerMove);
throughCamCanvas.addEventListener('pointerleave', onHoverPointerLeave);

toggleHideFieldBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.hideField = !cam.settings.hideField;
  toggleHideFieldBtn.classList.toggle('active', cam.settings.hideField);
  updateDistortedPreview(cam);
  // hideField changes nothing the PIPELINE produces -- it only blanks the
  // painted field view -- so this one still just repaints.
  updateContaminationOverlays(cam);
});
// recomputeFromLastCapture, not updateContaminationOverlays: these overlays
// read the POSE RUN's fx/fy now (see overlays/pipelineField.ts), and the run
// only produces them when the toggle was already on -- see
// axesReconstruction.ts's displayIntermediates. Turning one on therefore has to
// re-run the stages, which also repaints every overlay through the visual tail.
// The same applies to all four handlers below.
toggleTrueContamBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showTrueContamination = !cam.settings.showTrueContamination;
  toggleTrueContamBtn.classList.toggle('active', cam.settings.showTrueContamination);
  updateDistortedPreview(cam);
  recomputeFromLastCapture(cam);
});
toggleReconContamBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showReconstructedContamination = !cam.settings.showReconstructedContamination;
  toggleReconContamBtn.classList.toggle('active', cam.settings.showReconstructedContamination);
  updateDistortedPreview(cam);
  recomputeFromLastCapture(cam);
});
toggleTopGradientBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showTopGradient = !cam.settings.showTopGradient;
  toggleTopGradientBtn.classList.toggle('active', cam.settings.showTopGradient);
  updateDistortedPreview(cam);
  recomputeFromLastCapture(cam);
});
// These 3 (rectangles/rejected/raw-regions) don't touch the gradient/
// level-line arrow groups at all -- rectangles draw into their own SVG
// group and raw-regions/rejected paint their own raster textures, both
// handled entirely inside updateLsdOverlay -- so no updateHoverOverlays
// call is needed after any of them.
//
// recomputeFromLastCapture rather than updateLsdOverlay directly, same as the
// overlays above: these views draw the pose run's own rects/regions, and a run
// that was not asked for them has none to draw. The recompute repaints through
// the visual tail.
toggleLsdSegmentsBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLsdSegments = !cam.settings.showLsdSegments;
  toggleLsdSegmentsBtn.classList.toggle('active', cam.settings.showLsdSegments);
  persistConfig();
  recomputeFromLastCapture(cam);
});
toggleLsdRejectedBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLsdRejected = !cam.settings.showLsdRejected;
  toggleLsdRejectedBtn.classList.toggle('active', cam.settings.showLsdRejected);
  persistConfig();
  recomputeFromLastCapture(cam);
});
toggleLsdRawRegionsBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLsdRawRegions = !cam.settings.showLsdRawRegions;
  toggleLsdRawRegionsBtn.classList.toggle('active', cam.settings.showLsdRawRegions);
  persistConfig();
  recomputeFromLastCapture(cam);
});
// THE JOIN TOGGLE USED TO LIVE HERE, and it moved to ui/cameraPanel.ts to sit
// beside the `joinKSigma` slider it now shares a setting with. It was the one
// toggle in this file that moved the POSE rather than the view, and keeping the
// button and the slider apart would mean each having to reach into the other to
// stay in sync -- which this file cannot do, since cameraPanel imports it.
// ── THE TWO LINE TOGGLES, WHICH ARE NOT THROUGH-CAM TOGGLES ─────────────
//
// Each drives all three views at once, so neither can repaint just this one --
// and both gate a readback (`compLines`, or `lines`+`anchorOf`+`joinScan`), so a
// run taken while one was off has nothing to draw FROM. Hence the recompute,
// same as every other readback-gating toggle here. refreshModeVisualizations
// then redraws whichever view is actually on screen.
toggleLineSegmentsBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLineSegments = !cam.settings.showLineSegments;
  toggleLineSegmentsBtn.classList.toggle('active', cam.settings.showLineSegments);
  persistConfig();
  recomputeFromLastCapture(cam);
});
toggleCompositeLinesBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showCompositeLines = !cam.settings.showCompositeLines;
  toggleCompositeLinesBtn.classList.toggle('active', cam.settings.showCompositeLines);
  persistConfig();
  recomputeFromLastCapture(cam);
});
// It DOES recolour the same lines rather than recomputing them -- but the colour
// now comes from a readback (`samples`/`family`, see inspectFor), so a run taken
// while this was off has nothing to recolour FROM. Hence the recompute, same as
// every other toggle that gates a buffer.
toggleColorByFamilyBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.colorLinesByFamily = !cam.settings.colorLinesByFamily;
  toggleColorByFamilyBtn.classList.toggle('active', cam.settings.colorLinesByFamily);
  persistConfig();
  recomputeFromLastCapture(cam);
});
// The two arrow toggles also gained a recompute, for the same reason as the
// raster overlays above: the arrows read the pose run's fx/fy, which the run
// only produces when one of these was already on. updateHoverOverlays still
// runs after it -- the recompute repaints through the visual tail, which does
// not know where the cursor is.
toggleGradientArrowBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showGradientArrow = !cam.settings.showGradientArrow;
  toggleGradientArrowBtn.classList.toggle('active', cam.settings.showGradientArrow);
  recomputeFromLastCapture(cam);
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
toggleLevelLineArrowBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLevelLineArrow = !cam.settings.showLevelLineArrow;
  toggleLevelLineArrowBtn.classList.toggle('active', cam.settings.showLevelLineArrow);
  recomputeFromLastCapture(cam);
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
// Purely a display-time rotation (see settings.ts's useTrueCardinalOrientation
// doc comment) -- no recompute to trigger here, main.ts's animate() reads
// this setting fresh every frame for the Projected-Cam render.
toggleTrueCardinalOrientationBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.useTrueCardinalOrientation = !cam.settings.useTrueCardinalOrientation;
  toggleTrueCardinalOrientationBtn.classList.toggle('active', cam.settings.useTrueCardinalOrientation);
});
// Lives in the Projected-view overlay group rather than the left panel: it only
// draws in Projected Cam, and that group is the one mode.ts already shows and
// hides with the mode.
//
// NOT display-only any more, unlike the toggle above. It gates `packed` in
// inspectFor, so a run that happened while this was off has no per-cell verdict
// to draw and animate() reading the flag fresh would just find nothing -- which
// is why this recomputes like every other readback-gating toggle rather than
// only repainting.
toggleSampleLatticeBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showSampleLattice = !cam.settings.showSampleLattice;
  toggleSampleLatticeBtn.classList.toggle('active', cam.settings.showSampleLattice);
  persistConfig();
  recomputeFromLastCapture(cam);
});
// (The rectified-lines toggle was here. It is gone: the Projected-Cam view of
// the composites is the SAME overlay in another projection, so it turns on with
// `showCompositeLines` like the other two views do.)


// No-op hook -- see contaminationOverlays.ts's updateContaminationAvailability
// for why this stays a named function now that there's no field view to gate on.
export function updateGradientArrowAvailability() {
  const cam = activeCamera(); if (!cam) return;
  toggleGradientArrowBtn.disabled = false;
  toggleLevelLineArrowBtn.disabled = false;
}
