import { type Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { flipDy, pipelineField } from './pipelineField.ts';
import { recomputeFromLastCapture } from '../pipeline/axesReconstruction.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { globalState } from '../state.ts';
import { canvas, gradientArrowGroup, levelLineArrowGroup, lsdCompositeGroup, throughCamCanvas, toggleCompositeLineFamiliesBtn, toggleGradientArrowBtn, toggleHideFieldBtn, toggleLevelLineArrowBtn, toggleLsdCompositeBtn, toggleLsdRawRegionsBtn, toggleLsdRejectedBtn, toggleLsdSegmentsBtn, toggleReconContamBtn, toggleTopGradientBtn, toggleSampleLatticeBtn, toggleTrueCardinalOrientationBtn, toggleTrueContamBtn } from '../ui/dom.ts';
import { computeThroughRect } from '../ui/layout.ts';
import { persistConfig } from '../config.ts';
import { updateContaminationOverlays } from './contaminationOverlays.ts';
import { hashSeedIndexToHueDeg, repaintLsdRawRegionsHighlight } from './lsdOverlay.ts';
import { drawOneArrow, svgEl } from './svgUtil.ts';

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





// Draws the pose's own voteComposites -- the composite lines actually fed to
// fitPairOfPlanes and classified by pose/stages/period/gridPeriodPhase.ts (pipeline/
// votes.ts's computeGradient2x2Composites). Drawing this exact same array
// (rather than an independently-recomputed copy) guarantees a line's root
// here means the same thing as it does in gpp's own row/col maps.
//
// Family coloring (pose/stages/period/gridPeriodPhase.ts): blue = row family, red =
// column family, black -> full-color by each line's own RANK within its
// family (sorted by its rectified `value` -- the same order the
// period/phase fit itself assigns integer indices in), so this literally
// shows the sequence the fit will register each line as. Gray for any line
// gridPeriodPhase itself skipped (e.g. a degenerate gnomonic projection).
// Draws the pose's own voteComposites -- the composite lines actually fed to
// fitPairOfPlanes and classified by pose/stages/period/gridPeriodPhase.ts (pipeline/
// votes.ts's computeGradient2x2Composites) -- as SVG lines in
// lsdCompositeGroup. Populated once per REAL capture (not live-recomputed
// per LSD slider tweak, unlike the rectangle/rejected/raw-region views in
// overlays/lsdOverlay.ts): their pixel coords come from the
// row-flipped `gray` axesReconstruction.ts feeds the vote/grid-period-phase
// pipeline (its own flipRowsF64, kept separate from
// camera.lastNoisedPreviewGray), and recomputing that same row-flip
// live from the raw preview would risk exactly the kind of "two different
// computations of the same thing" mismatch a past version of this file
// already had and fixed -- see this file's git history.
//
// ── THE COORDINATES ARE TOP-DOWN, whatever the paragraph above once said ──
//
// An earlier version of this comment claimed the endpoints came from a
// row-flipped gray. They do not, and the arithmetic below never matched that
// claim: `rasterY = fieldH - 1 - fy` followed by measuring UP from the rect's
// bottom is the composition that maps a TOP-DOWN fy=0 to the TOP of the rect.
// src/pose2's `lines` are in the pipeline's own top-down pixel space, which is
// the dominant convention everywhere except the preview textures, so this is
// correct as written -- and it was correct-looking either way round, which is
// why the stale comment survived. See overlays/pipelineField.ts.
//
// COLOUR: a unique per-line hue hashed from the index of the REGION the segment
// was fitted to (overlays/lsdOverlay.ts's hashSeedIndexToHueDeg). The
// showCompositeLineFamilies mode -- blue=row, red=column, shaded by each line's
// rank within its family -- needs the period search's rectified values, which
// are `rowSamples`/`colSamples` and are not requested yet, so the toggle
// currently recolours nothing.
function drawCompositeLines(camera: Camera) {
  while (lsdCompositeGroup.firstChild) lsdCompositeGroup.removeChild(lsdCompositeGroup.firstChild);
  const settings = camera.settings;
  const composites = camera.pose?.voteComposites;
  if (!settings.showLsdComposite || !composites) return;
  const rect = computeThroughRect(camera);
  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const toScreen = (fx: number, fy: number) => {
    const rasterY = fieldH - 1 - fy;
    return {
      x: rect.x + (fx + 0.5) * (rect.w / fieldW),
      y: rect.y + rect.h - (rasterY + 0.5) * (rect.h / fieldH),
    };
  };

  // The row/column FAMILY colouring ranked each detected line by its rectified
  // periodic coordinate, which came off the deleted period search's rowLines/
  // colLines. Nothing produces those now, so both ranks stay null and the lines
  // draw in their un-ranked colour -- which is exactly what already happened
  // whenever the toggle was off or the search had not run.
  for (const { root, line } of composites) {
    const a = toScreen(line.x1, line.y1), b = toScreen(line.x2, line.y2);
    let strokeColor: string;
    {
      // The blue/red FAMILY colouring used to go here, ranking each line by its
      // rectified periodic coordinate. That came off the deleted period search
      // (rowLines/colLines), so every line now takes the per-root hash colour --
      // which is the branch that already ran whenever the toggle was off.
      const [hr, hg, hb] = hsvToRgb(hashSeedIndexToHueDeg(root), 0.85, 1);
      strokeColor = `rgb(${hr},${hg},${hb})`;
    }
    // Grouped (not two independently-alpha'd strokes) so the halo+color
    // pair composites as one opaque unit first, THEN that unit is 50%
    // see-through against whatever's underneath -- two separately-alpha'd
    // strokes would instead let the color line partially show the halo
    // through itself, muddying the line's own color.
    const lineGroup = svgEl('g', { opacity: 0.5 });
    lineGroup.appendChild(svgEl('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: 'rgba(0,0,0,0.85)', 'stroke-width': 5 }));
    lineGroup.appendChild(svgEl('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: strokeColor, 'stroke-width': 2.5 }));
    lsdCompositeGroup.appendChild(lineGroup);
  }
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

  drawCompositeLines(camera);

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
// throughCamCanvas is NOT pointer-events:none (see sphere-lab.html's own
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
toggleLsdCompositeBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLsdComposite = !cam.settings.showLsdComposite;
  toggleLsdCompositeBtn.classList.toggle('active', cam.settings.showLsdComposite);
  persistConfig();
  drawCompositeLines(cam);
});
toggleCompositeLineFamiliesBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showCompositeLineFamilies = !cam.settings.showCompositeLineFamilies;
  toggleCompositeLineFamiliesBtn.classList.toggle('active', cam.settings.showCompositeLineFamilies);
  persistConfig();
  drawCompositeLines(cam); // recolors the SAME lines, doesn't recompute them
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
// hides with the mode. Same display-only story as the toggle above -- animate()
// reads the setting fresh each frame, so there is nothing to recompute.
toggleSampleLatticeBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showSampleLattice = !cam.settings.showSampleLattice;
  toggleSampleLatticeBtn.classList.toggle('active', cam.settings.showSampleLattice);
  persistConfig();
});


// No-op hook -- see contaminationOverlays.ts's updateContaminationAvailability
// for why this stays a named function now that there's no field view to gate on.
export function updateGradientArrowAvailability() {
  const cam = activeCamera(); if (!cam) return;
  toggleGradientArrowBtn.disabled = false;
  toggleLevelLineArrowBtn.disabled = false;
}
