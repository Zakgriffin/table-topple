import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { CameraSettingsCommon } from '../camera/settings.ts';
import { activeCamera } from '../camera/store.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { globalState } from '../state.ts';
import { canvas, gradientArrowCanvas, gradientArrowCtx, lsdCompositeGroup, persistControl, toggleCompositeLineFamiliesBtn, toggleGradientArrowBtn, toggleGradientArrowModeBtn, toggleHideFieldBtn, toggleLsdCompositeBtn, toggleLsdRawRegionsBtn, toggleLsdRejectedBtn, toggleLsdSegmentsBtn, toggleReconContamBtn, toggleTopGradientBtn, toggleTrueCardinalOrientationBtn, toggleTrueContamBtn } from '../ui/dom.ts';
import { computeThroughRect } from '../ui/layout.ts';
import { updateContaminationOverlays } from './contaminationOverlays.ts';
import { updateTopGradientOverlay } from './gradientHighlightOverlays.ts';
import { hashSeedIndexToHueDeg, updateLsdOverlay } from './lsdOverlay.ts';
import { svgEl } from './svgUtil.ts';

export function clearGradientArrowOverlay() {
  gradientArrowCtx.clearRect(0, 0, gradientArrowCanvas.width, gradientArrowCanvas.height);
}
export function drawOneArrow(px: number, py: number, dirVecX: number, dirVecY: number, color: string, scale: number) {
  const tipX = px + dirVecX * scale, tipY = py + dirVecY * scale;
  const headLen = 8, headAngle = Math.PI / 7;
  const backAngle = Math.atan2(tipY - py, tipX - px);
  const headPath = new Path2D();
  headPath.moveTo(tipX, tipY);
  headPath.lineTo(tipX - headLen * Math.cos(backAngle - headAngle), tipY - headLen * Math.sin(backAngle - headAngle));
  headPath.lineTo(tipX - headLen * Math.cos(backAngle + headAngle), tipY - headLen * Math.sin(backAngle + headAngle));
  headPath.closePath();

  const ctx = gradientArrowCtx;
  ctx.strokeStyle = 'rgba(0,0,0,0.85)'; ctx.fillStyle = 'rgba(0,0,0,0.85)'; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tipX, tipY); ctx.stroke();
  ctx.fill(headPath);
  ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();

  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tipX, tipY); ctx.stroke();
  ctx.fill(headPath);
  ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
}

// Diagnostic-only re-run of guidedTangentDirection's walk that also records
// which pixels actually got incorporated -- must stay in exact lockstep
// with guidedTangentDirection's own logic.
export function computeTangentWalkIncludedPixels(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { x: number; y: number }[] {
  const included: { x: number; y: number }[] = [{ x, y }];
  const seedTheta = Math.atan2(seedFy, seedFx);
  const tdx = -Math.sin(seedTheta), tdy = Math.cos(seedTheta);
  const seedMag = Math.hypot(seedFx, seedFy);
  let sumCos = Math.cos(2 * seedTheta) * seedMag;
  let sumSin = Math.sin(2 * seedTheta) * seedMag;
  let runningMag = seedMag;
  let sampleCount = 1;
  const maxSteps = settings.tangentWalkMaxSteps;
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(settings.tangentWalkDeviationDeg));
  const magFraction = settings.tangentWalkMagFraction;
  const grace = settings.tangentWalkGraceSamples;
  for (const sign of [1, -1]) {
    let violations = 0;
    for (let k = 1; k <= maxSteps; k++) {
      const sx = Math.round(x + sign * k * tdx), sy = Math.round(y + sign * k * tdy);
      if (sx < 0 || sx >= w || sy < 0 || sy >= h) break;
      const si = sy * w + sx;
      const sfx = fx[si], sfy = fy[si];
      const mag = Math.hypot(sfx, sfy);
      if (mag === 0 || mag < runningMag * magFraction) {
        violations++;
        if (violations >= grace) break;
        continue;
      }
      const theta = Math.atan2(sfy, sfx);
      const c2 = Math.cos(2 * theta), s2 = Math.sin(2 * theta);
      const avgLen = Math.hypot(sumCos, sumSin);
      const cosDeviation = avgLen > 0 ? (c2 * sumCos + s2 * sumSin) / avgLen : 1;
      if (cosDeviation < devCos) {
        violations++;
        if (violations >= grace) break;
        continue;
      }
      violations = 0;
      sumCos += c2 * mag; sumSin += s2 * mag;
      runningMag = (runningMag * sampleCount + mag) / (sampleCount + 1);
      sampleCount++;
      included.push({ x: sx, y: sy });
    }
  }
  return included;
}

export function computeTangentWalkIncludedPixelsAdaptive(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { x: number; y: number }[] {
  const included: { x: number; y: number }[] = [{ x, y }];
  const seedTheta = Math.atan2(seedFy, seedFx);
  const seedMag = Math.hypot(seedFx, seedFy);
  const seedCos = Math.cos(2 * seedTheta) * seedMag, seedSin = Math.sin(2 * seedTheta) * seedMag;
  const maxSteps = settings.tangentWalkMaxSteps;
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(settings.tangentWalkDeviationDeg));
  const magFraction = settings.tangentWalkMagFraction;
  const grace = settings.tangentWalkGraceSamples;
  for (const sign of [1, -1]) {
    let sumCos = seedCos, sumSin = seedSin, runningMag = seedMag, sampleCount = 1;
    let curX = x, curY = y;
    let violations = 0;
    for (let k = 1; k <= maxSteps; k++) {
      const avgTheta = Math.atan2(sumSin, sumCos) / 2;
      const tdx = -Math.sin(avgTheta), tdy = Math.cos(avgTheta);
      curX += sign * tdx; curY += sign * tdy;
      const sx = Math.round(curX), sy = Math.round(curY);
      if (sx < 0 || sx >= w || sy < 0 || sy >= h) break;
      const si = sy * w + sx;
      const sfx = fx[si], sfy = fy[si];
      const mag = Math.hypot(sfx, sfy);
      if (mag === 0 || mag < runningMag * magFraction) {
        violations++;
        if (violations >= grace) break;
        continue;
      }
      const theta = Math.atan2(sfy, sfx);
      const c2 = Math.cos(2 * theta), s2 = Math.sin(2 * theta);
      const avgLen = Math.hypot(sumCos, sumSin);
      const cosDeviation = avgLen > 0 ? (c2 * sumCos + s2 * sumSin) / avgLen : 1;
      if (cosDeviation < devCos) {
        violations++;
        if (violations >= grace) break;
        continue;
      }
      violations = 0;
      sumCos += c2 * mag; sumSin += s2 * mag;
      runningMag = (runningMag * sampleCount + mag) / (sampleCount + 1);
      sampleCount++;
      included.push({ x: sx, y: sy });
    }
  }
  return included;
}

// Paints the tangent-walk path straight into the camera's own per-field-
// pixel texture (same DataTexture-plus-quad mechanism contamination/top-
// gradient overlays use, see overlays/contaminationOverlays.ts and
// overlays/gradientHighlightOverlays.ts) instead of stroking rects onto
// gradientArrowCanvas at window resolution -- that 2D canvas is sized 1:1 to
// CSS pixels (see ui/layout.ts) with no devicePixelRatio scaling, so it went
// blurry on any HiDPI display; painting one texel per field pixel and
// letting the WebGL quad blit handle the upscale (same as every other field
// overlay) fixes that by construction, no rect/cellW/cellH mapping needed.
export function clearTangentWalkPathOverlay(camera: Camera) {
  camera.tangentWalkPathData.fill(0);
  camera.tangentWalkPathTex.needsUpdate = true;
}
export function paintTangentWalkPathOverlay(
  camera: Camera, fieldW: number, fx: Float64Array, fy: Float64Array, included: { x: number; y: number }[],
) {
  const data = camera.tangentWalkPathData;
  for (let idx = 0; idx < included.length; idx++) {
    const { x: fc, y: fr } = included[idx];
    const o = (fr * fieldW + fc) * 4;
    if (idx === 0) {
      const si = fr * fieldW + fc;
      let hueTheta = Math.atan2(fy[si], fx[si]);
      if (hueTheta < 0) hueTheta += Math.PI;
      if (hueTheta >= Math.PI) hueTheta -= Math.PI;
      const [rr, gg, bb] = hsvToRgb((hueTheta / Math.PI) * 360, 1, 1);
      data[o] = 255 - rr; data[o + 1] = 255 - gg; data[o + 2] = 255 - bb; data[o + 3] = 255;
    } else {
      data[o] = 0; data[o + 1] = 0; data[o + 2] = 0; data[o + 3] = 200;
    }
  }
  camera.tangentWalkPathTex.needsUpdate = true;
}

// Same idea, for the 2-axis/4-spoke jacobian walk (see
// pipeline/localJacobian.ts) -- axis 1 (dominant eigenvector, "across the
// edge") in the same orange used for its arrow, axis 2 (subordinate, "along
// the edge") in the matching cyan, seed pixel in white.
export function paintSpokedWalkOverlay(camera: Camera, fieldW: number, included: { x: number; y: number; axis: 1 | 2 }[]) {
  const data = camera.tangentWalkPathData;
  for (let idx = 0; idx < included.length; idx++) {
    const { x: fc, y: fr, axis } = included[idx];
    const o = (fr * fieldW + fc) * 4;
    if (idx === 0) {
      data[o] = 255; data[o + 1] = 255; data[o + 2] = 255; data[o + 3] = 255;
    } else if (axis === 1) {
      data[o] = 255; data[o + 1] = 120; data[o + 2] = 0; data[o + 3] = 200;
    } else {
      data[o] = 0; data[o + 1] = 200; data[o + 2] = 255; data[o + 3] = 200;
    }
  }
  camera.tangentWalkPathTex.needsUpdate = true;
}

// Draws camera.lastVoteComposites -- the composite lines actually fed to
// fitPairOfPlanes and classified by pipeline/gridPeriodPhase.ts (pipeline/
// votes.ts's computeGradient2x2Composites). Drawing this exact same array
// (rather than an independently-recomputed copy) guarantees a line's root
// here means the same thing as it does in gpp's own row/col maps.
//
// Family coloring (pipeline/gridPeriodPhase.ts): blue = row family, red =
// column family, black -> full-color by each line's own RANK within its
// family (sorted by its rectified `value` -- the same order the
// period/phase fit itself assigns integer indices in), so this literally
// shows the sequence the fit will register each line as. Gray for any line
// gridPeriodPhase itself skipped (e.g. a degenerate gnomonic projection).
// Draws camera.lastVoteComposites -- the composite lines actually fed to
// fitPairOfPlanes and classified by pipeline/gridPeriodPhase.ts (pipeline/
// votes.ts's computeGradient2x2Composites) -- as SVG lines in
// lsdCompositeGroup. Populated once per REAL capture (not live-recomputed
// per LSD slider tweak, unlike the rectangle/rejected/raw-region views in
// overlays/lsdOverlay.ts): lastVoteComposites' pixel coords come from the
// row-flipped `gray` axesReconstruction.ts feeds the vote/grid-period-phase
// pipeline (its own flipRowsF64, kept separate from
// camera.lastNoisedPreviewGray), and recomputing that same row-flip
// live from the raw preview would risk exactly the kind of "two different
// computations of the same thing" mismatch a past version of this file
// already had and fixed -- see this file's git history.
//
// Two color modes over the SAME line set, not two different line sets:
// default is a unique per-line color hashed from that line's own merge-
// group root (overlays/lsdOverlay.ts's hashSeedIndexToHueDeg); toggling
// showCompositeLineFamilies switches every line to blue=row family /
// red=column family instead, shaded by each line's own RANK within its
// family (sorted by its rectified `value` -- the same order the
// period/phase fit itself assigns integer indices in) -- gray for any line
// gridPeriodPhase itself skipped (e.g. a degenerate gnomonic projection).
function drawCompositeLines(camera: Camera) {
  while (lsdCompositeGroup.firstChild) lsdCompositeGroup.removeChild(lsdCompositeGroup.firstChild);
  const settings = camera.settings;
  if (!settings.showLsdComposite || !camera.lastVoteComposites) return;
  const rect = computeThroughRect(camera);
  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const toScreen = (fx: number, fy: number) => {
    const rasterY = fieldH - 1 - fy;
    return {
      x: rect.x + (fx + 0.5) * (rect.w / fieldW),
      y: rect.y + rect.h - (rasterY + 0.5) * (rect.h / fieldH),
    };
  };

  let rowRank: Map<number, number> | null = null, colRank: Map<number, number> | null = null;
  const gpp = camera.lastGridPeriodPhase;
  if (settings.showCompositeLineFamilies && gpp) {
    const sortedRow = [...gpp.rowLines].sort((a, b) => a.value - b.value);
    rowRank = new Map(sortedRow.map((s, i) => [s.root, sortedRow.length > 1 ? i / (sortedRow.length - 1) : 1]));
    const sortedCol = [...gpp.colLines].sort((a, b) => a.value - b.value);
    colRank = new Map(sortedCol.map((s, i) => [s.root, sortedCol.length > 1 ? i / (sortedCol.length - 1) : 1]));
  }

  for (const { root, line } of camera.lastVoteComposites) {
    const a = toScreen(line.x1, line.y1), b = toScreen(line.x2, line.y2);
    let strokeColor: string;
    if (rowRank || colRank) {
      if (rowRank && rowRank.has(root)) strokeColor = `rgb(0,0,${Math.round(rowRank.get(root)! * 255)})`;
      else if (colRank && colRank.has(root)) strokeColor = `rgb(${Math.round(colRank.get(root)! * 255)},0,0)`;
      else strokeColor = 'rgb(150,150,150)';
    } else {
      const [hr, hg, hb] = hsvToRgb(hashSeedIndexToHueDeg(root), 0.85, 1);
      strokeColor = `rgb(${hr},${hg},${hb})`;
    }
    lsdCompositeGroup.appendChild(svgEl('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: 'rgba(0,0,0,0.85)', 'stroke-width': 5 }));
    lsdCompositeGroup.appendChild(svgEl('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: strokeColor, 'stroke-width': 2.5 }));
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
  if (!camera) { clearGradientArrowOverlay(); return; }
  const settings = camera.settings;
  const arrowsOn = settings.showGradientArrow || settings.showGradientArrowPerpendicular;

  clearGradientArrowOverlay();

  if (globalState.mode !== 'through') return;

  drawCompositeLines(camera);

  if (!arrowsOn) return;
  const rect = computeThroughRect(camera);
  if (clientX < rect.x || clientX >= rect.x + rect.w || clientY < rect.y || clientY >= rect.y + rect.h) return; // cursor outside -- markers (if any) stay drawn, just no hover-specific content below

  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const nx = (clientX - rect.x) / rect.w, ny = (clientY - rect.y) / rect.h;
  const fieldCol = Math.min(fieldW - 1, Math.max(0, Math.floor(nx * fieldW)));
  const fieldRow = Math.min(fieldH - 1, Math.max(0, Math.floor((1 - ny) * fieldH)));
  const i = fieldRow * fieldW + fieldCol;

  if (arrowsOn && camera.lastDisplayedVectorField) {
    const { fx, fy } = camera.lastDisplayedVectorField;
    const gx = fx[i], gy = fy[i];
    const mag = Math.hypot(gx, gy);
    if (mag > 0) {
      const px = rect.x + (fieldCol + 0.5) * (rect.w / fieldW);
      const py = rect.y + rect.h - (fieldRow + 0.5) * (rect.h / fieldH);
      let hueTheta = Math.atan2(gy, gx);
      if (hueTheta < 0) hueTheta += Math.PI;
      if (hueTheta >= Math.PI) hueTheta -= Math.PI;
      const [rr, gg, bb] = hsvToRgb((hueTheta / Math.PI) * 360, 1, 1);
      const color = `rgb(${rr},${gg},${bb})`;

      if (settings.showGradientArrow) {
        const theta = Math.atan2(gy, gx);
        drawOneArrow(px, py, Math.cos(theta) * mag, -Math.sin(theta) * mag, color, settings.gradientArrowScale);
      }
      if (settings.showGradientArrowPerpendicular) {
        const theta = Math.atan2(gx, -gy);
        drawOneArrow(px, py, Math.cos(theta) * mag, -Math.sin(theta) * mag, color, settings.gradientArrowScale);
      }
    }
  }
}
export let lastHoverClientX = -1, lastHoverClientY = -1;
canvas.addEventListener('pointermove', (e) => {
  lastHoverClientX = e.clientX; lastHoverClientY = e.clientY;
  updateHoverOverlays(e.clientX, e.clientY);
});
canvas.addEventListener('pointerleave', () => {
  lastHoverClientX = -1; lastHoverClientY = -1;
  clearGradientArrowOverlay();
});

toggleHideFieldBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.hideField = !cam.settings.hideField;
  toggleHideFieldBtn.classList.toggle('active', cam.settings.hideField);
  updateDistortedPreview(cam);
  updateContaminationOverlays(cam);
});
toggleTrueContamBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showTrueContamination = !cam.settings.showTrueContamination;
  toggleTrueContamBtn.classList.toggle('active', cam.settings.showTrueContamination);
  updateDistortedPreview(cam);
  updateContaminationOverlays(cam);
});
toggleReconContamBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showReconstructedContamination = !cam.settings.showReconstructedContamination;
  toggleReconContamBtn.classList.toggle('active', cam.settings.showReconstructedContamination);
  updateDistortedPreview(cam);
  updateContaminationOverlays(cam);
});
toggleTopGradientBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showTopGradient = !cam.settings.showTopGradient;
  toggleTopGradientBtn.classList.toggle('active', cam.settings.showTopGradient);
  updateDistortedPreview(cam);
  updateTopGradientOverlay(cam);
});
// These 3 (rectangles/rejected/raw-regions) no longer touch the shared
// gradientArrowCtx canvas at all -- rectangles draw into their own SVG
// group and raw-regions/rejected paint their own raster textures, both
// handled entirely inside updateLsdOverlay -- so no updateHoverOverlays
// call is needed after any of them.
toggleLsdSegmentsBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLsdSegments = !cam.settings.showLsdSegments;
  toggleLsdSegmentsBtn.classList.toggle('active', cam.settings.showLsdSegments);
  persistControl('toggleLsdSegments', cam.settings.showLsdSegments ? '1' : '0');
  updateLsdOverlay(cam);
});
toggleLsdRejectedBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLsdRejected = !cam.settings.showLsdRejected;
  toggleLsdRejectedBtn.classList.toggle('active', cam.settings.showLsdRejected);
  persistControl('toggleLsdRejected', cam.settings.showLsdRejected ? '1' : '0');
  updateLsdOverlay(cam); // may be the first of the 3 toggles turned on
});
toggleLsdRawRegionsBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLsdRawRegions = !cam.settings.showLsdRawRegions;
  toggleLsdRawRegionsBtn.classList.toggle('active', cam.settings.showLsdRawRegions);
  persistControl('toggleLsdRawRegions', cam.settings.showLsdRawRegions ? '1' : '0');
  updateLsdOverlay(cam);
});
toggleLsdCompositeBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showLsdComposite = !cam.settings.showLsdComposite;
  toggleLsdCompositeBtn.classList.toggle('active', cam.settings.showLsdComposite);
  persistControl('toggleLsdComposite', cam.settings.showLsdComposite ? '1' : '0');
  drawCompositeLines(cam);
});
toggleCompositeLineFamiliesBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showCompositeLineFamilies = !cam.settings.showCompositeLineFamilies;
  toggleCompositeLineFamiliesBtn.classList.toggle('active', cam.settings.showCompositeLineFamilies);
  persistControl('toggleCompositeLineFamilies', cam.settings.showCompositeLineFamilies ? '1' : '0');
  drawCompositeLines(cam); // recolors the SAME lines, doesn't recompute them
});
toggleGradientArrowBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showGradientArrow = !cam.settings.showGradientArrow;
  toggleGradientArrowBtn.classList.toggle('active', cam.settings.showGradientArrow);
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
toggleGradientArrowModeBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showGradientArrowPerpendicular = !cam.settings.showGradientArrowPerpendicular;
  toggleGradientArrowModeBtn.classList.toggle('active', cam.settings.showGradientArrowPerpendicular);
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


export function updateGradientArrowAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.fieldView === 'gradient2x2';
  toggleGradientArrowBtn.disabled = !relevant;
  toggleGradientArrowModeBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showGradientArrow = false;
    cam.settings.showGradientArrowPerpendicular = false;
    toggleGradientArrowBtn.classList.remove('active');
    toggleGradientArrowModeBtn.classList.remove('active');
    clearGradientArrowOverlay();
  }
}
