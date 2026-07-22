import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { CameraSettingsCommon } from '../camera/settings.ts';
import { activeCamera } from '../camera/store.ts';
import { segmentLength } from '../pipeline/bucketFillSegments.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { computeSpokedWalkIncludedPixels } from '../pipeline/localJacobian.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { globalState } from '../state.ts';
import { canvas, gradientArrowCanvas, gradientArrowCtx, persistControl, toggleBucketFillBtn, toggleBucketFillCompositeBtn, toggleBucketFillJoinBtn, toggleBucketFillMarkersBtn, toggleGradientArrowBtn, toggleGradientArrowModeBtn, toggleHideFieldBtn, toggleReconContamBtn, toggleTangentWalkPathBtn, toggleTopGradientBtn, toggleTrueContamBtn } from '../ui/dom.ts';
import { updateBucketFillOverlay } from './bucketFillOverlay.ts';
import { updateBucketFillCompositeAvailability, updateBucketFillJoinAvailability, updateBucketFillJoinOverlay } from './bucketFillJoinOverlay.ts';
import { updateContaminationOverlays } from './contaminationOverlays.ts';
import { updateTopGradientOverlay } from './gradientHighlightOverlays.ts';

// Letterbox rect for the fixed-aspect gizmo camera within whatever shape the
// window currently is.
export function computeThroughRect(camera: Camera): { x: number; y: number; w: number; h: number } {
  const winAspect = innerWidth / innerHeight;
  let w = innerWidth, h = innerHeight, x = 0, y = 0;
  if (winAspect > camera.aspect) { w = innerHeight * camera.aspect; x = (innerWidth - w) / 2; }
  else { h = innerWidth / camera.aspect; y = (innerHeight - h) / 2; }
  return { x, y, w, h };
}

export function clearGradientArrowOverlay() {
  gradientArrowCtx.clearRect(0, 0, gradientArrowCanvas.width, gradientArrowCanvas.height);
}
// Same black-outline-then-colored-fill dot drawOneArrow plants at its own
// origin point -- extracted so other markers (bucket-fill endpoints) can
// reuse the exact same visual language without also drawing a shaft/arrow.
export function drawMarkerDot(px: number, py: number, color: string) {
  const ctx = gradientArrowCtx;
  ctx.fillStyle = 'rgba(0,0,0,0.85)';
  ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
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

// Two dots per bucket-fill segment (see pipeline/bucketFillSegments.ts and
// overlays/bucketFillOverlay.ts) -- one at each of the segment's tracked
// endpoints (endAlong/endAgainst), same dot styling the center-of-mass
// marker used before. Colored to match that segment's own random fill
// color, so a marker can be visually traced back to its blob in the
// bucket-fill raster overlay underneath. Unlike the hover arrows below,
// this is PERSISTENT (tied to whenever segments were last recomputed, not
// to cursor position) -- see this function's caller for why it still lives
// inside the same clear-then-redraw cycle as the hover-only content.
function drawBucketFillSegmentMarkers(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showBucketFillSegments || !settings.showBucketFillMarkers || !camera.lastBucketFillSegments || !camera.lastBucketFillColors) return;
  const rect = computeThroughRect(camera);
  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const segments = camera.lastBucketFillSegments;
  const colors = camera.lastBucketFillColors;
  const toScreen = (fx: number, fy: number) => ({
    px: rect.x + (fx + 0.5) * (rect.w / fieldW),
    py: rect.y + rect.h - (fy + 0.5) * (rect.h / fieldH),
  });
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (segmentLength(seg) < settings.bucketFillMinLengthPx) continue; // too short -- disappears the same as in the base raster
    const [rr, gg, bb] = colors[i];
    const color = `rgb(${rr},${gg},${bb})`;
    const a = toScreen(seg.endAlongX, seg.endAlongY);
    const b = toScreen(seg.endAgainstX, seg.endAgainstY);
    drawMarkerDot(a.px, a.py, color);
    drawMarkerDot(b.px, b.py, color);
  }
}

// One line per merge GROUP (see pipeline/bucketFillJoin.ts's
// computeCompositeLines) -- the two farthest-apart endpoints across every
// segment the join walk decided belongs to that group, i.e. the group's own
// "composite" line, not any one member segment's own extent. Same black-
// outline-then-colored-stroke language as drawOneArrow/drawMarkerDot, in
// the group's own blended color (matching the join overlay's raster).
function drawBucketFillCompositeLines(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showBucketFillComposite || !camera.lastBucketFillComposite) return;
  const rect = computeThroughRect(camera);
  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const toScreen = (fx: number, fy: number) => ({
    px: rect.x + (fx + 0.5) * (rect.w / fieldW),
    py: rect.y + rect.h - (fy + 0.5) * (rect.h / fieldH),
  });
  const ctx = gradientArrowCtx;
  for (const c of camera.lastBucketFillComposite) {
    const a = toScreen(c.x1, c.y1), b = toScreen(c.x2, c.y2);
    ctx.strokeStyle = 'rgba(0,0,0,0.85)'; ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py); ctx.stroke();
    const [rr, gg, bb] = c.color;
    ctx.strokeStyle = `rgb(${rr},${gg},${bb})`; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py); ctx.stroke();
  }
}

// Single per-hover entry point -- operates on the ACTIVE camera, since only
// its Through-Cam view is ever on screen. Also doubles as the redraw
// entry point for the PERSISTENT bucket-fill segment markers above (see
// their callers in overlays/bucketFillOverlay.ts's own callers) -- they
// don't depend on cursor position, but share this canvas/clear cycle with
// the cursor-driven content below, so anything that recomputes them re-
// invokes this function (with the last-known cursor position) to get them
// back on screen without waiting for the next pointermove.
export function updateHoverOverlays(clientX: number, clientY: number) {
  const camera = activeCamera();
  if (!camera) { clearGradientArrowOverlay(); return; }
  const settings = camera.settings;
  const arrowsOn = settings.showGradientArrow || settings.showGradientArrowPerpendicular;
  const walkOn = settings.showTangentWalkPath;
  const markersOn = settings.showBucketFillSegments;

  clearGradientArrowOverlay();
  clearTangentWalkPathOverlay(camera);

  if (globalState.mode !== 'through') return;

  if (markersOn) drawBucketFillSegmentMarkers(camera);
  if (settings.showBucketFillComposite) drawBucketFillCompositeLines(camera);

  if (!arrowsOn && !walkOn) return;
  const rect = computeThroughRect(camera);
  if (clientX < rect.x || clientX >= rect.x + rect.w || clientY < rect.y || clientY >= rect.y + rect.h) return; // cursor outside -- markers (if any) stay drawn, just no hover-specific content below

  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const nx = (clientX - rect.x) / rect.w, ny = (clientY - rect.y) / rect.h;
  const fieldCol = Math.min(fieldW - 1, Math.max(0, Math.floor(nx * fieldW)));
  const fieldRow = Math.min(fieldH - 1, Math.max(0, Math.floor((1 - ny) * fieldH)));
  const i = fieldRow * fieldW + fieldCol;

  if (settings.fieldView === 'jacobian' && camera.lastJacobianField) {
    const jac = camera.lastJacobianField;
    const { e1x: E1x, e1y: E1y, lambda1, e2x: E2x, e2y: E2y, lambda2 } = jac;
    const e1x = E1x[i], e1y = E1y[i], e2x = E2x[i], e2y = E2y[i];
    const hasStructure = lambda1[i] !== 0 || lambda2[i] !== 0;

    if (arrowsOn && hasStructure) {
      const px = rect.x + (fieldCol + 0.5) * (rect.w / fieldW);
      const py = rect.y + rect.h - (fieldRow + 0.5) * (rect.h / fieldH);
      // e1 (dominant, "across the edge") in orange, e2 (subordinate, "along
      // the edge") in cyan -- matches drawSpokedWalkOutline's colors below.
      if (settings.showGradientArrow) {
        const m = Math.abs(lambda1[i]);
        drawOneArrow(px, py, e1x * m, -e1y * m, 'rgb(255,120,0)', settings.gradientArrowScale);
      }
      if (settings.showGradientArrowPerpendicular) {
        const m = Math.abs(lambda2[i]);
        drawOneArrow(px, py, e2x * m, -e2y * m, 'rgb(0,200,255)', settings.gradientArrowScale);
      }
    }

    if (walkOn && hasStructure && camera.lastDisplayedVectorField) {
      const { fx, fy } = camera.lastDisplayedVectorField; // the raw field the Jacobian was built from
      const included = computeSpokedWalkIncludedPixels(settings, fx, fy, fieldW, fieldH, fieldCol, fieldRow, e1x, e1y, e2x, e2y);
      paintSpokedWalkOverlay(camera, fieldW, included);
    }
  } else {
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

    if (walkOn && camera.lastEffectiveField) {
      const { fx, fy } = camera.lastEffectiveField;
      const seedFx = fx[i], seedFy = fy[i];
      if (seedFx !== 0 || seedFy !== 0) {
        const included = settings.tangentWalkAdaptive
          ? computeTangentWalkIncludedPixelsAdaptive(settings, fx, fy, fieldW, fieldH, fieldCol, fieldRow, seedFx, seedFy)
          : computeTangentWalkIncludedPixels(settings, fx, fy, fieldW, fieldH, fieldCol, fieldRow, seedFx, seedFy);
        paintTangentWalkPathOverlay(camera, fieldW, fx, fy, included);
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
  const cam = activeCamera(); if (cam) clearTangentWalkPathOverlay(cam);
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
toggleBucketFillBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showBucketFillSegments = !cam.settings.showBucketFillSegments;
  toggleBucketFillBtn.classList.toggle('active', cam.settings.showBucketFillSegments);
  persistControl('toggleBucketFill', cam.settings.showBucketFillSegments ? '1' : '0');
  updateDistortedPreview(cam);
  updateBucketFillOverlay(cam);
  updateBucketFillJoinAvailability(); // the join button's enabled state depends on this toggle -- refresh it now, not just on fieldView change
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
toggleBucketFillMarkersBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showBucketFillMarkers = !cam.settings.showBucketFillMarkers;
  toggleBucketFillMarkersBtn.classList.toggle('active', cam.settings.showBucketFillMarkers);
  persistControl('toggleBucketFillMarkers', cam.settings.showBucketFillMarkers ? '1' : '0');
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
toggleBucketFillJoinBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showBucketFillJoin = !cam.settings.showBucketFillJoin;
  toggleBucketFillJoinBtn.classList.toggle('active', cam.settings.showBucketFillJoin);
  persistControl('toggleBucketFillJoin', cam.settings.showBucketFillJoin ? '1' : '0');
  updateBucketFillJoinOverlay(cam);
  updateBucketFillCompositeAvailability(); // composite's enabled state depends on this toggle -- refresh it now, not just on some other trigger
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
toggleBucketFillCompositeBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showBucketFillComposite = !cam.settings.showBucketFillComposite;
  toggleBucketFillCompositeBtn.classList.toggle('active', cam.settings.showBucketFillComposite);
  persistControl('toggleBucketFillComposite', cam.settings.showBucketFillComposite ? '1' : '0');
  updateBucketFillJoinOverlay(cam); // composites are computed inside this, gated on the toggle just flipped
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
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
toggleTangentWalkPathBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showTangentWalkPath = !cam.settings.showTangentWalkPath;
  toggleTangentWalkPathBtn.classList.toggle('active', cam.settings.showTangentWalkPath);
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});


export function updateGradientArrowAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.fieldView === 'gradient' || cam.settings.fieldView === 'gradient2x2' || cam.settings.fieldView === 'effective' || cam.settings.fieldView === 'walked' || cam.settings.fieldView === 'jacobian';
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
export function updateTangentWalkPathAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.fieldView === 'effective' || cam.settings.fieldView === 'walked' || cam.settings.fieldView === 'jacobian';
  toggleTangentWalkPathBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showTangentWalkPath = false;
    toggleTangentWalkPathBtn.classList.remove('active');
    clearGradientArrowOverlay();
    clearTangentWalkPathOverlay(cam);
  }
}
