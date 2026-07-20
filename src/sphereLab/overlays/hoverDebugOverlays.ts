import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { CameraSettingsCommon } from '../camera/settings.ts';
import { activeCamera } from '../camera/store.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { computeSpokedWalkIncludedPixels } from '../pipeline/localJacobian.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { globalState } from '../state.ts';
import { canvas, gradientArrowCanvas, gradientArrowCtx, toggleGradientArrowBtn, toggleGradientArrowModeBtn, toggleHideFieldBtn, toggleReconContamBtn, toggleTangentWalkPathBtn, toggleTrueContamBtn } from '../ui/dom.ts';
import { updateContaminationOverlays } from './contaminationOverlays.ts';

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

export function drawTangentWalkOutline(
  rect: { x: number; y: number; w: number; h: number }, fieldW: number, fieldH: number,
  fx: Float64Array, fy: Float64Array, included: { x: number; y: number }[],
) {
  const cellW = rect.w / fieldW, cellH = rect.h / fieldH;
  const ctx = gradientArrowCtx;
  for (let idx = 0; idx < included.length; idx++) {
    const { x: fc, y: fr } = included[idx];
    const boxLeft = rect.x + fc * cellW;
    const boxTop = rect.y + rect.h - (fr + 1) * cellH;
    const isSeed = idx === 0;
    if (isSeed) {
      const si = fr * fieldW + fc;
      let hueTheta = Math.atan2(fy[si], fx[si]);
      if (hueTheta < 0) hueTheta += Math.PI;
      if (hueTheta >= Math.PI) hueTheta -= Math.PI;
      const [rr, gg, bb] = hsvToRgb((hueTheta / Math.PI) * 360, 1, 1);
      ctx.strokeStyle = `rgb(${255 - rr},${255 - gg},${255 - bb})`;
      ctx.lineWidth = 2.5;
    } else {
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1.5;
    }
    ctx.strokeRect(boxLeft + 0.5, boxTop + 0.5, Math.max(1, cellW - 1), Math.max(1, cellH - 1));
  }
}

// Same idea as drawTangentWalkOutline, but for the 2-axis/4-spoke jacobian
// walk (see pipeline/localJacobian.ts) -- axis 1 (dominant eigenvector,
// "across the edge") outlined in the same orange used for its arrow, axis 2
// (subordinate, "along the edge") in the matching cyan, seed pixel in white.
export function drawSpokedWalkOutline(
  rect: { x: number; y: number; w: number; h: number }, fieldW: number, fieldH: number,
  included: { x: number; y: number; axis: 1 | 2 }[],
) {
  const cellW = rect.w / fieldW, cellH = rect.h / fieldH;
  const ctx = gradientArrowCtx;
  for (let idx = 0; idx < included.length; idx++) {
    const { x: fc, y: fr, axis } = included[idx];
    const boxLeft = rect.x + fc * cellW;
    const boxTop = rect.y + rect.h - (fr + 1) * cellH;
    const isSeed = idx === 0;
    if (isSeed) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2.5;
    } else {
      ctx.strokeStyle = axis === 1 ? 'rgb(255,120,0)' : 'rgb(0,200,255)';
      ctx.lineWidth = 1.5;
    }
    ctx.strokeRect(boxLeft + 0.5, boxTop + 0.5, Math.max(1, cellW - 1), Math.max(1, cellH - 1));
  }
}

// Single per-hover entry point -- operates on the ACTIVE camera, since only
// its Through-Cam view is ever on screen.
export function updateHoverOverlays(clientX: number, clientY: number) {
  const camera = activeCamera();
  if (!camera) { clearGradientArrowOverlay(); return; }
  const settings = camera.settings;
  const arrowsOn = settings.showGradientArrow || settings.showGradientArrowPerpendicular;
  const walkOn = settings.showTangentWalkPath;
  if (globalState.mode !== 'through' || (!arrowsOn && !walkOn)) { clearGradientArrowOverlay(); return; }
  const rect = computeThroughRect(camera);
  if (clientX < rect.x || clientX >= rect.x + rect.w || clientY < rect.y || clientY >= rect.y + rect.h) { clearGradientArrowOverlay(); return; }

  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const nx = (clientX - rect.x) / rect.w, ny = (clientY - rect.y) / rect.h;
  const fieldCol = Math.min(fieldW - 1, Math.max(0, Math.floor(nx * fieldW)));
  const fieldRow = Math.min(fieldH - 1, Math.max(0, Math.floor((1 - ny) * fieldH)));
  const i = fieldRow * fieldW + fieldCol;

  clearGradientArrowOverlay();

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
      drawSpokedWalkOutline(rect, fieldW, fieldH, included);
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
        drawTangentWalkOutline(rect, fieldW, fieldH, fx, fy, included);
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
  const relevant = cam.settings.fieldView === 'gradient' || cam.settings.fieldView === 'effective' || cam.settings.fieldView === 'walked' || cam.settings.fieldView === 'jacobian';
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
  }
}
