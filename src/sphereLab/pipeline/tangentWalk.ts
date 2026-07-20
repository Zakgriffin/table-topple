import * as THREE from 'three';
import { CameraSettingsCommon } from '../camera/settings.ts';
import { GradientField } from '../types.ts';

// ── Guided tangent walk ──────────────────────────────────────────────────
//
// Fixed-direction walk: seeded once from the seed pixel's own gradient, not
// adaptively re-steered. Every tunable comes from `settings` (the active
// camera's own CameraSettingsCommon) now, instead of a module-level `state`.
export function guidedTangentDirection(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { fx: number; fy: number } {
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
    }
  }
  const avgTheta = Math.atan2(sumSin, sumCos) / 2;
  return { fx: Math.cos(avgTheta) * seedMag, fy: Math.sin(avgTheta) * seedMag };
}

// Adaptive variant -- re-steers at every step using the CURRENT running-
// average direction instead of always sampling along a fixed straight line
// from the seed. settings.tangentWalkAdaptive toggles between the two.
export function guidedTangentDirectionAdaptive(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { fx: number; fy: number } {
  const seedTheta = Math.atan2(seedFy, seedFx);
  const seedMag = Math.hypot(seedFx, seedFy);
  const seedCos = Math.cos(2 * seedTheta) * seedMag, seedSin = Math.sin(2 * seedTheta) * seedMag;
  const maxSteps = settings.tangentWalkMaxSteps;
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(settings.tangentWalkDeviationDeg));
  const magFraction = settings.tangentWalkMagFraction;
  const grace = settings.tangentWalkGraceSamples;

  let totalCos = 0, totalSin = 0;
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
    }
    totalCos += sumCos - seedCos;
    totalSin += sumSin - seedSin;
  }
  totalCos += seedCos; totalSin += seedSin;
  const avgTheta = Math.atan2(totalSin, totalCos) / 2;
  return { fx: Math.cos(avgTheta) * seedMag, fy: Math.sin(avgTheta) * seedMag };
}

// Single dispatch point used by every REAL (non-diagnostic) caller.
export function guidedTangentDirectionForWalk(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { fx: number; fy: number } {
  return settings.tangentWalkAdaptive
    ? guidedTangentDirectionAdaptive(settings, fx, fy, w, h, x, y, seedFx, seedFy)
    : guidedTangentDirection(settings, fx, fy, w, h, x, y, seedFx, seedFy);
}

export function computeWalkedGradientField(settings: CameraSettingsCommon, field: GradientField): GradientField {
  const { fx, fy, w, h, r } = field;
  const walkedFx = new Float64Array(fx.length), walkedFy = new Float64Array(fy.length);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      if (fx[i] === 0 && fy[i] === 0) continue;
      const walked = guidedTangentDirectionForWalk(settings, fx, fy, w, h, x, y, fx[i], fy[i]);
      walkedFx[i] = walked.fx; walkedFy[i] = walked.fy;
    }
  }
  return { fx: walkedFx, fy: walkedFy, w, h, r };
}
