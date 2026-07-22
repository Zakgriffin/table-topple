import * as THREE from 'three';
import { GradientField } from '../types.ts';
import { hsvToRgb, separableBoxBlur } from './distortion.ts';

// ── Value fields (no color) ─────────────────────────────────────────────

export function computeGradientField(gray: Float64Array, w: number, h: number, gradRadius: number): GradientField {
  const r = gradRadius;
  const fx = new Float64Array(w * h), fy = new Float64Array(w * h);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      fx[i] = gray[i + r] - gray[i - r];
      fy[i] = gray[i + r * w] - gray[i - r * w];
    }
  }
  return { fx, fy, w, h, r };
}

// Quick throwaway comparison view -- forward difference over a bare 2x2
// window (this pixel, one right, one down) instead of the radius-driven
// centered difference above.
export function computeGradient2x2Field(gray: Float64Array, w: number, h: number): GradientField {
  const fx = new Float64Array(w * h), fy = new Float64Array(w * h);
  for (let y = 0; y < h - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const i = y * w + x;
      fx[i] = gray[i + 1] - gray[i];
      fy[i] = gray[i + w] - gray[i];
    }
  }
  return { fx, fy, w, h, r: 1 };
}

// Magnitude of the local VECTOR SUM of gradients (double-angle folded so
// alternating-polarity edges reinforce instead of cancelling) -- see
// pre-Stage-A history for the full derivation. Normalized against this
// frame's own RAW (unsmoothed) max magnitude.
export function computeGradientAgreementField(field: GradientField, aggRadius: number): Float64Array {
  const { fx, fy, w, h } = field;
  const n = w * h;
  const cx = new Float64Array(n), cy = new Float64Array(n);
  let maxRawMag = 0;
  for (let i = 0; i < n; i++) {
    const mag = Math.hypot(fx[i], fy[i]);
    if (mag > maxRawMag) maxRawMag = mag;
    if (mag === 0) continue;
    const theta = Math.atan2(fy[i], fx[i]);
    cx[i] = mag * Math.cos(2 * theta);
    cy[i] = mag * Math.sin(2 * theta);
  }
  const sx = separableBoxBlur(cx, w, h, aggRadius);
  const sy = separableBoxBlur(cy, w, h, aggRadius);
  const agreement = new Float64Array(n);
  for (let i = 0; i < n; i++) agreement[i] = Math.hypot(sx[i], sy[i]);
  if (maxRawMag > 0) for (let i = 0; i < n; i++) agreement[i] /= maxRawMag;
  return agreement;
}

export function computeEffectiveGradientField(field: GradientField, agreement: Float64Array): GradientField {
  const { fx, fy, w, h, r } = field;
  const n = w * h;
  const efx = new Float64Array(n), efy = new Float64Array(n);
  for (let i = 0; i < n; i++) { efx[i] = fx[i] * agreement[i]; efy[i] = fy[i] * agreement[i]; }
  return { fx: efx, fy: efy, w, h, r };
}

// Triangular fold of a grayscale buffer (0..maxVal, see toGrayscale/
// fillGrayscalePreview): pixels in the lower half keep their exact
// brightness, pixels in the upper half mirror back down toward 0 at maxVal
// (slope -1 past the midpoint) -- 0 and maxVal both fold to 0, maxVal/2
// stays at maxVal/2. Applied to whichever grayscale is about to feed the
// gradient pipeline (see paintFieldViewFromGray's callers), so it's the same
// source both camera types already use for that pipeline, not a distortion
// stage of its own.
export function computeTriangleFold(gray: Float64Array, maxVal = 255): Float64Array {
  const half = maxVal / 2;
  const out = new Float64Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    const v = gray[i];
    out[i] = v <= half ? v : maxVal - v;
  }
  return out;
}

// ── Display: colorizes a value field, only for whichever one is on screen ─

export function paintVectorFieldAsColor(field: GradientField, out: Uint8Array) {
  const { fx, fy, w, h } = field;
  const n = w * h;
  const mags = new Float64Array(n);
  let maxMag = 0;
  for (let i = 0; i < n; i++) {
    const mag = Math.hypot(fx[i], fy[i]);
    mags[i] = mag;
    if (mag > maxMag) maxMag = mag;
  }
  for (let i = 0; i < n; i++) {
    let theta = Math.atan2(fy[i], fx[i]);
    if (theta < 0) theta += Math.PI;
    if (theta >= Math.PI) theta -= Math.PI;
    const sat = maxMag > 0 ? mags[i] / maxMag : 0;
    const [rr, gg, bb] = hsvToRgb((theta / Math.PI) * 360, sat, 1);
    const o = i * 4;
    out[o] = rr; out[o + 1] = gg; out[o + 2] = bb; out[o + 3] = 255;
  }
}

export function paintScalarFieldAsGray(field: Float64Array, out: Uint8Array) {
  for (let i = 0; i < field.length; i++) {
    const v = Math.round(THREE.MathUtils.clamp(field[i], 0, 1) * 255);
    const o = i * 4;
    out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
  }
}

export function fillGrayscalePreview(gray: Float64Array, out: Uint8Array) {
  for (let i = 0; i < gray.length; i++) {
    const v = Math.max(0, Math.min(255, gray[i]));
    const o = i * 4;
    out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
  }
}

