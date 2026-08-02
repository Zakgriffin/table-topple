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

// LSD's own gradient definition (von Gioi et al. 2010): for the 2x2 block
// at (x,y)-(x+1,y+1), gx is the average of the block's two HORIZONTAL edge
// differences and gy the average of its two VERTICAL edge differences --
// uses all 4 corners of the block for both components, not just one edge of
// it (a bare forward difference, this function's previous behavior, reads
// only 2 of the 4 corners per component and ignores the other edge
// entirely). Leaves the last row/column at zero (no full 2x2 block there),
// same footprint as before.
//
// Normalized by GRAYSCALE_MAX so hypot(fx,fy)'s own true ceiling is exactly
// 1, not 255: fx/fy are two ORTHOGONAL unit-norm combinations of the same 4
// corner pixels, so (being a convex quadratic over a box) the max of
// hypot(fx,fy) is attained at a corner-pixel vertex (each corner at 0 or
// GRAYSCALE_MAX) -- enumerating those, the true max is GRAYSCALE_MAX itself
// (one column of the block fully black, the other fully white), reached
// BEFORE this division. This makes `rho`/lsdRhoNoiseThreshold's own scale
// (and every other absolute-magnitude constant downstream) a plain [0,1]
// fraction instead of an arbitrary pixel-brightness-derived number.
const GRAYSCALE_MAX = 255;
export function computeGradient2x2Field(gray: Float64Array, w: number, h: number): GradientField {
  const fx = new Float64Array(w * h), fy = new Float64Array(w * h);
  const norm = 1 / (2 * GRAYSCALE_MAX);
  for (let y = 0; y < h - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const i = y * w + x;
      fx[i] = ((gray[i + 1] + gray[i + 1 + w]) - (gray[i] + gray[i + w])) * norm;
      fy[i] = ((gray[i + w] + gray[i + 1 + w]) - (gray[i] + gray[i + 1])) * norm;
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

// `directed` picks which of the two hue conventions to paint, and the choice
// is genuinely load-bearing rather than cosmetic:
//
// AXIAL (directed=false, the long-standing default, fieldView 'gradient2x2'):
// theta is folded into [0, PI) before mapping to a full 0-360 hue sweep, so a
// black-to-white edge and the white-to-black edge facing it get the SAME hue.
// That matches every mod-PI consumer in the codebase (computeGradientAgreementField,
// tangentWalk's guided walk) -- for them a line is a line regardless of which
// side is darker.
//
// DIRECTED (directed=true, fieldView 'gradient2x2Directed'): theta maps over
// its full [-PI, PI) range, so those two opposite-facing edges land exactly
// 180 degrees apart on the hue wheel -- opposite hues, not identical ones.
// This is the view that matches pipeline/lsdSegments.ts's segment growing,
// which is DIRECTED: the two edges of a single bright stripe are two different
// lines to it, and in the axial view they are literally indistinguishable. A
// region that looks like it should obviously have merged (uniform hue across a
// stripe) but didn't is explained instantly here -- the stripe turns out to be
// two opposing hues, exactly as the grower saw it. See lsdSegments.ts's own
// header for why growth reverted to directed angles.
export function paintVectorFieldAsColor(field: GradientField, out: Uint8Array, directed: boolean = false) {
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
    const o = i * 4;
    const raw = Math.atan2(fy[i], fx[i]); // [-PI, PI)
    let hueDeg: number;
    if (directed) {
      hueDeg = ((raw + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI) * 360;
    } else {
      let theta = raw;
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      hueDeg = (theta / Math.PI) * 360;
    }
    const sat = maxMag > 0 ? mags[i] / maxMag : 0;
    const [rr, gg, bb] = hsvToRgb(hueDeg, sat, 1);
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

