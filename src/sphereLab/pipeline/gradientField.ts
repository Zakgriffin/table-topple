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

// ── The two candidate LSD driving fields, on ONE absolute scale ──────────
//
// Both of these return magnitudes on the SAME scale rho is thresholded
// against, with no frame-relative normalization -- which is what makes them
// directly comparable to each other AND directly meaningful against
// lsdRhoNoiseThreshold / lsdRhoHighThreshold.
//
// That they share a scale is not a coincidence worth trusting blindly, so:
// computeGradient2x2Field divides by 2*GRAYSCALE_MAX, giving hypot(fx,fy) a
// true ceiling of exactly 1. separableBoxBlur AVERAGES rather than sums, so
// the agreement value is the magnitude of the MEAN double-angle vector over
// the window -- bounded by the mean of the magnitudes, hence by that same 1.
//
// The behavioural difference, which is the point of comparing them: agreement
// at pixel i is a NEIGHBOURHOOD quantity, so it can come out HIGHER than the
// raw magnitude there. A weak pixel inside a coherent edge is lifted by its
// neighbours; an isolated strong spike whose neighbours disagree is pulled
// toward zero. Raw magnitude does neither.

// Raw per-pixel gradient magnitude. Absolute, ceiling 1.
export function gradientMagnitudes(field: GradientField): Float64Array {
  const { fx, fy } = field;
  const out = new Float64Array(fx.length);
  for (let i = 0; i < fx.length; i++) out[i] = Math.hypot(fx[i], fy[i]);
  return out;
}

// Magnitude of the local MEAN of gradients. Absolute, same ceiling.
//
// aggRadius 0 is the identity (a radius-0 box blur returns its input), so this
// degenerates exactly to gradientMagnitudes -- which is why radius is the
// single knob separating the two fields being compared.
//
// ── `directed` is NOT a display choice here, it decides what is being measured ──
//
// UNDIRECTED (mod π) folds each gradient to double angle before averaging, so
// θ and θ+π land on the same vector and opposing edges REINFORCE. That is the
// right convention for the contamination overlays, whose question is "is there
// a strong oriented structure here", polarity irrelevant.
//
// DIRECTED (mod 2π) averages the raw vectors, so opposing edges CANCEL. That is
// the right convention for judging this field as an LSD driver, because
// lsdSegments' growth rule is directed: every orientation comparison there is a
// plain SIGNED cos-dot against cos(τ). Aggregating mod π and then feeding a
// directed grower would inflate exactly the pixel pairs the grower goes on to
// refuse -- the two sides of a thin stripe boosting each other, then not being
// allowed to join.
export function agreementMagnitudes(field: GradientField, aggRadius: number, directed: boolean): Float64Array {
  const { fx, fy, w, h } = field;
  const n = w * h;
  let cx: Float64Array, cy: Float64Array;
  if (directed) {
    cx = fx; cy = fy;
  } else {
    cx = new Float64Array(n); cy = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const mag = Math.hypot(fx[i], fy[i]);
      if (mag === 0) continue;
      const theta = Math.atan2(fy[i], fx[i]);
      cx[i] = mag * Math.cos(2 * theta);
      cy[i] = mag * Math.sin(2 * theta);
    }
  }
  const sx = separableBoxBlur(cx, w, h, aggRadius);
  const sy = separableBoxBlur(cy, w, h, aggRadius);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.hypot(sx[i], sy[i]);
  return out;
}

// Frame-normalized agreement, i.e. divided by this frame's own RAW max
// magnitude. The contamination overlays want this rather than the absolute
// form: there it is an ALPHA, and a frame-relative one keeps a low-contrast
// capture's overlay visible instead of nearly transparent. Do NOT use it to
// reason about rho -- the divisor changes frame to frame.
export function computeGradientAgreementField(field: GradientField, aggRadius: number): Float64Array {
  const { fx, fy } = field;
  let maxRawMag = 0;
  for (let i = 0; i < fx.length; i++) {
    const mag = Math.hypot(fx[i], fy[i]);
    if (mag > maxRawMag) maxRawMag = mag;
  }
  // Undirected: the contamination question is polarity-agnostic. See
  // agreementMagnitudes on why the LSD-driver view uses the other convention.
  const agreement = agreementMagnitudes(field, aggRadius, false);
  if (maxRawMag > 0) for (let i = 0; i < agreement.length; i++) agreement[i] /= maxRawMag;
  return agreement;
}

// Frame-normalized raw magnitude -- the aggRadius 0 counterpart to
// computeGradientAgreementField, carrying the identical normalization so the
// two contamination alphas are directly comparable to each other. Same caveat:
// not a rho-scale quantity.
export function computeGradientMagnitudeField(field: GradientField): Float64Array {
  const out = gradientMagnitudes(field);
  let maxMag = 0;
  for (let i = 0; i < out.length; i++) if (out[i] > maxMag) maxMag = out[i];
  if (maxMag > 0) for (let i = 0; i < out.length; i++) out[i] /= maxMag;
  return out;
}

export function computeEffectiveGradientField(field: GradientField, agreement: Float64Array): GradientField {
  const { fx, fy, w, h, r } = field;
  const n = w * h;
  const efx = new Float64Array(n), efy = new Float64Array(n);
  for (let i = 0; i < n; i++) { efx[i] = fx[i] * agreement[i]; efy[i] = fy[i] * agreement[i]; }
  return { fx: efx, fy: efy, w, h, r };
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


// Paints a magnitude field as the THREE BANDS the LSD grower actually sees,
// against the live rho sliders. Not a grayscale ramp on purpose: the question
// this view exists to answer is "what would rho keep", and a ramp makes the
// reader do the thresholding by eye.
//
//   black  mag <= rhoLow            never eligible, cannot form a growth edge
//   gray   rhoLow < mag <= rhoHigh   grows, but its component is discarded
//                                    unless some member clears rhoHigh
//   white  mag > rhoHigh            clears survival on its own
//
// rhoHigh <= rhoLow (the shipping default has rhoHigh 0) is the documented
// degenerate case -- hysteresis collapses to a single threshold, so the middle
// band correctly does not appear rather than being special-cased away.
export function paintRhoBands(mag: Float64Array, rhoLow: number, rhoHigh: number, out: Uint8Array) {
  const survive = Math.max(rhoLow, rhoHigh);
  for (let i = 0; i < mag.length; i++) {
    const m = mag[i];
    const v = m <= rhoLow ? 0 : (m > survive ? 255 : 96);
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

