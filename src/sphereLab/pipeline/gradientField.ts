import { type GradientField } from '../types.ts';
import { hsvToRgb } from './distortion.ts';

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

// Frame-normalized per-pixel gradient magnitude, i.e. each pixel's |g| divided
// by this frame's own max |g|. This is the contamination overlays' WEIGHT, and
// frame-relative is what they want: there it is an ALPHA, so a low-contrast
// capture stays visible instead of coming out nearly transparent. That divisor
// changes frame to frame, so it is NOT a quantity to reason about rho with --
// computeGradient2x2Field's own output is (ceiling exactly 1, see its comment).
export function computeGradientMagnitudeField(field: GradientField): Float64Array {
  const { fx, fy } = field;
  const out = new Float64Array(fx.length);
  let maxMag = 0;
  for (let i = 0; i < fx.length; i++) {
    const mag = Math.hypot(fx[i], fy[i]);
    out[i] = mag;
    if (mag > maxMag) maxMag = mag;
  }
  if (maxMag > 0) for (let i = 0; i < out.length; i++) out[i] /= maxMag;
  return out;
}


// ── Display: colorizes a value field, only for whichever one is on screen ─

// `directed` picks which of the two hue conventions to paint, and the choice
// is genuinely load-bearing rather than cosmetic:
//
// AXIAL (directed=false, the long-standing default, fieldView 'gradient2x2'):
// theta is folded into [0, PI) before mapping to a full 0-360 hue sweep, so a
// black-to-white edge and the white-to-black edge facing it get the SAME hue.
// That matches every mod-PI consumer in the codebase (computeContaminationAlpha
// is the last one left) -- for it a line is a line regardless of which side is
// darker.
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


export function fillGrayscalePreview(gray: Float64Array, out: Uint8Array) {
  for (let i = 0; i < gray.length; i++) {
    const v = Math.max(0, Math.min(255, gray[i]));
    const o = i * 4;
    out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
  }
}

