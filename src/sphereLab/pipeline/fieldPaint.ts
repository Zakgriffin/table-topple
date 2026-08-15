import { type GradientField } from '../types.ts';
import { hsvToRgb } from './distortion.ts';

// ── Painting a value field, for whichever one is on screen ────────────────
//
// Split out of gradientField.ts along the "── Display: ──" divider that file
// had already drawn through itself. What made the divider worth making
// structural: `hsvToRgb` was the ONLY thing the pose pipeline imported from
// distortion.ts, and it was imported solely for paintVectorFieldAsColor below.
// So this split is what drops distortion.ts -- a noise/blur/downsample/flip
// grab-bag whose every other consumer is capture, preview, an overlay, or the
// phone -- out of the pose library's import closure entirely.
//
// computeGradientMagnitudeField comes along despite being a "value field"
// rather than a painter, because its own comment already says what it is: the
// contamination overlays' ALPHA. It is frame-normalized, which is exactly the
// property that makes it a display quantity and not one to reason about rho
// with.

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
// This is the view that matches the segment growing, which is DIRECTED: the
// two edges of a single bright stripe are two different lines to it, and in
// the axial view they are literally indistinguishable. A region that looks
// like it should obviously have merged (uniform hue across a stripe) but
// didn't is explained instantly here -- the stripe turns out to be two
// opposing hues, exactly as the grower saw it. See the grow stage's own
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
