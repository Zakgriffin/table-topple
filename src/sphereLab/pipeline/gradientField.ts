import { type GradientField } from '../types.ts';

// MOVED HERE when the old pipeline was deleted. These are pure IMAGE functions -- a
// gradient is a property of a picture, not of a pose -- and the app computes
// them from its own preview image (pipeline/projectedBins.ts) and its own
// capture (pipeline/preview.ts), never from a pipeline intermediate. So they
// outlived the pipeline that also used them.

// ── Stage 1: the gradient field (no color, no display) ───────────────────
//
// The painters that used to live below these -- paintVectorFieldAsColor,
// fillGrayscalePreview, computeGradientMagnitudeField -- moved to
// fieldPaint.ts, along with this file's only import of distortion.ts.

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
