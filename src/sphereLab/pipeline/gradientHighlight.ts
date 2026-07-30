// ── Top-gradient overlay paint (pure) ────────────────────────────────────
//
// Alpha scales linearly with each pixel's own gradient2x2 magnitude: 0 at
// magnitude 0, fully opaque at MAX_GRADIENT_2X2_MAGNITUDE (the hard
// theoretical ceiling for a gradient2x2 field, not a per-frame max -- see
// computeGradient2x2Field, whose fx/fy are each an average of two grayscale
// edge differences, so neither component can exceed a full black-to-white
// swing).

import { GradientField } from '../types.ts';

const GRAYSCALE_MAX = 255;
export const MAX_GRADIENT_2X2_MAGNITUDE = Math.hypot(GRAYSCALE_MAX, GRAYSCALE_MAX);

export function paintTopGradientOverlay(color: readonly [number, number, number], field: GradientField, out: Uint8Array) {
  const { fx, fy } = field;
  for (let i = 0; i < fx.length; i++) {
    const o = i * 4;
    const mag = Math.hypot(fx[i], fy[i]);
    const alpha = Math.min(1, mag / MAX_GRADIENT_2X2_MAGNITUDE);
    out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2]; out[o + 3] = Math.round(alpha * 255);
  }
}

export const TOP_GRADIENT_COLOR = [60, 230, 90] as const;
