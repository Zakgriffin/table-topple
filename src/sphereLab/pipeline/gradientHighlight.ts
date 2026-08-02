// ── Top-gradient overlay paint (pure) ────────────────────────────────────
//
// Alpha scales linearly with each pixel's own gradient2x2 magnitude: 0 at
// magnitude 0, fully opaque at magnitude 1 -- computeGradient2x2Field is
// itself normalized so hypot(fx,fy)'s true theoretical ceiling is exactly 1
// (see that function's own comment), so no separate max constant is needed
// here anymore.

import { GradientField } from '../types.ts';

export function paintTopGradientOverlay(color: readonly [number, number, number], field: GradientField, out: Uint8Array) {
  const { fx, fy } = field;
  for (let i = 0; i < fx.length; i++) {
    const o = i * 4;
    const alpha = Math.min(1, Math.hypot(fx[i], fy[i]));
    out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2]; out[o + 3] = Math.round(alpha * 255);
  }
}

export const TOP_GRADIENT_COLOR = [60, 230, 90] as const;
