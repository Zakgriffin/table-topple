// ── Top-gradient overlay paint (pure) ────────────────────────────────────
//
// Alpha scales linearly with each pixel's own gradient2x2 magnitude: 0 at
// magnitude 0, fully opaque at magnitude 1 -- computeGradient2x2Field is
// itself normalized so hypot(fx,fy)'s true theoretical ceiling is exactly 1
// (see that function's own comment), so no separate max constant is needed
// here anymore.

import { type GradientField } from '../../shared/types.ts';

// `field` is the pipeline's, so it is TOP-DOWN, while `out` is a flipY=false
// preview texture and is bottom-up. The row reversal happens here, at the
// paint, which is the same rule axesReconstruction.ts follows for the gray
// itself: the flip goes on the way OUT to display, never on the math going in.
// See overlays/pipelineField.ts.
export function paintTopGradientOverlay(color: readonly [number, number, number], field: GradientField, out: Uint8Array) {
  const { fx, fy, w, h } = field;
  for (let y = 0; y < h; y++) {
    const src = y * w, dst = (h - 1 - y) * w;
    for (let x = 0; x < w; x++) {
      const o = (dst + x) * 4;
      const alpha = Math.min(1, Math.hypot(fx[src + x], fy[src + x]));
      out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2]; out[o + 3] = Math.round(alpha * 255);
    }
  }
}

export const TOP_GRADIENT_COLOR = [60, 230, 90] as const;
