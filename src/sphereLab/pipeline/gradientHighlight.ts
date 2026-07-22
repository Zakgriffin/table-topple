import { GradientField } from '../types.ts';

// ── Top-N%-by-magnitude overlay math (pure) ──────────────────────────────
//
// Same [minPercent, maxPercent) percentile-rank band votesInMagnitudeBand
// (pipeline/votes.ts) uses to pick circle-visualization votes, applied to
// every pixel of a vector field's magnitude instead of a sparse vote list --
// reuses the same circleSamplePercentMin/Max settings as the band bounds, so
// "highlight top N%" always means the same N% the circles overlay is already
// selecting.

export function computeTopGradientAlpha(field: GradientField, minPercent: number, maxPercent: number): Float64Array {
  const { fx, fy, w, h, r } = field;
  const n = w * h;
  const mags = new Float64Array(n);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      mags[i] = Math.hypot(fx[i], fy[i]);
    }
  }
  const alpha = new Float64Array(n);
  const lo = Math.round(n * (minPercent / 100));
  const hi = Math.round(n * (maxPercent / 100));
  if (hi <= lo) return alpha;
  const sorted = Float64Array.from(mags).sort((a, b) => b - a);
  const upperThresh = sorted[lo];
  const lowerThresh = sorted[hi - 1];
  for (let i = 0; i < n; i++) {
    if (mags[i] <= upperThresh && mags[i] >= lowerThresh) alpha[i] = 1;
  }
  return alpha;
}

export function paintTopGradientOverlay(alpha: Float64Array, color: readonly [number, number, number], out: Uint8Array) {
  for (let i = 0; i < alpha.length; i++) {
    const o = i * 4;
    out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2];
    out[o + 3] = Math.round(alpha[i] * 200);
  }
}

export const TOP_GRADIENT_COLOR = [60, 230, 90] as const;
