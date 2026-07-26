// ── Top-gradient overlay paint (pure) ────────────────────────────────────
//
// Used to gate which pixels get painted via a [minPercent, maxPercent)
// percentile-rank band over the field's own magnitude (computeTopGradientAlpha,
// same band shape votesInMagnitudeBand in pipeline/votes.ts uses) -- every
// call site always passed (0, 100), which is mathematically a no-op (the
// resulting [min, max] magnitude band trivially contains every pixel's
// magnitude, by definition of min/max), so that dead computation (a full
// sort of the field on every repaint, for a guaranteed-uniform result) was
// removed along with every no-op seedEligible band it fed elsewhere (see
// pipeline/bucketFillSegments.ts's computeBucketFillRegions and its two
// callers) -- see this session's chat.

export function paintTopGradientOverlay(color: readonly [number, number, number], out: Uint8Array) {
  for (let o = 0; o < out.length; o += 4) {
    out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2]; out[o + 3] = 200;
  }
}

export const TOP_GRADIENT_COLOR = [60, 230, 90] as const;
