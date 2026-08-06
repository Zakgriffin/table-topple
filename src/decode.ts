// Grid decode primitives shared by the line-based rectification pipeline
// (src/lines.ts -> src/vp.ts -> src/lattice.ts -> src/homography.ts,
// orchestrated in src/main.ts): grayscale/binarize conversion, and
// pickBestCandidate's patch-tiling + correlation scoring, which the line
// pipeline reuses to decode whatever grid it samples via its fitted
// homography. Pure logic, no DOM/camera dependency, so it can run both in
// the browser and under Node for testing.
//
// pickBestCandidate picks among candidate sampled grids (the line pipeline
// passes exactly 2: the sampled grid and its row-mirrored twin, to cover the
// row-axis sort-direction ambiguity — see scripts/test-lines-decode.ts) by
// tiling each into discrete order x order patches, looking up each patch's
// exact-match seed anchor, then scoring every distinct seed by CORRELATING
// the ENTIRE sampled grid (not just one patch) against the actual known
// pattern at the position that seed implies (see scoreCorrelation). This
// tolerates individual misread bits gracefully (a genuinely correct anchor
// stays close to 1.0 even with some noise) while a wrong anchor sits close
// to 0.5 (uncorrelated with a random binary pattern) — with only 16 bits of
// key space at order 4, 65535 of 65536 possible windows are "valid", so a
// single patch's exact-match hit alone is weak evidence; this correlation
// check is what actually distinguishes a correct decode from noise that
// happened to hash to *some* valid-but-wrong position.
//
// This module previously also had an autocorrelation-based pitch/phase
// estimator (detectGrid) and a gradient-orientation rotation estimator
// (estimateRotationRad), used only to manufacture a single GLOBAL apparent
// cell-pitch number as a resolution hint for src/lines.ts's Hough transform
// and an alias-check reference for src/lattice.ts. Both were removed: a
// single global scalar is systematically wrong under real perspective for
// content far from wherever it was measured, and both of its consumers now
// derive what they need locally instead — src/lattice.ts's
// estimateLocalSpacing measures real neighboring detections directly, and
// src/main.ts's Hough resolution is now a small fixed constant rather than
// adaptive (see its HOUGH_RHO_BIN_PX comment for why fixed-fine is safe).

// valid is false when a cell couldn't actually be sampled (the homography
// mapped it outside the image, see src/main.ts's sampleFromHomography) — x/
// y/bit are meaningless placeholders in that case. cornerCount is a leftover
// per-cell confidence field from an earlier corner-mesh-based sampler,
// unused by the line pipeline (always 0).
interface SampledCell { x: number; y: number; bit: number; valid: boolean; cornerCount: number; }



export interface Patch {
  tileRow: number; tileCol: number; // position in the tile grid (not torus coords)
  cells: SampledCell[][]; // order x order
  match: { row: number; col: number } | null;
  // Per-cell ground-truth correctness (does this cell's bit match the actual
  // known pattern at the position the frame's winning anchor implies?), for
  // visual debugging — see pickBestCandidate, which populates this only for
  // the winning candidate (needs a resolved anchor to compare against).
  correct: boolean[][] | null;
}










export function toGrayscale(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Float64Array {
  const gray = new Float64Array(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return gray;
}

