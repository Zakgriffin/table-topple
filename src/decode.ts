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
export interface SampledCell { x: number; y: number; bit: number; valid: boolean; cornerCount: number; }

// rows/cols is however many cells the sampler produced; originRow/originCol
// give the array index of whichever cell the sampler treats as its anchor,
// for callers that need to map back to pixel positions without recomputing
// it themselves. src/main.ts's sampleFromHomography (the line pipeline's
// sampler) always sets both to 0.
export interface SampledGrid {
  rows: number; cols: number; cells: SampledCell[][];
  originRow: number; originCol: number;
}

// Packs an order x order cell block into a lookup key, reading it under one
// of 4 orientations (0/90/180/270 degree rotation of the reading order).
// Orientation 0 matches src/debruijn.ts's windowKey order exactly
// (top-to-bottom, left-to-right, MSB-first); the others apply the standard
// NxN matrix rotation index mapping before packing.
function packPatchCells(cells: SampledCell[][], order: number, orientation: number): number {
  let key = 0;
  for (let i = 0; i < order; i++) {
    for (let j = 0; j < order; j++) {
      let bit: number;
      if (orientation === 1) bit = cells[order - 1 - j][i].bit;       // 90 CW
      else if (orientation === 2) bit = cells[order - 1 - i][order - 1 - j].bit; // 180
      else if (orientation === 3) bit = cells[j][order - 1 - i].bit;  // 270 CW
      else bit = cells[i][j].bit;                                    // 0
      key = (key << 1) | bit;
    }
  }
  return key >>> 0;
}

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

export interface PatchDecodeResult {
  patches: Patch[];
  orientation: number | null; // resolved 0/90/180/270 reading orientation, shared across all patches this frame
  consistency: number; // fraction of adjacent-tile pairs whose decoded positions agree, 0..1
}

interface Tile { tileRow: number; tileCol: number; cells: SampledCell[][]; }

function buildTiles(sg: SampledGrid, order: number): Tile[] {
  const tileRows = Math.floor(sg.rows / order);
  const tileCols = Math.floor(sg.cols / order);
  const tiles: Tile[] = [];
  for (let I = 0; I < tileRows; I++) {
    for (let J = 0; J < tileCols; J++) {
      const cells: SampledCell[][] = [];
      for (let i = 0; i < order; i++) cells.push(sg.cells[I * order + i].slice(J * order, J * order + order));
      tiles.push({ tileRow: I, tileCol: J, cells });
    }
  }
  return tiles;
}

function patchesForOrientation(tiles: Tile[], order: number, lookup: Int32Array, C: number, o: number): Patch[] {
  return tiles.map(tile => {
    // A gap anywhere in the tile (see SampledCell.valid) means an
    // exact-match lookup can't be trusted; skip straight to no-match rather
    // than packing a bogus bit into the key. The tile still contributes to
    // scoreCorrelation later via whichever of its cells ARE valid, just not
    // as an exact-match seed candidate here.
    const complete = tile.cells.every(row => row.every(c => c.valid));
    const packed = complete ? lookup[packPatchCells(tile.cells, order, o)] : -1;
    const match = packed === -1 ? null : { row: Math.floor(packed / C), col: packed % C };
    return { tileRow: tile.tileRow, tileCol: tile.tileCol, cells: tile.cells, match, correct: null };
  });
}

// Applies the same orientation-dependent rotation packPatchCells uses (see
// its comment) to a (sampled-grid row shift, col shift) vector, giving the
// torus (row shift, col shift) a correct decode should show between two
// windows offset by that amount in sampled-grid space. Derived by tracking
// how packPatchCells's per-orientation index mapping responds to shifting
// its window's reference point by one cell in each sampled-grid direction —
// same underlying rotation, just applied to a shift vector instead of a
// cell index.
export function rotateShift(da: number, db: number, orientation: number): [number, number] {
  if (orientation === 1) return [db, -da];
  if (orientation === 2) return [-da, -db];
  if (orientation === 3) return [-db, da];
  return [da, db];
}

// Alternative seed generation to patchesForOrientation's non-overlapping
// tiling — the line pipeline's homography-based sampling (src/main.ts's
// sampleFromHomography) can produce gaps near the image edge, where a
// tile-aligned order x order block landing entirely inside the valid region
// is far less likely than SOME order x order window doing so. Sliding a
// window across EVERY possible position, not just tile-aligned ones, finds a
// valid window far more often. Only called when the grid actually has gaps
// (see pickBestCandidate) — a fully-valid grid never needs this, so its
// behavior and cost are unaffected.
function findSlidingSeeds(sg: SampledGrid, order: number, lookup: Int32Array, R: number, C: number, orientation: number): { row: number; col: number }[] {
  const seen = new Set<string>();
  const anchors: { row: number; col: number }[] = [];
  for (let i0 = 0; i0 + order <= sg.rows; i0++) {
    for (let j0 = 0; j0 + order <= sg.cols; j0++) {
      let complete = true;
      for (let i = 0; i < order && complete; i++) {
        for (let j = 0; j < order; j++) {
          if (!sg.cells[i0 + i][j0 + j].valid) { complete = false; break; }
        }
      }
      if (!complete) continue;
      const window = sg.cells.slice(i0, i0 + order).map(row => row.slice(j0, j0 + order));
      const packed = lookup[packPatchCells(window, order, orientation)];
      if (packed === -1) continue;
      const matchRow = Math.floor(packed / C), matchCol = packed % C;
      const [dr, dc] = rotateShift(i0, j0, orientation);
      const anchorRow = ((matchRow - dr) % R + R) % R, anchorCol = ((matchCol - dc) % C + C) % C;
      const key = `${anchorRow},${anchorCol}`;
      if (seen.has(key)) continue;
      seen.add(key);
      anchors.push({ row: anchorRow, col: anchorCol });
    }
  }
  return anchors;
}

// Scores a candidate torus anchor (the torus position implied for the
// sampled grid's own cell (0,0), i.e. its top-left corner) by comparing
// EVERY sampled cell — not just one patch, not just adjacent pairs — against
// the actual torus content at the positions that anchor implies. A genuinely
// correct anchor stays close to 1.0 even with a handful of misread bits
// (graceful degradation, rather than exact-match's all-or-nothing), while a
// wrong anchor — even one some patch found via a "valid" exact-match lookup,
// which is weak evidence on its own given only 16 bits of key space at order
// 4 (65535 of 65536 possible windows are valid) — should sit close to 0.5
// (uncorrelated with a random binary pattern), giving much better separation
// than checking patches only agree with each other.
function scoreCorrelation(sg: SampledGrid, torus: Uint8Array[], R: number, C: number, anchorRow: number, anchorCol: number, orientation: number): number {
  let agree = 0, total = 0;
  for (let i = 0; i < sg.rows; i++) {
    for (let j = 0; j < sg.cols; j++) {
      if (!sg.cells[i][j].valid) continue; // gap (e.g. sampled point fell outside the image) — no data to compare
      total++;
      const [dr, dc] = rotateShift(i, j, orientation);
      const torusRow = ((anchorRow + dr) % R + R) % R;
      const torusCol = ((anchorCol + dc) % C + C) % C;
      if (sg.cells[i][j].bit === torus[torusRow][torusCol]) agree++;
    }
  }
  return total === 0 ? 0 : agree / total;
}

// Runs across every candidate sampled grid the caller passes (the line
// pipeline passes 2: the sampled grid and its row-mirrored twin, to cover
// the row-axis sort-direction ambiguity — see scripts/test-lines-decode.ts),
// and across all 4 tile-reading orientations, via scoreCorrelation directly
// rather than the weaker patch-vs-patch agreement check — necessary because
// a grid with edge gaps can have too few adjacent-patch pairs for that check
// to discriminate anything, whereas scoreCorrelation always has real
// pattern content to compare against.
//
// Seeds come from two sources: patchesForOrientation's non-overlapping tiles
// (fast, works well when most cells are valid) and, when the grid has any
// gaps, findSlidingSeeds's exhaustive window search (needed because a
// randomly tile-aligned window landing on an all-valid block is rare once a
// meaningful fraction of cells are gaps — see findSlidingSeeds). Both feed
// the same scoreCorrelation-based selection, so a grid with gaps pays the
// extra sliding-window cost only when it actually needs it. match is the
// torus position of the sampled grid's CENTER cell (roughly where the
// camera is actually pointed), not (0,0).
export interface CandidateResult extends PatchDecodeResult { candidateIndex: number; match: { row: number; col: number } | null; }

export function pickBestCandidate(sampledGrids: SampledGrid[], order: number, lookup: Int32Array, torus: Uint8Array[], R: number, C: number): CandidateResult {
  let best: CandidateResult | null = null;
  let winningAnchor: { row: number; col: number } | null = null;

  for (let i = 0; i < sampledGrids.length; i++) {
    const sg = sampledGrids[i];
    const tiles = buildTiles(sg, order);
    const hasGaps = sg.cells.some(row => row.some(c => !c.valid));

    for (let o = 0; o < 4; o++) {
      const patches = patchesForOrientation(tiles, order, lookup, C, o);
      const seenAnchors = new Set<string>();
      let bestAnchor: { row: number; col: number } | null = null;
      let bestScore = -1;

      for (const patch of patches) {
        if (!patch.match) continue;
        const [dr, dc] = rotateShift(patch.tileRow * order, patch.tileCol * order, o);
        const anchorRow = ((patch.match.row - dr) % R + R) % R;
        const anchorCol = ((patch.match.col - dc) % C + C) % C;
        const key = `${anchorRow},${anchorCol}`;
        if (seenAnchors.has(key)) continue;
        seenAnchors.add(key);
        const score = scoreCorrelation(sg, torus, R, C, anchorRow, anchorCol, o);
        if (score > bestScore) { bestScore = score; bestAnchor = { row: anchorRow, col: anchorCol }; }
      }

      if (hasGaps) {
        for (const anchor of findSlidingSeeds(sg, order, lookup, R, C, o)) {
          const key = `${anchor.row},${anchor.col}`;
          if (seenAnchors.has(key)) continue;
          seenAnchors.add(key);
          const score = scoreCorrelation(sg, torus, R, C, anchor.row, anchor.col, o);
          if (score > bestScore) { bestScore = score; bestAnchor = anchor; }
        }
      }

      const consistency = Math.max(0, bestScore);
      let match: { row: number; col: number } | null = null;
      if (bestAnchor) {
        const centerI = Math.floor(sg.rows / 2), centerJ = Math.floor(sg.cols / 2);
        const [dr, dc] = rotateShift(centerI, centerJ, o);
        match = { row: ((bestAnchor.row + dr) % R + R) % R, col: ((bestAnchor.col + dc) % C + C) % C };
      }

      if (!best || consistency > best.consistency) {
        best = { patches, orientation: o, consistency, candidateIndex: i, match };
        winningAnchor = bestAnchor;
      }
    }
  }

  if (!best) return { patches: [], orientation: null, consistency: 0, candidateIndex: -1, match: null };

  // Per-cell ground-truth correctness, computed only for the winning
  // candidate (see the Patch.correct comment) — cheap enough (one pass over
  // the sampled cells) to not bother gating behind whether a caller actually
  // wants it for display.
  if (winningAnchor && best.orientation !== null) {
    const o = best.orientation, anchor = winningAnchor;
    best.patches = best.patches.map(patch => ({
      ...patch,
      correct: patch.cells.map((row, i) => row.map((cell, j) => {
        const globalI = patch.tileRow * order + i, globalJ = patch.tileCol * order + j;
        const [dr, dc] = rotateShift(globalI, globalJ, o);
        const torusRow = ((anchor.row + dr) % R + R) % R;
        const torusCol = ((anchor.col + dc) % C + C) % C;
        return cell.bit === torus[torusRow][torusCol];
      })),
    }));
  }

  return best;
}

export function toGrayscale(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Float64Array {
  const gray = new Float64Array(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return gray;
}

// Binarizes a grayscale buffer via global mean threshold (dark -> 1, i.e. a
// black cell).
export function binarize(gray: Float64Array): Uint8Array {
  let mean = 0;
  for (let i = 0; i < gray.length; i++) mean += gray[i];
  mean /= gray.length;
  const bin = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) bin[i] = gray[i] < mean ? 1 : 0;
  return bin;
}
