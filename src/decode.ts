// Grid decode primitives shared by the line-based rectification pipeline
// (src/lines.ts -> src/vp.ts -> src/lattice.ts -> src/homography.ts,
// orchestrated in src/main.ts): grayscale/binarize conversion, the
// rotation-mod-90 estimator, autocorrelation-based pitch/phase detection
// (used only as an adaptive-resolution hint for the Hough transform, not as
// the geometry solution), and pickBestCandidate's patch-tiling + correlation
// scoring, which the line pipeline reuses to decode whatever grid it samples
// via its fitted homography. Pure logic, no DOM/camera dependency, so it can
// run both in the browser and under Node for testing.
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

export interface GridDetection {
  px: number; py: number; // phase (px offset of first cell boundary)
  pitchX: number; pitchY: number;
}

// Finds the pitch (dominant period) of a 1D energy profile via autocorrelation.
//
// A sub-pixel (parabolic-interpolation) refinement was attempted here, on
// the theory that a real camera's cell size is essentially never an exact
// integer number of pixels, so rounding to the nearest integer lag should
// compound into visible drift over many cells. In practice it made the
// rotated-input synthetic test regress badly (300-trial pass rate dropped
// from ~99% to ~1%), including with a stricter validity threshold and with
// pre-smoothing the score profile — nearest-neighbor resampling during
// derotation introduces staircase aliasing that the parabolic fit reacts to.
// Reverted rather than ship something destabilized; worth retrying later
// with a test setup whose ground-truth pitch isn't coincidentally an exact
// integer (this test's source PNG is rendered at exactly 8px/cell, which
// hid whether the refinement was helping or hurting on real-world input).
export function findPitch(energy: Float64Array, minLag: number, maxLag: number): number {
  let bestLag = minLag, bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let score = 0;
    for (let x = 0; x + lag < energy.length; x++) score += energy[x] * energy[x + lag];
    if (score > bestScore) { bestScore = score; bestLag = lag; }
  }
  return bestLag;
}

// Finds the phase in [0, pitch) whose boundary positions (phase, phase+pitch,
// phase+2*pitch, ...) best align with peaks in the energy profile.
export function findPhase(energy: Float64Array, pitch: number): number {
  let bestPhase = 0, bestScore = -Infinity;
  for (let phase = 0; phase < pitch; phase++) {
    let score = 0;
    for (let x = phase; x < energy.length; x += pitch) score += energy[x];
    if (score > bestScore) { bestScore = score; bestPhase = phase; }
  }
  return bestPhase;
}

// Runs pitch/phase detection restricted to the sub-rectangle [x0,x1) x
// [y0,y1) of the buffer (still indexed in full-buffer coordinates in the
// result). detectGrid is just this called on the whole buffer.
function detectGridInRegion(bin: Uint8Array, w: number, x0: number, y0: number, x1: number, y1: number): GridDetection {
  const rw = x1 - x0, rh = y1 - y0;
  const colEnergy = new Float64Array(rw);
  for (let x = 1; x < rw; x++) {
    let e = 0;
    for (let y = y0; y < y1; y++) e += Math.abs(bin[y * w + (x0 + x)] - bin[y * w + (x0 + x - 1)]);
    colEnergy[x] = e;
  }
  const rowEnergy = new Float64Array(rh);
  for (let y = 1; y < rh; y++) {
    let e = 0;
    for (let x = x0; x < x1; x++) e += Math.abs(bin[(y0 + y) * w + x] - bin[(y0 + y - 1) * w + x]);
    rowEnergy[y] = e;
  }

  const minLag = 4, maxLagX = Math.floor(rw / 4), maxLagY = Math.floor(rh / 4);
  const pitchX = findPitch(colEnergy, minLag, maxLagX);
  const pitchY = findPitch(rowEnergy, minLag, maxLagY);
  let px = findPhase(colEnergy, pitchX) + x0;
  let py = findPhase(rowEnergy, pitchY) + y0;

  // Re-anchor the phase to the boundary nearest the REGION's center, rather
  // than leaving it wherever findPhase's [0, pitch) search happened to land
  // it (always near the region's own index 0) — anchoring near the center
  // keeps this estimate more stable when only used as a coarse pitch HINT
  // (see src/main.ts's apparentPitch), rather than as a basis for
  // extrapolating cell positions outward.
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  px += pitchX * Math.round((cx - px) / pitchX);
  py += pitchY * Math.round((cy - py) / pitchY);

  return { px, py, pitchX, pitchY };
}

export function detectGrid(bin: Uint8Array, w: number, h: number): GridDetection {
  return detectGridInRegion(bin, w, 0, 0, w, h);
}

// Estimates the grid's rotation, folded into [0, PI/2) radians, via a
// gradient-orientation histogram. Edges at cell boundaries produce gradient
// vectors perpendicular to the boundary line; a square grid's two line
// families (rows, columns) are perpendicular to each other, so their
// gradient directions are too — folding all edge orientations modulo 90
// degrees collapses both families onto the same histogram peak, directly
// giving the grid's rotation mod 90 in one shot (weighted by edge strength
// so faint/noisy gradients don't skew the estimate).
//
// Operates on blurred GRAYSCALE, not the binarized image: a rotated hard
// edge in an already-thresholded binary image becomes a staircase, whose
// micro-edges are biased toward 0/90 degrees rather than the true diagonal —
// same reason edge detectors like Canny blur before computing gradients.
function boxBlur(gray: Float64Array, w: number, h: number, radius: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          sum += gray[yy * w + xx];
          count++;
        }
      }
      out[y * w + x] = sum / count;
    }
  }
  return out;
}

// Uses a circular mean rather than a histogram peak search: orientation data
// is periodic mod PI/2 (both edge families and the +-direction ambiguity of
// each gradient all collapse together), so scaling each angle by 4 maps that
// periodicity onto the full circle (4 * PI/2 = 2*PI). Averaging as vectors in
// that scaled space, then dividing the result back down by 4, gives a
// continuous estimate with no bin-quantization error — a histogram's ~0.5
// degree bin width was enough residual error to visibly drift pixel sampling
// across a couple hundred pixels of buffer.
export function estimateRotationRad(gray: Float64Array, w: number, h: number): number {
  const blurred = boxBlur(gray, w, h, 2);
  let sumX = 0, sumY = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx = blurred[y * w + x + 1] - blurred[y * w + x - 1];
      const gy = blurred[(y + 1) * w + x] - blurred[(y - 1) * w + x];
      const mag = gx * gx + gy * gy; // squared magnitude — emphasizes strong edges
      if (mag < 1) continue;
      const angle = Math.atan2(gy, gx); // -PI..PI
      sumX += mag * Math.cos(4 * angle);
      sumY += mag * Math.sin(4 * angle);
    }
  }
  const meanAngle4 = Math.atan2(sumY, sumX); // -PI..PI
  let theta = meanAngle4 / 4; // -PI/4..PI/4
  if (theta < 0) theta += Math.PI / 2;
  return theta; // [0, PI/2)
}

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
