import * as THREE from 'three';
import { cornerDir } from '../math/geometry.ts';
import { C, R } from '../floorPattern.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { globalState } from '../state.ts';
import { sweepResultantsGPU } from '../pipelineGPU/periodSweep.ts';
import { CompositeLine } from '../types.ts';

// ── Grid period/phase recovery from composite lines (pure) ────────────────
//
// Given the orientation already recovered by fitPairOfPlanes (Drow, Dcol,
// Dnormal), figures out WHICH row/column of the physical grid each detected
// composite line actually is -- the piece fitPairOfPlanes doesn't attempt
// (it only recovers the grid's ORIENTATION, not its registration/spacing).
//
// The derivation (worked out in conversation, not copied from any existing
// reference) has two equivalent forms: one using a line's own vote NORMAL,
// one using an actual POINT/ray on the line, gnomonically projected onto the
// plane tangent to the sphere at -Dnormal. Both are affine-linear in the
// line's true physical offset, but they're numerically DIFFERENT quantities
// (different slope/intercept) -- so this file deliberately uses ONLY the
// point-projection form throughout, via `gnomonic()` below, even though the
// normal-vector form is still used to CLASSIFY a line as row/column-type
// (that test only cares about orientation, not position, so either form
// works there identically). Using one consistent form everywhere means the
// period/phase values computed here land in the exact same coordinate space
// a caller would use to draw the rectified composite lines and a derived
// sample lattice -- mixing the two forms would silently misalign them.
//
// gnomonic(r) = (-(r.Drow)/(r.Dnormal), -(r.Dcol)/(r.Dnormal)) for a ray r:
// every point on a given row line shares the same xCol (that's what makes a
// row line project to a straight, constant-xCol line at all), and that
// shared xCol is affine-linear in the row's true offset -- symmetrically,
// xRow for column lines. That's "period and phase" reduced to: classify
// each line (free, same |n.Drow| vs |n.Dcol| test the plane-pair fit itself
// relies on), gnomonically project one representative point per line, then
// find the period/phase of the resulting 1D point set.
//
// Finding that period is NOT a further closed-form step (frequency
// parameters don't reduce to one polynomial eigen-solve the way
// fitPairOfPlanes' orientation did) -- it's a small, deliberately NARROW
// bracketed search around a pairwise-gap seed estimate, scored at each
// candidate period by the weighted circular-mean resultant length (the same
// "fold onto a circle, measure how tightly it clusters" construction
// computeGradientAgreementField itself uses, via its own double-angle
// folding). Row and
// column periods are pooled into one shared search (square cells force the
// same physical period on both axes), which sharpens the search and is more
// robust than fitting each axis independently.

export interface GnomonicPoint { xRow: number; xCol: number }

export interface GridLineSample {
  root: number; // this line's merge-group root (pipeline/bucketFillJoin.ts) -- lets a caller map a sample back to its actual composite line/pixels
  value: number; // rectified periodic coordinate -- dimensionless (a ratio of unit-vector dot products, effectively tan of an angle), NOT pixels or degrees. xCol for a row line, xRow for a column line (see gnomonic() below)
  weight: number; // projected arc length, same weighting fitPairOfPlanes already uses for its own votes
  index: number; // recovered integer row/column index: round((value - phase) / period)
  p1: GnomonicPoint; p2: GnomonicPoint; // this line's own two endpoints, gnomonically projected -- for drawing the rectified line directly, in the SAME coordinate space `value`/period/phase are expressed in
}

export interface PeriodSearchSample { period: number; score: number }

// `rank` is each gap's two composite lines' averaged position within their
// family's own value-sorted order, 0..1 -- the SAME rank convention
// overlays/hoverDebugOverlays.ts's drawVoteFamilyLines colors composite
// lines by (rank*255 in the blue or red channel), so a caller can color a
// gap tick to match the two lines' own colors instead of a flat family
// color, letting a tick be traced back to which pair of lines formed it.
export interface NeighborGapSample { gap: number; rank: number }

export interface GridPeriodPhaseResult {
  period: number;
  phiRow: number;
  phiCol: number;
  height: number | null; // cellPitch / period, null if cellPitch wasn't supplied
  rowLines: GridLineSample[];
  colLines: GridLineSample[];
  // Everything the debug plot (overlays/gridPeriodPhaseOverlays.ts) needs to
  // show the search's work: the [Pmin, Pmax] bracket, the full integer-count
  // resultant curve, the period finally chosen, and each image-tested
  // candidate's (period, resultant, distinctness) so the plot can mark the
  // rejected sub-multiples and why the winner won. The pooled all-pairs gap
  // histogram is NOT here -- it's an O(n^2) computation that fed only that
  // plot, so computePooledGaps below recomputes it on-demand from rowLines/
  // colLines, only when the plot is actually drawn.
  debug: {
    bracket: [number, number];
    coarseSamples: PeriodSearchSample[];
    chosenPeriod: number;
    candidates: { period: number; resultant: number; distinctness: number | null }[];
  };
}

// Folds `values` onto the unit circle at the given candidate `period` and
// returns how tightly they cluster (resultant, 0..1: 0 = no periodicity at
// this period, 1 = perfect) and the weighted circular-mean phase, in the
// SAME units as `values` (not radians -- inverted back through the same
// theta = 2*pi*value/period relationship used to fold them in).
// Exported so overlays/gridPeriodPhaseOverlays.ts can extend the search
// curve visualization outside the actual searched bracket PURELY for
// display (e.g. when the user pans/zooms past it) without duplicating this
// math or pretending it was ever part of the real bracketed search.
export function circularFit(values: number[], weights: number[], period: number): { resultant: number; phase: number } {
  let sumCos = 0, sumSin = 0, sumW = 0;
  for (let i = 0; i < values.length; i++) {
    const theta = (2 * Math.PI * values[i]) / period;
    sumCos += weights[i] * Math.cos(theta);
    sumSin += weights[i] * Math.sin(theta);
    sumW += weights[i];
  }
  if (sumW < 1e-9) return { resultant: 0, phase: 0 };
  const resultant = Math.hypot(sumCos, sumSin) / sumW;
  const phaseTheta = Math.atan2(sumSin, sumCos);
  const phase = (phaseTheta * period) / (2 * Math.PI);
  return { resultant, phase };
}

// Even-length-safe median, shared by the real seed derivation below and by
// overlays/gridPeriodPhaseOverlays.ts's own live preview of the same
// per-family medians (drawn while the user drags the gap-lower-bound
// slider, before the next real capture bakes a new value in via
// computeGridPeriodPhase's own `gapLowerBound` argument).
export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

// Gnomonic projection of a ray direction onto the plane tangent to the unit
// sphere at -Dnormal ("the bottom of the sphere") -- Drow/Dcol double as the
// tangent plane's own in-plane basis for free, since {Drow,Dcol,Dnormal}
// being orthonormal means "perpendicular to Dnormal" already IS span{Drow,
// Dcol}. null for a ray nearly parallel to the tangent plane (r.Dnormal ~
// 0), which shouldn't happen for a real ray toward the floor but is checked
// defensively.
function gnomonic(r: THREE.Vector3, Drow: THREE.Vector3, Dcol: THREE.Vector3, Dnormal: THREE.Vector3): GnomonicPoint | null {
  const dn = r.dot(Dnormal);
  if (Math.abs(dn) < 1e-9) return null;
  return { xRow: -r.dot(Drow) / dn, xCol: -r.dot(Dcol) / dn };
}

// Pooled pairwise gaps (row and column values pooled together, every
// combination -- not just adjacent ones), sorted ascending -- ONLY used by
// overlays/gridPeriodPhaseOverlays.ts's debug histogram plot, and not cheap
// (O(n^2) in the total line count). Deliberately NOT computed inside
// computeGridPeriodPhase itself (see GridPeriodPhaseResult's own comment on
// why) -- call this separately, only when that plot is actually being
// drawn, from its already-computed rowLines/colLines.
export function computePooledGaps(rowLines: readonly GridLineSample[], colLines: readonly GridLineSample[]): number[] {
  const allValues = [...rowLines.map((s) => s.value), ...colLines.map((s) => s.value)];
  const pooledGaps: number[] = [];
  for (let i = 0; i < allValues.length; i++) {
    for (let j = i + 1; j < allValues.length; j++) {
      const g = Math.abs(allValues[i] - allValues[j]);
      if (g > 1e-9) pooledGaps.push(g);
    }
  }
  pooledGaps.sort((a, b) => a - b);
  return pooledGaps;
}

// ── Cell-centre distinctness (harmonic disambiguation, idea 2) ────────────
//
// Reverse-projects a small patch of CELL CENTRES (phase + cellPitch/2) at a
// candidate period back to image pixels -- exactly buildDecodeSampleGrid's
// per-cell construction, with distance = cellPitch/P for THIS candidate -- and
// returns the mean absolute adjacent grayscale difference. High at the true
// period (neighbours are distinct De Bruijn cells), low at a sub-multiple
// (neighbours are the same cell, repeated). null if there's no cell pitch to
// turn a period into a distance, or too little of the patch lands on-image.
//
// A FACTORY rather than a plain function because everything except the period
// itself is fixed for a given capture -- the basis flip, the inverted
// quaternion, the patch centre medians -- and this gets called across a whole
// sweep of candidate periods, twice over: once by computeGridPeriodPhase for
// the handful of peaks it actually decides between, and again by
// overlays/gridPeriodPhaseOverlays.ts to draw the distinctness and
// resultant x distinctness curves across whatever range the plot is showing.
// That second caller is why this is exported at all: the debug curves adapt to
// pan/zoom, so they cannot be precomputed in the pipeline -- the view range
// isn't known until draw time, and it changes on every wheel event.
export interface CellCentreDistinctnessInputs {
  gray: Float64Array; w: number; h: number;
  quat: THREE.Quaternion; vFovRad: number; aspect: number;
  Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3;
  cellPitch: number | null; minGrazingCos: number;
  rowValues: number[]; colValues: number[]; // only for the patch-centre medians
}

// phiXCol/phiXRow are arguments rather than being recomputed inside, because
// every caller already runs circularFit at this period to get the resultant
// and the phase falls out of the same complex sum for free. Recomputing them
// here would double the trig work of any sweep that draws a curve.
export function makeCellCentreDistinctness(
  inp: CellCentreDistinctnessInputs,
): (P: number, phiXCol: number, phiXRow: number) => number | null {
  const { gray, w, h, quat, vFovRad, aspect, Drow, Dcol, Dnormal, cellPitch, minGrazingCos } = inp;
  const flipped = cornerDir(0, 0, quat, vFovRad, aspect).dot(Dnormal) > 0;
  const normalToFloor = flipped ? Dnormal.clone().negate() : Dnormal.clone();
  const invQuat = quat.clone().invert();
  const halfV = vFovRad / 2;
  const med = (arr: number[]) => (arr.length ? [...arr].sort((a, b) => a - b)[arr.length >> 1] : 0);
  const centreXRow = med(inp.colValues), centreXCol = med(inp.rowValues); // col family value = xRow, row family value = xCol
  const tanHalfV = Math.tan(halfV);

  return (P: number, phiXCol: number, phiXRow: number): number | null => {
    if (cellPitch === null || !(P > 0)) return null;
    const distance = cellPitch / P;
    const uvScale = flipped ? -distance : distance;
    // (u,v) = uvScale*(xRow, xCol); boundaries at uvScale*phi, centres +cellPitch/2 (matches buildDecodeSampleGrid).
    const uBoundaryRaw = uvScale * phiXRow, vBoundaryRaw = uvScale * phiXCol;
    const uPhase = (uBoundaryRaw - Math.round(uBoundaryRaw / cellPitch) * cellPitch) + cellPitch / 2;
    const vPhase = (vBoundaryRaw - Math.round(vBoundaryRaw / cellPitch) * cellPitch) + cellPitch / 2;
    const jC = Math.round((uvScale * centreXRow - uPhase) / cellPitch), iC = Math.round((uvScale * centreXCol - vPhase) / cellPitch);
    const M = 6; // (2M+1)^2 cell patch, centred on the visible region's median cell
    const p = new THREE.Vector3(), local = new THREE.Vector3();
    const patch: (number | null)[][] = [];
    for (let di = -M; di <= M; di++) {
      const rowArr: (number | null)[] = [];
      const v = vPhase + (iC + di) * cellPitch;
      for (let dj = -M; dj <= M; dj++) {
        const u = uPhase + (jC + dj) * cellPitch;
        p.copy(Drow).multiplyScalar(u).addScaledVector(Dcol, v).addScaledVector(normalToFloor, -distance);
        if (!(p.dot(normalToFloor) / p.length() < -minGrazingCos)) { rowArr.push(null); continue; }
        local.copy(p).applyQuaternion(invQuat);
        const ndcU = -local.x / (local.z * tanHalfV * aspect), ndcV = -local.y / (local.z * tanHalfV);
        const xx = Math.round(((ndcU + 1) / 2) * w), yy = Math.round(((1 - ndcV) / 2) * h);
        rowArr.push((xx >= 0 && xx < w && yy >= 0 && yy < h) ? gray[yy * w + xx] : null);
      }
      patch.push(rowArr);
    }
    let sum = 0, count = 0;
    for (let i = 0; i < patch.length; i++) for (let j = 0; j < patch[i].length; j++) {
      const a = patch[i][j]; if (a === null) continue;
      const right = patch[i][j + 1], down = i + 1 < patch.length ? patch[i + 1][j] : undefined;
      if (right != null) { sum += Math.abs(a - right); count++; }
      if (down != null) { sum += Math.abs(a - down); count++; }
    }
    return count < 8 ? null : sum / count; // too little on-image to judge
  };
}

// Async ONLY because of the coarse sweep's optional GPU dispatch (see
// pipelineGPU/periodSweep.ts). Everything else in here is pure and synchronous,
// and the CPU path never awaits anything real.
export async function computeGridPeriodPhase(
  composites: { root: number; line: CompositeLine }[],
  gray: Float64Array,
  w: number, h: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
  Drow: THREE.Vector3, Dcol: THREE.Vector3, Dnormal: THREE.Vector3,
  cellPitch: number | null,
  minGrazingCos: number,
): Promise<GridPeriodPhaseResult | null> {
  // `composites` comes from pipeline/votes.ts's computeGradient2x2Composites
  // -- the SAME composite lines (same root numbering) computeSegmentVotes
  // casts its own votes from, computed exactly once per reconstruction pass
  // (see pipeline/axesReconstruction.ts). Used to rebuild them here from
  // scratch instead; that recompute agreed with computeSegmentVotes (same
  // inputs, deterministic) but NOT with the "color composite lines by
  // row/col family" debug overlay, which correlated against a third,
  // independently-recomputed set of composites built over a DIFFERENT field
  // entirely -- a different segmentation whose root numbers only matched
  // this one by coincidence.
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];

  // Step 1+2: classify each composite line as row- or column-type via its
  // own vote normal (orientation only -- same |n.Drow| vs |n.Dcol| test the
  // plane-pair fit itself relies on), then rectify to one scalar via the
  // gnomonic projection of its own two endpoints -- NOT the normal-vector
  // ratio, see this file's header for why the two forms don't mix.
  const classifySpan = spanStart('classify lines (cornerDir+gnomonic x2 per line)');
  const rowSamples: { root: number; value: number; weight: number; p1: GnomonicPoint; p2: GnomonicPoint }[] = [];
  const colSamples: { root: number; value: number; weight: number; p1: GnomonicPoint; p2: GnomonicPoint }[] = [];
  for (const { root, line } of composites) {
    const [u1, v1] = toNDC(line.x1, line.y1);
    const [u2, v2] = toNDC(line.x2, line.y2);
    const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
    const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
    const nRaw = ray1.clone().cross(ray2);
    const arcLen = nRaw.length();
    if (arcLen < 1e-12) continue;
    const n = nRaw.divideScalar(arcLen);

    const p1 = gnomonic(ray1, Drow, Dcol, Dnormal);
    const p2 = gnomonic(ray2, Drow, Dcol, Dnormal);
    if (!p1 || !p2) continue; // ray parallel to the tangent plane -- shouldn't happen for a real floor ray, skip defensively

    if (Math.abs(n.dot(Drow)) < Math.abs(n.dot(Dcol))) {
      // Row-type: every point on this line shares the same xCol (that's
      // what "projects to a straight, constant-xCol line" means) --
      // averaging both endpoints is a cheap noise-robustness bonus over
      // picking just one, since in theory they'd already be identical.
      rowSamples.push({ root, value: (p1.xCol + p2.xCol) / 2, weight: arcLen, p1, p2 });
    } else {
      colSamples.push({ root, value: (p1.xRow + p2.xRow) / 2, weight: arcLen, p1, p2 });
    }
  }
  spanEnd(classifySpan);
  if (rowSamples.length === 0 && colSamples.length === 0) return null;

  // ── Seed-free period search: integer-count candidates -> image-content ────
  // ── harmonic disambiguation -> golden-section polish ──────────────────────
  //
  // Replaces the old median-neighbor-gap seed + 25x-disagreement bracket (and
  // the brief "largest-period peak wins" + 0.6 heuristic that first replaced
  // them) -- see this session's decode-robustness work and the A/B bake-off.
  //
  //  (1) NO SEED, PHYSICALLY-BOUNDED, INTEGER-COUNT CANDIDATES. A detected line
  //      IS a grid line, and the two extreme detected lines sit an INTEGER
  //      number of periods apart, so the only physically-possible periods are
  //      P_n = spread/n for integer n (= number of periods spanning the data).
  //      n runs from 3 (fewer isn't credibly periodic; decode needs ORDER=5
  //      cells anyway) up to the looser of two independent physical bounds --
  //      the board's own cell count, and a maximum assumed camera height -- see
  //      the nMax derivation below. Both track the board-size slider live. The
  //      candidate set falls straight out of the geometry: no seed, no tunable
  //      width, no arbitrary sample count.
  //
  //  (2) HARMONIC DISAMBIGUATION BY IMAGE CONTENT (replaces "largest peak" +
  //      its 0.6 fudge). The circular resultant is high at the true period P0
  //      AND at every sub-multiple P0/2, P0/3 (a lattice at spacing P0 trivially
  //      lies on any finer sub-lattice), so the resultant alone can't separate
  //      them. But a sub-multiple OVERSAMPLES -- each real cell becomes a kxk
  //      block of IDENTICAL samples. The De Bruijn pattern is locally unique, so
  //      at the TRUE period adjacent cell CENTRES differ while at a sub-multiple
  //      they repeat. So for the top few resultant peaks we sample grayscale at
  //      cell centres (phase + HALF a period -- the same +cell/2 offset decode
  //      uses to hit solid interiors, NOT the gray edges the phase sits on) and
  //      pick the period whose neighbours are most DISTINCT. Bonus: this earns
  //      compute back downstream -- it keeps decode fed with the coarsest
  //      correct period, not a sub-multiple that would balloon its grid 4-9x.
  //
  //  (3) GOLDEN-SECTION polish on the resultant within the winner's one-count
  //      neighbourhood -- no parabola-shape assumption.
  //
  // KNOWN LIMIT / ALTERNATIVE (Option B): all of this still runs on the SPARSE
  // detected lines. Under heavy line dropout at extreme grazing (measured ~70
  // lines vs ~400, at 40-80 cells up / -18..-25deg) the sparse set can grow a
  // genuine longer-period spurious structure -- which also has HIGH distinctness
  // (it UNDER-samples, so neighbours still differ), so (2) won't catch it. That
  // regime needs a DENSE per-pixel gnomonic histogram + autocorrelation, which
  // degrades gracefully (~25-35% error) instead of exploding; measured to beat
  // this only at those extremes, deferred as the fallback if an elevated/raked
  // camera regime ever matters. See the A/B bake-off.
  const searchSpan = spanStart('period search (integer-count + distinctness + golden)');
  const rowValues = rowSamples.map((s) => s.value), rowWeights = rowSamples.map((s) => s.weight);
  const colValues = colSamples.map((s) => s.value), colWeights = colSamples.map((s) => s.weight);
  // PER-FAMILY extent, then the larger of the two -- NOT the extent of the two
  // pooled together, which is what this used to compute and was a real bug.
  // A row line's `value` is its xCol coordinate and a column line's is its
  // xRow (see GridLineSample) -- two DIFFERENT axes. Pooling them and taking
  // min/max measures the extent of the union of two unrelated coordinate sets,
  // which exceeds either one whenever the camera's nadir is off-centre
  // asymmetrically between the axes (generic for anything oblique). Example:
  // nadir at the floor origin with the board at u in [0,32], v in [-16,16]
  // gives xRow in [0,32/d] and xCol in [-16/d,16/d] -- each family spans 32
  // cells, but the pooled extent spans 48. Since nMax below is a PER-FAMILY
  // cell count, that inflated numerator pushed the true period left of the
  // bracket entirely and the search never evaluated it.
  const extentOf = (vals: number[]) => {
    if (vals.length < 2) return 0;
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return hi - lo;
  };
  const spread = Math.max(extentOf(rowValues), extentOf(colValues));
  if (!(spread > 1e-9)) return null; // no extent to search over (all lines coincident)
  const resultantAt = (P: number) => circularFit(rowValues, rowWeights, P).resultant + circularFit(colValues, colWeights, P).resultant;

  // Golden-section maximizer -- unimodal near the resultant peak, no parabola
  // assumption. Returns the argmax within [a0, b0].
  const goldenSectionMax = (f: (x: number) => number, a0: number, b0: number, iters = 14): number => {
    const gr = (Math.sqrt(5) - 1) / 2;
    let a = a0, b = b0, c = b0 - gr * (b0 - a0), d = a0 + gr * (b0 - a0);
    let fc = f(c), fd = f(d);
    for (let i = 0; i < iters; i++) {
      if (fc > fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f(c); }
      else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f(d); }
    }
    return (a + b) / 2;
  };

  const cellCentreDistinctness = makeCellCentreDistinctness({
    gray, w, h, quat, vFovRad, aspect, Drow, Dcol, Dnormal, cellPitch, minGrazingCos,
    rowValues, colValues,
  });

  // (1) integer-count candidate curve, ASCENDING period order (n descending).
  // Two INDEPENDENT physical bounds on how many periods can span the data, and
  // the search uses whichever is looser so a shortfall in one can't hide the
  // true period from the other.
  //
  // BOARD BOUND: n is exactly "how many cells the detected lines span" (spread
  // = n*P and P = cellPitch/d, so n = spread/P = visible world extent /
  // cellPitch), and you can't span more cells of a family than the board has.
  // Tight and correct -- but only once `spread` is per-family, which it now is.
  //
  // HEIGHT BOUND: assume the camera never gets further from the floor than
  // MAX_CAMERA_HEIGHT_IN_BOARDS times the board's own linear size. Since
  // P = cellPitch/d, a maximum distance is a MINIMUM period:
  //   d <= H * N * cellPitch   =>   P >= cellPitch / (H * N * cellPitch) = 1/(H*N)
  // Note cellPitch cancels -- the floor on P is a pure function of the board's
  // cell count and the height assumption, independent of world scale and of
  // whether a cellPitch was even supplied. Converting that floor to a count
  // gives n <= spread * H * N.
  //
  // These agree at exactly the assumed maximum height and the height bound is
  // the looser one below it. That is where the headroom comes from, and it is
  // physically argued rather than a tuning fudge -- real effects (axis-fit
  // error, the rectification not being perfect, detection noise near the
  // extremes) can push the APPARENT count slightly past the board's own hard
  // limit, and the board bound alone leaves nowhere to go.
  //
  // Worth knowing what `spread` actually is here: it's ANGULAR, not metric.
  // The gnomonic extent of the field of view is ~2*tan(halfVFov) regardless of
  // camera height (values are u/d, and both scale together), so it sits near 1
  // for a typical phone -- a 65deg/4:3 camera gives ~0.96, making the height
  // bound ~1.4x the board bound. A very wide lens pushes it further (~2.9x at
  // 120deg), which is why the result is capped: n can never physically exceed
  // boardCells (you can't span more cells than the board has), so everything
  // past it is headroom, and HEADROOM_CAP bounds both how much of it we buy
  // and the cost -- every extra n is two more circularFits over every detected
  // line, and this sweep is already the search's main expense.
  const MAX_CAMERA_HEIGHT_IN_BOARDS = 1.5;
  const HEADROOM_CAP = 2; // multiples of boardCells; 2x always covers the hard bound with room to spare
  const boardCells = Math.max(R, C);
  const nFromHeight = Math.floor(MAX_CAMERA_HEIGHT_IN_BOARDS * boardCells * spread);
  const nMin = 3;
  const nMax = Math.min(HEADROOM_CAP * boardCells, Math.max(boardCells, nFromHeight));
  const bracket: [number, number] = [spread / nMax, spread / nMin];

  // The sweep itself -- ~256-512 candidates x 2 families x every detected line,
  // and the search's dominant cost. Optionally dispatched to GPU (one workgroup
  // per candidate, see pipelineGPU/periodSweep.ts); the CPU loop below stays the
  // source of truth and the fallback.
  //
  // The GPU gets values PRE-SCALED to [0,1] as (value - familyMin)/spread,
  // divided in f64 here rather than in f32 there. That is a real precision
  // requirement, not tidiness: period = spread/n shrinks as n grows, so
  // value/period reaches ~nMax (~512), where f32 has only ~3-4 digits left after
  // the point. Each family gets its OWN min because a constant shift rotates
  // every one of that family's unit vectors equally and so cannot change its
  // resultant -- it would move the PHASE, which is exactly why the GPU never
  // computes phase and circularFit still runs on CPU for the chosen candidates.
  const sweepSpan = spanStart(globalState.forceCPU ? 'period sweep (CPU)' : 'period sweep (GPU)');
  let scores: Float32Array | null = null;
  if (!globalState.forceCPU) {
    const total = rowValues.length + colValues.length;
    const scaled = new Float32Array(total), weights = new Float32Array(total);
    let at = 0;
    for (const [vals, wts] of [[rowValues, rowWeights], [colValues, colWeights]] as const) {
      let lo = Infinity;
      for (const v of vals) if (v < lo) lo = v;
      for (let i = 0; i < vals.length; i++) { scaled[at] = (vals[i] - lo) / spread; weights[at] = wts[i]; at++; }
    }
    scores = await sweepResultantsGPU(scaled, weights, rowValues.length, colValues.length, nMin, nMax);
  }
  const coarseSamples: PeriodSearchSample[] = [];
  for (let n = nMax, i = 0; n >= nMin; n--, i++) {
    coarseSamples.push({ period: spread / n, score: scores ? scores[i] : resultantAt(spread / n) });
  }
  spanEnd(sweepSpan);

  // interior local-maxima of the resultant, restricted to REAL periodicity
  // peaks. Two signals are needed and neither suffices alone:
  //   - distinctness (below) only demotes SUB-multiples (short periods, whose
  //     cell centres repeat). It does NOT demote spurious LONG periods -- those
  //     under-sample, so their neighbours still differ and they look "distinct"
  //     too. Nor does it demote pure noise bumps.
  //   - the resultant demotes exactly those (a spurious-long or noise period
  //     has LOW resultant -- the lines don't fold coherently there), but can't
  //     tell the true period from its sub-multiples (both fold perfectly).
  // So: keep only peaks whose resultant clears SIGNIFICANCE x the best (this is
  // a real-vs-noise cut with a ~10x margin -- true/sub-multiple peaks sit at
  // ~0.7-1.0 of max, noise/spurious-long at <0.2 -- NOT a fiddly decision
  // boundary; distinctness makes the actual decision among the survivors).
  const SIGNIFICANCE = 0.5;
  const MAX_CANDIDATES = 6; // compute budget for the (cheap) distinctness test
  let gMax = 0; for (const s of coarseSamples) if (s.score > gMax) gMax = s.score;
  const peaks: { period: number; idx: number; score: number }[] = [];
  for (let i = 1; i < coarseSamples.length - 1; i++) {
    const s = coarseSamples[i];
    if (s.score > coarseSamples[i - 1].score && s.score >= coarseSamples[i + 1].score && s.score >= SIGNIFICANCE * gMax) {
      peaks.push({ period: s.period, idx: i, score: s.score });
    }
  }
  peaks.sort((a, b) => b.score - a.score);
  const candidatePeaks = peaks.slice(0, MAX_CANDIDATES);

  // (2) among the real periodicity peaks, pick the MOST DISTINCT (the
  // fundamental; its sub-multiples repeat and score lower). Falls back to the
  // largest-period significant peak when distinctness can't be sampled (no
  // cell pitch, or too little on-image), or the plain global best if the
  // significance cut left nothing.
  const candidates: { period: number; resultant: number; distinctness: number | null }[] = [];
  let chosenIdx = -1, bestDistinct = -Infinity;
  for (const cand of candidatePeaks) {
    const d = cellCentreDistinctness(
      cand.period,
      circularFit(rowValues, rowWeights, cand.period).phase,
      circularFit(colValues, colWeights, cand.period).phase,
    );
    candidates.push({ period: cand.period, resultant: cand.score, distinctness: d });
    if (d !== null && d > bestDistinct) { bestDistinct = d; chosenIdx = cand.idx; }
  }
  if (chosenIdx < 0) {
    if (candidatePeaks.length > 0) { let largestP = -1; for (const cand of candidatePeaks) if (cand.period > largestP) { largestP = cand.period; chosenIdx = cand.idx; } }
    else { let gBest = -1; for (let i = 0; i < coarseSamples.length; i++) if (coarseSamples[i].score > gBest) { gBest = coarseSamples[i].score; chosenIdx = i; } }
  }
  const chosenCandidatePeriod = coarseSamples[chosenIdx].period; // pre-refine -- what the debug plot marks as "chosen"
  let bestP = chosenCandidatePeriod;

  // (3) golden-section refine on the resultant within the chosen count's
  // immediate neighbourhood [P(n+1), P(n-1)].
  if (chosenIdx > 0 && chosenIdx < coarseSamples.length - 1) {
    bestP = goldenSectionMax(resultantAt, coarseSamples[chosenIdx - 1].period, coarseSamples[chosenIdx + 1].period);
  }
  spanEnd(searchSpan);

  // Step 5: final phases at the winning period.
  const finalRow = circularFit(rowValues, rowWeights, bestP);
  const finalCol = circularFit(colValues, colWeights, bestP);

  // Step 6: assign an integer row/column index to every detected line, and
  // convert the period to a camera-height estimate if the physical cell
  // pitch is known (h = cellPitch / period).
  const rowLines: GridLineSample[] = rowSamples.map((s) => ({
    root: s.root, value: s.value, weight: s.weight, index: Math.round((s.value - finalRow.phase) / bestP), p1: s.p1, p2: s.p2,
  }));
  const colLines: GridLineSample[] = colSamples.map((s) => ({
    root: s.root, value: s.value, weight: s.weight, index: Math.round((s.value - finalCol.phase) / bestP), p1: s.p1, p2: s.p2,
  }));

  return {
    period: bestP, phiRow: finalRow.phase, phiCol: finalCol.phase,
    height: cellPitch !== null ? cellPitch / bestP : null,
    rowLines, colLines,
    debug: { bracket, coarseSamples, chosenPeriod: chosenCandidatePeriod, candidates },
  };
}
