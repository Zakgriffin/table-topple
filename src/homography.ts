// Fits and applies a planar homography mapping lattice (col,row) <-> image
// (x,y). A flat grid under perspective is exactly a homography (8 DOF) —
// used by the line-based rectification pipeline (src/lattice.ts's
// buildLatticeCorrespondences feeds fitHomographyDLT) to turn every indexed
// row/col line crossing into a single global model, letting every lattice
// cell be addressed directly (rectify + round) rather than by chaining local
// neighbor hops.

export type Mat3 = Float64Array; // row-major 3x3, 9 entries

// weight is optional (defaults to 1, i.e. unweighted) -- src/lattice.ts's
// buildLatticeCorrespondences sets it from the originating lines' Hough vote
// mass (see src/lines.ts's LineCandidate.weight), so a crossing built from
// two confidently-detected lines counts more in the fit than one involving a
// line that barely cleared the peak threshold.
export interface PointCorrespondence { u: number; v: number; x: number; y: number; weight?: number; }

function mat3Multiply(a: Mat3, b: Mat3): Mat3 {
  const out = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[r * 3 + k] * b[k * 3 + c];
      out[r * 3 + c] = s;
    }
  }
  return out;
}

function mat3Invert(m: Mat3): Mat3 | null {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return null;
  const D = -(b * i - c * h), E = a * i - c * g, F = -(a * h - b * g);
  const G = b * f - c * e, H = -(a * f - c * d), I = a * e - b * d;
  const inv = new Float64Array(9);
  const invDet = 1 / det;
  inv[0] = A * invDet; inv[1] = D * invDet; inv[2] = G * invDet;
  inv[3] = B * invDet; inv[4] = E * invDet; inv[5] = H * invDet;
  inv[6] = C * invDet; inv[7] = F * invDet; inv[8] = I * invDet;
  return inv;
}

// Solves the general linear system A x = b (A is n x n, row-major) via
// Gaussian elimination with partial pivoting. Small, general-purpose — used
// here for the 8x8 normal-equations solve in fitHomographyDLT.
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    if (Math.abs(M[pivot][col]) < 1e-12) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

// Hartley normalization: translate to centroid, scale so average distance
// from origin is sqrt(2). Standard prerequisite for a numerically stable
// DLT fit — without it, lattice coords (small integers) and pixel coords
// (hundreds of units) sit at wildly different scales and the least-squares
// system is poorly conditioned. Weighted so a low-confidence point doesn't
// pull the centroid/scale estimate as much as a high-confidence one, same
// reasoning as the weighted fit itself below.
function normalize(pts: { x: number; y: number; weight?: number }[]): { normalized: { x: number; y: number }[]; T: Mat3 } {
  let cx = 0, cy = 0, wSum = 0;
  for (const p of pts) { const w = p.weight ?? 1; cx += w * p.x; cy += w * p.y; wSum += w; }
  cx /= wSum; cy /= wSum;
  let avgDist = 0;
  for (const p of pts) avgDist += (p.weight ?? 1) * Math.hypot(p.x - cx, p.y - cy);
  avgDist /= wSum;
  const scale = avgDist > 1e-9 ? Math.SQRT2 / avgDist : 1;
  const T: Mat3 = new Float64Array([scale, 0, -scale * cx, 0, scale, -scale * cy, 0, 0, 1]);
  const normalized = pts.map(p => ({ x: scale * (p.x - cx), y: scale * (p.y - cy) }));
  return { normalized, T };
}

function applyMat3(m: Mat3, x: number, y: number): [number, number] | null {
  const w = m[6] * x + m[7] * y + m[8];
  if (Math.abs(w) < 1e-12) return null;
  return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w];
}

// Fits a homography H (lattice (u,v) -> image (x,y)) from >= 4 point
// correspondences via DLT with Hartley normalization. Returns null if the
// correspondences are too few or too degenerate (e.g. collinear). Each
// correspondence's optional weight scales its two equations' contribution to
// the normal-equations accumulation below — unweighted (weight 1) unless the
// caller set one, so existing behavior is unchanged for callers that don't
// care about per-point confidence.
export function fitHomographyDLT(correspondences: PointCorrespondence[]): Mat3 | null {
  if (correspondences.length < 4) return null;

  const { normalized: uv, T: T1 } = normalize(correspondences.map(c => ({ x: c.u, y: c.v, weight: c.weight })));
  const { normalized: xy, T: T2 } = normalize(correspondences.map(c => ({ x: c.x, y: c.y, weight: c.weight })));

  // h22 = 1 (inhomogeneous DLT): 2 equations per correspondence in 8 unknowns.
  const A: number[][] = [];
  const b: number[] = [];
  const weights: number[] = [];
  for (let i = 0; i < correspondences.length; i++) {
    const { x: u, y: v } = uv[i];
    const { x, y } = xy[i];
    const w = correspondences[i].weight ?? 1;
    A.push([u, v, 1, 0, 0, 0, -u * x, -v * x]); b.push(x); weights.push(w);
    A.push([0, 0, 0, u, v, 1, -u * y, -v * y]); b.push(y); weights.push(w);
  }
  // Normal equations: (A^T W A) h = A^T W b — 8x8, cheap for a handful of points.
  const n = 8;
  const AtA: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const Atb: number[] = new Array(n).fill(0);
  for (let i = 0; i < A.length; i++) {
    const w = weights[i];
    for (let r = 0; r < n; r++) {
      Atb[r] += w * A[i][r] * b[i];
      for (let c = 0; c < n; c++) AtA[r][c] += w * A[i][r] * A[i][c];
    }
  }

  const h = solveLinearSystem(AtA, Atb);
  if (!h) return null;
  const Hn: Mat3 = new Float64Array([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1]);

  // Denormalize: H = T2^-1 * Hn * T1
  const T2inv = mat3Invert(T2);
  if (!T2inv) return null;
  return mat3Multiply(mat3Multiply(T2inv, Hn), T1);
}

export function applyHomography(H: Mat3, u: number, v: number): [number, number] | null {
  return applyMat3(H, u, v);
}

export function invertHomography(H: Mat3): Mat3 | null {
  return mat3Invert(H);
}

// Fits once via fitHomographyDLT, then rejects whichever correspondences
// that initial fit reprojects worst and refits from the survivors — the
// same "fit -> find inliers -> refit from inliers only" shape already used
// by src/lattice.ts's Mobius fit, one level up. Found via real end-to-end
// testing (scripts/test-lines-decode.ts): a handful of correspondences built
// from weak/marginal lines (see PointCorrespondence.weight) can drag an
// otherwise-good DLT fit measurably off, even though every individual stage
// (line detection, index recovery) tests as accurate in isolation against
// clean synthetic input — real detected lines carry correlated, non-Gaussian
// noise that isolated per-stage tests don't reproduce, and this compounds
// enough across a whole grid's worth of correspondences to matter.
//
// The rejection threshold is a multiple of the fit's OWN median reprojection
// error rather than a fixed pixel constant: a good fit's residuals are
// naturally small, so even a modest absolute error is a meaningful outlier
// relative to it, while a noisier scene's larger median residual needs more
// slack before flagging anything — self-calibrating per fit, no new tunable
// knob to carry around. The small floor guards the degenerate case where the
// median is already ~0 (a very clean fit), which would otherwise reject
// almost everything over trivial noise.
export function fitHomographyRobust(correspondences: PointCorrespondence[]): Mat3 | null {
  let H = fitHomographyDLT(correspondences);
  if (!H || correspondences.length < 8) return H; // too few points to safely reject any

  // Repeats reject-and-refit up to 3 rounds: the FIRST round's fit is itself
  // dragged toward any outliers, so residuals measured against it can still
  // underestimate how wrong an outlier really is -- each subsequent round
  // re-measures against a progressively cleaner fit, converging on which
  // points are genuinely inconsistent rather than judging them all against
  // a single already-biased baseline. Stops early once a round rejects
  // nothing new.
  let current = correspondences;
  for (let round = 0; round < 3; round++) {
    const residuals = current.map(c => {
      const p = applyMat3(H!, c.u, c.v);
      return p ? Math.hypot(p[0] - c.x, p[1] - c.y) : Infinity;
    });
    const finite = residuals.filter(Number.isFinite).slice().sort((a, b) => a - b);
    const median = finite.length ? finite[Math.floor(finite.length / 2)] : 0;
    const threshold = Math.max(2, median * 4);

    const survivors = current.filter((_, i) => residuals[i] <= threshold);
    if (survivors.length < 8 || survivors.length === current.length) break;
    const refit = fitHomographyDLT(survivors);
    if (!refit) break;
    H = refit;
    current = survivors;
  }
  return H;
}
