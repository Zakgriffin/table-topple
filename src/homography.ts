// Fits and applies a planar homography mapping lattice (col,row) <-> image
// (x,y). A flat grid under perspective is exactly a homography (8 DOF) —
// used by the line-based rectification pipeline (src/lattice.ts's
// buildLatticeCorrespondences feeds fitHomographyDLT) to turn every indexed
// row/col line crossing into a single global model, letting every lattice
// cell be addressed directly (rectify + round) rather than by chaining local
// neighbor hops.

export type Mat3 = Float64Array; // row-major 3x3, 9 entries

export interface PointCorrespondence { u: number; v: number; x: number; y: number; }

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
// system is poorly conditioned.
function normalize(pts: { x: number; y: number }[]): { normalized: { x: number; y: number }[]; T: Mat3 } {
  let cx = 0, cy = 0;
  for (const p of pts) { cx += p.x; cy += p.y; }
  cx /= pts.length; cy /= pts.length;
  let avgDist = 0;
  for (const p of pts) avgDist += Math.hypot(p.x - cx, p.y - cy);
  avgDist /= pts.length;
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
// correspondences are too few or too degenerate (e.g. collinear).
export function fitHomographyDLT(correspondences: PointCorrespondence[]): Mat3 | null {
  if (correspondences.length < 4) return null;

  const { normalized: uv, T: T1 } = normalize(correspondences.map(c => ({ x: c.u, y: c.v })));
  const { normalized: xy, T: T2 } = normalize(correspondences.map(c => ({ x: c.x, y: c.y })));

  // h22 = 1 (inhomogeneous DLT): 2 equations per correspondence in 8 unknowns.
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < correspondences.length; i++) {
    const { x: u, y: v } = uv[i];
    const { x, y } = xy[i];
    A.push([u, v, 1, 0, 0, 0, -u * x, -v * x]); b.push(x);
    A.push([0, 0, 0, u, v, 1, -u * y, -v * y]); b.push(y);
  }
  // Normal equations: (A^T A) h = A^T b — 8x8, cheap for a handful of points.
  const n = 8;
  const AtA: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const Atb: number[] = new Array(n).fill(0);
  for (let i = 0; i < A.length; i++) {
    for (let r = 0; r < n; r++) {
      Atb[r] += A[i][r] * b[i];
      for (let c = 0; c < n; c++) AtA[r][c] += A[i][r] * A[i][c];
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
