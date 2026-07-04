// Generic small linear-algebra helpers shared by src/vp.ts (vanishing point
// estimation, a 3x3 problem) and src/lattice.ts (1D Mobius line-index
// recovery, a 4x4 problem) — both reduce to "find the eigenvector of a small
// symmetric matrix's SMALLEST eigenvalue" (the least-squares null space of a
// set of homogeneous linear constraints), just at different sizes, so this
// is factored out once rather than duplicated per size.

// Classic cyclic Jacobi eigenvalue algorithm for symmetric n x n matrices.
// Fine for the tiny (3x3, 4x4) fixed sizes used here — avoids pulling in a
// linear-algebra dependency for what's always a small, fixed-size problem.
export function jacobiEigenSymmetric(Ain: number[][]): { values: number[]; vectors: number[][] } {
  const n = Ain.length;
  const A = Ain.map(row => row.slice());
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        const app = A[p][p], aqq = A[q][q], apq = A[p][q];
        A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        A[p][q] = 0; A[q][p] = 0;
        for (let k = 0; k < n; k++) {
          if (k !== p && k !== q) {
            const akp = A[k][p], akq = A[k][q];
            A[k][p] = c * akp - s * akq; A[p][k] = A[k][p];
            A[k][q] = s * akp + c * akq; A[q][k] = A[k][q];
          }
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = A.map((row, i) => row[i]);
  const vectors = values.map((_, k) => V.map(row => row[k]));
  return { values, vectors };
}

// Eigenvector of the SMALLEST eigenvalue — the recurring operation both
// estimateVanishingPoint and fitMobius1D actually need; sized generically
// from M's dimensions.
export function smallestEigenvector(M: number[][]): number[] {
  const { values, vectors } = jacobiEigenSymmetric(M);
  let minIdx = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[minIdx]) minIdx = i;
  return vectors[minIdx];
}
