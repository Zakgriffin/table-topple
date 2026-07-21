import * as THREE from 'three';
import { ProfileSpan, spanEnd, spanStart } from '../profiling/profiler.ts';
import { OrientationFit, Vote } from '../types.ts';

// ── Orientation refinement (Levenberg-Marquardt) ─────────────────────────

export function fourFoldResidual(n: THREE.Vector3, Drow: THREE.Vector3, Dcol: THREE.Vector3): number {
  const psi = Math.atan2(n.dot(Dcol), n.dot(Drow));
  return Math.sin(4 * psi);
}

export function orientationCost(votes: Vote[], Drow: THREE.Vector3, Dcol: THREE.Vector3): number {
  let cost = 0;
  for (const { n, weight } of votes) {
    const r = weight * fourFoldResidual(n, Drow, Dcol);
    cost += r * r;
  }
  return cost;
}

export function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
    if (Math.abs(M[pivot][col]) < 1e-18) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let k = col; k <= n; k++) M[row][k] -= f * M[col][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let s = M[row][n];
    for (let k = row + 1; k < n; k++) s -= M[row][k] * x[k];
    x[row] = s / M[row][row];
  }
  return x;
}

export function refineOrientationLM(votes: Vote[], initial: OrientationFit, maxIterations = 20): OrientationFit & { iterations: number; initialCost: number; finalCost: number } {
  const q = new THREE.Quaternion();
  const Drow0 = initial.Drow.clone(), Dcol0 = initial.Dcol.clone(), Dnormal0 = initial.Dnormal.clone();
  const candidateDrow = (qq: THREE.Quaternion) => Drow0.clone().applyQuaternion(qq);
  const candidateDcol = (qq: THREE.Quaternion) => Dcol0.clone().applyQuaternion(qq);

  const initialCost = orientationCost(votes, Drow0, Dcol0);
  let cost = initialCost;
  let lambda = 1e-3;
  const EPS = 1e-5;
  const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];

  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    const iterSpan: ProfileSpan | null = spanStart(`LM iter ${iterations}`);
    try {
      const Drow = candidateDrow(q), Dcol = candidateDcol(q);
      const n = votes.length;
      const residuals = new Float64Array(n);
      const residSpan = spanStart('residuals+jacobian');
      for (let i = 0; i < n; i++) residuals[i] = votes[i].weight * fourFoldResidual(votes[i].n, Drow, Dcol);

      const J: Float64Array[] = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
      for (let k = 0; k < 3; k++) {
        const qPlus = new THREE.Quaternion().setFromAxisAngle(axes[k], EPS).multiply(q);
        const DrowP = candidateDrow(qPlus), DcolP = candidateDcol(qPlus);
        for (let i = 0; i < n; i++) {
          const rP = votes[i].weight * fourFoldResidual(votes[i].n, DrowP, DcolP);
          J[k][i] = (rP - residuals[i]) / EPS;
        }
      }
      spanEnd(residSpan);

      const solveSpan = spanStart('JtJ/solve');
      const JtJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      const Jtr = [0, 0, 0];
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          let s = 0;
          for (let i = 0; i < n; i++) s += J[a][i] * J[b][i];
          JtJ[a][b] = s;
        }
        let s = 0;
        for (let i = 0; i < n; i++) s += J[a][i] * residuals[i];
        Jtr[a] = s;
      }
      const A = JtJ.map((row, a) => row.map((v, b) => v + (a === b ? lambda * (JtJ[a][a] || 1) : 0)));
      const rhs = Jtr.map((v) => -v);
      const delta = solveLinearSystem(A, rhs);
      spanEnd(solveSpan);
      if (!delta) break;

      const deltaVec = new THREE.Vector3(delta[0], delta[1], delta[2]);
      const deltaAngle = deltaVec.length();
      if (deltaAngle < 1e-10) break;
      const deltaAxis = deltaVec.normalize();
      const qTry = new THREE.Quaternion().setFromAxisAngle(deltaAxis, deltaAngle).multiply(q).normalize();

      const DrowTry = candidateDrow(qTry), DcolTry = candidateDcol(qTry);
      const tryCost = orientationCost(votes, DrowTry, DcolTry);
      if (tryCost < cost) {
        q.copy(qTry);
        cost = tryCost;
        lambda = Math.max(lambda * 0.5, 1e-8);
      } else {
        lambda = Math.min(lambda * 3, 1e8);
      }
    } finally {
      spanEnd(iterSpan);
    }
  }

  return {
    Drow: candidateDrow(q), Dcol: candidateDcol(q), Dnormal: Dnormal0.clone().applyQuaternion(q),
    iterations, initialCost, finalCost: cost,
  };
}

