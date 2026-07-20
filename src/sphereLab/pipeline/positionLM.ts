import * as THREE from 'three';
import { cornerDir } from '../math/geometry.ts';
import { C, R, torus } from '../scene/floor.ts';
import { Marginals, OrientationFit, PhotometricSample, PositionFit } from '../types.ts';
import { hsvToRgb } from './distortion.ts';
import { solveLinearSystem } from './orientationLM.ts';

// ── Phase 3 (Option B): joint orientation + ABSOLUTE position refinement ─

export function torusBrightness(row: number, col: number): number {
  const r = ((row % R) + R) % R, c = ((col % C) + C) % C;
  return torus[r][c] ? 20 : 235;
}

export function predictedBilinear(worldX: number, worldZ: number): number {
  const xf = worldX + C / 2 - 0.5, zf = worldZ + R / 2 - 0.5;
  const c0 = Math.floor(xf), r0 = Math.floor(zf);
  const fx = xf - c0, fz = zf - r0;
  const b00 = torusBrightness(r0, c0), b10 = torusBrightness(r0, c0 + 1);
  const b01 = torusBrightness(r0 + 1, c0), b11 = torusBrightness(r0 + 1, c0 + 1);
  return b00 * (1 - fx) * (1 - fz) + b10 * fx * (1 - fz) + b01 * (1 - fx) * fz + b11 * fx * fz;
}

export function computePhotometricSamples(gray: Float64Array, w: number, h: number, stride: number): PhotometricSample[] {
  const samples: PhotometricSample[] = [];
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      samples.push({ px: x, py: y, observed: gray[y * w + x] });
    }
  }
  return samples;
}

export function refineOrientationAndPositionLM(
  samples: PhotometricSample[], w: number, h: number,
  initial: OrientationFit, distance: number, initialWorldX0: number, initialWorldZ0: number,
  camQuat: THREE.Quaternion, vFovRad: number, aspect: number,
  maxIterations = 20,
): PositionFit & { iterations: number; initialCost: number; finalCost: number } {
  const q = new THREE.Quaternion();
  const Drow0 = initial.Drow.clone(), Dcol0 = initial.Dcol.clone(), Dnormal0 = initial.Dnormal.clone();
  let worldX0 = initialWorldX0, worldZ0 = initialWorldZ0;
  const MIN_GRAZING_COS = 0.15;
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, (py / h) * 2 - 1];

  const candidateNormal = (qq: THREE.Quaternion) => {
    const n = Dnormal0.clone().applyQuaternion(qq);
    if (cornerDir(0, 0, camQuat, vFovRad, aspect).dot(n) > 0) n.negate();
    return n;
  };

  function residualsFor(qq: THREE.Quaternion, wx0: number, wz0: number): Float64Array {
    const Drow = Drow0.clone().applyQuaternion(qq), Dcol = Dcol0.clone().applyQuaternion(qq);
    const normal = candidateNormal(qq);
    const out: number[] = [];
    for (const s of samples) {
      const [ndcU, ndcV] = toNDC(s.px, s.py);
      const rayDir = cornerDir(ndcU, ndcV, camQuat, vFovRad, aspect);
      const denom = rayDir.dot(normal);
      if (denom >= -MIN_GRAZING_COS) continue;
      const hit = rayDir.multiplyScalar(-distance / denom);
      const u = hit.dot(Drow), v = hit.dot(Dcol);
      const predicted = predictedBilinear(wx0 + u, wz0 + v);
      out.push(predicted - s.observed);
    }
    return new Float64Array(out);
  }

  const cost = (r: Float64Array) => { let s = 0; for (let i = 0; i < r.length; i++) s += r[i] * r[i]; return s; };
  const initialCost = cost(residualsFor(q, worldX0, worldZ0));
  let curCost = initialCost;
  let lambda = 1e-3;
  const EPS_ROT = 1e-5, EPS_POS = 1e-3;
  const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const P = 5;

  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    const r0 = residualsFor(q, worldX0, worldZ0);
    const n = r0.length;
    if (n === 0) break;

    const J: Float64Array[] = [];
    for (let k = 0; k < 3; k++) {
      const qPlus = new THREE.Quaternion().setFromAxisAngle(axes[k], EPS_ROT).multiply(q);
      const rP = residualsFor(qPlus, worldX0, worldZ0);
      const len = Math.min(n, rP.length);
      const col = new Float64Array(n);
      for (let i = 0; i < len; i++) col[i] = (rP[i] - r0[i]) / EPS_ROT;
      J.push(col);
    }
    for (const [dx, dz] of [[EPS_POS, 0], [0, EPS_POS]]) {
      const rP = residualsFor(q, worldX0 + dx, worldZ0 + dz);
      const len = Math.min(n, rP.length);
      const col = new Float64Array(n);
      const eps = dx || dz;
      for (let i = 0; i < len; i++) col[i] = (rP[i] - r0[i]) / eps;
      J.push(col);
    }

    const JtJ: number[][] = Array.from({ length: P }, () => new Array(P).fill(0));
    const Jtr: number[] = new Array(P).fill(0);
    for (let a = 0; a < P; a++) {
      for (let b = 0; b < P; b++) {
        let s = 0; for (let i = 0; i < n; i++) s += J[a][i] * J[b][i];
        JtJ[a][b] = s;
      }
      let s = 0; for (let i = 0; i < n; i++) s += J[a][i] * r0[i];
      Jtr[a] = s;
    }
    const A = JtJ.map((row, a) => row.map((v, b) => v + (a === b ? lambda * (JtJ[a][a] || 1) : 0)));
    const rhs = Jtr.map((v) => -v);
    const delta = solveLinearSystem(A, rhs);
    if (!delta) break;

    const deltaRotVec = new THREE.Vector3(delta[0], delta[1], delta[2]);
    const deltaRotAngle = deltaRotVec.length();
    const deltaWX = delta[3], deltaWZ = delta[4];
    if (deltaRotAngle < 1e-10 && Math.abs(deltaWX) < 1e-10 && Math.abs(deltaWZ) < 1e-10) break;

    const qTry = deltaRotAngle > 1e-12
      ? new THREE.Quaternion().setFromAxisAngle(deltaRotVec.normalize(), deltaRotAngle).multiply(q).normalize()
      : q.clone();
    const wx0Try = worldX0 + deltaWX, wz0Try = worldZ0 + deltaWZ;

    const tryCost = cost(residualsFor(qTry, wx0Try, wz0Try));
    if (tryCost < curCost) {
      q.copy(qTry); worldX0 = wx0Try; worldZ0 = wz0Try;
      curCost = tryCost;
      lambda = Math.max(lambda * 0.5, 1e-8);
    } else {
      lambda = Math.min(lambda * 3, 1e8);
    }
  }

  return {
    Drow: Drow0.clone().applyQuaternion(q), Dcol: Dcol0.clone().applyQuaternion(q), Dnormal: candidateNormal(q),
    worldX0, worldZ0, distance,
    iterations, initialCost, finalCost: curCost,
  };
}

// De-means the profile, then finds the lag (in bins) of the strongest
// non-trivial autocorrelation peak -- see pre-Stage-A history for the full
// derivation (detrend, local-peak requirement, sub-bin parabolic refinement).
export function autocorrelationPeriod(profile: Float64Array): number | null {
  const n = profile.length;

  const detrendWin = 41;
  const half = Math.floor(detrendWin / 2);
  const detrended = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half), hi = Math.min(n - 1, i + half);
    let s = 0, c = 0;
    for (let j = lo; j <= hi; j++) { s += profile[j]; c++; }
    detrended[i] = profile[i] - s / c;
  }

  let mean = 0;
  for (let i = 0; i < n; i++) mean += detrended[i];
  mean /= n;
  const centered = new Float64Array(n);
  for (let i = 0; i < n; i++) centered[i] = detrended[i] - mean;

  const minLag = Math.max(2, Math.floor(n * 0.005));
  const maxLag = Math.floor(n / 2);
  const scores = new Float64Array(maxLag - minLag);
  let bestLagAny = -1, bestScoreAny = -Infinity;
  for (let lag = minLag; lag < maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) sum += centered[i] * centered[i + lag];
    scores[lag - minLag] = sum;
    if (sum > bestScoreAny) { bestScoreAny = sum; bestLagAny = lag; }
  }

  let bestScorePeak = -Infinity;
  const peaks: number[] = [];
  for (let lag = minLag + 1; lag < maxLag - 1; lag++) {
    const s = scores[lag - minLag];
    if (s > scores[lag - minLag - 1] && s > scores[lag - minLag + 1]) {
      peaks.push(lag);
      if (s > bestScorePeak) bestScorePeak = s;
    }
  }
  const peakThreshold = bestScorePeak * 0.5;
  let bestLagPeak = -1;
  for (const lag of peaks) {
    if (scores[lag - minLag] >= peakThreshold) { bestLagPeak = lag; break; }
  }

  const bestLag = bestLagPeak > 0 ? bestLagPeak : bestLagAny;
  if (bestLag <= 0) return null;

  const i = bestLag - minLag;
  if (i > 0 && i < scores.length - 1) {
    const y0 = scores[i - 1], y1 = scores[i], y2 = scores[i + 1];
    const denom = y0 - 2 * y1 + y2;
    if (denom !== 0) {
      const delta = 0.5 * (y0 - y2) / denom;
      if (Math.abs(delta) < 1) return bestLag + delta;
    }
  }
  return bestLag;
}

export function computeProjectedMarginals(w: number, h: number, counts: Float64Array, gradCxSum: Float64Array, gradCySum: Float64Array): Marginals {
  const colSum = new Float64Array(w);
  const colSumCy = new Float64Array(w);
  const rowSum = new Float64Array(h);
  const rowHueCx = new Float64Array(h);
  const rowSumCy = new Float64Array(h);
  for (let bv = 0; bv < h; bv++) {
    for (let bu = 0; bu < w; bu++) {
      const bi = bv * w + bu;
      const c = counts[bi];
      if (c === 0) continue;
      const cx = gradCxSum[bi] / c;
      const cy = gradCySum[bi] / c;
      colSum[bu] += cx; colSumCy[bu] += cy;
      rowSum[bv] -= cx; rowHueCx[bv] += cx; rowSumCy[bv] += cy;
    }
  }
  const colMag = new Float64Array(w);
  for (let bu = 0; bu < w; bu++) colMag[bu] = Math.hypot(colSum[bu], colSumCy[bu]);
  const rowMag = new Float64Array(h);
  for (let bv = 0; bv < h; bv++) rowMag[bv] = Math.hypot(rowHueCx[bv], rowSumCy[bv]);

  const colPeriod = autocorrelationPeriod(colMag);
  const rowPeriod = autocorrelationPeriod(rowMag);
  const colPhase = colPeriod ? findPhase(colMag, colPeriod) : 0;
  const rowPhase = rowPeriod ? findPhase(rowMag, rowPeriod) : 0;
  return { colSum, rowSum, colSumCy, rowHueCx, rowSumCy, colMag, rowMag, colPeriod, rowPeriod, colPhase, rowPhase };
}

export function findPhase(profile: Float64Array, period: number): number {
  let mean = 0;
  for (let i = 0; i < profile.length; i++) mean += profile[i];
  mean /= profile.length;
  let sc = 0, ss = 0;
  for (let i = 0; i < profile.length; i++) {
    const wgt = profile[i] - mean;
    const theta = (2 * Math.PI * i) / period;
    sc += wgt * Math.cos(theta);
    ss += wgt * Math.sin(theta);
  }
  let phase = (Math.atan2(ss, sc) / (2 * Math.PI)) * period;
  if (phase < 0) phase += period;
  return phase;
}

export function marginalHueColor(cx: number, cy: number): string {
  let theta = Math.atan2(cy, cx) / 2;
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  const [r, g, b] = hsvToRgb((theta / Math.PI) * 360, 1, 1);
  return `rgb(${r},${g},${b})`;
}

