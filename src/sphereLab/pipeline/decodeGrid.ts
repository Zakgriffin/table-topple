import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { GRID_STEP, MATH_QUAT } from '../constants.ts';
import { binarize } from '../../decode.ts';
import { cornerDir } from '../math/geometry.ts';
import { C, ORDER, R, debruijnLookup, torus } from '../scene/floor.ts';
import { DecodeCellDebug, DecodeSampleGrid, DecodeSamplePoint, Marginals, ProjectedBins, VoteResult } from '../types.ts';
import { getAnalysisVFovRad } from './capture.ts';
import { computeGradientField } from './gradientField.ts';
import { computeProjectedMarginals } from './positionLM.ts';

// ── Grid rotation helpers (pure) ─────────────────────────────────────────

export function rotatedDims(rows: number, cols: number, o: number): [number, number] {
  return (o === 1 || o === 3) ? [cols, rows] : [rows, cols];
}
export function readRotated(grid: DecodeSampleGrid, o: number, a: number, b: number): DecodeSamplePoint {
  const { rows: gr, cols: gc, points } = grid;
  if (o === 1) return points[gr - 1 - b][a];
  if (o === 2) return points[gr - 1 - a][gc - 1 - b];
  if (o === 3) return points[b][gc - 1 - a];
  return points[a][b];
}
export function rotateGrid(grid: DecodeSampleGrid, o: number): DecodeSampleGrid {
  if (o === 0) return grid;
  const [rr, cc] = rotatedDims(grid.rows, grid.cols, o);
  const points: DecodeSamplePoint[][] = Array.from({ length: rr }, (_, a) =>
    Array.from({ length: cc }, (_, b) => readRotated(grid, o, a, b)));
  let zeroI = 0, zeroJ = 0, bestD2 = Infinity;
  for (let a = 0; a < rr; a++) {
    for (let b = 0; b < cc; b++) {
      const pt = points[a][b];
      if (!pt.valid) continue;
      const d2 = pt.u * pt.u + pt.v * pt.v;
      if (d2 < bestD2) { bestD2 = d2; zeroI = a; zeroJ = b; }
    }
  }
  return { rows: rr, cols: cc, zeroI, zeroJ, points };
}

// Every valid order x order window, in EACH of the 4 whole-grid rotations,
// votes for a torus anchor -- see pre-Stage-A history for the full
// derivation. Pure function of the grid + the shared De Bruijn lookup.
export function tallyPositionVotes(grid: DecodeSampleGrid): VoteResult | null {
  const tally = new Map<string, number>();
  let totalWindows = 0;
  const block: number[][] = Array.from({ length: ORDER }, () => new Array(ORDER).fill(0));
  for (let o = 0; o < 4; o++) {
    const [rr, cc] = rotatedDims(grid.rows, grid.cols, o);
    for (let i0 = 0; i0 + ORDER <= rr; i0++) {
      for (let j0 = 0; j0 + ORDER <= cc; j0++) {
        let complete = true;
        for (let di = 0; di < ORDER && complete; di++) {
          for (let dj = 0; dj < ORDER; dj++) {
            const pt = readRotated(grid, o, i0 + di, j0 + dj);
            if (!pt.valid) { complete = false; break; }
            block[di][dj] = pt.bit;
          }
        }
        if (!complete) continue;
        totalWindows++;
        let key = 0;
        for (let di = 0; di < ORDER; di++) for (let dj = 0; dj < ORDER; dj++) key = (key << 1) | block[di][dj];
        key = key >>> 0;
        const packed = debruijnLookup.get(key);
        if (packed === undefined) continue;
        const matchRow = Math.floor(packed / C), matchCol = packed % C;
        const anchorRow = ((matchRow - i0) % R + R) % R;
        const anchorCol = ((matchCol - j0) % C + C) % C;
        const voteKey = `${o},${anchorRow},${anchorCol}`;
        tally.set(voteKey, (tally.get(voteKey) ?? 0) + 1);
      }
    }
  }
  let best: VoteResult | null = null;
  for (const [key, votes] of tally) {
    if (best && votes <= best.votes) continue;
    const [o, ar, ac] = key.split(',').map(Number);
    best = { orientation: o, anchorRow: ar, anchorCol: ac, votes, totalWindows };
  }
  return best;
}

// Solves for the camera's ACTUAL world orientation, entirely from the
// pattern -- see pre-Stage-A history (solveRecoveredCamQuat's own comment)
// for the full derivation. Pure function of the (already math-frame)
// geometry plus the shared torus's true world layout.
export function solveRecoveredCamQuat(
  rotated: DecodeSampleGrid, anchorRow: number, anchorCol: number,
  Drow: THREE.Vector3, Dcol: THREE.Vector3, normal: THREE.Vector3, distance: number,
): THREE.Quaternion | null {
  const mathPos = (i: number, j: number) => new THREE.Vector3()
    .addScaledVector(Drow, rotated.points[i][j].u)
    .addScaledVector(Dcol, rotated.points[i][j].v)
    .addScaledVector(normal, -distance);
  const worldPos = (i: number, j: number) => {
    const tRow = ((anchorRow + i) % R + R) % R, tCol = ((anchorCol + j) % C + C) % C;
    return new THREE.Vector3((tCol + 0.5 - C / 2) * GRID_STEP, 0, (tRow + 0.5 - R / 2) * GRID_STEP);
  };
  function findStep(i0: number, j0: number, di: number, dj: number): { i: number; j: number } | null {
    const maxSteps = di !== 0 ? rotated.rows : rotated.cols;
    for (let k = 1; k <= maxSteps; k++) {
      const i = i0 + di * k, j = j0 + dj * k;
      if (i < 0 || i >= rotated.rows || j < 0 || j >= rotated.cols) return null;
      if (rotated.points[i][j].valid) return { i, j };
    }
    return null;
  }

  const zi = rotated.zeroI, zj = rotated.zeroJ;
  const rowStep = findStep(zi, zj, 1, 0) ?? findStep(zi, zj, -1, 0);
  const colStep = findStep(zi, zj, 0, 1) ?? findStep(zi, zj, 0, -1);
  if (!rowStep || !colStep) return null;

  const originMath = mathPos(zi, zj), originWorld = worldPos(zi, zj);
  const rowMath = mathPos(rowStep.i, rowStep.j).sub(originMath).normalize();
  const rowWorld = worldPos(rowStep.i, rowStep.j).sub(originWorld).normalize();
  const colMath = mathPos(colStep.i, colStep.j).sub(originMath).normalize();
  const colWorld = worldPos(colStep.i, colStep.j).sub(originWorld).normalize();

  const thirdMath = new THREE.Vector3().crossVectors(rowMath, colMath).normalize();
  const thirdWorld = new THREE.Vector3().crossVectors(rowWorld, colWorld).normalize();
  if (thirdMath.lengthSq() < 1e-9 || thirdWorld.lengthSq() < 1e-9) return null;

  const mathBasis = new THREE.Matrix4().makeBasis(rowMath, colMath, thirdMath);
  const worldBasis = new THREE.Matrix4().makeBasis(rowWorld, colWorld, thirdWorld);
  const mathBasisInv = mathBasis.clone().invert();
  return new THREE.Quaternion().setFromRotationMatrix(worldBasis.clone().multiply(mathBasisInv));
}


// Casts one ray per SCREEN pixel and bins the hits into a bucketW x bucketH
// grid -- see pre-Stage-A history for the full derivation (grazing-angle
// cutoff, gradient-covector re-expression in the (u,v) frame, the U-mirror
// that cancels a handedness mismatch).
export function castAndBucketProjectedSamples(camera: Camera, bucketW: number, bucketH: number): {
  bins: ProjectedBins; sums: Float64Array; counts: Float64Array; gradCxSum: Float64Array; gradCySum: Float64Array;
} | null {
  if (!camera.lastRecoveredAxes) return null;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const vFovRad = getAnalysisVFovRad(camera);
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, (py / h) * 2 - 1];

  const MIN_GRAZING_COS = 0.15;
  const hit = new THREE.Vector3();
  const hit2 = new THREE.Vector3();
  const us: number[] = [], vs: number[] = [], srcIdx: number[] = [];
  const gradCxAtSample: number[] = [], gradCyAtSample: number[] = [];
  const srcGrad = camera.lastNoisedPreviewGray ? computeGradientField(camera.lastNoisedPreviewGray, w, h, 1) : null;
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [ndcU, ndcV] = toNDC(x, y);
      const rayDir = cornerDir(ndcU, ndcV, MATH_QUAT, vFovRad, camera.aspect);
      const denom = rayDir.dot(normal);
      if (denom >= -MIN_GRAZING_COS) continue;
      const t = -distance / denom;
      hit.copy(rayDir).multiplyScalar(t);
      const u = hit.dot(Drow), v = hit.dot(Dcol);
      us.push(u); vs.push(v); srcIdx.push(y * w + x);
      if (u < minU) minU = u; if (u > maxU) maxU = u;
      if (v < minV) minV = v; if (v > maxV) maxV = v;

      let cxAtSample = 0, cyAtSample = 0;
      if (srcGrad) {
        const si = y * w + x;
        const fx = srcGrad.fx[si], fy = srcGrad.fy[si];
        const mag = Math.hypot(fx, fy);
        if (mag > 0) {
          const theta = Math.atan2(fy, fx);
          const tdx = -Math.sin(theta), tdy = Math.cos(theta);
          const [ndcU2, ndcV2] = toNDC(x + tdx, y + tdy);
          const rayDir2 = cornerDir(ndcU2, ndcV2, MATH_QUAT, vFovRad, camera.aspect);
          const denom2 = rayDir2.dot(normal);
          if (denom2 < -MIN_GRAZING_COS) {
            const t2 = -distance / denom2;
            hit2.copy(rayDir2).multiplyScalar(t2);
            const u2 = hit2.dot(Drow), v2 = hit2.dot(Dcol);
            const du = u2 - u, dv = v2 - v;
            if (Math.hypot(du, dv) > 1e-9) {
              const phiUV = Math.atan2(dv, du);
              cxAtSample = -mag * Math.cos(2 * phiUV);
              cyAtSample = -mag * Math.sin(2 * phiUV);
            }
          }
        }
      }
      gradCxAtSample.push(cxAtSample); gradCyAtSample.push(cyAtSample);
    }
  }
  if (!isFinite(minU) || !isFinite(minV)) return null;

  const binWidthU = (maxU - minU) / bucketW || 1;
  const binWidthV = (maxV - minV) / bucketH || 1;
  const bins: ProjectedBins = { minU, maxU, minV, maxV, binWidthU, binWidthV, w: bucketW, h: bucketH };
  const sums = new Float64Array(bucketW * bucketH * 3);
  const counts = new Float64Array(bucketW * bucketH);
  const gradCxSum = new Float64Array(bucketW * bucketH);
  const gradCySum = new Float64Array(bucketW * bucketH);
  for (let k = 0; k < us.length; k++) {
    const bu = Math.min(bucketW - 1, Math.max(0, Math.floor((maxU - us[k]) / binWidthU)));
    const bv = Math.min(bucketH - 1, Math.max(0, Math.floor((vs[k] - minV) / binWidthV)));
    const bi = bv * bucketW + bu;
    const si = srcIdx[k];
    const srcO = si * 4;
    sums[bi * 3] += camera.distortedPreviewData[srcO];
    sums[bi * 3 + 1] += camera.distortedPreviewData[srcO + 1];
    sums[bi * 3 + 2] += camera.distortedPreviewData[srcO + 2];
    counts[bi]++;
    gradCxSum[bi] += gradCxAtSample[k];
    gradCySum[bi] += gradCyAtSample[k];
  }
  return { bins, sums, counts, gradCxSum, gradCySum };
}

// Rebuilds projectedPreviewData: a bird's-eye, floor-plane-rectified view of
// whichever field view is currently in distortedPreviewData.
export function buildProjectedTexture(camera: Camera) {
  const result = camera.lastRecoveredAxes ? castAndBucketProjectedSamples(camera, camera.rtSize.w, camera.rtSize.h) : null;
  if (!result) { camera.projectedPreviewData.fill(0); camera.projectedPreviewTex.needsUpdate = true; camera.lastProjectedBins = null; camera.lastMarginals = null; return; }
  const { bins, sums, counts, gradCxSum, gradCySum } = result;
  camera.lastProjectedBins = bins;
  for (let bi = 0; bi < bins.w * bins.h; bi++) {
    const c = counts[bi];
    const o = bi * 4;
    if (c > 0) {
      camera.projectedPreviewData[o] = Math.round(sums[bi * 3] / c);
      camera.projectedPreviewData[o + 1] = Math.round(sums[bi * 3 + 1] / c);
      camera.projectedPreviewData[o + 2] = Math.round(sums[bi * 3 + 2] / c);
      camera.projectedPreviewData[o + 3] = 255;
    } else {
      camera.projectedPreviewData[o] = 0; camera.projectedPreviewData[o + 1] = 0; camera.projectedPreviewData[o + 2] = 0; camera.projectedPreviewData[o + 3] = 255;
    }
  }
  camera.projectedPreviewTex.needsUpdate = true;
  camera.lastMarginals = computeProjectedMarginals(bins.w, bins.h, counts, gradCxSum, gradCySum);
}

// Re-buckets castAndBucketProjectedSamples' rays at a resolution sized to
// keep a fixed target of buckets per grid cell -- see pre-Stage-A history.
export function measurePeriodDistance(camera: Camera, currentDistance: number, extentU: number, extentV: number): { distanceU: number; distanceV: number } | null {
  const TARGET_BUCKETS_PER_CELL = 20;
  const MAX_REFINE_BUCKETS = 2048;
  const refineW = Math.min(MAX_REFINE_BUCKETS, Math.max(camera.rtSize.w, Math.ceil(extentU / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const refineH = Math.min(MAX_REFINE_BUCKETS, Math.max(camera.rtSize.h, Math.ceil(extentV / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const refined = castAndBucketProjectedSamples(camera, refineW, refineH);
  const refinedMarginals = refined ? computeProjectedMarginals(refineW, refineH, refined.counts, refined.gradCxSum, refined.gradCySum) : null;
  if (!refined || !refinedMarginals || refinedMarginals.colPeriod === null || refinedMarginals.rowPeriod === null) return null;
  return {
    distanceU: currentDistance * (GRID_STEP / (refinedMarginals.colPeriod * refined.bins.binWidthU)),
    distanceV: currentDistance * (GRID_STEP / (refinedMarginals.rowPeriod * refined.bins.binWidthV)),
  };
}

// Own, axis-symmetric-bucket bins/marginals -- deliberately NOT
// lastProjectedBins/lastMarginals (the display pipeline's own state) -- see
// pre-Stage-A history for why sharing that state caused a real bug.
export function computeDecodeMarginals(camera: Camera): { bins: ProjectedBins; marginals: Marginals } | null {
  if (!camera.lastRecoveredAxes || !camera.lastProjectedBins) return null;
  const TARGET_BUCKETS_PER_CELL = 20;
  const MAX_REFINE_BUCKETS = 2048;
  const floor = Math.max(camera.rtSize.w, camera.rtSize.h);
  const extentU = camera.lastProjectedBins.maxU - camera.lastProjectedBins.minU;
  const extentV = camera.lastProjectedBins.maxV - camera.lastProjectedBins.minV;
  const bucketW = Math.min(MAX_REFINE_BUCKETS, Math.max(floor, Math.ceil(extentU / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const bucketH = Math.min(MAX_REFINE_BUCKETS, Math.max(floor, Math.ceil(extentV / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const result = castAndBucketProjectedSamples(camera, bucketW, bucketH);
  if (!result) return null;
  const marginals = computeProjectedMarginals(bucketW, bucketH, result.counts, result.gradCxSum, result.gradCySum);
  return { bins: result.bins, marginals };
}

// Builds a sampling grid covering the FULL observed quadrilateral -- see
// pre-Stage-A history for the full derivation.
export function buildDecodeSampleGrid(camera: Camera, gray: Float64Array, w: number, h: number, vFovRad: number): DecodeSampleGrid | null {
  if (!camera.lastRecoveredAxes) return null;
  const decodeMarginals = computeDecodeMarginals(camera);
  if (!decodeMarginals || decodeMarginals.marginals.colPeriod === null || decodeMarginals.marginals.rowPeriod === null) {
    return null;
  }
  const { bins, marginals } = decodeMarginals;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const invQuat = MATH_QUAT.clone().invert();
  const halfV = vFovRad / 2;
  const bin = binarize(gray);

  const uBoundaryRaw = bins.maxU - marginals.colPhase * bins.binWidthU;
  const vBoundaryRaw = bins.minV + marginals.rowPhase * bins.binWidthV;
  const uPhase = (uBoundaryRaw - Math.round(uBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;
  const vPhase = (vBoundaryRaw - Math.round(vBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;

  const { minU, maxU, minV, maxV } = bins;
  const kMinU = Math.floor((minU - uPhase) / GRID_STEP), kMaxU = Math.ceil((maxU - uPhase) / GRID_STEP);
  const kMinV = Math.floor((minV - vPhase) / GRID_STEP), kMaxV = Math.ceil((maxV - vPhase) / GRID_STEP);
  const cols = kMaxU - kMinU + 1, rows = kMaxV - kMinV + 1;
  const zeroI = Math.min(rows - 1, Math.max(0, Math.round(-vPhase / GRID_STEP) - kMinV));
  const zeroJ = Math.min(cols - 1, Math.max(0, Math.round(-uPhase / GRID_STEP) - kMinU));

  const p = new THREE.Vector3();
  const local = new THREE.Vector3();
  const points: DecodeSamplePoint[][] = [];
  for (let i = 0; i < rows; i++) {
    const v = vPhase + (kMinV + i) * GRID_STEP;
    const rowPoints: DecodeSamplePoint[] = [];
    for (let j = 0; j < cols; j++) {
      const u = uPhase + (kMinU + j) * GRID_STEP;
      p.copy(Drow).multiplyScalar(u).addScaledVector(Dcol, v).addScaledVector(normal, -distance);
      local.copy(p).applyQuaternion(invQuat);
      const ndcU = -local.x / (local.z * Math.tan(halfV) * camera.aspect);
      const ndcV = -local.y / (local.z * Math.tan(halfV));
      const px = ((ndcU + 1) / 2) * w, py = ((1 - ndcV) / 2) * h;
      const valid = Number.isFinite(px) && Number.isFinite(py) && px >= 0 && px < w && py >= 0 && py < h;
      if (!valid) { rowPoints.push({ u, v, px, py, valid: false, bit: 0 }); continue; }
      const xx = Math.round(px), yy = Math.round(py);
      rowPoints.push({ u, v, px, py, valid: true, bit: bin[yy * w + xx] });
    }
    points.push(rowPoints);
  }
  return { rows, cols, zeroI, zeroJ, points };
}

// Decodes the camera's absolute world position -- see pre-Stage-A history
// for the full derivation.
export function runPositionDecode(camera: Camera, gray: Float64Array, w: number, h: number, vFovRad: number) {
  const grid = buildDecodeSampleGrid(camera, gray, w, h, vFovRad);
  camera.lastDecodeGrid = grid;
  camera.lastDecodeRotated = null;
  if (!grid) { camera.lastPositionDecode = null; camera.lastDecodeCorrectness = null; return; }
  const winner = tallyPositionVotes(grid);
  if (!winner) { camera.lastPositionDecode = null; camera.lastDecodeCorrectness = null; return; }

  const { anchorRow, anchorCol } = winner;
  const rotated = rotateGrid(grid, winner.orientation);
  camera.lastDecodeRotated = rotated;
  const correctness: (DecodeCellDebug | null)[][] = Array.from({ length: rotated.rows }, () => new Array(rotated.cols).fill(null));
  let correctCount = 0, wrongCount = 0;
  for (let i = 0; i < rotated.rows; i++) {
    for (let j = 0; j < rotated.cols; j++) {
      const pt = rotated.points[i][j];
      if (!pt.valid) continue;
      const torusRow = ((anchorRow + i) % R + R) % R;
      const torusCol = ((anchorCol + j) % C + C) % C;
      const correct = pt.bit === torus[torusRow][torusCol];
      correctness[i][j] = { bit: pt.bit, correct };
      correct ? correctCount++ : wrongCount++;
    }
  }
  camera.lastDecodeCorrectness = correctness;
  const consistency = correctCount + wrongCount > 0 ? correctCount / (correctCount + wrongCount) : 0;

  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes!; // buildDecodeSampleGrid returning non-null guarantees this
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const refTorusRow = ((anchorRow + rotated.zeroI) % R + R) % R;
  const refTorusCol = ((anchorCol + rotated.zeroJ) % C + C) % C;

  const recoveredCamQuat = solveRecoveredCamQuat(rotated, anchorRow, anchorCol, Drow, Dcol, normal, distance);
  if (!recoveredCamQuat) { camera.lastPositionDecode = null; return; }

  const DrowWorld = Drow.clone().applyQuaternion(recoveredCamQuat);
  const DcolWorld = Dcol.clone().applyQuaternion(recoveredCamQuat);
  const normalWorld = normal.clone().applyQuaternion(recoveredCamQuat);
  const refPt = rotated.points[rotated.zeroI][rotated.zeroJ];
  const hitRelWorld = new THREE.Vector3()
    .addScaledVector(DrowWorld, refPt.u).addScaledVector(DcolWorld, refPt.v).addScaledVector(normalWorld, -distance);

  const worldPosTrue = new THREE.Vector3((refTorusCol + 0.5 - C / 2) * GRID_STEP, 0, (refTorusRow + 0.5 - R / 2) * GRID_STEP);
  camera.lastPositionDecode = {
    row: refTorusRow, col: refTorusCol, consistency, votes: winner.votes, totalWindows: winner.totalWindows,
    camPos: worldPosTrue.sub(hitRelWorld),
    recoveredCamQuat,
  };
}

