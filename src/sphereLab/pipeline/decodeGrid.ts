import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { GRID_STEP, MATH_QUAT } from '../constants.ts';
import { binarize } from '../../decode.ts';
import { cornerDir } from '../math/geometry.ts';
import { tallyPositionVotesGPU } from '../pipelineGPU/decodeTally.ts';
import { projectSamplesGPU } from '../pipelineGPU/projectSamples.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { C, ORDER, R, debruijnLookup, torus } from '../scene/floor.ts';
import { globalState } from '../state.ts';
import { DecodeCellDebug, DecodeSampleGrid, DecodeSamplePoint, GradientField, Marginals, ProjectedBins, ProjectedSamplesDense, VoteResult } from '../types.ts';
import { getAnalysisVFovRad } from './capture.ts';
import { computeGradientField } from './gradientField.ts';
import { computeProjectedMarginals } from './positionLM.ts';

// castAndBucketProjectedSamples reruns this same full-frame gradient field
// EVERY call (computeGradientField(gray, w, h, 1)), but camera.lastNoisedPreviewGray
// never changes within one runAxesReconstruction invocation -- yet this
// function gets called 3-6 times per reconstruction (once per
// computeProjectedBinsAndMarginals/measurePeriodDistance call, see
// axesReconstruction.ts). A plain oversight, not intentional recomputation
// -- cached per-camera (WeakMap, so multiple simultaneously-existing
// cameras never collide), invalidated automatically whenever the gray
// array reference or dimensions change (a new capture/reconstruction).
const srcGradCache = new WeakMap<Camera, { src: Float64Array; w: number; h: number; grad: GradientField }>();
function getCachedSrcGradientField(camera: Camera, gray: Float64Array, w: number, h: number): GradientField {
  const cached = srcGradCache.get(camera);
  if (cached && cached.src === gray && cached.w === w && cached.h === h) return cached.grad;
  const grad = computeGradientField(gray, w, h, 1);
  srcGradCache.set(camera, { src: gray, w, h, grad });
  return grad;
}

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


// Stage 1 (CPU) -- casts one ray per SCREEN pixel, and if it clears the
// grazing-angle cutoff, projects it onto the recovered floor plane's (u,v)
// frame plus its gradient covector. Dense output (one slot per pixel,
// valid=0 for misses) specifically so this and projectSamplesGPU (see
// pipelineGPU/projectSamples.ts) can feed the exact same stage-2 bucketing
// code below -- see pre-Stage-A history for the full derivation
// (grazing-angle cutoff, gradient-covector re-expression in the (u,v)
// frame, the U-mirror that cancels a handedness mismatch).
function projectSamplesCPU(camera: Camera): ProjectedSamplesDense | null {
  if (!camera.lastRecoveredAxes) return null;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const vFovRad = getAnalysisVFovRad(camera);
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, (py / h) * 2 - 1];
  const minGrazingCos = camera.settings.minGrazingCos;

  const hit = new THREE.Vector3();
  const hit2 = new THREE.Vector3();
  const n = w * h;
  const uArr = new Float32Array(n), vArr = new Float32Array(n), cxArr = new Float32Array(n), cyArr = new Float32Array(n);
  const validArr = new Uint8Array(n);
  const srcGrad = camera.lastNoisedPreviewGray ? getCachedSrcGradientField(camera, camera.lastNoisedPreviewGray, w, h) : null;
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const [ndcU, ndcV] = toNDC(x, y);
      const rayDir = cornerDir(ndcU, ndcV, MATH_QUAT, vFovRad, camera.aspect);
      const denom = rayDir.dot(normal);
      if (denom >= -minGrazingCos) continue;
      const t = -distance / denom;
      hit.copy(rayDir).multiplyScalar(t);
      const u = hit.dot(Drow), v = hit.dot(Dcol);
      uArr[i] = u; vArr[i] = v; validArr[i] = 1;
      if (u < minU) minU = u; if (u > maxU) maxU = u;
      if (v < minV) minV = v; if (v > maxV) maxV = v;

      if (srcGrad) {
        const fx = srcGrad.fx[i], fy = srcGrad.fy[i];
        const mag = Math.hypot(fx, fy);
        if (mag > 0) {
          const theta = Math.atan2(fy, fx);
          const tdx = -Math.sin(theta), tdy = Math.cos(theta);
          const [ndcU2, ndcV2] = toNDC(x + tdx, y + tdy);
          const rayDir2 = cornerDir(ndcU2, ndcV2, MATH_QUAT, vFovRad, camera.aspect);
          const denom2 = rayDir2.dot(normal);
          if (denom2 < -minGrazingCos) {
            const t2 = -distance / denom2;
            hit2.copy(rayDir2).multiplyScalar(t2);
            const u2 = hit2.dot(Drow), v2 = hit2.dot(Dcol);
            const du = u2 - u, dv = v2 - v;
            if (Math.hypot(du, dv) > 1e-9) {
              const phiUV = Math.atan2(dv, du);
              cxArr[i] = -mag * Math.cos(2 * phiUV);
              cyArr[i] = -mag * Math.sin(2 * phiUV);
            }
          }
        }
      }
    }
  }
  if (!isFinite(minU) || !isFinite(minV)) return null;
  return { u: uArr, v: vArr, cx: cxArr, cy: cyArr, valid: validArr, minU, maxU, minV, maxV };
}

// Exported so other debug overlays (currently gridPeriodPhaseOverlays.ts)
// can convert their OWN gnomonic-unit {xRow,xCol} points (see
// pipeline/gridPeriodPhase.ts's gnomonic()) into this same u/v space --
// u = k*xRow, v = k*xCol for a single shared scalar k -- instead of
// re-deriving their own bounding box and mirror convention, which is what
// caused the Projected-Cam misalignment this was added to fix. k folds in
// both the camera height (distance) and the same grazing-angle normal-flip
// projectSamplesCPU applies above, since that flip changes u/v's sign but
// not gnomonic()'s (gnomonic always uses the raw, unflipped Dnormal).
export function projectedUVScale(camera: Camera): number | null {
  if (!camera.lastRecoveredAxes) return null;
  const { Dnormal, distance } = camera.lastRecoveredAxes;
  const vFovRad = getAnalysisVFovRad(camera);
  const flipped = cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(Dnormal) > 0;
  return flipped ? -distance : distance;
}

// Stage 2 (CPU only, for now -- see this session's chat for why: bucketed
// float accumulation needs either fixed-point atomic<i32> encoding or
// something else GPU-side, deliberately not tackled yet). Bins stage 1's
// dense per-pixel samples into a bucketW x bucketH grid -- shared,
// unchanged, by both the CPU and GPU stage-1 paths below.
function bucketSamples(camera: Camera, bucketW: number, bucketH: number, proj: ProjectedSamplesDense): {
  bins: ProjectedBins; sums: Float64Array; counts: Float64Array; gradCxSum: Float64Array; gradCySum: Float64Array;
} {
  const { u, v, cx, cy, valid, minU, maxU, minV, maxV } = proj;
  const binWidthU = (maxU - minU) / bucketW || 1;
  const binWidthV = (maxV - minV) / bucketH || 1;
  const bins: ProjectedBins = { minU, maxU, minV, maxV, binWidthU, binWidthV, w: bucketW, h: bucketH };
  const sums = new Float64Array(bucketW * bucketH * 3);
  const counts = new Float64Array(bucketW * bucketH);
  const gradCxSum = new Float64Array(bucketW * bucketH);
  const gradCySum = new Float64Array(bucketW * bucketH);
  const n = valid.length;
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    const bu = Math.min(bucketW - 1, Math.max(0, Math.floor((maxU - u[i]) / binWidthU)));
    const bv = Math.min(bucketH - 1, Math.max(0, Math.floor((v[i] - minV) / binWidthV)));
    const bi = bv * bucketW + bu;
    const srcO = i * 4;
    sums[bi * 3] += camera.distortedPreviewData[srcO];
    sums[bi * 3 + 1] += camera.distortedPreviewData[srcO + 1];
    sums[bi * 3 + 2] += camera.distortedPreviewData[srcO + 2];
    counts[bi]++;
    gradCxSum[bi] += cx[i];
    gradCySum[bi] += cy[i];
  }
  return { bins, sums, counts, gradCxSum, gradCySum };
}

export function castAndBucketProjectedSamples(camera: Camera, bucketW: number, bucketH: number): {
  bins: ProjectedBins; sums: Float64Array; counts: Float64Array; gradCxSum: Float64Array; gradCySum: Float64Array;
} | null {
  const proj = projectSamplesCPU(camera);
  if (!proj) return null;
  return bucketSamples(camera, bucketW, bucketH, proj);
}

// GPU-resident counterpart -- only stage 1 (the ray-cast+project, see
// pipelineGPU/projectSamples.ts) runs on GPU; stage 2 (bucketing) stays the
// exact same CPU code as the fully-CPU path above, fed by the GPU's dense
// output. Returns null if WebGPU isn't available; caller falls back to the
// CPU version, which stays the source of truth.
export async function castAndBucketProjectedSamplesGPU(camera: Camera, bucketW: number, bucketH: number): Promise<{
  bins: ProjectedBins; sums: Float64Array; counts: Float64Array; gradCxSum: Float64Array; gradCySum: Float64Array;
} | null> {
  const proj = await projectSamplesGPU(camera);
  if (!proj) return null;
  return bucketSamples(camera, bucketW, bucketH, proj);
}

type ProjectedSampleResult = ReturnType<typeof castAndBucketProjectedSamples>;

// The numeric half of what used to be buildProjectedTexture -- bins feed the
// spacing refinement in runAxesReconstruction regardless of which mode is on
// screen (World view's recovered-pose overlay depends on an accurate
// distance for every camera, not just the one being displayed), so this
// always runs. Returns the raw result too, so a caller that also wants to
// paint doesn't have to re-cast every ray a second time. No longer computes
// marginals (autocorrelation) here -- see this session's chat: that was
// display-only (the marginal-graph overlay, now removed) and decode gets its
// own phase from gridPeriodPhase instead.
export function computeProjectedBinsAndMarginals(camera: Camera): ProjectedSampleResult {
  const result = camera.lastRecoveredAxes ? castAndBucketProjectedSamples(camera, camera.rtSize.w, camera.rtSize.h) : null;
  if (!result) { camera.lastProjectedBins = null; return null; }
  camera.lastProjectedBins = result.bins;
  return result;
}

// GPU-aware twin, deliberately kept separate rather than folded into
// computeProjectedBinsAndMarginals above -- that function has several
// call sites outside the reconstruction pipeline (throttled preview
// updates, mode switches, camera creation) that are perfectly fine staying
// synchronous, and making it async would force all of those to become
// async too. Only runAxesReconstruction (already async) calls this one.
export async function computeProjectedBinsAndMarginalsGPU(camera: Camera): Promise<ProjectedSampleResult> {
  const result = camera.lastRecoveredAxes ? await castAndBucketProjectedSamplesGPU(camera, camera.rtSize.w, camera.rtSize.h) : null;
  if (!result) { camera.lastProjectedBins = null; return null; }
  camera.lastProjectedBins = result.bins;
  return result;
}

// The display half -- an actual GPU texture upload (needsUpdate = true),
// worth skipping whenever nobody's looking at this camera's Projected-Cam
// view. Takes the already-computed result so callers that only need the
// numeric half (see above) never pay for this at all.
export function paintProjectedTexture(camera: Camera, result: ProjectedSampleResult) {
  if (!result) { camera.projectedPreviewData.fill(0); camera.projectedPreviewTex.needsUpdate = true; return; }
  const { bins, sums, counts } = result;
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
}

// Convenience for call sites that always want both (buildProjectedTexture's
// old all-in-one behavior) -- currently just the throttled preview-update
// path in main.ts's animate loop, which already only runs for the active
// camera while Projected-Cam mode is on screen.
export function buildProjectedTexture(camera: Camera) {
  paintProjectedTexture(camera, computeProjectedBinsAndMarginals(camera));
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

// GPU-aware twin, same reasoning as computeProjectedBinsAndMarginalsGPU
// above -- kept separate so measurePeriodDistance's other (synchronous)
// callers are untouched.
export async function measurePeriodDistanceGPU(camera: Camera, currentDistance: number, extentU: number, extentV: number): Promise<{ distanceU: number; distanceV: number } | null> {
  const TARGET_BUCKETS_PER_CELL = 20;
  const MAX_REFINE_BUCKETS = 2048;
  const refineW = Math.min(MAX_REFINE_BUCKETS, Math.max(camera.rtSize.w, Math.ceil(extentU / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const refineH = Math.min(MAX_REFINE_BUCKETS, Math.max(camera.rtSize.h, Math.ceil(extentV / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const refined = await castAndBucketProjectedSamplesGPU(camera, refineW, refineH);
  const refinedMarginals = refined ? computeProjectedMarginals(refineW, refineH, refined.counts, refined.gradCxSum, refined.gradCySum) : null;
  if (!refined || !refinedMarginals || refinedMarginals.colPeriod === null || refinedMarginals.rowPeriod === null) return null;
  return {
    distanceU: currentDistance * (GRID_STEP / (refinedMarginals.colPeriod * refined.bins.binWidthU)),
    distanceV: currentDistance * (GRID_STEP / (refinedMarginals.rowPeriod * refined.bins.binWidthV)),
  };
}

// Own, axis-symmetric-bucket bins/marginals -- deliberately NOT
// lastProjectedBins (the display pipeline's own state) -- see pre-Stage-A
// history for why sharing that state caused a real bug.
// Superseded by buildDecodeSampleGrid's own corner-projection + gridPeriodPhase
// sourced bounds/phase (see this session's chat) -- left defined, unreferenced,
// in case this autocorrelation-based approach is wanted again later.
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

// Forward-projects the 4 image corners onto the recovered floor plane --
// same per-ray math projectSamplesCPU uses (normal flipped toward the floor,
// grazing-cutoff rejected), just 4 rays instead of a full-image pass. Used
// only to size buildDecodeSampleGrid's (u,v) bounding rectangle -- see this
// session's chat for why this replaces the old autocorrelation-derived
// bins.minU/maxU/minV/maxV as that extent's source, and why it's NOT also
// used as a per-cell containment test (the reverse-projection-and-pixel-
// bounds check every surviving cell already does for its own pixel read is
// the same containment test, computed a different way, under the same
// grazing-cutoff assumption this function itself relies on).
export function projectImageCornersToPlane(camera: Camera): { u: number; v: number }[] | null {
  if (!camera.lastRecoveredAxes) return null;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const vFovRad = getAnalysisVFovRad(camera);
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const minGrazingCos = camera.settings.minGrazingCos;
  const corners: { u: number; v: number }[] = [];
  for (const [ndcU, ndcV] of [[-1, -1], [1, -1], [1, 1], [-1, 1]] as const) {
    const rayDir = cornerDir(ndcU, ndcV, MATH_QUAT, vFovRad, camera.aspect);
    const denom = rayDir.dot(normal);
    if (denom >= -minGrazingCos) return null;
    const t = -distance / denom;
    const hit = rayDir.clone().multiplyScalar(t);
    corners.push({ u: hit.dot(Drow), v: hit.dot(Dcol) });
  }
  return corners;
}

// Builds a sampling grid covering the FULL observed quadrilateral -- see
// pre-Stage-A history for the full derivation.
export function buildDecodeSampleGrid(camera: Camera, gray: Float64Array, w: number, h: number, vFovRad: number): DecodeSampleGrid | null {
  if (!camera.lastRecoveredAxes || !camera.lastGridPeriodPhase) return null;
  const gpp = camera.lastGridPeriodPhase;
  const corners = projectImageCornersToPlane(camera);
  if (!corners) return null;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const invQuat = MATH_QUAT.clone().invert();
  const halfV = vFovRad / 2;
  const bin = binarize(gray);
  const minGrazingCos = camera.settings.minGrazingCos;

  // phiCol/phiRow are gnomonic xRow/xCol-space phases (pipeline/gridPeriodPhase.ts);
  // u = uvScale*xRow, v = uvScale*xCol is the same conversion projectedUVScale's
  // own doc comment establishes, so this reuses it rather than re-deriving it.
  const uvScale = projectedUVScale(camera);
  if (uvScale === null) return null;
  const uBoundaryRaw = uvScale * gpp.phiCol;
  const vBoundaryRaw = uvScale * gpp.phiRow;
  const uPhase = (uBoundaryRaw - Math.round(uBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;
  const vPhase = (vBoundaryRaw - Math.round(vBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;

  const cornerUs = corners.map((c) => c.u), cornerVs = corners.map((c) => c.v);
  const minU = Math.min(...cornerUs), maxU = Math.max(...cornerUs);
  const minV = Math.min(...cornerVs), maxV = Math.max(...cornerVs);
  const kMinU = Math.floor((minU - uPhase) / GRID_STEP), kMaxU = Math.ceil((maxU - uPhase) / GRID_STEP);
  const kMinV = Math.floor((minV - vPhase) / GRID_STEP), kMaxV = Math.ceil((maxV - vPhase) / GRID_STEP);
  const cols = kMaxU - kMinU + 1, rows = kMaxV - kMinV + 1;
  const zeroI = Math.min(rows - 1, Math.max(0, Math.round(-vPhase / GRID_STEP) - kMinV));
  const zeroJ = Math.min(cols - 1, Math.max(0, Math.round(-uPhase / GRID_STEP) - kMinU));

  const p = new THREE.Vector3();
  const local = new THREE.Vector3();
  const rayDir = new THREE.Vector3();
  const points: DecodeSamplePoint[][] = [];
  for (let i = 0; i < rows; i++) {
    const v = vPhase + (kMinV + i) * GRID_STEP;
    const rowPoints: DecodeSamplePoint[] = [];
    for (let j = 0; j < cols; j++) {
      const u = uPhase + (kMinU + j) * GRID_STEP;
      p.copy(Drow).multiplyScalar(u).addScaledVector(Dcol, v).addScaledVector(normal, -distance);
      // p is this floor point's position relative to the camera (at the
      // analysis-frame origin) -- same grazing-angle cutoff
      // projectSamplesCPU applies going the OTHER direction (screen pixel
      // -> floor point), so a lattice point only counts as "in the
      // projected quad" (and gets fed to decode) if the ray to it is
      // within the same cutoff the actual Projected-Cam image respects.
      // Without this, decode could read bits from a region of the image
      // that's blank/unreliable there (past the true quad, into a
      // near-horizon sliver only the reverse on-screen-bounds check below
      // would have let through).
      rayDir.copy(p).normalize();
      const grazingOk = rayDir.dot(normal) < -minGrazingCos;
      local.copy(p).applyQuaternion(invQuat);
      const ndcU = -local.x / (local.z * Math.tan(halfV) * camera.aspect);
      const ndcV = -local.y / (local.z * Math.tan(halfV));
      const px = ((ndcU + 1) / 2) * w, py = ((1 - ndcV) / 2) * h;
      const valid = grazingOk && Number.isFinite(px) && Number.isFinite(py) && px >= 0 && px < w && py >= 0 && py < h;
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
    orientation: winner.orientation,
  };
}

