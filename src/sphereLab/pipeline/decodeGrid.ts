import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { GRID_STEP, MATH_QUAT } from '../constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../math/geometry.ts';
import { tallyPositionVotesGPU } from '../pipelineGPU/decodeTally.ts';
import { buildAndTallyDecodeGPU } from '../pipelineGPU/decodeGridBuild.ts';
import { projectSamplesGPU } from '../pipelineGPU/projectSamples.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { C, ORDER, R, debruijnLookup, torus } from '../floorPattern.ts';
import { DecodeCellDebug, DecodeSampleGrid, DecodeSamplePoint, GradientField, PositionDecodeResult, ProjectedBins, ProjectedSamplesDense, RecoveredAxes, VoteResult } from '../types.ts';
import { Backend } from './backend.ts';
import { computeGradientField } from './gradientField.ts';
import { GridPeriodPhaseResult } from './gridPeriodPhase.ts';

// Minimal shape projectImageCornersToPlane/projectedUVScale/
// buildDecodeSampleGrid/runPositionDecode actually need -- narrowed off the
// full `Camera` (which also carries THREE objects/GPU textures no bare data
// object has) so these stay callable from pipeline/poseCompute.ts's
// PoseComputeState, a plain object literal with none of that -- see this
// session's on-device-pose-recovery plan. Declared locally (not imported
// from poseCompute.ts) since that file imports runPositionDecode FROM here;
// PoseComputeState satisfies this structurally without either file needing
// to import the other's type.
interface PoseCameraLike {
  aspect: number;
  settings: { horizFovDeg: number; minGrazingCos: number };
  lastRecoveredAxes: RecoveredAxes | null;
  lastGridPeriodPhase: GridPeriodPhaseResult | null;
  lastDecodeGrid: DecodeSampleGrid | null;
  lastDecodeRotated: DecodeSampleGrid | null;
  lastDecodeCorrectness: (DecodeCellDebug | null)[][] | null;
  lastPositionDecode: PositionDecodeResult | null;
  pendingDecodeGrid: PendingDecodeGrid | null;
}

// ── The decode grid's readback, handed to the caller instead of taken here ──
//
// The fused GPU decode produces the pose from `winner` plus two correctness
// counts (8 bytes); the GRID itself -- 0.45MB across decode:gridGeom and
// decode:gridPacked -- feeds only lastDecodeGrid/lastDecodeRotated/
// lastDecodeCorrectness, which is display. So the pose does not have to wait
// for it, and this is what lets a caller say so.
//
// `resolve` is IDEMPOTENT and `release` is safe to call twice, in either order,
// because the two callers use them differently: a phone or a harness resolves
// immediately, while the desktop parks this on the camera and drains it a frame
// later -- possibly never, if a newer reconstruction supersedes it first, which
// is exactly when release-without-resolve has to be free.
//
// NOT view-conditional, and the distinction is the whole reason this is a
// deferral rather than the skip that was rejected before (see runPositionDecode
// below): whoever holds this WILL resolve it, so every observable ends up
// identical and only the timing moves.
export interface PendingDecodeGrid {
  resolve(): Promise<void>;
  release(): void;
}

// bucketSamples reruns this same full-frame gradient field EVERY call
// (computeGradientField(gray, w, h, 1)), but camera.lastNoisedPreviewGray never
// changes within one runAxesReconstruction invocation -- yet this function used
// to get called 3-6 times per reconstruction, once per projection. (The
// autocorrelation callers that made it 3-6 are deleted; one projection per
// reconstruction remains.) A plain oversight, not intentional recomputation
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
function readRotated(grid: DecodeSampleGrid, o: number, a: number, b: number): DecodeSamplePoint {
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
  // Carries the ORIGINAL grid's own zero-reference index through the same
  // rotation every other cell goes through -- the inverse of readRotated's
  // index map above -- rather than rescanning the whole rotated grid for
  // "closest to the true origin, and valid". That rescan (and its "and
  // valid" filter) existed only to dodge an invalid reference point for
  // solveRecoveredCamQuat's old per-cell neighbor search, which no longer
  // exists -- see this session's chat. With that gone, no downstream
  // consumer cares whether the reference cell is valid (u/v and the torus
  // row/col it maps to are defined for every index either way), so simply
  // preserving the SAME physical point buildDecodeSampleGrid already chose
  // (nearest the true world origin) is both simpler and just as correct.
  const [zeroI, zeroJ] = rotatedZeroIndex(grid.rows, grid.cols, grid.zeroI, grid.zeroJ, o);
  return { rows: rr, cols: cc, zeroI, zeroJ, points };
}

// Where the grid's zero-reference cell ENDS UP after rotating by `o`. Split out
// of rotateGrid because the GPU decode path needs it without materializing a
// rotated grid at all -- it has no CPU-side points array to permute, but still
// needs the rotated (zeroI, zeroJ) to index the torus with.
//
// Note this maps the reference cell's INDEX; the cell itself is unchanged, so
// its u/v are still the original grid's (zeroI, zeroJ) -- see decodeGridCellUV.
export function rotatedZeroIndex(
  rows: number, cols: number, zeroI: number, zeroJ: number, o: number,
): [number, number] {
  if (o === 0) return [zeroI, zeroJ];
  if (o === 1) return [zeroJ, rows - 1 - zeroI];
  if (o === 2) return [rows - 1 - zeroI, cols - 1 - zeroJ];
  return [cols - 1 - zeroJ, zeroI];
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

// Solves for the camera's ACTUAL world orientation -- closed form, from
// the analysis-frame row/col axes fitPairOfPlanes already recovered, no
// per-cell grid/validity lookup needed. Drow/Dcol are defined against the
// UNROTATED sample grid's own row/col indexing (buildDecodeSampleGrid,
// before any rotation) -- but tallyPositionVotes doesn't know in advance
// which of the 4 index-rotations of that grid actually matches the
// board's own pattern, so it tries all 4 and reports the winner as
// `orientation`. Whenever that winner isn't the identity, the sample
// grid's row/col axes are offset from Drow/Dcol by exactly `orientation`
// steps of 90 degrees, and rowMath/colMath below apply that SAME 90-degree
// rotation `orientation` times: (x,y) -> (y,-x), a proper rotation
// (determinant +1, cycles back to the start after 4 applications -- a
// reflection would have order 2, not 4), not a per-case reflection fix.
// fitPairOfPlanes' own handedness enforcement (axesReconstruction.ts,
// right after the fit) is what makes ONE rotation formula sufficient
// uniformly across all 4 orientations -- it guarantees {Drow,Dcol,Dnormal}
// is consistently right-handed, so thirdMath below never needs its own
// independent handedness correction; without that guarantee, rotating
// (Dcol,Drow) alone wouldn't be enough. Verified numerically against a
// simulated camera's true position at all 4 orientations via dev-bridge
// (position error 0.05-0.5, matching a working capture's own baseline
// noise) -- an earlier version of this got the o=1/3 axis swap right but
// missed these sign flips entirely, silently correct only at o=0 (most
// captures) while still reporting perfect bit consistency at o=1/2/3
// (period/phase/anchor recovery are all independently correct; only this
// rotation was missing) -- caught live via dev-bridge at yaw=+2
// (orientation 3), where a +2/-2 yaw swing flipped which orientation won.
// See this session's chat, and the yaw=-65 dev-bridge investigation for
// the separate, earlier bug this same closed form replaced (walking
// outward from one fixed grid index for ANY valid neighbor, which could
// fail outright if that index landed on an invalid quad edge).
function solveRecoveredCamQuat(Drow: THREE.Vector3, Dcol: THREE.Vector3, orientation: number): THREE.Quaternion {
  let rowMath = Dcol.clone(), colMath = Drow.clone();
  for (let step = 0; step < orientation; step++) {
    const nextRow = colMath, nextCol = rowMath.negate();
    rowMath = nextRow; colMath = nextCol;
  }
  const thirdMath = new THREE.Vector3().crossVectors(rowMath, colMath).normalize();
  const mathBasis = new THREE.Matrix4().makeBasis(rowMath, colMath, thirdMath);
  const rowWorld = new THREE.Vector3(0, 0, 1), colWorld = new THREE.Vector3(1, 0, 0);
  const thirdWorld = new THREE.Vector3().crossVectors(rowWorld, colWorld).normalize();
  const worldBasis = new THREE.Matrix4().makeBasis(rowWorld, colWorld, thirdWorld);
  return new THREE.Quaternion().setFromRotationMatrix(worldBasis.multiply(mathBasis.invert()));
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
export function projectedUVScale(camera: PoseCameraLike): number | null {
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

// The bins a successful projection produces, or null when there was no
// recovered pose to project against. Named off bucketSamples now that the
// castAndBucketProjectedSamples wrappers are gone -- they existed only to pair
// a projection with a bucketing, and their last callers were the retired
// autocorrelation distance path.
export type ProjectedSampleResult = ReturnType<typeof bucketSamples> | null;

// Picks bucketW/bucketH so every bucket is a SQUARE in world (floor-plane)
// units -- binWidthU === binWidthV -- rather than one bucket per screen
// pixel on each axis independently (the old behavior, castAndBucket*'s
// default call pattern below used to use), which only produced square cells
// when the projected floor extent itself happened to be square. The longer
// of the two floor axes gets a full max(rtSize.w, rtSize.h) buckets; the
// shorter axis gets proportionally fewer, so the shared bin width tracks
// the EXTENT's own aspect ratio instead of the viewport's -- meaning
// bucketW and bucketH will generally differ from each other (and from
// rtSize.w/h), unlike a same-count-both-axes "square grid" (which does NOT
// by itself guarantee square cells). `|| 1` / `Math.max(1, ...)` guard the
// degenerate near-zero-extent case (e.g. a single valid ray) from producing
// a zero-width bin or a zero-bucket axis.
function squareCellBucketDims(camera: Camera, extentU: number, extentV: number): { bucketW: number; bucketH: number } {
  const longAxisBuckets = Math.max(camera.rtSize.w, camera.rtSize.h);
  const binWidth = Math.max(extentU, extentV) / longAxisBuckets || 1;
  return { bucketW: Math.max(1, Math.round(extentU / binWidth)), bucketH: Math.max(1, Math.round(extentV / binWidth)) };
}

// The numeric half of what used to be buildProjectedTexture -- bins feed the
// spacing refinement in runAxesReconstruction regardless of which mode is on
// screen (World view's recovered-pose overlay depends on an accurate
// distance for every camera, not just the one being displayed), so this
// always runs. Returns the raw result too, so a caller that also wants to
// paint doesn't have to re-cast every ray a second time.
//
// It was called computeProjectedBinsAndMarginals until the marginals it named
// were deleted -- autocorrelation was display-only (the marginal-graph overlay,
// since removed) and decode takes its phase from gridPeriodPhase. The name
// outlived the work by long enough to be worth noting as a hazard.
//
// Projects once (stage 1) so the resulting extent can size a square-cell bucket
// grid (stage 2) BEFORE bucketing, which is why it calls projectSamplesCPU and
// bucketSamples separately rather than in one step.
function computeProjectedBinsCPU(camera: Camera): ProjectedSampleResult {
  const proj = camera.lastRecoveredAxes ? projectSamplesCPU(camera) : null;
  if (!proj) { camera.lastProjectedBins = null; return null; }
  const { bucketW, bucketH } = squareCellBucketDims(camera, proj.maxU - proj.minU, proj.maxV - proj.minV);
  const result = bucketSamples(camera, bucketW, bucketH, proj);
  camera.lastProjectedBins = result.bins;
  return result;
}

// GPU-aware twin, deliberately kept separate rather than folded into
// computeProjectedBinsCPU above -- that function has several
// call sites outside the reconstruction pipeline (throttled preview
// updates, mode switches, camera creation) that used to be perfectly fine
// staying synchronous; now that those call sites (see modeRefresh.ts/
// main.ts/ui/mode.ts) have all gone async too (see this session's chat --
// buildProjectedTexture used to silently bypass the backend choice
// entirely, running a redundant CPU-only re-projection on every
// reconstruction pass and every throttled preview tick), every caller can
// safely go through computeProjectedBinsAuto below instead of
// picking CPU vs GPU itself.
async function computeProjectedBinsGPU(camera: Camera): Promise<ProjectedSampleResult> {
  const proj = camera.lastRecoveredAxes ? await projectSamplesGPU(camera) : null;
  if (!proj) { camera.lastProjectedBins = null; return null; }
  const { bucketW, bucketH } = squareCellBucketDims(camera, proj.maxU - proj.minU, proj.maxV - proj.minV);
  const result = bucketSamples(camera, bucketW, bucketH, proj);
  camera.lastProjectedBins = result.bins;
  return result;
}

// Single dispatch point for every caller (axesReconstruction.ts's
// recomputeStages, and modeRefresh.ts's buildProjectedTexture) --
// centralizes the backend check once instead of duplicating the
// GPU-with-CPU-fallback ternary at each call site.
export async function computeProjectedBinsAuto(camera: Camera, backend: Backend): Promise<ProjectedSampleResult> {
  const s = spanStart(backend === 'cpu' ? 'projectBins (CPU)' : 'projectBins (GPU stage 1 + CPU bucket)');
  const result = backend === 'gpu'
    ? (await computeProjectedBinsGPU(camera)) ?? computeProjectedBinsCPU(camera)
    : computeProjectedBinsCPU(camera);
  spanEnd(s);
  return result;
}

// The display half -- an actual GPU texture upload (needsUpdate = true),
// worth skipping whenever nobody's looking at this camera's Projected-Cam
// view. Takes the already-computed result so callers that only need the
// numeric half (see above) never pay for this at all.
export function paintProjectedTexture(camera: Camera, result: ProjectedSampleResult) {
  if (!result) { camera.projectedPreviewData.fill(0); camera.projectedPreviewTex.needsUpdate = true; return; }
  const { bins, sums, counts } = result;
  // bins.w x bins.h (squareCellBucketDims' square-CELL grid) varies per
  // capture with the recovered floor extent's own aspect ratio -- unlike
  // every other preview buffer here, fixed at viewport-resize time --
  // reallocate whenever this capture's bucket grid differs from the buffer
  // camera/factory.ts or pipeline/capture.ts's resizeCaptureBuffers last
  // sized this to (or the previous capture left it at).
  const img = camera.projectedPreviewTex.image as { width: number; height: number };
  if (img.width !== bins.w || img.height !== bins.h) {
    camera.projectedPreviewData = new Uint8Array(bins.w * bins.h * 4);
    camera.projectedPreviewTex.image = { data: camera.projectedPreviewData, width: bins.w, height: bins.h };
    camera.projectedPreviewTex.dispose();
  }
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

// Convenience for call sites that always want both (paint AND, unless
// already provided, compute) -- the throttled preview-update path in
// main.ts's animate loop, mode switches (ui/mode.ts), and camera creation
// (camera/lifecycle.ts) all still need a real computation here since they
// have no fresher result lying around; modeRefresh.ts's
// refreshModeVisualizations, called at the tail of a reconstruction pass
// that JUST computed this same result a few lines earlier, passes it in via
// `precomputed` instead of paying for a second (possibly GPU) round trip.
// Wrapped in `{ value }` rather than passed bare so "not provided" (recompute)
// and "provided, and IS null" (recovered axes genuinely missing -- paint
// blank) stay distinguishable.
//
// Per-camera sequence guard: main.ts's throttled preview loop can call this
// again (captureDirty re-set by some other slider) before a slow -- e.g.
// GPU -- in-flight call finishes; without this, whichever call happened to
// finish LAST would win even if it was started FIRST, painting a stale
// result over a newer one. Keyed per-camera (not one shared counter) since
// World mode can have several cameras' textures updating concurrently (every
// camera's own recovered-floor decal, not just the active one) -- a shared
// counter would let one camera's call wrongly invalidate another's.
const projTextureSeq = new WeakMap<Camera, number>();
export async function buildProjectedTexture(
  camera: Camera, backend: Backend, precomputed?: { value: ProjectedSampleResult },
): Promise<void> {
  const seq = (projTextureSeq.get(camera) ?? 0) + 1;
  projTextureSeq.set(camera, seq);
  const result = precomputed ? precomputed.value : await computeProjectedBinsAuto(camera, backend);
  if (projTextureSeq.get(camera) !== seq) return; // a newer call started meanwhile -- its result wins instead
  paintProjectedTexture(camera, result);
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
export function projectImageCornersToPlane(camera: PoseCameraLike): { u: number; v: number }[] | null {
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

// The (u,v) bounding rectangle of the RESOLVABLE visible floor region, for
// sizing buildDecodeSampleGrid's sample lattice. Unlike
// projectImageCornersToPlane above -- which needs ALL 4 image corners to
// clear the grazing cutoff and returns null the instant one grazes past the
// horizon -- this samples a grid of image points and bounds only those that
// DO clear the same strict per-cell cutoff. That all-or-nothing corner bail
// was silently killing decode outright at any oblique/grazing view whose top
// edge points past the horizon, discarding an 80-94%-usable frame along with
// the two grazing corners (see this session's decode-failure investigation:
// orientation/period/distance were all fine at those poses, decode just never
// built a grid). The per-lattice-point grazing check inside
// buildDecodeSampleGrid already drops individual far cells, so bounding the
// resolvable region here and letting that per-cell check exclude the rest is
// both correct and strictly more permissive. For a non-grazing view every
// sample clears the cutoff and the floor projects to a straight-edged
// quadrilateral, so this reduces to the same min/max the 4 corners gave --
// no behavior change there. Returns null only when almost nothing projects
// (a genuinely degenerate, near-horizon-only view).
function projectedUVBounds(camera: PoseCameraLike): { minU: number; maxU: number; minV: number; maxV: number } | null {
  if (!camera.lastRecoveredAxes) return null;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const vFovRad = getAnalysisVFovRad(camera);
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const minGrazingCos = camera.settings.minGrazingCos;
  const N = 48; // image sampling resolution; extremes lie on the boundary, so this over-covers slightly, which is safe (per-cell cutoff still drops far cells)
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity, count = 0;
  for (let iy = 0; iy <= N; iy++) {
    for (let ix = 0; ix <= N; ix++) {
      const ndcU = (ix / N) * 2 - 1, ndcV = (iy / N) * 2 - 1;
      const rayDir = cornerDir(ndcU, ndcV, MATH_QUAT, vFovRad, camera.aspect);
      const denom = rayDir.dot(normal);
      if (denom >= -minGrazingCos) continue;
      const t = -distance / denom;
      const u = rayDir.dot(Drow) * t, v = rayDir.dot(Dcol) * t;
      if (u < minU) minU = u; if (u > maxU) maxU = u;
      if (v < minV) minV = v; if (v > maxV) maxV = v;
      count++;
    }
  }
  if (count < 4 || !isFinite(minU)) return null;
  return { minU, maxU, minV, maxV };
}

// Builds a sampling grid covering the FULL observed quadrilateral -- see
// pre-Stage-A history for the full derivation.
// Everything about the sample lattice EXCEPT the per-cell ray-casting: the
// axes, the phase/extent derivation, the grid dimensions, and the binarization
// threshold. Extracted so buildDecodeSampleGrid and its GPU twin
// (pipelineGPU/decodeGridBuild.ts) derive the lattice from ONE piece of code
// rather than two copies that could drift -- a half-cell disagreement in
// uPhase would put every sampled bit in the wrong place while still looking
// like a plausible grid.
export interface DecodeGridLayout {
  Drow: THREE.Vector3; Dcol: THREE.Vector3; normal: THREE.Vector3;
  invQuat: THREE.Quaternion;
  distance: number; halfV: number; aspect: number; binThreshold: number; minGrazingCos: number;
  uPhase: number; vPhase: number; kMinU: number; kMinV: number;
  rows: number; cols: number; zeroI: number; zeroJ: number;
}

export function decodeGridLayout(
  camera: PoseCameraLike, gray: Float64Array, vFovRad: number,
): DecodeGridLayout | null {
  if (!camera.lastRecoveredAxes || !camera.lastGridPeriodPhase) return null;
  const gpp = camera.lastGridPeriodPhase;
  const bounds = projectedUVBounds(camera);
  if (!bounds) return null;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const invQuat = MATH_QUAT.clone().invert();
  const halfV = vFovRad / 2;
  // binarize() materializes a full-image Uint8Array, but this grid only ever
  // READS rows*cols of those pixels -- on the order of a thousand out of ~200k.
  // All it actually needs is binarize's global-mean threshold, so take the mean
  // and compare at sample time: same decision per sampled pixel, without the
  // 196KB allocation and the per-pixel write for the ~99.5% nothing looks at.
  let binThreshold = 0;
  for (let i = 0; i < gray.length; i++) binThreshold += gray[i];
  binThreshold /= gray.length;

  // phiCol/phiRow are gnomonic xRow/xCol-space phases (pipeline/gridPeriodPhase.ts);
  // u = uvScale*xRow, v = uvScale*xCol is the same conversion projectedUVScale's
  // own doc comment establishes, so this reuses it rather than re-deriving it.
  const uvScale = projectedUVScale(camera);
  if (uvScale === null) return null;
  const uBoundaryRaw = uvScale * gpp.phiCol;
  const vBoundaryRaw = uvScale * gpp.phiRow;
  const uPhase = (uBoundaryRaw - Math.round(uBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;
  const vPhase = (vBoundaryRaw - Math.round(vBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;

  const { minU, maxU, minV, maxV } = bounds;
  const kMinU = Math.floor((minU - uPhase) / GRID_STEP), kMaxU = Math.ceil((maxU - uPhase) / GRID_STEP);
  const kMinV = Math.floor((minV - vPhase) / GRID_STEP), kMaxV = Math.ceil((maxV - vPhase) / GRID_STEP);
  const cols = kMaxU - kMinU + 1, rows = kMaxV - kMinV + 1;
  const zeroI = Math.min(rows - 1, Math.max(0, Math.round(-vPhase / GRID_STEP) - kMinV));
  const zeroJ = Math.min(cols - 1, Math.max(0, Math.round(-uPhase / GRID_STEP) - kMinU));

  return {
    Drow, Dcol, normal, invQuat, distance, halfV, aspect: camera.aspect, binThreshold,
    minGrazingCos: camera.settings.minGrazingCos,
    uPhase, vPhase, kMinU, kMinV, rows, cols, zeroI, zeroJ,
  };
}

// The u/v of one lattice cell -- pure arithmetic in (i, j), no image and no
// ray-cast. This is what lets the GPU path recover its reference point's u/v
// on the host in f64 instead of reading a vec4 back off the device.
export function decodeGridCellUV(layout: DecodeGridLayout, i: number, j: number): { u: number; v: number } {
  return {
    u: layout.uPhase + (layout.kMinU + j) * GRID_STEP,
    v: layout.vPhase + (layout.kMinV + i) * GRID_STEP,
  };
}

export function buildDecodeSampleGrid(camera: PoseCameraLike, gray: Float64Array, w: number, h: number, vFovRad: number): DecodeSampleGrid | null {
  const layout = decodeGridLayout(camera, gray, vFovRad);
  if (!layout) return null;
  const {
    Drow, Dcol, normal, invQuat, distance, halfV, binThreshold, minGrazingCos,
    uPhase, vPhase, kMinU, kMinV, rows, cols, zeroI, zeroJ,
  } = layout;

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
      // CLAMPED, because `valid` above tests px < w on the UNROUNDED value: a
      // sample at px = w - 0.3 passes it and then rounds to w, one past the
      // last column. Measured 2 such points in a 1635-sample grid, always at
      // the bottom/right edge. The old binarize path indexed its Uint8Array
      // out of bounds there and produced bit: undefined -- harmless in the
      // tally by luck ((key << 1) | undefined === key << 1, i.e. it acted as
      // 0), but a genuine undefined leaking into the per-cell correctness
      // overlay. Clamping is what a nearest-pixel sample should do anyway.
      const xx = Math.min(w - 1, Math.max(0, Math.round(px)));
      const yy = Math.min(h - 1, Math.max(0, Math.round(py)));
      rowPoints.push({ u, v, px, py, valid: true, bit: gray[yy * w + xx] < binThreshold ? 1 : 0 });
    }
    points.push(rowPoints);
  }
  return { rows, cols, zeroI, zeroJ, points };
}

// Decodes the camera's absolute world position -- see pre-Stage-A history
// for the full derivation.
export async function runPositionDecode(
  camera: PoseCameraLike, gray: Float64Array, w: number, h: number, vFovRad: number,
  // The LSD chain's device-resident gray, when the chain ran on GPU and its
  // residency is still alive. Purely an optimization -- null means decode
  // uploads its own copy exactly as it always did.
  sharedGray: GPUBuffer | null,
  backend: Backend,
  // When set, the fused path's grid readback is parked on
  // camera.pendingDecodeGrid instead of awaited here. Only a caller with a drain
  // to resolve it in should pass true -- see PendingDecodeGrid.
  defer = false,
) {
  // A previous frame's handle can still be sitting here if the drain never got
  // to it. Released BEFORE this frame overwrites the pointer, since nothing else
  // holds it and the GPU buffers would otherwise live until device loss.
  camera.pendingDecodeGrid?.release();
  camera.pendingDecodeGrid = null;
  // ── Fused GPU path: grid built on device, tally consumes it in place ────
  //
  // The fused path: grid built AND tallied on device. It supersedes the
  // tally-only GPU route below, which now runs only as the fused path's own
  // fallback (it packs and uploads a CPU-built grid -- 298KB per call at a
  // 270x276 lattice). Here the
  // packed grid never crosses the bus in either direction on the pose path:
  // what comes back is the winner and two correctness counts, and the reference
  // cell's u/v are recomputed on the host in f64 rather than read back.
  //
  // The grid is still read back UNCONDITIONALLY -- but as of 2026-08-05 not
  // necessarily HERE. `defer` hands the readback to the caller as a
  // PendingDecodeGrid instead of awaiting it on the pose path.
  //
  // Read the distinction carefully, because the obvious version of this was
  // tried and rejected. SKIPPING the readback when no view wants it was the
  // rejected one: it leaves lastDecodeGrid/lastDecodeRotated/
  // lastDecodeCorrectness null outside Projected-Cam mode, and consumers read
  // those fields synchronously -- mobileCapture's AR readout among them -- so it
  // buys ~1.3ms in exchange for a real behavioural difference between the CPU
  // and GPU routes. That trade is still refused.
  //
  // DEFERRING is not that trade. Whoever takes the handle resolves it, so the
  // same three fields end up populated with the same values; only the moment
  // moves. A caller with nowhere to defer TO (the phone, the timing harness)
  // passes defer=false and gets exactly the old behaviour.
  //
  // The win that stands either way: the PACKED grid never crosses the bus at all
  // (298KB of upload per call at a 270x276 lattice, gone).
  //
  // Falls through to the CPU pair below on failure -- but the two ways of
  // failing are NOT alike, and only one of them is a recovery:
  //
  //   fused === null is an ENVIRONMENT failure (no device, a validation error,
  //   device loss, a limit exceeded). The input is perfectly decodable and the
  //   CPU is simply not subject to whatever refused, so the fall-through
  //   genuinely rescues the frame.
  //
  //   layout === null is a DATA state -- no recovered axes, no period/phase, no
  //   uvScale, i.e. "there is no pose to decode yet". buildDecodeSampleGrid
  //   calls the SAME decodeGridLayout with the same arguments, so it
  //   deterministically reproduces the same null and gives up. The
  //   fall-through recomputes an answer it already has.
  //
  // That redundancy is deliberate and worth keeping: the alternative is
  // returning early here, which means duplicating the four assignments that
  // define the give-up state (lastDecodeGrid/lastDecodeRotated/
  // lastPositionDecode/lastDecodeCorrectness all cleared). Two places that must
  // agree on what "no decode this frame" means is a worse failure mode than one
  // extra decodeGridLayout call, which does no per-pixel work.
  if (backend === 'gpu') {
    const fusedSpan = spanStart('decode (fused GPU build+tally)');
    const layout = decodeGridLayout(camera, gray, vFovRad);
    const fused = layout ? await buildAndTallyDecodeGPU(layout, gray, w, h, sharedGray) : null;
    if (layout && fused) {
      try {
        // Fills the three display fields off the device-side grid. Closed over
        // rather than inlined so the deferred and immediate paths run the SAME
        // code -- a deferral that quietly did something slightly different from
        // what it replaced would be indistinguishable from a bug in the pose.
        const applyGrid = async () => {
          const built = await fused.readGrid();
          camera.lastDecodeGrid = built;
          const rot = rotateGrid(built, fused.winner.orientation);
          camera.lastDecodeRotated = rot;
          camera.lastDecodeCorrectness =
            buildCorrectnessArray(rot, fused.winner.anchorRow, fused.winner.anchorCol).correctness;
        };

        // The pose needs NONE of that. The reference cell is unchanged by
        // rotation -- only its INDEX moves -- so its u/v are the original
        // lattice's (zeroI, zeroJ), computed here in f64 rather than read back;
        // the counts came down with the winner. Hoisted above the grid work so
        // the ordering says so out loud, and so the deferred path finalizes the
        // pose at exactly the same point the immediate one does.
        const ref = decodeGridCellUV(layout, layout.zeroI, layout.zeroJ);
        const [rzI, rzJ] = rotatedZeroIndex(layout.rows, layout.cols, layout.zeroI, layout.zeroJ, fused.winner.orientation);
        finishPositionDecode(
          camera, vFovRad, fused.winner, ref.u, ref.v, rzI, rzJ,
          fused.correctCount, fused.wrongCount,
        );

        if (defer) {
          // STALE FIELDS ARE CLEARED, not left showing the previous frame. The
          // drain is a frame or two away and an overlay repainting in between
          // would otherwise mix this frame's pose with the last frame's decode
          // -- the exact "picture that never existed" drainVisuals' own guard
          // exists to prevent.
          camera.lastDecodeGrid = null;
          camera.lastDecodeRotated = null;
          camera.lastDecodeCorrectness = null;
          let done = false;
          camera.pendingDecodeGrid = {
            resolve: async () => {
              if (done) return;
              done = true;
              try { await applyGrid(); } finally { fused.release(); }
            },
            release: () => { done = true; fused.release(); },
          };
        } else {
          await applyGrid();
        }
      } finally {
        // Only the immediate path frees here; the deferred one handed ownership
        // to the pendingDecodeGrid above and releases through it. fused.release()
        // is idempotent (see buildAndTallyDecodeGPU), so the double call on the
        // deferred path's own error unwind is harmless.
        if (!camera.pendingDecodeGrid) fused.release();
        spanEnd(fusedSpan);
      }
      return;
    }
    spanEnd(fusedSpan);
    // fall through to CPU
  }

  // Sub-spans so the decode stage's two halves are separable in the profiler.
  // They answer a specific question: buildDecodeSampleGrid is a candidate for a
  // GPU port, but only if it is actually expensive relative to a `gray` upload
  // -- rows x cols is on the order of a thousand ray-casts, which may well be
  // cheaper on CPU than moving the image across. Measure before porting.
  const buildSpan = spanStart('buildDecodeSampleGrid');
  const grid = buildDecodeSampleGrid(camera, gray, w, h, vFovRad);
  spanEnd(buildSpan);
  camera.lastDecodeGrid = grid;
  camera.lastDecodeRotated = null;
  if (!grid) { camera.lastPositionDecode = null; camera.lastDecodeCorrectness = null; return; }
  // Same GPU-source-of-truth-verified-by-CPU-fallback pattern as every other
  // GPU sub-pipeline (axesReconstruction.ts's gradient2x2Field/projectBins/
  // fitPairOfPlanes) -- see tallyPositionVotesGPU's own header for why this
  // one is currently a manual toggle rather than always-on (measured SLOWER
  // than CPU for typical grid sizes, expected to flip in the GPU's favor for
  // larger decode grids).
  const tallySpan = spanStart(backend === 'cpu' ? 'tallyPositionVotes (CPU)' : 'tallyPositionVotes (GPU)');
  const winner = backend === 'gpu'
    ? (await tallyPositionVotesGPU(grid)) ?? tallyPositionVotes(grid)
    : tallyPositionVotes(grid);
  spanEnd(tallySpan);
  if (!winner) { camera.lastPositionDecode = null; camera.lastDecodeCorrectness = null; return; }

  const rotated = rotateGrid(grid, winner.orientation);
  camera.lastDecodeRotated = rotated;
  const { correctness, correctCount, wrongCount } = buildCorrectnessArray(rotated, winner.anchorRow, winner.anchorCol);
  camera.lastDecodeCorrectness = correctness;
  const refPt = rotated.points[rotated.zeroI][rotated.zeroJ];
  finishPositionDecode(
    camera, vFovRad, winner, refPt.u, refPt.v, rotated.zeroI, rotated.zeroJ, correctCount, wrongCount,
  );
}

// Per-cell correctness for the projected-cam overlay, plus the two counts that
// feed `consistency`. Only the counts matter off the display path, which is why
// the fused GPU route computes THOSE on device (see decodeGridBuild.wgsl.ts's
// correctness entry point) and calls this only when the array itself is wanted.
export function buildCorrectnessArray(
  rotated: DecodeSampleGrid, anchorRow: number, anchorCol: number,
): { correctness: (DecodeCellDebug | null)[][]; correctCount: number; wrongCount: number } {
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
  return { correctness, correctCount, wrongCount };
}

// Everything downstream of "we have a winner": torus registration, the closed-
// form camera orientation, and the world position. Shared verbatim by the CPU
// and fused-GPU routes so the two can't drift on the pose itself.
//
// Takes the reference cell's u/v and its ROTATED index as plain numbers rather
// than a grid, because the fused route has no CPU-side grid to read them from --
// it recomputes u/v arithmetically (decodeGridCellUV) and maps the index with
// rotatedZeroIndex. That is precisely what lets it skip the grid readback.
function finishPositionDecode(
  camera: PoseCameraLike, vFovRad: number, winner: VoteResult,
  refPtU: number, refPtV: number, rotZeroI: number, rotZeroJ: number,
  correctCount: number, wrongCount: number,
) {
  const consistency = correctCount + wrongCount > 0 ? correctCount / (correctCount + wrongCount) : 0;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes!; // a non-null grid/layout guarantees this
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const refTorusRow = ((winner.anchorRow + rotZeroI) % R + R) % R;
  const refTorusCol = ((winner.anchorCol + rotZeroJ) % C + C) % C;

  const recoveredCamQuat = solveRecoveredCamQuat(Drow, Dcol, winner.orientation);
  const DrowWorld = Drow.clone().applyQuaternion(recoveredCamQuat);
  const DcolWorld = Dcol.clone().applyQuaternion(recoveredCamQuat);
  const normalWorld = normal.clone().applyQuaternion(recoveredCamQuat);
  const hitRelWorld = new THREE.Vector3()
    .addScaledVector(DrowWorld, refPtU).addScaledVector(DcolWorld, refPtV).addScaledVector(normalWorld, -distance);

  const worldPosTrue = new THREE.Vector3((refTorusCol + 0.5 - C / 2) * GRID_STEP, 0, (refTorusRow + 0.5 - R / 2) * GRID_STEP);
  camera.lastPositionDecode = {
    row: refTorusRow, col: refTorusCol, consistency, votes: winner.votes, totalWindows: winner.totalWindows,
    camPos: worldPosTrue.sub(hitRelWorld),
    recoveredCamQuat,
    orientation: winner.orientation,
  };
}

