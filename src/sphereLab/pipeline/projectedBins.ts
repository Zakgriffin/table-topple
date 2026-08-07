import * as THREE from 'three';
import { type Camera } from '../camera/model.ts';
import { MATH_QUAT } from '../constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../math/geometry.ts';
import { projectSamplesGPU } from './projectSamples.gpu.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { type ProjectedBins, type ProjectedSamplesDense } from '../types.ts';
import { type GradientField } from '../../pose/results.ts';
import { type Backend } from '../../pose/backend.ts';
import { computeGradientField } from '../../pose/stages/gradient/gradientField.ts';

// ── The projection stage: DISPLAY, not pose ───────────────────────────────
//
// Split out of decodeGrid.ts, where it had been sitting next to the decode it
// has nothing to do with. poseCompute.ts's own header has said the whole time
// that computeProjectedBinsAuto/paintProjectedTexture are deliberately NOT on
// the path to a pose -- distance is already final by the time gridPeriodPhase
// returns, and these exist only to feed Projected-Cam and the World-view floor
// decal. This file is that statement made structural.
//
// It is the reason the split is worth doing rather than a tidiness argument:
// every function here takes a full `Camera` and reads RENDER RESOURCES off it
// -- `distortedPreviewData` (an RGBA raster), `projectedPreviewTex` (a THREE
// DataTexture), `rtSize`, and `lastNoisedPreviewGray`, which is the ROW-FLIPPED
// display copy rather than the top-down buffer the pipeline runs on. None of
// that can exist on the library side of the boundary. Meanwhile everything left
// in decodeGrid.ts takes the narrow `PoseCameraLike` and touches none of it.
//
// So this extraction is what lets decodeGrid.ts stop importing `Camera` at all,
// and that single edge is what was dragging camera/model.ts (and, through it,
// camera/settings.ts) into the pose pipeline's RUNTIME import closure --
// `import { type Camera }` is a real module load under verbatimModuleSyntax,
// not an erased one. Measured before this change: 41 modules loaded when node
// imports poseCompute.ts, and those two were in it.

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

// Stage 1 (CPU) -- casts one ray per SCREEN pixel, and if it clears the
// grazing-angle cutoff, projects it onto the recovered floor plane's (u,v)
// frame plus its gradient covector. Dense output (one slot per pixel,
// valid=0 for misses) specifically so this and projectSamplesGPU (see
// pipeline/projectSamples.gpu.ts) can feed the exact same stage-2 bucketing
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
