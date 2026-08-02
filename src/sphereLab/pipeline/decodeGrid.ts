import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { GRID_STEP, MATH_QUAT } from '../constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../math/geometry.ts';
import { tallyPositionVotesGPU } from '../pipelineGPU/decodeTally.ts';
import { projectSamplesGPU } from '../pipelineGPU/projectSamples.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { C, ORDER, R, debruijnLookup, torus } from '../floorPattern.ts';
import { globalState } from '../state.ts';
import { DecodeCellDebug, DecodeSampleGrid, DecodeSamplePoint, GradientField, Marginals, PositionDecodeResult, ProjectedBins, ProjectedSamplesDense, RecoveredAxes, VoteResult } from '../types.ts';
import { computeGradientField } from './gradientField.ts';
import { GridPeriodPhaseResult } from './gridPeriodPhase.ts';
import { computeProjectedMarginals } from './positionLM.ts';

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
}

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
  const { rows: gr, cols: gc, zeroI: i, zeroJ: j } = grid;
  const [zeroI, zeroJ] = o === 1 ? [j, gr - 1 - i] : o === 2 ? [gr - 1 - i, gc - 1 - j] : [gc - 1 - j, i];
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
export function solveRecoveredCamQuat(Drow: THREE.Vector3, Dcol: THREE.Vector3, orientation: number): THREE.Quaternion {
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

export type ProjectedSampleResult = ReturnType<typeof castAndBucketProjectedSamples>;

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
// paint doesn't have to re-cast every ray a second time. No longer computes
// marginals (autocorrelation) here -- see this session's chat: that was
// display-only (the marginal-graph overlay, now removed) and decode gets its
// own phase from gridPeriodPhase instead. Projects once (stage 1) so the
// resulting extent can size a square-cell bucket grid (stage 2) BEFORE
// bucketing -- can't use castAndBucketProjectedSamples' single-call
// convenience here since that picks bucketW/bucketH before the extent
// (which stage 1 alone produces) is known.
export function computeProjectedBinsAndMarginals(camera: Camera): ProjectedSampleResult {
  const proj = camera.lastRecoveredAxes ? projectSamplesCPU(camera) : null;
  if (!proj) { camera.lastProjectedBins = null; return null; }
  const { bucketW, bucketH } = squareCellBucketDims(camera, proj.maxU - proj.minU, proj.maxV - proj.minV);
  const result = bucketSamples(camera, bucketW, bucketH, proj);
  camera.lastProjectedBins = result.bins;
  return result;
}

// GPU-aware twin, deliberately kept separate rather than folded into
// computeProjectedBinsAndMarginals above -- that function has several
// call sites outside the reconstruction pipeline (throttled preview
// updates, mode switches, camera creation) that used to be perfectly fine
// staying synchronous; now that those call sites (see modeRefresh.ts/
// main.ts/ui/mode.ts) have all gone async too (see this session's chat --
// buildProjectedTexture used to silently bypass globalState.useGPUProject
// entirely, running a redundant CPU-only re-projection on every
// reconstruction pass and every throttled preview tick), every caller can
// safely go through computeProjectedBinsAndMarginalsAuto below instead of
// picking CPU vs GPU itself.
export async function computeProjectedBinsAndMarginalsGPU(camera: Camera): Promise<ProjectedSampleResult> {
  const proj = camera.lastRecoveredAxes ? await projectSamplesGPU(camera) : null;
  if (!proj) { camera.lastProjectedBins = null; return null; }
  const { bucketW, bucketH } = squareCellBucketDims(camera, proj.maxU - proj.minU, proj.maxV - proj.minV);
  const result = bucketSamples(camera, bucketW, bucketH, proj);
  camera.lastProjectedBins = result.bins;
  return result;
}

// Single dispatch point for every caller (axesReconstruction.ts's
// recomputeStages, and modeRefresh.ts's buildProjectedTexture) --
// centralizes the globalState.useGPUProject check once instead of
// duplicating the GPU-with-CPU-fallback ternary at each call site.
export async function computeProjectedBinsAndMarginalsAuto(camera: Camera): Promise<ProjectedSampleResult> {
  const s = spanStart(globalState.useGPUProject ? 'projectBins (GPU stage 1 + CPU bucket)' : 'projectBins (CPU)');
  const result = globalState.useGPUProject
    ? (await computeProjectedBinsAndMarginalsGPU(camera)) ?? computeProjectedBinsAndMarginals(camera)
    : computeProjectedBinsAndMarginals(camera);
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
export async function buildProjectedTexture(camera: Camera, precomputed?: { value: ProjectedSampleResult }): Promise<void> {
  const seq = (projTextureSeq.get(camera) ?? 0) + 1;
  projTextureSeq.set(camera, seq);
  const result = precomputed ? precomputed.value : await computeProjectedBinsAndMarginalsAuto(camera);
  if (projTextureSeq.get(camera) !== seq) return; // a newer call started meanwhile -- its result wins instead
  paintProjectedTexture(camera, result);
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
export function projectedUVBounds(camera: PoseCameraLike): { minU: number; maxU: number; minV: number; maxV: number } | null {
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
export function buildDecodeSampleGrid(camera: PoseCameraLike, gray: Float64Array, w: number, h: number, vFovRad: number): DecodeSampleGrid | null {
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

  const { minU, maxU, minV, maxV } = bounds;
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
export async function runPositionDecode(camera: PoseCameraLike, gray: Float64Array, w: number, h: number, vFovRad: number) {
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
  const tallySpan = spanStart(globalState.useGPUDecode ? 'tallyPositionVotes (GPU)' : 'tallyPositionVotes (CPU)');
  const winner = globalState.useGPUDecode
    ? (await tallyPositionVotesGPU(grid)) ?? tallyPositionVotes(grid)
    : tallyPositionVotes(grid);
  spanEnd(tallySpan);
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

  const recoveredCamQuat = solveRecoveredCamQuat(Drow, Dcol, winner.orientation);

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

