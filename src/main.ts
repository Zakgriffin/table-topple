import { generateTorus, buildLookupTable } from './debruijn.ts';
import { detectGrid, sampleFullGrid, sampleFromMesh, pickBestCandidate, toGrayscale, binarize, estimateRotationRad, asSignedResidual, buildBoundaries } from './decode.ts';
import type { GridDetection, SampledGrid, CandidateResult } from './decode.ts';
import type { Patch } from './decode.ts';
import { computeJunctionField, detectJunctions, refineJunctionSubPixel } from './cornerdetect.ts';
import type { JunctionType } from './cornerdetect.ts';
import { buildMesh, pruneInconsistentNodes } from './mesh.ts';
import type { Mesh } from './mesh.ts';

// ── Pattern setup ─────────────────────────────────────────────────────────────
//
// Order is fixed per pattern (must match whatever was printed / is on
// screen). Override via ?order=N in the URL if testing a different one.
const ORDER = parseInt(new URLSearchParams(location.search).get('order') ?? '4', 10);

const status = document.getElementById('status')!;
status.textContent = `Building order-${ORDER} torus + lookup table...`;

const debruijn = generateTorus(ORDER);
const lookup = buildLookupTable(debruijn);
const { R, C } = debruijn;
console.log(`Torus ready: ${R}x${C} cells, order ${ORDER}, ${lookup.length} lookup entries.`);

// ── Camera ────────────────────────────────────────────────────────────────────

const canvas = document.getElementById('c') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const video = document.createElement('video');
video.setAttribute('playsinline', '');

const switchCamBtn = document.getElementById('switchCam') as HTMLButtonElement;
const zoomSlider = document.getElementById('zoom') as HTMLInputElement;
let currentStream: MediaStream | null = null;
let currentFacing = 'environment';

// Hardware zoom is a non-standard MediaTrackCapabilities extension (missing
// from TS's DOM lib types, hence the `any`s) with patchy real-world support —
// mainly some Android Chrome + camera hardware combos. No software fallback:
// if the device doesn't report a zoom capability, the slider stays disabled
// and red so that's visibly true rather than silently doing something else.
//
// Zoom is a multiplicative factor (2x, 4x, 8x, ...), so a slider mapped
// linearly to it "feels" wrong: equal slider movement near the low end
// (1x -> 2x) is a much bigger visual change than the same movement near the
// high end (7x -> 8x). The slider's own position stays linear (0..1); we map
// that position to the zoom value geometrically (zoomMin * (zoomMax/zoomMin)^t)
// so equal slider steps feel like equal visual zoom changes throughout.
let zoomMin = 1, zoomMax = 1;

function sliderToZoom(t: number): number {
  return zoomMin * Math.pow(zoomMax / zoomMin, t);
}

function setupZoomControl() {
  const track = currentStream?.getVideoTracks()[0];
  let caps: any = null;
  try { caps = track && 'getCapabilities' in track ? (track as any).getCapabilities() : null; }
  catch { caps = null; }

  if (caps && caps.zoom && caps.zoom.min > 0 && caps.zoom.max > caps.zoom.min) {
    zoomSlider.disabled = false;
    zoomMin = caps.zoom.min;
    zoomMax = caps.zoom.max;
    zoomSlider.min = '0';
    zoomSlider.max = '1';
    zoomSlider.step = '0.001';
    zoomSlider.value = '0'; // default to fully zoomed out
    track!.applyConstraints({ advanced: [{ zoom: zoomMin } as any] }).catch(() => {});
  } else {
    zoomSlider.disabled = true;
  }
}

zoomSlider.addEventListener('input', () => {
  const track = currentStream?.getVideoTracks()[0];
  const zoom = sliderToZoom(parseFloat(zoomSlider.value));
  track?.applyConstraints({ advanced: [{ zoom } as any] }).catch(() => {});
});

async function startCamera(desiredFacing: string) {
  const newStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: desiredFacing } }
  });
  if (currentStream) currentStream.getTracks().forEach(t => t.stop());
  currentStream = newStream;
  video.srcObject = currentStream;
  await video.play();
  const settings = currentStream.getVideoTracks()[0].getSettings();
  currentFacing = settings.facingMode || 'user';
  canvas.classList.toggle('mirror', currentFacing === 'user');
  setupZoomControl();
}

switchCamBtn.addEventListener('click', async () => {
  const next = currentFacing === 'user' ? 'environment' : 'user';
  switchCamBtn.disabled = true;
  try { await startCamera(next); }
  catch (e: any) { status.textContent = 'Camera error: ' + e.message; }
  finally { switchCamBtn.disabled = false; }
});

startCamera('environment')
  .then(() => render())
  .catch((e: any) => { status.textContent = 'Camera error: ' + e.message; });

// ── Mesh-based geometry pipeline (Option B) — on-demand debug analysis ─────────
// Unlike the Stage 2 pipeline below, corner detection is rotation-robust by
// construction (validated up to 75deg, see scripts/test-junctions.ts), so
// this skips the derotation-candidate search entirely — one capture,
// analyzed directly. Still far too slow for every frame (~2.4s/analysis in
// Node testing, 500-1500 junctions on a real capture), so it runs on
// demand (the Analyze button) rather than every render().
//
// End-to-end decode through this pipeline isn't reliable yet — a whole
// connected region of the mesh can drift together during construction and
// still look locally self-consistent to pruneInconsistentNodes, so the
// decoded row/col here should not be trusted. This is wired in as a
// DIAGNOSTIC view of each stage (what got detected, what got kept vs
// pruned, how many corners sourced each cell), not as the app's actual
// position source — the status text above still comes from decodeFrame's
// Stage 2 result.

const meshStatus = document.getElementById('meshStatus')!;
const analyzeBtn = document.getElementById('analyzeBtn') as HTMLButtonElement;
const toggleJunctions = document.getElementById('toggleJunctions') as HTMLInputElement;
const toggleMesh = document.getElementById('toggleMesh') as HTMLInputElement;
const toggleCells = document.getElementById('toggleCells') as HTMLInputElement;
const toggleStage2 = document.getElementById('toggleStage2') as HTMLInputElement;

const MESH_ANALYSIS_BUDGET = 320 * 320;
const MESH_PATCH = 120; // small central patch, only for an accurate pitch seed under roll
const meshRawCanvas = document.createElement('canvas');
const meshRawCtx = meshRawCanvas.getContext('2d', { willReadFrequently: true })!;
const meshPatchCanvas = document.createElement('canvas');
const meshPatchCtx = meshPatchCanvas.getContext('2d', { willReadFrequently: true })!;

interface MeshAnalysisResult {
  cropSx: number; cropSy: number; rawScale: number; rawW: number; rawH: number;
  theta0: number;
  refinedJunctions: { x: number; y: number; type: JunctionType }[];
  rawMesh: Mesh;
  prunedMesh: Mesh;
  sampledGrid: SampledGrid | null;
  decodeResult: CandidateResult | null;
}

let meshResult: MeshAnalysisResult | null = null;

function rawToVideo(rx: number, ry: number, d: MeshAnalysisResult): { x: number; y: number } {
  return { x: d.cropSx + rx * d.rawScale, y: d.cropSy + ry * d.rawScale };
}

function runMeshAnalysis() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;

  const cropW = vw * 0.95, cropH = vh * 0.95;
  const cropSx = (vw - cropW) / 2, cropSy = (vh - cropH) / 2;
  const aspect = cropW / cropH;
  const rawH = Math.round(Math.sqrt(MESH_ANALYSIS_BUDGET / aspect));
  const rawW = Math.round(rawH * aspect);
  const rawScale = cropW / rawW;

  meshRawCanvas.width = rawW; meshRawCanvas.height = rawH;
  meshRawCtx.drawImage(video, cropSx, cropSy, cropW, cropH, 0, 0, rawW, rawH);
  const rawRgba = meshRawCtx.getImageData(0, 0, rawW, rawH).data;
  const gray = toGrayscale(rawRgba, rawW, rawH);
  const bin = binarize(gray);

  const theta0 = estimateRotationRad(gray, rawW, rawH);

  // Pitch estimate needs derotating first to be accurate under roll (it
  // assumes axis-aligned grid lines to measure pitch, which roll breaks) —
  // a small central patch only, via canvas rotate.
  meshPatchCanvas.width = MESH_PATCH; meshPatchCanvas.height = MESH_PATCH;
  meshPatchCtx.fillStyle = '#fff';
  meshPatchCtx.fillRect(0, 0, MESH_PATCH, MESH_PATCH);
  meshPatchCtx.save();
  meshPatchCtx.translate(MESH_PATCH / 2, MESH_PATCH / 2);
  meshPatchCtx.rotate(-theta0);
  meshPatchCtx.translate(-rawW / 2, -rawH / 2);
  meshPatchCtx.drawImage(meshRawCanvas, 0, 0);
  meshPatchCtx.restore();
  const patchGray = toGrayscale(meshPatchCtx.getImageData(0, 0, MESH_PATCH, MESH_PATCH).data, MESH_PATCH, MESH_PATCH);
  const patchBin = binarize(patchGray);
  const coarseGrid = detectGrid(patchBin, MESH_PATCH, MESH_PATCH);

  // tensorRadius/minDistance scale with the actual apparent pitch — a fixed
  // default doesn't transfer across different capture distances/zoom.
  const apparentPitch = (coarseGrid.pitchX + coarseGrid.pitchY) / 2;
  const tensorRadius = Math.max(2, Math.round(apparentPitch / 4));
  const minDistance = Math.max(5, Math.round(tensorRadius * 2.5));

  const field = computeJunctionField(gray, rawW, rawH, 1, tensorRadius);
  const coarseJ = detectJunctions(field, 0.15, minDistance);
  const refinedJunctions = coarseJ.map(j => {
    const r = refineJunctionSubPixel(gray, rawW, rawH, j.x, j.y);
    return { x: r.x, y: r.y, type: j.type };
  });

  const rawMesh = buildMesh(refinedJunctions, rawW / 2, rawH / 2, coarseGrid.pitchX, coarseGrid.pitchY, -theta0, 3, 0.45);
  const prunedMesh = pruneInconsistentNodes(rawMesh, apparentPitch);

  let sampledGrid: SampledGrid | null = null;
  let decodeResult: CandidateResult | null = null;
  if (prunedMesh.nodes.length >= ORDER * ORDER) {
    sampledGrid = sampleFromMesh(bin, rawW, rawH, prunedMesh);
    if (sampledGrid.rows >= ORDER && sampledGrid.cols >= ORDER) {
      decodeResult = pickBestCandidate([sampledGrid], ORDER, lookup, debruijn.torus, R, C);
    }
  }

  meshResult = { cropSx, cropSy, rawScale, rawW, rawH, theta0, refinedJunctions, rawMesh, prunedMesh, sampledGrid, decodeResult };

  const locked = !!decodeResult?.match && decodeResult.consistency >= CONFIDENCE_THRESHOLD;
  meshStatus.textContent = [
    `mesh: ${refinedJunctions.length} junctions, ${rawMesh.nodes.length}->${prunedMesh.nodes.length} nodes (pruned)`,
    sampledGrid ? `sampled ${sampledGrid.rows}x${sampledGrid.cols}` : 'sampled: n/a',
    decodeResult ? `score ${(decodeResult.consistency * 100).toFixed(0)}%${locked ? ` -> row ${decodeResult.match!.row} col ${decodeResult.match!.col}` : ' (no lock)'}` : 'decode: n/a',
  ].join('\n');
}

// Runs one analysis pass, painting "analyzing..." first since the pass
// itself blocks the main thread for a while (double rAF: the DOM update is
// guaranteed painted by the time the heavy synchronous work in the second
// callback runs — a single rAF doesn't guarantee the paint has happened
// yet). Calls `after` once done, whether or not anything was actually drawn.
function triggerAnalysis(after: () => void) {
  analyzeBtn.disabled = true;
  meshStatus.textContent = 'analyzing...';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    try { runMeshAnalysis(); }
    finally { analyzeBtn.disabled = false; after(); }
  }));
}

// Auto-refresh: junctions/mesh/cells are all derived from the same
// runMeshAnalysis() pass, so one toggle being on is enough reason to keep
// it current — otherwise checking "junctions" just shows an increasingly
// stale single snapshot as you move the camera. Re-triggers itself only
// after the PREVIOUS pass finishes (not a fixed-cadence interval), since a
// pass can take a while and overlapping runs would just queue up jank.
// Stops entirely once every layer is toggled off, so idle cost is zero.
const MESH_REFRESH_DELAY_MS = 1200;
let autoRefreshTimer: number | null = null;

function anyMeshLayerVisible(): boolean {
  return toggleJunctions.checked || toggleMesh.checked || toggleCells.checked;
}

function scheduleAutoRefresh() {
  if (autoRefreshTimer !== null || !anyMeshLayerVisible()) return;
  autoRefreshTimer = window.setTimeout(() => {
    autoRefreshTimer = null;
    if (!anyMeshLayerVisible()) return;
    triggerAnalysis(scheduleAutoRefresh);
  }, MESH_REFRESH_DELAY_MS);
}

for (const toggle of [toggleJunctions, toggleMesh, toggleCells]) {
  toggle.addEventListener('change', scheduleAutoRefresh);
}

analyzeBtn.addEventListener('click', () => triggerAnalysis(scheduleAutoRefresh));

// ── Grid decode pipeline (Stage 2: rotation + uniform scale) ───────────────────
// See src/decode.ts's module header for the full algorithm writeup. Summary:
//   1. Crop nearly the WHOLE video viewport (not just a small centered
//      square) — when the camera is close enough that only one patch's
//      worth of cells is visible, cross-patch consistency has no neighbors
//      to check, so every available cell matters. The capture rectangle
//      matches the video's real aspect ratio rather than being limited to a
//      square bounded by the shorter dimension.
//   2. Estimate rotation mod 90 degrees via gradient-orientation histogram.
//   3. Derotate at that angle plus 90/180/270-degree offsets (4 candidates —
//      edge orientation alone can't tell which of the 4 is the true one).
//      The square RAW buffer that rotation happens in is sized off the
//      ALIGNED rectangle's diagonal (not a fixed side), since a wider or
//      taller capture needs more rotation margin to stay fully covered.
//   4. For each candidate: detect grid pitch/phase, sample every visible
//      cell, tile into discrete order x order patches, decode each patch.
//   5. Keep whichever candidate's exact-match patches yield the best-
//      correlated position against the actual known pattern (see
//      pickBestCandidate / scoreCorrelation in decode.ts) — this tolerates
//      individual misread bits gracefully rather than exact-match's
//      all-or-nothing, and gives a MUCH better-separated confidence score
//      than checking patches only agree with each other: empirically,
//      correct decodes score ~1.0 (degrading smoothly with real bit noise),
//      wrong ones sit around ~0.55-0.6 (close to the 0.5 "uncorrelated"
//      baseline) — see scripts/test-decode.ts's noise-injection test.

const CONFIDENCE_THRESHOLD = 0.85; // correlation score vs the known pattern; wrong matches empirically top out ~0.6
const ANALYSIS_BUDGET = 190 * 190; // target ALIGNED_w * ALIGNED_h, keeps analysis cost roughly constant

const rawCanvas = document.createElement('canvas');
const rawCtx = rawCanvas.getContext('2d', { willReadFrequently: true })!;
const alignedCanvas = document.createElement('canvas');
const alignedCtx = alignedCanvas.getContext('2d', { willReadFrequently: true })!;

interface DecodeResult {
  match: { row: number; col: number } | null;
  patches: Patch[];
  consistency: number;
  theta: number; // resolved full rotation angle (radians) for the winning candidate
  grid: GridDetection;
  alignedW: number; alignedH: number;
  cropSx: number; cropSy: number; rawScale: number; rawSide: number; contentDx: number; contentDy: number;
}

// Maps a point in the (winning candidate's) ALIGNED buffer back to video
// coordinates, for overlay drawing: aligned -> raw (inverse of the
// ctx.rotate(-theta) derotation, see src/decode.ts's header) -> video. RAW's
// content is drawn centered with a padding margin (contentDx/Dy) since RAW
// is sized for rotation coverage, not 1:1 with the real capture rectangle.
function alignedToVideo(ax: number, ay: number, d: DecodeResult): { x: number; y: number } {
  const relX = ax - d.alignedW / 2, relY = ay - d.alignedH / 2;
  const cosT = Math.cos(d.theta), sinT = Math.sin(d.theta);
  const rx = relX * cosT - relY * sinT + d.rawSide / 2;
  const ry = relX * sinT + relY * cosT + d.rawSide / 2;
  return { x: d.cropSx + (rx - d.contentDx) * d.rawScale, y: d.cropSy + (ry - d.contentDy) * d.rawScale };
}

function decodeFrame(): DecodeResult | null {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;

  const cropW = vw * 0.95, cropH = vh * 0.95;
  const cropSx = (vw - cropW) / 2, cropSy = (vh - cropH) / 2;

  const aspect = cropW / cropH;
  const alignedH = Math.round(Math.sqrt(ANALYSIS_BUDGET / aspect));
  const alignedW = Math.round(alignedH * aspect);
  const rawScale = cropW / alignedW; // == cropH / alignedH
  const rawSide = Math.ceil(Math.sqrt(alignedW * alignedW + alignedH * alignedH)); // covers ALIGNED's diagonal at any rotation
  const contentDx = (rawSide - alignedW) / 2, contentDy = (rawSide - alignedH) / 2;

  rawCanvas.width = rawSide; rawCanvas.height = rawSide;
  alignedCanvas.width = alignedW; alignedCanvas.height = alignedH;

  rawCtx.fillStyle = '#fff';
  rawCtx.fillRect(0, 0, rawSide, rawSide);
  rawCtx.drawImage(video, cropSx, cropSy, cropW, cropH, contentDx, contentDy, alignedW, alignedH);

  const rawImg = rawCtx.getImageData(0, 0, rawSide, rawSide).data;
  const rawGray = toGrayscale(rawImg, rawSide, rawSide);

  // Derotates rawCanvas by theta into alignedCanvas and returns its grayscale.
  const derotateToGray = (theta: number): Float64Array => {
    alignedCtx.save();
    alignedCtx.translate(alignedW / 2, alignedH / 2);
    alignedCtx.rotate(-theta);
    alignedCtx.translate(-rawSide / 2, -rawSide / 2);
    alignedCtx.drawImage(rawCanvas, 0, 0);
    alignedCtx.restore();
    return toGrayscale(alignedCtx.getImageData(0, 0, alignedW, alignedH).data, alignedW, alignedH);
  };

  const thetaCoarse = estimateRotationRad(rawGray, rawSide, rawSide);

  // Coarse-to-fine: derotate once with the coarse estimate, then re-run the
  // SAME estimator on that now-mostly-aligned result to correct residual
  // angular error. Error from a wrong rotation grows with distance from the
  // pivot, so this matters more now that capture uses the full viewport
  // (bigger radius) than it did with the old small centered crop. Reuses
  // estimateRotationRad rather than adding new rotation-specific machinery,
  // since this single-angle model is scoped to be superseded by full
  // homography estimation later anyway.
  const previewGray = derotateToGray(thetaCoarse);
  const residual = asSignedResidual(estimateRotationRad(previewGray, alignedW, alignedH));
  const theta0 = thetaCoarse + residual;

  const grids: GridDetection[] = [];
  const sampledGrids = [0, 1, 2, 3].map(k => {
    const theta = theta0 + k * (Math.PI / 2);
    const alignedBin = binarize(derotateToGray(theta));
    const grid = detectGrid(alignedBin, alignedW, alignedH);
    grids.push(grid);
    return sampleFullGrid(alignedBin, alignedW, alignedH, grid);
  });

  const best = pickBestCandidate(sampledGrids, ORDER, lookup, debruijn.torus, R, C);
  if (best.candidateIndex === -1) return null; // no candidate found any valid reading orientation at all
  const theta = theta0 + best.candidateIndex * (Math.PI / 2);
  const grid = grids[best.candidateIndex];
  const match = best.consistency >= CONFIDENCE_THRESHOLD ? best.match : null;

  return {
    match, patches: best.patches, consistency: best.consistency, theta, grid,
    alignedW, alignedH, cropSx, cropSy, rawScale, rawSide, contentDx, contentDy,
  };
}

// ── Render loop ───────────────────────────────────────────────────────────────

function render() {
  requestAnimationFrame(render);
  if (video.readyState < 2) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const scaleToFit = Math.max(window.innerWidth / canvas.width, window.innerHeight / canvas.height);
  canvas.style.width = (canvas.width * scaleToFit) + 'px';
  canvas.style.height = (canvas.height * scaleToFit) + 'px';
  ctx.drawImage(video, 0, 0);

  const result = decodeFrame();

  if (result && toggleStage2.checked) {
    const toVideo = (ax: number, ay: number) => alignedToVideo(ax, ay, result);
    const confident = result.consistency >= CONFIDENCE_THRESHOLD;

    // Blue lines for the estimated grid edges (both line families), so you
    // can visually check the detected grid against the real one on screen.
    // Uses the SAME buildBoundaries walk sampleFullGrid uses internally,
    // so the drawn lines stay truthful to what's actually being sampled.
    const { px, py, pitchX, pitchY } = result.grid;
    ctx.strokeStyle = 'rgba(60,140,255,0.8)';
    ctx.lineWidth = Math.max(1, result.rawScale * 1.2);
    const { boundaries: xB } = buildBoundaries(px, pitchX, 0, result.alignedW);
    const { boundaries: yB } = buildBoundaries(py, pitchY, 0, result.alignedH);
    for (const x of xB) {
      const a = toVideo(x, 0), b = toVideo(x, result.alignedH);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    for (const y of yB) {
      const a = toVideo(0, y), b = toVideo(result.alignedW, y);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }

    // Each patch drawn as a discrete outlined quad (green if it decoded to a
    // valid position AND the frame is confident overall, orange if it found
    // a lookup hit but overall confidence is low, red if no hit), containing
    // its own black/white dots so patches are visually distinguishable.
    for (const patch of result.patches) {
      // Derived from the patch's own sampled cell positions (always correct)
      // rather than px + tileCol*order*pitchX — that formula assumed tile 0
      // starts AT px, which broke once px was anchored near the buffer
      // center with tiles extending both directions from it (same root
      // cause as the blue-line fix above).
      const x0 = patch.cells[0][0].x - pitchX / 2, x1 = patch.cells[0][ORDER - 1].x + pitchX / 2;
      const y0 = patch.cells[0][0].y - pitchY / 2, y1 = patch.cells[ORDER - 1][0].y + pitchY / 2;
      const corners = [toVideo(x0, y0), toVideo(x1, y0), toVideo(x1, y1), toVideo(x0, y1)];
      ctx.strokeStyle = patch.match ? (confident ? '#0f0' : '#fa0') : '#f00';
      ctx.lineWidth = Math.max(1, result.rawScale);
      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      for (const c of corners.slice(1)) ctx.lineTo(c.x, c.y);
      ctx.closePath();
      ctx.stroke();

      const dotRadius = Math.max(2, pitchX * result.rawScale * 0.25);
      for (let i = 0; i < ORDER; i++) {
        for (let j = 0; j < ORDER; j++) {
          const cell = patch.cells[i][j];
          const p = toVideo(cell.x, cell.y);
          ctx.beginPath();
          ctx.arc(p.x, p.y, dotRadius, 0, Math.PI * 2);
          ctx.fillStyle = cell.bit ? '#000' : '#fff';
          ctx.fill();
          ctx.lineWidth = Math.max(1, dotRadius * 0.3);
          // Per-cell ground truth when available (does THIS cell match the
          // actual pattern, not just "did the patch find a match") — falls
          // back to the coarser patch-level match when no anchor was found
          // to compare against at all (see Patch.correct in decode.ts).
          ctx.strokeStyle = patch.correct ? (patch.correct[i][j] ? '#0f0' : '#f00') : (patch.match ? '#0f0' : '#f00');
          ctx.stroke();
        }
      }
    }
  }

  status.textContent = result?.match
    ? `order ${ORDER}  torus ${R}x${C}\nrow ${result.match.row}  col ${result.match.col}\nconfidence ${(result.consistency * 100).toFixed(0)}%`
    : `order ${ORDER}  torus ${R}x${C}\nno lock — move closer / center the pattern`;

  if (meshResult) drawMeshDebug(meshResult);
}

// Draws whichever of the mesh-pipeline's diagnostic layers are toggled on,
// from the last Analyze snapshot (see runMeshAnalysis — this data is stale
// relative to the live video by however long ago the button was pressed,
// unlike everything else in render() which is per-frame).
function drawMeshDebug(d: MeshAnalysisResult) {
  const toVideo = (rx: number, ry: number) => rawToVideo(rx, ry, d);
  const dotR = Math.max(2, d.rawScale * 4);

  if (toggleJunctions.checked) {
    for (const j of d.refinedJunctions) {
      const p = toVideo(j.x, j.y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2);
      ctx.fillStyle = j.type === 'saddle' ? '#0ff' : '#fa0'; // saddle=cyan (precise), lcorner=amber (majority, less precise)
      ctx.fill();
    }
  }

  if (toggleMesh.checked) {
    const kept = d.prunedMesh.byCoord;
    // Links between adjacent surviving nodes, so the mesh reads as a graph
    // rather than a scatter of dots.
    ctx.strokeStyle = 'rgba(0,255,0,0.6)';
    ctx.lineWidth = Math.max(1, d.rawScale);
    for (const node of d.prunedMesh.nodes) {
      const right = kept.get(`${node.row},${node.col + 1}`);
      const down = kept.get(`${node.row + 1},${node.col}`);
      for (const nb of [right, down]) {
        if (!nb) continue;
        const a = toVideo(node.x, node.y), b = toVideo(nb.x, nb.y);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    // Surviving nodes as small green dots.
    for (const node of d.prunedMesh.nodes) {
      const p = toVideo(node.x, node.y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, dotR * 0.6, 0, Math.PI * 2);
      ctx.fillStyle = '#0f0';
      ctx.fill();
    }
    // Nodes buildMesh found but pruneInconsistentNodes rejected — shown as
    // red X marks, so it's visible what got thrown out and why the surviving
    // mesh is sparser than raw detection alone would suggest.
    ctx.strokeStyle = '#f00';
    ctx.lineWidth = Math.max(1, d.rawScale * 1.5);
    for (const node of d.rawMesh.nodes) {
      if (kept.has(`${node.row},${node.col}`)) continue;
      const p = toVideo(node.x, node.y);
      const r = dotR * 0.7;
      ctx.beginPath(); ctx.moveTo(p.x - r, p.y - r); ctx.lineTo(p.x + r, p.y + r); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p.x + r, p.y - r); ctx.lineTo(p.x - r, p.y + r); ctx.stroke();
    }
  }

  if (toggleCells.checked && d.sampledGrid) {
    // Dot size/opacity encodes how many of the cell's 4 corners were
    // actually known (2, 3, or 4 — see sampleFromMesh's diagonal-pair
    // comment) — a rough per-cell confidence signal: a full, opaque dot
    // means all 4 corners were found, a small, faint one means the position
    // was estimated from just one diagonal pair.
    for (const row of d.sampledGrid.cells) {
      for (const cell of row) {
        if (!cell.valid) continue;
        const p = toVideo(cell.x, cell.y);
        const frac = Math.max(0, Math.min(1, (cell.cornerCount - 2) / 2)); // 2->0, 3->0.5, 4->1
        const r = dotR * (0.5 + 0.5 * frac);
        ctx.globalAlpha = 0.5 + 0.5 * frac;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = cell.bit ? '#000' : '#fff';
        ctx.fill();
        ctx.lineWidth = Math.max(1, r * 0.3);
        ctx.strokeStyle = '#888';
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }
  }
}
