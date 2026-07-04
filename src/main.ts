import { generateTorus, buildLookupTable } from './debruijn.ts';
import { pickBestCandidate, toGrayscale, binarize } from './decode.ts';
import type { SampledGrid, SampledCell, CandidateResult } from './decode.ts';
import { buildLineAccumulator, findLinePeaks } from './lines.ts';
import type { LineCandidate } from './lines.ts';
import { splitIntoTwoFamilies, vpIsFinite, vpToPoint } from './vp.ts';
import type { LineFamily, VanishingPoint as LineVP } from './vp.ts';
import { indexFamilyLines, buildLatticeCorrespondences } from './lattice.ts';
import type { IndexedLine } from './lattice.ts';
import { fitHomographyDLT, applyHomography } from './homography.ts';
import type { Mat3 } from './homography.ts';

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

// ── Position-tracking pipeline ──────────────────────────────────────────────
// Detect grid lines directly via a gradient-oriented Hough transform
// (src/lines.ts), split them into the two row/col families by their
// vanishing points (src/vp.ts), recover each line's true gap-tolerant
// integer index via a Mobius-model fit (src/lattice.ts), then treat every
// row-line x col-line crossing as a correspondence for a single DLT
// homography fit (src/homography.ts). Sampling every lattice cell through
// that homography and decoding via src/decode.ts's pickBestCandidate gives
// this app's actual reported position — this is the only pipeline left; an
// earlier corner/mesh/vanishing-point approach (retired) needed to chain
// local neighbor-to-neighbor links to cover the ~55% of lattice points its
// corner detector could find individually, and a single bad link could drift
// a whole connected region undetected. Recovering a full global row/col
// index directly, then fitting one global homography from all of them at
// once, has no such local-chaining step to drift in the first place.
//
// A full pass (Hough transform over the whole capture, VP split, index
// recovery, homography fit, decode) is too heavy to run every rendered
// frame, so it runs on a self-rescheduling timer (see scheduleNextPipelinePass)
// instead — the render loop just redraws the last completed pass's result
// every frame, including the live video underneath it.

const pipelineStatus = document.getElementById('pipelineStatus')!;
const analyzeBtn = document.getElementById('analyzeBtn') as HTMLButtonElement;
const toggleBinarized = document.getElementById('toggleBinarized') as HTMLInputElement;
const toggleHoughLines = document.getElementById('toggleHoughLines') as HTMLInputElement;
const toggleLineVPs = document.getElementById('toggleLineVPs') as HTMLInputElement;
const toggleLineHomography = document.getElementById('toggleLineHomography') as HTMLInputElement;
const togglePatches = document.getElementById('togglePatches') as HTMLInputElement;

// Live-tunable pipeline parameters — each backed by a <input type=range> in
// the "tuning" details panel (index.html). Reads the slider's CURRENT value
// on every call via .get(), so a change takes effect on the next scheduled
// pass or Analyze click with no extra plumbing.
function bindSlider(id: string): { get: () => number } {
  const input = document.getElementById(id) as HTMLInputElement;
  const valSpan = document.getElementById(id + 'Val');
  const sync = () => { if (valSpan) valSpan.textContent = input.value; };
  input.addEventListener('input', sync);
  return { get: () => parseFloat(input.value) };
}

// Level 1 — hough lines (src/lines.ts)
const tBlurRadius = bindSlider('tBlurRadius');
const tMinMag = bindSlider('tMinMag');
const tPeakThreshold = bindSlider('tPeakThreshold');
const tNmsTheta = bindSlider('tNmsTheta');
const tNmsRho = bindSlider('tNmsRho');
// Level 2 — VP split (src/vp.ts)
const tSplitInlierPx = bindSlider('tSplitInlierPx');
const tMaxCandidates = bindSlider('tMaxCandidates');
// Level 3 — index recovery (src/lattice.ts)
const tIndexInlierPx = bindSlider('tIndexInlierPx');
const tMaxGap = bindSlider('tMaxGap');
const tSpanCap = bindSlider('tSpanCap');
// Orchestration
const tAnalysisRes = bindSlider('tAnalysisRes');
const tConfidence = bindSlider('tConfidence');
const tRefreshDelay = bindSlider('tRefreshDelay');
const tMinFamily = bindSlider('tMinFamily');

// Level 1's Hough resolution is fixed rather than adaptive: at the point
// this is needed, no lines have been detected yet, so there's nothing real
// to measure a "local" pitch from — any adaptive estimate here could only
// ever be a GLOBAL proxy (an earlier version derotated a small patch near
// the image center and autocorrelated it), which is systematically wrong
// under perspective for lines anywhere else in the frame. A small fixed bin
// size is the safer default because the two failure modes aren't symmetric:
// too FINE a resolution just duplicates a peak into two close detections,
// which Level 3's gap-tolerant indexing absorbs as harmless noise (both
// round to the same integer index); too COARSE a resolution merges distinct
// real lines into one, an unrecoverable loss of information downstream.
const HOUGH_RHO_BIN_PX = 1.5;
const HOUGH_THETA_BINS = Math.round(360 / HOUGH_RHO_BIN_PX);

const analysisCanvas = document.createElement('canvas');
const analysisCtx = analysisCanvas.getContext('2d', { willReadFrequently: true })!;
const binCanvas = document.createElement('canvas');
const binCtx = binCanvas.getContext('2d')!;

interface PipelineResult {
  cropSx: number; cropSy: number; rawScale: number; rawW: number; rawH: number;
  bin: Uint8Array;
  peaks: LineCandidate[];
  familyA: LineFamily | null;
  familyB: LineFamily | null;
  unassigned: LineCandidate[];
  rowIndexed: IndexedLine[];
  colIndexed: IndexedLine[];
  H: Mat3 | null;
  rows: number; cols: number;
  sampledGrid: SampledGrid | null;
  decodeResult: CandidateResult | null;
}

let pipelineResult: PipelineResult | null = null;

function rawToVideo(rx: number, ry: number, d: PipelineResult): { x: number; y: number } {
  return { x: d.cropSx + rx * d.rawScale, y: d.cropSy + ry * d.rawScale };
}

// Samples one bit per lattice cell (its center, the midpoint of its 4
// corners) directly from H — see scripts/test-lines-decode.ts, where this
// was validated end-to-end against real perspective tilt.
function sampleFromHomography(bin: Uint8Array, w: number, h: number, H: Mat3, rowCount: number, colCount: number): SampledGrid {
  const cells: SampledCell[][] = [];
  for (let i = 0; i < rowCount; i++) {
    const rowCells: SampledCell[] = [];
    for (let j = 0; j < colCount; j++) {
      const p = applyHomography(H, i + 0.5, j + 0.5);
      if (!p) { rowCells.push({ x: NaN, y: NaN, bit: 0, valid: false, cornerCount: 0 }); continue; }
      const [px, py] = p;
      const xx = Math.round(px), yy = Math.round(py);
      if (xx < 0 || xx >= w || yy < 0 || yy >= h) { rowCells.push({ x: px, y: py, bit: 0, valid: false, cornerCount: 0 }); continue; }
      rowCells.push({ x: px, y: py, bit: bin[yy * w + xx], valid: true, cornerCount: 4 });
    }
    cells.push(rowCells);
  }
  return { rows: rowCount, cols: colCount, cells, originRow: 0, originCol: 0 };
}
function mirrorRowsGrid(sg: SampledGrid): SampledGrid {
  return { ...sg, cells: sg.cells.slice().reverse() };
}

function runPipeline() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;

  const cropW = vw * 0.95, cropH = vh * 0.95;
  const cropSx = (vw - cropW) / 2, cropSy = (vh - cropH) / 2;
  const aspect = cropW / cropH;
  const analysisBudget = tAnalysisRes.get() * tAnalysisRes.get();
  const rawH = Math.round(Math.sqrt(analysisBudget / aspect));
  const rawW = Math.round(rawH * aspect);
  const rawScale = cropW / rawW;

  analysisCanvas.width = rawW; analysisCanvas.height = rawH;
  analysisCtx.drawImage(video, cropSx, cropSy, cropW, cropH, 0, 0, rawW, rawH);
  const rawRgba = analysisCtx.getImageData(0, 0, rawW, rawH).data;
  const gray = toGrayscale(rawRgba, rawW, rawH);
  const bin = binarize(gray);

  const field = buildLineAccumulator(gray, rawW, rawH, HOUGH_THETA_BINS, HOUGH_RHO_BIN_PX, Math.round(tBlurRadius.get()), tMinMag.get());
  const peaks = findLinePeaks(field, tPeakThreshold.get(), Math.round(tNmsTheta.get()), Math.round(tNmsRho.get()));

  let familyA: LineFamily | null = null, familyB: LineFamily | null = null, unassigned: LineCandidate[] = [];
  let rowIndexed: IndexedLine[] = [], colIndexed: IndexedLine[] = [];
  let H: Mat3 | null = null;
  let rows = 0, cols = 0;
  let sampledGrid: SampledGrid | null = null;
  let decodeResult: CandidateResult | null = null;

  if (peaks.length >= 8) {
    try {
      const split = splitIntoTwoFamilies(peaks, rawW, rawH, tSplitInlierPx.get(), Math.round(tMaxCandidates.get()));
      familyA = split.familyA; familyB = split.familyB; unassigned = split.unassigned;
    } catch { /* fewer than 2 usable lines — leave families null */ }
  }

  const minFamilySize = Math.round(tMinFamily.get());
  if (familyA && familyB && familyA.lines.length >= minFamilySize && familyB.lines.length >= minFamilySize) {
    // indexFamilyLines resolves a real ambiguity found via live testing: from
    // line positions alone, "no gaps" and "a uniform pattern of missing
    // lines" are indistinguishable, and under real noise this caused
    // genuinely gap-free lines to occasionally be mis-indexed as 2x/3x
    // sparser, splitting real cells into phantom half-cells in the
    // rectified-grid overlay. It resolves this using each family's OWN
    // locally-measured real line spacing (see src/lattice.ts's
    // estimateLocalSpacing) — no external pitch estimate needed here.
    rowIndexed = indexFamilyLines(familyA, familyB.vp, rawW, rawH, tIndexInlierPx.get(), Math.round(tMaxGap.get()), Math.round(tSpanCap.get()));
    colIndexed = indexFamilyLines(familyB, familyA.vp, rawW, rawH, tIndexInlierPx.get(), Math.round(tMaxGap.get()), Math.round(tSpanCap.get()));
    const correspondences = buildLatticeCorrespondences(rowIndexed, colIndexed, rawW, rawH);
    H = fitHomographyDLT(correspondences);
    if (H) {
      rows = rowIndexed.length ? Math.max(...rowIndexed.map(r => r.index)) : 0;
      cols = colIndexed.length ? Math.max(...colIndexed.map(c => c.index)) : 0;
      if (rows >= ORDER && cols >= ORDER) {
        sampledGrid = sampleFromHomography(bin, rawW, rawH, H, rows, cols);
        // indexFamilyLines' sort direction per family is arbitrary, so a
        // single-axis mirror is possible alongside the 0/90/180/270 rotation
        // ambiguity pickBestCandidate already searches — see
        // scripts/test-lines-decode.ts's comment on why both candidates are
        // needed to cover all 8 dihedral symmetries.
        decodeResult = pickBestCandidate([sampledGrid, mirrorRowsGrid(sampledGrid)], ORDER, lookup, debruijn.torus, R, C);
      }
    }
  }

  pipelineResult = {
    cropSx, cropSy, rawScale, rawW, rawH, bin,
    peaks, familyA, familyB, unassigned, rowIndexed, colIndexed, H, rows, cols, sampledGrid, decodeResult,
  };

  const locked = !!decodeResult?.match && decodeResult.consistency >= tConfidence.get();
  pipelineStatus.textContent = [
    `lines: ${peaks.length} peaks, ${unassigned.length} unassigned`,
    `families: A=${familyA?.lines.length ?? 0} B=${familyB?.lines.length ?? 0}`,
    `indexed: rows 0..${rows} (${rowIndexed.length} lines), cols 0..${cols} (${colIndexed.length} lines)`,
    H ? `homography: fit ok, sampled ${sampledGrid?.rows}x${sampledGrid?.cols}` : 'homography: n/a',
    decodeResult ? `score ${(decodeResult.consistency * 100).toFixed(0)}%${locked ? ` -> row ${decodeResult.match!.row} col ${decodeResult.match!.col}` : ' (no lock)'}` : 'decode: n/a',
  ].join('\n');
}

// Runs one pipeline pass, painting "analyzing..." first since the pass
// itself blocks the main thread for a while (double rAF: the DOM update is
// guaranteed painted by the time the heavy synchronous work in the second
// callback runs — a single rAF doesn't guarantee the paint has happened
// yet). Calls `after` once done, whether or not anything was actually found.
function triggerAnalysis(after: () => void) {
  analyzeBtn.disabled = true;
  pipelineStatus.textContent = 'analyzing...';
  requestAnimationFrame(() => requestAnimationFrame(() => {
    try { runPipeline(); }
    finally { analyzeBtn.disabled = false; after(); }
  }));
}

// The pipeline result drives BOTH the debug overlays below and the app's
// actual reported position, so — unlike when this was purely a diagnostic
// view — it must keep running continuously, not just while a debug toggle
// happens to be checked. Reschedules itself only after the PREVIOUS pass
// finishes (not a fixed-cadence interval), since a pass can take a while and
// overlapping runs would just queue up jank.
function scheduleNextPipelinePass() {
  window.setTimeout(() => triggerAnalysis(scheduleNextPipelinePass), tRefreshDelay.get());
}

analyzeBtn.addEventListener('click', () => triggerAnalysis(() => {}));
scheduleNextPipelinePass();

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

  if (pipelineResult) drawDebugOverlays(pipelineResult);

  const decodeResult = pipelineResult?.decodeResult;
  const locked = !!decodeResult?.match && decodeResult.consistency >= tConfidence.get();
  status.textContent = locked
    ? `order ${ORDER}  torus ${R}x${C}\nrow ${decodeResult!.match!.row}  col ${decodeResult!.match!.col}\nconfidence ${(decodeResult!.consistency * 100).toFixed(0)}%`
    : `order ${ORDER}  torus ${R}x${C}\nno lock — move closer / center the pattern`;
}

// Draws whichever of the pipeline's diagnostic layers are toggled on, from
// the last completed pass (see runPipeline — this data is stale relative to
// the live video by however long ago that pass ran, unlike the live video
// feed itself which render() draws fresh every frame).
function drawDebugOverlays(d: PipelineResult) {
  const toVideo = (rx: number, ry: number) => rawToVideo(rx, ry, d);
  const dotR = Math.max(2, d.rawScale * 4);

  if (toggleBinarized.checked) {
    // Renders the same bit array the pipeline reads bits from, so you can
    // see exactly what it's working with — drawn as a plain image
    // (nearest-neighbor, no smoothing) rather than per-pixel canvas calls,
    // since that'd be tens of thousands of draw calls/frame.
    binCanvas.width = d.rawW; binCanvas.height = d.rawH;
    const imgData = binCtx.createImageData(d.rawW, d.rawH);
    for (let i = 0; i < d.bin.length; i++) {
      const v = d.bin[i] ? 0 : 255;
      imgData.data[i * 4] = v; imgData.data[i * 4 + 1] = v; imgData.data[i * 4 + 2] = v; imgData.data[i * 4 + 3] = 255;
    }
    binCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(binCanvas, d.cropSx, d.cropSy, d.rawW * d.rawScale, d.rawH * d.rawScale);
    ctx.imageSmoothingEnabled = true;
  }

  // Every detected Hough line (src/lines.ts), drawn full-length across the
  // frame (a line has no endpoints of its own, just a theta/rho) — colored
  // by which family it ended up in, so the split (src/vp.ts) is visually
  // checkable against the real grid: family A / family B / unassigned
  // (neither VP's inlier — noise, or a genuine line the split couldn't
  // place) each get a distinct color.
  if (toggleHoughLines.checked) {
    const cx = d.rawW / 2, cy = d.rawH / 2;
    const big = 2 * Math.max(d.rawW, d.rawH);
    const drawLine = (line: LineCandidate, color: string, width: number) => {
      const a = Math.cos(line.theta), b = Math.sin(line.theta);
      const px = cx + line.rho * a, py = cy + line.rho * b;
      const tx = -b, ty = a; // direction along the line (perpendicular to its normal)
      const p1 = toVideo(px - tx * big, py - ty * big);
      const p2 = toVideo(px + tx * big, py + ty * big);
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1, d.rawScale * width);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    };
    for (const line of d.unassigned) drawLine(line, 'rgba(150,150,150,0.4)', 0.75);
    for (const line of d.familyA?.lines ?? []) drawLine(line, 'rgba(0,255,255,0.7)', 1);
    for (const line of d.familyB?.lines ?? []) drawLine(line, 'rgba(255,0,255,0.7)', 1);
  }

  // The two families' vanishing points (src/vp.ts) — drawn as full lines
  // through the raw capture's center rather than segments toward the VP
  // itself, since it's very often off-screen (or, for a near-fronto-parallel
  // family, has no position at all — only a direction, the homogeneous w~0
  // case). A finite VP additionally gets a marker dot at its actual
  // position when that's within the drawable area.
  if (toggleLineVPs.checked && d.familyA && d.familyB) {
    const cx = d.rawW / 2, cy = d.rawH / 2;
    const big = 2 * Math.max(d.rawW, d.rawH);
    const drawVPLine = (vp: LineVP, color: string) => {
      let dx: number, dy: number;
      if (vpIsFinite(vp)) {
        const p = vpToPoint(vp);
        dx = p.x - cx; dy = p.y - cy;
        const n = Math.hypot(dx, dy) || 1;
        dx /= n; dy /= n;
      } else {
        const n = Math.hypot(vp.x, vp.y) || 1;
        dx = vp.x / n; dy = vp.y / n;
      }
      const p1 = toVideo(cx - dx * big, cy - dy * big);
      const p2 = toVideo(cx + dx * big, cy + dy * big);
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1, d.rawScale * 1.5);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      if (vpIsFinite(vp)) {
        const p = vpToPoint(vp);
        const pv = toVideo(p.x, p.y);
        ctx.beginPath(); ctx.arc(pv.x, pv.y, dotR * 1.2, 0, Math.PI * 2);
        ctx.fillStyle = color; ctx.fill();
      }
    };
    drawVPLine(d.familyA.vp, 'rgba(255,60,60,0.85)');
    drawVPLine(d.familyB.vp, 'rgba(255,60,255,0.85)');
  }

  // The fitted homography's implied grid (src/homography.ts) — every
  // integer row/col lattice line projected back into the image, so the
  // rectification can be visually checked against the real grid on screen:
  // if H is right, these lines should sit exactly on the real cell
  // boundaries even in regions with no detected Hough line at all (a gap
  // indexFamilyLines' Mobius fit bridged).
  if (toggleLineHomography.checked && d.H) {
    ctx.strokeStyle = 'rgba(255,200,0,0.7)';
    ctx.lineWidth = Math.max(1, d.rawScale);
    for (let i = 0; i <= d.rows; i++) {
      const p1 = applyHomography(d.H, i, 0), p2 = applyHomography(d.H, i, d.cols);
      if (!p1 || !p2) continue;
      const a = toVideo(p1[0], p1[1]), b = toVideo(p2[0], p2[1]);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    for (let j = 0; j <= d.cols; j++) {
      const p1 = applyHomography(d.H, 0, j), p2 = applyHomography(d.H, d.rows, j);
      if (!p1 || !p2) continue;
      const a = toVideo(p1[0], p1[1]), b = toVideo(p2[0], p2[1]);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
  }

  // Each decoded patch as a discrete outlined quad (green if it decoded to a
  // valid position AND the frame is confident overall, orange if it found a
  // lookup hit but overall confidence is low, red if no hit), containing its
  // own black/white bit dots (stroked green/red per-cell against the actual
  // known pattern, see Patch.correct) so patches are visually distinguishable
  // and individually verifiable.
  //
  // The quad corners are extrapolated half a cell beyond this patch's own 4
  // corner CELL positions using THIS patch's own local row/col step vectors
  // — not a global constant pitch (the old axis-aligned pipeline's
  // approach) — since sampled cell spacing varies smoothly across the frame
  // under real perspective (foreshortened farther from the camera). A single
  // cell's extrapolation is small enough that this local-linear
  // approximation of the true (homography-curved) boundary is accurate.
  if (togglePatches.checked && d.decodeResult) {
    const dr = d.decodeResult;
    const confident = dr.consistency >= tConfidence.get();
    const denom = Math.max(1, ORDER - 1);
    for (const patch of dr.patches) {
      const tl = patch.cells[0][0], tr = patch.cells[0][ORDER - 1];
      const bl = patch.cells[ORDER - 1][0], br = patch.cells[ORDER - 1][ORDER - 1];
      const rowStep = { x: (bl.x - tl.x) / denom, y: (bl.y - tl.y) / denom };
      const colStep = { x: (tr.x - tl.x) / denom, y: (tr.y - tl.y) / denom };
      const corner = (base: SampledCell, rowSign: number, colSign: number) => toVideo(
        base.x + rowSign * rowStep.x / 2 + colSign * colStep.x / 2,
        base.y + rowSign * rowStep.y / 2 + colSign * colStep.y / 2,
      );
      const corners = [corner(tl, -1, -1), corner(tr, -1, 1), corner(br, 1, 1), corner(bl, 1, -1)];
      ctx.strokeStyle = patch.match ? (confident ? '#0f0' : '#fa0') : '#f00';
      ctx.lineWidth = Math.max(1, d.rawScale);
      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      for (const c of corners.slice(1)) ctx.lineTo(c.x, c.y);
      ctx.closePath();
      ctx.stroke();

      const dotRadius = Math.max(2, Math.hypot(colStep.x, colStep.y) * d.rawScale * 0.25);
      for (let i = 0; i < ORDER; i++) {
        for (let j = 0; j < ORDER; j++) {
          const cell = patch.cells[i][j];
          const p = toVideo(cell.x, cell.y);
          ctx.beginPath();
          ctx.arc(p.x, p.y, dotRadius, 0, Math.PI * 2);
          ctx.fillStyle = cell.bit ? '#000' : '#fff';
          ctx.fill();
          ctx.lineWidth = Math.max(1, dotRadius * 0.3);
          ctx.strokeStyle = patch.correct ? (patch.correct[i][j] ? '#0f0' : '#f00') : (patch.match ? '#0f0' : '#f00');
          ctx.stroke();
        }
      }
    }
  }
}
