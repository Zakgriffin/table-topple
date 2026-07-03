import { generateTorus, buildLookupTable } from './debruijn.ts';
import { detectGrid, sampleFullGrid, pickBestCandidate, toGrayscale, binarize, estimateRotationRad } from './decode.ts';
import type { Patch } from './decode.ts';

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
//   5. Keep whichever candidate's patches are most mutually consistent.

const CONFIDENCE_THRESHOLD = 0.5; // min fraction of agreeing adjacent patches to trust a frame
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
  grid: { px: number; py: number; pitchX: number; pitchY: number };
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
  const theta0 = estimateRotationRad(rawGray, rawSide, rawSide);

  const grids: { px: number; py: number; pitchX: number; pitchY: number }[] = [];
  const sampledGrids = [0, 1, 2, 3].map(k => {
    const theta = theta0 + k * (Math.PI / 2);
    alignedCtx.save();
    alignedCtx.translate(alignedW / 2, alignedH / 2);
    alignedCtx.rotate(-theta);
    alignedCtx.translate(-rawSide / 2, -rawSide / 2);
    alignedCtx.drawImage(rawCanvas, 0, 0);
    alignedCtx.restore();

    const alignedImg = alignedCtx.getImageData(0, 0, alignedW, alignedH).data;
    const alignedBin = binarize(toGrayscale(alignedImg, alignedW, alignedH));
    const grid = detectGrid(alignedBin, alignedW, alignedH);
    grids.push(grid);
    return sampleFullGrid(alignedBin, alignedW, alignedH, grid);
  });

  const best = pickBestCandidate(sampledGrids, ORDER, lookup, R, C);
  const theta = theta0 + best.candidateIndex * (Math.PI / 2);
  const grid = grids[best.candidateIndex];

  const matched = best.patches.filter(p => p.match !== null);
  const match = (best.consistency >= CONFIDENCE_THRESHOLD && matched.length > 0) ? matched[0].match : null;

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

  if (result) {
    const toVideo = (ax: number, ay: number) => alignedToVideo(ax, ay, result);
    const confident = result.consistency >= CONFIDENCE_THRESHOLD;

    // Blue lines for the estimated grid edges (both line families), so you
    // can visually check the detected grid against the real one on screen.
    const { px, py, pitchX, pitchY } = result.grid;
    ctx.strokeStyle = 'rgba(60,140,255,0.8)';
    ctx.lineWidth = Math.max(1, result.rawScale * 1.2);
    const numCellsX = Math.floor((result.alignedW - px) / pitchX);
    const numCellsY = Math.floor((result.alignedH - py) / pitchY);
    for (let k = 0; k <= numCellsX; k++) {
      const x = px + k * pitchX;
      const a = toVideo(x, 0), b = toVideo(x, result.alignedH);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    for (let k = 0; k <= numCellsY; k++) {
      const y = py + k * pitchY;
      const a = toVideo(0, y), b = toVideo(result.alignedW, y);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }

    // Each patch drawn as a discrete outlined quad (green if it decoded to a
    // valid position AND the frame is confident overall, orange if it found
    // a lookup hit but overall confidence is low, red if no hit), containing
    // its own black/white dots so patches are visually distinguishable.
    for (const patch of result.patches) {
      const x0 = px + patch.tileCol * ORDER * pitchX, x1 = x0 + ORDER * pitchX;
      const y0 = py + patch.tileRow * ORDER * pitchY, y1 = y0 + ORDER * pitchY;
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
          ctx.strokeStyle = patch.match ? '#0f0' : '#f00';
          ctx.stroke();
        }
      }
    }
  }

  status.textContent = result?.match
    ? `order ${ORDER}  torus ${R}x${C}\nrow ${result.match.row}  col ${result.match.col}\nconfidence ${(result.consistency * 100).toFixed(0)}%`
    : `order ${ORDER}  torus ${R}x${C}\nno lock — move closer / center the pattern`;
}
