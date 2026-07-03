import { generateTorus, buildLookupTable } from './debruijn.ts';
import { detectGrid, sampleWindow, binarizeRGBA } from './decode.ts';

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

// ── Grid decode pipeline (Stage 1: axis-aligned, no rotation) ──────────────────
//
// 1. Crop a square region from the center of the video frame, downscaled to a
//    fixed working resolution — keeps analysis cost constant regardless of
//    native camera resolution.
// 2. Binarize (global mean threshold — the pattern is pure black/white, so
//    this is robust as long as lighting is roughly even across the crop).
// 3. Find the cell pitch in each axis via autocorrelation of an "edge energy"
//    profile (sum of adjacent-pixel differences) — periodic peaks in that
//    profile land at cell boundaries, spaced by the pitch.
// 4. Find the phase (sub-pitch offset) that best aligns with those boundaries.
// 5. Sample the ORDER x ORDER grid of cells nearest the crop center, average
//    brightness per cell, threshold to a bit.
// 6. Pack into a window key (top-to-bottom, left-to-right, MSB-first — must
//    match src/debruijn.ts's windowKey order) and look it up.

const CROP = 200; // working resolution for the analysis crop, in px
const cropCanvas = document.createElement('canvas');
cropCanvas.width = CROP;
cropCanvas.height = CROP;
const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true })!;

interface DecodeResult {
  match: { row: number; col: number } | null;
  cells: { x: number; y: number; bit: number }[]; // crop-local (0..CROP) coords
  cropSx: number; cropSy: number; cropSrc: number; // maps crop-local -> video coords
}

function decodeFrame(): DecodeResult | null {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;
  const cropSrc = Math.min(vw, vh) * 0.6;
  const cropSx = (vw - cropSrc) / 2, cropSy = (vh - cropSrc) / 2;
  cropCtx.drawImage(video, cropSx, cropSy, cropSrc, cropSrc, 0, 0, CROP, CROP);

  const img = cropCtx.getImageData(0, 0, CROP, CROP).data;
  const bin = binarizeRGBA(img, CROP, CROP);

  const grid = detectGrid(bin, CROP, CROP);
  const sampled = sampleWindow(bin, CROP, CROP, grid, ORDER);
  if (sampled === null) return { match: null, cells: [], cropSx, cropSy, cropSrc };

  const packed = lookup[sampled.key];
  const match = packed === -1 ? null : { row: Math.floor(packed / C), col: packed % C };
  return { match, cells: sampled.cells, cropSx, cropSy, cropSrc };
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

  // Show the analyzed crop region for visual alignment feedback.
  const vw = video.videoWidth, vh = video.videoHeight;
  const cropSrc = Math.min(vw, vh) * 0.6;
  const sx = (vw - cropSrc) / 2, sy = (vh - cropSrc) / 2;
  ctx.strokeStyle = '#0f0';
  ctx.lineWidth = 2;
  ctx.strokeRect(sx, sy, cropSrc, cropSrc);

  const result = decodeFrame();

  // Overlay a circle on every sampled cell: filled with the bit this pipeline
  // detected (black/white), with a bright outline so it's visible regardless
  // of the cell's own color. Lets you visually cross-check detected bits
  // against what's actually on screen, cell by cell.
  if (result) {
    const scale = result.cropSrc / CROP;
    const radius = Math.max(3, scale * 3);
    for (const cell of result.cells) {
      const vx = result.cropSx + cell.x * scale;
      const vy = result.cropSy + cell.y * scale;
      ctx.beginPath();
      ctx.arc(vx, vy, radius, 0, Math.PI * 2);
      ctx.fillStyle = cell.bit ? '#000' : '#fff';
      ctx.fill();
      ctx.lineWidth = Math.max(1, radius * 0.3);
      ctx.strokeStyle = result.match ? '#0f0' : '#f00';
      ctx.stroke();
    }
  }

  status.textContent = result?.match
    ? `order ${ORDER}  torus ${R}x${C}\nrow ${result.match.row}  col ${result.match.col}`
    : `order ${ORDER}  torus ${R}x${C}\nno lock — move closer / center the pattern`;
}
