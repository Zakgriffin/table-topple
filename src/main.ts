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
const { R, C, N } = debruijn;
console.log(`Torus ready: ${R}x${C} cells, order ${ORDER}, ${lookup.length} lookup entries.`);

// ── Camera ────────────────────────────────────────────────────────────────────

const canvas = document.getElementById('c') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const video = document.createElement('video');
video.setAttribute('playsinline', '');

const switchCamBtn = document.getElementById('switchCam') as HTMLButtonElement;
let currentStream: MediaStream | null = null;
let currentFacing = 'environment';

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

function decodeFrame(): { row: number; col: number } | null {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;
  const cropSrc = Math.min(vw, vh) * 0.6;
  const sx = (vw - cropSrc) / 2, sy = (vh - cropSrc) / 2;
  cropCtx.drawImage(video, sx, sy, cropSrc, cropSrc, 0, 0, CROP, CROP);

  const img = cropCtx.getImageData(0, 0, CROP, CROP).data;
  const bin = binarizeRGBA(img, CROP, CROP);

  const grid = detectGrid(bin, CROP, CROP);
  const key = sampleWindow(bin, CROP, CROP, grid, ORDER);
  if (key === null) return null;

  const packed = lookup[key];
  if (packed === -1) return null;
  return { row: Math.floor(packed / C), col: packed % C };
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
  status.textContent = result
    ? `order ${ORDER}  torus ${R}x${C}\nrow ${result.row}  col ${result.col}`
    : `order ${ORDER}  torus ${R}x${C}\nno lock — move closer / center the pattern`;
}
