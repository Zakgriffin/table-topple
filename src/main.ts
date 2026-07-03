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

interface GridDetection {
  px: number; py: number; // phase (px offset of first cell boundary)
  pitchX: number; pitchY: number;
}

// Finds the pitch (dominant period) of a 1D energy profile via autocorrelation.
function findPitch(energy: Float64Array, minLag: number, maxLag: number): number {
  let bestLag = minLag, bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let score = 0;
    for (let x = 0; x + lag < energy.length; x++) score += energy[x] * energy[x + lag];
    if (score > bestScore) { bestScore = score; bestLag = lag; }
  }
  return bestLag;
}

// Finds the phase in [0, pitch) whose boundary positions (phase, phase+pitch,
// phase+2*pitch, ...) best align with peaks in the energy profile.
function findPhase(energy: Float64Array, pitch: number): number {
  let bestPhase = 0, bestScore = -Infinity;
  for (let phase = 0; phase < pitch; phase++) {
    let score = 0;
    for (let x = phase; x < energy.length; x += pitch) score += energy[x];
    if (score > bestScore) { bestScore = score; bestPhase = phase; }
  }
  return bestPhase;
}

function detectGrid(bin: Uint8Array, w: number, h: number): GridDetection {
  const colEnergy = new Float64Array(w);
  for (let x = 1; x < w; x++) {
    let e = 0;
    for (let y = 0; y < h; y++) e += Math.abs(bin[y * w + x] - bin[y * w + x - 1]);
    colEnergy[x] = e;
  }
  const rowEnergy = new Float64Array(h);
  for (let y = 1; y < h; y++) {
    let e = 0;
    for (let x = 0; x < w; x++) e += Math.abs(bin[y * w + x] - bin[(y - 1) * w + x]);
    rowEnergy[y] = e;
  }

  const minLag = 4, maxLagX = Math.floor(w / 4), maxLagY = Math.floor(h / 4);
  const pitchX = findPitch(colEnergy, minLag, maxLagX);
  const pitchY = findPitch(rowEnergy, minLag, maxLagY);
  const px = findPhase(colEnergy, pitchX);
  const py = findPhase(rowEnergy, pitchY);
  return { px, py, pitchX, pitchY };
}

// Samples the ORDER x ORDER cells nearest the crop center and packs them into
// a window key, or returns null if not enough cells are visible.
function sampleWindow(bin: Uint8Array, w: number, h: number, grid: GridDetection): number | null {
  const { px, py, pitchX, pitchY } = grid;
  const numCellsX = Math.floor((w - px) / pitchX);
  const numCellsY = Math.floor((h - py) / pitchY);
  if (numCellsX < ORDER || numCellsY < ORDER) return null;

  const startX = Math.floor((numCellsX - ORDER) / 2);
  const startY = Math.floor((numCellsY - ORDER) / 2);

  let key = 0;
  for (let i = 0; i < ORDER; i++) {
    const cy = py + pitchY * (startY + i + 0.5);
    for (let j = 0; j < ORDER; j++) {
      const cx = px + pitchX * (startX + j + 0.5);
      // Average a small box around the cell center to reduce edge noise.
      const bx = Math.max(2, Math.floor(pitchX * 0.2));
      const by = Math.max(2, Math.floor(pitchY * 0.2));
      let sum = 0, count = 0;
      for (let dy = -by; dy <= by; dy++) {
        const yy = Math.round(cy + dy);
        if (yy < 0 || yy >= h) continue;
        for (let dx = -bx; dx <= bx; dx++) {
          const xx = Math.round(cx + dx);
          if (xx < 0 || xx >= w) continue;
          sum += bin[yy * w + xx];
          count++;
        }
      }
      const bit = count > 0 && sum / count > 0.5 ? 1 : 0;
      key = (key << 1) | bit;
    }
  }
  return key >>> 0;
}

function decodeFrame(): { row: number; col: number } | null {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;
  const cropSrc = Math.min(vw, vh) * 0.6;
  const sx = (vw - cropSrc) / 2, sy = (vh - cropSrc) / 2;
  cropCtx.drawImage(video, sx, sy, cropSrc, cropSrc, 0, 0, CROP, CROP);

  const img = cropCtx.getImageData(0, 0, CROP, CROP).data;
  const gray = new Float64Array(CROP * CROP);
  let mean = 0;
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const luma = 0.299 * img[p] + 0.587 * img[p + 1] + 0.114 * img[p + 2];
    gray[i] = luma;
    mean += luma;
  }
  mean /= gray.length;
  const bin = new Uint8Array(CROP * CROP);
  for (let i = 0; i < gray.length; i++) bin[i] = gray[i] < mean ? 1 : 0; // dark -> 1 (black cell)

  const grid = detectGrid(bin, CROP, CROP);
  const key = sampleWindow(bin, CROP, CROP, grid);
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
