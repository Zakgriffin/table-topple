// ── The camera: one stream, one resolution, and the frames it hands on ───
//
// Everything hard-won about NEGOTIATING a stream is src/capture/stream.ts --
// the platform layer both clients share. This is the page's half, and it is
// far smaller than Pose Viewer's because this page has no resolution UI:
// config.capture names one resolution, boot asks for it once, and whatever the
// camera actually delivers is what the pose context is built for.
//
// ── NOTHING IS EVER MIRRORED, AND THAT IS THE ONE RULE HERE ──
//
// Pose Viewer's client bakes a horizontal flip into the front-camera preview so
// a selfie viewfinder reads the way a mirror does. This page must not, anywhere,
// for a reason that is not cosmetic: the pixels drawn on screen and the pixels
// handed to the pipeline are the SAME pixels, and a mirrored image decodes to a
// mirrored -- wrong-handed -- pose. An AR overlay has to land on what the camera
// actually sees, so a flip would put the virtual board on a reflection of the
// real one and every gameplay position would be laterally inverted.
//
// The practical consequence: this page is a rear-camera activity. `facing` is
// still a config field because a front camera is a legitimate way to check the
// plumbing without pointing anything at a board, but a `user` capture will
// decode a mirrored board and should not be trusted for a pose.

import { requestPlainStream, requestStreamAt } from '../../capture/stream.ts';
import { toGrayscaleF32 } from '../../pose/grayscale.ts';
import { config } from '../shared/config.ts';
import { video, viewfinder, viewfinderCtx } from './dom.ts';

let currentStream: MediaStream | null = null;

// The off-DOM canvas the pipeline's input is drawn through, at the camera's
// full delivered resolution. Kept separate from the visible viewfinder, which
// is downscaled to the screen -- drawing and compositing a full-resolution
// canvas at display rate is what crashes a real phone, while the analysis
// needs every pixel the camera gave us.
const frameCanvas = document.createElement('canvas');
const frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true })!;

// Allocated once, for the life of the page, and reused every frame. This page
// runs a reconstruction continuously, so a per-frame 1.2 MB allocation would be
// competing with the pipeline it is feeding -- see toGrayscaleF32's own comment.
let grayBuffer: Float32Array | null = null;

/**
 * The resolution the camera is ACTUALLY delivering, which is not necessarily
 * the one that was requested.
 *
 * Everything downstream sizes itself off this rather than off config.capture:
 * a real device can resolve a getUserMedia request successfully and still land
 * somewhere else, and some platforms report the axes swapped (see
 * capture/stream.ts). Believing the request instead of the video element is how
 * a PoseContext gets built for a shape the frames do not have -- which fails as
 * a length mismatch inside `runPose`, a long way from the cause.
 */
export function frameDims(): { w: number; h: number } {
  return { w: video.videoWidth, h: video.videoHeight };
}

export function hasLiveFrame(): boolean {
  return !!currentStream && video.videoWidth > 0 && video.videoHeight > 0;
}

async function adoptStream(stream: MediaStream) {
  if (currentStream && currentStream !== stream) currentStream.getTracks().forEach((t) => t.stop());
  currentStream = stream;
  video.srcObject = stream;
  await video.play();
}

/**
 * Opens the camera at the configured resolution, falling back to an
 * unconstrained stream if the device refuses.
 *
 * A refusal is NORMAL, not an error: `requestStreamAt` returns null rather than
 * throwing precisely because a camera declining an exact resolution is an
 * everyday answer. Falling back keeps the page usable -- the pose context is
 * built from what arrives, so an off-spec resolution costs comparability with
 * the sweep's numbers, not correctness.
 *
 * Resolves once the video element actually reports dimensions, so a caller can
 * read frameDims() immediately afterward and get real numbers.
 */
export async function startCamera(): Promise<void> {
  const { width, height, facing } = config.capture;
  const exact = await requestStreamAt(undefined, width, height);
  if (exact) await adoptStream(exact);
  else await adoptStream(await requestPlainStream(undefined, facing));
  await waitForDimensions();
}

// `video.play()` resolving does not guarantee videoWidth is populated yet --
// the first decoded frame can land a tick later, and reading dimensions in that
// window gives 0x0. Polling on rAF rather than listening for 'loadedmetadata'
// because that event may already have fired by the time this runs.
function waitForDimensions(): Promise<void> {
  return new Promise((resolve, reject) => {
    const deadline = performance.now() + 5000;
    const check = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) return resolve();
      if (performance.now() > deadline) return reject(new Error('camera delivered no frames within 5s'));
      requestAnimationFrame(check);
    };
    check();
  });
}

// The screen's own long edge in physical pixels. Rendering the viewfinder any
// larger than this buys zero visible detail and costs a full-resolution
// composite on every display frame.
function previewLongEdgeCap(): number {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return Math.max(window.innerWidth, window.innerHeight) * dpr;
}

/**
 * Draws the live feed into the visible canvas, downscaled to the screen.
 *
 * Returns the canvas's intrinsic size, which the AR renderer matches exactly --
 * both canvases carry the same CSS letterbox, so identical intrinsic dimensions
 * are what keep the overlay registered with the picture underneath it.
 */
export function drawViewfinder(): { cw: number; ch: number } {
  const { w: vw, h: vh } = frameDims();
  if (!vw || !vh) { viewfinder.width = 0; viewfinder.height = 0; return { cw: 0, ch: 0 }; }
  const scale = Math.min(1, previewLongEdgeCap() / Math.max(vw, vh));
  const cw = Math.max(1, Math.round(vw * scale)), ch = Math.max(1, Math.round(vh * scale));
  // Guards the resize -- reassigning canvas.width/height clears the backing
  // bitmap even when set to the same value, so skipping it when unchanged
  // avoids a pointless reallocation on every display frame.
  if (viewfinder.width !== cw || viewfinder.height !== ch) { viewfinder.width = cw; viewfinder.height = ch; }
  viewfinderCtx.drawImage(video, 0, 0, cw, ch);
  return { cw, ch };
}

/**
 * One full-resolution frame, as the f32 grayscale `runPose` takes.
 *
 * Returns a view into a REUSED buffer, so the caller must be done with it
 * before asking for the next frame. That is true of the only caller by
 * construction -- the pose loop awaits its reconstruction before starting
 * another -- and it is what keeps this page off the allocator.
 *
 * Returns null when the camera dimensions have changed out from under the
 * buffer, rather than resizing silently: the PoseContext is built for one
 * shape, so a resolution change is a page-level event, not something to paper
 * over one frame at a time.
 */
export function grabGray(): Float32Array | null {
  const { w, h } = frameDims();
  if (!w || !h) return null;
  if (grayBuffer && grayBuffer.length !== w * h) return null;
  if (frameCanvas.width !== w || frameCanvas.height !== h) { frameCanvas.width = w; frameCanvas.height = h; }
  frameCtx.drawImage(video, 0, 0, w, h);
  const rgba = frameCtx.getImageData(0, 0, w, h).data;
  grayBuffer ??= new Float32Array(w * h);
  return toGrayscaleF32(rgba, w, h, grayBuffer);
}
