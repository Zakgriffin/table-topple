import { type Camera } from '../../shared/camera/model.ts';
import { throughCamCanvas, throughCamCtx } from '../ui/dom.ts';

// ── Through-Cam's own 2D rendering path -- deliberately NOT the shared
// WebGL #gl canvas World/Inside-Sphere use ──────────────────────────────
//
// Through-Cam used to be a full-screen textured quad rendered into the same
// renderer.domElement every other mode shares (see scene/quadRenderers.ts's
// now-removed renderTrueContamOverlay/etc). That meant a screenshot of
// "Through-Cam" was actually a screenshot of the whole shared canvas --
// sized to the DESKTOP WINDOW, not the camera's own captured resolution --
// so it silently returned the wrong pixel dimensions regardless of what
// resolution the phone/camera was actually set to (see this session's
// chat). throughCamCanvas's own intrinsic width/height are set to
// camera.rtSize directly (resizeThroughCamCanvas below); CSS max-width/
// max-height (pose-viewer-server.html) letterboxes that for on-screen display, the
// same split pose-viewer-client.html's own captureCanvas already uses. A
// screenshot/download of THIS canvas therefore always returns the camera's
// true resolution, independent of window size.
//
// World/Inside-Sphere's small PIP preview box still goes through the old
// WebGL quad path (scene/quadRenderers.ts's renderPreviewViewport) --
// that's a small overlay INSIDE an already-3D-view-mode frame, not a mode
// of its own, so it legitimately belongs on the shared canvas.

// Reusable scratch canvas for alpha-blended overlay layers -- putImageData
// can't alpha-blend (it replaces destination pixels outright, alpha
// channel included), so an overlay buffer has to land on its own canvas
// first and get drawImage'd onto throughCamCanvas, which DOES respect
// alpha compositing (same as the WebGL quads' own `transparent: true`
// blending used to). Reused across every overlay/every frame rather than
// allocated per-layer; resized in lockstep with throughCamCanvas itself.
const scratch = document.createElement('canvas');
const scratchCtx = scratch.getContext('2d')!;

// RGBA Uint8Array -> ImageData, zero-copy (Uint8ClampedArray view over the
// SAME underlying buffer) -- these buffers (camera.distortedPreviewData
// etc, see camera/model.ts) are already exactly w*h*4 bytes, the same
// layout ImageData requires.
function bufferToImageData(data: Uint8Array, w: number, h: number): ImageData {
  // Cast: these buffers are always plain `new Uint8Array(...)` (see
  // camera/model.ts), never backed by a SharedArrayBuffer -- .buffer's
  // ArrayBufferLike type is just wider than ImageData's constructor wants.
  return new ImageData(new Uint8ClampedArray(data.buffer as ArrayBuffer, data.byteOffset, data.length), w, h);
}

// Guards the resize the same way client/camera.ts's own preview canvas
// does -- reassigning canvas.width/height clears the backing bitmap even
// when set to the same value.
export function resizeThroughCamCanvas(camera: Camera) {
  const { w, h } = camera.rtSize;
  if (throughCamCanvas.width === w && throughCamCanvas.height === h) return;
  throughCamCanvas.width = w; throughCamCanvas.height = h;
  scratch.width = w; scratch.height = h;
}

// Every one of these buffers is stored BOTTOM-UP (row 0 = the bottom of the
// image, Y increasing upward) -- the field-space Y-up convention the
// analysis pipeline uses throughout (see overlays/hoverDebugOverlays.ts's
// own screen<->field mapping: `rasterY = fieldH - 1 - fy` and
// `fieldRow = (1 - ny) * fieldH`), and exactly what the OLD WebGL quads
// compensated for via `flipY = false` on every one of these textures (see
// camera/factory.ts) -- flipY=false meant "don't re-flip, this data is
// already oriented for GL's own bottom-up upload convention." putImageData
// has no such option (always interprets row 0 as the TOP), so without an
// explicit compensating flip here the image renders upside down -- confirmed
// live this session. The scratch canvas + Y-flip transform on the drawImage
// that composites it onto throughCamCtx is that compensation; every layer
// (including the opaque base preview, not just the alpha-blended overlays)
// needs it, since putImageData can't be scaled/transformed directly.
function drawLayer(camera: Camera, data: Uint8Array) {
  const { w, h } = camera.rtSize;
  scratchCtx.putImageData(bufferToImageData(data, w, h), 0, 0);
  throughCamCtx.save();
  throughCamCtx.translate(0, h);
  throughCamCtx.scale(1, -1);
  throughCamCtx.drawImage(scratch, 0, 0);
  throughCamCtx.restore();
}

export function drawThroughCamPreview(camera: Camera) { drawLayer(camera, camera.distortedPreviewData); }
export function drawThroughCamTrueContam(camera: Camera) { drawLayer(camera, camera.trueContamData); }
export function drawThroughCamReconContam(camera: Camera) { drawLayer(camera, camera.reconContamData); }
export function drawThroughCamTopGradient(camera: Camera) { drawLayer(camera, camera.topGradientData); }
export function drawThroughCamLsdRawRegions(camera: Camera) { drawLayer(camera, camera.lsdRawRegionsData); }
export function drawThroughCamLsdRejected(camera: Camera) { drawLayer(camera, camera.lsdRejectedData); }
