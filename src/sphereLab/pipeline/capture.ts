import * as THREE from 'three';
import { Camera, PhysicalCamera, SimulatedCamera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { toGrayscale } from '../../decode.ts';
import { renderer, scene } from '../scene/renderer.ts';
import { globalState } from '../state.ts';
import { layoutPip } from '../ui/layout.ts';
import { runAxesReconstruction } from './axesReconstruction.ts';
import { buildProjectedTexture } from './decodeGrid.ts';
import { addGaussianNoise, applyAntialiasFilter, downsampleBoxAverage, flipRowsF64, separableBoxBlur } from './distortion.ts';
import { updateDistortedPreview } from './preview.ts';

// ── Per-camera capture/analysis pipeline ─────────────────────────────────

// Central place for "what FOV should ray-casting assume" -- settings.
// horizFovDeg is HORIZONTAL (shared by both camera types now, see
// CameraSettingsCommon's own comment on why); this function always returns
// VERTICAL (THREE.js's camera.fov convention), via the camera's own current
// aspect ratio. updateGizmo sets a SimulatedCamera's actual gizmoCam.fov via
// this exact same formula, so reading it back here (rather than
// recomputing) would give the identical answer either way -- recomputing
// directly from settings just means this function works uniformly for both
// types without needing a per-type branch at all.
export function getAnalysisVFovRad(camera: Camera): number {
  const hFovRad = THREE.MathUtils.degToRad(camera.settings.horizFovDeg);
  return 2 * Math.atan(Math.tan(hFovRad / 2) / camera.aspect);
}

export function markCaptureDirty(camera: Camera) {
  camera.captureDirty = true;
}

// Called once at camera creation and again whenever the viewportW/H/
// captureSupersample sliders change -- or, with an explicit override,
// whenever a real capture arrives at a different resolution than whatever's
// currently allocated (see ingestRealCapture).
export function resizeCaptureBuffers(camera: Camera, explicitSize?: { w: number; h: number }) {
  camera.captureDirty = true;
  camera.rtSize = explicitSize ?? { w: Math.round(camera.settings.viewportW), h: Math.round(camera.settings.viewportH) };
  camera.aspect = camera.rtSize.w / camera.rtSize.h;
  const { w, h } = camera.rtSize;

  if (isSimulated(camera)) {
    camera.captureRTSize = { w: w * camera.settings.captureSupersample, h: h * camera.settings.captureSupersample };
    camera.camRT.setSize(camera.captureRTSize.w, camera.captureRTSize.h);
    camera.gizmoCam.aspect = camera.aspect;
    camera.gizmoCam.updateProjectionMatrix();
  }

  camera.distortedPreviewData = new Uint8Array(w * h * 4);
  camera.distortedPreviewTex.image = { data: camera.distortedPreviewData, width: w, height: h };
  // WebGL2 typically allocates a texture's GPU storage immutably on first
  // upload -- dispose() forces three.js to drop the old GL texture object so
  // the next upload allocates fresh storage at the new size.
  camera.distortedPreviewTex.dispose();
  camera.distortedPreviewTex.needsUpdate = true;

  camera.projectedPreviewData = new Uint8Array(w * h * 4);
  camera.projectedPreviewTex.image = { data: camera.projectedPreviewData, width: w, height: h };
  camera.projectedPreviewTex.dispose();
  camera.projectedPreviewTex.needsUpdate = true;

  camera.trueContamData = new Uint8Array(w * h * 4);
  camera.trueContamTex.image = { data: camera.trueContamData, width: w, height: h };
  camera.trueContamTex.dispose();
  camera.trueContamTex.needsUpdate = true;

  camera.reconContamData = new Uint8Array(w * h * 4);
  camera.reconContamTex.image = { data: camera.reconContamData, width: w, height: h };
  camera.reconContamTex.dispose();
  camera.reconContamTex.needsUpdate = true;

  if (camera === activeCamera()) layoutPip(camera);
}

// Renders gizmoCam's view into camRT -- pulled out into its own function so
// the real analysis path (captureDistortedGrayscale) can always force a
// truly fresh capture regardless of the passive preview's dirty/throttle
// gating in animate().
export function renderCamRT(camera: SimulatedCamera) {
  const dpr = renderer.getPixelRatio();
  const prevRT = renderer.getRenderTarget();
  renderer.setRenderTarget(camera.camRT);
  renderer.setViewport(0, 0, camera.captureRTSize.w / dpr, camera.captureRTSize.h / dpr);
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.render(scene, camera.gizmoCam);
  renderer.setRenderTarget(prevRT);
}

// Render+blur happen at captureSupersample x rtSize, THEN get box-downsampled
// to rtSize -- see pre-Stage-A history for why (physical lens blur acts on a
// near-continuous image; only the sensor's final discretization should
// introduce the pixel grid). Returned in GL's native bottom-up row order.
export function captureDistortedGrayscale(camera: SimulatedCamera): { gray: Float64Array; w: number; h: number } {
  renderCamRT(camera);
  const { w: cw, h: ch } = camera.captureRTSize;
  const raw = new Uint8Array(cw * ch * 4);
  renderer.readRenderTargetPixels(camera.camRT, 0, 0, cw, ch, raw);
  const hiResGray = toGrayscale(raw, cw, ch);
  const antialiased = applyAntialiasFilter(hiResGray, cw, ch, camera.settings.captureSupersample);
  const hiResBlurred = separableBoxBlur(antialiased, cw, ch, Math.round(camera.settings.simBlur * camera.settings.captureSupersample));
  const gray = downsampleBoxAverage(hiResBlurred, cw, ch, camera.settings.captureSupersample, camera.rtSize.w, camera.rtSize.h);
  addGaussianNoise(gray, camera.settings.simNoise);
  return { gray, w: camera.rtSize.w, h: camera.rtSize.h };
}

// Decodes an incoming data URL, resamples it to the current analysis
// resolution, converts to grayscale, and flips it to bottom-up.
export async function ingestRealCapture(camera: PhysicalCamera, dataUrl: string): Promise<void> {
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('failed to decode incoming capture image'));
    img.src = dataUrl;
  });
  const w = img.naturalWidth, h = img.naturalHeight;
  if (w !== camera.rtSize.w || h !== camera.rtSize.h) resizeCaptureBuffers(camera, { w, h });

  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = w; tmpCanvas.height = h;
  const tctx = tmpCanvas.getContext('2d')!;
  tctx.drawImage(img, 0, 0);
  const topDown = tctx.getImageData(0, 0, w, h).data;
  const grayTopDown = toGrayscale(topDown, w, h);
  camera.lastRealCaptureGray = flipRowsF64(grayTopDown, w, h);
  camera.lastRealCaptureW = w; camera.lastRealCaptureH = h;

  updateDistortedPreview(camera);
  if (globalState.mode === 'projected' && camera === activeCamera()) buildProjectedTexture(camera);
  runAxesReconstruction(camera);
}
