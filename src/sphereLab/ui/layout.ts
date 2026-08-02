import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { renderer, scene } from '../scene/renderer.ts';
import { insideCam, viewerCam } from '../scene/viewerControls.ts';
import { lsdSvgOverlay, pipFrame, pipLabel } from './dom.ts';

export function renderViewport(cam: THREE.Camera, x: number, y: number, w: number, h: number) {
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(scene, cam);
}

// Letterbox rect for the fixed-aspect gizmo camera within whatever shape the
// window currently is -- shared by every overlay that needs to map a field-
// space (pixel) coordinate to screen space for the Through-Cam view (hover
// arrows, LSD rectangles/composite lines, vote-family lines). Lives here
// (not in an overlays/ file) specifically so overlays/lsdOverlay.ts and
// overlays/hoverDebugOverlays.ts can both import it without one importing
// the other.
export function computeThroughRect(camera: Camera): { x: number; y: number; w: number; h: number } {
  const winAspect = innerWidth / innerHeight;
  let w = innerWidth, h = innerHeight, x = 0, y = 0;
  if (winAspect > camera.aspect) { w = innerHeight * camera.aspect; x = (innerWidth - w) / 2; }
  else { h = innerWidth / camera.aspect; y = (innerHeight - h) / 2; }
  return { x, y, w, h };
}

export function layoutPip(camera: Camera) {
  const w = Math.min(320, innerWidth * 0.28);
  const h = w / camera.aspect;
  const margin = 20;
  camera.pipRect = { x: innerWidth - w - margin, y: innerHeight - h - margin, w, h };
  pipFrame.style.left = camera.pipRect.x + 'px';
  pipFrame.style.top = camera.pipRect.y + 'px';
  pipFrame.style.width = w + 'px';
  pipFrame.style.height = h + 'px';
  pipLabel.style.left = camera.pipRect.x + 'px';
  pipLabel.style.top = (camera.pipRect.y - 16) + 'px';
}

export function resize() {
  renderer.setSize(innerWidth, innerHeight);
  viewerCam.aspect = innerWidth / innerHeight;
  viewerCam.updateProjectionMatrix();
  insideCam.aspect = innerWidth / innerHeight;
  insideCam.updateProjectionMatrix();
  const cam = activeCamera();
  if (cam) layoutPip(cam);
  // viewBox in raw pixel units so SVG user-space coordinates (rectangles,
  // composite lines, gradient/level-line arrows) match computeThroughRect's
  // screen-pixel output 1:1.
  lsdSvgOverlay.setAttribute('viewBox', `0 0 ${innerWidth} ${innerHeight}`);
}
addEventListener('resize', resize);
resize();

