import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { renderer, scene } from '../scene/renderer.ts';
import { insideCam, viewerCam } from '../scene/viewerControls.ts';
import { gradientArrowCanvas, pipFrame, pipLabel } from './dom.ts';

export function renderViewport(cam: THREE.Camera, x: number, y: number, w: number, h: number) {
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(scene, cam);
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
  gradientArrowCanvas.width = innerWidth;
  gradientArrowCanvas.height = innerHeight;
  gradientArrowCanvas.style.width = innerWidth + 'px';
  gradientArrowCanvas.style.height = innerHeight + 'px';
}
addEventListener('resize', resize);
resize();

