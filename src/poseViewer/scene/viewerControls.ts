import * as THREE from 'three';
import { DEBUG_LAYER } from '../constants.ts';
import { globalState } from '../state.ts';
import { canvas } from '../ui/dom.ts';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// ── Controls: world-orbit (mode A) + free look-around (mode C) ─────────

export const viewerCam = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.05, 500);
viewerCam.position.set(6, 6, 14);
viewerCam.layers.enable(DEBUG_LAYER);
export const worldOrbit = new OrbitControls(viewerCam, canvas);
worldOrbit.target.set(0, 3, 0);

export const insideCam = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.02, 500);
insideCam.layers.enable(DEBUG_LAYER);
export let insideYaw = 0, insidePitch = 0;
export let dragging = false, lastPX = 0, lastPY = 0;

canvas.addEventListener('pointerdown', (e) => {
  if (globalState.mode !== 'inside') return;
  dragging = true; lastPX = e.clientX; lastPY = e.clientY;
});
addEventListener('pointerup', () => { dragging = false; });
addEventListener('pointermove', (e) => {
  if (!dragging || globalState.mode !== 'inside') return;
  const dx = e.clientX - lastPX, dy = e.clientY - lastPY;
  lastPX = e.clientX; lastPY = e.clientY;
  insideYaw -= dx * 0.004;
  insidePitch = THREE.MathUtils.clamp(insidePitch - dy * 0.004, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
});
canvas.addEventListener('wheel', (e) => {
  if (globalState.mode !== 'inside') return;
  e.preventDefault();
  insideCam.fov = THREE.MathUtils.clamp(insideCam.fov + e.deltaY * 0.02, 20, 110);
  insideCam.updateProjectionMatrix();
}, { passive: false });

