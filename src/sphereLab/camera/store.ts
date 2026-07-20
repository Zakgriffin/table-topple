import * as THREE from 'three';
import { Camera, PhysicalCamera, SimulatedCamera } from './model.ts';

export const cameras = new Map<string, Camera>();
export let activeCameraId = '';
// ES module bindings are read-only outside their own module -- everything
// that used to just write `activeCameraId = ...` inline (now spread across
// camera/lifecycle.ts and ui/cameraPanel.ts) goes through this instead.
export function setActiveCameraId(id: string) { activeCameraId = id; }
export function activeCamera(): Camera | undefined { return cameras.get(activeCameraId); }
export function isSimulated(camera: Camera): camera is SimulatedCamera { return camera.type === 'simulated'; }
export function isPhysical(camera: Camera): camera is PhysicalCamera { return camera.type === 'physical'; }

export let nextCameraSerial = 1;
// Same read-only-outside-module reason as setActiveCameraId above.
export function bumpCameraSerial() { nextCameraSerial++; }
// Assigned in creation order, keyed off nextCameraSerial (never reused, even
// across deletions -- reusing a color the moment a camera's slot frees up
// would risk two SIMULTANEOUSLY existing cameras sharing a color, which
// defeats the entire point). Falls back to a random, well-saturated HSL hue
// once the fixed palette runs out, rather than capping how many cameras can
// exist.
export const CAMERA_COLOR_PALETTE = [0xffcc44, 0x33dd55, 0xff5588, 0x55ccff, 0xcc88ff, 0xff8833, 0x33ffcc, 0xdd4444];
export function nextCameraColor(): THREE.Color {
  const idx = nextCameraSerial - 1;
  if (idx < CAMERA_COLOR_PALETTE.length) return new THREE.Color(CAMERA_COLOR_PALETTE[idx]);
  return new THREE.Color().setHSL(Math.random(), 0.65, 0.55);
}
