import * as THREE from 'three';
import { CIRCLE_SEGMENTS, GRID_STEP, VIS_HALF_EXTENT } from '../constants.ts';
import { HALF_C, HALF_R } from '../floorPattern.ts';

// ── Math helpers (pure, shared) ──────────────────────────────────────────

export function slerpUnit(a: THREE.Vector3, b: THREE.Vector3, t: number): THREE.Vector3 {
  const dot = THREE.MathUtils.clamp(a.dot(b), -1, 1);
  const omega = Math.acos(dot);
  if (omega < 1e-6) return a.clone();
  const s = Math.sin(omega);
  return a.clone().multiplyScalar(Math.sin((1 - t) * omega) / s).addScaledVector(b, Math.sin(t * omega) / s);
}

// Plane through the camera origin containing a line (a point on it + its
// direction) — normal via point-direction cross product, no far-point
// approximation needed since the direction is exact and constant.
export function greatCircleNormal(pointOnLine: THREE.Vector3, direction: THREE.Vector3, camPos: THREE.Vector3): THREE.Vector3 | null {
  const toPoint = pointOnLine.clone().sub(camPos);
  const n = toPoint.clone().cross(direction);
  if (n.lengthSq() < 1e-10) return null; // camera sits (nearly) on the line — degenerate
  return n.normalize();
}

export function cornerDir(u: number, v: number, quat: THREE.Quaternion, vFovRad: number, aspect: number): THREE.Vector3 {
  const halfV = vFovRad / 2;
  const yc = Math.tan(halfV) * v;
  const xc = Math.tan(halfV) * aspect * u;
  return new THREE.Vector3(xc, yc, -1).normalize().applyQuaternion(quat);
}

// A vote normal's residual against a candidate row/col pole pair, under the
// grid's FOUR-FOLD symmetry: exactly 0 when n lies on either family's great
// circle, and sin(4*psi) rather than a plain angle so all four quarter-turn-
// equivalent alignments score identically.
//
// Lived in pipeline/orientationLM.ts until the Levenberg-Marquardt orientation
// refinement it was written for was deleted -- it was the only thing left in
// that file, and it is pure geometry with two consumers, so it belongs here.
export function fourFoldResidual(n: THREE.Vector3, Drow: THREE.Vector3, Dcol: THREE.Vector3): number {
  const psi = Math.atan2(n.dot(Dcol), n.dot(Drow));
  return Math.sin(4 * psi);
}

export function angleBetweenDegV(a: THREE.Vector3, b: THREE.Vector3): number {
  return THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(Math.abs(a.dot(b)), -1, 1)));
}

// Central place for "what FOV should ray-casting assume" -- settings.
// horizFovDeg is HORIZONTAL (shared by both camera types, see
// CameraSettingsCommon's own comment on why); this function always returns
// VERTICAL (THREE.js's camera.fov convention), via the caller's current
// aspect ratio. updateGizmo sets a SimulatedCamera's actual gizmoCam.fov via
// this exact same formula, so reading it back here (rather than
// recomputing) would give the identical answer either way -- recomputing
// directly from settings just means this function works uniformly for both
// types without needing a per-type branch at all.
//
// Originally lived on pipeline/capture.ts (taking a full `Camera`), which
// made it (and everything downstream that called it: decodeGrid.ts,
// pipelineGPU/projectSamples.ts) transitively hazardous to import on a page
// with no #gl canvas -- capture.ts imports the scene/renderer.ts singleton.
// Relocated here (already dependency-free, home of cornerDir) and narrowed
// to just the two fields it actually needs; re-exported from capture.ts so
// every existing desktop import keeps working unchanged.
export function getAnalysisVFovRad(camera: { aspect: number; settings: { horizFovDeg: number } }): number {
  const hFovRad = THREE.MathUtils.degToRad(camera.settings.horizFovDeg);
  return 2 * Math.atan(Math.tan(hFovRad / 2) / camera.aspect);
}

export function writeCirclePoints(line: THREE.Line, normal: THREE.Vector3, radius: number) {
  const helper = Math.abs(normal.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const u = helper.clone().cross(normal).normalize();
  const v = normal.clone().cross(u);
  const pos = (line.geometry.attributes.position as THREE.BufferAttribute);
  for (let i = 0; i <= CIRCLE_SEGMENTS; i++) {
    const a = (i / CIRCLE_SEGMENTS) * Math.PI * 2;
    const ca = Math.cos(a) * radius, sa = Math.sin(a) * radius;
    pos.setXYZ(i, u.x * ca + v.x * sa, u.y * ca + v.y * sa, u.z * ca + v.z * sa);
  }
  pos.needsUpdate = true;
}

// Row/col great-circle family "k" values, shared by every camera's own
// circle-pool meshes -- purely derived from the (global, shared) pattern
// extent, not camera state. Mutated in place (never reassigned) so every
// import of the array binding keeps seeing the same live array; per-camera
// circle pools (camera/factory.ts) are sized off rowLineKs.length/
// colLineKs.length at CAMERA-creation time only, so a board-size change
// that shrinks these below what an already-existing camera's pool was
// built for just leaves that camera's extra pool entries showing whatever
// they last displayed (cosmetic only, see overlays/sphereOverlays.ts's
// updateFamily -- it only ever writes pool[0..ks.length-1]) -- not expected
// to matter in practice, since VIS_HALF_EXTENT caps both arrays' length the
// same way for any board size roughly >= 2*VIS_HALF_EXTENT/GRID_STEP cells,
// comfortably below the board-size slider's whole useful range.
export const rowLineKs: number[] = [];
export const colLineKs: number[] = [];
export function rebuildGridLineKs() {
  rowLineKs.length = 0;
  for (let k = -Math.min(VIS_HALF_EXTENT, HALF_R); k <= Math.min(VIS_HALF_EXTENT, HALF_R); k += GRID_STEP) rowLineKs.push(k);
  colLineKs.length = 0;
  for (let k = -Math.min(VIS_HALF_EXTENT, HALF_C); k <= Math.min(VIS_HALF_EXTENT, HALF_C); k += GRID_STEP) colLineKs.push(k);
}
rebuildGridLineKs();
