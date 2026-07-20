import * as THREE from 'three';

export const SPHERE_RADIUS = 2.5;
export const GRID_STEP = 1; // world units per pattern cell
export const VIS_HALF_EXTENT = 20; // cap on how many grid lines get a reference line / great circle drawn (perf + clutter, independent of the floor's true size)
export const CIRCLE_SEGMENTS = 96;
export const PATCH_RES = 48; // patch-mesh tessellation, shared by every camera's own patch geometry

// ── World-frame constants ────────────────────────────────────────────────

export const ROW_DIR = new THREE.Vector3(1, 0, 0); // world +X — direction shared by every "row" floor line
export const COL_DIR = new THREE.Vector3(0, 0, 1); // world +Z — direction shared by every "col" floor line
// Scratch, reused sequentially by updateGizmo (each call fully overwrites it
// before use, so sharing across cameras/frames is safe).
export const euler = new THREE.Euler(0, 0, 0, 'YXZ');

// Fixed identity, NEVER mutated -- the reference frame every ray-casting call
// inside the recovery pipeline is expressed in. It does NOT need to equal
// the camera's true orientation for any of that math to work -- see
// pre-Stage-A history (solveRecoveredCamQuat's own comment) for how the
// camera's actual world orientation gets recovered afterward, entirely from
// the pattern, with zero dependency on this being "correct". A simulated
// camera's own camQuat (ground truth) must never leak into the recovery
// math itself.
export const MATH_QUAT = new THREE.Quaternion();

// Debug layer: gizmoCam (whichever camera is rendering "what the real camera
// sees") never sees layer 1, so its capture is a clean shot of just the
// floor -- every debug/overlay object (grid lines, gizmo bodies, sphere
// shell, poles, frustum outline, patch mesh, camera helper) lives on it.
export const DEBUG_LAYER = 1;
