import * as THREE from 'three';

export const SPHERE_RADIUS = 2.5;
export const GRID_STEP = 1; // world units per pattern cell

// ── The one place the PHYSICAL board's size is written down ──────────────
//
// Everything upstream of here is deliberately scale-invariant: the pose
// pipeline recovers position in board units (GRID_STEP per cell) and never
// needs to know how big a cell actually is. That invariance is why this
// constant did not exist until IMU fusion needed it -- an accelerometer
// reports m/s^2, so SOMETHING has to relate metres to board units, and until
// now nothing in the codebase did.
//
// Measured from the printed board: 1/2 cm cells. Change this if the board is
// reprinted at another size; nothing else needs to move.
//
// Sanity check it against a live capture rather than trusting it blind: with
// this value the recovered `distance` of ~31.9 board units reads as ~16cm,
// which is the actual working distance. A wrong scale here does not break the
// pose pipeline at all -- it only mis-scales IMU-predicted translation, which
// is exactly the kind of error that looks like bad filter tuning.
const CELL_SIZE_METRES = 0.005;
export const BOARD_UNITS_PER_METRE = GRID_STEP / CELL_SIZE_METRES; // 200
export const VIS_HALF_EXTENT = 20; // cap on how many grid lines get a reference line / great circle drawn (perf + clutter, independent of the floor's true size)
export const CIRCLE_SEGMENTS = 96;
export const PATCH_RES = 48; // patch-mesh tessellation, shared by every camera's own patch geometry

// ── TRUE vs RECOVERED: one visual language, two knobs ────────────────────
//
// The convention, which every "we measured this" overlay follows:
//
//   TRUE       opaque, base size      only simulated cameras have one
//   RECOVERED  translucent, LARGER    every camera has one
//
// so a recovered marker reads as a shell AROUND the truth it is trying to hit,
// and the two are legible while nested. A physical camera has no ground truth at
// all, so it shows only the translucent outer form -- which is why translucency
// means "recovered" rather than "less important".
//
// ADDITIVE, NOT MULTIPLICATIVE -- a constant SHELL THICKNESS in world units,
// added to the true form's half-extent. This is the number to change.
//
// A ratio was tried first and is wrong for the job: at 1.25x the camera box
// gains 1.25^3 = 1.95x the VOLUME, which is what the eye actually judges, so a
// "quarter larger" box reads as roughly double. Worse, a ratio makes the visual
// gap depend on the size of the thing wrapped, so a 0.06 pole and a 0.4 box get
// shells differing by almost an order of magnitude and the pair stops reading as
// one convention. A fixed offset gives every recovered form the same skin.
//
// MIND THE UNITS AT THE CALL SITE. SphereGeometry takes a RADIUS, which is a
// half-extent, so a pole adds this once. BoxGeometry takes FULL SIDE LENGTHS, so
// a box adds it TWICE per dimension to push each face out by the same amount.
// Getting that wrong gives the box a half-thickness shell and is invisible
// except as the two pairs quietly disagreeing again.
export const RECOVERED_SHELL = 0.015;

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
