import * as THREE from 'three';
import { camera, scene } from './game/scene.ts';
import { step } from './game/sim.ts';
import { BOARD_SIZE } from './game/constants.ts';
import { board } from './poseViewer/floorPattern.ts';
import { GRID_STEP } from './poseViewer/constants.ts';

// The board game, drawn over the live camera feed.
//
// This is the AR overlay on mobile-capture.html: the same world game.html
// renders, on a transparent canvas laid exactly over the viewfinder, seen
// through a camera at the pose the reconstruction pipeline recovered. When the
// decode is good, the virtual board lands on the printed De Bruijn pattern in
// the real feed underneath it.
//
// It replaces what used to be here -- a translucent blue rectangle and a red
// marker cube. Those were a pose-correctness readout and nothing more. The
// board game is the thing all of the pose work is FOR, so putting the actual
// board there makes the overlay show whether the pose is good AND what it is
// good for, in the same picture.
//
// ── What this module is, and is not ──────────────────────────────────────
//
// It is the SECOND host of src/game, next to game.html's own main.ts, and it
// exists as a separate file for a merge reason as much as a design one: the
// board game is developed on its own branch, which touches only src/game.
// Every adaptation needed to render that world here lives in this file and in
// mobileCapture.ts, and nothing in src/game is edited on this side -- so the
// two branches never diverge inside src/game and merges between them stay
// clean. The seam that makes that possible is the split in src/game itself
// (scene.ts = world, view.ts = game.html's presentation, sim.ts = one frame of
// world with no drawing), which was cut once on the game branch and merged
// here rather than being reinvented on main.
//
// So: no keyboard, no pointer, no orbit controls, no pointer lock. src/game
// only wires those when a host asks it to (wireKeys, wireAim, ...), and this
// host never asks. The capture page's own taps stay the capture page's.

const canvas = document.getElementById('arCanvas') as HTMLCanvasElement;

// alpha + a zero clear alpha is the whole reason this can't reuse the game's
// own renderer from view.ts: that one is opaque, because a full-screen game
// has nothing to show through. Here the camera feed underneath is the point.
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

// Near plane pulled in from the game's own 0.1: game.html's camera is always
// at least controls.minDistance (3 units) from what it looks at, while this
// one is wherever the phone happens to be -- including close enough over the
// board that 0.1 would clip the near cells away.
camera.near = 0.05;

// ── Fitting the game's board onto the physical one ───────────────────────
//
// Both boards are already square, centered on the world origin, lying in the
// XZ plane with +Y up -- the two coordinate conventions agree, so this is a
// pure scale with nothing to rotate or offset.
//
// The DEFAULT is 1:1 and deliberately so: the De Bruijn board now crops to
// 144x144 (debruijn.ts's ORDER5_CANDIDATE) to match the game's own 144 cells
// (constants.ts's BOARD_CELLS), so one printed pattern cell IS one board-game
// cell and a soldier is exactly one cell tall. The scale below only stops
// being 1 if the board-size slider on the desktop moves the physical board off
// 144, in which case the game is scaled to keep filling it.
//
// Scaling the Scene itself rather than parenting the world under a Group: the
// game adds to `scene` at runtime (arrows in combat.ts, health bars, retiring
// denizens), so anything that isn't the scene root would need src/game to know
// about a container it never had -- exactly the kind of edit this file exists
// to avoid. The lights ride along harmlessly; a directional light's direction
// is scale-invariant and a hemisphere light has no position at all.
export function fitBoardToPattern() {
  // min, not the two axes separately: C and R are equal for every board the
  // crop can produce, but a non-square one would have to letterbox rather than
  // stretch -- a non-uniform scale would squash the denizens standing on it.
  const cells = Math.min(board.C, board.R);
  scene.scale.setScalar((cells * GRID_STEP) / BOARD_SIZE);
}
fitBoardToPattern();

// ── Pose ─────────────────────────────────────────────────────────────────

/** A camera pose to draw the board from. Two sources produce it -- this
 *  device's own reconstruction, and one synced down from the desktop -- and
 *  both normalize to this before arriving here (see mobileCapture.ts). */
export interface ARCameraPose {
  camPos: THREE.Vector3; recoveredCamQuat: THREE.Quaternion; aspect: number; fovDeg: number;
}

/** No fix means nothing is drawn at all, rather than the world being left at
 *  a stale pose. A frozen overlay silently claims to still know where the
 *  camera is; a blank one is the honest failure. */
let hasPose = false;

/** Only ever moves the CAMERA -- the board is bolted to the world origin,
 *  where the pose pipeline's own reconstruction puts it. */
export function updateOverlayCamera(pose: ARCameraPose | null) {
  hasPose = !!pose;
  if (!pose) return;
  camera.position.copy(pose.camPos);
  camera.quaternion.copy(pose.recoveredCamQuat);
  camera.fov = pose.fovDeg;
  camera.aspect = pose.aspect;
  camera.updateProjectionMatrix();
}

// Mirrors captureCanvas's own INTRINSIC (cw,ch) directly, not its rendered CSS
// box -- both canvases share the exact same CSS letterbox fit (see
// mobile-capture.html), so matching intrinsic dimensions is what keeps this
// canvas scaled/positioned identically to the viewfinder underneath. `false`
// (skip three.js's own inline-style sizing) since the stylesheet already owns
// display sizing for both canvases identically. Only touches the GL backing
// store when (cw,ch) genuinely changed, since this is called every rAF tick.
let sizedCw = 0, sizedCh = 0;
export function syncOverlayRendererSize(cw: number, ch: number) {
  if (cw === sizedCw && ch === sizedCh) return;
  sizedCw = cw; sizedCh = ch;
  if (cw > 0 && ch > 0) renderer.setSize(cw, ch, false);
}

/**
 * One frame: advance the world, then draw it.
 *
 * Called only while the overlay is switched on, so a hidden overlay costs
 * neither the simulation nor the draw -- this page is already CPU/GPU bound by
 * continuous pose recovery in video mode.
 *
 * `dt` comes from the caller rather than a clock of this module's own, because
 * the caller is the one that knows how long the frame took and already caps it
 * (a backgrounded tab stops rAF, and the first frame back would otherwise
 * carry the whole away-time in one step and teleport everybody).
 */
export function renderOverlay(dt: number) {
  step(dt);
  if (!hasPose) { renderer.clear(); return; }
  renderer.render(scene, camera);
}
