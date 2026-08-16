import * as THREE from 'three';
import { camera, scene } from '../shared/scene.ts';
import { step } from '../shared/sim.ts';
import { BOARD_SIZE } from '../shared/constants.ts';
import { CELL_PITCH, board } from '../shared/floorPattern.ts';
import { arCanvas } from './dom.ts';

// The board game, drawn over the live camera feed.
//
// This is table-topple-client.html: the same world table-topple-server.html
// renders, on a transparent canvas laid exactly over the viewfinder, seen
// through a camera at the pose this device's own reconstruction recovered. When
// the decode is good, the virtual board lands on the printed De Bruijn pattern
// in the real feed underneath it.
//
// It replaces what used to be here -- a translucent blue rectangle and a red
// marker cube. Those were a pose-correctness readout and nothing more. The
// board game is the thing all of the pose work is FOR, so putting the actual
// board there makes the overlay show whether the pose is good AND what it is
// good for, in the same picture.
//
// ── What this module is, and is not ──────────────────────────────────────
//
// It is the SECOND host of this project's world, next to
// table-topple-server.html's own main.ts. The seam that makes two hosts
// possible is the split in shared/ itself: scene.ts is the world, server/view.ts
// is the standalone page's presentation, sim.ts is one frame of world with no
// drawing in it. This file is the presentation half for a host that owns no
// canvas of its own, orbits nothing, and is told where the camera is.
//
// So: no keyboard, no pointer, no orbit controls, no pointer lock. shared/ only
// wires those when a host asks it to (wireKeys, wireAim, ...), and this host
// never asks -- there is no mouse on a phone held over a table, and the page's
// own taps stay the page's.
//
// THIS FILE SPENT A PHASE AS DEAD CODE. It was written against the AR overlay
// that used to live on Pose Viewer's capture page, survived the split into two
// projects, and then sat imported by nothing, reaching for an `#arCanvas` no
// page served. Phase 6 is the page. Its history explains the shape: the
// adapters it was originally paired with converted Pose Viewer's pose and the
// desktop's poseSync, and neither came across, because this client computes its
// own pose locally and hands it here directly (see client/pose.ts).

const canvas = arCanvas;

// alpha + a zero clear alpha is the whole reason this can't reuse the game's
// own renderer from view.ts: that one is opaque, because a full-screen game
// has nothing to show through. Here the camera feed underneath is the point.
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

// Near plane pulled in from the game's own 0.1: the standalone page's camera is
// always at least controls.minDistance (3 units) from what it looks at, while
// this one is wherever the phone happens to be -- including close enough over
// the board that 0.1 would clip the near cells away.
camera.near = 0.05;

// ── Fitting the game's board onto the physical one ───────────────────────
//
// Both boards are already square, centered on the world origin, lying in the
// XZ plane with +Y up -- the two coordinate conventions agree, so this is a
// pure scale with nothing to rotate or offset.
//
// The DEFAULT is 1:1 and deliberately so: shared/floorPattern.ts crops the De
// Bruijn board to BOARD_CELLS, so one printed pattern cell IS one board-game
// cell and a soldier is exactly one cell tall.
//
// This USED TO READ POSE VIEWER'S BOARD, which meant a slider in the other
// project silently rescaled this world. It now reads Table Topple's own, so the
// scale is 1 unless THIS project changes its own numbers -- the ratio is kept
// rather than hard-coded because the two constants are independently editable
// and a mismatch should letterbox, not lie.
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
  scene.scale.setScalar((cells * CELL_PITCH) / BOARD_SIZE);
}
fitBoardToPattern();

// ── Pose ─────────────────────────────────────────────────────────────────

/** A camera pose to draw the board from. ONE source produces it -- this
 *  device's own reconstruction, via client/pose.ts's toCameraPose. There was
 *  briefly a second (a pose synced down from a desktop), and it is gone with
 *  the decision that pose never leaves the client that computed it. */
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

// Mirrors the viewfinder's own INTRINSIC (cw,ch) directly, not its rendered CSS
// box -- both canvases share the exact same CSS letterbox fit (see
// table-topple-client.html), so matching intrinsic dimensions is what keeps this
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
 * Advancing the world happens even with no fix, and only the DRAW is skipped.
 * That is deliberate: the simulation is the shared world, and pausing it
 * whenever the camera loses the board would make the game's state a function of
 * how steady someone's hand is. A lost decode should cost you the picture, not
 * the match.
 *
 * This page is CPU/GPU bound by continuous pose recovery, so the cost matters --
 * but `step` is the cheap half, and the renderer.clear() path below skips the
 * expensive one.
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
