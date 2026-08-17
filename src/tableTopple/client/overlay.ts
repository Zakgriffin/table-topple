import * as THREE from 'three';
import { camera, scene } from '../shared/scene.ts';
import { renderWorld, writeDenizenSnapshot } from '../shared/sim.ts';
import { updateAttackVisuals } from '../shared/combat.ts';
import type { DenizenStateEntry } from '../shared/protocol.ts';
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
  camPos: THREE.Vector3; recoveredCamQuat: THREE.Quaternion; fovDeg: number;
}

/** No fix means nothing is drawn at all, rather than the world being left at
 *  a stale pose. A frozen overlay silently claims to still know where the
 *  camera is; a blank one is the honest failure. */
let hasPose = false;

// ── Extending the frustum to the full viewport ────────────────────────────
//
// This canvas covers the phone's FULL viewport, on both axes -- deliberately
// bigger than #viewfinder's own on-screen box, which stays letterboxed to the
// camera feed's true aspect (see table-topple-client.html). A phone can be
// pillarboxed (bars left/right, a desktop webcam's usual shape) OR letterboxed
// (bars top/bottom -- what a phone's own portrait camera turned out to be, at
// 480x640 against a 390x669 viewport). Both need the same fix, just on
// different axes, so nothing here picks one.
//
// The naive version -- pin camera.fov to the pose's true vertical FOV and just
// widen aspect -- only works when the render height still EQUALS the
// viewfinder's own height (the pillarboxed case, extending width alone). Stretch
// height too and a fixed fov changes the vertical pixels-per-degree scale,
// which drifts the shared middle of the picture out of register with the real
// feed underneath -- a subtler version of the same misalignment the old
// "identical letterbox" comment warned about.
//
// The fix that stays correct on EITHER axis: hold the camera's FOCAL LENGTH
// (pixels per radian) fixed, not its FOV in degrees, and let both fov and
// aspect fall out of however large the render surface actually is. That is
// exactly what a physically bigger sensor at the same focal length would
// produce -- extending the frustum outward from the real feed's own footprint
// rather than rescaling it.
let trueVFovDeg = 0; // the pose's own vertical FOV, UNSCALED -- what the real camera actually sees
let renderH = 0; // the renderer's current buffer height, set by syncOverlayRendererSize
let extendScale = 1; // renderH / viewfinder's own on-screen height; >1 means this axis is being stretched

function applyExtendedFov() {
  if (trueVFovDeg <= 0) return;
  const trueVFovRad = THREE.MathUtils.degToRad(trueVFovDeg);
  const extendedVFovRad = 2 * Math.atan(Math.tan(trueVFovRad / 2) * extendScale);
  camera.fov = THREE.MathUtils.radToDeg(extendedVFovRad);
  camera.updateProjectionMatrix();
}

/** Only ever moves the CAMERA -- the board is bolted to the world origin,
 *  where the pose pipeline's own reconstruction puts it.
 *
 *  Stores the pose's TRUE vertical FOV rather than applying it directly --
 *  applyExtendedFov (called every render, see renderOverlay) is what actually
 *  sets camera.fov, so a window resize between pose fixes still gets a correct
 *  frustum without waiting on the next fix. */
export function updateOverlayCamera(pose: ARCameraPose | null) {
  hasPose = !!pose;
  if (!pose) return;
  camera.position.copy(pose.camPos);
  camera.quaternion.copy(pose.recoveredCamQuat);
  trueVFovDeg = pose.fovDeg;
}

/**
 * Sizes the renderer to the FULL VIEWPORT, in CSS pixels -- always
 * (window.innerWidth, window.innerHeight), never the viewfinder's own
 * letterboxed box. `true` (let three.js own the inline style too) because
 * there is no longer a shared CSS letterbox to defer to.
 *
 * Only touches the GL backing store when (renderW, renderH) genuinely
 * changed, since this is called every rAF tick.
 */
let sizedW = 0, sizedH = 0;
export function syncOverlayRendererSize(renderW: number, renderCh: number) {
  if (renderW === sizedW && renderCh === sizedH) return;
  sizedW = renderW; sizedH = renderCh; renderH = renderCh;
  if (renderW > 0 && renderCh > 0) {
    renderer.setSize(renderW, renderCh, true);
    camera.aspect = renderW / renderCh;
  }
}

/**
 * The viewfinder's own on-screen height (camera.ts's viewfinderBoxHeight),
 * called every rAF tick alongside syncOverlayRendererSize -- this is what
 * extendScale is measured against, and it can change (a resize, a camera
 * dimension change) independently of whether the render buffer itself did.
 */
export function setViewfinderBoxHeight(boxH: number) {
  extendScale = boxH > 0 && renderH > 0 ? renderH / boxH : 1;
}

/**
 * The latest `denizenState` message, held as a mailbox rather than applied
 * the instant it arrives -- see receiveDenizenState's own comment.
 */
let pendingSnapshot: readonly DenizenStateEntry[] | null = null;

/**
 * Hands this page a fresh `denizenState` snapshot -- client/main.ts's NetHost
 * calls this the instant a message arrives. Only STORES it; writing it into
 * `denizens` happens in renderOverlay below, on the display loop's own clock.
 * The alternative -- writing straight in here -- would mean a health bar's
 * camera-facing billboard only refreshing on a network tick instead of every
 * rAF frame the AR camera itself moves (see renderOverlay).
 */
export function receiveDenizenState(entries: readonly DenizenStateEntry[]) {
  pendingSnapshot = entries;
}

/**
 * One frame: apply whatever's arrived from the authoritative host, render
 * every denizen off it, then draw the scene.
 *
 * This page never simulates -- there is no step() here, deliberately (see
 * shared/sim.ts's own header on the split). Pending state is written in
 * unconditionally, but the actual paint (renderWorld) happens even on a
 * frame with nothing new, off whatever `denizens` already holds: the render
 * clock and the network clock are different rates, and a lost/late message
 * should cost staleness, not a frozen scene.
 *
 * updateAttackVisuals is called here for the same reason renderWorld is:
 * an arrow or a blast wedge spawned by this page's own spawnArrowFx/
 * spawnBlastFx (client/main.ts's attackFx handler) lives in combat.ts's
 * arrows/blasts arrays, and NOTHING else on this page ever advances or
 * removes them -- step()'s updateCombat, which does that on the desktop, is
 * never called here. Skipping this call is exactly how a blast's wedge
 * used to sit at full opacity forever on a phone while fading correctly on
 * the desktop.
 *
 * Only the DRAW is skipped with no pose fix, not the render pass above it --
 * the world's presentation state has to stay current even while the camera
 * doesn't know where it is, or the picture jumps the instant the fix returns.
 *
 * This page is CPU/GPU bound by continuous pose recovery, so the cost matters
 * -- but rendering the roster is the cheap half, and the renderer.clear()
 * path below skips the expensive one.
 *
 * `dt` comes from the caller rather than a clock of this module's own, because
 * the caller is the one that knows how long the frame took and already caps it
 * (a backgrounded tab stops rAF, and the first frame back would otherwise
 * carry the whole away-time in one step and teleport everybody).
 */
export function renderOverlay(dt: number) {
  if (pendingSnapshot) { writeDenizenSnapshot(pendingSnapshot); pendingSnapshot = null; }
  renderWorld(dt);
  updateAttackVisuals(dt);
  if (!hasPose) { renderer.clear(); return; }
  // Every frame, not just on a pose fix or a resize -- either one alone can
  // leave the frustum stale for the OTHER's next change (see applyExtendedFov's
  // own comment above).
  applyExtendedFov();
  renderer.render(scene, camera);
}
