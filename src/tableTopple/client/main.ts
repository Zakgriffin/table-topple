// Table Topple Client -- the board game, played through a phone held over the
// printed board.
//
// The fourth page, and the only genuinely new one in the four-page restructure.
// It renders the SAME world table-topple-server.html does, from a camera placed
// where this device's own reconstruction says the phone is. Point it at the
// printed De Bruijn pattern and the virtual board lands on the real one.
//
// ── THIS IS THE PAGE THAT PUTS A PHONE ON POSE ───────────────────────────
//
// For a long stretch there was no such page. `src/pose` was built, verified and
// pooled against the desktop lab; the phone relayed JPEGs to a laptop and got
// nothing back but a picture. Pose Viewer's client still has a `TODO(phone-on-
// pose)` where its own span belongs. This file is the first thing on a phone to
// call `runPose`.
//
//   dom.ts        every element, resolved once. the LEAF -- imports nothing
//   camera.ts     one stream at one resolution, and the frames it hands on
//   pose.ts       the WebGPU device, the one PoseContext, and the placement
//   overlay.ts    the world drawn over the feed, from that placement
//
// ── TWO LOOPS, ON PURPOSE ────────────────────────────────────────────────
//
// A DISPLAY loop on requestAnimationFrame, and a POSE loop that paces itself.
// They are separate because they are bound by different things: the display
// should run at the screen's rate so the game animates smoothly and the
// viewfinder never stutters, while a reconstruction takes tens of milliseconds
// and should start the next one as soon as the last resolves. Driving the
// overlay off the pose loop would render the world at reconstruction rate --
// visibly juddering, and slower the better the camera is. Driving pose off rAF
// would either skip frames or block the display.
//
// The consequence to be honest about: between fixes the camera placement is
// STALE by up to one reconstruction. That is the same delayed-measurement
// problem Pose Viewer's client documents at its own anchor site, and the same
// thing IMU fusion exists to fix. Nothing here pretends otherwise -- there is
// no IMU on this page yet, deliberately, because the A/B that judges it needs a
// working vision-only baseline to be judged against, and this is that baseline.
//
// ── NETWORKED, BUT ONE-DIRECTIONAL ────────────────────────────────────────
//
// This page is a PURE RECEIVER of table-topple-server.html's game state --
// see shared/sim.ts's own header on the simulate/render split this depends
// on. It never runs step(), never touches AI or physics, and has no opinion
// about who's winning; it only paints whatever the last `denizenState`/
// `attackFx` message said (see overlay.ts's receiveDenizenState and this
// file's onMessage below).
//
// POSE still never leaves this device -- that half of the original "no
// network" decision stands. What changed is game state, which now flows the
// other way: down from the desktop, never up from here.

import * as THREE from 'three';
import type { PoseResult } from '../../pose/pose.ts';
import { drawViewfinder, frameDims, grabGray, hasLiveFrame, startCamera, viewfinderBoxHeight } from './camera.ts';
import { chromeToggleBtn, statusEl } from './dom.ts';
import { fitBoardToPattern, receiveDenizenState, renderOverlay, setViewfinderBoxHeight, syncOverlayRendererSize, updateOverlayCamera } from './overlay.ts';
import { initPose, isPoseReady, recoverPose, toCameraPose } from './pose.ts';
import { spawnArrowFx, spawnBlastFx } from '../shared/combat.ts';
import { writeRoadSnapshot } from '../shared/roads.ts';
import { attachNetHost, connect } from '../shared/net.ts';

// This page never sends a game message (see the header above) -- only
// receives. denizenState/roadState are handed to their own mailboxes;
// attackFx is a one-shot, so it is applied the instant it arrives rather
// than held.
attachNetHost({
  onMessage(msg) {
    if (msg.type === 'denizenState') {
      receiveDenizenState(msg.denizens);
    } else if (msg.type === 'roadState') {
      writeRoadSnapshot(msg.roads);
    } else if (msg.kind === 'arrow') {
      spawnArrowFx(msg.origin.x, msg.origin.y, msg.origin.z, msg.target.x, msg.target.y, msg.target.z, msg.speed, msg.team);
    } else {
      spawnBlastFx(msg.origin.x, msg.origin.y, msg.target.x, msg.target.y, msg.range, msg.halfAngle);
    }
  },
});
connect('phone');

// ── The status line ──────────────────────────────────────────────────────
//
// The whole readout this page has. It exists because every failure here is
// invisible otherwise: a page with no WebGPU, a camera that never started, and
// a board the decode cannot find all look identical -- a live picture with no
// board drawn on it. Saying WHICH is the difference between a bug report and a
// shrug, and on a phone there is no console to check.
function setStatus(text: string): void {
  statusEl.textContent = text;
}

// One switch that takes the readout off the screen, leaving the game and the
// feed and nothing else -- see the page's own `body.chromeHidden` rule for what
// counts as chrome. Purely presentational: nothing downstream reads it, so
// hiding the status does not pause the simulation or the reconstruction.
chromeToggleBtn.addEventListener('click', () => {
  const hidden = document.body.classList.toggle('chromeHidden');
  chromeToggleBtn.classList.toggle('active', hidden);
  chromeToggleBtn.setAttribute('aria-pressed', String(hidden));
  // A focused button stays outlined over an otherwise-clean picture, and also
  // treats the next Enter/Space as another click.
  chromeToggleBtn.blur();
});

// ── What the display loop knows about pose ───────────────────────────────
//
// The last RESULT, not the last SUCCESS. A failed decode overwrites a good one
// with null, which blanks the overlay -- see overlay.ts on why a frozen board
// is the dishonest failure and a blank one is the honest one. Holding the last
// good pose instead would let the world sit convincingly still on a board the
// camera has entirely lost.
let poseFixes = 0;
let poseAttempts = 0;

// ── The display loop ─────────────────────────────────────────────────────

const clock = new THREE.Clock();

function displayLoop() {
  requestAnimationFrame(displayLoop);
  const { cw, ch } = drawViewfinder();
  const boxH = viewfinderBoxHeight(cw, ch);
  // Always the FULL viewport, on both axes -- see overlay.ts's own comment on
  // why extending only one axis (as an earlier version of this did) isn't
  // enough: a phone's own camera can be letterboxed on EITHER axis depending
  // on how its aspect compares to the screen's.
  syncOverlayRendererSize(window.innerWidth, window.innerHeight);
  setViewfinderBoxHeight(boxH);
  // Capped: a backgrounded tab stops rAF, and the first frame after refocusing
  // would otherwise carry the entire away-time as one step and teleport
  // everybody. The same cap the standalone page applies, for the same reason.
  renderOverlay(Math.min(clock.getDelta(), 0.1));
}

// ── The pose loop ────────────────────────────────────────────────────────
//
// Self-paced: start the next reconstruction as soon as the last one resolves.
// No rAF gating and no artificial delay, so the fix rate is the phone's true
// reconstruction rate and the profiler's span measures exactly that.
async function poseLoop() {
  while (isPoseReady()) {
    if (!hasLiveFrame()) { await nextFrame(); continue; }
    const gray = grabGray();
    if (!gray) { await nextFrame(); continue; }
    const { w, h } = frameDims();
    let pose: PoseResult | null = null;
    try {
      pose = await recoverPose(gray, w / h);
    } catch (err) {
      // A throw here is a real defect (a buffer-size mismatch, a lost device),
      // not a failed decode -- so it is reported rather than swallowed into the
      // same blank overlay a bad frame produces. The loop continues, because a
      // lost device that comes back should not need a reload.
      setStatus(`pose error: ${err instanceof Error ? err.message : String(err)}`);
      await nextFrame();
      continue;
    }
    poseAttempts++;
    if (pose?.ok) poseFixes++;
    updateOverlayCamera(pose ? toCameraPose(pose, w / h) : null);
    reportFixRate(pose);
  }
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

// The one number worth showing continuously. A fix RATE rather than the latest
// verdict, because a board at a grazing angle decodes intermittently and a
// single-frame readout flickers between two states without saying which is
// typical. `consistency` rides along on a good fix because it is what separates
// a confident decode from a lucky one.
function reportFixRate(pose: PoseResult | null): void {
  const pct = poseAttempts > 0 ? Math.round((poseFixes / poseAttempts) * 100) : 0;
  const detail = pose?.ok
    ? `consistency ${pose.consistency.toFixed(2)}`
    : 'no board in view';
  setStatus(`fixes ${pct}% of ${poseAttempts} — ${detail}`);
}

// ── Boot ─────────────────────────────────────────────────────────────────
//
// Strictly ordered, and each step reports its own failure: the camera has to be
// delivering frames before the pose context can be sized, because the context is
// built for the resolution that ACTUALLY arrived rather than the one that was
// requested (see camera.ts).
//
// The display loop starts regardless of whether pose ever works. A page that
// shows a live viewfinder and says why there is no board is diagnosable; a
// blank page is not.
async function boot() {
  fitBoardToPattern();

  try {
    setStatus('starting camera…');
    await startCamera();
  } catch (err) {
    setStatus(`camera error: ${err instanceof Error ? err.message : String(err)}`);
    return;
  }

  const { w, h } = frameDims();
  displayLoop();

  setStatus(`camera ${w}x${h} — starting WebGPU…`);
  const ready = await initPose(w, h);
  if (!ready) {
    setStatus(`camera ${w}x${h} — NO WEBGPU on this browser, so no pose and no board`);
    return;
  }

  setStatus(`camera ${w}x${h} — looking for the board…`);
  poseLoop();
}

boot();
