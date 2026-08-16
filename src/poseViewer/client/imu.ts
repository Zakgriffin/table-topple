import * as THREE from 'three';
import { config } from '../shared/config.ts';
import { BOARD_UNITS_PER_METRE } from '../shared/constants.ts';
import { nowMs } from '../../clock.ts';
import {
  type MotionPermission, type MotionSample,
  isMotionListening, motionRateHz, startMotion, stopMotion,
} from '../../capture/motion.ts';
import { ImuTracker, defaultImuTrackerConfig } from '../../capture/imuTracker.ts';
import {
  axisDiversity, defaultDeviceToCamera, scoreConventions, snapToNearestOctahedral, triadSolve,
  type RotationPair,
} from '../../capture/imuFrames.ts';
import { imuCheckbox, imuCorrectionCheckbox, imuReadoutEl } from './dom.ts';

// ── The IMU: recording it, judging it, and correcting with it ────────────
//
// Split out of the page's one file, and it is the largest coherent piece of it:
// the recording, the fusion toggle, and the device->camera calibration are one
// subject with one lifetime, and nothing else on the page needs their internals.
//
// ── WHY THIS TAKES A HOST INSTEAD OF IMPORTING ONE ──
//
// Three things here belong to the page rather than to the IMU: whether pose is
// being computed locally, when the desktop last synced settings, and the
// websocket a batch goes out on. Importing them would make this module and
// main.ts mutually dependent -- and a cycle here is not a style problem, it is
// the temporal-dead-zone fault this page already hit once (see main.ts). So the
// page REGISTERS them, and the arrow only ever points one way.

export interface ImuHost {
  /** Whether the page is computing pose locally -- gates the sync warning. */
  computingOnDevice(): boolean;
  /** Which camera is live. The device->camera default depends on it. */
  facing(): 'user' | 'environment';
  /** When the desktop last synced settings down, or null if it never has. */
  settingsSyncedAt(): number | null;
  /** Ships one batch. Returns false when there is no open socket to ship on. */
  sendBatch(payload: unknown): boolean;
}

// Null until main.ts attaches. Nothing here runs before that except module
// initialization, which touches no host field.
let host: ImuHost | null = null;

export function attachImuHost(h: ImuHost): void {
  host = h;
}

// ── IMU capture (phase A of the motion-blur/IMU plan) ────────────────────
//
// RECORDING ONLY. Nothing here feeds the pose pipeline, by design: the plan's
// phase A deliverable is a REPLAYABLE DATASET plus the one plot that says
// whether IMU prediction tracks measured pose at all, and a filter tuned live
// against a moving phone is a filter tuned against noise. It is also the
// pattern this project already learned the hard way -- verifyLsdChain got
// built before the ports it judges.
//
// WHY THE ACCELEROMETER IS THE POINT AND THE GYRO IS INFRASTRUCTURE:
// a hand-held phone rotates about a wrist/elbow, not its own optical centre,
// so a rotation w drags the camera sideways with lever arm r (v = w*r). The
// two image-motion terms are f*w for rotation and f*v/Z for translation, so
// their ratio is just
//
//     translation / rotation  =  r / Z
//
// -- translation dominates whenever the camera is nearer the surface than it
// is to the pivot, which at this project's working distance it always is.
// But the accelerometer measures a_true + g, and subtracting 9.81 m/s^2
// requires attitude: a 1 degree attitude error leaks 0.17 m/s^2, against
// ~0.016 m/s^2 of actual sensor noise at this bandwidth. So gravity leakage
// beats sensor noise by an order of magnitude, attitude accuracy is the
// binding constraint on accelerometer-based prediction, and attitude DURING
// motion needs the gyro. Hence: both are recorded, the accelerometer is what
// the result is judged on.
//
// AXES ARE NOT THE CAMERA'S AXES. DeviceMotion reports in the device frame
// (x right, y up, z out of the screen, with the phone held upright in
// portrait) -- the REAR camera looks along -z of that frame, and
// screen.orientation rotates the whole thing again. No rotation into the
// optical frame is applied here on purpose: raw is what a recording should
// hold, and the device->camera transform is a phase B/C calibration with a
// sign error waiting in it. `screenAngle` is captured per-batch so a replay
// can reconstruct the frame it was actually recorded in.
//
// Units, per the DeviceOrientation spec: rotationRate in DEGREES/SECOND,
// both acceleration fields in m/s^2.
const IMU_SEND_INTERVAL_MS = 100;

// A recorded sample is capture/motion.ts's `MotionSample` -- the device frame,
// the units, and the iOS interval quirk are all documented there, because they
// are platform facts rather than facts about this page.
export type ImuSample = MotionSample;

// THE AUTHORITATIVE RECORDING LIVES HERE, ON THE PHONE. This was briefly the
// other way round (desktop-side only, with just the newest sample kept here)
// and that was reversed on 2026-08-04 by an explicit architectural decision:
// the shipping product runs entirely on the device, so every timestamp in a
// dataset must come from ONE clock. The desktop relay measured -38ms of
// cross-device skew -- a third of the whole prediction horizon -- and a
// recording split across two clocks would have that baked in irrecoverably.
//
// The desktop still receives the same batches (devBridge/client.ts appends
// them for live inspection) but is no longer where a session is dumped from;
// pull it with `cli.js eval --phone "dumpRecording()"` instead. That is not
// the desktop being "in the loop" -- the data was stamped here, and reading
// it out afterwards cannot introduce a sync error.
const IMU_RING_CAPACITY = 3600; // 60s at 60Hz
export let imuRing: ImuSample[] = [];
export let imuLastSample: ImuSample | null = null;
let imuUnsent: ImuSample[] = [];
let imuPermission: MotionPermission = 'unknown';
let imuTotalSamples = 0;

function onMotionSample(s: MotionSample) {
  // Fed unconditionally, not only when the toggle is on: the tracker needs
  // continuous integration to have anything to say the moment it IS turned on,
  // and pushSample is a handful of flops. Whether its output is USED is the
  // toggle's business, further down.
  imuTracker.pushSample({
    t: s.t,
    rotationRate: { alpha: s.rrAlpha, beta: s.rrBeta, gamma: s.rrGamma },
    acceleration: { x: s.ax, y: s.ay, z: s.az },
  });

  // Live "up" in the device frame, for the direct calibration readout.
  const agLen = Math.hypot(s.agx, s.agy, s.agz);
  if (agLen > 1e-6) upDevice = new THREE.Vector3(s.agx, s.agy, s.agz).divideScalar(agLen);

  imuLastSample = s;
  imuRing.push(s);
  if (imuRing.length > IMU_RING_CAPACITY) imuRing.splice(0, imuRing.length - IMU_RING_CAPACITY);
  imuUnsent.push(s);
  imuTotalSamples++;
}

// iOS (13+) gates devicemotion behind an explicit permission prompt that MUST
// be requested from a user gesture -- hence this hanging off the checkbox's
// change event rather than running at load. Android/desktop have no
// requestPermission at all, where feature-detecting it away and just
// subscribing is the correct behaviour, not a fallback.
export async function startImu(): Promise<void> {
  if (isMotionListening()) return;
  imuPermission = await startMotion(onMotionSample);
  updateImuReadout();
}

export function stopImu() {
  if (!isMotionListening()) return;
  stopMotion();
  imuUnsent = [];
  updateImuReadout();
}

// The de-risking readout: whether the sensor produces anything at all, at
// what rate, and with what magnitudes -- checkable on the phone itself
// before any of this is trusted downstream. |a| is the gravity-removed
// magnitude (should sit near 0 at rest) and |ag| the raw one (should sit
// near 9.81 at rest); the pair disagreeing at rest is the fastest way to see
// that the OS's own gravity estimate is off, which is the error that
// dominates everything else in the plan.
export function updateImuReadout() {
  // Shown even with the IMU toggle off, and BEFORE every other state below:
  // computing on-device without a settingsSync produces confident garbage
  // (see settingsSyncedAt), so this outranks anything else this readout has
  // to say. Deliberately phrased as an instruction, not a status.
  if (host?.computingOnDevice() && host.settingsSyncedAt() === null) {
    imuReadoutEl.textContent = '⚠ NO SETTINGS SYNC — open pose-viewer-server.html on the desktop.\n'
      + 'Pose results are INVALID until it does (wrong board size).';
    return;
  }
  if (!imuCheckbox.checked) { imuReadoutEl.textContent = ''; return; }
  if (imuPermission === 'unsupported') { imuReadoutEl.textContent = 'IMU: not supported on this browser'; return; }
  if (imuPermission === 'denied') { imuReadoutEl.textContent = 'IMU: permission denied — re-tap the checkbox'; return; }
  if (!isMotionListening()) { imuReadoutEl.textContent = 'IMU: starting…'; return; }
  const last = imuLastSample;
  if (!last) { imuReadoutEl.textContent = 'IMU: on, no samples yet'; return; }
  const aMag = Math.hypot(last.ax, last.ay, last.az);
  const agMag = Math.hypot(last.agx, last.agy, last.agz);
  const rrMag = Math.hypot(last.rrAlpha, last.rrBeta, last.rrGamma);
  const rateHz = motionRateHz();
  const rate = rateHz !== null ? `${rateHz.toFixed(0)}Hz` : '—';
  const lines = [
    `IMU ${rate} (self-reported ${last.interval ? (1000 / last.interval).toFixed(0) + 'Hz' : '—'})  n=${imuTotalSamples}`,
    `|a| ${aMag.toFixed(2)}  |a+g| ${agMag.toFixed(2)} m/s²  |w| ${rrMag.toFixed(1)} °/s`,
  ];
  // The direct-calibration readout. Both rows are the SAME physical
  // direction (up) in two frames, so a correct transform makes them match --
  // and with the phone held flat above a horizontal board, both should read
  // roughly (0.00, 0.00, 1.00). Any disagreement IS the transform, visible
  // by eye without solving anything.
  //
  // |a| is shown beside them because upDevice is only trustworthy while
  // stationary: real acceleration adds to gravity, so a reading taken mid-
  // motion is wrong in a way the vector itself does not reveal. Near 0 means
  // the reading is good.
  if (upDevice || upCamera) {
    const f = (v: THREE.Vector3 | null) => v
      ? `[${v.x >= 0 ? ' ' : ''}${v.x.toFixed(2)}, ${v.y >= 0 ? ' ' : ''}${v.y.toFixed(2)}, ${v.z >= 0 ? ' ' : ''}${v.z.toFixed(2)}]`
      : '[    —          ]';
    lines.push(`up(dev) ${f(upDevice)}  ${aMag < 0.15 ? 'still' : 'MOVING'}`);
    lines.push(`up(cam) ${f(upCamera)}  snaps ${upSnaps.length}`);
    if (upDevice && upCamera) {
      const mapped = upDevice.clone().multiplyScalar(gravitySign)
        .applyQuaternion(imuTracker.cfg.deviceToCamera);
      lines.push(`transform err ${THREE.MathUtils.radToDeg(mapped.angleTo(upCamera)).toFixed(1)}°  gSign ${gravitySign > 0 ? '+' : '-'}`);
    }
  }
  if (useImuCorrection) {
    // The rejected count is the number worth watching: it should track the
    // ~8% cardinal-flip rate. Near zero means the gate is too loose to be
    // doing anything; far above it means the gate is eating good fixes, and
    // the vision/gyro pair on the same line says which.
    const pctRej = imuStats.fixes ? (100 * imuStats.rejected / imuStats.fixes).toFixed(0) : '—';
    lines.push(`IMU-CORR  fixes ${imuStats.fixes}  rejected ${imuStats.rejected} (${pctRej}%)`);
    lines.push(`last Δvision ${imuStats.lastVisionDeltaDeg.toFixed(1)}° vs gyro ${imuStats.lastGyroPathDeg.toFixed(1)}°`);
    // The drift, watchable live. Hold the phone STILL and coast should tick
    // up while drift stays near zero; if drift climbs while stationary that
    // is bias, and its rate is the thing to fix. While actually moving this
    // is displacement plus drift and cannot separate them.
    const dbg = imuTracker.debugState(nowMs());
    if (dbg) {
      lines.push(`coast ${dbg.coastMs.toFixed(0)}ms  drift ${dbg.driftMm.toFixed(1)}mm  v ${dbg.speedMps.toFixed(2)} m/s`);
    }
  }
  imuReadoutEl.textContent = lines.join('\n');
}

// Batched, not one message per sample: at 60Hz a message-per-sample would be
// 60 websocket writes/second competing with the JPEG frames for the same
// socket, and the frames are what the backpressure gate (canSend) is already
// fighting over. A ~100ms batch is ~6 samples, well inside the prediction
// horizon this data is for.
setInterval(() => {
  if (!isMotionListening() || imuUnsent.length === 0) return;
  const batch = imuUnsent;
  imuUnsent = [];
  host?.sendBatch({
    type: 'imu',
    // Orientation of the SCREEN relative to the device's natural orientation,
    // needed to get from the DeviceMotion frame to anything camera-relative
    // (see the axes note above). Per-batch rather than per-sample: it changes
    // at human speed, and per-sample would trip the size argument this
    // batching exists to make.
    screenAngle: (screen as any).orientation?.angle ?? 0,
    rateHz: motionRateHz(),
    samples: batch.map((s) => [s.t, s.rrAlpha, s.rrBeta, s.rrGamma, s.ax, s.ay, s.az, s.agx, s.agy, s.agz, s.interval]),
  });
}, IMU_SEND_INTERVAL_MS);

setInterval(updateImuReadout, 250);

imuCheckbox.addEventListener('change', () => {
  if (imuCheckbox.checked) void startImu();
  else stopImu();
});
// A config.phone.imuEnabled of true seeds the box checked (see the seeding
// block up by the element lookups), and a checked box that is not actually
// recording would be a lie -- so honour it here, where startImu is in scope.
// Note iOS gates the motion permission behind a user gesture, so this can
// legitimately fail at load; startImu owns reporting that, exactly as it does
// when the box is ticked by hand.
if (imuCheckbox.checked) void startImu();

// ── IMU correction ───────────────────────────────────────────────────────
//
// Behind its own checkbox, independent of "record IMU", so the two things it
// does can be compared directly against having them off. What it does:
//   1. REJECTS flipped poses (the 4-way cardinal ambiguity, measured at 8.2%
//      of consecutive pairs at ~179deg). Magnitude-only, so it is correct even
//      before deviceToCamera is verified.
//   2. PREDICTS forward to display time, covering the ~147ms between fixes.
//
// Position prediction starts OFF (see the config default): it is the only
// part that depends on deviceToCamera being right AND on boardUnitsPerMetre,
// neither of which is verified yet, and both fail silently rather than
// loudly. Orientation prediction and the gate are safe to run today.
export const imuTracker = new ImuTracker(defaultImuTrackerConfig(BOARD_UNITS_PER_METRE));
export let useImuCorrection = config.phone.imuCorrection;
// The last pose the tracker ACCEPTED, so the frame-convention pairs below
// measure vision rotation across the same span the gyro integrated over. Using
// the last pose RECEIVED instead would silently include rejected flips.
export const lastAcceptedQuat = new THREE.Quaternion();
// Rolling tally so the toggle can be judged rather than believed.
export let imuStats = { fixes: 0, rejected: 0, predicted: 0, lastVisionDeltaDeg: 0, lastGyroPathDeg: 0 };
// Pairs of (device net rotation, vision net rotation) over each inter-fix
// interval, for picking deviceToCamera FROM DATA -- see imuFrames.ts's
// scoreConventions. Collected regardless of whether correction is on, since
// it costs one vector per fix and is the prerequisite for turning position
// prediction on at all.
export const framePairs: RotationPair[] = [];
export const FRAME_PAIR_CAP = 400;

// Net rotation taking `a` to `b`, as an axis*angle vector in the frame `a`
// lives in -- the vision-side counterpart of the gyro's integrated rotation
// vector, so the two can be compared by scoreConventions.
export function rotationVectorBetween(a: THREE.Quaternion, b: THREE.Quaternion): THREE.Vector3 {
  const d = a.clone().invert().multiply(b).normalize();
  // Shortest arc: q and -q are the same rotation, but only one of them gives
  // an angle below pi, and a 350-degree turn recorded as such would swamp the
  // residual it is meant to inform.
  if (d.w < 0) { d.x = -d.x; d.y = -d.y; d.z = -d.z; d.w = -d.w; }
  const sinHalf = Math.hypot(d.x, d.y, d.z);
  if (sinHalf < 1e-9) return new THREE.Vector3();
  const angle = 2 * Math.atan2(sinHalf, d.w);
  return new THREE.Vector3(d.x, d.y, d.z).multiplyScalar(angle / sinHalf);
}

// ── "Up", seen from both sides ───────────────────────────────────────────
//
// The two live vectors the direct calibration compares (see imuFrames.ts's
// triadSolve). Both are the SAME physical direction -- up -- so with a
// correct transform, deviceToCamera applied to upDevice should equal upCamera.
//
// upDevice: normalized accelerationIncludingGravity. At rest this is the
// proper acceleration, +9.81 UPWARD, so it points up directly with no sign
// flip. Only meaningful while roughly stationary; real acceleration adds to
// it, which is why the readout shows |a| alongside so a moving reading can be
// discounted rather than trusted.
//
// upCamera: world up pushed through the INVERSE camera orientation. Assumes
// the board is horizontal -- true for a floor pattern, and the one assumption
// this method rests on.
// Raw sensor sign, as reported. NOT inverted here on purpose: the readout
// should show what the device actually said, so a convention difference is
// visible rather than silently normalized away. imuSolveFromSnaps determines
// the sign and imuApplySnapSolve writes it here and into the tracker.
export let upDevice: THREE.Vector3 | null = null;
export let upCamera: THREE.Vector3 | null = null;
export let gravitySign = 1;
export const WORLD_UP = new THREE.Vector3(0, 1, 0);
export const upSnaps: Array<{ dev: THREE.Vector3; cam: THREE.Vector3; label: string }> = [];

(globalThis as any).imuSnap = function imuSnap(label = '') {
  if (!upDevice || !upCamera) return { error: 'need both a still IMU reading and a successful pose' };
  upSnaps.push({ dev: upDevice.clone(), cam: upCamera.clone(), label: label || `snap${upSnaps.length + 1}` });
  return {
    n: upSnaps.length,
    dev: upDevice.toArray().map((v) => +v.toFixed(3)),
    cam: upCamera.toArray().map((v) => +v.toFixed(3)),
    // Angle to each earlier snap: two snaps must differ to pin the third
    // degree of freedom, and near-parallel ones cannot.
    separationDeg: upSnaps.slice(0, -1).map((s) =>
      +THREE.MathUtils.radToDeg(s.cam.angleTo(upCamera!)).toFixed(1)),
  };
};
(globalThis as any).imuClearSnaps = () => { upSnaps.length = 0; return { n: 0 }; };
(globalThis as any).imuSolveFromSnaps = function imuSolveFromSnaps() {
  if (upSnaps.length < 2) return { error: 'need 2 snaps at different tilts' };
  // Use the most widely separated pair available -- the larger the angle
  // between the two observations, the better conditioned the third axis is.
  let bi = 0, bj = 1, bestSep = -1;
  for (let i = 0; i < upSnaps.length; i++) {
    for (let j = i + 1; j < upSnaps.length; j++) {
      const sep = upSnaps[i].cam.angleTo(upSnaps[j].cam);
      if (sep > bestSep) { bestSep = sep; bi = i; bj = j; }
    }
  }
  const A = upSnaps[bi], B = upSnaps[bj];

  // ── BOTH GRAVITY SIGN CONVENTIONS ARE TRIED, NOT ASSUMED ───────────────
  //
  // Measured on the target device 2026-08-05: held flat and face-up, with the
  // rear camera looking down at the board, accelerationIncludingGravity reads
  // (0,0,-1). Device +z points out of the screen, which is UP in that pose,
  // and a spec-compliant sensor at rest reports PROPER acceleration -- i.e.
  // +9.81 upward, or (0,0,+1). So this device reports the gravity DIRECTION
  // instead, opposite to the W3C spec and to Android. A well-known iOS
  // inversion, and one that a consistent sign error would hide completely:
  // TRIAD fed two identically-negated observations does not return a cleanly
  // wrong answer, it returns a subtly wrong one that still fits its own
  // inputs.
  //
  // So rather than hardcode a platform, solve BOTH and let an objective
  // criterion pick: the correct sign lands near an axis-aligned mounting,
  // the wrong one generally does not. Reported side by side so a close call
  // is visible rather than silently resolved.
  const trySign = (sign: number) => {
    const raw = triadSolve(
      A.dev.clone().multiplyScalar(sign), B.dev.clone().multiplyScalar(sign), A.cam, B.cam);
    if (!raw) return null;
    const snapped = snapToNearestOctahedral(raw);
    return {
      sign,
      quat: snapped.q,
      deviationFromAxisAlignedDeg: +snapped.deviationDeg.toFixed(2),
      // Residual against EVERY snap, including ones that did not feed the
      // solve -- the two that did are fitted by construction and prove
      // nothing. A third snap is what makes this line meaningful.
      residuals: upSnaps.map((s) => ({
        label: s.label,
        errDeg: +THREE.MathUtils.radToDeg(
          s.dev.clone().multiplyScalar(sign).applyQuaternion(snapped.q).angleTo(s.cam)).toFixed(2),
      })),
    };
  };
  const cands = [trySign(1), trySign(-1)].filter((c) => c !== null) as NonNullable<ReturnType<typeof trySign>>[];
  if (!cands.length) return { error: 'snaps too close to parallel — tilt further between them' };
  const worst = (c: typeof cands[0]) => Math.max(...c.residuals.map((r) => r.errDeg));
  cands.sort((a, b) => (a.deviationFromAxisAlignedDeg + worst(a)) - (b.deviationFromAxisAlignedDeg + worst(b)));
  const best = cands[0];
  return {
    usedSnaps: [A.label, B.label], separationDeg: +THREE.MathUtils.radToDeg(bestSep).toFixed(1),
    gravitySign: best.sign,
    gravitySignNote: best.sign === -1
      ? 'device reports gravity DIRECTION (iOS convention) — inverted before use'
      : 'device reports proper acceleration (W3C spec convention)',
    quat: best.quat.toArray().map((v) => +v.toFixed(4)),
    deviationFromAxisAlignedDeg: best.deviationFromAxisAlignedDeg,
    residuals: best.residuals,
    // The loser, so a marginal decision is never invisible. If these two are
    // close, take a third snap at a different tilt before trusting either.
    rejectedAlternative: cands[1] ? {
      sign: cands[1].sign,
      deviationFromAxisAlignedDeg: cands[1].deviationFromAxisAlignedDeg,
      worstResidualDeg: worst(cands[1]),
    } : null,
    verdict: cands[1] && Math.abs(
      (best.deviationFromAxisAlignedDeg + worst(best))
      - (cands[1].deviationFromAxisAlignedDeg + worst(cands[1]))) < 10
      ? 'MARGINAL — take a third snap at a different tilt'
      : worst(best) > 12 ? 'POOR FIT — check the board is level and the phone was still'
        : 'CLEAR',
  };
};
(globalThis as any).imuApplySnapSolve = function imuApplySnapSolve() {
  const r: any = (globalThis as any).imuSolveFromSnaps();
  if (r.error) return r;
  imuTracker.cfg.deviceToCamera = new THREE.Quaternion().fromArray(r.quat);
  // The sign the solve chose has to reach the TRACKER too, not just the
  // readout -- it consumes raw acceleration on every sample, and a transform
  // fitted under one convention applied to data in the other is wrong in
  // exactly the way that is hardest to see.
  imuTracker.cfg.gravitySign = r.gravitySign;
  gravitySign = r.gravitySign;
  deviceToCameraPinned = true;
  return { applied: r.quat, gravitySign: r.gravitySign, residuals: r.residuals, verdict: r.verdict };
};

// Set by imuApplyFrame: once a transform has been SOLVED from data, the
// spec-derived guess must stop overwriting it on every reconstruction.
let deviceToCameraPinned = false;

export function refreshImuFrameContext() {
  if (deviceToCameraPinned) return;
  imuTracker.cfg.deviceToCamera = defaultDeviceToCamera({
    screenAngle: (screen as any).orientation?.angle ?? 0,
    facing: host?.facing() ?? 'environment',
  });
}

// Scores the 24 axis-aligned candidates against everything collected so far.
// Read over the dev bridge with `eval --phone "imuSolveFrames()"`.
(globalThis as any).imuSolveFrames = function imuSolveFrames(minDeg = 3) {
  const ranked = scoreConventions(framePairs, minDeg);
  const div = axisDiversity(framePairs, minDeg);
  const gap = ranked.length > 1 ? ranked[1].residualDeg - ranked[0].residualDeg : Infinity;
  const weakestAxis = Math.min(...div.scatter);
  return {
    pairs: framePairs.length, usable: div.n,
    // Both guards are reported as a plain verdict rather than left to be
    // eyeballed, because "the top two are close" is exactly the situation
    // where the winner still looks entirely plausible.
    axisScatter: div.scatter.map((v) => +v.toFixed(3)),
    gapDeg: +gap.toFixed(2),
    verdict:
      div.n < 40 ? 'NOT ENOUGH DATA — record more rotation'
        : weakestAxis < 0.05 ? 'ROTATION TOO PLANAR — one axis barely exercised, answer not trustworthy'
          : gap < 5 ? 'AMBIGUOUS — top two candidates too close to separate'
            : 'CLEAR — winner is well separated',
    top: ranked.slice(0, 4).map((c) => ({
      quat: [c.q.x, c.q.y, c.q.z, c.q.w].map((v) => +v.toFixed(4)),
      residualDeg: +c.residualDeg.toFixed(2),
    })),
  };
};
// Applies a solved transform without a reload (which would cost the motion
// permission). Pass the index into imuSolveFrames().top.
(globalThis as any).imuApplyFrame = function imuApplyFrame(rank = 0, minDeg = 3) {
  const ranked = scoreConventions(framePairs, minDeg);
  const pick = ranked[rank];
  if (!pick) return { error: 'no candidate at that rank' };
  imuTracker.cfg.deviceToCamera = pick.q.clone();
  // Pinned so refreshImuFrameContext stops overwriting it with the
  // spec-derived guess every reconstruction -- otherwise applying a solved
  // transform would silently last exactly one frame.
  deviceToCameraPinned = true;
  return { applied: [pick.q.x, pick.q.y, pick.q.z, pick.q.w].map((v) => +v.toFixed(4)),
    residualDeg: +pick.residualDeg.toFixed(2) };
};
(globalThis as any).imuStatus = () => ({
  on: useImuCorrection, ...imuStats, hasFix: imuTracker.hasFix(),
  cfg: {
    boardUnitsPerMetre: imuTracker.cfg.boardUnitsPerMetre,
    enableOrientationPrediction: imuTracker.cfg.enableOrientationPrediction,
    enablePositionPrediction: imuTracker.cfg.enablePositionPrediction,
    deviceToCamera: imuTracker.cfg.deviceToCamera.toArray().map((v) => +v.toFixed(4)),
  },
});
// Deliberately exposed so the config can be tuned live over the bridge
// without a reload -- a phone reload costs the motion-permission grant.
(globalThis as any).imuTracker = imuTracker;

imuCorrectionCheckbox.addEventListener('change', () => {
  useImuCorrection = imuCorrectionCheckbox.checked;
  refreshImuFrameContext();
  imuTracker.reset();
  imuStats = { fixes: 0, rejected: 0, predicted: 0, lastVisionDeltaDeg: 0, lastGyroPathDeg: 0 };
});

// ── What the page is allowed to write ────────────────────────────────────
//
// Two pieces of this module's state are set from outside, and they go through
// functions rather than exported `let`s because an imported binding cannot be
// assigned -- which is the language enforcing the thing phase 2 had to enforce
// by hand. Ambient mutable state reachable from anywhere is what made the pose
// library depend on an app; keeping the write surface named and small is how
// that does not come back one module down.

/** Clears the recording. The dataset's lifetime is the page's to decide. */
export function clearImuRing(): void {
  imuRing = [];
}

/**
 * World up expressed in the camera's own frame -- the VISION half of the direct
 * calibration, so only the pose path can produce it. Written only on an
 * ACCEPTED pose, which is what stops a flipped one being snapped.
 */
export function setUpCamera(v: THREE.Vector3 | null): void {
  upCamera = v;
}
