import * as THREE from 'three';

// ── IMU dead-reckoning between absolute pose fixes ───────────────────────
//
// Integrates rotationRate/acceleration forward from the last vision pose, and
// gets HARD RESET every time a new one arrives. Not a Kalman filter, on
// purpose: with a fix arriving at ~6.8Hz and measured at-rest drift of ~0.7mm
// over a 147ms gap, there is nothing for optimal blending to recover that
// overwriting does not. A filter earns its place when the gaps get long or the
// measurement gets noisy enough that trusting it wholesale is worse than
// averaging -- neither is true yet. Everything here is deliberately the
// simplest thing that can work, so that when it does not work the reason is
// visible.
//
// USES THE OS'S GRAVITY-COMPENSATED `acceleration`, not
// accelerationIncludingGravity: measured residual at rest on the target device
// was 0.0609 m/s^2, implying the OS attitude estimate is good to ~0.36deg,
// which is far better than anything this code would derive on its own.
//
// WHAT IS AND IS NOT ROBUST TO A WRONG deviceToCamera (see imuFrames.ts):
//   - the FLIP GATE is robust. It compares rotation MAGNITUDES, which are
//     invariant under any rotation of the source frame, so it works correctly
//     even with the transform completely wrong.
//   - orientation and position prediction are NOT. They rotate vectors
//     through it, so a wrong transform yields confident nonsense.
// That split is why they are separately switchable below: the gate is usable
// on day one, prediction only once the transform is verified.

interface ImuTrackerConfig {
  // Subtracted from every sample. Measure by averaging a static window --
  // vision can detect "not moving" for free.
  gyroBiasDegPerSec: THREE.Vector3;
  accelBiasMps2: THREE.Vector3;
  // Board units per metre -- from constants.ts's CELL_SIZE_METRES, which is
  // the only place the physical board's size is recorded. Overridable here
  // because it is also RECOVERABLE FROM DATA (regress integrated-accel
  // displacement against vision displacement; the slope IS this number), and
  // that measurement is worth having as an independent check on the ruler.
  boardUnitsPerMetre: number;
  deviceToCamera: THREE.Quaternion;
  // +1 if the device reports PROPER acceleration (W3C spec: at rest, +9.81
  // UPWARD); -1 if it reports the gravity direction instead, which iOS does.
  // Determined empirically by imuSolveFromSnaps rather than assumed from the
  // platform. Applies to the gravity-free `acceleration` too: both fields
  // come from the same sensor pipeline and share its sign convention, so a
  // device that inverts one inverts the other.
  gravitySign: number;

  // Flip gate. A measurement is rejected when vision claims a rotation the
  // gyro says did not happen -- the 4-way cardinal ambiguity of the square
  // lattice, measured at 8.2% of consecutive pairs with a median of 178.9deg.
  // Rejecting is correct rather than merely safe: a flipped pose has its
  // POSITION in the wrong frame too (median jump 61 board units), so there is
  // no salvageable half to keep.
  flipGateMinDeg: number;    // never gate rotations smaller than this
  flipGateSlack: number;     // multiple of the gyro-measured path allowed
  flipGateSlackDeg: number;  // plus this absolute margin

  // How much of the vision-differenced velocity to adopt on each fix. THE ONE
  // REAL TUNING KNOB. Vision measures position, never velocity, so velocity is
  // the single quantity no fix can correct directly -- and velocity error is
  // exactly what makes position drift between fixes (delta_v * 0.147s).
  // 0 = trust integrated velocity only, 1 = replace it wholesale each fix.
  velocityBlend: number;

  // Refuse to predict further than this past the last fix; beyond it the
  // answer is worse than admitting ignorance. At-rest drift reaches ~33mm by
  // 1s, so this should stay well inside that.
  maxCoastMs: number;

  enableOrientationPrediction: boolean;
  enablePositionPrediction: boolean;
}

/**
 * `boardUnitsPerMetre` is REQUIRED rather than defaulted, because it is the one
 * field here that is a fact about somebody's PRINTED BOARD rather than about an
 * IMU. It used to read Pose Viewer's constant directly, which made this
 * platform layer depend on one of the apps -- and silently gave any other app
 * that app's board scale. Each caller passes its own.
 */
export function defaultImuTrackerConfig(boardUnitsPerMetre: number): ImuTrackerConfig {
  return {
    gyroBiasDegPerSec: new THREE.Vector3(0, 0, 0),
    accelBiasMps2: new THREE.Vector3(0, 0, 0),
    boardUnitsPerMetre,
    // SOLVED 2026-08-05 from three tilt snaps, hand-verified against the
    // prediction that a positive device-x tilt must show as negative camera-x.
    // Identity is the real answer, not a placeholder: the rear camera looks
    // along device -z and a THREE camera looks along its own -Z.
    deviceToCamera: new THREE.Quaternion(),
    // MEASURED, not assumed: held flat and face-up the device reports
    // (0,0,-1) where the W3C spec calls for (0,0,+1). It reports the gravity
    // DIRECTION rather than proper acceleration -- the known iOS inversion.
    gravitySign: -1,
    flipGateMinDeg: 25,
    flipGateSlack: 2.5,
    flipGateSlackDeg: 12,
    velocityBlend: 0.5,
    maxCoastMs: 400,
    enableOrientationPrediction: true,
    // ON as of 2026-08-05. deviceToCamera and boardUnitsPerMetre are both
    // settled, so the remaining objection is not correctness of this code but
    // a SEPARATE defect upstream: the vision pipeline over-reports tilt by
    // ~1.46x (measured against the accelerometer over two independent tilt
    // axes), almost certainly a camera-intrinsics/FOV error. That rotates the
    // acceleration into the board frame slightly wrong, so expect a
    // systematic cross-track component of roughly 20% of any displacement.
    //
    // Turned on anyway, deliberately: the point is to SEE the drift and to
    // get gap-filling when a decode fails, and both are visible through that
    // bias. What must NOT be done until the FOV is fixed is TUNING against
    // it -- velocityBlend would absorb the orientation error, and the
    // empirical boardUnitsPerMetre regression would launder it into the scale.
    enablePositionPrediction: true,
  };
}

interface ImuSampleIn {
  t: number;                                       // phone clock, epoch ms
  rotationRate: { alpha: number; beta: number; gamma: number } | null; // deg/s
  acceleration: { x: number; y: number; z: number } | null;            // m/s^2
}

interface AnchorResult {
  accepted: boolean;
  reason: 'ok' | 'flip-rejected' | 'first-fix' | 'no-imu';
  visionDeltaDeg: number;
  gyroPathDeg: number;
}

export class ImuTracker {
  cfg: ImuTrackerConfig;
  // Nominal state, all in the VISION/board frame.
  private p = new THREE.Vector3();
  private v = new THREE.Vector3();
  private q = new THREE.Quaternion();
  private tState = 0;
  private haveFix = false;

  // Since the last anchor: total angular path (deg) and the net rotation
  // vector. PATH, not net rotation, is what the gate wants -- it is an upper
  // bound on how far the orientation can possibly have moved, so exceeding it
  // is unambiguous evidence rather than a heuristic.
  private gyroPathDeg = 0;
  private gyroNetRotVec = new THREE.Vector3();
  private lastSampleT: number | null = null;
  private lastOmegaCam = new THREE.Vector3();
  private lastFixT = 0;
  private lastFixQuat = new THREE.Quaternion();
  private lastFixPos = new THREE.Vector3();
  private haveLastFix = false;
  private sawAnySample = false;

  constructor(cfg: ImuTrackerConfig) { this.cfg = cfg; }

  reset() {
    this.haveFix = false; this.haveLastFix = false; this.sawAnySample = false;
    this.v.set(0, 0, 0); this.clearInterval();
  }

  private clearInterval() {
    this.gyroPathDeg = 0; this.gyroNetRotVec.set(0, 0, 0); this.lastSampleT = null;
  }

  pushSample(s: ImuSampleIn) {
    this.sawAnySample = true;
    const dt = this.lastSampleT === null ? 0 : (s.t - this.lastSampleT) / 1000;
    this.lastSampleT = s.t;
    // A gap this large means the page was backgrounded or the handler was
    // starved; integrating across it would invent motion that never happened.
    if (dt <= 0 || dt > 0.25) { this.tState = s.t; return; }

    // Angular rate, device frame, deg/s -> rad/s, bias removed.
    const rr = s.rotationRate;
    const wDev = new THREE.Vector3(
      ((rr?.alpha ?? 0) - this.cfg.gyroBiasDegPerSec.x) * Math.PI / 180,
      ((rr?.beta ?? 0) - this.cfg.gyroBiasDegPerSec.y) * Math.PI / 180,
      ((rr?.gamma ?? 0) - this.cfg.gyroBiasDegPerSec.z) * Math.PI / 180,
    );
    this.gyroPathDeg += THREE.MathUtils.radToDeg(wDev.length()) * dt;
    this.gyroNetRotVec.addScaledVector(wDev, dt);

    const wCam = wDev.clone().applyQuaternion(this.cfg.deviceToCamera);
    this.lastOmegaCam.copy(wCam);

    if (this.haveFix && this.cfg.enableOrientationPrediction) {
      this.q.multiply(smallRotationQuat(wCam, dt)).normalize();
    }
    if (this.haveFix && this.cfg.enablePositionPrediction) {
      const a = s.acceleration;
      const g = this.cfg.gravitySign;
      const aDev = new THREE.Vector3(
        (a?.x ?? 0) * g - this.cfg.accelBiasMps2.x,
        (a?.y ?? 0) * g - this.cfg.accelBiasMps2.y,
        (a?.z ?? 0) * g - this.cfg.accelBiasMps2.z,
      );
      // Device -> camera body -> world/board, then metres -> board units.
      const aBoard = aDev.applyQuaternion(this.cfg.deviceToCamera)
        .applyQuaternion(this.q)
        .multiplyScalar(this.cfg.boardUnitsPerMetre);
      this.p.addScaledVector(this.v, dt).addScaledVector(aBoard, 0.5 * dt * dt);
      this.v.addScaledVector(aBoard, dt);
    }
    this.tState = s.t;
  }

  // A new absolute fix from the pose pipeline. Returns whether it was taken.
  anchor(t: number, camPos: THREE.Vector3, camQuat: THREE.Quaternion): AnchorResult {
    const gyroPathDeg = this.gyroPathDeg;
    const visionDeltaDeg = this.haveLastFix
      ? THREE.MathUtils.radToDeg(quatAngle(this.lastFixQuat, camQuat)) : 0;

    if (!this.sawAnySample) {
      this.acceptFix(t, camPos, camQuat);
      return { accepted: true, reason: 'no-imu', visionDeltaDeg, gyroPathDeg };
    }
    if (!this.haveLastFix) {
      this.acceptFix(t, camPos, camQuat);
      return { accepted: true, reason: 'first-fix', visionDeltaDeg, gyroPathDeg };
    }
    // THE GATE. Magnitude-only, so it holds regardless of deviceToCamera.
    const allowed = Math.max(
      this.cfg.flipGateMinDeg,
      gyroPathDeg * this.cfg.flipGateSlack + this.cfg.flipGateSlackDeg,
    );
    if (visionDeltaDeg > allowed) {
      // Rejected: keep dead-reckoning from the previous good fix. The interval
      // accumulator is NOT cleared, so the next fix is judged against the gyro
      // path since the last ACCEPTED pose rather than since this bad one --
      // otherwise a rejected measurement would shrink the budget and cause a
      // cascade of rejections after any single flip.
      return { accepted: false, reason: 'flip-rejected', visionDeltaDeg, gyroPathDeg };
    }
    // Velocity is the one thing the fix cannot set directly; difference the
    // last two accepted positions and blend.
    const dtFix = (t - this.lastFixT) / 1000;
    if (dtFix > 1e-3 && this.cfg.enablePositionPrediction) {
      const vMeas = camPos.clone().sub(this.lastFixPos).divideScalar(dtFix);
      this.v.lerp(vMeas, THREE.MathUtils.clamp(this.cfg.velocityBlend, 0, 1));
    }
    this.acceptFix(t, camPos, camQuat);
    return { accepted: true, reason: 'ok', visionDeltaDeg, gyroPathDeg };
  }

  private acceptFix(t: number, camPos: THREE.Vector3, camQuat: THREE.Quaternion) {
    this.p.copy(camPos); this.q.copy(camQuat);
    this.lastFixT = t; this.lastFixPos.copy(camPos); this.lastFixQuat.copy(camQuat);
    this.haveFix = true; this.haveLastFix = true;
    this.tState = t;
    this.clearInterval();
  }

  // The corrected pose at time t. Extrapolates past the newest sample with
  // constant angular rate and constant velocity -- over the ~15ms between the
  // last IMU sample and display, anything more elaborate is noise.
  predictAt(t: number): { camPos: THREE.Vector3; camQuat: THREE.Quaternion } | null {
    if (!this.haveFix) return null;
    if (t - this.lastFixT > this.cfg.maxCoastMs) return null;
    const dt = Math.max(0, (t - this.tState) / 1000);
    const q = this.q.clone();
    if (this.cfg.enableOrientationPrediction && dt > 0) {
      q.multiply(smallRotationQuat(this.lastOmegaCam, dt)).normalize();
    }
    const p = this.p.clone();
    if (this.cfg.enablePositionPrediction && dt > 0) p.addScaledVector(this.v, dt);
    return { camPos: p, camQuat: q };
  }

  // For the calibration path in imuFrames.ts: the net rotation the DEVICE has
  // turned through since the last accepted fix, pairable with the vision
  // rotation over the same interval.
  // How far dead reckoning has carried the state away from the fix it started
  // at, and for how long. This is the drift, made watchable -- the whole
  // reason to run position prediction before the upstream tilt error is fixed.
  // `driftMm` is the honest name for it only while genuinely stationary; when
  // the phone is actually moving it is displacement plus drift, and nothing
  // here can separate the two without ground truth.
  debugState(t: number): { coastMs: number; driftMm: number; speedMps: number } | null {
    if (!this.haveFix) return null;
    return {
      coastMs: t - this.lastFixT,
      driftMm: (this.p.distanceTo(this.lastFixPos) / this.cfg.boardUnitsPerMetre) * 1000,
      speedMps: this.v.length() / this.cfg.boardUnitsPerMetre,
    };
  }

  netDeviceRotation(): THREE.Vector3 { return this.gyroNetRotVec.clone(); }
  lastFixQuaternion(): THREE.Quaternion { return this.lastFixQuat.clone(); }
  hasFix(): boolean { return this.haveFix; }
}

// Exact for a constant angular-velocity step, rather than the usual
// first-order approximation: at 60Hz the difference is negligible, but the
// exact form costs one sin/cos and cannot slowly denormalize the quaternion
// over a long run of samples.
function smallRotationQuat(omega: THREE.Vector3, dt: number): THREE.Quaternion {
  const theta = omega.length() * dt;
  if (theta < 1e-12) return new THREE.Quaternion();
  return new THREE.Quaternion().setFromAxisAngle(omega.clone().normalize(), theta);
}

function quatAngle(a: THREE.Quaternion, b: THREE.Quaternion): number {
  const dot = Math.min(1, Math.abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w));
  return 2 * Math.acos(dot);
}
