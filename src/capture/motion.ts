import { nowMs } from '../clock.ts';

// ── DeviceMotion, and the platform quirks that come with it ──────────────
//
// The device-input half of the IMU path: permission, subscription, sample
// normalization, and the delivery-rate estimate. What a sample MEANS -- the
// ring buffer, the fusion, the readout, the recording -- belongs to whoever is
// capturing, and lives in that app.
//
// This is a neutral platform layer, a peer of `src/gpu/`, because BOTH clients
// need it independently and the isolation rule forbids them sharing it through
// each other. Duplicating it would mean two divergent copies of the quirks
// below, which were expensive to find and are invisible until they are wrong.
//
// ── AXES ARE NOT THE CAMERA'S AXES ──
//
// DeviceMotion reports in the device frame (x right, y up, z out of the screen,
// with the phone held upright in portrait) -- the REAR camera looks along -z of
// that frame, and `screen.orientation` rotates the whole thing again. NO
// rotation into the optical frame is applied here, on purpose: raw is what a
// recording should hold, and the device->camera transform is a calibration with
// a sign error waiting in it.
//
// Units, per the DeviceOrientation spec: rotationRate in DEGREES/SECOND, both
// acceleration fields in m/s^2.

export interface MotionSample {
  /**
   * When the HANDLER RAN, not when the sensor sampled. Between them sit the
   * OS's own batching and the browser's task queue, and that delay is a
   * constant offset a fusion step has to estimate rather than assume. Recorded
   * as-is rather than "corrected" by a guess.
   */
  t: number;
  /** rotationRate, deg/s. */
  rrAlpha: number; rrBeta: number; rrGamma: number;
  /** acceleration, gravity REMOVED by the OS. */
  ax: number; ay: number; az: number;
  /** accelerationIncludingGravity, raw-er. */
  agx: number; agy: number; agz: number;
  /** Sampling interval, NORMALIZED TO ms -- see normalizeMotionInterval. */
  interval: number;
}

export type MotionPermission = 'unknown' | 'granted' | 'denied' | 'unsupported';

/**
 * `DeviceMotionEvent.interval` IS IN SECONDS ON iOS, despite the spec saying
 * milliseconds -- measured 2026-08-04, the device reported 0.0166667 while
 * actually delivering 58.8Hz. Taken literally as ms that reads as a 60kHz
 * sensor.
 *
 * A 60Hz sensor is 16.7ms or 0.0167s and nothing real sits near 1, so the units
 * are separable by magnitude with an enormous margin either side.
 */
export function normalizeMotionInterval(raw: number | undefined): number {
  if (!raw || raw <= 0) return 0;
  return raw < 1 ? raw * 1000 : raw;
}

let listening = false;
let permission: MotionPermission = 'unknown';
let handler: ((e: DeviceMotionEvent) => void) | null = null;

// Rolling rate estimate, from the handler's own ARRIVAL TIMES rather than the
// event's self-reported `interval` -- the two disagreeing is itself worth
// seeing, since it means the browser is coalescing or throttling delivery.
let arrivalTimes: number[] = [];
let lastRateHz: number | null = null;

/** Measured delivery rate, or null before two samples have arrived. */
export function motionRateHz(): number | null {
  return lastRateHz;
}

export function motionPermission(): MotionPermission {
  return permission;
}

export function isMotionListening(): boolean {
  return listening;
}

/**
 * Subscribes to devicemotion, asking for permission first where that exists.
 *
 * **iOS (13+) gates devicemotion behind a prompt that MUST be requested from a
 * user gesture**, so this has to hang off a real interaction rather than run at
 * load. Android and desktop have no `requestPermission` at all, where
 * feature-detecting it away and just subscribing is the CORRECT behaviour, not
 * a fallback.
 *
 * Returns the resulting permission state; the caller is what renders it.
 */
export async function startMotion(onSample: (s: MotionSample) => void): Promise<MotionPermission> {
  if (listening) return permission;
  if (typeof DeviceMotionEvent === 'undefined') {
    permission = 'unsupported';
    return permission;
  }
  const DME = DeviceMotionEvent as any;
  if (typeof DME.requestPermission === 'function') {
    try {
      // Called AS A METHOD on DeviceMotionEvent, not via a detached reference
      // -- it is a static method, and pulling it into a local first would
      // invoke it with `this` undefined, which some WebKit versions reject
      // outright.
      const res = await DME.requestPermission();
      permission = res === 'granted' ? 'granted' : 'denied';
    } catch {
      // Thrown when the call did not originate from a user gesture, which is a
      // DIFFERENT failure from the user tapping "deny" -- surfaced as denied
      // either way, but the distinction is why a readout should say "re-tap"
      // rather than "permission refused".
      permission = 'denied';
    }
  } else {
    permission = 'granted';
  }
  if (permission !== 'granted') return permission;

  handler = (e: DeviceMotionEvent) => {
    const t = nowMs();
    const rr = e.rotationRate;
    const a = e.acceleration;
    const ag = e.accelerationIncludingGravity;
    arrivalTimes.push(t);
    if (arrivalTimes.length > 60) arrivalTimes.shift();
    if (arrivalTimes.length >= 2) {
      const span = arrivalTimes[arrivalTimes.length - 1]! - arrivalTimes[0]!;
      lastRateHz = span > 0 ? ((arrivalTimes.length - 1) / span) * 1000 : null;
    }
    // Every component is individually nullable in the DOM types; resolving that
    // HERE, once, is what stops a consumer and a recording disagreeing about
    // what a given sample was.
    onSample({
      t,
      rrAlpha: rr?.alpha ?? 0, rrBeta: rr?.beta ?? 0, rrGamma: rr?.gamma ?? 0,
      ax: a?.x ?? 0, ay: a?.y ?? 0, az: a?.z ?? 0,
      agx: ag?.x ?? 0, agy: ag?.y ?? 0, agz: ag?.z ?? 0,
      interval: normalizeMotionInterval(e.interval),
    });
  };
  window.addEventListener('devicemotion', handler);
  listening = true;
  return permission;
}

export function stopMotion(): void {
  if (!listening) return;
  if (handler) window.removeEventListener('devicemotion', handler);
  handler = null;
  listening = false;
  arrivalTimes = [];
  lastRateHz = null;
}
