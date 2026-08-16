import * as THREE from 'three';

// ── Device frame -> camera frame ─────────────────────────────────────────
//
// ISOLATED IN ITS OWN FILE BECAUSE THIS IS THE DANGEROUS PART. Everything
// else in the IMU work is arithmetic that fails loudly; a wrong rotation here
// produces a predictor that is smooth, stable, self-consistent and pointed the
// wrong way. That is the exact shape of the level-line component-order bug
// (`(fx,-fy)` for `(-fy,fx)`) which shipped and survived two sessions because
// every check in place was invariant to it. So: derive a DEFAULT from the
// spec, but treat it as a hypothesis, and provide the means to pick the right
// one FROM DATA rather than from reasoning.
//
// THE SPEC'S DEVICE FRAME (DeviceOrientation/DeviceMotion): with the phone
// held upright in its natural portrait orientation, +x is right, +y is up,
// +z points OUT OF THE SCREEN toward the viewer. These axes are fixed to the
// DEVICE and do NOT rotate when the screen orientation changes -- which is
// why screenAngle has to be applied separately below.
//
// THREE'S CAMERA FRAME: looks along its own -Z, with +X right and +Y up.
//
// So for a REAR camera on a portrait phone, the camera looks out the back,
// i.e. along -z of the device -- which is exactly where a THREE camera
// already looks. The nominal transform is therefore IDENTITY, which is a
// suspiciously convenient answer and precisely the reason not to trust it
// without evidence.
//
// WHAT THIS DOES NOT KNOW: whether the video frames getUserMedia hands over
// are already rotated to match display orientation (platforms differ, and it
// can depend on the track's own settings). If they are, the effective camera
// frame has an extra screen rotation baked in that no amount of reasoning
// about sensor axes will reveal. Hence pickBestConvention below.

interface FrameContext {
  screenAngle: number;   // screen.orientation.angle, degrees: 0 / 90 / 180 / 270
  facing: 'environment' | 'user';
}

// The default hypothesis: undo the screen rotation about the viewing axis,
// then mirror for a front camera (a selfie stream is horizontally flipped, so
// its handedness differs -- represented here as a 180 degree turn about Y,
// which is the closest PROPER rotation; a true mirror is not a rotation at all
// and if the front camera is ever actually used for pose this needs revisiting
// rather than papering over).
export function defaultDeviceToCamera(ctx: FrameContext): THREE.Quaternion {
  const q = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 0, 1), THREE.MathUtils.degToRad(-ctx.screenAngle));
  if (ctx.facing === 'user') {
    q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI));
  }
  return q;
}

// ── Picking the transform from data instead of from argument ─────────────
//
// The 24 rotations of the octahedral group -- every way of mapping the axes
// onto each other with a right-handed result. The true device->camera
// transform is essentially certain to be one of these (phones do not mount
// their sensors at 37 degrees), so SEARCHING this finite set is both more
// robust than a least-squares fit and incapable of returning a non-physical
// answer. No SVD, no local minima, no sign ambiguity.
function octahedralRotations(): THREE.Quaternion[] {
  const out: THREE.Quaternion[] = [];
  const axes = [0, 1, 2];
  const perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
  for (const p of perms) {
    for (let signBits = 0; signBits < 8; signBits++) {
      const s = [signBits & 1 ? -1 : 1, signBits & 2 ? -1 : 1, signBits & 4 ? -1 : 1];
      // Columns of the candidate matrix: where each source axis lands.
      const m = new THREE.Matrix4();
      const e = m.elements;
      for (const c of axes) {
        e[c * 4 + 0] = p[c] === 0 ? s[c] : 0;
        e[c * 4 + 1] = p[c] === 1 ? s[c] : 0;
        e[c * 4 + 2] = p[c] === 2 ? s[c] : 0;
      }
      // Keep proper rotations only -- a determinant of -1 is a reflection,
      // which no physical mounting can produce.
      const det =
        e[0] * (e[5] * e[10] - e[6] * e[9])
        - e[4] * (e[1] * e[10] - e[2] * e[9])
        + e[8] * (e[1] * e[6] - e[2] * e[5]);
      if (det < 0) continue;
      out.push(new THREE.Quaternion().setFromRotationMatrix(m));
    }
  }
  return out;
}

// One observation: how the DEVICE said it rotated over an interval, against
// how VISION said the camera rotated over the same interval. Both as rotation
// vectors (axis * angle, radians).
export interface RotationPair { device: THREE.Vector3; camera: THREE.Vector3 }

// ── The direct route: two known directions, seen in both frames ──────────
//
// Far better than the statistical fit below, and it is what should be used
// when it is available. At rest the accelerometer measures a direction we
// already know the name of -- UP -- expressed in the device frame. The
// recovered pose independently gives UP in the camera frame (world up pushed
// through the inverse camera orientation). Same physical arrow, two frames:
// one observation constrains two of the three degrees of freedom, and a
// SECOND observation at a different tilt pins the last one exactly.
//
// No motion diversity to worry about, no residual to interpret, no statistics
// at all -- and it can be checked by eye, since with the phone flat above a
// horizontal board both vectors should read (0,0,1).
//
// TRIAD: build a right-handed triad from each pair and compose. Exact for two
// non-parallel observations.
export function triadSolve(
  aDev: THREE.Vector3, bDev: THREE.Vector3,
  aCam: THREE.Vector3, bCam: THREE.Vector3,
): THREE.Quaternion | null {
  const frame = (u: THREE.Vector3, v: THREE.Vector3) => {
    const e1 = u.clone().normalize();
    const cross = e1.clone().cross(v);
    // Parallel observations carry no information about the third axis -- the
    // cross product vanishes and the triad is undefined. Caller must tilt
    // further rather than receive a confidently wrong answer.
    if (cross.length() < 1e-6) return null;
    const e2 = cross.normalize();
    const e3 = e1.clone().cross(e2);
    return new THREE.Matrix4().makeBasis(e1, e2, e3);
  };
  const A = frame(aDev, bDev), B = frame(aCam, bCam);
  if (!A || !B) return null;
  // R = B * A^T maps device -> camera.
  const R = B.multiply(A.transpose());
  return new THREE.Quaternion().setFromRotationMatrix(R).normalize();
}

// Rounds a solved rotation to the nearest physically-plausible mounting. The
// true transform is axis-aligned, so snapping removes measurement noise
// outright rather than merely reducing it -- a 4-degree error in the snaps
// becomes a 0-degree error in the answer.
export function snapToNearestOctahedral(q: THREE.Quaternion):
{ q: THREE.Quaternion; deviationDeg: number } {
  let best = octahedralRotations()[0], bestAngle = Infinity;
  for (const cand of octahedralRotations()) {
    const dot = Math.min(1, Math.abs(q.dot(cand)));
    const ang = 2 * Math.acos(dot);
    if (ang < bestAngle) { bestAngle = ang; best = cand; }
  }
  return { q: best.clone(), deviationDeg: THREE.MathUtils.radToDeg(bestAngle) };
}

// ── Is the data even capable of answering the question? ──────────────────
//
// THE FAILURE MODE THAT WASTES A RECORDING: rotations all about ONE axis
// cannot distinguish candidates that agree on that axis and differ elsewhere.
// Spin the phone only about its optical axis and several transforms will tie
// at a low residual, all of them looking equally convincing -- a confident
// wrong answer, which is the exact class of error this whole module exists to
// avoid.
//
// Reports the diagonal of the scatter matrix of normalized rotation axes,
// which sums to 1. [0.95, 0.03, 0.02] means everything turned about one axis
// and the recording is not usable. Something like [0.4, 0.35, 0.25] means all
// three were exercised.
//
// LIMITATION, stated because it would otherwise be a trap of its own: the
// diagonal only detects dominance along the COORDINATE axes. Rotations
// concentrated about a diagonal direction would spread across all three
// entries and read as healthy. A full eigen-decomposition would catch that;
// this is the cheap version, and the candidate residual GAP below is the real
// backstop -- if the data cannot separate the candidates, the gap says so
// regardless of why.
export function axisDiversity(pairs: RotationPair[], minDeg = 3):
{ n: number; scatter: [number, number, number] } {
  const usable = pairs.filter((p) => p.camera.length() > THREE.MathUtils.degToRad(minDeg));
  const s: [number, number, number] = [0, 0, 0];
  for (const p of usable) {
    const u = p.camera.clone().normalize();
    s[0] += u.x * u.x; s[1] += u.y * u.y; s[2] += u.z * u.z;
  }
  if (usable.length) { s[0] /= usable.length; s[1] /= usable.length; s[2] /= usable.length; }
  return { n: usable.length, scatter: s };
}

// Scores every octahedral candidate against recorded pairs and returns them
// best-first. The score is mean angular residual in DEGREES between the
// rotated device vector and the observed camera vector.
//
// ONLY PAIRS WITH REAL ROTATION ARE INFORMATIVE: a near-zero rotation vector
// has an ill-conditioned direction, so it contributes noise to every candidate
// equally and simply dilutes the contrast between them. minDeg drops those.
export function scoreConventions(pairs: RotationPair[], minDeg = 3):
Array<{ q: THREE.Quaternion; residualDeg: number; n: number }> {
  const usable = pairs.filter((p) => p.camera.length() > THREE.MathUtils.degToRad(minDeg));
  const out = octahedralRotations().map((q) => {
    let sum = 0;
    for (const p of usable) {
      const rotated = p.device.clone().applyQuaternion(q);
      // Angle between the two rotation vectors, which is what a wrong axis
      // mapping corrupts -- deliberately NOT the magnitude difference, since
      // magnitude is exactly the quantity that is invariant to this transform
      // and so cannot distinguish any candidate from any other.
      const la = rotated.length(), lb = p.camera.length();
      if (la < 1e-9 || lb < 1e-9) continue;
      const cos = THREE.MathUtils.clamp(rotated.dot(p.camera) / (la * lb), -1, 1);
      sum += THREE.MathUtils.radToDeg(Math.acos(cos));
    }
    return { q, residualDeg: usable.length ? sum / usable.length : Infinity, n: usable.length };
  });
  out.sort((a, b) => a.residualDeg - b.residualDeg);
  return out;
}
