// Generates synthetic camera captures of the pattern PNG under a real pinhole
// projective transform — not just in-plane rotation (see scripts/test-decode.ts's
// cropAtRotated, which only ever simulates a camera looking straight down at
// the mat). World coordinates match the PNG's own pixel space directly (the
// pattern lies flat in the Z=0 plane); the camera is positioned via spherical
// coordinates around a target point on the mat, so tilt alone controls how
// much genuine perspective foreshortening appears across the capture.

export interface CameraPose {
  targetX: number; targetY: number; // world point the camera is centered on (PNG pixel coords)
  dist: number;    // camera height above the target, along the sphere radius
  tilt: number;    // radians from straight-down (0 = looking straight down, no perspective)
  azimuth: number; // radians, which horizontal direction the tilt leans
  roll: number;    // radians, extra in-plane camera rotation (in addition to tilt direction)
  focal: number;   // focal length in the same pixel units as dist/target
}

type Vec3 = [number, number, number];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

// Builds the camera's position and (right, up, forward) basis for a pose.
// tilt=0 places the camera directly above target, looking straight down —
// the degenerate case that matches the old rotation-only harness exactly
// (azimuth becomes meaningless, roll alone reproduces the old theta param).
function buildCamera(pose: CameraPose) {
  const { targetX, targetY, dist, tilt, azimuth, roll } = pose;
  const target: Vec3 = [targetX, targetY, 0];
  const camPos: Vec3 = add(target, [
    dist * Math.sin(tilt) * Math.cos(azimuth),
    dist * Math.sin(tilt) * Math.sin(azimuth),
    dist * Math.cos(tilt),
  ]);
  const forward = norm(sub(target, camPos));
  const worldUpHint: Vec3 = [0, 1, 0]; // arbitrary reference, safe unless tilt -> 90deg exactly
  let right = norm(cross(forward, worldUpHint));
  let up = cross(right, forward);
  const cosR = Math.cos(roll), sinR = Math.sin(roll);
  const right2 = add(scale(right, cosR), scale(up, sinR));
  const up2 = add(scale(right, -sinR), scale(up, cosR));
  right = right2; up = up2;
  return { camPos, right, up, forward };
}

// Returns a function mapping a destination pixel (u,v), centered on a
// rawW x rawH buffer, to the source-plane (x,y) it corresponds to under this
// camera pose — i.e. the inverse of the pose's homography. Ray-plane
// intersection with Z=0 makes this a genuine projective (not affine) map:
// apparent pitch varies smoothly across the frame once tilt > 0, exactly the
// distortion a locally-constant or locally-linear pitch model can't capture.
export function makeHomographySampler(pose: CameraPose, rawW: number, rawH: number) {
  const { camPos, right, up, forward } = buildCamera(pose);
  const f = pose.focal;
  return (u: number, v: number): [number, number] | null => {
    const du = u - rawW / 2, dv = v - rawH / 2;
    const dir: Vec3 = add(forward, add(scale(right, du / f), scale(up, dv / f)));
    if (dir[2] >= -1e-9) return null; // ray doesn't point toward the plane
    const t = -camPos[2] / dir[2];
    if (t <= 0) return null;
    return [camPos[0] + t * dir[0], camPos[1] + t * dir[1]];
  };
}

export interface SourceImage { width: number; height: number; data: Uint8Array | Buffer; }

// Renders a rawW x rawH RGBA capture by nearest-neighbor sampling `png`
// through the pose's homography, wrapping at the pattern's tile edges (same
// torus wraparound cropAtRotated uses) so the target is never clipped by
// running off the image boundary.
export function captureHomography(png: SourceImage, pose: CameraPose, rawW: number, rawH: number): Uint8ClampedArray {
  const sampler = makeHomographySampler(pose, rawW, rawH);
  const out = new Uint8ClampedArray(rawW * rawH * 4);
  for (let v = 0; v < rawH; v++) {
    for (let u = 0; u < rawW; u++) {
      const dstIdx = (rawW * v + u) << 2;
      const hit = sampler(u, v);
      if (!hit) { out[dstIdx] = out[dstIdx + 1] = out[dstIdx + 2] = 255; out[dstIdx + 3] = 255; continue; }
      const sx = Math.round(hit[0]), sy = Math.round(hit[1]);
      const wx = ((sx % png.width) + png.width) % png.width;
      const wy = ((sy % png.height) + png.height) % png.height;
      const srcIdx = (png.width * wy + wx) << 2;
      out[dstIdx] = png.data[srcIdx];
      out[dstIdx + 1] = png.data[srcIdx + 1];
      out[dstIdx + 2] = png.data[srcIdx + 2];
      out[dstIdx + 3] = 255;
    }
  }
  return out;
}
