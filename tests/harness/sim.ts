import * as THREE from 'three';
import type { Board } from '../../src/pose/board.ts';

// ── The simulator: render what a camera at a known pose would see ─────────
//
// This is the foundation of the pose sweep, and the reason the sweep can give
// GROUND TRUTH rather than a second opinion: the image is generated FROM a
// pose, so the true axes, height, period, phase and camera pose are all known
// exactly. Nothing here is an approximation of the answer -- it IS the answer,
// and the pipeline's job is to get back to it.
//
// ── Why not reuse the app's simulated camera ──
//
// That one renders through the three.js scene, which would tie the sweep to a
// browser. A flat textured plane under a pinhole camera is closed-form, so this
// runs headless -- and it is at parity, because the existing simulator has no
// lens model either.
//
// ── The one convention that must not drift ──
//
// Rays are cast with the SAME projection the pose pipeline uses. If this file
// rolled its own, a sign or axis disagreement would show up as pipeline "error"
// that is really a simulator bug, and the two are indistinguishable from inside
// a sweep.
//
// It is not literally the same CALL: `cornerDir` allocates a Vector3 per ray,
// and a 480x640 frame at 2x2 supersampling is 1.23M rays -- 180 such frames
// exhausted the V8 heap. `rayDirInto` below is that function written out into
// scalars, and it is held to the original BY TEST rather than by sharing:
// tests/poseSim.test.ts asserts the two agree bit-for-bit across the NDC range
// and a spread of orientations. Verified equivalence, not assumed.
//
// Everything here is WORLD space: the floor is the y = 0 plane, and board cell
// (row, col) has its centre at ((col + 0.5 - C/2) * cellPitch, 0,
// (row + 0.5 - R/2) * cellPitch) -- the exact formula finishPositionDecode
// uses to turn a decoded anchor back into a position, so a recovered camPos is
// directly comparable to the one that generated the image.

/**
 * The world being rendered: which board is printed on the floor, and how big a
 * cell is in world units.
 *
 * Both are the CALLER's to choose, and they have to be the same two values the
 * reconstruction is given -- the board because the simulator has to render the
 * pattern the decoder decodes, and `cellPitch` because it is what
 * `GppSettings`/`LayoutSettings` carry into the pipeline. A disagreement here
 * is not a subtle error: the sweep would be scoring one world against another.
 */
export interface SimWorld {
  board: Board;
  /** World units per pattern cell. The app's GRID_STEP. */
  cellPitch: number;
}

/**
 * A camera pose, parameterized the way a sweep wants to vary it rather than
 * the way the maths wants to consume it.
 */
export interface SimPose {
  /** Height above the floor, in board units (1 unit = 1 cell = cellPitch). */
  height: number;
  /** Where the camera sits over the board, in fractional cell coordinates. */
  overRow: number;
  overCol: number;
  /** Degrees away from straight down. 0 = nadir, 60 = strongly oblique. */
  tiltDeg: number;
  /** Compass direction of the tilt, degrees. Also the board-relative yaw. */
  yawDeg: number;
  /** Rotation about the view axis, degrees. */
  rollDeg?: number;
}

export interface SimDims {
  w: number;
  h: number;
  /** HORIZONTAL field of view. Vertical is derived via the aspect, matching
   *  getAnalysisVFovRad -- the pipeline's own convention. */
  horizFovDeg: number;
}

// ── The three levels this renderer paints, MEASURED off the app ──────────
//
// These were read out of pose-viewer-server.html rather than chosen, because
// the app's floor is a lit MeshStandardMaterial behind THREE's tone mapping and
// sRGB encode, and reproducing that analytically in a CPU ray caster would be
// guesswork. The recipe, on the dev bridge, is short enough to repeat whenever
// the scene's lighting or the floor texture changes:
//
//   updateGizmo(cam); renderCamRT(cam);
//   renderer.readRenderTargetPixels(cam.camRT, ...) -> luma -> histogram
//
// at a nadir pose low enough for the board to fill the frame (dark/light), and
// again high enough to see past its edge (background).
//
// WHAT THE OLD VALUES COST. This renderer used to paint 0 and 255 with a
// neutral 127.5 sky, so it fed the detector 52% MORE CONTRAST than the app ever
// produces -- straight into lsdRhoNoiseThreshold, which is a gradient magnitude
// -- and a background that could not drag decode's binarization mean, which the
// app's very dark one does. Two sweeps' worth of conclusions were drawn across
// that gap before it was noticed.
//
// The floor texture's own values are 20 and 235 (scene/floor.ts). 20 survives
// lighting unchanged and 235 comes back as 188, which is the tone curve
// compressing the highlight -- so these are NOT a linear rescale of the texture
// and cannot be derived from it.
/** A dark cell (torus bit 1) as the app renders it. */
export const DARK_LEVEL = 20;
/** A light cell (torus bit 0) as the app renders it. */
export const LIGHT_LEVEL = 188;
/**
 * Everything that is not board: above the horizon, and the floor plane beyond
 * the board's own edge.
 *
 * The app's scene background is 0x0a0a0f, which is DARKER THAN ANY BOARD CELL.
 * That is load-bearing for decode, whose binarization threshold is the image
 * mean: a frame with much background off the board's edge binarizes at a
 * different point than one filled by board.
 */
export const BACKGROUND = 10.6;

export function vFovRadOf(d: SimDims): number {
  const aspect = d.w / d.h;
  return 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(d.horizFovDeg) / 2) / aspect);
}

/**
 * World position of the camera. Board col maps to world +X and board row to
 * world +Z, which is finishPositionDecode's mapping, not a fresh choice.
 */
export function camPosOf(world: SimWorld, p: SimPose): THREE.Vector3 {
  const { board: { R, C }, cellPitch } = world;
  return new THREE.Vector3(
    (p.overCol + 0.5 - C / 2) * cellPitch,
    p.height,
    (p.overRow + 0.5 - R / 2) * cellPitch,
  );
}

/**
 * World orientation. THREE's camera convention: an unrotated camera looks down
 * -Z with +Y up, so pitch = -90deg points it at the floor and `tiltDeg` lifts
 * it back toward the horizon.
 *
 * 'YXZ' is yaw-pitch-roll, which is what makes these three parameters
 * independent in the way a sweep expects -- yaw spins the camera about the
 * world vertical whatever the tilt is.
 */
export function camQuatOf(p: SimPose): THREE.Quaternion {
  const pitch = -Math.PI / 2 + THREE.MathUtils.degToRad(p.tiltDeg);
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(
    pitch, THREE.MathUtils.degToRad(p.yawDeg), THREE.MathUtils.degToRad(p.rollDeg ?? 0), 'YXZ',
  ));
}

/**
 * `cornerDir` with no allocation, written out into three scalars.
 *
 * ── Why this exists, and why it is not a shortcut ──
 *
 * `cornerDir` returns a fresh THREE.Vector3, which is fine everywhere else in
 * the codebase and ruinous here: a 480x640 frame at 2x2 supersampling is 1.23M
 * rays, and 180 of those frames exhausted the V8 heap outright.
 *
 * Duplicating projection maths is exactly the drift this file's header warns
 * about -- so it is not trusted, it is TESTED. `tests/poseSim.test.ts` asserts
 * this agrees with cornerDir to the last bit across the whole NDC range and a
 * range of orientations. The guarantee is preserved by verification rather than
 * by sharing the call.
 *
 * `out` is caller-owned scratch, reused across every ray.
 */
export function rayDirInto(
  out: { x: number; y: number; z: number },
  ndcU: number, ndcV: number, q: THREE.Quaternion, tanHalf: number, aspect: number,
): void {
  const xc = tanHalf * aspect * ndcU;
  const yc = tanHalf * ndcV;
  const zc = -1;
  // Vector3.normalize() is divideScalar(length()), and divideScalar is
  // multiplyScalar(1 / scalar) -- so the reciprocal is formed once and
  // multiplied, not divided three times.
  const s = 1 / Math.sqrt(xc * xc + yc * yc + zc * zc);
  const vx = xc * s, vy = yc * s, vz = zc * s;
  // Vector3.applyQuaternion's EXACT operation order:
  //   t = 2 * cross(q.xyz, v);  out = v + q.w * t + cross(q.xyz, t)
  //
  // Algebraically this equals the more familiar
  // `v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)`, and writing that form
  // instead made the equivalence test fail -- same value, different bits. The
  // point of the test is that a copy of projection maths cannot be allowed to
  // drift AT ALL, so it matches the original operation for operation.
  const qx = q.x, qy = q.y, qz = q.z, qw = q.w;
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  out.x = vx + qw * tx + qy * tz - qz * ty;
  out.y = vy + qw * ty + qz * tx - qx * tz;
  out.z = vz + qw * tz + qx * ty - qy * tx;
}

/**
 * Renders the De Bruijn floor as this camera would see it.
 *
 * ── SUPERSAMPLING IS LOAD-BEARING, AND 4 IS THE FLOOR ──
 *
 * `supersample` is the linear rate: 4 means 4x4 = 16 samples per pixel.
 *
 * Measured 2026-08-12 at 480x640, height 10, nadir, yaw 35 (so the grid lines
 * run DIAGONALLY across the image):
 *
 *     ss=1     0 votes, no pose at all
 *     ss=2     2 votes, period 3.26x wrong, consistency 0.49 (chance)
 *     ss=4   171 votes, period exact, consistency 1.000
 *     ss=8   171 votes, identical
 *
 * A diagonal edge at 2x2 is a staircase, and the level-line directions it
 * produces are quantized to that staircase rather than to the true edge -- so
 * the directed-growth predicate splits what should be one line at a 9.5deg
 * tolerance. AXIS-ALIGNED edges survive it, which is why a nadir yaw-0 view
 * validates cleanly at ss=2 and hides the problem entirely.
 *
 * This nearly produced a false finding against the pipeline. If a sweep ever
 * reports mass line-detection failure, re-run the pose at ss=8 BEFORE
 * concluding anything about the detector.
 *
 * The floor TILES. The torus is periodic by construction and the decode's
 * anchor arithmetic is mod R/C, so a lattice spanning more than one period is
 * an ordinary case rather than an edge case.
 */
/**
 * The geometric render, at FULL supersampled resolution and undistorted.
 *
 * Split out from `renderPose` so a caller can run the app's own distortion
 * chain -- antialias, lens blur, box downsample, sensor noise -- over the same
 * hi-res image the app blurs, rather than over an already-downsampled one. The
 * app blurs BEFORE discretizing on purpose (physical lens blur acts on a
 * near-continuous image; only the sensor introduces the pixel grid), so doing
 * it in the other order is a different image, not a rounding difference.
 */
export function renderPoseHiRes(
  world: SimWorld, p: SimPose, dims: SimDims, supersample = 4,
): { gray: Float64Array; w: number; h: number } {
  const { board: { R, C, torus }, cellPitch } = world;
  const { w, h } = dims;
  const s = Math.max(1, Math.floor(supersample));
  const hw = w * s, hh = h * s;
  const gray = new Float64Array(hw * hh);
  const aspect = w / h;
  const tanHalf = Math.tan(vFovRadOf(dims) / 2);
  const quat = camQuatOf(p);
  const cam = camPosOf(world, p);
  const dir = { x: 0, y: 0, z: 0 };

  for (let Y = 0; Y < hh; Y++) {
    for (let X = 0; X < hw; X++) {
      // The same sample points renderPose's subsample loop visits: hi-res pixel
      // X maps to full-res px (X + 0.5) / s, which for X = x*s + sx is exactly
      // x + (sx + 0.5) / s.
      const ndcU = ((X + 0.5) / hw) * 2 - 1;
      const ndcV = 1 - ((Y + 0.5) / hh) * 2;
      rayDirInto(dir, ndcU, ndcV, quat, tanHalf, aspect);
      if (dir.y >= -1e-12) { gray[Y * hw + X] = BACKGROUND; continue; }
      const t = -cam.y / dir.y;
      const col = Math.floor((cam.x + t * dir.x) / cellPitch + C / 2);
      const row = Math.floor((cam.z + t * dir.z) / cellPitch + R / 2);
      if (row < 0 || row >= R || col < 0 || col >= C) { gray[Y * hw + X] = BACKGROUND; continue; }
      gray[Y * hw + X] = torus[row]![col]! === 1 ? DARK_LEVEL : LIGHT_LEVEL;
    }
  }
  return { gray, w: hw, h: hh };
}

export function renderPose(world: SimWorld, p: SimPose, dims: SimDims, supersample = 4): Float64Array {
  const { board: { R, C, torus }, cellPitch } = world;
  const { w, h } = dims;
  const gray = new Float64Array(w * h);
  const aspect = w / h;
  const tanHalf = Math.tan(vFovRadOf(dims) / 2);
  const quat = camQuatOf(p);
  const cam = camPosOf(world, p);
  const s = Math.max(1, Math.floor(supersample));
  const inv = 1 / (s * s);
  const dir = { x: 0, y: 0, z: 0 };

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let sy = 0; sy < s; sy++) {
        for (let sx = 0; sx < s; sx++) {
          // Pixel centres at +0.5, subsamples spread evenly inside the pixel.
          const px = x + (sx + 0.5) / s;
          const py = y + (sy + 0.5) / s;
          // NDC, matching decodeGridBuild's inverse mapping exactly:
          // px = ((ndcU + 1) * 0.5) * w  and  py = ((1 - ndcV) * 0.5) * h.
          const ndcU = (px / w) * 2 - 1;
          const ndcV = 1 - (py / h) * 2;
          rayDirInto(dir, ndcU, ndcV, quat, tanHalf, aspect);
          // The floor is y = 0 and the camera is above it, so a ray reaches the
          // floor only while descending.
          if (dir.y >= -1e-12) { acc += BACKGROUND; continue; }
          const t = -cam.y / dir.y;
          const hx = cam.x + t * dir.x;
          const hz = cam.z + t * dir.z;
          const col = Math.floor(hx / cellPitch + C / 2);
          const row = Math.floor(hz / cellPitch + R / 2);
          // ── ONE FINITE BOARD, NOT AN INFINITE TILING ──
          //
          // This used to wrap (`row % R`), painting the pattern repeated
          // forever, on the reasoning that the pattern IS a torus so tiling it
          // is seamless. Seamless it is; PHYSICAL it is not. The app renders a
          // single PlaneGeometry(C, R) with the texture at repeat(1,1), and a
          // printed board has an edge.
          //
          // The difference is not cosmetic and it invalidated real conclusions.
          // A view spanning more than one board period saw the SAME pattern
          // twice, so decode could lock onto either copy and be right both
          // times -- while the sweep scored position against an UNWRAPPED
          // truth and called it an error. That is where "anchor off (144, 1)"
          // came from: 144 is exactly R. Those poses were then read as a
          // regression in the line-join work, which live A/B against the app
          // could not reproduce at any of them.
          if (row < 0 || row >= R || col < 0 || col >= C) { acc += BACKGROUND; continue; }
          // torus 1 is a DARK cell: decodeGridBuild reads a bit as
          // `gray < binThreshold`, so a set bit has to be the dark one.
          acc += torus[row]![col]! === 1 ? DARK_LEVEL : LIGHT_LEVEL;
        }
      }
      gray[y * w + x] = acc * inv;
    }
  }
  return gray;
}

/**
 * Everything the pipeline is supposed to recover, known exactly.
 *
 * ── What is and is not unambiguous ──
 *
 * `camPos`, `height` and `period` are unambiguous and are what a sweep should
 * score on. The AXES are not: the pipeline recovers them only up to the
 * pattern's 4-fold rotational ambiguity (which decode resolves separately) and
 * up to the handedness correction poseCompute applies, so comparing them
 * directly needs to account for both. The floor NORMAL is unambiguous up to
 * sign, which makes it the useful axis-level check.
 */
export interface PoseTruth {
  pose: SimPose;
  camPos: THREE.Vector3;
  camQuat: THREE.Quaternion;
  /** Camera height, which is also recoveredAxes.distance. */
  height: number;
  /**
   * gridPeriodPhase's period. It reports `height = cellPitch / period`, so the
   * true period is the inverse relation -- a dimensionless gnomonic quantity,
   * not pixels.
   */
  period: number;
  /** Board cell directly under the camera, wrapped into the torus. */
  anchorRow: number;
  anchorCol: number;
  /** World floor axes. Board col is +X and board row is +Z. */
  colDirWorld: THREE.Vector3;
  rowDirWorld: THREE.Vector3;
  normalWorld: THREE.Vector3;
  /** The same three in the pipeline's MATH frame, i.e. camera-relative.
   *  NOTE the naming swap: solveRecoveredCamQuat maps the pipeline's `Dcol`
   *  onto world (0,0,1) and its `Drow` onto world (1,0,0). */
  DrowMath: THREE.Vector3;
  DcolMath: THREE.Vector3;
  DnormalMath: THREE.Vector3;
}

export function truthFor(world: SimWorld, p: SimPose): PoseTruth {
  const { board: { R, C }, cellPitch } = world;
  const camPos = camPosOf(world, p);
  const camQuat = camQuatOf(p);
  const invQuat = camQuat.clone().invert();
  const colDirWorld = new THREE.Vector3(1, 0, 0);
  const rowDirWorld = new THREE.Vector3(0, 0, 1);
  const normalWorld = new THREE.Vector3(0, 1, 0);
  return {
    pose: p,
    camPos,
    camQuat,
    height: p.height,
    period: cellPitch / p.height,
    anchorRow: ((Math.floor(p.overRow) % R) + R) % R,
    anchorCol: ((Math.floor(p.overCol) % C) + C) % C,
    colDirWorld, rowDirWorld, normalWorld,
    DrowMath: colDirWorld.clone().applyQuaternion(invQuat),
    DcolMath: rowDirWorld.clone().applyQuaternion(invQuat),
    DnormalMath: normalWorld.clone().applyQuaternion(invQuat),
  };
}
