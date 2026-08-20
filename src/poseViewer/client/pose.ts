// ── Pose, on this device ─────────────────────────────────────────────────
//
// This is what `TODO(phone-on-pose)` was waiting for. Pose Viewer Client used to
// run the WHOLE reconstruction locally through the old pipeline; that pipeline
// was deleted and this page went dark, reporting a failed decode on every
// capture. This module puts it back, on `src/pose`.
//
// ── WHAT IS DIFFERENT FROM TABLE TOPPLE'S OWN pose.ts ────────────────────
//
// The two are deliberately similar in shape and MUST NOT share code -- the
// isolation rule is that the two projects import nothing from one another, and
// the whole point of the four-page restructure was that each project owns its
// own board. So this file reads Pose Viewer's `board` and `GRID_STEP`, and Table
// Topple's reads Table Topple's. Three real differences follow from the page,
// not from taste:
//
//   1. THE RESOLUTION MOVES. This page has a resolution dropdown; Table Topple's
//      client deliberately has none. A PoseContext bakes its buffer sizes from
//      (w,h), so the context has to be rebuilt when the camera settles somewhere
//      new -- see ensurePoseContext, which is the whole reason this file tracks
//      dims at all.
//   2. THE SETTINGS ARRIVE OVER THE WIRE. The desktop pushes a settingsSync down
//      per camera (server/devBridge/client.ts), so the tuning this page
//      reconstructs with is whatever the lab last set, and it can change between
//      frames. They are therefore a PARAMETER of every call rather than read
//      from a config module at import time.
//   3. IT REPORTS AXES, NOT A CAMERA. The pose goes back over the relay as a
//      `poseResult` message whose shape the desktop's ingestRemotePose already
//      parses, so the translation target is that message's `recoveredAxes` +
//      `positionDecode`, not an overlay camera. The overlay takes its camera
//      from the same result, one step later (see overlay.ts).
//
// ── THE BOARD CAN BE REBUILT UNDER US ────────────────────────────────────
//
// `rebuildFloorPatternData` swaps the board when the desktop's board-size slider
// moves, and a context's buffers are sized from the board's dimensions too. So
// the context is keyed on the board identity as well as on (w,h) -- a stale
// context against a new board decodes against a pattern that is no longer on the
// table, which fails as a plausible-looking wrong answer rather than as an error.

import * as THREE from 'three';
import { getGPUDevice } from '../../gpu/device.ts';
import { type Board, boardDims } from '../../pose/board.ts';
import { POSE_STATUS, type PoseResult, decodeLayout } from '../../pose/pose.ts';
import {
  type PoseContext, type PoseFrame, type PoseSettings,
  createPoseContext, destroyPoseContext, runPose,
} from '../../pose/run.ts';
import { ingestGpuFrame } from '../../profiling/clocks.ts';
import { spanEnd, spanStart } from '../../profiling/profiler.ts';
import type { PositionDecodeResult, RecoveredAxes } from '../shared/camera/model.ts';
import { GRID_STEP } from '../shared/constants.ts';
import { board } from '../shared/floorPattern.ts';
import type { PipelineSettings } from '../shared/harness/input.ts';
import { getAnalysisVFovRad } from '../shared/math/geometry.ts';

/**
 * What one reconstruction produces, in the shape this page already speaks.
 *
 * Declared HERE rather than in main.ts, where it used to sit as a placeholder
 * with both fields permanently null. The module that produces a thing is the one
 * that should name it.
 *
 * `null` rather than an absent key on both, because that is what the relay's
 * `poseResult` message and the IMU anchor site already branch on -- the desktop's
 * own CameraPose omits keys instead, and the two conventions meet at the wire
 * (see server/pipeline/capture.ts's ingestRemotePose).
 */
export type LocalPose = {
  recoveredAxes: RecoveredAxes | null;
  positionDecode: PositionDecodeResult | null;
};

// The same caps the desktop uses (server/pipeline/axesReconstruction.ts). They
// are overflow bounds, not tuning: going over sets a status bit rather than
// corrupting a result. Written here rather than read from the config because
// they are a property of how much GPU memory a context may claim.
import { DEFAULT_MAX_LINES, DEFAULT_MAX_REGIONS } from '../../pose/pipeline.ts';
const MAX_REGIONS = DEFAULT_MAX_REGIONS;
const MAX_LINES = DEFAULT_MAX_LINES;

// ── The declared inspect catalogue: exactly two buffers ──────────────────
//
// The desktop declares fifteen because it has fifteen overlays. This page has
// ONE overlay and it needs a camera, not a field -- so it declares only what the
// `recoveredAxes` gate is built from. `triad` is the axis frame and `layout`
// carries the lattice's validity and its distance; the position and quaternion
// ride in the 128-byte pose block itself and cost nothing extra.
//
// This is the cheap end of the mechanism on purpose: the catalogue sizes the
// staging buffer, and a phone reconstructing continuously should not be pulling
// 1.17 MiB rasters back across the bus for nobody. Adding a name here is what
// unlocks an overlay that wants a field -- see project_pose_display_wiring's
// recipe -- and it costs a context rebuild, which is why it is not per-frame.
const INSPECT = ['triad', 'layout'] as const;

// ── Device + context lifecycle ───────────────────────────────────────────

let ctx: PoseContext | null = null;
let ctxW = 0, ctxH = 0;
let ctxBoard: Board | null = null;

// The PROMISE, not the device: `ensurePoseContext` is called from a self-paced
// loop and from a shutter tap, so two calls can overlap before the first
// resolves. Holding the promise makes the second one wait for the first rather
// than requesting a second adapter.
let devicePromise: Promise<GPUDevice | null> | null = null;

/**
 * Acquires the device (once) and builds a context for this frame size.
 *
 * Returns false when this browser has no WebGPU -- a normal answer on iOS
 * Safari, not a crash. The page still captures and relays exactly as it did
 * before; it simply cannot compute a pose locally, and main.ts says so rather
 * than leaving "compute on this device" silently doing nothing.
 *
 * Rebuilds whenever (w,h) or the board changed, and is a no-op otherwise, so it
 * is safe to call on every frame -- which is what the loop does, because the
 * resolution can change out from under it at any time.
 */
export async function ensurePoseContext(w: number, h: number): Promise<boolean> {
  devicePromise ??= getGPUDevice();
  const device = await devicePromise;
  if (!device) return false;
  if (ctx && ctxW === w && ctxH === h && ctxBoard === board) return true;
  if (ctx) { destroyPoseContext(ctx); ctx = null; }
  ctx = createPoseContext(
    device,
    { w, h, maxRegions: MAX_REGIONS, maxLines: MAX_LINES, ...boardDims(board) },
    board,
    { inspect: INSPECT },
  );
  ctxW = w; ctxH = h; ctxBoard = board;
  return true;
}

export function isPoseReady(): boolean {
  return ctx !== null;
}

/** What the live context is sized for -- `0x0` before the first build. Read by
 *  the readout, so a resolution change that failed to rebuild is visible rather
 *  than showing up later as a length mismatch. */
export function poseContextDims(): { w: number; h: number } {
  return { w: ctxW, h: ctxH };
}

export function disposePose(): void {
  if (ctx) destroyPoseContext(ctx);
  ctx = null;
  ctxW = 0; ctxH = 0; ctxBoard = null;
}

// ── The settings the pipeline is handed every frame ──────────────────────
//
// A pure REGROUPING of this page's live PipelineSettings into the library's
// per-stage shape -- the same mapping the desktop makes in
// server/pipeline/axesReconstruction.ts, against the same field names, because
// both are driven by the same synced `camera.common` block. No value is invented
// and none is defaulted.
//
// `vFovRad` is the one derived quantity, and it goes through shared/math/
// geometry.ts's getAnalysisVFovRad rather than being re-derived here: that
// function is the single source of truth the desktop's analysis path already
// uses, and this page has to agree with it exactly or a pose computed here and
// the same frame replayed on the desktop would disagree by a lens.
function poseSettingsFor(aspect: number, s: PipelineSettings): PoseSettings {
  const vFovRad = getAnalysisVFovRad({ aspect, settings: s });
  return {
    grow: { rhoLow: s.lsdRhoNoiseThreshold, toleranceDeg: s.lsdToleranceDeg },
    collect: { rhoHigh: s.lsdRhoHighThreshold, minRegionSize: s.lsdMinRegionSize },
    lsdFit: {
      rho: s.lsdRhoNoiseThreshold, toleranceDeg: s.lsdToleranceDeg,
      nfaTestExponent: s.lsdNfaTestExponent, nfaEpsilon: s.lsdNfaEpsilon,
    },
    lines: { minLengthPx: s.lsdMinLengthPx },
    votes: { vFovRad },
    // Join DISABLED, pending measurement and config wiring. kSigma 0 means no
    // pair can pass the gate, so every line becomes its own composite and the
    // pose is bit-for-bit the pre-join one. The three knobs have no entry in
    // pose-viewer.config.json yet; see scripts/sweep.ts, which drives them from
    // the command line while they are being measured.
    join: { vFovRad, endpointNoisePx: 0.15, kSigma: 0, maxAngleDeg: 0.2, maxResidualPx: 2, polarityAbs: false },
    gpp: { vFovRad, cellPitch: GRID_STEP, minGrazingCos: s.minGrazingCos },
    layout: { vFovRad, cellPitch: GRID_STEP, minGrazingCos: s.minGrazingCos },
  };
}

/**
 * One reconstruction: grayscale in, pose out.
 *
 * `gray` is one f32 per pixel, 0..255, TOP-DOWN -- which is the convention the
 * whole project settled on and the orientation `runPose` validates against. The
 * caller gets it from `toGrayscaleF32` over `getImageData`, which is natively
 * top-down, and applies NO flip. See main.ts's capture site: a single flip there
 * was once confirmed live to be the root cause of an on-device Drow/Dcol axis
 * swap, and it is the easiest mistake to make in this file's vicinity.
 *
 * Returns null only when there is no context. A FAILED DECODE IS NOT NULL -- it
 * is a real result whose `positionDecode` is null, and the caller has to be able
 * to tell those apart, because they mean opposite things about whether the
 * pipeline ran.
 */
export async function recoverPose(
  gray: Float32Array, w: number, h: number, settings: PipelineSettings,
): Promise<{ result: PoseResult; local: LocalPose } | null> {
  if (!ctx) return null;
  // A SPAN, not a stopwatch. This project has exactly one way to record a
  // duration -- and this is the span the TODO at the old capture site asked for.
  // The stopwatch that used to live there was deleted rather than converted
  // because it was timing an empty statement; there is something to time now.
  const span = spanStart('poseViewer.client.pose');
  const frame = await runPose(ctx, gray, poseSettingsFor(w / h, settings), INSPECT);
  spanEnd(span);
  // The phone gets the same per-pass GPU breakdown the desktop has, for free,
  // off an instrument that already exists -- absent on a device without
  // `timestamp-query`, which is a normal state and not a degraded one.
  if (frame.gpu) ingestGpuFrame(frame.gpu, 'poseViewer.client.pose');
  return { result: frame.pose, local: toLocalPose(frame) };
}

// ── The library's two shapes, as this page's one ─────────────────────────
//
// Deliberately the same two success gates the desktop applies, spelled the same
// way (server/pipeline/axesReconstruction.ts's toCameraPose):
//
//   recoveredAxes    a triad, and a lattice to scale it   !fitDegenerate && layout.valid
//   positionDecode   ...and an anchor in the board        pose.ok
//
// They are NOT derived from one another. A frame can have a good axis frame and
// no position -- that is an ordinary outcome at a grazing angle, not an error --
// and collapsing the two into one "did it work" flag is exactly what the pose
// library's §15 warns against.
function toLocalPose(frame: PoseFrame): LocalPose {
  const { pose, inspected } = frame;
  // Three vec3<f32> at STRIDE 16 -- the stride is why this is `i * 4` and not
  // `i * 3`, and getting it wrong reads Drow.w as Dcol.x. Same trap, same
  // arithmetic, as the desktop's own unpack.
  const t = new Float32Array(inspected['triad']!);
  const axis = (i: number) => new THREE.Vector3(t[i * 4]!, t[i * 4 + 1]!, t[i * 4 + 2]!);
  const layout = decodeLayout(inspected['layout']!);
  const fitOk = (pose.status & POSE_STATUS.fitDegenerate) === 0;
  return {
    recoveredAxes: fitOk && layout.valid === 1
      // `distance` IS the camera's height above the floor, read off the same
      // block that carries the axes it belongs with, so the two cannot be paired
      // across a frame.
      // tanHalf/aspect ride along for the same reason: they are the projection
      // the DEVICE used, and a consumer that recomputed them from its own
      // viewport would be describing a different camera.
      ? {
        Drow: axis(0), Dcol: axis(1), Dnormal: axis(2), distance: layout.distance,
        tanHalf: layout.tanHalf, aspect: layout.aspect,
      }
      : null,
    positionDecode: pose.ok
      ? {
        row: pose.boardRow, col: pose.boardCol, consistency: pose.consistency,
        camPos: new THREE.Vector3(pose.position.x, pose.position.y, pose.position.z),
        recoveredCamQuat: new THREE.Quaternion(
          pose.quaternion.x, pose.quaternion.y, pose.quaternion.z, pose.quaternion.w,
        ),
        orientation: pose.orientation,
      }
      : null,
  };
}

/**
 * The recovered pose, as a camera placement for the AR overlay.
 *
 * Returns null on a failed decode so the overlay can blank rather than freeze --
 * see overlay.ts on why a stale overlay is the dishonest failure.
 *
 * THREE's `camera.fov` is VERTICAL degrees, which is why this converts rather
 * than passing `horizFovDeg` straight through: handing it the horizontal number
 * renders a too-wide frustum and the cubes sit visibly small inside the cells
 * they should cover -- a near-miss that reads as a calibration error rather than
 * as a unit mix-up.
 */
export function toOverlayCamera(
  local: LocalPose, aspect: number, settings: PipelineSettings,
): { camPos: THREE.Vector3; camQuat: THREE.Quaternion; aspect: number; fovDeg: number } | null {
  const pd = local.positionDecode;
  if (!pd) return null;
  return {
    camPos: pd.camPos,
    camQuat: pd.recoveredCamQuat,
    aspect,
    fovDeg: THREE.MathUtils.radToDeg(getAnalysisVFovRad({ aspect, settings })),
  };
}
