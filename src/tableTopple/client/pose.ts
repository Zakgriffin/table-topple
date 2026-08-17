// ── Pose, on this device ─────────────────────────────────────────────────
//
// The WebGPU device, the one PoseContext, and the translation from what the
// library returns into what the overlay's camera needs. This is the module
// phase 6 exists for: it is the first time anything on a phone runs
// `src/pose`.
//
// ── WHY POSE IS COMPUTED HERE AND NEVER ANYWHERE ELSE ──
//
// A settled decision of the four-page restructure: Table Topple clients
// reconstruct locally and the Table Topple server NEVER computes a pose. The
// server has no pose pipeline and no need for WebGPU at all. So there is no
// wire format for a pose in this project, no relay to fall back to, and no
// second implementation to disagree with -- the camera feed goes in on this
// device and a camera placement comes out on this device.
//
// ── LIFECYCLE ──
//
// One device and one context, both for the life of the page, both built from
// the resolution the camera ACTUALLY delivered rather than the one that was
// requested. The context's buffer sizes are baked from those dims, so a
// resolution change means a new context, which is exactly why this page does
// not have a resolution control (see shared/configSchema.ts).

import * as THREE from 'three';
import { getGPUDevice } from '../../gpu/device.ts';
import { boardDims } from '../../pose/board.ts';
import type { PoseResult } from '../../pose/pose.ts';
import {
  type PoseContext, type PoseSettings,
  createPoseContext, destroyPoseContext, runPose,
} from '../../pose/run.ts';
import { ingestGpuFrame } from '../../profiling/clocks.ts';
import { spanEnd, spanStart } from '../../profiling/profiler.ts';
import { config } from '../shared/config.ts';
import { CELL_PITCH, board } from '../shared/floorPattern.ts';
import type { ARCameraPose } from './overlay.ts';

// The same caps the desktop uses. They are overflow bounds, not tuning: going
// over sets a status bit rather than corrupting a result, and at 480x640 a real
// frame is nowhere near either. Written here rather than read from the config
// because they are a property of how much GPU memory a context may claim, which
// is not a thing anyone should be adjusting from a JSON file to fix a decode.
const MAX_REGIONS = 16384;
const MAX_LINES = 16384;

let ctx: PoseContext | null = null;

/**
 * Acquires the device and builds the context for a fixed frame size.
 *
 * Returns false when this browser has no WebGPU -- a normal answer, not a
 * crash. The page still runs: the viewfinder is live and the overlay simply
 * never gets a pose, which is the same visible state as a board the camera
 * cannot decode. Reporting "no WebGPU" is the caller's job (see main.ts's
 * status line), because a blank overlay with no explanation is the failure this
 * project keeps finding.
 */
export async function initPose(w: number, h: number): Promise<boolean> {
  const device = await getGPUDevice();
  if (!device) return false;
  ctx = createPoseContext(device, {
    w, h, maxRegions: MAX_REGIONS, maxLines: MAX_LINES, ...boardDims(board),
  }, board, {});
  return true;
}

export function disposePose(): void {
  if (ctx) destroyPoseContext(ctx);
  ctx = null;
}

export function isPoseReady(): boolean {
  return ctx !== null;
}

// ── The settings the pipeline is handed every frame ──────────────────────
//
// Purely a REGROUPING of table-topple.config.json into the library's per-stage
// shape. No value is invented here and none is defaulted -- that is the whole
// point of the config file, and it is why the field names match the library's
// rather than being translated.
//
// vFovRad is the one DERIVED quantity: a spec sheet quotes horizontal FOV, the
// pipeline works in vertical, and the two are related through the frame's
// aspect. Recomputed per call rather than cached because aspect is a property
// of the frames actually arriving.
function poseSettingsFor(aspect: number): PoseSettings {
  const p = config.pose;
  const vFovRad = verticalFovRad(aspect);
  return {
    grow: { rhoLow: p.lsdRhoNoiseThreshold, toleranceDeg: p.lsdToleranceDeg },
    collect: { rhoHigh: p.lsdRhoHighThreshold, minRegionSize: p.lsdMinRegionSize },
    lsdFit: {
      rho: p.lsdRhoNoiseThreshold, toleranceDeg: p.lsdToleranceDeg,
      nfaTestExponent: p.lsdNfaTestExponent, nfaEpsilon: p.lsdNfaEpsilon,
    },
    lines: { minLengthPx: p.lsdMinLengthPx },
    votes: { vFovRad },
    gpp: { vFovRad, cellPitch: CELL_PITCH, minGrazingCos: p.minGrazingCos },
    layout: { vFovRad, cellPitch: CELL_PITCH, minGrazingCos: p.minGrazingCos },
  };
}

// Written out rather than imported: the identity is two lines of trigonometry,
// and the only existing copies live in Pose Viewer's math/geometry.ts (which
// the isolation rule forbids) and in the pose library's sim.ts (which is the
// simulator, and importing it would drag THREE's renderer-side world and a
// whole dev surface into a client that needs neither).
function verticalFovRad(aspect: number): number {
  const hFovRad = THREE.MathUtils.degToRad(config.pose.horizFovDeg);
  return 2 * Math.atan(Math.tan(hFovRad / 2) / aspect);
}

/**
 * One reconstruction: grayscale in, pose out.
 *
 * Returns null when there is no context, and a PoseResult otherwise -- INCLUDING
 * one whose `ok` is false. A failed decode is a real answer and the caller has
 * to be able to tell it apart from "the pipeline did not run", because those two
 * mean opposite things about whether to trust the last known pose.
 */
export async function recoverPose(gray: Float32Array, aspect: number): Promise<PoseResult | null> {
  if (!ctx) return null;
  // A SPAN, not a stopwatch. This project has exactly one way to record a
  // duration, and the phone getting the same per-pass GPU breakdown the desktop
  // has -- for free, off an instrument that already exists -- is why profiling/
  // was made a neutral platform layer rather than staying inside Pose Viewer.
  const span = spanStart('tableTopple.pose');
  const frame = await runPose(ctx, gray, poseSettingsFor(aspect));
  spanEnd(span);
  // Absent on a device without timestamp-query, which is a normal state and not
  // a degraded one -- the pipeline runs identically and there is simply nothing
  // to anchor.
  if (frame.gpu) ingestGpuFrame(frame.gpu, 'tableTopple.pose');
  return frame.pose;
}

/**
 * The recovered pose, as a camera placement for the overlay.
 *
 * Returns null on a failed decode so the overlay can blank rather than freeze
 * -- see overlay.ts on why a stale pose is the dishonest failure.
 *
 * The position and quaternion pass through UNCHANGED. That is worth stating
 * because it looks too easy: the pipeline reports in cells, this project's
 * CELL_PITCH is 1, and shared/floorPattern.ts crops the De Bruijn board to
 * BOARD_CELLS so one printed cell IS one board-game cell. The overlay's own
 * fitBoardToPattern then scales the scene by a ratio that is 1 in the ordinary
 * case. Nothing here is a coincidence -- it is the same choice made in three
 * places on purpose -- but if any of the three moves, this is a conversion.
 */
export function toCameraPose(pose: PoseResult, aspect: number): ARCameraPose | null {
  if (!pose.ok) return null;
  return {
    camPos: new THREE.Vector3(pose.position.x, pose.position.y, pose.position.z),
    recoveredCamQuat: new THREE.Quaternion(
      pose.quaternion.x, pose.quaternion.y, pose.quaternion.z, pose.quaternion.w,
    ),
    // THREE's camera.fov is VERTICAL degrees, which is why this converts rather
    // than passing horizFovDeg straight through. Handing it the horizontal
    // number would render a too-wide frustum and the board would sit visibly
    // small inside its printed outline -- a near-miss that reads as a
    // calibration error rather than a unit mix-up.
    fovDeg: THREE.MathUtils.radToDeg(verticalFovRad(aspect)),
  };
}
