// ── Pose records, the other half of a phase-A dataset ────────────────────
//
// Split out of the page's one file. One entry per ON-DEVICE reconstruction
// (main.ts's captureComputeAndSendPose), stamped on the SAME clock as imuRing.
// That co-stamping is the entire point of keeping both on the phone: the
// phase-A deliverable is a plot of measured pose against IMU-predicted pose,
// and any clock difference between the two series would show up as a
// velocity-proportional error -- indistinguishable from the prediction being
// wrong, which is the one confusion this whole plan cannot afford.
//
// Records regardless of whether the reconstruction SUCCEEDED. A failed decode
// is not a gap in the data, it is a labelled event, and the fraction of
// frames that fail under motion is itself one of the numbers phase A is
// meant to produce -- dropping failures would silently make the pipeline look
// better the faster the phone moves.
//
// It also owns the recording's VALIDITY state (settingsSyncedAt, knownBoardSize)
// rather than taking it from the page through a host. Those two exist to say
// whether a dump is trustworthy at all -- see the un-synced-defaults trap below
// -- so they belong with the ring they qualify, and main.ts's settingsSync
// handler writes them through noteSettingsSync().

import { motionRateHz, isMotionListening } from '../../capture/motion.ts';
import { nowMs } from '../../clock.ts';
import type { FrameMeta } from '../shared/camera/model.ts';
import { currentFacing, frameDims } from './camera.ts';
import { clearImuRing, imuRing } from './imu.ts';

// ── The un-synced-defaults trap ──────────────────────────────────────────
//
// null until the desktop has pushed a settingsSync. THIS PAGE WILL HAPPILY
// COMPUTE POSES WITHOUT ONE, and silently produce garbage: the De Bruijn
// table is built at module load from ORDER5_CANDIDATE's own cropSize of 256,
// so a board that is actually 128 gets decoded against a completely
// different torus. It does not error -- it returns ~chance-level matches
// that still pass the gate for `positionDecode` being non-null.
//
// This cost a full 60-second dataset on 2026-08-05: 12% "success" rate,
// consistency 0.515, 2 votes of 2924 windows, position jumping hundreds of
// units between consecutive frames -- all of it initially read as a
// motion-blur problem, which it was not. pose-viewer-server.html had simply been
// closed, so main.ts's neverSyncedSettings push never fired.
//
// Surfaced THREE ways on purpose, because a warning nobody is looking at is
// how this happened in the first place: on screen, and recorded into every
// pose record AND the dump header so a dataset carries its own validity and
// the analysis can reject it even if nobody saw the warning.
let syncedAt: number | null = null;

// Tracks the last boardSize a settingsSync actually applied, so main.ts's
// rebuildFloorPatternData (which rebuilds the whole De Bruijn lookup table --
// not cheap) only runs when boardSize genuinely changed, not on every
// unrelated slider nudge riding along in the same message -- mirrors how
// only the desktop's own board-size slider handler calls the THREE-side
// rebuildFloorPattern today (see ui/cameraPanel.ts's 'boardSize' binding).
let knownBoardSize: number | null = null;

/** When the desktop last synced settings down, or null if it never has. */
export function settingsSyncedAt(): number | null {
  return syncedAt;
}

/** What the De Bruijn table was ACTUALLY last built at, or null if never. */
export function syncedBoardSize(): number | null {
  return knownBoardSize;
}

/**
 * A settingsSync landed. `boardSize` is whatever the message carried, or null
 * when it carried none -- passing null leaves the known size alone rather than
 * clearing it, since a message without a board size is not a message saying the
 * board has no size.
 */
export function noteSettingsSync(boardSize: number | null): void {
  if (boardSize !== null) knownBoardSize = boardSize;
  // THE one clock -- the same one imuRing and tDrawn are stamped on. A dump's
  // validity marker being on a different clock from the samples it qualifies
  // is exactly the skew this whole recording is arranged to avoid.
  syncedAt = nowMs();
}

const POSE_RING_CAPACITY = 600;

export interface PoseRecord {
  tDrawn: number;              // nowMs() at the moment the frame was pulled off the video element
  frameMeta: FrameMeta | null;
  // `computeMs` is GONE: a ring of durations
  // alongside the profiler was a second way to measure one, and this one was
  // measuring an empty statement anyway -- see the TODO at its write site. When
  // this page runs a pipeline again, its duration is a SPAN, not a field here.
  ok: boolean;                 // did positionDecode produce anything
  // Whether the desktop had pushed a settingsSync by the time THIS pose was
  // computed. Per-record rather than per-session because a sync can land
  // mid-run, which splits one recording into an invalid prefix and a valid
  // remainder -- a session-level flag would have to call the whole thing one
  // or the other, and would pick wrong either way.
  synced: boolean;
  boardSize: number | null;    // what the De Bruijn table was ACTUALLY built at
  camPos: number[] | null;     // world position, from the De Bruijn decode
  camQuat: number[] | null;
  distance: number | null;
  dnormal: number[] | null;
  consistency: number | null;
  // `votes` and `totalWindows` -- the winning anchor's vote count and how many
  // windows voted at all -- were here and are gone with the display type that
  // carried them (poseViewer/camera/model.ts's PositionDecodeResult). Nothing
  // read them on either side. They are pose-BLOCK quantities in src/pose, so
  // when this page is wired onto that pipeline they come back off the block
  // directly rather than through a display struct.
}

export let poseRing: PoseRecord[] = [];

/**
 * Appends one reconstruction. The two validity fields are filled IN HERE
 * rather than by the caller, so a record can never claim a sync state that
 * disagrees with the module that tracks it.
 */
export function recordPose(r: Omit<PoseRecord, 'synced' | 'boardSize'>): void {
  if (!isMotionListening()) return; // one switch arms the whole recording
  poseRing.push({ ...r, synced: syncedAt !== null, boardSize: knownBoardSize });
  if (poseRing.length > POSE_RING_CAPACITY) poseRing.splice(0, poseRing.length - POSE_RING_CAPACITY);
}

// Pulled with `node scripts/dev-bridge/cli.js eval --phone "dumpRecording()"`.
// Returned as a plain object rather than written to a file because a phone
// browser has nowhere useful to write one -- the bridge IS the filesystem
// here. At 60s of IMU plus a few hundred poses this is a few hundred KB of
// JSON, comfortably inside one websocket message.
//
// `calibration` is computed here rather than offline because it needs the
// AT-REST window to be meaningful and only this side knows what the sensor
// actually reported; it is a summary, not a substitute for the raw arrays,
// both of which are returned.
export function dumpRecording() {
  const dts: number[] = [];
  for (let i = 1; i < imuRing.length; i++) dts.push(imuRing[i].t - imuRing[i - 1].t);
  const sorted = [...dts].sort((a, b) => a - b);
  const dims = frameDims();
  return {
    capturedAt: new Date().toISOString(),
    clock: 'phone performance.timeOrigin + performance.now(), epoch ms, monotonic',
    // Validity FIRST. A dump whose poses were computed before a settingsSync
    // landed is not a weak dataset, it is a wrong one -- the decode ran
    // against a different De Bruijn torus than the physical board.
    settingsSyncedAt: syncedAt, boardSize: knownBoardSize,
    posesBeforeSync: poseRing.filter((p) => !p.synced).length,
    screenAngle: (screen as any).orientation?.angle ?? 0,
    facing: currentFacing,
    video: { w: dims.w, h: dims.h },
    imu: {
      n: imuRing.length,
      spanSec: imuRing.length > 1 ? (imuRing[imuRing.length - 1].t - imuRing[0].t) / 1000 : 0,
      medianDtMs: sorted.length ? sorted[Math.floor(sorted.length / 2)] : null,
      rateHz: motionRateHz(),
      samples: imuRing,
    },
    poses: { n: poseRing.length, records: poseRing },
  };
}

export function clearRecording() {
  clearImuRing();
  poseRing = [];
  return { cleared: true };
}

// Kept on globalThis as well as exported, because the dev-bridge CLI's own
// recipes spell them unqualified -- `eval --phone "dumpRecording()"` -- and
// those are written down in scripts/dev-bridge/feval.sh and in the session
// notes. client/inspect.ts also puts the module in the eval scope as
// `poses`, so both `dumpRecording()` and `poses.dumpRecording()` work.
(globalThis as any).dumpRecording = dumpRecording;
(globalThis as any).clearRecording = clearRecording;
