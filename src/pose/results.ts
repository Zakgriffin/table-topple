import * as THREE from 'three';

// ── The pose library's own vocabulary ─────────────────────────────────────
//
// Every type here is either something a stage HANDS BACK (RecoveredAxes,
// PositionDecodeResult, VoteResult) or something one stage hands to the next
// (GradientField, CompositeLine, Vote, DecodeSampleGrid). Nothing in this file
// is about drawing anything.
//
// They lived in sphereLab/types.ts until the library moved to src/pose/, which
// left that file as the single reason the app was still in the pipeline's
// RUNTIME import closure -- it mixed these with the UI's Mode/FieldView
// re-exports, so importing a pose result type meant loading a module that
// exists to serve the app. Splitting the two halves is what takes
// sphereLab/types.ts off tests/libraryBoundary.test.ts's shared-leaf allowlist.
//
// What did NOT come along, deliberately: ProjectedBins and
// ProjectedSamplesDense. The projection stage is DISPLAY (it moved out of the
// pipeline in step 5a because every function in it takes a full Camera and
// reads render resources), so its shapes belong with the app.
//
// THREE stays a dependency of the library -- a decision made on purpose rather
// than inherited, so Vector3/Quaternion below are fine.

export interface Vote { n: THREE.Vector3; weight: number }
export interface GradientField { fx: Float64Array; fy: Float64Array; w: number; h: number; r: number }
// One detected line in pixel space -- what the pose pipeline casts votes from
// and what gridPeriodPhase classifies into row/column families.
//
// It lived in pipeline/bucketFillJoin.ts back when a "composite" genuinely was
// a composite: several raw segments merged into one line by the (since removed)
// join walk. That
// walk is retired (see its header), so today this is exactly ONE accepted LSD
// rectangle's own two endpoints and nothing is composed at all. The name is
// kept because it is threaded through Camera, the dev bridge wire format and
// the phone's own pose path, and renaming it would touch all of those to say
// the same thing.
export interface CompositeLine { x1: number; y1: number; x2: number; y2: number }
// Set by runAxesReconstruction on a successful capture; consumed by
// buildProjectedTexture. distance is the average of the U/V estimates.
export interface RecoveredAxes { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3; distance: number }
export interface PositionDecodeResult {
  row: number; col: number; consistency: number; votes: number; totalWindows: number;
  camPos: THREE.Vector3;
  // The camera's TRUE world orientation, solved entirely from the pattern --
  // see solveRecoveredCamQuat. Anything placed into the actual 3D scene
  // needs this to convert lastRecoveredAxes' Drow/Dcol/Dnormal (expressed in
  // MATH_QUAT's fixed math frame) into true world space first.
  recoveredCamQuat: THREE.Quaternion;
  // Which of the 4 cardinal rotations (see stages/decode/decodeGrid.ts's
  // readRotated/rotateGrid) tallyPositionVotes found the best De Bruijn match
  // at -- display-only consumers (Projected-Cam's "use true cardinal
  // orientation" toggle) use this to rotate what's shown, purely visually;
  // nothing in the actual decode pipeline reads this back.
  orientation: number;
}
// u,v are the sample's world position (relative to camera, in Drow/Dcol
// units); px,py are where that point projects to in the CURRENT capture's
// pixel space, TOP-DOWN row convention. valid is false when the point is
// behind the camera or projects outside the image entirely.
export interface DecodeSamplePoint { u: number; v: number; px: number; py: number; valid: boolean; bit: number }
export interface DecodeSampleGrid { rows: number; cols: number; zeroI: number; zeroJ: number; points: DecodeSamplePoint[][] }
export interface DecodeCellDebug { bit: number; correct: boolean }
export interface VoteResult { orientation: number; anchorRow: number; anchorCol: number; votes: number; totalWindows: number }
