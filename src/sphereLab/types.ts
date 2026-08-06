import * as THREE from 'three';

export type Mode = 'world' | 'through' | 'inside' | 'projected';
export type FieldView = 'raw' | 'antialiased' | 'downsampled' | 'noised' | 'gradient2x2' | 'gradient2x2Directed' | 'rhoMagnitude' | 'rhoAgreement';

// ── Shared result/field types (referenced by the Camera interfaces below) ─

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
// the same thing. Home moved here so live code does not import a type out of a
// retired module.
export interface CompositeLine { x1: number; y1: number; x2: number; y2: number }
export interface ProjectedBins { minU: number; maxU: number; minV: number; maxV: number; binWidthU: number; binWidthV: number; w: number; h: number }
// Stage-1 output of castAndBucketProjectedSamples (decodeGrid.ts) -- one
// ray-cast+project result per SCREEN pixel, dense (w*h, valid=0 for pixels
// that failed the grazing-angle cutoff) so the CPU and GPU implementations
// of that stage (pipelineGPU/projectSamples.ts) can feed the exact same
// stage-2 bucketing code.
export interface ProjectedSamplesDense {
  u: Float32Array; v: Float32Array; cx: Float32Array; cy: Float32Array; valid: Uint8Array;
  minU: number; maxU: number; minV: number; maxV: number;
}
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
  // Which of the 4 cardinal rotations (see decodeGrid.ts's readRotated/
  // rotateGrid) tallyPositionVotes found the best De Bruijn match at --
  // display-only consumers (Projected-Cam's "use true cardinal orientation"
  // toggle) use this to rotate what's shown, purely visually; nothing in
  // the actual decode pipeline reads this back.
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
export interface OrientationFit { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 }
