import * as THREE from 'three';

export type Mode = 'world' | 'through' | 'inside' | 'projected';
export type FieldView = 'raw' | 'antialiased' | 'downsampled' | 'noised' | 'triangleFold' | 'gradient' | 'gradient2x2' | 'walked' | 'agreement' | 'effective';

// ── Shared result/field types (referenced by the Camera interfaces below) ─

export interface Vote { n: THREE.Vector3; weight: number }
export interface GradientField { fx: Float64Array; fy: Float64Array; w: number; h: number; r: number }
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
export interface Marginals {
  colSum: Float64Array; rowSum: Float64Array; colSumCy: Float64Array; rowHueCx: Float64Array; rowSumCy: Float64Array;
  colMag: Float64Array; rowMag: Float64Array;
  colPeriod: number | null; rowPeriod: number | null; colPhase: number; rowPhase: number;
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
export interface PositionFit extends OrientationFit { worldX0: number; worldZ0: number; distance: number }
export interface PhotometricSample { px: number; py: number; observed: number }
// Per-pixel local Jacobian of the gradient FIELD (i.e. a discrete Hessian of
// the image), split into its symmetric part's closed-form eigen-decomposition
// -- e1/lambda1 is the dominant (larger |eigenvalue|) eigenvector, which for
// a clean single edge sits almost exactly along the gradient; e2/lambda2 is
// the subordinate one, almost exactly along the edge tangent -- plus the
// antisymmetric part's scalar (asym), which is exactly zero for an ideal
// edge and grows where the local 1-edge model breaks down (corners/
// junctions), independent of the eigenvalue-ratio signal. See
// pipeline/localJacobian.ts for the derivation.
export interface JacobianField {
  e1x: Float64Array; e1y: Float64Array; lambda1: Float64Array;
  e2x: Float64Array; e2y: Float64Array; lambda2: Float64Array;
  asym: Float64Array;
  w: number; h: number; r: number;
}
