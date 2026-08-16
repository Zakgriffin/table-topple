// ── The app's shared types ────────────────────────────────────────────────
//
// This file's header used to say the pose pipeline's vocabulary was NOT here,
// it was in the old pipeline's results.ts, and that keeping the two apart was what let the
// library stay out of the app's import closure. the old pipeline is DELETED, so there is
// no library on the other side of that boundary any more and the split has
// nothing left to protect.
//
// What came back here is only what the app computes and draws for ITSELF. The
// deleted pipeline's result vocabulary -- Vote, RecoveredAxes,
// PositionDecodeResult, DecodeSampleGrid, LsdRectangle, GrownRegion -- did NOT
// come with it. Those described a pose pipeline that no longer exists, and
// src/pose hands back a plain struct instead, so re-homing them would have kept
// the old shape alive under a new path.

// Both of these are DERIVED from the config schemas rather than declared
// here, and re-exported so the dozen existing importers keep their import
// path. Each used to be a hand-written union that had to be kept in step with
// the schema (and with sphere-lab.html's radio/button lists) by eye; now a
// config file naming a field view or a mode that does not exist is a
// validation error rather than a silently accepted string.
export type { Mode } from './configSchema.ts';
export type { FieldView } from './camera/settings.ts';

// ── The projection stage's shapes ─────────────────────────────────────────
//
// These stayed behind when the rest moved to the old pipeline's results.ts, because the
// projection stage is DISPLAY: step 5a moved it out of the pipeline on the
// grounds that every function in it takes a full Camera and reads render
// resources. Produced by pipeline/projectedBins.ts and
// pipeline/projectSamples.gpu.ts; the only other reader is Camera itself.

export interface ProjectedBins { minU: number; maxU: number; minV: number; maxV: number; binWidthU: number; binWidthV: number; w: number; h: number }
// Stage-1 output of projectSamplesCPU / projectSamplesGPU -- one
// ray-cast+project result per SCREEN pixel, dense (w*h, valid=0 for pixels
// that failed the grazing-angle cutoff) so the two implementations
// (pipeline/projectedBins.ts and pipeline/projectSamples.gpu.ts) can feed the
// exact same stage-2 bucketing code.
export interface ProjectedSamplesDense {
  u: Float32Array; v: Float32Array; cx: Float32Array; cy: Float32Array; valid: Uint8Array;
  minU: number; maxU: number; minV: number; maxV: number;
}

// ── The gradient field ────────────────────────────────────────────────────
//
// An image gradient is not pose vocabulary -- it is a property of a picture --
// and the app paints it for its own display modes (pipeline/fieldPaint.ts,
// contamination.ts, gradientHighlight.ts, overlays/pipelineField.ts). So it
// lives here now rather than dying with the pipeline that also happened to
// consume it.
//
// fx/fy are the components at radius r, in a w x h grid.
//
// EITHER WIDTH, and that is not laziness. The app's own gradients
// (pipeline/gradientField.ts, computed from a preview image) are f64; the pose
// run's are f32, because src/pose works in f32 on the device and hands its
// buffers back as raw bytes. Every consumer here only INDEXES these -- nothing
// writes through the interface -- so widening the type is the whole change, and
// the alternative is converting 307k values twice a frame to satisfy a
// declaration. That is the same narrowing loop §4 deleted from the input side.
export type FloatField = Float64Array | Float32Array;
export interface GradientField { fx: FloatField; fy: FloatField; w: number; h: number; r: number }
