// ── The app's shared types ────────────────────────────────────────────────
//
// The pose pipeline's own vocabulary is NOT here -- it is in
// src/pose/results.ts. This file used to hold both, which is the only reason
// the library still loaded a module out of the app it is supposed to be
// independent of; see that file's header and tests/libraryBoundary.test.ts.

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
// These stayed behind when the rest moved to src/pose/results.ts, because the
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
