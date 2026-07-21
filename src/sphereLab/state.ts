import { Mode } from './types.ts';

// ── Global settings ──────────────────────────────────────────────────────
//
// Everything that applies regardless of which camera is active/exists --
// per the N-camera plan's explicit global/per-camera split. Deliberately
// tiny: `mode` because the 3D canvas has exactly one current view regardless
// of camera count/selection, `showFloor`/`floorCellOutlineSubdiv` because
// the floor itself is one shared object, not owned by any camera.
export const globalState = {
  mode: 'world' as Mode,
  showFloor: true,
  floorCellOutlineSubdiv: 0,
  // Manual dev-time switch for the vote-generation pipeline (see
  // pipelineGPU/voteGeneration.ts) -- not auto-detected/fallback yet, per an
  // explicit choice to keep that decision simple while the GPU path is
  // still being trusted. Silently no-ops back to the CPU path if WebGPU
  // isn't available even when this is true (see axesReconstruction.ts).
  useGPUVotes: false,
  // Same idea, independent toggle, for Phase 3's photometric position LM
  // (see pipelineGPU/positionLM.ts) -- kept separate from useGPUVotes so
  // either GPU sub-pipeline can be compared against its CPU counterpart on
  // its own.
  useGPUPositionLM: false,
  // Same idea, independent toggle, for the plane-fit reduction (see
  // pipelineGPU/fitPlanes.ts).
  useGPUFit: false,
  // Same idea, independent toggle, for the decode window-tally histogram
  // (see pipelineGPU/decodeTally.ts).
  useGPUDecode: false,
  // Same idea, independent toggle, for the projected-sample ray-cast (see
  // pipelineGPU/projectSamples.ts) -- only stage 1 of
  // castAndBucketProjectedSamples; the bucket-accumulation stage 2 stays on
  // CPU regardless of this toggle.
  useGPUProject: false,
};
