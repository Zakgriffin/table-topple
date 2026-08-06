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
  // The De Bruijn floor pattern's board size (cells per side) -- mirrors
  // debruijn.ts's ORDER5_CANDIDATE.cropSize, which this now overrides at
  // runtime via scene/floor.ts's rebuildFloorPattern (see the "De Bruijn
  // board size" slider in ui/cameraPanel.ts). Must stay in step with that
  // candidate's cropSize, since this is only the value the slider STARTS at
  // -- 144, so that one pattern cell is one board-game cell (BOARD_CELLS in
  // src/game/constants.ts) and the AR overlay's board lands on the physical
  // one 1:1 rather than at some scale factor.
  boardSize: 144,
  // ── The one GPU/CPU switch, replacing nine per-stage ones (2026-08-05) ──
  //
  // There used to be a `useGPU*` toggle per stage: gradient, growRegions,
  // collectRegions, lsdFit, fit, periodSweep, decode, decodeFused, project. All
  // nine defaulted true. They were the INSTRUMENT that produced every measured
  // number on record, and the answer they produced was "everything is GPU" --
  // at which point a runtime switch for a choice nobody makes at runtime is
  // pure carrying cost. Nine independent booleans is 2^9 of nominal correctness
  // surface, and every new stage doubled it.
  //
  // They also structurally blocked the endgame: you cannot record a fixed
  // command list, or fuse the pipeline into one submit, if any stage might run
  // on the CPU this frame. And they made FieldResidency a transfer-DECISION
  // engine (which side is each stage on, and what must therefore cross?) rather
  // than a plain named-slot arena.
  //
  // What is NOT deleted is the CPU implementations. They earned their keep
  // twice over -- they caught the `layout:'auto'` binding prune that silently
  // no-op'd every submit while returning plausible garbage, and the
  // BOUNDARY_EPS bug that was real on BOTH sides. They are now REFERENCE
  // implementations rather than production branches, reachable through this
  // single switch, and pipelineGPU/lsdChainVerify.ts's two-configuration
  // differential is what keeps them from rotting unnoticed now that production
  // never runs them.
  //
  // Note this is NOT the no-WebGPU fallback and does not need to be: every GPU
  // stage already falls back on its own when there is no device or a dispatch
  // fails validation, and that path is independent of this flag (the flag gates
  // ENTRY, a `return null/false` gates FALLBACK). This exists for harnesses and
  // for answering "is the GPU lying to me" by hand.
  forceCPU: false,
};
