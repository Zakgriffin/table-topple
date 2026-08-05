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
  // Mailbox-style pipelining for a physical camera's video-mode capture
  // stream (see devBridge/client.ts's realCapture handler and main.ts's
  // animate loop): when on, the phone is told it's always ready and free-
  // runs its own capture/encode/send cadence instead of stalling on a full
  // desktop<->phone round trip per frame; the desktop always processes
  // whichever frame is freshest when it becomes free, silently dropping any
  // that arrived and were superseded while it was still busy. Removes the
  // idle round-trip gap between reconstructions (profiled at a little under
  // half of video mode's total per-frame time); off falls back to the
  // original strict one-frame-in-flight handshake.
  useCapturePipelining: true,
  // Defers the reconstruction's DISPLAY tail (projected bins + texture paint,
  // pole markers/floor overlay, the mode-specific overlays) out of the pipeline
  // window and into a one-slot per-camera mailbox drained by animate() -- see
  // pipeline/axesReconstruction.ts's drainVisuals.
  //
  // The tail is idempotent in camera state, not a queue of work items, so the
  // mailbox is a single `visualsDirty` boolean rather than a list: three
  // captures landing before visuals get a turn repaint ONCE, from the newest
  // state, instead of three times. What it buys is that ~20ms of display GPU
  // work (`projectBins` measured at 19.8ms of a 158.9ms reconstruction, and
  // every reader of lastProjectedBins is an overlay/decal/lattice -- nothing on
  // the pose path) stops being AWAITED inside the pose window, where it
  // serialized against the same device queue.
  //
  // Cost, stated plainly: overlays land one animation frame (~16ms) after the
  // pose gizmos rather than in the same frame. Off runs the identical tail
  // inline at the end of recomputeStages, which is the baseline to measure
  // against.
  useDeferredVisuals: true,
};
