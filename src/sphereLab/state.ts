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
  // board size" slider in ui/cameraPanel.ts). 256 matches that candidate's
  // original fixed cropSize, i.e. today's actual default board.
  boardSize: 256,
  // Manual dev-time switch for the plane-fit reduction (see
  // pipelineGPU/fitPlanes.ts) -- not auto-detected/fallback yet, per an
  // explicit choice to keep that decision simple while the GPU path is
  // still being trusted. Silently no-ops back to the CPU path if WebGPU
  // isn't available even when this is true (see axesReconstruction.ts).
  // Defaults to true (all useGPU* toggles do) -- each one has an
  // independently-verified CPU fallback, so the only downside of leaving
  // one on is a silent no-op on a machine without WebGPU.
  useGPUFit: true,
  // Same idea, independent toggle, for the decode window-tally histogram
  // (see pipelineGPU/decodeTally.ts).
  useGPUDecode: true,
  // Same idea, independent toggle, for the projected-sample ray-cast (see
  // pipelineGPU/projectSamples.ts) -- only stage 1 of
  // castAndBucketProjectedSamples; the bucket-accumulation stage 2 stays on
  // CPU regardless of this toggle.
  useGPUProject: true,
  // Same idea, independent toggle, for the 2x2 forward-difference gradient
  // field that feeds computeGradient2x2Composites (see
  // pipelineGPU/gradient2x2.ts).
  useGPUGradient: true,
  // Same idea, independent toggle, for the LSD pipeline's stage 4 (PCA
  // rectangle fit) + stage 5's first NFA pass (see pipelineGPU/lsdFit.ts) --
  // the retry loop and everything before stage 4 (region growing etc.) stay
  // CPU-only regardless, see pipeline/lsdSegments.ts's own header.
  //
  // PINNED false (checkbox disabled in sphere-lab.html too, see
  // cameraPanel.ts's own comment). The parity problem that pinned it is gone
  // -- lsdFit.wgsl.ts's directed NFA alignment count now MATCHES the CPU
  // path's countRectanglePixels again, since that reverted to directed too
  // (pipeline/lsdSegments.ts). What's still missing is an actual verification
  // run against the CPU path; do that before flipping this back on.
  useGPULsdFit: false,
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
};
