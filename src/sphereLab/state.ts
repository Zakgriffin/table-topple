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
  // Default false, and now for a PERFORMANCE reason rather than a correctness
  // one. The GPU path is verified bit-for-decision against the CPU path (see
  // pipelineGPU/lsdFitVerify.ts and lsdFit.ts's own header) -- but it is
  // currently SLOWER (3.5ms CPU vs 8ms GPU on a 2931-region capture), because
  // fitAndTestRegionsGPU still has to upload mag/theta and the CSR member
  // arrays from CPU every call. Toggleable for A/B; worth defaulting on once
  // growRegionsCCL is GPU-resident and that upload disappears.
  useGPULsdFit: false,
  // Same idea, independent toggle, for the LSD pipeline's stage 2+3 (directed
  // connected-component region growing, see pipelineGPU/growRegions.ts).
  //
  // Defaults TRUE like the rest, on measured evidence: verifyGrowRegions()
  // reported 2931/2931 exactly-matching regions, identical size distribution,
  // meanAngle delta 0, and 35.9ms CPU -> 15.3ms GPU on a real capture.
  //
  // Worth knowing WHY that was exact rather than merely close, because it is
  // capture-dependent. This is the one GPU stage that cannot be bit-identical in
  // principle -- the edge predicate is evaluated in f32 here and f64 on CPU, so
  // a neighbour pair within ~1e-7 of exactly the tolerance can fall on opposite
  // sides, and one flipped edge merges or splits a whole component rather than
  // perturbing a number. The harness's `borderlinePairs` counts pairs actually
  // sitting in that window; it measured 0, so there was nothing exposed. A
  // different lsdToleranceDeg could put pairs in it, which is what the harness
  // is for -- re-run it after changing tau if exactness matters.
  useGPUGrowRegions: true,
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
