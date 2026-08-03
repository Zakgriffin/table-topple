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
  // NOW DEFAULTS TRUE. It was pinned false for a performance reason -- 3.5ms CPU
  // vs 8ms GPU -- that was entirely the mag/theta upload and the hand-packed CSR,
  // both of which FieldResidency removed (pipelineGPU/fieldResidency.ts).
  //
  // Correctness was never the issue and is unchanged: verifyLsdFit reports
  // 1258/1258 accepted, zero accept disagreements, zero n/k count mismatches,
  // max nfaLog10 delta 7.7e-6.
  //
  // The measurement worth remembering is that this flag's cost DEPENDS ON ITS
  // NEIGHBOUR, which is the whole point of the residency. Median of 3
  // interleaved reps, whole computeLsdRectanglesAuto:
  //
  //   grow on CPU:  fit off 30.9ms, fit on 31.1ms  (neutral)
  //   grow on GPU:  fit off 15.8ms, fit on 13.9ms  (a ~1.9ms WIN, 3/3 reps)
  //
  // Same kernel, same work, opposite verdict -- because with the grower on GPU
  // there is no transfer left between the two stages. Turning this on while the
  // grower is on CPU still buys nothing; that is honest and expected, not a
  // regression.
  useGPULsdFit: true,
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
  // Same idea, independent toggle, for gridPeriodPhase.ts's coarse period sweep
  // (see pipelineGPU/periodSweep.ts). An ISLAND, not part of the LSD chain --
  // it's fenced by CPU-only stages on both sides, so it shares no buffers with
  // anything and its transfer cost is a few KB regardless of image size.
  //
  // Defaults TRUE on measurement, not argument: verifyPeriodSweep() on a real
  // capture (286 candidates, 1258 lines) reported identical peak periods,
  // identical argmax, and 14.9ms CPU -> 1.8ms GPU. The sweep is a bigger share
  // of the distance stage than it looks.
  //
  // Not bit-identical -- the fold is evaluated in f32 -- but maxAbsDelta was
  // 3.4e-6 against a 0..2 score scale. The host pre-scales values into [0,1] so
  // the sin/cos argument never leaves [0,2pi) (see the shader's header), which
  // is what keeps it there; ported naively the error would be ~100x worse. The
  // relative delta looks larger (1.4e-4) only because it is dominated by
  // near-zero scores, which are exactly the candidates the SIGNIFICANCE cut
  // throws away.
  useGPUPeriodSweep: true,
  // Hysteresis + CSR build on GPU (pipelineGPU/collectRegions.ts), consuming the
  // label/mag/theta buffers the grower already has on device instead of reading
  // the labeling back. Only meaningful with useGPUGrowRegions on.
  //
  // VERIFIED EXACT (verifyCollectRegions): 1782/1782 regions, 0 member-set, 0
  // member-order and 0 regionId mismatches; maxMeanAngleDelta 3.6e-7, which is
  // just the f32 cos/sin summation.
  //
  // STILL DEFAULT OFF AFTER THE RESIDENCY LANDED, and the prediction that it
  // would flip was WRONG -- worth recording as a measurement rather than
  // quietly re-arguing. It improved (9.7-vs-11.7ms before, 12.8-vs-13.8 in its
  // own harness now) but stayed on the wrong side of zero: ~0.7ms worse through
  // the real dispatch point at both fit settings (15.8 -> 16.5 with fit off,
  // 13.9 -> 14.7 with fit on).
  //
  // The reason is now understood, which is the useful part. Residency could not
  // save the transfer this stage actually pays, because THE MEMBERS HAVE TO LAND
  // ON CPU NO MATTER WHAT: lsdRectanglesToBucketFillShape rebuilds a per-pixel
  // regionId from LsdRectangle.rawMembers to seed the join walk. So GPU collect
  // trades a 786KB label readback for six dispatches plus TWO 4-BYTE COUNT
  // READBACKS (totalRegions/totalMembers, each a full mapAsync round trip) and
  // still reads the members back afterwards. At this scale that latency
  // outweighs the bytes saved -- a bandwidth win losing to a latency cost.
  //
  // What WOULD flip it is removing the members readback, i.e. moving the join
  // walk (or at least its regionId seeding) onto the GPU. Not the CSR handoff,
  // which is already free. Caveat on the numbers: at 3 reps the deltas overlap
  // between configurations, so the direction is consistent but not separated
  // from noise -- re-measure with more reps before treating 0.7ms as exact.
  useGPUCollectRegions: false,
  // Fused decode: buildDecodeSampleGrid AND the tally on GPU, with the packed
  // grid staying device-resident between them (pipelineGPU/decodeGridBuild.ts).
  // Distinct from useGPUDecode, which moves only the tally and still uploads a
  // CPU-built grid -- 298KB per call at a 270x276 lattice.
  //
  // Defaults TRUE on measurement: verifyDecodeGridBuild() on a 187x188 lattice
  // reported 0 valid-flag diffs, 0 bit diffs, an exactly matching winner and
  // identical consistency, at 11-18ms CPU vs 4.2-4.6ms fused GPU (warm; the
  // first call is ~88ms of shader compilation).
  //
  // Observably identical to the CPU route -- lastDecodeGrid/lastDecodeRotated/
  // lastDecodeCorrectness are populated exactly as before. Skipping that grid
  // readback outside Projected-Cam mode was tried and deliberately dropped: it
  // saves only ~1.3ms of the ~4.5ms and would make those fields null for
  // synchronous consumers like mobileCapture's AR readout.
  useGPUDecodeFused: true,
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
