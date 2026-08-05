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
  // pipelineGPU/gradient2x2.ts). This is stage 1 of the LSD chain and shares
  // the chain's FieldResidency, so its cost depends on its NEIGHBOUR just like
  // the rest: on its own it uploads gray and reads fx/fy straight back, but
  // with useGPUGrowRegions on, fx/fy never cross at all and the gray upload is
  // the only traffic left at the front of the pipeline.
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
  // FLIPPED ON 2026-08-03, after stage 1 joined the chain. The history is worth
  // keeping, because this toggle reversed TWICE and the reason it finally moved
  // is not the reason anyone predicted.
  //
  // It was OFF through the residency work, at ~0.7ms worse, and the diagnosis
  // was that residency could not save the transfer it actually pays: the members
  // had to land on CPU no matter what (lsdRectanglesToBucketFillShape rebuilt
  // a per-pixel regionId from LsdRectangle.rawMembers to seed the join walk), so
  // GPU collect traded a 786KB label readback for six dispatches plus two
  // 4-byte count readbacks and still read the members back afterwards -- a
  // bandwidth win losing to a latency cost. That diagnosis was correct and the
  // conclusion drawn from it ("only moving the join walk would flip this") was
  // too narrow.
  //
  // UPDATE 2026-08-05: the join walk IS gone now, and that half of the
  // diagnosis resolved -- rawMembers is no longer required by the pose path.
  // It is still read every frame though, because the raw-region/rejected debug
  // overlays consume it, so nothing has actually changed for this flag yet.
  // What would change it is deferring that readback off the pose path
  // (perf TODO item 4), not the join walk removal on its own.
  //
  // What flipped it: the CPU collect needs fx/fy. Once stage 1 could keep them
  // on the device, leaving collect on CPU stopped costing one readback and
  // started costing THREE -- label, fx and fy. So the comparison is no longer
  // six dispatches vs 786KB, it is six dispatches vs 2.3MB, and the dispatches
  // win. Measured by verifyLsdChain(cam, 5) at 512x384, median of 5 interleaved
  // reps with the page verified focused throughout, full chain, gradient and
  // grow and fit all on GPU:
  //   collect CPU: 5 crossings, 3.17MB, 12.2ms
  //   collect GPU: 2 crossings, 0.92MB, 10.8ms  <- the chain's transfer floor
  // and with the fitter on CPU, 15.5 -> 13.9ms. The members readback is still
  // there and still unavoidable; it is simply no longer the only thing crossing.
  //
  // Honest caveat on the win, because the transfer floor is NOT the fastest
  // configuration: gradient=CPU with collect on measures 10.1ms, a hair faster
  // than the 2-crossing 10.8ms. Once the grower is on GPU the top four
  // configurations sit within 10.1-10.8ms and crossings stop predicting time --
  // this toggle is chosen on bytes and on being the honest floor, not on a
  // wall-clock difference that survives scrutiny at this resolution.
  //
  // VERIFIED EXACT (verifyCollectRegions): 1782/1782 regions, 0 member-set, 0
  // member-order and 0 regionId mismatches; maxMeanAngleDelta 3.6e-7, which is
  // just the f32 cos/sin summation. verifyLsdChain's 12-configuration sweep
  // additionally shows 0 delta in rectangle count, accepted count and member
  // count against the all-CPU baseline.
  useGPUCollectRegions: true,
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
