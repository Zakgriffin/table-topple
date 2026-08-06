import { savedControls } from '../ui/dom.ts';
import { FieldView } from '../types.ts';

// bindSlider/bindCheckbox (ui/dom.ts) already WRITE every control's value to
// localStorage on change, keyed by the DOM element's own id -- but per-
// camera settings are otherwise pure hardcoded defaults below, so nothing
// ever reads that back in and a fresh camera (including after a reload,
// since cameras themselves aren't persisted -- see main.ts's header) always
// started from scratch regardless of what was saved. These two helpers read
// a saved value back in, by the SAME id bindSlider/bindCheckbox persist
// under, falling back to the literal default when nothing's been saved yet
// (first-ever load) or the value doesn't parse.
function savedBool(id: string, fallback: boolean): boolean {
  const v = savedControls[id];
  return v === undefined ? fallback : v === '1';
}
function savedNum(id: string, fallback: number): number {
  const v = savedControls[id];
  if (v === undefined) return fallback;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : fallback;
}

// ── Per-camera settings ──────────────────────────────────────────────────
//
// Everything that used to live in the single module-level `state` object,
// split into what's common to both camera types and what's type-specific.
// See createDefaultCameraSettings below for the actual default values
// (mirrors this file's pre-Stage-A `state` initializer exactly).

export interface CameraSettingsCommon {
  showSphere: boolean; showCircles: boolean; showPoles: boolean; showFrustum: boolean; showPatch: boolean;
  showGizmoBody: boolean; showRecoveredFloor: boolean; recoveredFloorOpacity: number;
  showTrueContamination: boolean; showReconstructedContamination: boolean; hideField: boolean;
  showTopGradient: boolean;
  // ── From-scratch traditional LSD pipeline (pipeline/lsdSegments.ts) --
  // this is now the PRODUCTION composite-line source: pipeline/votes.ts's
  // computeGradient2x2Composites feeds pipeline/bucketFillJoin.ts's join
  // walk from lsdRectanglesToBucketFillShape's output, not
  // bucketFillSegments.ts's own BFS growing (still defined, just no longer
  // invoked live). None of these are gated behind a show/hide toggle --
  // they're tuning knobs for a live feature, not just a debug view.
  showLsdSegments: boolean;
  showLsdRejected: boolean; // also draw candidates that failed NFA validation (dashed red), not just accepted rectangles
  // Scatters each drawn rectangle's OWN stage-3 raw region membership
  // (LsdRectangle.rawMembers -- the actual flood-filled pixels, before any
  // retry tightened/shrank it) as small dots, so the raw growing result can
  // be compared directly against the fitted rectangle it produced.
  showLsdRawRegions: boolean;
  showLsdComposite: boolean;
  // The two histograms in the grid period/phase debug plot
  // (overlays/gridPeriodPhaseOverlays.ts). Both are DRAWING-ONLY -- neither
  // feeds the period search, which reads the O(n) line values directly via
  // circularFit. Toggling one off skips computing its data entirely, which
  // matters for the gap histogram: computePooledGaps is O(n^2) in the
  // detected line count and exists only for this plot.
  showGapHistogram: boolean;
  showValueHistogram: boolean;
  // The two debug curves in the period/phase plot, each independently
  // toggleable: cellCentreDistinctness on its own, and its product with the
  // resultant -- the conjunction the search's two-stage filter is
  // approximating. Both are computed in the OVERLAY across the current view
  // range (see overlays/gridPeriodPhaseOverlays.ts), not in the pipeline, so
  // they follow pan/zoom and can show a joint peak sitting outside the
  // bracket the integer-count search ever looked at.
  showDistinctnessCurve: boolean;
  showProductCurve: boolean; // draw the merged composite lines fed by the segments above (the join walk that produced them is deleted; see votes.ts's compositesFromLsdRectangles for what fills this now)
  lsdToleranceDeg: number; // tau -- the one angle tolerance growRegionsCCL's edge predicate, countRectanglePixels' NFA alignment count and the retry-1 refilter all test against. LSD default 22.5deg.
  // rho LOW -- hysteresis' participation floor: a pixel below this is
  // excluded entirely (from growth edges and from NFA alignment counts).
  // Scale is [0,1], matching computeGradient2x2Field's own normalized output.
  lsdRhoNoiseThreshold: number;
  // rho HIGH -- hysteresis' survival bar: a grown component is discarded
  // unless at least one of its members exceeds this. Canny's two-threshold
  // idea, and what replaces the magnitude-priority ORDERING that the old
  // competitive growers relied on to stop weak noise out-competing a real
  // ridge (a symmetric edge predicate has no notion of "wins"). Set at or
  // below lsdRhoNoiseThreshold to degrade to plain single-threshold
  // behavior. See pipeline/lsdSegments.ts's growRegionsCCL.
  lsdRhoHighThreshold: number;
  // DEBUG SCRUBBER ONLY (0 = run to the fixpoint, the real algorithm): caps
  // how many hook+compress rounds growRegionsCCL runs so the overlay can
  // watch components coalesce. Unlike the lsdGrowSteps it replaces, this
  // cannot change the converged answer -- connected components are a
  // fixpoint, not an iteration budget -- only how far along you're looking.
  lsdCclSteps: number;
  // Minimum member count for a grown component to become a region at all --
  // applied in collectRegionsFromLabels, so BOTH the CPU and GPU growers get it
  // and neither ever materializes the region. 2 is the behavior-preserving
  // floor: the rectangle fit needs two members to have an axis at all, so a
  // singleton could only ever be fitted and discarded. Measured on a real
  // capture, that alone is 1149 of 2931 regions -- ~40% of the fit stage's work,
  // for zero change in output. Above 2 it becomes a real (output-changing)
  // tuning knob: a floor on how much evidence a line segment needs.
  lsdMinRegionSize: number;
  lsdNfaEpsilon: number; // epsilon -- accept a candidate rectangle iff NFA < this (LSD default 1: expect <1 false detection per image)
  lsdNfaTestExponent: number; // N_tests = N^exponent, N = max(image w,h) -- LSD's own estimate of "how many rectangles could plausibly have been tested" (~5 degrees of freedom: 2 position, 1 angle, 2 size)
  // ── RETIRED (see pipeline/lsdSegments.ts's fitRegionWithRetries) ─────────
  // Stage 5's retry loop is no longer referenced by any live path -- the fitter
  // is attempt-0-only now. These three stay defined so the retired function
  // still typechecks and so persisted values survive, but nothing reads them
  // and their sliders are disabled.
  lsdMaxRetries: number; // how many tighter-tolerance-then-shrink attempts before giving up on a candidate that fails NFA
  lsdRetryToleranceFactor: number; // tau is multiplied by this on the first retry (LSD: retighten before shrinking)
  lsdRetryShrinkFraction: number; // fraction of a region's own farthest-from-center pixels dropped on each shrink retry
  // The join walk's own 4 parameters (pipeline/bucketFillJoin.ts's
  // computeJoinWalk) -- unchanged in meaning from the old bucketFillJoinSteps/
  // MergeMinSimilarity/MaxTravelFactor/MinLengthPx names, just relocated
  // here since the join walk is now exclusively fed by this LSD pipeline.
  // lsdMergeMinSimilarity is now a FLOOR, not a fixed gate: computeJoinWalk's
  // own requiredSimilarityFor scales the real bar up from this value toward
  // 1 as the two segments/groups' shared lineScore confidence drops, so this
  // is exactly the old fixed threshold only when both sides are maximally
  // confident.
  lsdJoinSteps: number; lsdMergeMinSimilarity: number; lsdMaxTravelFactor: number; lsdMinLengthPx: number;
  // showLevelLineArrow: the gradient rotated -90deg (LSD's own level-line
  // convention, see pipeline/lsdSegments.ts's level-line vector block) -- was named
  // "perpendicular" before, renamed to match that shared terminology.
  showGradientArrow: boolean; showLevelLineArrow: boolean; gradientArrowScale: number;
  coherenceRadius: number;
  tangentWalkMaxSteps: number; tangentWalkDeviationDeg: number; tangentWalkMagFraction: number; tangentWalkGraceSamples: number;
  tangentWalkAdaptive: boolean;
  showRecoveredPoles: boolean;
  showAxisVectors: boolean;
  showTopCircles: boolean;
  topCirclesLineWidth: number;
  weightSharpenPower: number;
  // Orientation-fit source: false (default) is today's composite-line path
  // (computeSegmentVotes -> fitPairOfPlanes), true swaps in
  // pipeline/votes.ts's computePixelVotes2x2 -> refineOrientationIRLS --
  // one vote per pixel straight off the 2x2 gradient field, iteratively
  // reweighted instead of segmented. Orthogonal to gridPeriodPhase, which
  // keeps reading composite lines (see computePoseFromCapture) either way --
  // this only changes which Drow/Dcol/Dnormal it's handed.
  useWorldVoteOrientation: boolean;
  // Cap on refineOrientationIRLS's reweight-and-refit loop -- 0 disables
  // refinement (the loop's own single-shot initial fit only), for direct
  // A/B against fitPairOfPlanes. The loop itself stops early on convergence
  // well before this in the typical case, so this is a worst-case bound,
  // not a fixed per-frame cost -- see refineOrientationIRLS's own comment.
  worldVoteRefineSteps: number;
  // Grazing-angle cutoff (cosine) shared by projectSamplesCPU (forward:
  // screen pixel -> floor point) and buildDecodeSampleGrid (reverse: floor
  // point -> screen pixel) -- see pipeline/decodeGrid.ts's own comment.
  // Higher = stricter (excludes more of the near-horizon view).
  minGrazingCos: number;
  gridPeriodPhaseBinCount: number;
  // Below this, a red/blue neighbor gap (gridPeriodPhaseOverlays.ts's
  // per-family median lines) is excluded from the median -- filters out the
  // same near-duplicate-line noise gaps pipeline/gridPeriodPhase.ts's own
  // seed-period mode search is built to shrug off, which would otherwise
  // drag a small-sample median toward ~0 instead of the true spacing.
  gridPeriodPhaseGapLowerBound: number;
  showCompositeLineFamilies: boolean;
  showSampleLattice: boolean;
  // Purely a display-time rotation of the Projected-Cam view (WebGL texture
  // + debug overlay) by camera.lastPositionDecode.orientation * 90 degrees,
  // so "up" matches the pattern's true cardinal orientation instead of
  // whichever of the 4 the raw sample buffer happened to land in -- doesn't
  // touch the decode pipeline itself, see main.ts's projected-mode branch.
  useTrueCardinalOrientation: boolean;
  fieldView: FieldView;
  axesAutoCapture: boolean; axesCaptureIntervalMs: number;
  viewportW: number; viewportH: number; aspectLocked: boolean;
  // HORIZONTAL field of view, in degrees -- shared by both camera types
  // (see getAnalysisVFovRad, the one place that turns this into the
  // vertical FOV every ray-casting call site actually needs, via whatever
  // the camera's own current aspect ratio is). Used to be simulated-only
  // focalMM, converted through a fixed 36mm "35mm-equivalent" sensor width
  // -- that conversion quietly assumed a 3:2 sensor, so it drifted from
  // what a real lens at that focal length would actually show once the
  // camera's own aspect ratio (viewportW/H) wasn't 3:2, which by default it
  // isn't (512x384 = 4:3). Specifying FOV directly, the same way a
  // physical camera already had to (there's no focal-length spec sheet for
  // a real phone lens to convert from), sidesteps the whole issue and
  // means both camera types now go through the exact same formula.
  horizFovDeg: number;
}
export interface SimulatedCameraSettings extends CameraSettingsCommon {
  camX: number; camY: number; camZ: number;
  camYawDeg: number; camPitchDeg: number;
  simNoise: number; simBlur: number; captureSupersample: number;
}
export interface PhysicalCameraSettings extends CameraSettingsCommon {
}

export function createDefaultCommonSettings(): CameraSettingsCommon {
  return {
    showSphere: true, showCircles: false, showPoles: true, showFrustum: true, showPatch: true, showGizmoBody: true, showRecoveredFloor: true, recoveredFloorOpacity: savedNum('recoveredFloorOpacity', 0.9),
    showTrueContamination: false, showReconstructedContamination: false, hideField: false,
    showTopGradient: false,
    showLsdSegments: savedBool('toggleLsdSegments', false),
    showLsdRejected: savedBool('toggleLsdRejected', false),
    showLsdRawRegions: savedBool('toggleLsdRawRegions', false),
    showLsdComposite: savedBool('toggleLsdComposite', false),
    // All four plot series default OFF. Gap used to default ON (it was
    // unconditional before these toggles existed), but it was turned off in
    // practice and the working default was rebaselined from the live UI on
    // 2026-08-05 -- see this file's header note.
    showGapHistogram: savedBool('toggleGapHistogram', false),
    showValueHistogram: savedBool('toggleValueHistogram', false),
    showDistinctnessCurve: savedBool('toggleDistinctnessCurve', false),
    showProductCurve: savedBool('toggleProductCurve', false),
    lsdToleranceDeg: savedNum('lsdToleranceDeg', 9.5),
    // Rebaselined from the live UI 2026-08-05. Was 4/255 (~0.0157), which
    // preserved the pre-normalization default exactly once
    // computeGradient2x2Field's output started topping out at 1 instead of 255
    // -- that lineage is over; this is a tuned value.
    lsdRhoNoiseThreshold: savedNum('lsdRhoNoiseThreshold', 0.132),
    // ZERO, rebaselined from the live UI 2026-08-05 -- which makes hysteresis
    // DEGENERATE by construction: every pixel clearing the noise floor also
    // clears the high bar, so the survival path never discriminates. Was 12/255
    // (3x the low bar). Worth knowing because it is exactly the condition that
    // made an earlier hysteresis verification VACUOUS (see the perf TODO's
    // cleanup list) -- any check of the high threshold against this default
    // proves nothing.
    lsdRhoHighThreshold: savedNum('lsdRhoHighThreshold', 0),
    lsdCclSteps: savedNum('lsdCclSteps', 0), // 0 = run to fixpoint (the real algorithm); 1+ scrubs rounds
    lsdNfaEpsilon: savedNum('lsdNfaEpsilon', 1),
    lsdNfaTestExponent: savedNum('lsdNfaTestExponent', 5),
    lsdMinRegionSize: savedNum('lsdMinRegionSize', 2),
    lsdMaxRetries: savedNum('lsdMaxRetries', 2),
    lsdRetryToleranceFactor: savedNum('lsdRetryToleranceFactor', 0.5),
    lsdRetryShrinkFraction: savedNum('lsdRetryShrinkFraction', 0.2),
    lsdJoinSteps: savedNum('lsdJoinSteps', 0),
    lsdMergeMinSimilarity: savedNum('lsdMergeMinSimilarity', 0.9),
    lsdMaxTravelFactor: savedNum('lsdMaxTravelFactor', 1),
    lsdMinLengthPx: savedNum('lsdMinLengthPx', 3),
    // 10*255 preserves the pre-normalization arrow length exactly, now that
    // computeGradient2x2Field's own output (the only field this arrow ever
    // draws -- overlays/hoverDebugOverlays.ts derives it per-hover from
    // lastNoisedPreviewGray) tops out at 1 instead of 255.
    showGradientArrow: false, showLevelLineArrow: false, gradientArrowScale: 10 * 255,
    coherenceRadius: 1,
    // See the pre-Stage-A history for the full derivation of these tangent-walk
    // defaults (guided tangent walk, simNoise=8 stability etc.) -- unchanged.
    tangentWalkMaxSteps: 76, tangentWalkDeviationDeg: 45, tangentWalkMagFraction: 0, tangentWalkGraceSamples: 50,
    tangentWalkAdaptive: false,
    showRecoveredPoles: true,
    showAxisVectors: false,
    showTopCircles: true,
    topCirclesLineWidth: savedNum('topCirclesLineWidth', 1),
    weightSharpenPower: 1,
    useWorldVoteOrientation: savedBool('useWorldVoteOrientation', false),
    worldVoteRefineSteps: savedNum('worldVoteRefineSteps', 0),
    minGrazingCos: savedNum('minGrazingCos', 0.1),
    gridPeriodPhaseBinCount: savedNum('gridPeriodPhaseBinCount', 150),
    gridPeriodPhaseGapLowerBound: savedNum('gridPeriodPhaseGapLowerBound', 0.005),
    // Key matches the BUTTON's own id (toggleCompositeLineFamilies), not
    // this field's name -- same persistence convention every other button-
    // driven boolean here uses (e.g. showLsdSegments/toggleLsdSegments).
    showCompositeLineFamilies: savedBool('toggleCompositeLineFamilies', false),
    showSampleLattice: savedBool('showSampleLattice', true),
    useTrueCardinalOrientation: false,
    fieldView: 'noised',
    axesAutoCapture: false, axesCaptureIntervalMs: 500,
    viewportW: 480, viewportH: 640, aspectLocked: false,
    horizFovDeg: 65,
  };
}
export function createDefaultSimulatedSettings(): SimulatedCameraSettings {
  return {
    ...createDefaultCommonSettings(),
    camX: 0, camY: 16.5, camZ: 8,
    camYawDeg: -43, camPitchDeg: -32,
    simNoise: 2, simBlur: 0, captureSupersample: 2,
  };
}
export function createDefaultPhysicalSettings(): PhysicalCameraSettings {
  return {
    ...createDefaultCommonSettings(),
    // Overrides the common default of 'noised' -- that (and antialiased/
    // downsampled, the other simulated-distortion-pipeline stages) don't
    // exist for a real photo, and are hidden from the field-view list
    // entirely for a physical camera (see refreshCameraPanel) -- 'raw'
    // (labeled "capture" in that case) is the only one of the four that
    // still means something.
    fieldView: 'raw',
  };
}
