import { FieldView } from '../types.ts';

// ── Per-camera settings: the SHAPE only ──────────────────────────────────
//
// Everything that used to live in the single module-level `state` object,
// split into what's common to both camera types and what's type-specific.
//
// There are no default VALUES here anymore -- every one of them lives in
// sphere-lab.config.json, and config.ts is what turns that file into a
// settings object (createDefaultSimulatedSettings/createDefaultPhysicalSettings
// moved there with them). This file deliberately imports nothing but types, so
// that config.ts can import it without a cycle.

export interface CameraSettingsCommon {
  showSphere: boolean; showCircles: boolean; showPoles: boolean; showFrustum: boolean; showPatch: boolean;
  showGizmoBody: boolean; showRecoveredFloor: boolean; recoveredFloorOpacity: number;
  showTrueContamination: boolean; showReconstructedContamination: boolean;
  hideField: boolean;
  showTopGradient: boolean;
  // ── From-scratch traditional LSD pipeline (pipeline/lsdSegments.ts) --
  // the PRODUCTION composite-line source: pipeline/votes.ts's
  // computeGradient2x2Composites turns each accepted rectangle straight into
  // one line. None of these are gated behind a show/hide toggle --
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
  // The last survivor of the join walk's four parameters -- the other three
  // (join steps, merge similarity, max travel) went with the walk itself.
  //
  // NOT the same filter as lsdMinRegionSize, and the two are easy to confuse:
  // minRegionSize counts PIXELS in a connected component and runs BEFORE any
  // rectangle is fitted; this compares the FITTED rectangle's long-axis extent
  // and runs after NFA acceptance, in compositesFromLsdRectangles. A fat
  // 8-pixel blob passes the first and fails this one.
  lsdMinLengthPx: number;
  // showLevelLineArrow: the gradient rotated -90deg (LSD's own level-line
  // convention, see pipeline/lsdSegments.ts's level-line vector block) -- was named
  // "perpendicular" before, renamed to match that shared terminology.
  showGradientArrow: boolean; showLevelLineArrow: boolean; gradientArrowScale: number;
  tangentWalkMaxSteps: number; tangentWalkDeviationDeg: number; tangentWalkMagFraction: number; tangentWalkGraceSamples: number;
  tangentWalkAdaptive: boolean;
  showRecoveredPoles: boolean;
  showAxisVectors: boolean;
  showTopCircles: boolean;
  topCirclesLineWidth: number;
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


// ── Key manifests ────────────────────────────────────────────────────────
//
// The runtime spelling of the two interfaces above, which TypeScript's own
// types cannot provide (they are erased). config.ts needs them twice over: to
// VALIDATE that sphere-lab.config.json actually carries every setting -- there
// is no literal fallback left anywhere, so a missing key has to be a loud
// startup failure rather than a silent undefined -- and to COPY a camera's
// settings back into the config object when one changes.
//
// `satisfies Record<keyof T, true>` is what keeps these honest: adding a field
// to either interface without adding it here is a compile error, not a
// setting that silently stops being saved.
export const COMMON_KEYS = {
  showSphere: true, showCircles: true, showPoles: true, showFrustum: true, showPatch: true,
  showGizmoBody: true, showRecoveredFloor: true, recoveredFloorOpacity: true,
  showTrueContamination: true, showReconstructedContamination: true,
  hideField: true, showTopGradient: true,
  showLsdSegments: true, showLsdRejected: true, showLsdRawRegions: true, showLsdComposite: true,
  showGapHistogram: true, showValueHistogram: true, showDistinctnessCurve: true, showProductCurve: true,
  lsdToleranceDeg: true, lsdRhoNoiseThreshold: true, lsdRhoHighThreshold: true, lsdCclSteps: true,
  lsdNfaEpsilon: true, lsdNfaTestExponent: true, lsdMinRegionSize: true, lsdMinLengthPx: true,
  showGradientArrow: true, showLevelLineArrow: true, gradientArrowScale: true,
  tangentWalkMaxSteps: true, tangentWalkDeviationDeg: true, tangentWalkMagFraction: true,
  tangentWalkGraceSamples: true, tangentWalkAdaptive: true,
  showRecoveredPoles: true, showAxisVectors: true, showTopCircles: true, topCirclesLineWidth: true,
  minGrazingCos: true, gridPeriodPhaseBinCount: true, gridPeriodPhaseGapLowerBound: true,
  showCompositeLineFamilies: true, showSampleLattice: true,
  useTrueCardinalOrientation: true, fieldView: true,
  axesAutoCapture: true, axesCaptureIntervalMs: true,
  viewportW: true, viewportH: true, aspectLocked: true, horizFovDeg: true,
} satisfies Record<keyof CameraSettingsCommon, true>;

// Only the fields a simulated camera adds -- the common ones are merged in
// separately, so listing them again here would be a second place to forget.
export const SIM_ONLY_KEYS = {
  camX: true, camY: true, camZ: true, camYawDeg: true, camPitchDeg: true,
  simNoise: true, simBlur: true, captureSupersample: true,
} satisfies Record<Exclude<keyof SimulatedCameraSettings, keyof CameraSettingsCommon>, true>;

export type SimulatedOnlySettings = Pick<SimulatedCameraSettings, keyof typeof SIM_ONLY_KEYS>;
