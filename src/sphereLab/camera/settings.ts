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
  showLsdComposite: boolean; // draw the join walk's own merged composite lines (pipeline/bucketFillJoin.ts's computeCompositeLines), fed by the segments above
  lsdToleranceDeg: number; // tau -- region-growing angle tolerance, LSD default 22.5deg
  lsdRhoNoiseThreshold: number; // rho -- gradient magnitude floor below which a pixel is excluded entirely (not just from growth -- from seeding and NFA counts too)
  lsdMagnitudeBuckets: number; // approximate bucket/counting-sort resolution for stage 2's magnitude-descending pixel ordering
  lsdNfaEpsilon: number; // epsilon -- accept a candidate rectangle iff NFA < this (LSD default 1: expect <1 false detection per image)
  lsdNfaTestExponent: number; // N_tests = N^exponent, N = max(image w,h) -- LSD's own estimate of "how many rectangles could plausibly have been tested" (~5 degrees of freedom: 2 position, 1 angle, 2 size)
  lsdMaxRetries: number; // how many tighter-tolerance-then-shrink attempts before giving up on a candidate that fails NFA
  lsdRetryToleranceFactor: number; // tau is multiplied by this on the first retry (LSD: retighten before shrinking)
  lsdRetryShrinkFraction: number; // fraction of a region's own farthest-from-center pixels dropped on each shrink retry
  // The join walk's own 4 parameters (pipeline/bucketFillJoin.ts's
  // computeJoinWalk) -- unchanged in meaning from the old bucketFillJoinSteps/
  // MergeMinSimilarity/MaxTravelFactor/MinLengthPx names, just relocated
  // here since the join walk is now exclusively fed by this LSD pipeline.
  lsdJoinSteps: number; lsdMergeMinSimilarity: number; lsdMaxTravelFactor: number; lsdMinLengthPx: number;
  showGradientArrow: boolean; showGradientArrowPerpendicular: boolean; gradientArrowScale: number;
  coherenceRadius: number;
  tangentWalkMaxSteps: number; tangentWalkDeviationDeg: number; tangentWalkMagFraction: number; tangentWalkGraceSamples: number;
  tangentWalkAdaptive: boolean;
  showRecoveredPoles: boolean;
  showAxisVectors: boolean;
  showTopCircles: boolean;
  topCirclesLineWidth: number;
  weightSharpenPower: number;
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
    showSphere: true, showCircles: false, showPoles: true, showFrustum: true, showPatch: true, showGizmoBody: true, showRecoveredFloor: true, recoveredFloorOpacity: savedNum('recoveredFloorOpacity', 0.92),
    showTrueContamination: false, showReconstructedContamination: false, hideField: false,
    showTopGradient: false,
    showLsdSegments: savedBool('toggleLsdSegments', false),
    showLsdRejected: savedBool('toggleLsdRejected', false),
    showLsdRawRegions: savedBool('toggleLsdRawRegions', false),
    showLsdComposite: savedBool('toggleLsdComposite', false),
    lsdToleranceDeg: savedNum('lsdToleranceDeg', 22.5),
    lsdRhoNoiseThreshold: savedNum('lsdRhoNoiseThreshold', 4),
    lsdMagnitudeBuckets: savedNum('lsdMagnitudeBuckets', 1024),
    lsdNfaEpsilon: savedNum('lsdNfaEpsilon', 1),
    lsdNfaTestExponent: savedNum('lsdNfaTestExponent', 5),
    lsdMaxRetries: savedNum('lsdMaxRetries', 2),
    lsdRetryToleranceFactor: savedNum('lsdRetryToleranceFactor', 0.5),
    lsdRetryShrinkFraction: savedNum('lsdRetryShrinkFraction', 0.2),
    lsdJoinSteps: savedNum('lsdJoinSteps', 0),
    lsdMergeMinSimilarity: savedNum('lsdMergeMinSimilarity', 0.9),
    lsdMaxTravelFactor: savedNum('lsdMaxTravelFactor', 1),
    lsdMinLengthPx: savedNum('lsdMinLengthPx', 3),
    showGradientArrow: false, showGradientArrowPerpendicular: false, gradientArrowScale: 10,
    coherenceRadius: 1,
    // See the pre-Stage-A history for the full derivation of these tangent-walk
    // defaults (guided tangent walk, simNoise=8 stability etc.) -- unchanged.
    tangentWalkMaxSteps: 76, tangentWalkDeviationDeg: 45, tangentWalkMagFraction: 0, tangentWalkGraceSamples: 50,
    tangentWalkAdaptive: false,
    showRecoveredPoles: true,
    showAxisVectors: false,
    showTopCircles: true,
    topCirclesLineWidth: savedNum('topCirclesLineWidth', 1),
    weightSharpenPower: 4,
    minGrazingCos: savedNum('minGrazingCos', 0.15),
    gridPeriodPhaseBinCount: savedNum('gridPeriodPhaseBinCount', 30),
    gridPeriodPhaseGapLowerBound: savedNum('gridPeriodPhaseGapLowerBound', 0.005),
    // Key matches the BUTTON's own id (toggleCompositeLineFamilies), not
    // this field's name -- same persistence convention every other button-
    // driven boolean here uses (e.g. showLsdSegments/toggleLsdSegments).
    showCompositeLineFamilies: savedBool('toggleCompositeLineFamilies', false),
    showSampleLattice: savedBool('showSampleLattice', false),
    useTrueCardinalOrientation: false,
    fieldView: 'gradient2x2',
    axesAutoCapture: false, axesCaptureIntervalMs: 500,
    viewportW: 512, viewportH: 384, aspectLocked: false,
    horizFovDeg: 65,
  };
}
export function createDefaultSimulatedSettings(): SimulatedCameraSettings {
  return {
    ...createDefaultCommonSettings(),
    camX: 0, camY: 20.7, camZ: 8,
    camYawDeg: -43, camPitchDeg: -50,
    simNoise: 1, simBlur: 0, captureSupersample: 2,
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
