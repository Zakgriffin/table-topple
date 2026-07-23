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
  showGizmoBody: boolean; showRecoveredFloor: boolean; showSampleLattice: boolean;
  showTrueContamination: boolean; showReconstructedContamination: boolean; hideField: boolean;
  showTopGradient: boolean;
  showBucketFillSegments: boolean; bucketFillToleranceDeg: number; bucketFillMagnitudeThreshold: number; bucketFillMinLengthPx: number;
  showBucketFillMarkers: boolean;
  showBucketFillJoin: boolean; bucketFillJoinSteps: number; bucketFillMergeMinSimilarity: number;
  bucketFillMaxTravelFactor: number;
  showBucketFillComposite: boolean;
  showBucketFillMergeMarkers: boolean;
  showGradientArrow: boolean; showGradientArrowPerpendicular: boolean; gradientArrowScale: number;
  simGradRadius: number; coherenceRadius: number;
  tangentWalkMaxSteps: number; tangentWalkDeviationDeg: number; tangentWalkMagFraction: number; tangentWalkGraceSamples: number;
  tangentWalkAdaptive: boolean;
  showRecoveredPoles: boolean;
  showAxisVectors: boolean;
  showTopCircles: boolean;
  topCirclesLineWidth: number;
  weightSharpenPower: number;
  useSegmentVotes: boolean;
  showGridPeriodPhaseDebug: boolean;
  gridPeriodPhaseBinCount: number;
  showCompositeLineFamilies: boolean;
  showNewSampleLattice: boolean;
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
    showSphere: true, showCircles: false, showPoles: true, showFrustum: true, showPatch: true, showGizmoBody: true, showRecoveredFloor: true, showSampleLattice: false,
    showTrueContamination: false, showReconstructedContamination: false, hideField: false,
    showTopGradient: false,
    showBucketFillSegments: savedBool('toggleBucketFill', false),
    bucketFillToleranceDeg: savedNum('bucketFillToleranceDeg', 22.5),
    bucketFillMagnitudeThreshold: savedNum('bucketFillMagnitudeThreshold', 0),
    bucketFillMinLengthPx: savedNum('bucketFillMinLengthPx', 3),
    showBucketFillMarkers: savedBool('toggleBucketFillMarkers', true),
    showBucketFillJoin: savedBool('toggleBucketFillJoin', false),
    bucketFillJoinSteps: savedNum('bucketFillJoinSteps', 0),
    bucketFillMergeMinSimilarity: savedNum('bucketFillMergeMinSimilarity', 0.9),
    bucketFillMaxTravelFactor: savedNum('bucketFillMaxTravelFactor', 1),
    showBucketFillComposite: savedBool('toggleBucketFillComposite', false),
    showBucketFillMergeMarkers: savedBool('toggleBucketFillMergeMarkers', false),
    showGradientArrow: false, showGradientArrowPerpendicular: false, gradientArrowScale: 10,
    simGradRadius: 1, coherenceRadius: 1,
    // See the pre-Stage-A history for the full derivation of these tangent-walk
    // defaults (guided tangent walk, simNoise=8 stability etc.) -- unchanged.
    tangentWalkMaxSteps: 76, tangentWalkDeviationDeg: 45, tangentWalkMagFraction: 0, tangentWalkGraceSamples: 50,
    tangentWalkAdaptive: false,
    showRecoveredPoles: true,
    showAxisVectors: false,
    showTopCircles: true,
    topCirclesLineWidth: savedNum('topCirclesLineWidth', 1),
    weightSharpenPower: 4,
    useSegmentVotes: savedBool('useSegmentVotes', false),
    showGridPeriodPhaseDebug: savedBool('showGridPeriodPhaseDebug', false),
    gridPeriodPhaseBinCount: savedNum('gridPeriodPhaseBinCount', 30),
    showCompositeLineFamilies: savedBool('showCompositeLineFamilies', false),
    showNewSampleLattice: savedBool('showNewSampleLattice', false),
    fieldView: 'walked',
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
