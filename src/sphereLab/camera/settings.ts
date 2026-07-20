import { FieldView } from '../types.ts';

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
  showGradientArrow: boolean; showGradientArrowPerpendicular: boolean; gradientArrowScale: number;
  showTangentWalkPath: boolean;
  simGradRadius: number; coherenceRadius: number;
  tangentWalkMaxSteps: number; tangentWalkDeviationDeg: number; tangentWalkMagFraction: number; tangentWalkGraceSamples: number;
  tangentWalkAdaptive: boolean;
  circleSamplePercentMin: number; circleSamplePercentMax: number;
  showRecoveredPoles: boolean;
  showAxisVectors: boolean;
  showTopCircles: boolean;
  weightSharpenPower: number;
  orientationLM: boolean;
  positionLM: boolean;
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
    showSphere: true, showCircles: true, showPoles: true, showFrustum: true, showPatch: true, showGizmoBody: true, showRecoveredFloor: true, showSampleLattice: false,
    showTrueContamination: false, showReconstructedContamination: false, hideField: false,
    showGradientArrow: false, showGradientArrowPerpendicular: false, gradientArrowScale: 10,
    showTangentWalkPath: false,
    simGradRadius: 1, coherenceRadius: 1,
    // See the pre-Stage-A history for the full derivation of these tangent-walk
    // defaults (guided tangent walk, simNoise=8 stability etc.) -- unchanged.
    tangentWalkMaxSteps: 76, tangentWalkDeviationDeg: 45, tangentWalkMagFraction: 0, tangentWalkGraceSamples: 50,
    tangentWalkAdaptive: false,
    circleSamplePercentMin: 0, circleSamplePercentMax: 10,
    showRecoveredPoles: true,
    showAxisVectors: false,
    showTopCircles: true,
    weightSharpenPower: 4,
    orientationLM: false,
    positionLM: false,
    fieldView: 'effective',
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
