import { Type } from '@sinclair/typebox';
import type { Static } from '@sinclair/typebox';

// ── Per-camera settings: the SHAPE only ──────────────────────────────────
//
// Everything that used to live in the single module-level `state` object,
// split into what's common to both camera types and what's type-specific.
//
// There are no default VALUES here -- every one of them lives in
// sphere-lab.config.json, and config.ts is what turns that file into a
// settings object. This file deliberately imports nothing but TypeBox and has
// no side effects at all, so config.ts can import it without a cycle and
// scripts/check-config.mjs can import it outside a browser.
//
// ── Why TypeBox schemas rather than plain interfaces ─────────────────────
//
// One declaration doing three jobs. `Static<typeof X>` derives the TypeScript
// type, so there is no second list of field names to keep in step -- which is
// exactly what the `satisfies Record<keyof T, true>` key manifests here used
// to be for. Those manifests could only check that a key was PRESENT in the
// config file; these check its TYPE, which is the failure that actually
// reaches a GPU uniform. A string where a number belongs is a present key.
//
// additionalProperties: false on every section is deliberate. A renamed or
// deleted setting leaves its old key behind in the JSON, and silently ignoring
// it is how a config file drifts into being half documentation. Loud instead.
const strict = { additionalProperties: false } as const;

// The field-view radio group's legal values, and the source of the FieldView
// type -- which used to be a hand-maintained union in types.ts that had to be
// kept in step with this list by eye. The radio options in sphere-lab.html are
// still a third copy: markup is beyond what a schema can reach.
export const FieldViewSchema = Type.Union([
  Type.Literal('raw'),
  Type.Literal('antialiased'),
  Type.Literal('downsampled'),
  Type.Literal('noised'),
  Type.Literal('gradient2x2'),
  Type.Literal('gradient2x2Directed'),
]);
export type FieldView = Static<typeof FieldViewSchema>;

export const CameraSettingsCommonSchema = Type.Object({
  showSphere: Type.Boolean(),
  showCircles: Type.Boolean(),
  showPoles: Type.Boolean(),
  showFrustum: Type.Boolean(),
  showPatch: Type.Boolean(),
  showGizmoBody: Type.Boolean(),
  showRecoveredFloor: Type.Boolean(),
  recoveredFloorOpacity: Type.Number(),
  showTrueContamination: Type.Boolean(),
  showReconstructedContamination: Type.Boolean(),
  hideField: Type.Boolean(),
  showTopGradient: Type.Boolean(),

  // ── From-scratch traditional LSD pipeline (pose/stages/lsd/) --
  // the PRODUCTION composite-line source: pose/stages/votes/votes.ts's
  // computeGradient2x2Composites turns each accepted rectangle straight into
  // one line. None of these are gated behind a show/hide toggle --
  // they're tuning knobs for a live feature, not just a debug view.
  showLsdSegments: Type.Boolean(),
  // also draw candidates that failed NFA validation (dashed red), not just
  // accepted rectangles
  showLsdRejected: Type.Boolean(),
  // Scatters each drawn rectangle's OWN stage-3 raw region membership
  // (LsdRectangle.rawMembers -- the actual flood-filled pixels, before any
  // retry tightened/shrank it) as small dots, so the raw growing result can
  // be compared directly against the fitted rectangle it produced.
  showLsdRawRegions: Type.Boolean(),
  // draw the merged composite lines fed by the segments above (the join walk
  // that produced them is deleted; see votes.ts's compositesFromLsdRectangles
  // for what fills this now)
  showLsdComposite: Type.Boolean(),

  // The two histograms in the grid period/phase debug plot
  // (overlays/gridPeriodPhaseOverlays.ts). Both are DRAWING-ONLY -- neither
  // feeds the period search, which reads the O(n) line values directly via
  // circularFit. Toggling one off skips computing its data entirely, which
  // matters for the gap histogram: computePooledGaps is O(n^2) in the
  // detected line count and exists only for this plot.
  showGapHistogram: Type.Boolean(),
  showValueHistogram: Type.Boolean(),
  // The two debug curves in the period/phase plot, each independently
  // toggleable: cellCentreDistinctness on its own, and its product with the
  // resultant -- the conjunction the search's two-stage filter is
  // approximating. Both are computed in the OVERLAY across the current view
  // range (see overlays/gridPeriodPhaseOverlays.ts), not in the pipeline, so
  // they follow pan/zoom and can show a joint peak sitting outside the
  // bracket the integer-count search ever looked at.
  showDistinctnessCurve: Type.Boolean(),
  showProductCurve: Type.Boolean(),

  // tau -- the one angle tolerance growRegionsCCL's edge predicate,
  // countRectanglePixels' NFA alignment count and the retry-1 refilter all
  // test against. LSD default 22.5deg.
  lsdToleranceDeg: Type.Number(),
  // rho LOW -- hysteresis' participation floor: a pixel below this is
  // excluded entirely (from growth edges and from NFA alignment counts).
  // Scale is [0,1], matching computeGradient2x2Field's own normalized output.
  lsdRhoNoiseThreshold: Type.Number(),
  // rho HIGH -- hysteresis' survival bar: a grown component is discarded
  // unless at least one of its members exceeds this. Canny's two-threshold
  // idea, and what replaces the magnitude-priority ORDERING that the old
  // competitive growers relied on to stop weak noise out-competing a real
  // ridge (a symmetric edge predicate has no notion of "wins"). Set at or
  // below lsdRhoNoiseThreshold to degrade to plain single-threshold
  // behavior. See pose/stages/lsd/regions.cpu.ts's growRegionsCCL.
  lsdRhoHighThreshold: Type.Number(),
  // DEBUG SCRUBBER ONLY (0 = run to the fixpoint, the real algorithm): caps
  // how many hook+compress rounds growRegionsCCL runs so the overlay can
  // watch components coalesce. Unlike the lsdGrowSteps it replaces, this
  // cannot change the converged answer -- connected components are a
  // fixpoint, not an iteration budget -- only how far along you're looking.
  lsdCclSteps: Type.Number(),
  // epsilon -- accept a candidate rectangle iff NFA < this (LSD default 1:
  // expect <1 false detection per image)
  lsdNfaEpsilon: Type.Number(),
  // N_tests = N^exponent, N = max(image w,h) -- LSD's own estimate of "how
  // many rectangles could plausibly have been tested" (~5 degrees of freedom:
  // 2 position, 1 angle, 2 size)
  lsdNfaTestExponent: Type.Number(),
  // Minimum member count for a grown component to become a region at all --
  // applied in collectRegionsFromLabels, so BOTH the CPU and GPU growers get
  // it and neither ever materializes the region. 2 is the behavior-preserving
  // floor: the rectangle fit needs two members to have an axis at all, so a
  // singleton could only ever be fitted and discarded. Measured on a real
  // capture, that alone is 1149 of 2931 regions -- ~40% of the fit stage's
  // work, for zero change in output. Above 2 it becomes a real
  // (output-changing) tuning knob: a floor on how much evidence a line
  // segment needs.
  lsdMinRegionSize: Type.Number(),
  // The last survivor of the join walk's four parameters -- the other three
  // (join steps, merge similarity, max travel) went with the walk itself.
  //
  // NOT the same filter as lsdMinRegionSize, and the two are easy to confuse:
  // minRegionSize counts PIXELS in a connected component and runs BEFORE any
  // rectangle is fitted; this compares the FITTED rectangle's long-axis extent
  // and runs after NFA acceptance, in compositesFromLsdRectangles. A fat
  // 8-pixel blob passes the first and fails this one.
  lsdMinLengthPx: Type.Number(),

  // showLevelLineArrow: the gradient rotated -90deg (LSD's own level-line
  // convention, see pose/stages/lsd/levelLine.ts) -- was
  // named "perpendicular" before, renamed to match that shared terminology.
  showGradientArrow: Type.Boolean(),
  showLevelLineArrow: Type.Boolean(),
  gradientArrowScale: Type.Number(),

  showRecoveredPoles: Type.Boolean(),
  showAxisVectors: Type.Boolean(),
  showTopCircles: Type.Boolean(),
  topCirclesLineWidth: Type.Number(),

  // Grazing-angle cutoff (cosine) shared by projectSamplesCPU (forward:
  // screen pixel -> floor point) and buildDecodeSampleGrid (reverse: floor
  // point -> screen pixel) -- see pose/stages/decode/decodeGrid.ts's own comment.
  // Higher = stricter (excludes more of the near-horizon view).
  minGrazingCos: Type.Number(),
  gridPeriodPhaseBinCount: Type.Number(),
  // Below this, a red/blue neighbor gap (gridPeriodPhaseOverlays.ts's
  // per-family median lines) is excluded from the median -- filters out the
  // same near-duplicate-line noise gaps pose/stages/period/gridPeriodPhase.ts's own
  // seed-period mode search is built to shrug off, which would otherwise
  // drag a small-sample median toward ~0 instead of the true spacing.
  gridPeriodPhaseGapLowerBound: Type.Number(),
  showCompositeLineFamilies: Type.Boolean(),
  showSampleLattice: Type.Boolean(),
  // Purely a display-time rotation of the Projected-Cam view (WebGL texture
  // + debug overlay) by camera.pose.positionDecode.orientation * 90 degrees,
  // so "up" matches the pattern's true cardinal orientation instead of
  // whichever of the 4 the raw sample buffer happened to land in -- doesn't
  // touch the decode pipeline itself, see main.ts's projected-mode branch.
  useTrueCardinalOrientation: Type.Boolean(),
  fieldView: FieldViewSchema,
  axesAutoCapture: Type.Boolean(),
  axesCaptureIntervalMs: Type.Number(),
  viewportW: Type.Number(),
  viewportH: Type.Number(),
  aspectLocked: Type.Boolean(),
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
  horizFovDeg: Type.Number(),
}, strict);

// Only the fields a simulated camera ADDS. Its own schema rather than a
// composite with the common one, because the config file stores them as two
// separate sections and these are exactly one section's contents.
export const SimulatedOnlySettingsSchema = Type.Object({
  camX: Type.Number(),
  camY: Type.Number(),
  camZ: Type.Number(),
  camYawDeg: Type.Number(),
  camPitchDeg: Type.Number(),
  simNoise: Type.Number(),
  simBlur: Type.Number(),
  captureSupersample: Type.Number(),
}, strict);

export type CameraSettingsCommon = Static<typeof CameraSettingsCommonSchema>;
export type SimulatedOnlySettings = Static<typeof SimulatedOnlySettingsSchema>;

// An intersection rather than `interface extends`, because both halves are
// derived types now. Structurally identical for every consumer.
export type SimulatedCameraSettings = CameraSettingsCommon & SimulatedOnlySettings;
// A physical camera adds nothing of its own -- it is a real phone, so it has
// no position or lens sliders. It only OVERRIDES a common setting (fieldView;
// see config.ts's camera.physical section).
export type PhysicalCameraSettings = CameraSettingsCommon;

// ── config.camera -> a settings object ───────────────────────────────────
//
// The rule for turning the config file's three camera sections into the
// settings a camera actually runs with. It lives HERE, in the pure module,
// rather than in config.ts, because there are now two callers with nothing in
// common: camera/factory.ts creating a live camera (through config.ts's
// createDefault*Settings, which are these functions applied to the live
// config), and fixture.ts rebuilding the settings a saved capture was
// processed under -- outside a browser, with no live config to read.
//
// They take `config.camera` rather than the whole config so this file does not
// have to import configSchema.ts, which imports this one.
//
// The parameter types are structural for the same reason: PhysicalOverrides
// is declared over in configSchema.ts, and `{ fieldView: FieldView }` is its
// whole contents. If that section ever grows a second key, this signature is
// what fails to compile.
export function physicalSettingsFrom(
  camera: { common: CameraSettingsCommon; physical: { fieldView: FieldView } },
): PhysicalCameraSettings {
  // camera.physical overrides the common block: 'raw' rather than 'noised',
  // because the simulated-distortion field views (noised/antialiased/
  // downsampled) do not exist for a real photo and are hidden from the
  // field-view list entirely for a physical camera (see refreshCameraPanel).
  return { ...structuredClone(camera.common), ...structuredClone(camera.physical) };
}

export function simulatedSettingsFrom(
  camera: { common: CameraSettingsCommon; simulated: SimulatedOnlySettings },
): SimulatedCameraSettings {
  return { ...structuredClone(camera.common), ...structuredClone(camera.simulated) };
}
