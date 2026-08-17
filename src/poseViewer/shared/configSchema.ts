import { Type } from '@sinclair/typebox';
import type { Static } from '@sinclair/typebox';
import { CameraSettingsCommonSchema, FieldViewSchema, SimulatedOnlySettingsSchema } from './camera/settings.ts';

// ── The config file's shape ──────────────────────────────────────────────
//
// Split from config.ts so that this module is PURE: no fetch, no
// localStorage, no top-level await, nothing that assumes a browser. That is
// what lets scripts/check-config.mjs validate pose-viewer.config.json from the
// command line against the very same schema the page will use, instead of
// finding out at boot.
//
// See camera/settings.ts for why these are TypeBox schemas at all, and for
// the per-camera half of the shape.

const strict = { additionalProperties: false } as const;

// The 3D canvas has exactly one current view regardless of camera
// count/selection, which is why the mode is global rather than per-camera.
// Also the source of the Mode type, which types.ts re-exports -- it was a
// hand-written union there, a second list to keep in step by eye.
export const ModeSchema = Type.Union([
  Type.Literal('world'),
  Type.Literal('through'),
  Type.Literal('inside'),
  Type.Literal('projected'),
]);
export type Mode = Static<typeof ModeSchema>;

export const GlobalSettingsSchema = Type.Object({
  mode: ModeSchema,
  // UI chrome, persisted for the same reason every slider is: reopening the
  // page should not undo how you left the workspace.
  panelCollapsed: Type.Boolean(),
  overlayPanelCollapsed: Type.Boolean(),
  // The floor is one shared object, not owned by any camera.
  showFloor: Type.Boolean(),
  floorCellOutlineSubdiv: Type.Number(),
  // The De Bruijn floor pattern's board size (cells per side) -- overrides
  // debruijn.ts's ORDER5_CANDIDATE.cropSize at runtime via scene/floor.ts's
  // rebuildFloorPattern. 144 so that one pattern cell is one board-game cell
  // (BOARD_CELLS in src/game/constants.ts) and the AR overlay's board lands on
  // the physical one 1:1 rather than at some scale factor.
  boardSize: Type.Number(),
  // ── The one GPU/CPU switch, replacing nine per-stage ones (2026-08-05) ──
  //
  // There used to be a `useGPU*` toggle per stage: gradient, growRegions,
  // collectRegions, lsdFit, fit, periodSweep, decode, decodeFused, project. All
  // nine defaulted true. They were the INSTRUMENT that produced every measured
  // number on record, and the answer they produced was "everything is GPU" --
  // at which point a runtime switch for a choice nobody makes at runtime is
  // pure carrying cost. Nine independent booleans is 2^9 of nominal correctness
  // surface, and every new stage doubled it.
  //
  // They also structurally blocked the endgame: you cannot record a fixed
  // command list, or fuse the pipeline into one submit, if any stage might run
  // on the CPU this frame. And they made FieldResidency a transfer-DECISION
  // engine (which side is each stage on, and what must therefore cross?) rather
  // than a plain named-slot arena.
  //
  // What is NOT deleted is the CPU implementations. They earned their keep
  // twice over -- they caught the `layout:'auto'` binding prune that silently
  // no-op'd every submit while returning plausible garbage, and the
  // BOUNDARY_EPS bug that was real on BOTH sides. They are now REFERENCE
  // implementations rather than production branches, reachable through this
  // single switch, and harness/lsdChainVerify.ts's two-configuration
  // differential is what keeps them from rotting unnoticed now that production
  // never runs them.
  //
  // Note this is NOT the no-WebGPU fallback and does not need to be: every GPU
  // stage already falls back on its own when there is no device or a dispatch
  // fails validation, and that path is independent of this flag (the flag gates
  // ENTRY, a `return null/false` gates FALLBACK). This exists for harnesses and
  // for answering "is the GPU lying to me" by hand.
  forceCPU: Type.Boolean(),
}, strict);

// The phone page's own debug toggles (pose-viewer-client.html). Not part of
// PoseInput['settings'] -- those are pushed down per-camera by
// devBridge/client.ts's settingsSync and are already covered by camera.common
// -- these are the switches that only mean anything on the device: whether it
// computes its own pose, what it ships back, and the IMU.
export const PhoneSettingsSchema = Type.Object({
  // The PHONE's own control panel, not the desktop's. `global.panelCollapsed`
  // already exists and is pose-viewer-server.html's; sharing it would mean
  // collapsing a panel on the laptop collapsed the one on the phone, which is
  // the kind of cross-talk the two pages have no reason to have.
  panelCollapsed: Type.Boolean(),
  computeOnDevice: Type.Boolean(),
  // Whether the cube overlay is drawn over the viewfinder. Independent of
  // computeOnDevice even though it needs a local pose to draw anything -- see
  // pose-viewer-client.html's #arToggle on why the two are separate switches.
  arOverlay: Type.Boolean(),
  sendDebugInfo: Type.Boolean(),
  sendCapturedImage: Type.Boolean(),
  imuEnabled: Type.Boolean(),
  imuCorrection: Type.Boolean(),
}, strict);

// A physical camera is a real phone: no position or lens sliders of its own,
// only the handful of common settings that mean something different for a
// photo than for a render. An explicit one-key object rather than a partial,
// so "which settings does a physical camera override" stays a listed fact.
export const PhysicalOverridesSchema = Type.Object({
  fieldView: FieldViewSchema,
}, strict);

export const PoseViewerConfigSchema = Type.Object({
  global: GlobalSettingsSchema,
  camera: Type.Object({
    common: CameraSettingsCommonSchema,
    simulated: SimulatedOnlySettingsSchema,
    physical: PhysicalOverridesSchema,
  }, strict),
  phone: PhoneSettingsSchema,
}, strict);

export type GlobalSettings = Static<typeof GlobalSettingsSchema>;
export type PhoneSettings = Static<typeof PhoneSettingsSchema>;
export type PoseViewerConfig = Static<typeof PoseViewerConfigSchema>;
