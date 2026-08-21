import { Type } from '@sinclair/typebox';
import type { Static } from '@sinclair/typebox';

// ── Table Topple's config file, its shape ────────────────────────────────
//
// Split from config.ts so that this module is PURE: no fetch, no
// localStorage, no top-level await, nothing that assumes a browser. That is
// what lets scripts/check-table-topple-config.mjs validate
// table-topple.config.json from the command line against the very same schema
// the page will use, instead of finding out at boot. Pose Viewer's
// configSchema.ts is pure for exactly the same reason; if this module ever
// grows a browser dependency, that script is what breaks first, which is the
// intended alarm.
//
// ── WHY A SECOND CONFIG FILE AT ALL ──
//
// The isolation rule: Table Topple imports nothing from Pose Viewer, so it
// cannot read pose-viewer.config.json, and the pose library deliberately ships
// no defaults of its own -- it takes a settings object per stage precisely so
// that thresholds stay a property of a board and a camera rather than of the
// library. Something had to own these numbers on this side, and a second JSON
// keeps the project's one hard rule intact: no default value exists anywhere
// except a config file. Not in TypeScript, not as an attribute in the HTML.
//
// ── WHAT IT DOES NOT HAVE, AND WHY ──
//
// No localStorage overlay, no save button, no per-camera section. Pose Viewer
// has all three because it is a lab with ~45 live controls; this client has no
// settings UI at all, so an overlay would be a second source of truth that
// nothing can write and nobody can inspect. If this page ever grows sliders,
// the overlay is the thing to add, and config.ts is where it goes.

const strict = { additionalProperties: false } as const;

// ── Capture ──────────────────────────────────────────────────────────────
//
// No resolution constraint, same as Pose Viewer: whatever the browser
// negotiates on its own is what the PoseContext gets built for. There is
// deliberately no dropdown and no probe sweep, but also no pinned width/height
// to request -- the old exact-resolution request (client/camera.ts's former
// requestStreamAt call) never named a facing direction at all, so it opened
// whichever camera the browser defaults to, not `facing` below. `facing` was
// only ever honoured on the fallback path that request's own `ideal`
// constraints made nearly unreachable. A single facingMode-only request, like
// Pose Viewer's, is what actually gets `facing` applied.
export const CaptureSettingsSchema = Type.Object({
  facing: Type.Union([Type.Literal('environment'), Type.Literal('user')]),
}, strict);
export type CaptureSettings = Static<typeof CaptureSettingsSchema>;

// ── Pose ─────────────────────────────────────────────────────────────────
//
// The thresholds `runPose` is handed every frame. The names match the library's
// own stage settings rather than being renamed here, so a value can be traced
// from this file to the stage that reads it without a translation table --
// client/pose.ts is the only place they are regrouped, and it does nothing but
// regroup them.
//
// THE VALUES START WHERE POSE VIEWER'S DO, AND THAT IS NOT LAZINESS: both
// projects decode the same printed De Bruijn board through a phone camera, so
// the tuning is a property of that board and that sensor, not of the app. They
// are written out here rather than inherited because the two are now
// independently editable -- if Table Topple ever prints a board at a different
// contrast, this is the file that moves, and Pose Viewer's does not.
export const PoseTuningSchema = Type.Object({
  // Vertical FOV is DERIVED from this and the frame's aspect (see
  // client/pose.ts). Horizontal is what a camera's spec sheet quotes, so it is
  // what gets written down.
  horizFovDeg: Type.Number(),
  // Gradient-orientation agreement, in degrees, for a pixel to join a region.
  lsdToleranceDeg: Type.Number(),
  // rho LOW -- the participation floor. Below this a pixel never joins.
  lsdRhoNoiseThreshold: Type.Number(),
  // rho HIGH -- the survival bar for a grown region. At or below the floor,
  // hysteresis degrades to plain single-threshold behaviour, which is what a 0
  // here means.
  lsdRhoHighThreshold: Type.Number(),
  // NFA acceptance: a region is a line if its number of false alarms is under
  // this, tested against 2^-exponent candidate rectangles.
  lsdNfaEpsilon: Type.Number(),
  lsdNfaTestExponent: Type.Number(),
  // Minimum PIXELS in a connected component, applied before any line is fitted.
  lsdMinRegionSize: Type.Number(),
  // Minimum fitted-segment LENGTH in pixels. Not the same filter as
  // lsdMinRegionSize, and the two are easy to confuse.
  lsdMinLengthPx: Type.Number(),

  // ── S5c join: collinear segments -> composite lines ────────────────────
  //
  // The iterative reach-bounded corridor join. See src/pose/pose.ts's
  // JoinSettings for the derivation of every one of these and pose.wgsl.ts's
  // JOIN_GATE_WGSL for the gate itself -- not repeated here, because this file
  // is the values and that one is the reasoning.
  //
  // TWO KNOBS, ONE NUMBER: joinKSigma and joinEndpointNoisePx reach the shader
  // only as their product (tol = kSigma * noise * sqrt(2)). The split is so
  // each means something -- one a confidence level, one a noise model.
  //
  // joinKSigma 0 is the EXACT OFF: no pair can pass the gate, every line
  // becomes its own singleton composite, and the pose is bit-for-bit the
  // pre-join one. On a page with no settings UI at all, editing this file is
  // the only way to run that A/B -- which is the first thing to reach for if a
  // pose on this device looks wrong.
  joinKSigma: Type.Number(),
  joinEndpointNoisePx: Type.Number(),
  // How far a segment's front may travel, as a multiple of its OWN length.
  // Measured at 4; the derivation fixes the form, not the constant. Do not
  // "correct" it to 1.
  joinReachFrac: Type.Number(),
  // Merge rounds. 1 reproduces one-shot star clustering exactly; measured
  // converged at 3. Overshooting is safe, and costs four dispatches a round.
  joinRounds: Type.Number(),
  // How far a member endpoint may sit off the finished composite before that
  // member is dropped.
  joinMaxResidualPx: Type.Number(),
  // Compare |dot| rather than dot on the vote normals, joining across a
  // gradient polarity flip. Measured WORSE; false is the shipping value.
  joinPolarityAbs: Type.Boolean(),

  // How close to edge-on the board may be before its geometry is refused.
  minGrazingCos: Type.Number(),
}, strict);
export type PoseTuning = Static<typeof PoseTuningSchema>;

export const TableToppleConfigSchema = Type.Object({
  capture: CaptureSettingsSchema,
  pose: PoseTuningSchema,
}, strict);
export type TableToppleConfig = Static<typeof TableToppleConfigSchema>;
