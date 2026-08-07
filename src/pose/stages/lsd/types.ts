// ── The LSD stage's data types ───────────────────────────────────────────
//
// A leaf, so the three modules that produce and consume these (regions.cpu.ts,
// rectangles.cpu.ts, chain.ts) can share them without any two of them having
// to import each other. LsdSettings in particular spans the whole chain --
// its first four fields configure the grower and its last two the NFA test --
// which is exactly why it does not belong to either stage's own file.

// meanU is the region's mean level-line DIRECTION, normalized. It was an angle
// (meanAngle) until the vector-space pass; its only consumer, fitRectangle,
// used it purely to pick which of the PCA axis's two opposite directions to
// keep, so only the direction has ever mattered and the normalization is for
// comparability rather than correctness.
export interface GrownRegion { members: Int32Array; meanUx: number; meanUy: number }

export interface LsdRectangle {
  cx: number; cy: number; theta: number; length: number; width: number;
  accepted: boolean;
  nfaLog10: number; // log10(NFA) -- more negative = more statistically confident
  lineScore: number; // nfaLog10 squashed to [0, 1] via nfaLog10ToLineScore -- see that function's own comment
  // Stage 3's grown-region membership (pixel indices into the field) -- the
  // true, complete flood-fill result this rectangle came from. For debug display
  // only (overlays/hoverDebugOverlays.ts's raw-region-pixels toggle) -- not
  // read anywhere in the accept/reject decision itself.
  //
  // EMPTY ON THE POSE PATH as of 2026-08-05 -- see NO_MEMBERS below and
  // fitRegionsGPU's header. Populated only by the CPU fitter, which is what
  // overlays/lsdOverlay.ts's own from-scratch recomputation runs.
  rawMembers: Int32Array;
}

// The one shared empty membership, handed to every rectangle the GPU fitter
// produces. Shared rather than allocated per rectangle because there are a few
// thousand per frame and nothing ever writes through it -- and a single identity
// makes "this rectangle has no members because nobody asked for them" greppable,
// which `new Int32Array(0)` scattered at four call sites would not be.
export const NO_MEMBERS = new Int32Array(0);

export interface LsdSettings {
  toleranceDeg: number;
  // rhoNoiseThreshold is hysteresis' LOW bar (participate in edges);
  // rhoHighThreshold is its HIGH bar (a component must contain at least one
  // pixel above it to survive at all). Kept under the original name so every
  // existing caller/persisted control id stays valid -- see growRegionsCCL's
  // own comment for what the pair does.
  rhoNoiseThreshold: number; rhoHighThreshold: number;
  cclSteps: number; // debug round scrubber only -- 0 = run to fixpoint, see growRegionsCCL
  minRegionSize: number; // components smaller than this never become regions -- see camera/settings.ts's lsdMinRegionSize
  nfaEpsilon: number; nfaTestExponent: number;
}
