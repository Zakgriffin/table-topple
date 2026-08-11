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

// FLAT: nine numbers and a flag, nothing that points anywhere. It used to
// carry `rawMembers`, its region's whole pixel-index list, which made a
// rectangle an object rather than a row -- and made "were the members read?" a
// property of when the rect was BUILT, since the field could only be filled
// inside the chain while the CSR was still device-resident. That is what the
// `wantMembers` request parameter existed to answer, and it is why the CSR was
// read twice per reconstruction whenever a debug view was on.
//
// ── THE INDEX JOIN, which replaces it ──
//
// `rects[i]` is the fit of `regions[i]`, one rect per grown region, on BOTH
// backends and in the same order. The GPU fitter dispatches over `regionCount`
// and writes `out[r]`; the CPU fitter walks its region array in order and emits
// a slot for every region including the degenerate ones it cannot fit (see
// fitRegionsCPU). So a caller holding both arrays recovers a rectangle's
// members with `regions[i].members` and pays nothing for the ones it does not
// look at.
//
// Keep that 1:1 -- it is the join's only premise. A fitter that drops or
// reorders regions breaks every member-drawing overlay silently, by
// mis-attributing pixels rather than by failing.
export interface LsdRectangle {
  cx: number; cy: number; theta: number; length: number; width: number;
  accepted: boolean;
  nfaLog10: number; // log10(NFA) -- more negative = more statistically confident
  lineScore: number; // nfaLog10 squashed to [0, 1] via nfaLog10ToLineScore -- see that function's own comment
}

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
