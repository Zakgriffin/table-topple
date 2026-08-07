import type { GrownRegion, LsdRectangle } from './lsdSegments.ts';

// ── Asking the pipeline for its intermediates, as DATA ────────────────────
//
// The problem, in the user's own diagnosis: this project has repeatedly spent
// effort relieving the friction between "expose a GPU intermediate to the CPU
// so it can be visualized" and "don't pay for it when nobody is looking." Every
// previous answer was a per-call-site convention -- a boolean parameter, a view
// check inside a stage, or, four times over, giving up and RECOMPUTING the
// stage on the CPU because the GPU copy had already been destroyed.
//
// The three capabilities this has to serve, in the order they matter:
//
//   1. "All I care about is pose reconstruction." Must be maximally fast:
//      asking for nothing costs nothing. An empty request destroys the
//      residency in computePoseFromCapture's finally exactly as before, and no
//      readback of any kind happens.
//   2. "I want this, this and this intermediate." Named, one at a time.
//   3. Timing on/off -- which is the profiler's job, not this one's.
//
// ── Why a SET OF NAMES and not booleans on the settings object ────────────
//
// Settings are algorithm parameters: what to compute. This is what to HAND
// BACK, which is a property of the call, not of the configuration -- the same
// settings on the same pixels should be able to produce a bare pose for a
// timing run and a full set of debug fields for an overlay, and comparing those
// two is only meaningful if nothing about the computation changed between them.
// Same argument that keeps `backend` out of LsdSettings (see backend.ts).
//
// It is data rather than a bag of flags for a second reason: a test is a
// first-class consumer. You cannot assert on fx/fy today at all -- the
// residency is destroyed before any caller could look -- and a parameterized
// differential harness wants to ITERATE the set, which it can do to a set of
// names and cannot do to a struct of booleans.
//
// ── What is NOT here in round one ─────────────────────────────────────────
//
// Intermediates the kernels do not currently emit. Every name below already
// exists on the device or on the host at some point during a run and is merely
// destroyed too early, so exposing it costs ZERO shader changes. Generating
// shader variants per requested intermediate is the expensive, bug-prone half
// and is deliberately deferred -- this type is the thing designed to make it
// addable later without changing any caller.

export type IntermediateName =
  // Stage 1, per-pixel, w*h each. The 2x2 gradient field, in the PIPELINE's
  // top-down convention -- see the note on orientation below.
  | 'fx' | 'fy'
  // Stage 3b, per-pixel: which region each pixel ended up in, -1 for none.
  // fieldResidency.ts has described this for a while as "read ONLY by the
  // raw-region debug overlay" -- which never read it, because it could not.
  | 'regionId'
  // Stage 3b proper: the grown regions with their member lists.
  | 'regions'
  // Stage 4: the fitted rectangles with their NFA scores and accept flags.
  | 'rects'
  // The decode sample grid -- 0.45MB, and the one intermediate that already had
  // a deferral mechanism of its own (the old PendingDecodeGrid). Populates
  // lastDecodeGrid / lastDecodeRotated / lastDecodeCorrectness rather than
  // landing in `Intermediates`, because those three fields already exist and
  // several consumers already read them.
  | 'decodeGrid';

export type IntermediatesRequest = ReadonlySet<IntermediateName>;

// The fast path, by name. Shared and frozen so that the overwhelmingly common
// case -- a caller that wants a pose -- allocates nothing and reads as an
// intention rather than as an empty literal.
export const NO_INTERMEDIATES: IntermediatesRequest = new Set<IntermediateName>();

export function wants(...names: IntermediateName[]): IntermediatesRequest {
  return new Set(names);
}

// What came back. Every field is optional and present exactly when it was
// asked for -- so a consumer that reads `intermediates.fx!` after requesting
// 'fx' is right, and one that reads it without asking gets undefined rather
// than a stale array from two frames ago.
//
// ── ORIENTATION, which is the trap ────────────────────────────────────────
//
// These are the PIPELINE's own buffers, so they are TOP-DOWN, the dominant
// convention everywhere except the preview textures (see axesReconstruction.ts:
// the one remaining flip in the whole pipeline is applied on the way OUT to
// lastNoisedPreviewGray). The four display recomputations these replace all ran
// on that flipped copy, so a consumer switching over has to flip at PAINT time
// rather than assume the two are interchangeable. They are not, and not merely
// by a row reversal: a vertical flip also negates the vertical derivative and
// shifts the 2x2 stencil by a row, so `flipRows(computeField(gray))` and
// `computeField(flipRows(gray))` are different arrays.
export interface Intermediates {
  fx?: Float64Array;
  fy?: Float64Array;
  regionId?: Int32Array;
  regions?: GrownRegion[];
  rects?: LsdRectangle[];
}

// ── The drain-once handle ─────────────────────────────────────────────────
//
// Generalized from PendingDecodeGrid, which was the same idea built once by
// hand for one payload. What made that a convention rather than a mechanism was
// its `deferDecodeGrid` boolean: per its own comment, "this function has three
// callers and only one of them can honour it".
//
// OWNERSHIP TRANSFERS. When a request is non-empty, computePoseFromCapture does
// NOT destroy the residency in its finally -- this handle owns it, and whoever
// holds the handle must resolve or release it. Left unresolved it holds the
// chain's device buffers until device loss.
//
// `resolve` is IDEMPOTENT and `release` is safe to call twice, in either order,
// because callers use them differently: a phone or a headless test resolves
// immediately, while the desktop parks this on the camera and drains it a frame
// later -- possibly never, if a newer reconstruction supersedes it first, which
// is exactly when release-without-resolve has to be free.
//
// NOT view-conditional, and that distinction is the whole reason this is a
// deferral rather than a skip: whoever holds this WILL resolve it, so every
// observable ends up identical and only the timing moves. What IS conditional
// is the REQUEST -- and that is the caller's declaration, made before the run,
// not the pipeline guessing what a view might want mid-flight.
export interface PendingIntermediates {
  resolve(): Promise<void>;
  release(): void;
}
