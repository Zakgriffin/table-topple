import { decodeGray, fixtureSettings } from '../fixture.ts';
import type { Fixture } from '../fixture.ts';
// Type-only, so this module stays pure and node can import it -- see
// fixture.ts's header for why that matters and what enforces it.

// ── What a harness runs ON ────────────────────────────────────────────────
//
// Every verify in this directory used to begin with the same four lines:
//
//     camera = camera ?? activeCamera() ?? null;
//     if (!camera) return 'no active camera';
//     const gray = camera.lastNoisedPreviewGray;   // or lastAxesCaptureGray
//     const w = camera.rtSize.w, h = camera.rtSize.h, s = camera.settings;
//
// which is what made them un-runnable outside a live page: the input was not a
// parameter, it was a reach into the app. This is that input, named -- so a
// harness takes what it needs and has no opinion about where it came from.
// cameraInput.ts builds one from a live camera; inputFromFixture below builds
// one from a file, with no browser involved.
//
// ── The gray is the PIPELINE's gray, which is a change ────────────────────
//
// Four of the five capture-driven verifies read `lastNoisedPreviewGray`. That
// is the ROW-FLIPPED display copy (axesReconstruction.ts applies the pipeline's
// one remaining flip on the way OUT to the preview), not the top-down
// `lastAxesCaptureGray` that computePoseFromCapture actually runs on.
//
// As a CPU-vs-GPU differential that was still sound -- both sides got the same
// array -- but it means those harnesses have never once been run on the image
// production processes, only on its mirror. Since a fixture stores the real
// capture, they now run on the real orientation. Deltas from before this change
// are not comparable to deltas after it, which costs nothing: every historical
// number here is void for the config-pinning reason anyway.
// ── The detector's tuning, DECLARED HERE now ─────────────────────────────
//
// This was `PoseInput['settings']` -- a projection of the deleted pipeline's
// input type, which is where the fields were declared. It is written out
// instead of re-homing PoseInput, because a settings bag is exactly the part of
// that type worth keeping and the rest (the aspect, the backend, the chain) was
// the pipeline's business.
//
// These are all real user-facing defaults in pose-viewer.config.json, and
// src/pose takes the same numbers per stage -- it deliberately does not import
// the config, so the values in the config file and the ones a pose stage is
// handed have to keep matching by convention rather than by type.
export interface PipelineSettings {
  lsdToleranceDeg: number;
  // rhoNoiseThreshold is hysteresis' LOW bar (participate in edges);
  // rhoHighThreshold is its HIGH bar (a component must contain at least one
  // pixel above it to survive at all).
  lsdRhoNoiseThreshold: number; lsdRhoHighThreshold: number;
  lsdCclSteps: number; // debug round scrubber only -- 0 = run to fixpoint
  lsdMinRegionSize: number;
  lsdNfaEpsilon: number; lsdNfaTestExponent: number;
  lsdMinLengthPx: number;
  horizFovDeg: number;
  gridPeriodPhaseGapLowerBound: number;
  minGrazingCos: number;
}

export interface HarnessInput {
  // Where this came from, carried into reports so a result names its input.
  // The entire point of the exercise: a delta with no named input is a delta
  // nobody can re-derive.
  label: string;
  gray: Float64Array;
  w: number;
  h: number;
  // A physical camera's aspect IS its capture's -- capture.ts sets
  // camera.aspect from rtSize, and rtSize is resized to the incoming photo.
  aspect: number;
  settings: PipelineSettings;
}

// `poseStateFor` USED TO BE HERE, and its deletion is the clearest single
// measure of what step 5f bought. It built a blank twelve-null-field
// PoseComputeState for computePoseFromCapture to mutate -- boilerplate
// the phone page and the since-deleted reconstructionTiming.ts each hand-rolled
// their own copy of, which was the reason to share it.
//
// There is nothing left to build. A HarnessInput already carries `aspect` and
// `settings`, so it satisfies PoseInput as it stands, and the results come back
// as a return value. The "detached state, never the live Camera" caution it
// carried is likewise moot: there is no state to detach.

// ── PROVENANCE: what a report has to carry to be re-derivable ─────────────
//
// The complaint that voided every historical timing number was not that the
// numbers were wrong. It was that no report said what it had measured, so a
// year later nobody could tell whether two of them were comparable. Step 1 made
// the configuration expressible and step 2 made it storable; this is the third
// piece, and it is the one that puts it ON THE OUTPUT.
//
// Six of the seven harnesses in this directory recorded NOTHING about their
// input -- not the label, not the size, not the settings. Their deltas were
// exactly the un-re-derivable kind, in a new place. One spread fixes all of
// them at once.
//
// ── TWO hashes, because "it changed" is not a useful answer on its own ──
//
// configHash covers what the pipeline READS: `settings` and `aspect`. That is
// provably the whole of it, and it is provable only because step 5 drew the
// boundary -- `PoseInput` is the entire surface computePoseFromCapture takes
// besides the pixels and the backend, and nothing under the old pipeline reads an
// ambient global any more. Hashing the whole config file (what fixture.ts
// snapshots) would have been the safe answer BEFORE that boundary existed; now
// it would just make the fingerprint churn on UI edits that cannot reach the
// pipeline, which trains a reader to ignore it.
//
// captureHash covers the pixels. A label does not pin them: `camera:sim-1` is a
// different image every capture, so two runs with an identical configHash can
// still be measuring different work. Separating the two is what makes the
// answer diagnostic -- "same config, different capture" and "same capture,
// different config" are different problems.
//
// The BACKEND is deliberately not in either. It is per-call, several harnesses
// sweep it, and the reports that care record it as its own field.
export interface InputProvenance {
  input: string;
  w: number;
  h: number;
  configHash: string;
  captureHash: string;
}

// FNV-1a, 32 bits. Zero dependencies and deterministic across engines, which is
// the whole requirement -- this identifies a configuration, it does not defend
// against one. Math.imul because the multiply overflows 2^53 otherwise and the
// hash silently degenerates.
function fnv1a(bytes: Uint8Array): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}

function hex(n: number): string { return n.toString(16).padStart(8, '0'); }

// Key order must not change the hash: two settings objects built by different
// code paths hold the same configuration, and a report that called them
// different would be worse than no report.
function canonical(v: unknown): string {
  if (v === null || typeof v !== 'object') return JSON.stringify(v) ?? 'null';
  if (Array.isArray(v)) return `[${v.map(canonical).join(',')}]`;
  const o = v as Record<string, unknown>;
  return `{${Object.keys(o).sort().map((k) => `${JSON.stringify(k)}:${canonical(o[k])}`).join(',')}}`;
}

/**
 * The same content hash `provenance` uses, over any plain value.
 *
 * Exported so a second caller identifies a configuration THE SAME WAY rather
 * than growing its own hash: the sweep tags every report with one of these over
 * the render settings it ran with, so changing pose-viewer.config.json visibly
 * changes the tag instead of silently changing the numbers underneath it.
 * Key-order-independent, per `canonical`.
 */
export function configHashOf(value: unknown): string {
  return hex(fnv1a(new TextEncoder().encode(canonical(value))));
}

export function provenance(input: HarnessInput): InputProvenance {
  return {
    input: input.label,
    w: input.w,
    h: input.h,
    configHash: hex(fnv1a(new TextEncoder().encode(
      canonical({ settings: input.settings, aspect: input.aspect }),
    ))),
    // Over the raw bytes of the f64 samples, not over a stringification: it is
    // ~2.4MB and one pass, paid once when a harness starts rather than per rep.
    captureHash: hex(fnv1a(new Uint8Array(
      input.gray.buffer, input.gray.byteOffset, input.gray.byteLength,
    ))),
  };
}

export function inputFromFixture(fixture: Fixture): HarnessInput {
  return {
    label: `fixture:${fixture.name}`,
    gray: decodeGray(fixture),
    w: fixture.capture.w,
    h: fixture.capture.h,
    aspect: fixture.capture.w / fixture.capture.h,
    settings: fixtureSettings(fixture),
  };
}
