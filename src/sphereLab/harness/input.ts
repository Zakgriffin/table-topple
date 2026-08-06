import { decodeGray, fixtureSettings } from '../fixture.ts';
import type { Fixture } from '../fixture.ts';
// Type-only, so this module stays pure and node can import it -- see
// fixture.ts's header for why that matters and what enforces it.
import type { PoseComputeState } from '../pipeline/poseCompute.ts';

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
export type PipelineSettings = PoseComputeState['settings'];

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

// A blank PoseComputeState wired to this input, ready for
// computePoseFromCapture to fill in. Twelve null fields a caller would
// otherwise hand-roll -- mobileCapture.ts and reconstructionTiming.ts already
// hand-roll one each, which is the limit before adding a field means hunting
// for the sites that forgot it.
//
// A DETACHED state, never the live Camera, even though a Camera structurally
// satisfies PoseComputeState. Passing the camera would work and would also
// overwrite whatever the app is displaying, and leave the last run's results
// behind afterwards -- reconstructionTiming.ts's freshState was built for
// exactly that reason and this is that function, shared.
export function poseStateFor(input: HarnessInput): PoseComputeState {
  return {
    aspect: input.aspect,
    settings: input.settings,
    lastVoteComposites: null, lastVotes: null, lastQuadricPair: null, lastGridPeriodPhase: null,
    lastRecoveredAxes: null, lastDecodeGrid: null, lastDecodeRotated: null, lastDecodeCorrectness: null,
    lastPositionDecode: null, lastChainTransfers: null,
    // A caller that wants the decode grid's readback deferred has to opt in and
    // own the handle -- see poseCompute's deferDecodeGrid.
    pendingDecodeGrid: null,
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
