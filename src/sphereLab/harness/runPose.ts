import type { Backend } from '../../pose/backend.ts';
import { type Intermediates, type IntermediatesRequest, NO_INTERMEDIATES } from '../../pose/intermediates.ts';
import { computePoseFromCapture } from '../../pose/poseCompute.ts';
import type { PoseResult } from '../../pose/poseCompute.ts';
import type { HarnessInput } from './input.ts';

// One whole reconstruction on a harness input, drained.
//
// The late-stage verifies (decodeGridBuild, periodSweep) do not check a
// function of the raw pixels -- they check a stage that consumes the OUTPUT of
// every stage before it, so they need a completed run to look at. They used to
// get one by reading `camera.lastGridPeriodPhase` etc. off whatever the app had
// most recently displayed, which is what made them un-re-derivable: the result
// depended on a run nobody recorded, triggered by a button, under whatever
// settings were live at the time.
//
// Running it here instead costs one reconstruction (tens of ms) and buys the
// property that matters: the verify's answer is a function of its argument.
//
// The request is RESOLVED here rather than handed on, so a caller gets settled
// data and never has to know a handle existed. verifyDecodeGridBuild asks for
// 'decodeGrid' this way -- the grid is absent from `intermediates` unless
// someone asked.
//
// `input` is passed straight through as the PoseInput: a HarnessInput carries
// `aspect` and `settings`, which is the whole of what the pipeline reads.
export async function runPoseOn(
  input: HarnessInput, backend: Backend, want: IntermediatesRequest = NO_INTERMEDIATES,
): Promise<{ pose: PoseResult; intermediates: Intermediates }> {
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, backend, want);
  const intermediates = (await pose.pending?.resolve()) ?? {};
  return { pose, intermediates };
}
