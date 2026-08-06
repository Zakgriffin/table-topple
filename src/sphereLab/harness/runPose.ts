import type { Backend } from '../pipeline/backend.ts';
import { computePoseFromCapture } from '../pipeline/poseCompute.ts';
import type { PoseComputeState } from '../pipeline/poseCompute.ts';
import { poseStateFor } from './input.ts';
import type { HarnessInput } from './input.ts';

// One whole reconstruction on a harness input, into a detached state.
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
// `deferDecodeGrid` is false -- unlike the timing harness, these verifies read
// lastDecodeGrid and need the readback to have happened.
export async function runPoseOn(input: HarnessInput, backend: Backend): Promise<PoseComputeState> {
  const state = poseStateFor(input);
  await computePoseFromCapture(state, input.gray, input.w, input.h, backend, false);
  return state;
}
