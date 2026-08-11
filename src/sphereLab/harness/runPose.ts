import type { Backend } from '../../pose/backend.ts';
import { readSliceF32, readSliceU32 } from '../../pose/gpu/device.ts';
import { readRegionMembers } from '../../pose/stages/lsd/lsdFit.gpu.ts';
import type { Intermediates } from '../camera/model.ts';
import type { GrownRegion } from '../../pose/stages/lsd/types.ts';
import { computePoseFromCapture } from '../../pose/poseCompute.ts';
import type { PoseResult } from '../../pose/poseCompute.ts';
import type { HarnessInput } from './input.ts';
import { poseSpan } from '../../pose/timing/stages.ts';
import { spanEnd } from '../profiling/profiler.ts';

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
// Everything is READ here rather than handed on, so a caller gets settled data
// and never has to hold a live arena slice. This is the one place in the
// harness that reads intermediates at all, and it reads ALL of them: a verify
// runs once and looks at whatever it needs, so there is nothing to be gained by
// making it declare that in advance -- which is the whole argument the request
// type used to make, and it was only ever true for the display path.
//
// `input` is passed straight through as the PoseInput: a HarnessInput carries
// `aspect` and `settings`, which is the whole of what the pipeline reads.
export async function runPoseOn(
  input: HarnessInput, backend: Backend, wantMembers = false,
): Promise<{ pose: PoseResult; intermediates: Intermediates }> {
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, backend, wantMembers);
  // Same span the app's drain opens, for the same reason: these transfers are
  // owned by `pose.drain` and have to join somewhere.
  const drainSpan = poseSpan('pose.drain');
  const out: Intermediates = { rects: pose.rects };
  const { chain, cpuChain } = pose;
  if (cpuChain) {
    out.fx = cpuChain.fx; out.fy = cpuChain.fy;
    out.regionId = cpuChain.regionId; out.regions = cpuChain.regions as GrownRegion[];
  } else if (chain) {
    const { arena, n } = chain;
    out.fx = new Float64Array(await readSliceF32(arena, chain.fx, n * 4, 'fx', 'pose.drain'));
    out.fy = new Float64Array(await readSliceF32(arena, chain.fy, n * 4, 'fy', 'pose.drain'));
    const raw = await readSliceU32(arena, chain.regionId, n * 4, 'regionId', 'pose.drain');
    out.regionId = new Int32Array(raw.buffer.slice(0));
    out.regions = (await readRegionMembers(
      arena, chain.regions, chain.regionCount, chain.memberCount, 'pose.drain',
    )) as GrownRegion[];
  }
  if (pose.readGrid) {
    const g = await pose.readGrid();
    out.decodeGrid = g.grid; out.decodeRotated = g.rotated; out.decodeCorrectness = g.correctness;
  }
  spanEnd(drainSpan);
  return { pose, intermediates: out };
}
