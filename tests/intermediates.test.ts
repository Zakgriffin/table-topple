import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { computeGradient2x2Field } from '../src/pose/stages/gradient/gradientField.ts';
import { NO_INTERMEDIATES, wants } from '../src/pose/intermediates.ts';
import { growRegionsCCL } from '../src/pose/stages/lsd/regions.cpu.ts';
import { computePoseFromCapture } from '../src/pose/poseCompute.ts';
import { closeTo, loadInput } from './helpers/fixtures.ts';

// ── The intermediates request ─────────────────────────────────────────────
//
// These are the assertions the old arrangement could not even express. You
// could not look at fx/fy at all: the residency was destroyed in
// computePoseFromCapture's finally before any caller could reach it, which is
// exactly why four display sites recomputed the stage on the CPU instead.
//
// The property that matters most is the LAST test here: asking for an
// intermediate must not change the computation. If it did, every debug view
// would be looking at a different pipeline than production runs.
//
// `input` is a HarnessInput and goes straight in as the PoseInput -- it carries
// `aspect` and `settings`, which is all the pipeline reads.

const input = loadInput();

test('asking for nothing leaves nothing behind and needs no drain', async () => {
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', NO_INTERMEDIATES);
  // No handle at all -- not an unresolved one. Capability (1): there is nothing
  // for a caller to remember to drain, so nothing it can leak.
  assert.equal(pose.pending, null);
  // ...and the POSE is still there, which is the whole point of asking for
  // nothing: the pose is not an intermediate. Note there is no field on the
  // result for an intermediate to hide in -- the only way to get one is
  // through a handle, and there is no handle.
  assert.ok(pose.recoveredAxes, 'no pose from a bare request');
  assert.ok(pose.positionDecode, 'no position decode from a bare request');
});

test('a request produces a handle that has to be drained', async () => {
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', wants('fx'));
  assert.ok(pose.pending, 'no handle for a non-empty request');
  const got = await pose.pending!.resolve();
  assert.ok(got.fx, 'fx missing after the drain');
  // Only what was asked for.
  assert.equal(got.fy, undefined);
  assert.equal(got.regions, undefined);
  assert.equal(got.decodeGrid, undefined);
});

test('resolve is idempotent and release is safe in either order', async () => {
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', wants('fx'));
  const handle = pose.pending!;
  const first = await handle.resolve();
  // A second drain over the same capture is safe -- runVisualTail can run twice
  // for one reconstruction, and the second one finds the handle already spent.
  // It re-hands the SAME arrays rather than reading a destroyed residency.
  const second = await handle.resolve();
  assert.equal(second.fx, first.fx, 'a second resolve did not re-hand the first drain');
  handle.release();
  handle.release();
});

test('releasing without resolving hands back nothing rather than throwing', async () => {
  // The two orders must not differ for a caller whose whole point is not
  // having to know which ran first -- see PendingIntermediates. A released
  // handle's buffers are gone, so the honest answer is the empty set.
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', wants('fx'));
  pose.pending!.release();
  assert.deepEqual(await pose.pending!.resolve(), {});
});

test('the fx/fy handed back ARE the field the pipeline ran on', async () => {
  // The claim that makes replacing the display recomputations sound. If these
  // differed, every overlay switched over to them would be drawing a different
  // pipeline than the one that produced the pose on screen next to it.
  //
  // WHAT IT IS BLIND TO: on the CPU backend the residency's fx IS the array
  // computeGradient2x2Field returned, so this is close to a tautology here. It
  // earns its keep on the GPU backend, where the same assertion becomes "what
  // came back down the bus is what went up" -- and that tier cannot run in
  // node. Read it as pinning the CONTRACT, with the GPU half owed to
  // harness/lsdChainVerify.ts until there is a runner with a device.
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', wants('fx', 'fy'));
  const { fx, fy } = await pose.pending!.resolve();
  const direct = computeGradient2x2Field(input.gray, input.w, input.h);
  assert.equal(fx!.length, input.w * input.h);
  assert.deepEqual(Array.from(fx!.subarray(0, 4096)), Array.from(direct.fx.subarray(0, 4096)));
  assert.deepEqual(Array.from(fy!.subarray(0, 4096)), Array.from(direct.fy.subarray(0, 4096)));
});

test('the regions and rects handed back are stage 3b/4 output', async () => {
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', wants('regions', 'regionId', 'rects'));
  const { regions, regionId, rects } = await pose.pending!.resolve();
  const s = input.settings;
  const field = computeGradient2x2Field(input.gray, input.w, input.h);
  const direct = growRegionsCCL(
    field.fx, field.fy, input.w, input.h,
    s.lsdToleranceDeg, s.lsdRhoNoiseThreshold, s.lsdRhoHighThreshold, s.lsdCclSteps, s.lsdMinRegionSize,
  );
  assert.equal(regions!.length, direct.regions.length);
  assert.equal(regionId!.length, input.w * input.h);
  assert.ok(rects!.length > 0, 'no rectangles');
  assert.ok(rects!.some((r) => r.accepted), 'no accepted rectangles');
  // Every rect belongs to a region, so there cannot be more of them.
  assert.ok(rects!.length <= regions!.length, `${rects!.length} rects from ${regions!.length} regions`);
});

test('the decode grid arrives through the same handle as everything else', async () => {
  // It used to be the one requestable name that landed on the caller's state
  // object instead of in `Intermediates` -- see IntermediateName. All three
  // fields come together, and only when asked for.
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', wants('decodeGrid'));
  const got = await pose.pending!.resolve();
  assert.ok(got.decodeGrid, 'decodeGrid was requested but is missing');
  assert.ok(got.decodeRotated, 'decodeRotated was requested but is missing');
  assert.ok(got.decodeCorrectness, 'decodeCorrectness was requested but is missing');
  assert.equal(got.decodeGrid!.rows * got.decodeGrid!.cols > 0, true);
  assert.equal(got.fx, undefined, 'decodeGrid pulled down a field nobody asked for');
});

test('asking for intermediates does not change the pose', async () => {
  // The invariant the whole design rests on. `want` is what to HAND BACK, never
  // what to compute -- so a debug view and a timing run must agree exactly.
  const bare = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', NO_INTERMEDIATES);
  const full = await computePoseFromCapture(
    input, input.gray, input.w, input.h, 'cpu',
    wants('fx', 'fy', 'regionId', 'regions', 'rects', 'decodeGrid'),
  );
  const drained = await full.pending!.resolve();

  assert.equal(bare.votes.length, full.votes.length);
  assert.ok(closeTo(bare.recoveredAxes!.distance, full.recoveredAxes!.distance, 0));
  assert.ok(closeTo(bare.positionDecode!.camPos.x, full.positionDecode!.camPos.x, 0));
  assert.ok(closeTo(bare.positionDecode!.camPos.z, full.positionDecode!.camPos.z, 0));
  assert.equal(bare.positionDecode!.consistency, full.positionDecode!.consistency);
  // ...and the full request really did bring the grid down, so the comparison
  // above is not two bare runs agreeing with each other.
  assert.ok(drained.decodeGrid, 'decodeGrid was requested but is missing');
});
