import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { computeGradient2x2Field } from '../src/pose/stages/gradient/gradientField.ts';
import { growRegionsCCL } from '../src/pose/stages/lsd/regions.cpu.ts';
import { computePoseFromCapture } from '../src/pose/poseCompute.ts';
import { closeTo, loadInput } from './helpers/fixtures.ts';

// ── Getting the pipeline's intermediates ──────────────────────────────────
//
// These used to test a REQUEST MECHANISM: a set of names went in, a handle came
// back, ownership of the chain's device buffers moved with it, and the tests
// pinned resolve-is-idempotent, release-is-safe-twice, and
// resolve-after-release-hands-back-nothing.
//
// None of those tests exist any more because none of those rules do. The whole
// mechanism was a workaround for the pipeline destroying its buffers in its own
// `finally`, before any caller could look; with the arena the buffers outlive
// the call and are freed by the NEXT reconstruction, so getting an intermediate
// is reading a field.
//
// What survives is what was always the real content: that the things handed
// back ARE the pipeline's own, and that getting them cannot change the pose.
//
// CPU backend, so it runs headless. On this path the intermediates are the
// pipeline's own arrays by identity, which makes some of these close to
// tautologies here -- they earn their keep on the GPU backend, where the same
// assertions become "what came back down the bus is what went up", and that
// tier is owed to harness/lsdChainVerify.ts until there is a runner with a
// device.

const input = loadInput();

test('a pose comes back with no reads made and nothing to clean up', async () => {
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu');
  assert.ok(pose.recoveredAxes, 'no pose');
  assert.ok(pose.positionDecode, 'no position decode');
  // There is no handle to leak and no release to forget. The old version of
  // this test asserted `pose.pending === null` to prove the fast path built no
  // handle; the field is gone, so the property is now structural.
  assert.ok(pose.cpuChain, 'a cpu run left no chain to read from');
  assert.equal(pose.chain, null, 'a cpu run produced a GPU chain');
});

test('the fx/fy handed back ARE the field the pipeline ran on', async () => {
  // The claim that makes replacing the display recomputations sound. If these
  // differed, every overlay switched over to them would be drawing a different
  // pipeline than the one that produced the pose on screen next to it.
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu');
  const { fx, fy } = pose.cpuChain!;
  const direct = computeGradient2x2Field(input.gray, input.w, input.h);
  assert.equal(fx.length, input.w * input.h);
  assert.deepEqual(Array.from(fx.subarray(0, 4096)), Array.from(direct.fx.subarray(0, 4096)));
  assert.deepEqual(Array.from(fy.subarray(0, 4096)), Array.from(direct.fy.subarray(0, 4096)));
});

test('the regions and rects handed back are the collect/fit output', async () => {
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu');
  const { regions, regionId } = pose.cpuChain!;
  const rects = pose.rects;
  const s = input.settings;
  const field = computeGradient2x2Field(input.gray, input.w, input.h);
  const direct = growRegionsCCL(
    field.fx, field.fy, input.w, input.h,
    s.lsdToleranceDeg, s.lsdRhoNoiseThreshold, s.lsdRhoHighThreshold, s.lsdCclSteps, s.lsdMinRegionSize,
  );
  assert.equal(regions.length, direct.regions.length);
  assert.equal(regionId.length, input.w * input.h);
  assert.ok(rects.length > 0, 'no rectangles');
  assert.ok(rects.some((r) => r.accepted), 'no accepted rectangles');
  // Every rect belongs to a region, so there cannot be more of them.
  assert.ok(rects.length <= regions.length, `${rects.length} rects from ${regions.length} regions`);
});

test('the decode grid is read through a function, not owned through a handle', async () => {
  // It used to be the one requestable name that landed on the caller's state
  // object rather than in the drained payload, and then the one thing still
  // carrying a resolve/release pair after everything else stopped. All three
  // fields still come together, because the rotation and the correctness array
  // are derived from the grid and are worthless without it.
  const pose = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu');
  assert.ok(pose.readGrid, 'no way to read the decode grid');
  const got = await pose.readGrid!();
  assert.ok(got.grid && got.rotated && got.correctness);
  assert.ok(got.grid.rows * got.grid.cols > 0);
});

test('reading intermediates does not change the pose', async () => {
  // The invariant the whole design rested on, and the reason the request type
  // existed at all: what you ask for must never be what gets computed, or a
  // debug view and a timing run would disagree.
  //
  // It is structural now rather than enforced -- a read is a read -- but the
  // test stays, because "structural" is a claim about code that can stop being
  // true. What WOULD break it: a future stage that skips work when nobody is
  // going to read its output. The `wantMembers` argument below is exactly that
  // shape and is the one thing to watch: it changes what the chain BRINGS BACK,
  // and this asserts it does not change what the chain COMPUTES.
  const bare = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', false);
  const full = await computePoseFromCapture(input, input.gray, input.w, input.h, 'cpu', true);
  const grid = await full.readGrid!();

  assert.equal(bare.votes.length, full.votes.length);
  assert.ok(closeTo(bare.recoveredAxes!.distance, full.recoveredAxes!.distance, 0));
  assert.ok(closeTo(bare.positionDecode!.camPos.x, full.positionDecode!.camPos.x, 0));
  assert.ok(closeTo(bare.positionDecode!.camPos.z, full.positionDecode!.camPos.z, 0));
  assert.equal(bare.positionDecode!.consistency, full.positionDecode!.consistency);
  // ...and the reads really did happen, so the comparison above is not two
  // identical runs agreeing with each other.
  assert.ok(grid.grid, 'the decode grid was not read');
  assert.ok(bare.rects.length > 0 && full.rects.length === bare.rects.length);
});
