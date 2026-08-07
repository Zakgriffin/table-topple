import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { computeGradient2x2Field } from '../src/pose/stages/gradient/gradientField.ts';
import { type Intermediates, NO_INTERMEDIATES, wants } from '../src/pose/intermediates.ts';
import { growRegionsCCL } from '../src/pose/stages/lsd/lsdSegments.ts';
import { computePoseFromCapture } from '../src/pose/poseCompute.ts';
import { poseStateFor } from '../src/sphereLab/harness/input.ts';
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

const input = loadInput();

test('asking for nothing leaves nothing behind and needs no drain', async () => {
  const state = poseStateFor(input);
  await computePoseFromCapture(state, input.gray, input.w, input.h, 'cpu', NO_INTERMEDIATES);
  // No handle at all -- not an unresolved one. Capability (1): there is nothing
  // for a caller to remember to drain, so nothing it can leak.
  assert.equal(state.pendingIntermediates, null);
  assert.equal(state.intermediates, null);
  // The decode grid is an intermediate like any other now, so it is absent too.
  assert.equal(state.lastDecodeGrid, null);
  assert.equal(state.lastDecodeRotated, null);
  assert.equal(state.lastDecodeCorrectness, null);
  // ...and the POSE is still there, which is the whole point of asking for
  // nothing: the pose is not an intermediate.
  assert.ok(state.lastRecoveredAxes, 'no pose from a bare request');
  assert.ok(state.lastPositionDecode, 'no position decode from a bare request');
});

test('a request produces a handle that has to be drained', async () => {
  const state = poseStateFor(input);
  await computePoseFromCapture(state, input.gray, input.w, input.h, 'cpu', wants('fx'));
  assert.ok(state.pendingIntermediates, 'no handle for a non-empty request');
  // Nothing is handed over until the drain runs.
  assert.equal(state.intermediates, null);
  await state.pendingIntermediates!.resolve();
  // Re-read rather than reusing the narrowed binding above: the assert that it
  // was null before the drain is a type guard, and the point of this test is
  // that it stopped being null.
  const got = state.intermediates as Intermediates | null;
  assert.ok(got?.fx, 'fx missing after the drain');
  // Only what was asked for.
  assert.equal(got!.fy, undefined);
  assert.equal(got!.regions, undefined);
});

test('resolve is idempotent and release is safe in either order', async () => {
  const state = poseStateFor(input);
  await computePoseFromCapture(state, input.gray, input.w, input.h, 'cpu', wants('fx'));
  const handle = state.pendingIntermediates!;
  await handle.resolve();
  const first = state.intermediates!.fx;
  // A second drain over the same capture is safe -- runVisualTail can run twice
  // for one reconstruction, and the second one finds the handle already spent.
  await handle.resolve();
  assert.equal(state.intermediates!.fx, first, 'a second resolve re-ran the drain');
  handle.release();
  handle.release();
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
  const state = poseStateFor(input);
  await computePoseFromCapture(state, input.gray, input.w, input.h, 'cpu', wants('fx', 'fy'));
  await state.pendingIntermediates!.resolve();
  const { fx, fy } = state.intermediates!;
  const direct = computeGradient2x2Field(input.gray, input.w, input.h);
  assert.equal(fx!.length, input.w * input.h);
  assert.deepEqual(Array.from(fx!.subarray(0, 4096)), Array.from(direct.fx.subarray(0, 4096)));
  assert.deepEqual(Array.from(fy!.subarray(0, 4096)), Array.from(direct.fy.subarray(0, 4096)));
});

test('the regions and rects handed back are stage 3b/4 output', async () => {
  const state = poseStateFor(input);
  await computePoseFromCapture(state, input.gray, input.w, input.h, 'cpu', wants('regions', 'regionId', 'rects'));
  await state.pendingIntermediates!.resolve();
  const { regions, regionId, rects } = state.intermediates!;
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

test('asking for intermediates does not change the pose', async () => {
  // The invariant the whole design rests on. `want` is what to HAND BACK, never
  // what to compute -- so a debug view and a timing run must agree exactly.
  const bare = poseStateFor(input);
  await computePoseFromCapture(bare, input.gray, input.w, input.h, 'cpu', NO_INTERMEDIATES);
  const full = poseStateFor(input);
  await computePoseFromCapture(full, input.gray, input.w, input.h, 'cpu', wants('fx', 'fy', 'regionId', 'regions', 'rects', 'decodeGrid'));
  await full.pendingIntermediates!.resolve();

  assert.equal(bare.lastVotes!.length, full.lastVotes!.length);
  assert.ok(closeTo(bare.lastRecoveredAxes!.distance, full.lastRecoveredAxes!.distance, 0));
  assert.ok(closeTo(bare.lastPositionDecode!.camPos.x, full.lastPositionDecode!.camPos.x, 0));
  assert.ok(closeTo(bare.lastPositionDecode!.camPos.z, full.lastPositionDecode!.camPos.z, 0));
  assert.equal(bare.lastPositionDecode!.consistency, full.lastPositionDecode!.consistency);
  // ...and the full request really did bring the grid down, so the comparison
  // above is not two bare runs agreeing with each other.
  assert.ok(full.lastDecodeGrid, 'decodeGrid was requested but is null');
});
