import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { computeGradient2x2Field } from '../src/sphereLab/pipeline/gradientField.ts';
import { growRegionsCCL } from '../src/sphereLab/pipeline/lsdSegments.ts';
import { runPoseOn } from '../src/sphereLab/harness/runPose.ts';
import { closeTo, loadInput } from './helpers/fixtures.ts';

// ── The CPU reference tier, headless ──────────────────────────────────────
//
// These are the implementations production no longer runs. lsdChainVerify's
// own header names the risk: the GPU took over every stage, so the CPU side
// can rot silently, and if it does two things break at once -- the no-WebGPU
// fallback ships broken, and the reference every future port is verified
// against is wrong. Until now the only thing standing between them and that
// was a differential run in a live browser, by hand.
//
// ── WHAT THESE CAN AND CANNOT TELL YOU ────────────────────────────────────
//
// Two different kinds of assertion live here and they are worth not confusing:
//
//   INVARIANTS (orthonormal axes, distance positive, consistency above chance)
//   are correctness claims. They would fail on a wrong answer even if the
//   wrong answer had always been produced.
//
//   GOLDEN VALUES are NOT. They say "unchanged", not "correct". Every one of
//   them was recorded from this same code on 2026-08-06, so a bug that was
//   already present is baked into them -- exactly the blind spot that let the
//   level-line component order survive two sessions. They catch drift and
//   nothing else, which is what they are for.
//
// The goldens are also all f64 arithmetic, deterministic on any IEEE-754 host
// except where libm differs (Math.sin/cos/atan2), which is why they compare
// with a relative tolerance rather than for equality. No wall-clock assertion
// appears anywhere here on purpose: a flaky perf threshold reproduces the same
// "numbers you cannot interpret later" failure in a new place.

const input = loadInput();

test('the 2x2 gradient field covers the image and is finite everywhere', () => {
  const { fx, fy, w, h } = computeGradient2x2Field(input.gray, input.w, input.h);
  assert.equal(w, input.w);
  assert.equal(h, input.h);
  assert.equal(fx.length, input.w * input.h);
  assert.equal(fy.length, input.w * input.h);
  for (let i = 0; i < fx.length; i++) {
    if (!Number.isFinite(fx[i]) || !Number.isFinite(fy[i])) {
      assert.fail(`non-finite gradient at ${i}: (${fx[i]}, ${fy[i]})`);
    }
  }
});

test('grown regions respect minRegionSize and stay inside the image', () => {
  const s = input.settings;
  const { fx, fy } = computeGradient2x2Field(input.gray, input.w, input.h);
  const { regions, regionId } = growRegionsCCL(
    fx, fy, input.w, input.h,
    s.lsdToleranceDeg, s.lsdRhoNoiseThreshold, s.lsdRhoHighThreshold, s.lsdCclSteps, s.lsdMinRegionSize,
  );
  assert.ok(regions.length > 0, 'the fixture should produce regions at its pinned settings');
  assert.equal(regionId.length, input.w * input.h);
  for (const r of regions) {
    assert.ok(r.members.length >= s.lsdMinRegionSize, `region of ${r.members.length} below minRegionSize`);
    for (const m of r.members) {
      assert.ok(m >= 0 && m < input.w * input.h, `member index ${m} outside the image`);
    }
    // meanUx/meanUy is a mean DIRECTION, so it must be a unit vector -- a
    // region whose members' angles cancelled would leave it near zero, and
    // every downstream angle read off it would be noise.
    const len = Math.hypot(r.meanUx, r.meanUy);
    assert.ok(closeTo(len, 1, 1e-12), `mean direction is not unit (|u| = ${len})`);
  }
});

test('grown region count and sizes are unchanged (golden, fixtures/default)', () => {
  const s = input.settings;
  const { fx, fy } = computeGradient2x2Field(input.gray, input.w, input.h);
  const { regions } = growRegionsCCL(
    fx, fy, input.w, input.h,
    s.lsdToleranceDeg, s.lsdRhoNoiseThreshold, s.lsdRhoHighThreshold, s.lsdCclSteps, s.lsdMinRegionSize,
  );
  assert.equal(regions.length, 542);
  assert.deepEqual(regions.slice(0, 5).map((r) => r.members.length), [25, 13, 31, 25, 5]);
});

// One whole reconstruction, CPU backend, from the fixture. ~60ms in node, so it
// runs once and every assertion below reads the same state.
const state = await runPoseOn(input, 'cpu');

test('a CPU reconstruction recovers a pose at all', () => {
  assert.ok(state.lastVotes && state.lastVotes.length > 0, 'no votes');
  assert.ok(state.lastRecoveredAxes, 'no recovered axes');
  assert.ok(state.lastGridPeriodPhase, 'no grid period/phase');
  assert.ok(state.lastPositionDecode, 'no position decode');
});

test('the recovered axes are an orthonormal frame with a positive distance', () => {
  const a = state.lastRecoveredAxes!;
  for (const [name, v] of [['Drow', a.Drow], ['Dcol', a.Dcol], ['Dnormal', a.Dnormal]] as const) {
    assert.ok(closeTo(v.length(), 1, 1e-9), `${name} is not unit (|${name}| = ${v.length()})`);
  }
  // Orthogonality is the real claim: the fit solves for two plane families and
  // a normal, and a frame that has drifted out of square would still produce a
  // plausible-looking pose. 1e-6 rather than 1e-12 because the fit is a
  // least-squares solution, not a construction.
  assert.ok(Math.abs(a.Drow.dot(a.Dcol)) < 1e-6, `Drow.Dcol = ${a.Drow.dot(a.Dcol)}`);
  assert.ok(Math.abs(a.Drow.dot(a.Dnormal)) < 1e-6, `Drow.Dnormal = ${a.Drow.dot(a.Dnormal)}`);
  assert.ok(Math.abs(a.Dcol.dot(a.Dnormal)) < 1e-6, `Dcol.Dnormal = ${a.Dcol.dot(a.Dnormal)}`);
  assert.ok(a.distance > 0, `distance ${a.distance} is not positive`);
});

test('the decode agrees with the torus well above chance', () => {
  // HOW TO READ THIS, from reconstructionTiming.ts's header: consistency is the
  // fraction of sampled cells whose decoded bit matches the torus at the
  // winning registration, and its FAILURE FLOOR IS ~50% -- chance, meaning the
  // winner means nothing. The 60s are a good local patch, not a warning. So the
  // bar is set just clear of the floor: this asserts "the decode found real
  // structure", which is the thing that silently stops being true.
  const pd = state.lastPositionDecode!;
  assert.ok(pd.consistency > 0.55, `consistency ${pd.consistency} is at or near the chance floor`);
  assert.ok(Number.isFinite(pd.camPos.x) && Number.isFinite(pd.camPos.y) && Number.isFinite(pd.camPos.z));
  assert.ok(closeTo(pd.recoveredCamQuat.length(), 1, 1e-9), 'the recovered camera quaternion is not unit');
});

test('the recovered pose is unchanged (golden, fixtures/default, backend=cpu)', () => {
  // Recorded 2026-08-06 from this code on this fixture. See the header: this
  // says UNCHANGED, not CORRECT.
  const a = state.lastRecoveredAxes!;
  const pd = state.lastPositionDecode!;
  assert.equal(state.lastVotes!.length, 273);
  assert.ok(closeTo(a.distance, 10.764384913521246), `distance ${a.distance}`);
  assert.ok(closeTo(state.lastGridPeriodPhase!.period, 0.09289894481048244), `period ${state.lastGridPeriodPhase!.period}`);
  for (const [name, got, want] of [
    ['camPos.x', pd.camPos.x, 59.47930056578264],
    ['camPos.y', pd.camPos.y, 10.764384913521246],
    ['camPos.z', pd.camPos.z, 68.6335463810696],
  ] as const) {
    assert.ok(closeTo(got, want), `${name}: ${got} != ${want}`);
  }
});
