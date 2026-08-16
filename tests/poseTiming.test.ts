import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { getBareTestDevice, getTestDevice, withDevice } from './helpers/gpu.ts';
import { countingDevice } from './helpers/countingDevice.ts';
import { canTimestamp } from '../src/gpu/device.ts';
import { boardDims } from '../src/pose/board.ts';
import { MAX_TIMED_PASSES, createPoseContext, destroyPoseContext, runPose } from '../src/pose/run.ts';
import type { Dims } from '../src/pose/pipeline.ts';
import { renderPose, vFovRadOf } from '../src/pose/sim.ts';
import { TEST_BOARD, TEST_CELL_PITCH, TEST_WORLD } from './helpers/board.ts';

// ── GPU timing: the device feature, and the path where it is ABSENT ───────
//
// `timestamp-query` is requested at device creation when the adapter offers it,
// and cannot be asked for later -- so a device either can be timed for its whole
// life or cannot.
//
// The half worth testing is the ABSENT one, because it is the half that will
// never run on the developer's machine. This adapter offers the feature (§5
// measured its 41.7 ns tick against this same Dawn build), so every ordinary
// test run exercises only the timed path, and a pipeline that had quietly come
// to REQUIRE timestamps would look perfectly healthy right up until it reached
// hardware that lacks them.
//
// So the absence is manufactured rather than waited for: a second device is
// created asking for no optional features at all. That is the only honest way to
// do it -- the absence is a property of a device, fixed at creation, and no flag
// on the shared device can imitate it.

const FRAME: Dims = {
  w: 96, h: 128, maxRegions: 4096, maxLines: 4096, ...boardDims(TEST_BOARD),
};
const FRAME_DIMS = { w: FRAME.w, h: FRAME.h, horizFovDeg: 60 };
const POSE = { height: 10, overRow: 40.1, overCol: 40.6, tiltDeg: 20, yawDeg: 15 };

const SETTINGS = {
  grow: { rhoLow: 0.132, toleranceDeg: 9.5 },
  collect: { rhoHigh: 0, minRegionSize: 2 },
  lsdFit: { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 },
  lines: { minLengthPx: 3 },
  votes: { vFovRad: vFovRadOf(FRAME_DIMS) },
  gpp: { vFovRad: vFovRadOf(FRAME_DIMS), cellPitch: TEST_CELL_PITCH, minGrazingCos: 0.15 },
  layout: { vFovRad: vFovRadOf(FRAME_DIMS), cellPitch: TEST_CELL_PITCH, minGrazingCos: 0.15 },
};

async function frameOn(device: GPUDevice, gray: Float64Array) {
  const ctx = createPoseContext(device, FRAME, TEST_BOARD, {});
  try {
    return await runPose(ctx, Float32Array.from(gray), SETTINGS);
  } finally {
    destroyPoseContext(ctx);
  }
}

async function poseOn(device: GPUDevice, gray: Float64Array) {
  return (await frameOn(device, gray)).pose;
}

/**
 * One frame with the WebGPU calls counted. The counters SHADOW methods on the
 * shared device, so restoring them is not optional -- see countingDevice.ts.
 */
async function countedFrame(real: GPUDevice) {
  const c = countingDevice(real);
  try {
    const frame = await frameOn(c.device, renderPose(TEST_WORLD, POSE, FRAME_DIMS, 4));
    return { frame, counts: c.counts };
  } finally {
    c.restore();
  }
}

// The precondition for everything below, and it is NOT a tautology: it asserts
// that requestDeviceWithOptionalTimestamps actually asked for the feature. A
// device only has what it was created with, so if the helper silently dropped
// the request this fails here rather than presenting later as "the GPU cannot
// be timed on this machine".
test('the shared test device asked for and got timestamp-query', async () => {
  const device = await getTestDevice();
  assert.ok(device, 'no WebGPU device available for tests');
  assert.equal(canTimestamp(device), true);
});

// The check that the check works. Without this, the absent-path test below
// could be running against a device that quietly HAS the feature, and would
// pass while testing nothing -- a success that came for free because nothing
// ever had to read the thing it claims to cover.
test('a bare device genuinely lacks the feature, so the absent path is real', async () => {
  const bare = await getBareTestDevice();
  assert.ok(bare, 'could not create a second device');
  assert.equal(canTimestamp(bare), false);

  const shared = await getTestDevice();
  assert.ok(shared);
  assert.notEqual(
    canTimestamp(bare), canTimestamp(shared),
    'the two devices must DIFFER in this feature, or nothing here distinguishes them',
  );
});

test('the pipeline recovers the same pose on a device that cannot be timed', async () => {
  const gray = renderPose(TEST_WORLD, POSE, FRAME_DIMS, 4);

  const bare = await getBareTestDevice();
  assert.ok(bare, 'could not create a second device');
  // withDevice's error scope, by hand: this device is not the shared one. A
  // WebGPU validation failure does not throw -- it makes commands silent no-ops
  // -- so without the scope a pipeline that failed validation on this device
  // would report plausible zeros and the assertions would blame the pose.
  bare.pushErrorScope('validation');
  let untimed;
  try {
    untimed = await poseOn(bare, gray);
  } finally {
    const err = await bare.popErrorScope();
    if (err) throw new Error(`validation error on the untimed device: ${err.message}`);
  }

  assert.equal(untimed.ok, true, 'the untimed device must still recover a pose');

  await withDevice(async (device) => {
    const timed = await poseOn(device, gray);
    assert.equal(timed.ok, true);
    // Identical, not merely close. TODAY THIS IS TRIVIALLY TRUE -- nothing yet
    // writes a timestamp, so the two runs execute the same commands and this
    // assertion is green for free. It is written now because it is Phase 3's
    // stated gate ("byte-identical pose with timing on and off"), and having it
    // already passing means that when Phase 3 adds timestampWrites, any drift it
    // introduces surfaces here immediately rather than being discovered as an
    // accuracy regression much later.
    assert.deepStrictEqual(untimed, timed);
  });
});

// ── Phase 3's gate ────────────────────────────────────────────────────────

test('timings are reported, one per encoded pass, in encode order', async () => {
  await withDevice(async (real) => {
    // Counted from OUTSIDE, off the WebGPU calls themselves. Asking the library
    // how many passes it encoded and comparing that to what it reported would
    // be one variable checked against itself.
    const { frame, counts } = await countedFrame(real);

    assert.ok(frame.gpu, 'a timed device must report gpu timings');
    assert.equal(
      frame.gpu.passes.length, counts.passes,
      'every encoded compute pass must be reported, and nothing else',
    );
    assert.ok(counts.passes > 0, 'the frame must have encoded some passes');

    // `index` is the pass's position in the submit, which is what lets a
    // consumer order them without re-deriving the encode order.
    frame.gpu.passes.forEach((t, i) => assert.equal(t.index, i));
    // The stage ids must match the pass LABELS the encoder actually saw, in
    // order -- this is what catches a timer that drifted out of step with the
    // passes (recording a stage it did not begin, or skipping one).
    assert.deepStrictEqual(frame.gpu.passes.map((t) => t.stage), counts.passLabels);
  });
});

test('the whole frame is still one submit, one fence, one map', async () => {
  await withDevice(async (real) => {
    const { frame, counts } = await countedFrame(real);

    assert.ok(frame.gpu, 'this must be the timed path, or the check is vacuous');
    // The property the whole pipeline is shaped around, and the one timestamps
    // could most easily have cost: resolveQuerySet and its copy go into the
    // encoder that was already there, so the counts must not move.
    assert.equal(counts.submits, 1, 'timing must not add a submit');
    assert.equal(counts.maps, 1, 'timing must not add a fence');
  });
});

test('grow repeats its stage ids, and the durations are real device time', async () => {
  await withDevice(async (real) => {
    const frame = await frameOn(real, renderPose(TEST_WORLD, POSE, FRAME_DIMS, 4));
    assert.ok(frame.gpu);

    // Repetition is expected, not a bug to normalize away: hook/compress/gate
    // are re-encoded once per convergence round. If these ever collapse to one
    // occurrence each, the round loop stopped being encoded.
    const hooks = frame.gpu.passes.filter((t) => t.stage === 'grow.hook');
    assert.ok(hooks.length > 1, `grow.hook should repeat per round, saw ${hooks.length}`);

    // Non-negative by construction (end minus begin on one monotonic counter),
    // and the total has to be a plausible frame rather than a pile of zeros --
    // which is what an unwritten or unresolved query set would produce.
    const totalNs = frame.gpu.passes.reduce((a, t) => a + t.ns, 0);
    assert.ok(frame.gpu.passes.every((t) => t.ns >= 0), 'a duration cannot be negative');
    assert.ok(totalNs > 0, 'the summed GPU time cannot be zero');
    // §5 measured a ~2.4 us floor for an EMPTY pass on this adapter, so a frame
    // of a few hundred passes cannot plausibly come in under a microsecond, and
    // a full second would mean the counter is not what we think it is.
    assert.ok(
      totalNs > 1_000 && totalNs < 1_000_000_000,
      `summed GPU time ${totalNs}ns is outside any plausible range`,
    );
  });
});

// ── THE COUNTER COARSENS AFTER ~2 FRAMES, AND IT NEVER RECOVERS ──────────
//
// Measured 2026-08-15 under the `webgpu` (Dawn) node binding. One context, ten
// frames, no destroys: the first two frames report ~90 DISTINCT per-pass
// durations on the expected 41.667 ns grid, and from the third frame onward
// every raw timestamp is quantized to 65536 ns (2^16) -- so most passes have
// begin and end in the SAME tick and read exactly zero.
//
// It is not reuse and not `destroy`: reordering the experiment moves the
// degradation with the frame COUNT, not with any object's lifecycle. The
// implementation reads the raw u64s faithfully; the device simply stops
// reporting fine ones.
//
// This QUALIFIES the design's §5, which measured a 41.667 ns tick with "no
// coarsening at any scale" against this same Dawn build -- that probe cannot
// have run many submits. Whether Chrome does the same is UNKNOWN and matters a
// great deal, because Phase 7's whole payoff is the sweep, which runs here.
//
// So this test asserts STRUCTURE, which is sound on every frame, and asserts
// resolution only on an EARLY frame, where it is real.
test('an early frame resolves individual passes, on the fine counter', async () => {
  await withDevice(async (device) => {
    const gray = renderPose(TEST_WORLD, POSE, FRAME_DIMS, 4);
    const ctx = createPoseContext(device, FRAME, TEST_BOARD, {});
    try {
      // The FIRST frame this context runs. Other tests in this file have
      // already run frames on the shared device, so this is not necessarily
      // the process's first -- which is exactly why the assertion below is a
      // floor on information, not on the tick size.
      const frame = await runPose(ctx, Float32Array.from(gray), SETTINGS);
      assert.ok(frame.gpu);
      const distinct = new Set(frame.gpu.passes.map((t) => t.ns)).size;
      const zeros = frame.gpu.passes.filter((t) => t.ns === 0).length;
      console.error(`      early frame: ${distinct} distinct durations, ${zeros}/${frame.gpu.passes.length} zero`);
      // A frame carrying only a handful of distinct values is the coarse
      // counter. Two is the floor for "this told me something per pass".
      assert.ok(distinct >= 2, `only ${distinct} distinct duration(s) -- the counter gave nothing`);
    } finally { destroyPoseContext(ctx); }
  });
});

// The instrument reading itself out. This is the first per-stage GPU breakdown
// this project has ever had for src/pose, and printing it is most of the point
// of building it -- the numbers are small here (96x128, not the sweep's 480x640)
// so treat the SHAPE as the finding and not the magnitudes.
test('the frame fits the query set, and reports where its GPU time went', async () => {
  await withDevice(async (real) => {
    const t0 = performance.now();
    const { frame, counts } = await countedFrame(real);
    const wallMs = performance.now() - t0;
    assert.ok(frame.gpu);

    // The capacity check that keeps MAX_TIMED_PASSES honest. It is a bound, not
    // a prediction, so what matters is that there is real headroom -- and that
    // this number moves visibly here when the pipeline gains passes.
    assert.ok(
      counts.passes < MAX_TIMED_PASSES,
      `${counts.passes} passes against a capacity of ${MAX_TIMED_PASSES}`,
    );

    const byStage = new Map<string, { ns: number; n: number }>();
    for (const t of frame.gpu.passes) {
      const e = byStage.get(t.stage) ?? { ns: 0, n: 0 };
      e.ns += t.ns; e.n++;
      byStage.set(t.stage, e);
    }
    const totalNs = frame.gpu.passes.reduce((a, t) => a + t.ns, 0);
    const top = [...byStage.entries()].sort((a, b) => b[1].ns - a[1].ns).slice(0, 8);

    console.error(
      `    ${counts.passes} passes / ${MAX_TIMED_PASSES} capacity, `
      + `${(totalNs / 1e6).toFixed(2)}ms GPU of ${wallMs.toFixed(2)}ms wall `
      + `(${((totalNs / 1e6 / wallMs) * 100).toFixed(0)}% on device)`);
    for (const [stage, e] of top) {
      console.error(
        `      ${stage.padEnd(24)} ${(e.ns / 1e6).toFixed(3)}ms`
        + `${e.n > 1 ? `  n=${e.n}` : ''}`);
    }
    // Printed so a reader can see whether this frame was on the fine counter or
    // the coarse one -- without it the breakdown above looks equally
    // authoritative either way, which is the trap this whole section documents.
    const distinct = new Set(frame.gpu.passes.map((t) => t.ns)).size;
    console.error(
      `      ${distinct} distinct durations, ${frame.gpu.passes.filter((t) => t.ns === 0).length} zero`
      + `${distinct < 10 ? '  <-- COARSE COUNTER, magnitudes above are not usable' : ''}`);
  });
});
