import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import * as THREE from 'three';
import { TEST_CELL_PITCH, TEST_WORLD } from './helpers/board.ts';
import { type SimDims, type SimPose, camPosOf, camQuatOf } from '../src/pose/sim.ts';
import { type PoseObservation, type SweepSpec, generatePoses, runSweep, summarize } from '../src/pose/sweep.ts';

// ── The sweep machinery, with a FAKE pipeline ─────────────────────────────
//
// These test the harness, not any pipeline. A fake runner returns a pose with a
// known, deliberately-constructed error, so the error decomposition can be
// checked against an answer computed by hand.
//
// That matters more than it sounds: the decomposition is the thing that made
// the first real sweep readable -- it separated "the anchor is one cell out"
// from "the projection is wrong", which a single RMS number merges into one
// uninterpretable value. If the split is computed wrongly, every conclusion
// drawn from a sweep is wrong in a way that still looks plausible.

const DIMS: SimDims = { w: 32, h: 32, horizFovDeg: 65 };

/**
 * A runner that reports the true pose displaced by a known amount.
 *
 * `dTiltDeg` and `dYawDeg` perturb the ORIENTATION, and the two are not
 * interchangeable: tilt moves the floor normal, yaw spins about it. The
 * orientation tests below turn exactly that distinction into assertions.
 */
function fakeRunner(offset: {
  dCol: number; dRow: number; dHeight?: number; dTiltDeg?: number; dYawDeg?: number;
}): (g: Float64Array, d: SimDims, p: SimPose) => Promise<PoseObservation> {
  return async (_gray, _dims, pose) => {
    const truth = camPosOf(TEST_WORLD, pose);
    return {
      camPos: new THREE.Vector3(
        truth.x + offset.dCol * TEST_CELL_PITCH,
        truth.y + (offset.dHeight ?? 0),
        truth.z + offset.dRow * TEST_CELL_PITCH,
      ),
      // Built by the SAME function that builds truth, from perturbed angles --
      // so a convention mistake cancels on both sides and cannot be mistaken
      // for the error being measured.
      camQuat: camQuatOf({
        ...pose,
        tiltDeg: pose.tiltDeg + (offset.dTiltDeg ?? 0),
        yawDeg: pose.yawDeg + (offset.dYawDeg ?? 0),
      }),
      height: pose.height + (offset.dHeight ?? 0),
      period: TEST_CELL_PITCH / pose.height,
      consistency: 0.9,
      ms: 1,
    };
  };
}

const ONE_POSE: SweepSpec = {
  heights: [10], tilts: [0], yaws: [0], offsets: [{ row: 70.5, col: 70.5 }],
  dims: DIMS, supersample: 1, world: TEST_WORLD,
};

test('generatePoses is the full cartesian product', () => {
  const poses = generatePoses({
    heights: [6, 10], tilts: [0, 20, 40], yaws: [0, 90],
    offsets: [{ row: 1, col: 2 }], dims: DIMS, world: TEST_WORLD,
  });
  assert.equal(poses.length, 2 * 3 * 2 * 1);
  assert.equal(new Set(poses.map((p) => JSON.stringify(p))).size, poses.length, 'poses must be distinct');
});

test('a perfect pipeline scores zero error', async () => {
  const rows = await runSweep(ONE_POSE, fakeRunner({ dCol: 0, dRow: 0 }));
  assert.equal(rows.length, 1);
  const r = rows[0]!;
  assert.equal(r.recovered, true);
  assert.equal(r.anchorErr, 0);
  assert.ok(r.subCellErr < 1e-12);
  assert.ok(Math.abs(r.heightErrRel) < 1e-12);
});

test('a whole-cell offset is DISCRETE, and leaves the continuous part clean', async () => {
  // Exactly one cell out in row. This is the anchor-off-by-N signature the
  // first real sweep found, and the split must attribute all of it to the
  // discrete side -- reporting 1.0 cells of "geometry error" would send anyone
  // reading it hunting for a projection bug that does not exist.
  const rows = await runSweep(ONE_POSE, fakeRunner({ dCol: 0, dRow: -1 }));
  const r = rows[0]!;
  assert.equal(r.cellRow, -1);
  assert.equal(r.cellCol, 0);
  assert.equal(r.anchorErr, 1);
  assert.ok(r.subCellErr < 1e-12, `sub-cell error should be zero, got ${r.subCellErr}`);
  assert.ok(Math.abs(r.posErr - 1) < 1e-12);
});

test('a sub-cell offset is CONTINUOUS, and leaves the anchor clean', async () => {
  const rows = await runSweep(ONE_POSE, fakeRunner({ dCol: 0.2, dRow: -0.1 }));
  const r = rows[0]!;
  assert.equal(r.anchorErr, 0, 'a fifth of a cell must not read as a wrong anchor');
  assert.ok(Math.abs(r.subCellErr - Math.hypot(0.2, 0.1)) < 1e-12);
});

test('the two parts separate when both are present', async () => {
  // 2.25 cells out: two whole cells of anchor error plus a quarter cell of
  // geometry. A sweep has to be able to say which of those got worse.
  const rows = await runSweep(ONE_POSE, fakeRunner({ dCol: 2.25, dRow: 0 }));
  const r = rows[0]!;
  assert.equal(r.cellCol, 2);
  assert.ok(Math.abs(r.subCellErr - 0.25) < 1e-12);
  assert.ok(Math.abs(r.posErr - 2.25) < 1e-12);
});

// ── The orientation split, which the 90-degree ambiguity forces ───────────
//
// A square grid fixes the camera's orientation only up to a quarter turn about
// the floor normal, and `pose.quaternion` is written AFTER decode resolves that
// from the De Bruijn tally. So orientation error is two unrelated quantities:
// a continuous one the line fit controls, and a discrete one decode controls.
// Reporting them as a single angle would let one wrong quadrant read as a
// 90-degree fit error and bury every continuous change underneath it.
//
// The pair of tests below is the proof that the split is REAL and not just
// computed: a pure tilt error must appear only in the normal, a pure yaw error
// only in the yaw. If either leaked into the other, both numbers would still
// look plausible and neither would mean anything.

test('a tilt error moves the normal and leaves the yaw clean', async () => {
  const rows = await runSweep(ONE_POSE, fakeRunner({ dCol: 0, dRow: 0, dTiltDeg: 3 }));
  const r = rows[0]!;
  assert.ok(Math.abs(r.normalErrDeg - 3) < 1e-9, `normal error should be 3 deg, got ${r.normalErrDeg}`);
  assert.ok(r.yawErrDeg < 1e-9, `tilt must not leak into yaw, got ${r.yawErrDeg}`);
  assert.equal(r.cardinalTurns, 0, 'a 3-degree tilt is not a quadrant change');
});

test('a yaw error moves the yaw and leaves the normal clean', async () => {
  const rows = await runSweep(ONE_POSE, fakeRunner({ dCol: 0, dRow: 0, dYawDeg: 5 }));
  const r = rows[0]!;
  assert.ok(r.normalErrDeg < 1e-9, `yaw must not leak into the normal, got ${r.normalErrDeg}`);
  assert.ok(Math.abs(r.yawErrDeg - 5) < 1e-9, `yaw error should be 5 deg, got ${r.yawErrDeg}`);
  assert.equal(r.cardinalTurns, 0);
});

test('a quarter turn is reported as a QUADRANT CHANGE, not as 90 degrees of error', async () => {
  // THE POINT OF THE WHOLE SPLIT. The board cannot tell this pose from the true
  // one geometrically -- only the pattern can -- so the continuous metrics must
  // read zero and the discrete counter must fire. A metric that reported 90
  // degrees here would make every real fit improvement invisible next to it.
  const rows = await runSweep(ONE_POSE, fakeRunner({ dCol: 0, dRow: 0, dYawDeg: 90 }));
  const r = rows[0]!;
  assert.ok(r.normalErrDeg < 1e-9, 'a quarter turn is about the normal, so the normal is untouched');
  assert.ok(r.yawErrDeg < 1e-9, `a quarter turn is exactly representable, got ${r.yawErrDeg}`);
  assert.notEqual(r.cardinalTurns, 0, 'the quadrant change must be COUNTED, not smoothed away');

  // Half a turn pins the count exactly, and unlike a quarter turn its value
  // does not depend on which way round the rotation is measured.
  const half = await runSweep(ONE_POSE, fakeRunner({ dCol: 0, dRow: 0, dYawDeg: 180 }));
  assert.equal(half[0]!.cardinalTurns, 2);
  assert.ok(half[0]!.yawErrDeg < 1e-9);

  // And the summary has to surface it, or the count is unreadable.
  assert.match(summarize(rows, TEST_WORLD), /cardinal exact\s+0\/1/);
});

test('a runner that reports no orientation is not scored as a perfect one', async () => {
  // NaN, not zero: `stats` drops non-finite values, so an absent measurement
  // leaves the quantiles alone instead of pulling them toward zero.
  const rows = await runSweep(ONE_POSE, async (_g, _d, pose) => ({
    camPos: camPosOf(TEST_WORLD, pose), camQuat: null,
    height: pose.height, period: TEST_CELL_PITCH / pose.height, consistency: 0.9, ms: 1,
  }));
  assert.equal(rows[0]!.recovered, true);
  assert.ok(Number.isNaN(rows[0]!.normalErrDeg));
  assert.ok(Number.isNaN(rows[0]!.yawErrDeg));
});

test('the scale of a pose is recorded alongside its error', async () => {
  // 1.274 cells per unit of height at a 65-degree horizontal fov, and the cell
  // pitch divides out -- so this is arithmetic, not a fixture to be updated.
  const rows = await runSweep(ONE_POSE, fakeRunner({ dCol: 0, dRow: 0 }));
  const r = rows[0]!;
  const expected = (2 * 10 * Math.tan((65 * Math.PI) / 360)) / TEST_CELL_PITCH;
  assert.ok(Math.abs(r.cellsAcross - expected) < 1e-9, `cellsAcross ${r.cellsAcross} vs ${expected}`);
  assert.ok(Math.abs(r.pxPerCell - DIMS.w / expected) < 1e-9);
});

test('a pipeline that recovers nothing is recorded, not dropped', async () => {
  const rows = await runSweep(ONE_POSE, async () => ({
    camPos: null, camQuat: null, height: null, period: null, consistency: null, ms: 5,
  }));
  assert.equal(rows[0]!.recovered, false);
  // And the summary has to SAY so rather than quietly averaging over one pose.
  const text = summarize(rows, TEST_WORLD);
  assert.match(text, /recovered\s+0\/1/);
  assert.match(text, /NO POSE/);
});

test('the summary names which poses failed', async () => {
  // An aggregate that hides WHICH pose broke is not actionable, and finding
  // where things break down is the entire purpose of the sweep.
  const spec: SweepSpec = { ...ONE_POSE, tilts: [0, 30] };
  const rows = await runSweep(spec, fakeRunner({ dCol: 0, dRow: -1 }));
  const text = summarize(rows, TEST_WORLD, 'unit');
  assert.match(text, /anchor exact\s+0\/2/);
  assert.match(text, /tilt=0/);
  assert.match(text, /tilt=30/);
  assert.match(text, /anchor off \(-1, 0\)/);
});
