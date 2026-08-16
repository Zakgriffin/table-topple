import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import * as THREE from 'three';
import { TEST_CELL_PITCH, TEST_WORLD } from './helpers/board.ts';
import { type SimDims, type SimPose, camPosOf } from '../src/pose2/sim.ts';
import { type PoseObservation, type SweepSpec, generatePoses, runSweep, summarize } from '../src/pose2/sweep.ts';

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

/** A runner that reports the true pose displaced by a known amount. */
function fakeRunner(offset: { dCol: number; dRow: number; dHeight?: number }): (g: Float64Array, d: SimDims, p: SimPose) => Promise<PoseObservation> {
  return async (_gray, _dims, pose) => {
    const truth = camPosOf(TEST_WORLD, pose);
    return {
      camPos: new THREE.Vector3(
        truth.x + offset.dCol * TEST_CELL_PITCH,
        truth.y + (offset.dHeight ?? 0),
        truth.z + offset.dRow * TEST_CELL_PITCH,
      ),
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

test('a pipeline that recovers nothing is recorded, not dropped', async () => {
  const rows = await runSweep(ONE_POSE, async () => ({
    camPos: null, height: null, period: null, consistency: null, ms: 5,
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
