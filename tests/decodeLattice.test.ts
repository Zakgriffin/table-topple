import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { withDevice } from './helpers/gpu.ts';
import { boardDims } from '../src/pose2/board.ts';
import { createPose2Context, destroyPose2Context, runPose2 } from '../src/pose2/run.ts';
import { decodeLayout } from '../src/pose2/pose.ts';
import type { Dims } from '../src/pose2/pipeline.ts';
import { renderPose, vFovRadOf } from '../src/pose2/sim.ts';
import type { SimWorld } from '../src/pose2/sim.ts';
import { GRID_STEP } from '../src/sphereLab/constants.ts';
import { board } from '../src/sphereLab/floorPattern.ts';
import { buildDecodeLattice } from '../src/sphereLab/pipeline/decodeLattice.ts';

// SPHERE LAB's board, not tests/helpers/board.ts's -- `buildDecodeLattice` is
// app code and reads the app's `board` module binding, so the frame it is
// checked against has to be rendered on that same floor.
const WORLD: SimWorld = { board, cellPitch: GRID_STEP };

// ── The app's host copy of decode.correctness ─────────────────────────────
//
// `buildDecodeLattice` re-derives, per cell, what the device reduced to two
// integers -- so the Projected-Cam lattice can ring each dot green or red. It
// carries a self-check (its counts must equal the device's) and returns
// `correct: null` when that fails, so what has to be established here is that the
// self-check is not VACUOUS.
//
// It is vacuous at orientation 0, where the rotation mapping is the identity and
// a completely wrong `origIndex` would still agree. That is the documented shape
// of this bug -- "silently correct at orientation 0, which is most captures, and
// wrong at 1, 2 and 3" -- so this test's real job is to reach the other three.

const FRAME: Dims = {
  w: 96, h: 128, maxRegions: 4096, maxLines: 4096, ...boardDims(board),
};
const FRAME_DIMS = { w: FRAME.w, h: FRAME.h, horizFovDeg: 60 };
const INSPECT = ['layout', 'packed', 'result'] as const;

const SETTINGS = {
  grow: { rhoLow: 0.132, toleranceDeg: 9.5 },
  collect: { rhoHigh: 0, minRegionSize: 2 },
  lsdFit: { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 },
  lines: { minLengthPx: 3 },
  votes: { vFovRad: vFovRadOf(FRAME_DIMS) },
  gpp: { vFovRad: vFovRadOf(FRAME_DIMS), cellPitch: GRID_STEP, minGrazingCos: 0.15 },
  layout: { vFovRad: vFovRadOf(FRAME_DIMS), cellPitch: GRID_STEP, minGrazingCos: 0.15 },
};

// Four yaws a quarter turn apart, which is what makes the winning orientation
// move -- the pattern is not rotationally symmetric, so turning the camera 90
// degrees is exactly the case `origIndex` exists for.
const YAWS = [15, 105, 195, 285];

test('the host lattice agrees with the device at every cardinal orientation', async () => {
  await withDevice(async (device) => {
    const ctx = createPose2Context(device, FRAME, board, { inspect: INSPECT });
    try {
      const orientations = new Set<number>();
      for (const yawDeg of YAWS) {
        const gray = renderPose(
          WORLD, { height: 10, overRow: 40.1, overCol: 40.6, tiltDeg: 20, yawDeg }, FRAME_DIMS, 4);
        const { pose, inspected } = await runPose2(ctx, Float32Array.from(gray), SETTINGS, INSPECT);
        assert.equal(pose.ok, true, `yaw ${yawDeg} did not decode -- nothing here is meaningful`);
        orientations.add(pose.orientation);

        const layout = decodeLayout(inspected['layout']!);
        const lattice = buildDecodeLattice(layout, inspected['packed']!, inspected['result']!, pose);
        assert.ok(lattice, `yaw ${yawDeg}: no lattice from a frame that decoded`);
        // NON-NULL IS THE ASSERTION. buildDecodeLattice nulls this out precisely
        // when its own counts disagree with the device's, so a wrong rotation
        // mapping arrives here as an absence.
        assert.ok(lattice.correct, `yaw ${yawDeg} (orientation ${pose.orientation}): ` +
          `the host re-derivation disagreed with the device`);

        // ...and the counts are re-checked here rather than taken on trust, so
        // this test would still fail if the self-check were ever weakened.
        let correct = 0, wrong = 0;
        for (const v of lattice.correct) { if (v === 1) correct++; else if (v === 0) wrong++; }
        assert.equal(correct, pose.correct, `yaw ${yawDeg}: correct count`);
        assert.equal(wrong, pose.wrong, `yaw ${yawDeg}: wrong count`);
        assert.ok(correct > 100, `yaw ${yawDeg}: only ${correct} agreeing cells`);
        // Both counters have to be able to move, or the ratio is untested at one
        // end -- the same one-sided-gate trap §12's mutation run recorded.
        assert.ok(correct + wrong > 0);
      }

      // THE DECISIVENESS GATE, and the reason this test exists at all: at
      // orientation 0 `origIndex` is the identity and every assertion above
      // passes under any mutation of it.
      assert.ok(orientations.size >= 3,
        `only reached orientations {${[...orientations].join(',')}} -- this fixture cannot ` +
        `see a wrong rotation mapping`);
    } finally {
      destroyPose2Context(ctx);
    }
  });
});
