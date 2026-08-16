import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { TEST_CELL_PITCH, TEST_WORLD } from './helpers/board.ts';
import { type SimDims, type SimPose, camPosOf, camQuatOf, rayDirInto, renderPose, vFovRadOf } from '../src/pose/sim.ts';
import { cornerDir } from '../src/poseViewer/math/geometry.ts';

// ── Validating the SIMULATOR, not the pipeline ────────────────────────────
//
// The sweep's whole value rests on the rendered image being a faithful inverse
// of what the pipeline projects. If the renderer has a sign or axis error, the
// sweep reports pipeline "error" that is really simulator error -- and the two
// are indistinguishable from inside the sweep.
//
// So the simulator WAS validated against the existing CPU pipeline first, while
// that pipeline was still the known-good one: a mature, independently-tested
// implementation recovering the pose that generated an image says the renderer
// and the truth derivation are both sound.
//
// THAT BOOTSTRAP IS OVER. The header used to say this dependency on the old pipeline was
// temporary and "retires when the old pipeline does" -- which is exactly what happened.
// See the note further down for what the three deleted tests checked, why they
// were not re-pointed at src/pose, and what the deletion costs.

const DIMS: SimDims = { w: 480, h: 640, horizFovDeg: 65 };


test('rayDirInto is bit-for-bit cornerDir', () => {
  // THE TEST THAT MAKES THE DUPLICATION SAFE. renderPose cannot call cornerDir
  // -- it allocates a Vector3 per ray and a sweep is hundreds of millions of
  // rays -- so the projection is written out into scalars instead. Duplicated
  // projection maths is exactly the drift that would make simulator error
  // indistinguishable from pipeline error, so the copy is held to the original
  // here rather than trusted.
  //
  // Exact equality, not a tolerance: the same operations in the same order on
  // the same doubles must produce the same bits. A tolerance would hide a
  // genuine reordering.
  const aspect = DIMS.w / DIMS.h;
  const tanHalf = Math.tan(vFovRadOf(DIMS) / 2);
  const out = { x: 0, y: 0, z: 0 };
  let checked = 0;
  for (const pose of [
    { height: 10, overRow: 0, overCol: 0, tiltDeg: 0, yawDeg: 0 },
    { height: 10, overRow: 0, overCol: 0, tiltDeg: 37, yawDeg: 0 },
    { height: 10, overRow: 0, overCol: 0, tiltDeg: 20, yawDeg: 123 },
    { height: 10, overRow: 0, overCol: 0, tiltDeg: 55, yawDeg: -80, rollDeg: 17 },
  ] as SimPose[]) {
    const q = camQuatOf(pose);
    for (let i = 0; i <= 12; i++) {
      for (let j = 0; j <= 12; j++) {
        const ndcU = -1 + (2 * i) / 12, ndcV = -1 + (2 * j) / 12;
        rayDirInto(out, ndcU, ndcV, q, tanHalf, aspect);
        const ref = cornerDir(ndcU, ndcV, q, vFovRadOf(DIMS), aspect);
        assert.equal(out.x, ref.x, `x at (${ndcU},${ndcV})`);
        assert.equal(out.y, ref.y, `y at (${ndcU},${ndcV})`);
        assert.equal(out.z, ref.z, `z at (${ndcU},${ndcV})`);
        checked++;
      }
    }
  }
  assert.equal(checked, 4 * 13 * 13);
});

test('render: produces a real image, not a constant', () => {
  const gray = renderPose(TEST_WORLD, { height: 12, overRow: 70, overCol: 70, tiltDeg: 20, yawDeg: 0 }, DIMS);
  assert.equal(gray.length, DIMS.w * DIMS.h);
  let min = Infinity, max = -Infinity, sum = 0;
  for (const g of gray) { if (g < min) min = g; if (g > max) max = g; sum += g; }
  assert.ok(min < 20, `expected dark cells, min was ${min}`);
  assert.ok(max > 235, `expected light cells, max was ${max}`);
  // A De Bruijn pattern is roughly balanced, so the mean should sit near
  // mid-grey. Far from it means most of the frame is sky or one bit value.
  assert.ok(sum / gray.length > 80 && sum / gray.length < 175, `mean ${sum / gray.length}`);
});

test('render: a nadir view is centred on the cell under the camera', () => {
  // Straight down, camera over the centre of cell (70, 70). The centre pixel's
  // ray goes straight down, so it must sample exactly that cell -- which pins
  // the world<->board mapping and the ray direction together.
  const pose: SimPose = { height: 6, overRow: 70.5, overCol: 70.5, tiltDeg: 0, yawDeg: 0 };
  const cam = camPosOf(TEST_WORLD, pose);
  assert.ok(Math.abs(cam.x - (70.5 + 0.5 - TEST_WORLD.board.C / 2) * TEST_CELL_PITCH) < 1e-12);
  assert.ok(Math.abs(cam.z - (70.5 + 0.5 - TEST_WORLD.board.R / 2) * TEST_CELL_PITCH) < 1e-12);
  assert.equal(cam.y, 6);
});

test('render: supersampling changes edge pixels but not cell interiors', () => {
  const pose: SimPose = { height: 10, overRow: 70, overCol: 70, tiltDeg: 15, yawDeg: 0 };
  const a = renderPose(TEST_WORLD, pose, DIMS, 1);
  const b = renderPose(TEST_WORLD, pose, DIMS, 3);
  let differing = 0, intermediate = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) differing++;
    if (b[i]! > 5 && b[i]! < 250) intermediate++;
  }
  // Antialiasing has to actually do something, or the gradient field it feeds
  // is a staircase and every downstream stage inherits that.
  assert.ok(differing > 0, 'supersampling changed nothing');
  assert.ok(intermediate > 100, `expected soft edges, got ${intermediate} intermediate pixels`);
});

// ── VALIDATING THE SIMULATOR ──────────────────────────────────────────────
//
// Position error decomposes into two independent parts, and keeping them apart
// is what lets these tests validate the RENDERER without also having to be
// right about the pipeline's accuracy:
//
//   CONTINUOUS  -- the geometry: period, height, and the sub-cell position.
//                  A renderer with a sign, axis or scale error shows up here.
//   DISCRETE    -- whole-cell jumps from the decode picking a neighbouring
//                  anchor. A property of the pipeline, not of the image.
//
// ── A RETRACTED FINDING, kept because the retraction is the lesson ──
//
// An earlier version of this comment recorded a table of whole-cell anchor
// errors at tilts 5-20 and called it a real accuracy result about the old pipeline.
// It was not. It was THIS FILE under-sampling diagonal edges: at supersample 2
// a diagonal renders as a staircase, the level-line directions quantize to the
// staircase, and directed growth splits one line into many. Raising the default
// to 4 made the anchor exact at all 180 sweep poses.
//
// The nadir validation below passed the whole time, because axis-aligned edges
// do not alias that way. **A simulator is validated at the poses you validated
// it at** -- agreeing to 0.038 cells at nadir said nothing whatever about
// diagonals. See renderPose's header for the measured supersampling table.

// ── THE THREE SIMULATOR-VALIDATION TESTS THAT USED TO SIT HERE ARE DELETED ──
//
// They ran `runPoseOn(..., 'cpu')` -- the old pipeline -- over a rendered frame and
// checked that the mature pipeline recovered the generating pose, which is how
// the renderer earned the right to judge anything (§19, Phase 1). the old pipeline is
// gone, so they went with it.
//
// They were NOT re-pointed at src/pose, and that is the deliberate part. Their
// whole value was that the oracle was an INDEPENDENT implementation: renderPose
// casts through the same `cornerDir` the pipeline projects with, so a shared
// projection error is invisible to any check that uses only those two. Running
// them against pose would have kept three green tests while quietly converting
// a cross-check into a circular one -- the worse outcome, because it reads as
// coverage.
//
// WHAT THE DELETION COSTS, stated rather than hidden: nothing now re-derives
// that the renderer's geometry is right from outside the pose + sim pair. What
// survives is the bit-for-bit `rayDirInto` vs `cornerDir` test below, the
// closed-form checks in this file, and the 180-pose sweep -- all of which share
// that projection. The measurements the deleted tests produced (0.038 cells at
// nadir, consistency 1.000, period error -0.01%) are recorded in §19 and are not
// reproducible any more.
