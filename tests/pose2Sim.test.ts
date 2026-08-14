import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { C, R } from '../src/sphereLab/floorPattern.ts';
import { GRID_STEP } from '../src/sphereLab/constants.ts';
import { runPoseOn } from '../src/sphereLab/harness/runPose.ts';
import { loadFixture } from './helpers/fixtures.ts';
import { inputFromFixture } from '../src/sphereLab/harness/input.ts';
import { type SimDims, type SimPose, camPosOf, camQuatOf, rayDirInto, renderPose, truthFor, vFovRadOf } from '../src/pose2/sim.ts';
import { cornerDir } from '../src/sphereLab/math/geometry.ts';

// ── Validating the SIMULATOR, not the pipeline ────────────────────────────
//
// The sweep's whole value rests on the rendered image being a faithful inverse
// of what the pipeline projects. If the renderer has a sign or axis error, the
// sweep reports pipeline "error" that is really simulator error -- and the two
// are indistinguishable from inside the sweep.
//
// So the simulator is validated against the EXISTING CPU pipeline first, while
// that pipeline is still the known-good one. If a mature, independently-tested
// implementation recovers the pose that generated an image, the renderer and
// the truth derivation are both sound. Only then is the simulator fit to judge
// anything.
//
// This is deliberately a dependency on src/pose, and it is temporary: it exists
// to bootstrap trust in the harness, and it retires when src/pose does.

const DIMS: SimDims = { w: 480, h: 640, horizFovDeg: 65 };

const fmt = (v: { x: number; y: number; z: number }) =>
  `(${v.x.toFixed(2)}, ${v.y.toFixed(2)}, ${v.z.toFixed(2)})`;

/** The fixture's settings, with our own image. Reuses tuning that is known to
 *  work on a real capture rather than inventing thresholds here. */
function inputFor(pose: SimPose, dims: SimDims = DIMS, supersample = 2) {
  const base = inputFromFixture(loadFixture());
  return {
    ...base,
    label: `sim:h${pose.height}/t${pose.tiltDeg}/y${pose.yawDeg}`,
    gray: renderPose(pose, dims, supersample),
    w: dims.w,
    h: dims.h,
    aspect: dims.w / dims.h,
  };
}

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
  const gray = renderPose({ height: 12, overRow: 70, overCol: 70, tiltDeg: 20, yawDeg: 0 }, DIMS);
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
  const cam = camPosOf(pose);
  assert.ok(Math.abs(cam.x - (70.5 + 0.5 - C / 2) * GRID_STEP) < 1e-12);
  assert.ok(Math.abs(cam.z - (70.5 + 0.5 - R / 2) * GRID_STEP) < 1e-12);
  assert.equal(cam.y, 6);
});

test('render: supersampling changes edge pixels but not cell interiors', () => {
  const pose: SimPose = { height: 10, overRow: 70, overCol: 70, tiltDeg: 15, yawDeg: 0 };
  const a = renderPose(pose, DIMS, 1);
  const b = renderPose(pose, DIMS, 3);
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
// errors at tilts 5-20 and called it a real accuracy result about src/pose.
// It was not. It was THIS FILE under-sampling diagonal edges: at supersample 2
// a diagonal renders as a staircase, the level-line directions quantize to the
// staircase, and directed growth splits one line into many. Raising the default
// to 4 made the anchor exact at all 180 sweep poses.
//
// The nadir validation below passed the whole time, because axis-aligned edges
// do not alias that way. **A simulator is validated at the poses you validated
// it at** -- agreeing to 0.038 cells at nadir said nothing whatever about
// diagonals. See renderPose's header for the measured supersampling table.

test('SIMULATOR VALIDATED: a nadir view round-trips through the existing CPU pipeline', async () => {
  // The strictest case and the one that pins every convention at once: looking
  // straight down, the recovered camera position must be the one that generated
  // the image. If the renderer had a sign flip, an axis swap, a half-cell phase
  // offset or the wrong bit polarity, this is where it would show.
  const pose: SimPose = { height: 10, overRow: 70.5, overCol: 70.5, tiltDeg: 0, yawDeg: 0 };
  const truth = truthFor(pose);
  const { pose: result } = await runPoseOn(inputFor(pose), 'cpu');

  assert.ok(result.recoveredAxes, 'no axes recovered from the simulated frame');
  assert.ok(result.positionDecode, 'no position decoded from the simulated frame');
  const pd = result.positionDecode!;

  assert.ok(pd.consistency > 0.99, `nadir should decode almost perfectly, got ${pd.consistency}`);
  const d = pd.camPos.distanceTo(truth.camPos);
  assert.ok(d < 0.15, `camPos off by ${d.toFixed(3)} cells: got ${fmt(pd.camPos)}, true ${fmt(truth.camPos)}`);

  // Period is the scale, and it is what height is derived from. Getting it to
  // a fraction of a percent means the gnomonic rectification and the lattice
  // spacing agree with the renderer exactly.
  const period = result.gridPeriodPhase!.period;
  assert.ok(Math.abs(period - truth.period) / truth.period < 0.01,
    `period: recovered ${period}, true ${truth.period}`);
});

test('SIMULATOR VALIDATED: the geometry stays right across tilt', async () => {
  // The continuous half, at tilts where the discrete half is known to
  // misbehave. Period and height are anchor-independent -- a wrong anchor moves
  // the position by whole cells and leaves the scale alone -- so asserting on
  // them isolates the renderer from the decode's problems.
  for (const tiltDeg of [0, 10, 20, 30, 40]) {
    const pose: SimPose = { height: 10, overRow: 70.5, overCol: 70.5, tiltDeg, yawDeg: 0 };
    const truth = truthFor(pose);
    const { pose: result } = await runPoseOn(inputFor(pose), 'cpu');
    assert.ok(result.recoveredAxes, `tilt ${tiltDeg}: no axes`);
    assert.ok(result.gridPeriodPhase, `tilt ${tiltDeg}: no period`);

    const period = result.gridPeriodPhase!.period;
    assert.ok(Math.abs(period - truth.period) / truth.period < 0.03,
      `tilt ${tiltDeg}: period ${period} vs true ${truth.period}`);
    assert.ok(Math.abs(result.recoveredAxes!.distance - truth.height) / truth.height < 0.03,
      `tilt ${tiltDeg}: height ${result.recoveredAxes!.distance} vs true ${truth.height}`);
  }
});

test('the horizontal axis is accurate at every tilt, which localizes the error to depth', async () => {
  // dx is the axis the tilt does NOT lean along. It stays accurate while dz
  // does not, and that asymmetry is what says the failure is the decode
  // choosing a neighbouring anchor along the tilt direction, rather than
  // anything wrong with the projection.
  for (const tiltDeg of [0, 10, 20, 30, 40]) {
    const pose: SimPose = { height: 10, overRow: 70.5, overCol: 70.5, tiltDeg, yawDeg: 0 };
    const truth = truthFor(pose);
    const { pose: result } = await runPoseOn(inputFor(pose), 'cpu');
    const pd = result.positionDecode;
    assert.ok(pd, `tilt ${tiltDeg}: no decode`);
    assert.ok(Math.abs(pd!.camPos.x - truth.camPos.x) < 0.25,
      `tilt ${tiltDeg}: dx ${(pd!.camPos.x - truth.camPos.x).toFixed(3)} should stay small`);
  }
});
