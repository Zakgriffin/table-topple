import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import * as THREE from 'three';
import { readF32, readU32, withDevice } from './helpers/gpu.ts';
import { createBuffers, planPool } from '../src/pose2/buffers.ts';
import {
  type Ctx, encodeCollect, encodeFit, encodeGradient, encodeGrow, encodeLines,
  encodeGpp, encodeDecodeBuild, encodeDecodeLayout, encodeDecodeTally, encodeFinish,
  encodeLsdFit, encodeVotes, decodePose, makeCtx,
} from '../src/pose2/pose.ts';
import type { Dims } from '../src/pose2/pipeline.ts';
import { rayDirInto, renderPose, truthFor, vFovRadOf } from '../src/pose2/sim.ts';
import { boardDims, buildBoard, hashU32, uploadBoard } from '../src/pose2/board.ts';
import { TEST_BOARD, TEST_CELL_PITCH, TEST_WORLD } from './helpers/board.ts';

// Destructured once, at the top, and safe here in a way it would not be in app
// code: TEST_BOARD is a const built at module load and never swapped.
const { R, C, torus, lookup: debruijnLookup, order: ORDER } = TEST_BOARD;

// The math frame is camera space -- every ray-casting call in the recovery
// pipeline is expressed against the identity. The app spells this MATH_QUAT;
// a library test has no business importing the app to say "identity".
const MATH_QUAT = new THREE.Quaternion();

// Where a lattice's zero-reference cell ENDS UP after rotating the whole grid by
// `o` quarter-turns. Used once, by the decode anchor test.
//
// This came from src/pose/stages/decode/decodeGrid.ts, which no longer exists.
// It was RE-DERIVED rather than copied across, for the same reason cpu.ts is:
// while src/pose was here, importing it made this an independent second source,
// and after the deletion a copy would have been an unowned duplicate of nothing.
//
// The derivation is one rule applied o times. A single quarter-turn takes (i, j)
// of a rows x cols grid to (j, rows-1-i) of a cols x rows grid -- so the four
// cases the original spelled out are just this loop unrolled, and there is no
// table of index expressions to get wrong.
//
// Checked exhaustively against the original before it was deleted: 24,336 cases
// over rows, cols in 1..12 and all four o, zero mismatches. That check was also
// shown to be DECISIVE -- a version using `rows` where the derivation calls for
// `cols` differs in 5,434 of them. Note such an error is invisible on a SQUARE
// lattice, so a fixture with rows === cols would not have settled anything.
function rotatedZeroIndex(
  rows: number, cols: number, zeroI: number, zeroJ: number, o: number,
): [number, number] {
  let r = rows, c = cols, ri = zeroI, rj = zeroJ;
  for (let k = 0; k < (o & 3); k++) {
    [ri, rj] = [rj, r - 1 - ri];
    [r, c] = [c, r];
  }
  return [ri, rj];
}

// ── Stage tests against SYNTHETIC GROUND TRUTH ────────────────────────────
//
// Not a differential against a CPU reference. Nearly every stage in this
// pipeline has a known inverse -- the input can be generated from the answer --
// and that is strictly stronger than comparing two implementations, which by
// construction cannot see an error both of them share.
//
// Every body runs inside withDevice, which fails the test if anything inside
// failed WebGPU validation. Without that, a stage that silently never executed
// returns zeros, and zero is a plausible answer for most of this pipeline.

const DIMS: Dims = {
  w: 32, h: 32, maxRegions: 1024, maxLines: 1024, maxCells: 144 * 144,
  torusR: 144, torusC: 144, hashSlots: 65536,
};
const W = DIMS.w, H = DIMS.h, N = W * H;

/** A frame big enough to hold a rendered board, and big enough that a region
 *  count can exceed one 64-thread workgroup. */
const FRAME: Dims = {
  w: 96, h: 128, maxRegions: 4096, maxLines: 4096, maxCells: 144 * 144,
  torusR: 144, torusC: 144, hashSlots: 65536,
};
const FRAME_DIMS = { w: FRAME.w, h: FRAME.h, horizFovDeg: 60 };


// ── ONE buffer set for the whole process, and that is deliberate ──────────
//
// Production allocates once per (w, h) and reuses across frames, so every
// buffer starts each run holding the PREVIOUS run's bytes. A test that got a
// fresh, implicitly-zeroed set every time would be exercising a pipeline nobody
// runs, and would be structurally blind to the whole zero-initialization
// failure class -- the one where run 2 inherits run 1's state. Sharing means a
// stage that forgets a clear gets caught by whichever test runs next, instead
// of by nobody.
//
// (An earlier version of this comment claimed a second buffer set segfaulted
// Dawn. That was FALSE and the cause was ours: the test helper dropped its
// reference to the GPU instance, so V8 collected it out from under the live
// device. See tests/helpers/gpu.ts. Sharing is kept because it is faithful, not
// because anything forces it.)
//
// Keyed by size, because the ground-truth vote test renders a real board frame
// and 32x32 is too small to hold one. One set per size, still shared across
// every test at that size.
const sets = new Map<string, { plan: ReturnType<typeof planPool>; bufs: Record<string, GPUBuffer> }>();

function buffers(device: GPUDevice, dims: Dims = DIMS, alias = false) {
  // Keyed by everything that changes a BUFFER SIZE OR ITS SHARING, not just the
  // frame: the clamp test reuses a frame size with a smaller board, and the
  // pooling test reuses one with a different slot assignment. A key that saw
  // only (w, h) would hand either of them the wrong set and quietly test
  // nothing.
  const key = `${dims.w}x${dims.h}:${dims.maxRegions}:${dims.maxLines}:${dims.torusR}x${dims.torusC}:${dims.maxCells}:${alias}`;
  let set = sets.get(key);
  if (!set) {
    const plan = planPool(dims, { alias });
    set = { plan, bufs: createBuffers(device, plan) };
    sets.set(key, set);
  }
  return set;
}

/** A fresh encoder over the shared buffers; submits when `body` returns. */
async function run(
  device: GPUDevice, body: (ctx: Ctx) => void, dims: Dims = DIMS, alias = false,
): Promise<Record<string, GPUBuffer>> {
  const { plan, bufs } = buffers(device, dims, alias);
  const ctx = makeCtx(device, plan, bufs, dims);
  body(ctx);
  device.queue.submit([ctx.enc.finish()]);
  return bufs;
}

// ── S1 gradient ───────────────────────────────────────────────────────────

test('gradient: a vertical step edge gives fx=1 in exactly one column', async () => {
  // GROUND TRUTH, derived by hand from the 2x2 kernel rather than from another
  // implementation. Gray is 0 left of column c and 255 from c rightwards, so at
  // x = c-1 the block is (0,255,0,255): fx = ((255+255)-(0+0))/510 = 1 and
  // fy = ((0+255)-(0+255))/510 = 0. At x = c the block is uniformly 255, so
  // both are 0. Everything else is flat.
  await withDevice(async (device) => {
    const c = 6;
    const gray = new Float32Array(N);
    for (let y = 0; y < H; y++) for (let x = c; x < W; x++) gray[y * W + x] = 255;

    const bufs = await run(device, (ctx) => {
      device.queue.writeBuffer(ctx.bufs.gray!, 0, gray);
      encodeGradient(ctx);
    });

    const fx = await readF32(device, bufs.fx!, N);
    const fy = await readF32(device, bufs.fy!, N);
    for (let y = 0; y < H - 1; y++) {          // last row is zeroed by design
      for (let x = 0; x < W - 1; x++) {        // last column likewise
        const expected = x === c - 1 ? 1 : 0;
        assert.ok(Math.abs(fx[y * W + x]! - expected) < 1e-6,
          `fx at (${x},${y}) expected ${expected}, got ${fx[y * W + x]}`);
        assert.ok(Math.abs(fy[y * W + x]!) < 1e-6, `fy at (${x},${y}) should be 0`);
      }
    }
  });
});

test('gradient: a horizontal step edge gives fy=1, and only the last row/col are margin', async () => {
  await withDevice(async (device) => {
    const r = 5;
    const gray = new Float32Array(N);
    for (let y = r; y < H; y++) for (let x = 0; x < W; x++) gray[y * W + x] = 255;

    const bufs = await run(device, (ctx) => {
      device.queue.writeBuffer(ctx.bufs.gray!, 0, gray);
      encodeGradient(ctx);
    });

    const fx = await readF32(device, bufs.fx!, N);
    const fy = await readF32(device, bufs.fy!, N);
    for (let y = 0; y < H - 1; y++) {
      for (let x = 0; x < W - 1; x++) {
        const expected = y === r - 1 ? 1 : 0;
        assert.ok(Math.abs(fy[y * W + x]! - expected) < 1e-6,
          `fy at (${x},${y}) expected ${expected}, got ${fy[y * W + x]}`);
        assert.ok(Math.abs(fx[y * W + x]!) < 1e-6);
      }
    }
    // The 2x2 block gradient leaves ONLY the last row and column zero. A
    // symmetric centered-difference kernel would zero all four edges and
    // silently discard the real data along the top and left; this asserts we
    // did not copy that one in.
    for (let y = 0; y < H; y++) assert.equal(fx[y * W + (W - 1)], 0);
    for (let x = 0; x < W; x++) assert.equal(fy[(H - 1) * W + x], 0);
    // ... and that the FIRST row and column carry real data, not margin.
    assert.ok(Math.abs(fy[(r - 1) * W + 0]! - 1) < 1e-6, 'column 0 must carry real gradient');
  });
});

// ── S2 grow ───────────────────────────────────────────────────────────────

/** Distinct non-negative labels, i.e. the component count. */
function componentCount(label: Int32Array): number {
  const seen = new Set<number>();
  for (const l of label) if (l >= 0) seen.add(l);
  return seen.size;
}

/**
 * Writes fx/fy directly, so grow is tested in isolation from the gradient.
 * fx = +/-1 with fy = 0 gives a level-line direction of (0, +/-1): vertical, so
 * a column is internally connected, and the SIGN is what the directed predicate
 * discriminates on.
 */
function stripeField(stripes: { col: number; sign: number }[]): { fx: Float32Array; fy: Float32Array } {
  const fx = new Float32Array(N), fy = new Float32Array(N);
  for (const { col, sign } of stripes) {
    for (let y = 2; y <= H - 4; y++) { fx[y * W + col] = sign; fy[y * W + col] = 0; }
  }
  return { fx, fy };
}

const GROW = { rhoLow: 0.5, toleranceDeg: 9.5 };

async function growOn(device: GPUDevice, stripes: { col: number; sign: number }[]): Promise<Record<string, GPUBuffer>> {
  const { fx, fy } = stripeField(stripes);
  return run(device, (ctx) => {
    device.queue.writeBuffer(ctx.bufs.fx!, 0, fx);
    device.queue.writeBuffer(ctx.bufs.fy!, 0, fy);
    encodeGrow(ctx, GROW);
  });
}

async function labelsOf(device: GPUDevice, bufs: Record<string, GPUBuffer>): Promise<Int32Array> {
  return new Int32Array((await readU32(device, bufs.label!, N)).buffer);
}

test('grow: two separated stripes are two components', async () => {
  await withDevice(async (device) => {
    const stripes = [{ col: 4, sign: 1 }, { col: 20, sign: 1 }];
    const label = await labelsOf(device, await growOn(device, stripes));
    assert.equal(componentCount(label), 2);

    // Every eligible pixel is labelled and every ineligible one is -1.
    const { fx } = stripeField(stripes);
    let eligible = 0;
    for (let i = 0; i < N; i++) {
      if (fx[i] !== 0) { eligible++; assert.ok(label[i]! >= 0, `pixel ${i} should be labelled`); }
      else assert.equal(label[i], -1, `pixel ${i} should be ineligible`);
    }
    assert.ok(eligible > 0);
  });
});

test('grow: ADJACENT ANTIPARALLEL stripes stay separate -- the directed predicate', async () => {
  // The one test that distinguishes a signed dot from an unsigned one, and the
  // reason the predicate has no abs(). Columns 4 and 5 are 8-adjacent; column 4
  // has level-line direction (0,1) and column 5 has (0,-1), so their dot is -1.
  //
  //   directed -> incompatible -> 3 components (4, 5, and 20)
  //   unsigned -> |dot| = 1 -> compatible -> columns 4 and 5 MERGE -> 2
  //
  // This is the physical case of a thin dark stripe: its two edges have
  // opposite polarity and must not fuse into one region.
  await withDevice(async (device) => {
    const label = await labelsOf(device,
      await growOn(device, [{ col: 4, sign: 1 }, { col: 5, sign: -1 }, { col: 20, sign: 1 }]));
    assert.equal(componentCount(label), 3, 'antiparallel adjacent stripes must not merge');
    // And they really are adjacent, or the test proves nothing.
    const a = label[3 * W + 4]!, b = label[3 * W + 5]!;
    assert.ok(a >= 0 && b >= 0, 'both stripes must be eligible');
    assert.notEqual(a, b);
  });
});

test('grow: converges on device and reports a true round count', async () => {
  await withDevice(async (device) => {
    const bufs = await growOn(device, [{ col: 4, sign: 1 }, { col: 20, sign: 1 }]);
    const args = await readU32(device, bufs.growArgs!, 4);
    // args.x == 0 IS the converged state -- gate zeroes the whole triple, and
    // ceil(w/8) is never legitimately 0, so there is no separate flag to keep in
    // sync with it.
    assert.equal(args[0], 0, 'grow did not converge within the encoded rounds');
    // Pointer jumping collapses a 27-pixel stripe in O(log n), so this must be
    // small -- and it must not be 0, which would mean the early-out fired before
    // any round ran and the labelling is still init's singletons.
    assert.ok(args[3]! > 0 && args[3]! < 12, `activeRounds = ${args[3]}, expected a small positive count`);
  });
});

test('grow: a label is the smallest pixel index in its component', async () => {
  // hook takes the MINIMUM neighbouring label, so at the fixpoint every member
  // carries its component's smallest pixel index. Pins the labelling exactly
  // rather than just counting components.
  await withDevice(async (device) => {
    const label = await labelsOf(device, await growOn(device, [{ col: 4, sign: 1 }]));
    const expected = 2 * W + 4; // topmost pixel of the stripe: y=2, x=4
    for (let y = 2; y <= H - 4; y++) assert.equal(label[y * W + 4], expected);
  });
});

// ── S4 lsdFit ─────────────────────────────────────────────────────────────

const COLLECT = { rhoHigh: 0.5, minRegionSize: 2 };
const LSDFIT = { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 };

/** grow + collect + lsdFit over a directly-written gradient field. */
async function fitOn(
  device: GPUDevice, fx: Float32Array, fy: Float32Array,
  collect: { rhoHigh: number; minRegionSize: number } = COLLECT,
): Promise<{ counts: Uint32Array; rects: Float32Array }> {
  const bufs = await run(device, (ctx) => {
    device.queue.writeBuffer(ctx.bufs.fx!, 0, fx);
    device.queue.writeBuffer(ctx.bufs.fy!, 0, fy);
    encodeGrow(ctx, GROW);
    encodeCollect(ctx, collect);
    encodeLsdFit(ctx, LSDFIT);
  });
  return {
    counts: await readU32(device, bufs.counts!, 2),
    rects: await readF32(device, bufs.rects!, DIMS.maxRegions * 10),
  };
}

test('lsdFit: a one-pixel-wide stripe fits the rectangle geometry demands', async () => {
  // GROUND TRUTH, derived by hand rather than from a second implementation.
  // stripeField puts 27 pixels in column c, rows 2..28, each with fx = 1 and
  // fy = 0 -- so every member has magnitude 1 and level-line direction (0, 1).
  //
  //   centroid    (c, 15) -- uniform weights over a symmetric run
  //   moments     Ixx = Ixy = 0, Iyy > 0, so the major axis is vertical and
  //               theta = pi/2 (the mean direction (0,1) agrees, so no flip)
  //   extent      proj spans [-13, 13] and perp is identically 0
  //               => centre (c, 15), length 26, width 0
  //   footprint   hw floors to 0.5, so the bbox is columns c-1..c+1 -- and the
  //               neighbours land at |perp| = 1, outside. n = the 27 members.
  //   aligned     every one of them: alignDot = 1. k = 27.
  //
  // With k == n the binomial tail collapses to its single j = n term,
  // log C(n,n) + n log p, so the NFA is exactly logNTests + 27 log(tol/180).
  await withDevice(async (device) => {
    const c = 4;
    const { fx, fy } = stripeField([{ col: c, sign: 1 }]);
    const { counts, rects } = await fitOn(device, fx, fy);
    assert.equal(counts[0], 1, 'one stripe is one region');

    const [cx, cy, theta, len, wid, nfaLog10, accepted, , n, k] = [...rects.subarray(0, 10)];
    assert.ok(Math.abs(cx! - c) < 1e-5, `cx ${cx}`);
    assert.ok(Math.abs(cy! - 15) < 1e-5, `cy ${cy}`);
    assert.ok(Math.abs(theta! - Math.PI / 2) < 1e-6, `theta ${theta}`);
    assert.ok(Math.abs(len! - 26) < 1e-5, `length ${len}`);
    assert.ok(Math.abs(wid!) < 1e-5, `width ${wid}`);
    assert.equal(n, 27, 'footprint pixel count');
    assert.equal(k, 27, 'aligned pixel count');

    // p = toleranceRad / pi is just toleranceDeg / 180.
    const p = LSDFIT.toleranceDeg / 180;
    const expected =
      (LSDFIT.nfaTestExponent * Math.log(Math.max(W, H)) + 27 * Math.log(p)) / Math.LN10;
    assert.ok(Math.abs(nfaLog10! - expected) < 1e-3, `nfaLog10 ${nfaLog10}, expected ${expected}`);
    assert.equal(accepted, 1, 'a 27-pixel perfectly aligned stripe must clear NFA');
  });
});

test('lsdFit: theta follows the level-line direction, not just the PCA axis', async () => {
  // PCA fixes the axis only up to 180 degrees. Two stripes of OPPOSITE polarity
  // have identical geometry -- same pixels, same moments, same eigenvector --
  // and differ only in their mean level-line direction, so this is the one test
  // that can see the disambiguation at all. Without it both would report the
  // same theta and the NFA alignment test would read one of them as entirely
  // anti-aligned, i.e. k = 0 and a rejected line.
  await withDevice(async (device) => {
    const { fx, fy } = stripeField([{ col: 4, sign: 1 }, { col: 20, sign: -1 }]);
    const { counts, rects } = await fitOn(device, fx, fy);
    assert.equal(counts[0], 2);

    // Ascending label order, and label 4 is the lower pixel index.
    const up = rects[2]!, down = rects[12]!;
    assert.ok(Math.abs(up - Math.PI / 2) < 1e-6, `+1 stripe theta ${up}`);
    assert.ok(Math.abs(down - 3 * Math.PI / 2) < 1e-6, `-1 stripe theta ${down}`);
    // Both are still fully aligned WITH THEIR OWN axis -- which is the property
    // the flip exists to preserve.
    assert.equal(rects[9], 27, '+1 stripe aligned count');
    assert.equal(rects[19], 27, '-1 stripe aligned count');
  });
});

test('lsdFit: n is geometry, k is evidence -- anti-aligned and weak pixels split them', async () => {
  // THE TEST THAT ASKS THE PREDICATE ITS OWN QUESTION. A mutation run found
  // that making the alignment dot UNSIGNED -- the documented trap, since the
  // NFA null model's p = tau/pi is only correct for a directed test -- changed
  // nothing anywhere else, including on a rendered frame. The reason is
  // structural rather than lucky: hw is half the region's own width, so a
  // footprint is essentially the region's own pixels, and directed growth
  // already made those one polarity. Nothing anti-aligned is ever inside.
  //
  // So the case has to be built. A 5x3 block, all level-line (0, 1), with
  // three pixels replaced:
  //
  //     col    10  11  12          the block is 15 pixels
  //   row 10    +   +   +          12 of them are members
  //   row 11    +   .   +          . = weak (0.1): inside the footprint,
  //   row 12    +   -   +              too faint for its angle to count
  //   row 13    +   .   +          - = ANTI-ALIGNED (-1): inside the
  //   row 14    +   +   +              footprint, pointing the other way
  //
  // Symmetric on purpose, so the fit stays hand-derivable: centroid (11, 12),
  // Ixy = 0, Ixx = 10 < Iyy = 28, so theta = pi/2, length 4, width 2, hw 1.
  // Every one of the 15 block pixels is inside that rectangle, so n = 15
  // whatever their gradients say -- n is a GEOMETRIC count. k is 12: the
  // anti-aligned pixel fails a signed dot, and the two weak ones never get
  // their angle tested at all.
  //
  //   unsigned dot     -> k = 13
  //   no sub-rho test  -> k = 14
  await withDevice(async (device) => {
    const fx = new Float32Array(N), fy = new Float32Array(N);
    for (let y = 10; y <= 14; y++) for (let x = 10; x <= 12; x++) fx[y * W + x] = 1;
    fx[12 * W + 11] = -1;    // strong, anti-aligned -- its own singleton region
    fx[11 * W + 11] = 0.1;   // below grow's rhoLow AND below lsdFit's rho
    fx[13 * W + 11] = 0.1;

    const { counts, rects } = await fitOn(device, fx, fy);
    // The singleton the anti-aligned pixel forms is dropped by minRegionSize,
    // and the weak pair never became eligible at all.
    assert.equal(counts[0], 1, 'the block is one region');

    const [cx, cy, theta, len, wid, , , , n, k] = [...rects.subarray(0, 10)];
    assert.ok(Math.abs(cx! - 11) < 1e-5, `cx ${cx}`);
    assert.ok(Math.abs(cy! - 12) < 1e-5, `cy ${cy}`);
    assert.ok(Math.abs(theta! - Math.PI / 2) < 1e-6, `theta ${theta}`);
    assert.ok(Math.abs(len! - 4) < 1e-5, `length ${len}`);
    assert.ok(Math.abs(wid! - 2) < 1e-5, `width ${wid}`);
    assert.equal(n, 15, 'every pixel of the block is inside the footprint');
    assert.equal(k, 12, 'only the 12 members are strong AND aligned');
  });
});

test('lsdFit: a degenerate region zeroes all ten slots, not just the accepted flag', async () => {
  // THE GREPPABLE SHAPE this stage was warned about: an early return that
  // writes one field. `rects` is reused frame to frame, so a degenerate region
  // that only clears its accept flag keeps the PREVIOUS frame's centre, angle
  // and length -- a full, plausible, stale rectangle that still feeds the line
  // compaction.
  //
  // Run 1 puts a real rectangle in slot 0. Run 2 makes slot 0 a single isolated
  // pixel, which has no axis to fit. Same buffers, so anything left unwritten
  // is run 1's answer and the assertion below sees it.
  await withDevice(async (device) => {
    const solo = { rhoHigh: 0.5, minRegionSize: 1 };

    const stripe = stripeField([{ col: 4, sign: 1 }]);
    const first = await fitOn(device, stripe.fx, stripe.fy, solo);
    assert.equal(first.counts[0], 1);
    assert.ok(first.rects[3]! > 20, 'run 1 must leave a real rectangle in slot 0');

    const fx = new Float32Array(N), fy = new Float32Array(N);
    fx[2 * W + 4] = 1; // one eligible pixel, no eligible neighbours
    const second = await fitOn(device, fx, fy, solo);
    assert.equal(second.counts[0], 1, 'the lone pixel is still a region');
    assert.deepEqual([...second.rects.subarray(0, 10)], new Array(10).fill(0),
      'a degenerate region must zero its whole slot');
  });
});

test('grow: back-to-back runs agree', async () => {
  // THE CHEAPEST TEST FOR THE ZERO-INITIALIZATION FAMILY, and nothing else
  // covers it: every bug in that class is "run 2 inherits run 1's bytes", so a
  // single run looks perfect.
  //
  // The sharpest instance lives right here. A converged run leaves growArgs at
  // [0,0,0]; inheriting that makes run 2 dispatch no rounds at all and emit
  // init's singleton labelling -- one region per eligible pixel, at full speed,
  // with no error and no validation message. encodeGrowInit WRITES [gx,gy,1,0]
  // rather than clearing, and this is what proves it.
  await withDevice(async (device) => {
    const stripes = [{ col: 4, sign: 1 }, { col: 5, sign: -1 }, { col: 20, sign: 1 }];
    const first = await labelsOf(device, await growOn(device, stripes));
    const second = await labelsOf(device, await growOn(device, stripes));
    assert.equal(componentCount(first), 3);
    assert.deepEqual([...second], [...first], 'run 2 disagreed with run 1 -- a buffer carried state across runs');
  });
});

// ── S5 lines + votes ──────────────────────────────────────────────────────

const LINES = { minLengthPx: 3 };
/** tanHalf = 1, so a hand-derived ray is exact rather than nearly exact. */
const VOTES = { vFovRad: Math.PI / 2 };

/** The whole front half plus the compaction, over a directly-written gradient. */
async function linesOn(
  device: GPUDevice, fx: Float32Array, fy: Float32Array,
  lines: { minLengthPx: number } = LINES, dims: Dims = DIMS,
): Promise<{ lineCount: Uint32Array; lines: Float32Array; args: Uint32Array; votes: Float32Array }> {
  const bufs = await run(device, (ctx) => {
    device.queue.writeBuffer(ctx.bufs.fx!, 0, fx);
    device.queue.writeBuffer(ctx.bufs.fy!, 0, fy);
    encodeGrow(ctx, GROW);
    encodeCollect(ctx, COLLECT);
    encodeLsdFit(ctx, LSDFIT);
    encodeLines(ctx, lines);
    encodeVotes(ctx, VOTES);
  }, dims);
  return {
    lineCount: await readU32(device, bufs.lineCount!, 2),
    lines: await readF32(device, bufs.lines!, dims.maxLines * 4),
    args: await readU32(device, bufs.lineArgs!, 3),
    votes: await readF32(device, bufs.votes!, dims.maxLines * 4),
  };
}

test('lines: a rectangle becomes its own two endpoints', async () => {
  // GROUND TRUTH from the rectangle geometry, not from a second implementation.
  // stripeField's column-c stripe fits centre (c, 15), theta = pi/2, length 26
  // (asserted independently in the lsdFit tests above), so hl = 13 and the axis
  // is (cos, sin)(pi/2) = (0, 1). The endpoints are centre +/- hl * axis:
  //
  //     (c, 15 + 13) = (c, 28)   and   (c, 15 - 13) = (c, 2)
  //
  // which are exactly the first and last rows stripeField wrote -- the endpoints
  // land back on the stripe that generated them.
  await withDevice(async (device) => {
    const c = 16;
    const { fx, fy } = stripeField([{ col: c, sign: 1 }]);
    const { lineCount, lines, args } = await linesOn(device, fx, fy);

    assert.equal(lineCount[0], 1, 'one accepted rectangle is one line');
    const [x1, y1, x2, y2] = [...lines.subarray(0, 4)];
    assert.ok(Math.abs(x1! - c) < 1e-4, `x1 ${x1}`);
    assert.ok(Math.abs(y1! - 28) < 1e-4, `y1 ${y1}`);
    assert.ok(Math.abs(x2! - c) < 1e-4, `x2 ${x2}`);
    assert.ok(Math.abs(y2! - 2) < 1e-4, `y2 ${y2}`);
    // The args triple is the propagation mechanism, and it is written by the
    // same pass -- one line is one workgroup of 64.
    assert.deepEqual([...args], [1, 1, 1], 'lineArgs must cover the emitted lines');
  });
});

test('lines: the length floor is THIS stage\'s filter, and it zeroes the dispatch', async () => {
  // The rectangle is still accepted -- NFA has not changed -- so this separates
  // the fitter's verdict from the vote stage's usability filter. It is also the
  // failure path: no lines means lineArgs is [0,1,1], so every downstream pass
  // dispatches nothing rather than reading an empty array.
  await withDevice(async (device) => {
    const { fx, fy } = stripeField([{ col: 16, sign: 1 }]);
    const long = await linesOn(device, fx, fy, { minLengthPx: 26 });
    assert.equal(long.lineCount[0], 1, 'length 26 must clear a floor of exactly 26');

    const tooLong = await linesOn(device, fx, fy, { minLengthPx: 26.001 });
    assert.equal(tooLong.lineCount[0], 0, 'a hair above the length is a hair too far');
    assert.deepEqual([...tooLong.args], [0, 1, 1], 'no lines must dispatch no workgroups');
  });
});

/**
 * A grid of short vertical stripes: as many independent regions as the frame
 * holds. Columns two apart so no two are 8-neighbours, and one blank row between
 * stripes in a column for the same reason.
 *
 * `rows` is per stripe, and it has an NFA floor rather than a free choice: with
 * n = k = rows the tail is rows * ln(tol/pi), which has to beat
 * nfaTestExponent * ln(max(w, h)). At 96x128 that needs 9 pixels; a shorter
 * stripe is a perfectly good REGION that is never an accepted LINE, and would
 * make this fixture silently measure nothing.
 */
function stripeGrid(d: Dims, rows: number): { fx: Float32Array; fy: Float32Array; count: number } {
  const fx = new Float32Array(d.w * d.h), fy = new Float32Array(d.w * d.h);
  let count = 0;
  for (let col = 0; col < d.w; col += 2) {
    for (let top = 0; top + rows <= d.h; top += rows + 1) {
      for (let y = top; y < top + rows; y++) fx[y * d.w + col] = 1;
      count++;
    }
  }
  return { fx, fy, count };
}

test('lines: the flag array is TOTAL -- a frame cannot inherit the last one\'s flags', async () => {
  // THE TEST THE §9 DECISION EXISTS FOR, and nothing else in the suite covers
  // it. `lineFlag` is scanned over maxRegions, but only the first regionCount
  // slots describe this frame. If the flag pass dispatched off regionArgs -- the
  // obvious choice, and what the declaration originally said -- the slots past
  // the region count would keep the PREVIOUS frame's flags, and a quiet frame
  // after a busy one would emit lines from stale rectangles.
  //
  // ── THE FIRST VERSION OF THIS TEST WAS GREEN UNDER THE MUTATION ──
  //
  // It ran 6 stripes and then 1, at 32x32, and the indirect dispatch passed it.
  // The workgroup is 64 threads wide, so BOTH runs dispatch one workgroup and
  // both write slots 0..63 -- the stale tail starts at 64 and neither run ever
  // reached it. Same shape as §8's four-pixel bar and §19's: a fixture that
  // cannot ask the question it claims to.
  //
  // The question needs run 1 to flag a slot run 2's dispatch CANNOT reach. Two
  // ways to guarantee that, and both are here because they fail differently:
  //
  //   A. run 1 makes far more than 64 regions, so run 2's single workgroup
  //      covers a fraction of them.
  //   B. run 2 has NO regions at all, so its dispatch is zero workgroups and
  //      reaches nothing whatsoever. This is also the failure-propagation path:
  //      an empty frame must produce an empty pipeline, not the last frame's.
  await withDevice(async (device) => {
    // ── A: 576 regions, then 1 ──
    const grid = stripeGrid(FRAME, 9);
    assert.ok(grid.count > 64, `the fixture must exceed one workgroup, has ${grid.count}`);
    const busy = await linesOn(device, grid.fx, grid.fy, LINES, FRAME);
    assert.equal(busy.lineCount[0], grid.count,
      'every stripe in the grid must become a line, or the fixture proves nothing');

    const one = stripeGrid(FRAME, 9);
    one.fx.fill(0);
    for (let y = 20; y < 29; y++) one.fx[y * FRAME.w + 40] = 1;
    const quiet = await linesOn(device, one.fx, one.fy, LINES, FRAME);
    assert.equal(quiet.lineCount[0], 1,
      'run 2 inherited run 1\'s flags -- lineFlag is not written over its whole range');
    assert.ok(Math.abs(quiet.lines[0]! - 40) < 1e-4, `x1 ${quiet.lines[0]}`);

    // ── B: regions, then none ──
    const empty = new Float32Array(FRAME.w * FRAME.h);
    const none = await linesOn(device, empty, empty, LINES, FRAME);
    assert.equal(none.lineCount[0], 0, 'a frame with no regions must produce no lines');
    assert.deepEqual([...none.args], [0, 1, 1], 'and must dispatch no votes');
  });
});

test('votes: a centred vertical line votes for the X axis, at sin(subtended angle)', async () => {
  // GROUND TRUTH, hand-derived in full. The line above runs from (16, 28) to
  // (16, 2) in a 32x32 frame at aspect 1 and tanHalf 1, so both endpoints sit at
  // ndcU = 0 and the unnormalized camera-space rays are
  //
  //     u1 = (0, ndcV1, -1) = (0, -0.75, -1)      |u1| = 1.25
  //     u2 = (0, ndcV2, -1) = (0,  0.875, -1)     |u2| = sqrt(1.765625)
  //
  // Both have x = 0 and z = -1, so their cross product is purely X:
  //     cross(u1, u2).x = (-0.75)(-1) - (-1)(0.875) = 1.625, y = z = 0.
  //
  // The stage crosses the NORMALIZED rays, so the normal is exactly (1, 0, 0)
  // and the weight is 1.625 / (1.25 * sqrt(1.765625)) = 1.3 / sqrt(1.765625) --
  // which is sin of the angle the segment subtends at the camera, i.e. its
  // projected arc length on the unit sphere.
  //
  // ── WHAT THIS FIXTURE CANNOT SEE, MEASURED RATHER THAN GUESSED ──
  //
  // An earlier version of this comment claimed the zero y and z components were
  // what a dropped aspect or a flipped ndcV sense would break. The mutation run
  // says otherwise, and the reason is that the fixture sits exactly on the
  // symmetry each mutation acts on:
  //
  //   dropping `u.aspect`  -- multiplies ndcU, which is 0 here. And DIMS is
  //                           square, so aspect is 1 at this size regardless.
  //   flipping ndcV        -- mirrors the two endpoints, which negates the
  //                           normal. This asserts |nx|, so it cannot tell.
  //
  // Both are caught by the rendered-board test below, which is the right oracle
  // for a projection: it has off-axis lines at a non-square aspect and a truth
  // to be perpendicular to. What this test pins is the ALGEBRA -- the cross
  // product, the normalization and the weight's identification with arc length
  // -- on numbers small enough to check by hand.
  await withDevice(async (device) => {
    const { fx, fy } = stripeField([{ col: 16, sign: 1 }]);
    const { lineCount, votes } = await linesOn(device, fx, fy);
    assert.equal(lineCount[0], 1);

    const [nx, ny, nz, weight] = [...votes.subarray(0, 4)];
    assert.ok(Math.abs(Math.abs(nx!) - 1) < 1e-5, `|nx| ${nx}`);
    assert.ok(Math.abs(ny!) < 1e-6, `ny ${ny} must be exactly zero`);
    assert.ok(Math.abs(nz!) < 1e-6, `nz ${nz} must be exactly zero`);

    const expected = 1.3 / Math.sqrt(1.765625);
    assert.ok(Math.abs(weight! - expected) < 1e-5, `weight ${weight}, expected ${expected}`);
  });
});

// ── The first stage scored by GROUND TRUTH rather than by a twin ──────────
//
// Everything up to lsdFit is checked against cpu.ts, because a labelling is an
// artifact of the algorithm and no pose implies one. A VOTE is different: it is
// a statement about the scene, and the scene is known exactly because the frame
// was rendered from a pose. So this is the first place the pipeline can be told
// it is WRONG rather than merely told it disagrees.
//
// The claim under test: a line detected on the printed floor lies along one of
// the two grid families, so the plane it sweeps through the camera centre
// contains that family's floor direction -- and therefore the vote NORMAL is
// perpendicular to it. In camera space that is truth.DrowMath or truth.DcolMath.
//
// This is also the geometric half of lsdFit's oracle (§21.8): a rectangle that
// is self-consistently wrong -- right shape, wrong place -- passes every twin
// comparison and fails here.

test('votes: on a rendered board, every normal is perpendicular to a true floor axis', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const gray = renderPose(TEST_WORLD, pose, FRAME_DIMS, 4);
    const truth = truthFor(TEST_WORLD, pose);

    const bufs = await run(device, (ctx) => {
      device.queue.writeBuffer(ctx.bufs.gray!, 0, Float32Array.from(gray));
      encodeGradient(ctx);
      encodeGrow(ctx, { rhoLow: 0.132, toleranceDeg: 9.5 });
      encodeCollect(ctx, { rhoHigh: 0, minRegionSize: 2 });
      encodeLsdFit(ctx, { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 });
      encodeLines(ctx, { minLengthPx: 3 });
      encodeVotes(ctx, { vFovRad: vFovRadOf(FRAME_DIMS) });
    }, FRAME);

    const lineCount = (await readU32(device, bufs.lineCount!, 2))[0]!;
    const votes = await readF32(device, bufs.votes!, FRAME.maxLines * 4);
    assert.ok(lineCount > 40, `only ${lineCount} lines -- the detector, not the votes, is the problem`);

    const rowAxis = truth.DrowMath, colAxis = truth.DcolMath;
    const residuals: number[] = [];
    let rowFamily = 0, colFamily = 0, degenerate = 0;
    for (let i = 0; i < lineCount; i++) {
      const [nx, ny, nz, w] = [...votes.subarray(i * 4, i * 4 + 4)];
      if (w! <= 0) { degenerate++; continue; }
      const dRow = Math.abs(nx! * rowAxis.x + ny! * rowAxis.y + nz! * rowAxis.z);
      const dCol = Math.abs(nx! * colAxis.x + ny! * colAxis.y + nz! * colAxis.z);
      if (dRow < dCol) rowFamily++; else colFamily++;
      residuals.push(Math.min(dRow, dCol));
    }
    residuals.sort((a, b) => a - b);
    const median = residuals[Math.floor(residuals.length / 2)]!;
    const worst = residuals[residuals.length - 1]!;
    const report =
      `${residuals.length} votes, median residual ${median.toExponential(2)}, ` +
      `worst ${worst.toExponential(2)}, families ${rowFamily}/${colFamily}, degenerate ${degenerate}`;

    // A residual is |sin(deviation)|. THE DECISIVENESS GATE, measured on both
    // sides rather than tuned until it passed:
    //
    //   observed, correct    median 8.1e-3 (0.47 deg), worst 3.5e-2 (2.0 deg)
    //   observed, mutated    dropping the aspect or flipping the ndcV sense both
    //                        push the median into the tenths -- the two families
    //                        are 90 degrees apart, so a normal perpendicular to
    //                        NEITHER has nowhere small to land.
    //
    // The gap between those two is an order of magnitude, and the thresholds sit
    // inside it. The honest caveat: `worst` has only ~1.4x headroom over the
    // observed value, so it is the assertion that will go flaky first if the
    // detector's endpoint precision changes. `median` is the load-bearing one.
    assert.ok(median < 0.02, `median vote is not perpendicular to a floor axis: ${report}`);
    assert.ok(worst < 0.05, `a vote is perpendicular to neither floor axis: ${report}`);

    // BOTH families must be populated. A geometry error that collapses the two
    // grid directions onto one leaves every residual small against a single
    // axis, and the assertions above cannot see it.
    assert.ok(rowFamily > residuals.length * 0.2 && colFamily > residuals.length * 0.2,
      `the two grid families are not both represented: ${report}`);
  });
});

// ── S6 fit ────────────────────────────────────────────────────────────────
//
// Ground truth all the way down, in two layers. The first feeds votes built
// FROM a known pair of floor axes, so the whole 6x6 -> 3x3 -> triad chain is
// checked against an answer that is exact by construction and has no detector in
// front of it. The second runs the real chain on a rendered board, which is the
// only thing that can catch a fit that is right about synthetic votes and wrong
// about the ones this pipeline actually produces.

type V3 = readonly [number, number, number];
const dot3 = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: V3, b: V3): V3 =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a: V3): V3 => {
  const s = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / s, a[1] / s, a[2] / s];
};
const asV3 = (v: { x: number; y: number; z: number }): V3 => [v.x, v.y, v.z];

/**
 * Votes generated from KNOWN floor directions instead of from an image.
 *
 * The generating identity, which is also the whole claim this stage rests on: a
 * line lying along floor direction `d` sweeps a plane through the camera centre
 * that CONTAINS d, so that plane's normal is perpendicular to d -- and any unit
 * vector perpendicular to d is a legal vote for it. So `cross(d, r)` for a
 * sweep of r produces the entire pencil of votes that axis can generate, which
 * is exactly the set the degenerate quadric has to fit.
 *
 * The sweep matters: votes clustered around one r would leave the 6x6's null
 * space bigger than one dimension, and the recovered axes would then be an
 * arbitrary member of it rather than the answer.
 */
function syntheticVotes(axes: readonly V3[], perAxis: number): {
  votes: Float32Array; count: number; maxWeight: number;
} {
  const out: number[] = [];
  let maxW = 0;
  for (const d of axes) {
    for (let k = 0; k < perAxis; k++) {
      const t = ((k + 1) / (perAxis + 1)) * Math.PI;
      const r: V3 = [Math.cos(t), Math.sin(t) * 0.5 + 0.3, Math.sin(t * 1.7)];
      const c = cross3(d, r);
      // A fixture that quietly generated a near-zero normal would be testing
      // the degenerate path while claiming to test the fit.
      assert.ok(Math.hypot(c[0], c[1], c[2]) > 0.1, 'the sweep put r nearly parallel to an axis');
      const n = norm3(c);
      // Varied and all positive, so the weighting is exercised rather than
      // being a constant that factors out.
      const w = 0.2 + (0.6 * ((k * 7) % 5)) / 4;
      maxW = Math.max(maxW, w);
      out.push(n[0], n[1], n[2], w);
    }
  }
  return { votes: Float32Array.from(out), count: out.length / 4, maxWeight: maxW };
}

interface Triad { Drow: V3; Dcol: V3; Dnormal: V3; status: number }

/** array<vec3<f32>, 3> has stride 16, so the three axes sit at 0, 4 and 8. */
function unpackTriad(t: Float32Array, status: number): Triad {
  return {
    Drow: [t[0]!, t[1]!, t[2]!],
    Dcol: [t[4]!, t[5]!, t[6]!],
    Dnormal: [t[8]!, t[9]!, t[10]!],
    status,
  };
}

/** fit alone, over votes written straight into the buffer. */
async function fitOnVotes(
  device: GPUDevice, v: { votes: Float32Array; count: number; maxWeight: number },
): Promise<Triad> {
  const bufs = await run(device, (ctx) => {
    device.queue.writeBuffer(ctx.bufs.votes!, 0, v.votes);
    device.queue.writeBuffer(ctx.bufs.lineCount!, 0, new Uint32Array([v.count, 0]));
    // maxWeight is an atomicMax over FLOAT BITS, so it is written as bits here
    // for the same reason fit.ata bitcasts them back.
    device.queue.writeBuffer(ctx.bufs.maxWeight!, 0, new Uint32Array(new Float32Array([v.maxWeight]).buffer));
    // status is cleared before its first bind, which is lines.emit -- and this
    // run does not encode lines.emit, so the schedule never fires.
    device.queue.writeBuffer(ctx.bufs.status!, 0, new Uint32Array([0]));
    encodeFit(ctx);
  });
  return unpackTriad(
    await readF32(device, bufs.triad!, 12),
    (await readU32(device, bufs.status!, 1))[0]!,
  );
}

/**
 * The three structural guarantees, asserted everywhere a triad is produced.
 *
 * None of them is a tolerance choice. Orthonormality is exact in the maths --
 * Jacobi's rotation matrix is orthogonal by construction, so b1 and b2 are
 * orthonormal, (b1+b2).(b1-b2) = |b1|^2 - |b2|^2 = 0, and both are perpendicular
 * to the third eigenvector -- so anything past round-off is a real defect.
 */
function assertTriadShape(t: Triad, what: string): void {
  for (const [name, v] of [['Drow', t.Drow], ['Dcol', t.Dcol], ['Dnormal', t.Dnormal]] as const) {
    assert.ok(Math.abs(Math.hypot(v[0], v[1], v[2]) - 1) < 1e-5, `${what}: ${name} is not a unit vector`);
  }
  assert.ok(Math.abs(dot3(t.Drow, t.Dcol)) < 1e-5, `${what}: Drow and Dcol are not perpendicular`);
  assert.ok(Math.abs(dot3(t.Drow, t.Dnormal)) < 1e-5, `${what}: Drow is not in the floor plane`);
  assert.ok(Math.abs(dot3(t.Dcol, t.Dnormal)) < 1e-5, `${what}: Dcol is not in the floor plane`);
  // The camera looks down -z, so a normal on the camera's side has z > 0. This
  // is the ONE assertion that distinguishes the oriented normal fit.eigen writes
  // from the raw eigenvector, whose sign the fit does not determine.
  assert.ok(t.Dnormal[2] > 0, `${what}: the normal points away from the camera (z = ${t.Dnormal[2]})`);
  // Right-handed, which §14's closed-form camera quaternion depends on.
  assert.ok(dot3(cross3(t.Drow, t.Dcol), t.Dnormal) <= 0,
    `${what}: the triad is left-handed`);
}

/** How far the recovered axis pair is from the true one, as an unordered set of
 *  undirected axes -- the fit cannot tell row from col or either from its
 *  negation, and does not have to: decode resolves the 4-fold ambiguity. */
function axisPairError(t: Triad, truthRow: V3, truthCol: V3): number {
  const pair = (a: V3, b: V3) => Math.max(1 - Math.abs(dot3(t.Drow, a)), 1 - Math.abs(dot3(t.Dcol, b)));
  return Math.min(pair(truthRow, truthCol), pair(truthCol, truthRow));
}

test('fit: votes built from two known floor axes recover exactly those axes', async () => {
  // The generating maths, worked through, because the answer being exact is the
  // point of this fixture. Votes perpendicular to d1 or d2 all satisfy
  // (n.d1)(n.d2) = 0, so the quadric they define is
  //
  //     M = (d1 d2^T + d2 d1^T) / 2
  //
  // whose eigenvectors for orthonormal d1, d2 are (d1+d2)/sqrt(2) at +1/2,
  // (d1-d2)/sqrt(2) at -1/2, and d1 x d2 at ZERO. So the near-zero eigenvector
  // is the floor normal and b1 +/- b2 are d1 and d2 back again -- which is
  // exactly what fit.eigen computes. Nothing here is approximate.
  //
  // The pose is oblique and yawed on purpose: at nadir the true axes are
  // coordinate axes, and a transposed unpack of the 21 upper-triangle entries or
  // a swapped 3x3 cross term would be invisible against them.
  await withDevice(async (device) => {
    // ── THREE POSES, AND THE SECOND AND THIRD ARE NOT DECORATION ──
    //
    // The fit ends with two sign decisions -- orient the normal toward the
    // camera, then make the triad right-handed -- and NEITHER is exercised by an
    // ordinary pose. Jacobi's eigenvector signs are deterministic but arbitrary,
    // and for a mildly oblique view they happen to come out already-correct, so
    // deleting either flip changes nothing and both assertions pass on evidence
    // they never saw. That is the same shape as §9's stale-flag test and §8's
    // four-pixel bar, and it was found the same way: by mutation.
    //
    // These two were found by disabling the flips and scanning 132 poses for one
    // where the RAW output is wrong. Measured 2026-08-13, raw (flips disabled):
    //
    //   tilt 20 yaw 15            normal z = +0.94, right-handed -- BLIND to both
    //   tilt 50 yaw 35            LEFT-HANDED (cross . normal = +1.000)
    //   tilt 50 yaw 35 roll 245   normal z = -0.643, pointing AWAY from the camera
    //
    // The scan is also the honest limit on how reachable these are: across tilts
    // 0-40 with no roll, 0 of 77 poses produced either. Both flips are cheap
    // insurance against a silently mirrored floor, and only the extremes prove
    // they are insurance rather than dead code.
    for (const pose of [
      { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 },
      { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 50, yawDeg: 35 },
      { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 50, yawDeg: 35, rollDeg: 245 },
    ]) {
      const what = `tilt ${pose.tiltDeg} yaw ${pose.yawDeg} roll ${pose.rollDeg ?? 0}`;
      const truth = truthFor(TEST_WORLD, pose);
      const d1 = asV3(truth.DrowMath), d2 = asV3(truth.DcolMath);
      const t = await fitOnVotes(device, syntheticVotes([d1, d2], 6));

      assertTriadShape(t, what);
      assert.equal(t.status, 0, `${what}: a well-posed fit must not report degeneracy`);

      // Observed 2026-08-13: 6.1e-9, i.e. f32 round-off. This is not a tuned
      // tolerance -- the fixture's answer is exact, so anything several decades
      // above the float epsilon is a defect rather than a precision limit.
      const err = axisPairError(t, d1, d2);
      assert.ok(err < 1e-5, `${what}: recovered axes are not the generating axes, 1 - |dot| = ${err.toExponential(2)}`);
      const nd = dot3(t.Dnormal, asV3(truth.DnormalMath));
      assert.ok(nd > 1 - 1e-5, `${what}: the normal is not the true floor normal, oriented: dot = ${nd}`);
    }
  });
});

test('fit: no votes is reported, not fitted -- and does not inherit the last frame\'s axes', async () => {
  // THE REACHABLE DEGENERACY, and it is not the one §10 declares. The host
  // guards |b1 + b2|^2 < 1e-9 before normalizing, which cannot fire: Jacobi's
  // eigenvectors are orthonormal, so that quantity is identically 2.
  //
  // What can happen is an all-zero scatter matrix -- no lines at all, or every
  // vote degenerate. Every eigenvalue is then zero, the "smallest eigenvector"
  // is whichever identity column the scan happens to reach first, and the
  // pipeline receives a perfectly orthonormal triad computed from no evidence.
  // Nothing downstream could tell.
  //
  // Ordered after the test above deliberately: the buffers are shared across the
  // process, so `triad` holds real axes going in. A failure path that returned
  // early without writing would leave them there and read as a working fit.
  await withDevice(async (device) => {
    const empty = await fitOnVotes(device, { votes: new Float32Array(4), count: 0, maxWeight: 0 });
    assert.equal(empty.status & 32, 32, 'an empty vote set must set fitDegenerate (bit 5)');
    assert.deepEqual([...empty.Drow], [1, 0, 0], 'triad must be written on the failure path too');
    assert.deepEqual([...empty.Dcol], [0, 1, 0]);
    assert.deepEqual([...empty.Dnormal], [0, 0, 1]);

    // Votes present but all zero-weight is the same condition by a different
    // route -- votes.cast writes weight 0 for a degenerate segment rather than
    // dropping it, so this is the path a real frame takes there.
    const zeroWeighted = new Float32Array([0.6, 0.8, 0, 0, 0, 0.6, 0.8, 0]);
    const unweighted = await fitOnVotes(device, { votes: zeroWeighted, count: 2, maxWeight: 0 });
    assert.equal(unweighted.status & 32, 32, 'zero-weight votes are no evidence either');
  });
});

test('fit: on a rendered board, the recovered floor is the true floor', async () => {
  // THE LOAD-BEARING TEST. The synthetic fixture proves the algebra against
  // votes that are exactly consistent; this one runs the whole chain -- gradient
  // through votes -- on an image rendered from a known pose, so the votes carry
  // the detector's real error and the question is whether the fit averages it
  // down rather than amplifying it.
  //
  // It is also the only test here that can see the 21-entry packing order, the
  // 3x3 cross-term halving, and the a<=b unpack, because it is the only one
  // where being self-consistently wrong is not enough to pass.
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const gray = renderPose(TEST_WORLD, pose, FRAME_DIMS, 4);
    const truth = truthFor(TEST_WORLD, pose);

    const bufs = await run(device, (ctx) => {
      device.queue.writeBuffer(ctx.bufs.gray!, 0, Float32Array.from(gray));
      encodeGradient(ctx);
      encodeGrow(ctx, { rhoLow: 0.132, toleranceDeg: 9.5 });
      encodeCollect(ctx, { rhoHigh: 0, minRegionSize: 2 });
      encodeLsdFit(ctx, { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 });
      encodeLines(ctx, { minLengthPx: 3 });
      encodeVotes(ctx, { vFovRad: vFovRadOf(FRAME_DIMS) });
      encodeFit(ctx);
    }, FRAME);

    const t = unpackTriad(
      await readF32(device, bufs.triad!, 12),
      (await readU32(device, bufs.status!, 1))[0]!,
    );
    assertTriadShape(t, 'rendered');
    assert.equal(t.status & 32, 0, 'the fit reported itself degenerate on a real frame');

    const normalDeg = (Math.acos(Math.min(1, dot3(t.Dnormal, asV3(truth.DnormalMath)))) * 180) / Math.PI;
    const axisDeg = (Math.acos(Math.min(1, 1 - axisPairError(t, asV3(truth.DrowMath), asV3(truth.DcolMath)))) * 180) / Math.PI;
    const report = `normal off by ${normalDeg.toFixed(3)} deg, axes off by ${axisDeg.toFixed(3)} deg`;

    // THE DECISIVENESS GATE, and the measurement behind it is worth keeping
    // because it says something the assertion does not.
    //
    // Observed 2026-08-13 on this frame: normal 0.420 deg, axes 0.457 deg, from
    // 68 votes whose own median residual is 0.47 deg. So the fit lands AT the
    // per-vote error, not a factor of sqrt(68) below it -- the detector's error
    // is correlated within a grid family, not independent noise the fit can
    // average away. That is a property of the input, not a defect here, and the
    // check is that it FALLS WITH RESOLUTION rather than sitting at a floor:
    //
    //     tilt 20 yaw 15    96x128  0.420 deg -> 192x256  0.260 -> 360x480  0.214
    //     nadir             96x128  0.357     -> 192x256  0.020 -> 360x480  0.000
    //
    // A bias in the fit would not do that. Nadir reaching exactly zero at 360x480
    // is the clearest statement available that the arithmetic is exact and
    // everything left is discretization.
    //
    // The gates are set against the worst thing this frame size produces rather
    // than against the observed value: tilt 35 yaw 40 at 96x128 measures 1.125
    // deg, so a gate of 1.0 is specific to THIS pose and has ~2.4x headroom. Not
    // generous -- this is the assertion that goes flaky first, and raising the
    // frame size rather than the threshold is the right response if it does.
    // Every structural mutation lands far outside: the unhalved cross term, the
    // transposed unpack and the single Jacobi sweep all fail both gates outright.
    assert.ok(normalDeg < 1, `the recovered floor normal is wrong: ${report}`);
    assert.ok(axisDeg < 2, `the recovered floor axes are wrong: ${report}`);
  });
});

// ── S7 gpp: lines -> two periodic 1D point sets ───────────────────────────
//
// The claim under test is the one Act II rests on, and it is exactly checkable
// from a known pose: gnomonic projection undoes the perspective, so a family's
// rectified values are a LATTICE whose spacing is cellPitch / height. Ground
// truth knows that number, so the test does not have to recover it -- it asks
// whether the points this stage emits actually sit on the lattice truth
// predicts. A wrong triad, a mixed-up projection form, or a swapped xRow/xCol
// destroys that structure without necessarily making any individual value look
// unreasonable.

const GPP = { cellPitch: TEST_CELL_PITCH, minGrazingCos: 0.1 };

/**
 * Weighted circular resultant: fold values onto the unit circle at `period` and
 * measure how tightly they cluster. 1 = a perfect lattice at that spacing,
 * 0 = no periodicity. This is the same construction the period search will
 * maximize, evaluated here at the TRUE period rather than searched for -- so it
 * checks the samples, not the search.
 */
function circularFit(values: number[], weights: number[], period: number): { resultant: number; phase: number } {
  let sc = 0, ss = 0, sw = 0;
  for (let i = 0; i < values.length; i++) {
    const t = (2 * Math.PI * values[i]!) / period;
    sc += weights[i]! * Math.cos(t);
    ss += weights[i]! * Math.sin(t);
    sw += weights[i]!;
  }
  if (sw < 1e-9) return { resultant: 0, phase: 0 };
  return { resultant: Math.hypot(sc, ss) / sw, phase: (Math.atan2(ss, sc) * period) / (2 * Math.PI) };
}

/** The whole chain from an image to the two compacted families. */
async function gppOn(device: GPUDevice, gray: Float64Array, dims: Dims, dd: { w: number; h: number; horizFovDeg: number }) {
  const bufs = await run(device, (ctx) => {
    device.queue.writeBuffer(ctx.bufs.gray!, 0, Float32Array.from(gray));
    encodeGradient(ctx);
    encodeGrow(ctx, { rhoLow: 0.132, toleranceDeg: 9.5 });
    encodeCollect(ctx, { rhoHigh: 0, minRegionSize: 2 });
    encodeLsdFit(ctx, { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 });
    encodeLines(ctx, { minLengthPx: 3 });
    encodeVotes(ctx, { vFovRad: vFovRadOf(dd) });
    encodeFit(ctx);
    encodeGpp(ctx, { ...GPP, vFovRad: vFovRadOf(dd) });
  }, dims);

  const counts = await readU32(device, bufs.familyCounts!, 2);
  const rowCount = counts[0]!, colCount = counts[1]!;
  // (value, weight, crossMin, crossMax) interleaved -- see `samples` in
  // pipeline.ts. The cross lanes are the decode hull's per-line contribution.
  const rowQuads = await readF32(device, bufs.rowSamples!, Math.max(1, rowCount) * 4);
  const colQuads = await readF32(device, bufs.colSamples!, Math.max(1, colCount) * 4);
  const lane = (a: Float32Array, count: number, k: number) =>
    Array.from({ length: count }, (_, i) => a[i * 4 + k]!);
  const rowValues = lane(rowQuads, rowCount, 0);
  const rowWeights = lane(rowQuads, rowCount, 1);
  const rowCross = Array.from({ length: rowCount }, (_, i) => [rowQuads[i * 4 + 2]!, rowQuads[i * 4 + 3]!] as const);
  const colValues = lane(colQuads, colCount, 0);
  const colWeights = lane(colQuads, colCount, 1);
  const colCross = Array.from({ length: colCount }, (_, i) => [colQuads[i * 4 + 2]!, colQuads[i * 4 + 3]!] as const);
  const extent = await readF32(device, bufs.extent!, 12);
  const lineCount = (await readU32(device, bufs.lineCount!, 2))[0]!;
  const result = await readF32(device, bufs.gppResult!, 4);
  // topK[0] is (count, 0, 0, 0); topK[1..] are (period, score, phiRow, phiCol).
  const tk = await readF32(device, bufs.topK!, 32);
  const dn = await readF32(device, bufs.distinctness!, 8);
  const candidates = Array.from({ length: Math.round(tk[0]!) }, (_, i) => ({
    period: tk[4 + i * 4]!, score: tk[5 + i * 4]!, distinctness: dn[i]!,
  }));
  return {
    bufs, rowCount, colCount, rowValues, rowWeights, rowCross, colValues, colWeights, colCross,
    extent, lineCount,
    period: result[0]!, phiRow: result[1]!, phiCol: result[2]!, height: result[3]!, candidates,
    // extent[5..8], named by AXIS: xRow is what u is scaled from, xCol is v.
    hull: { xRowMin: extent[5]!, xRowMax: extent[6]!, xColMin: extent[7]!, xColMax: extent[8]! },
  };
}

test('gpp: the rectified values are a lattice, and it is the true one', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const truth = truthFor(TEST_WORLD, pose);
    const g = await gppOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);

    assert.ok(g.lineCount > 40, `only ${g.lineCount} lines -- the detector, not gpp, is the problem`);
    // Both families populated. A classify that collapsed the two grid
    // directions onto one would leave a perfectly periodic single family and
    // every assertion below would still pass.
    assert.ok(g.rowCount > 5 && g.colCount > 5,
      `families ${g.rowCount}/${g.colCount} -- classify is not separating the two grid directions`);
    assert.equal(g.rowCount + g.colCount, g.lineCount,
      'a line went into neither family, or into both');

    // ── SCORED AT THE PEAK, NOT AT THE TRUE PERIOD, AND THAT IS THE FINDING ──
    //
    // The obvious test is the resultant evaluated AT truth.period, and a first
    // version of this asserted exactly that. It is the wrong metric, because it
    // is a JOINT statement about this stage and the triad it was handed --
    // measured 2026-08-13 across five poses at two frame sizes:
    //
    //   pose                   96x128                    192x256
    //                    axisErr  res@truth  peak  Perr   axisErr  res@truth  peak  Perr
    //   nadir             0.36     0.990    0.992  0.29%   0.02     0.997    0.998  0.14%
    //   tilt20 yaw15      0.42     0.987    0.996  0.48%   0.26     0.974    0.994  0.70%
    //   tilt30 yaw25      0.88     0.411    0.846  4.42%   0.25     0.828    0.979  1.54%
    //   tilt35 yaw40      0.76     0.978    0.981  0.21%   0.16     0.921    0.991  0.95%
    //   tilt20 yaw15 h16  0.46     0.859    0.963  1.37%   0.36     0.894    0.976  0.99%
    //
    // tilt30 yaw25 reads 0.411 at the true period while its peak is a perfectly
    // respectable 0.846. Nothing is wrong with the samples there: the fit's
    // 0.88 deg axis error rescales the gnomonic coordinates, which shows up as a
    // 4.4% PERIOD error, and the values span ~26 cells -- so folding at the true
    // period accumulates over a full period of drift across the data and the
    // phases spread right round the circle. The resultant at a fixed period is
    // hypersensitive to period error by construction, which is precisely WHY the
    // search downstream exists and is not a defect this stage can fix.
    //
    // So the two claims are separated. Both are still checked; they just stop
    // being conflated into one number that no threshold can serve.
    //
    //   1. THE SAMPLES ARE A CLEAN LATTICE -- gpp's own claim. Scored at the
    //      best period in a +/-20% window (narrow enough that a sub-multiple
    //      cannot win it, which is a separate stage's problem). Worst observed
    //      across all ten configurations above: 0.846.
    //   2. THAT LATTICE IS THE TRUE ONE -- gpp's claim jointly with the fit's.
    //      Worst observed: 4.42%, and it HALVES with resolution, which is what
    //      says it is upstream discretization rather than a scale error here.
    const scoreAt = (P: number) =>
      (circularFit(g.rowValues, g.rowWeights, P).resultant
        + circularFit(g.colValues, g.colWeights, P).resultant) / 2;
    let peak = { period: truth.period, score: -1 };
    for (let k = -2000; k <= 2000; k++) {
      const P = truth.period * (1 + k * 0.0001);
      const s = scoreAt(P);
      if (s > peak.score) peak = { period: P, score: s };
    }
    const periodErr = peak.period / truth.period - 1;
    const report =
      `families ${g.rowCount}/${g.colCount}, true period ${truth.period}, ` +
      `resultant at truth ${scoreAt(truth.period).toFixed(4)}, ` +
      `peak ${peak.score.toFixed(4)} at ${(periodErr * 100).toFixed(2)}% off truth`;

    // A wrong rectification does not land near either gate. Swapping which
    // coordinate a family is measured along makes a line's own two endpoints
    // disagree -- the value stops being a property of the line at all -- and the
    // peak falls into the low tenths with nowhere periodic to hide.
    assert.ok(peak.score > 0.8, `the rectified values are not a lattice at any period: ${report}`);
    assert.ok(Math.abs(periodErr) < 0.06, `the lattice is not the true one: ${report}`);
  });
});

// WHICH family is "row" is NOT recoverable here, and this test learned that the
// hard way by coming back 68 wrong out of 68 -- a perfect inversion, which is
// the signature of a labelling that is right about the PARTITION and arbitrary
// about the names. fit.eigen picks two plane normals out of a 3x3
// eigendecomposition and calls them Drow and Dcol in eigenvalue order; nothing
// in Act I can know which of them is the board's rows, and nothing needs to --
// decode's four-orientation search is what resolves it (§13), and §14's closed
// form consumes the pair rather than the labels.
//
// So the assertion is on the PARTITION, up to a global swap. That is still
// decisive: a classify that separated the families badly gives a MIX, which is
// neither 0 nor all.
test('gpp: classify splits the lines the way ground truth does, up to the swap', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const truth = truthFor(TEST_WORLD, pose);
    const g = await gppOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);

    // classify tests the recovered triad against itself. This tests it against
    // the axes that GENERATED the image, so a triad that is self-consistently
    // rotated -- which the fit's own tests would accept up to their tolerance --
    // shows up here as lines landing in the wrong family.
    const votes = await readF32(device, g.bufs.votes!, FRAME.maxLines * 4);
    const family = await readU32(device, g.bufs.family!, FRAME.maxLines * 2);
    const dotAbs = (i: number, a: { x: number; y: number; z: number }) =>
      Math.abs(votes[i * 4]! * a.x + votes[i * 4 + 1]! * a.y + votes[i * 4 + 2]! * a.z);

    let agree = 0, disagree = 0;
    for (let i = 0; i < g.lineCount; i++) {
      if (votes[i * 4 + 3]! <= 0) continue;
      const truthIsRow = dotAbs(i, truth.DrowMath) < dotAbs(i, truth.DcolMath);
      const gpuIsRow = family[i * 2] === 1;
      if (truthIsRow === gpuIsRow) agree++; else disagree++;
    }
    const total = agree + disagree;
    assert.ok(total > 40, `only ${total} classifiable lines`);
    assert.ok(agree === 0 || disagree === 0,
      `the partition does not match ground truth: ${agree} agree, ${disagree} disagree ` +
      `-- all of one or all of the other is a naming swap, a MIX is a classification error`);
  });
});

test('gpp: the extent is PER FAMILY, not the pooled range', async () => {
  await withDevice(async (device) => {
    // Deliberately oblique and yawed, because that is what makes the two
    // families' coordinate ranges differ. At nadir with the board centred they
    // very nearly coincide and this fixture could not tell the bug from the fix.
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 30, yawDeg: 25 };
    const g = await gppOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
    assert.ok(g.rowCount > 5 && g.colCount > 5, `families ${g.rowCount}/${g.colCount}`);

    const span = (v: number[]) => Math.max(...v) - Math.min(...v);
    const rowSpan = span(g.rowValues), colSpan = span(g.colValues);
    const pooled = span([...g.rowValues, ...g.colValues]);
    const [rlo, rhi, clo, chi, spread] = [...g.extent];

    // The reduction reports the actual per-family min and max.
    assert.ok(Math.abs(rlo! - Math.min(...g.rowValues)) < 1e-6, 'rowMin');
    assert.ok(Math.abs(rhi! - Math.max(...g.rowValues)) < 1e-6, 'rowMax');
    assert.ok(Math.abs(clo! - Math.min(...g.colValues)) < 1e-6, 'colMin');
    assert.ok(Math.abs(chi! - Math.max(...g.colValues)) < 1e-6, 'colMax');

    // THE FIXTURE HAS TO BE ABLE TO SEE THE BUG, and this asserts that it can
    // BEFORE asserting the behaviour -- the pooled range at this pose really is
    // wider than either family's, so `spread` picking the pooled one would be
    // visible. At a pose where the two families coincide it would not be, which
    // is the shape five earlier mutations in this rewrite hid behind.
    assert.ok(pooled > Math.max(rowSpan, colSpan) * 1.1,
      `this pose cannot distinguish per-family from pooled: ` +
      `row ${rowSpan.toFixed(4)} col ${colSpan.toFixed(4)} pooled ${pooled.toFixed(4)}`);
    assert.ok(Math.abs(spread! - Math.max(rowSpan, colSpan)) < 1e-6,
      `spread is ${spread} -- the per-family maximum is ${Math.max(rowSpan, colSpan)}, ` +
      `the pooled range is ${pooled}`);
  });
});

// ── THE DECODE HULL (§12) ─────────────────────────────────────────────────
//
// gpp.extent's second job: bound the decode lattice by the detected LINES
// instead of by the view quadrilateral. Measured and adopted 2026-08-14
// (scripts/hull-measure.ts) against src/pose; this is the same quantity computed
// inside src/pose2, so what needs testing here is the reduction, not the policy.

/**
 * The gnomonic extent of the floor the camera can actually resolve, from GROUND
 * TRUTH -- the same ray cast projectedUVBounds does, in the true triad rather
 * than a recovered one. Cell counts, so it is comparable to the hull without
 * carrying a scale.
 */
function trueVisibleCells(pose: { height: number; tiltDeg: number; yawDeg: number; overRow: number; overCol: number }, dd: typeof FRAME_DIMS, minGrazingCos: number) {
  const truth = truthFor(TEST_WORLD, pose);
  const tanHalf = Math.tan(vFovRadOf(dd) / 2);
  const aspect = dd.w / dd.h;
  const Dn = truth.DnormalMath.clone();
  // The pipeline's convention, decided once in fit.eigen: the normal points at
  // the camera, and the camera looks down -z.
  if (Dn.z < 0) Dn.negate();
  let rowMin = Infinity, rowMax = -Infinity, colMin = Infinity, colMax = -Infinity;
  const N = 48;
  const d = { x: 0, y: 0, z: 0 };
  for (let iy = 0; iy <= N; iy++) {
    for (let ix = 0; ix <= N; ix++) {
      rayDirInto(d, (ix / N) * 2 - 1, (iy / N) * 2 - 1, MATH_QUAT, tanHalf, aspect);
      const den = d.x * Dn.x + d.y * Dn.y + d.z * Dn.z;
      if (!(den < -minGrazingCos)) continue;
      const xRow = -(d.x * truth.DrowMath.x + d.y * truth.DrowMath.y + d.z * truth.DrowMath.z) / den;
      const xCol = -(d.x * truth.DcolMath.x + d.y * truth.DcolMath.y + d.z * truth.DcolMath.z) / den;
      rowMin = Math.min(rowMin, xRow); rowMax = Math.max(rowMax, xRow);
      colMin = Math.min(colMin, xCol); colMax = Math.max(colMax, xCol);
    }
  }
  // A gnomonic span divided by the period is a count of cells.
  return [(rowMax - rowMin) / truth.period, (colMax - colMin) / truth.period].sort((a, b) => a - b);
}

test('gpp: the hull is the min/max of every classified ENDPOINT, in both axes', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const g = await gppOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
    assert.ok(g.rowCount > 5 && g.colCount > 5, `families ${g.rowCount}/${g.colCount}`);

    // ── 1. THE REDUCTION, RE-DERIVED FROM ITS OWN INPUTS ──
    //
    // Each AXIS is bounded by one family's value and the OTHER family's cross,
    // and getting that pairing backwards is the mistake this stage invites --
    // "row family, row lane" reads right and is wrong, because a row line's
    // value is an xCol.
    const flat = (xs: readonly (readonly [number, number])[]) => xs.flatMap((p) => [p[0], p[1]]);
    const trueXRowMin = Math.min(...g.colValues, ...flat(g.rowCross));
    const trueXRowMax = Math.max(...g.colValues, ...flat(g.rowCross));
    const trueXColMin = Math.min(...g.rowValues, ...flat(g.colCross));
    const trueXColMax = Math.max(...g.rowValues, ...flat(g.colCross));
    assert.ok(Math.abs(g.hull.xRowMin - trueXRowMin) < 1e-6, `hull xRowMin ${g.hull.xRowMin} vs ${trueXRowMin}`);
    assert.ok(Math.abs(g.hull.xRowMax - trueXRowMax) < 1e-6, `hull xRowMax ${g.hull.xRowMax} vs ${trueXRowMax}`);
    assert.ok(Math.abs(g.hull.xColMin - trueXColMin) < 1e-6, `hull xColMin ${g.hull.xColMin} vs ${trueXColMin}`);
    assert.ok(Math.abs(g.hull.xColMax - trueXColMax) < 1e-6, `hull xColMax ${g.hull.xColMax} vs ${trueXColMax}`);

    // ── 2. THE CROSS LANES CARRY INFORMATION ──
    //
    // A version that forgot the cross lanes -- or paired them with the wrong
    // axis -- produces the per-family VALUE box (extent[0..3]) instead, and that
    // box is a perfectly plausible hull. So the gate is not containment, which
    // that version also satisfies; it is how much WIDER the hull is, measured
    // rather than assumed.
    //
    // Measured 2026-08-14 at this pose: the cross lanes add 1.78 cells in xRow
    // and 2.63 in xCol, against a value box of 14.8 x 17.2. Small by
    // construction -- a detected grid covers the view in both directions, so the
    // outermost perpendicular line already sits near the edge and what the cross
    // lanes add is only however far the lines run PAST it. 0.5 has ~3.5x headroom
    // over the smaller of the two and is not reachable at all without them.
    const [rlo, rhi, clo, chi] = [g.extent[0]!, g.extent[1]!, g.extent[2]!, g.extent[3]!];
    const addedXRow = ((clo - g.hull.xRowMin) + (g.hull.xRowMax - chi)) / g.period;
    const addedXCol = ((rlo - g.hull.xColMin) + (g.hull.xColMax - rhi)) / g.period;
    const added = `xRow +${addedXRow.toFixed(2)} cells, xCol +${addedXCol.toFixed(2)} cells`;
    assert.ok(addedXRow > 0.5 && addedXCol > 0.5,
      `the hull is barely wider than the per-family value box (${added}) -- ` +
      `the cross lanes are missing, mispaired, or reduced against the wrong family`);

    // ── 3. AGAINST GROUND TRUTH ──
    //
    // The hull spans about as many cells as the camera can actually resolve. The
    // comparison is on SPANS, sorted: which recovered axis is "row" is not
    // recoverable (see the classify test) and an eigenvector's sign is arbitrary,
    // so neither the pairing nor the offset survives into a truth comparison --
    // but the two spans do.
    const hullCells = [
      (g.hull.xRowMax - g.hull.xRowMin) / g.period,
      (g.hull.xColMax - g.hull.xColMin) / g.period,
    ].sort((a, b) => a - b);
    const trueCells = trueVisibleCells(pose, FRAME_DIMS, GPP.minGrazingCos);
    console.log(`    hull ${hullCells.map((c) => c.toFixed(1)).join('x')} cells ` +
      `(${added} over the value box), truly visible ${trueCells.map((c) => c.toFixed(1)).join('x')}`);
    for (let k = 0; k < 2; k++) {
      const ratio = hullCells[k]! / trueCells[k]!;
      assert.ok(ratio > 0.7 && ratio < 1.15,
        `hull axis ${k} covers ${(ratio * 100).toFixed(0)}% of the truly visible extent ` +
        `(${hullCells[k]!.toFixed(1)} of ${trueCells[k]!.toFixed(1)} cells)`);
    }

    // ── 4. THE NADIR IS INSIDE IT ──
    //
    // (xRow, xCol) = (0, 0) is the point directly beneath the camera, whatever
    // the axis naming or the eigenvector signs -- so this is the one part of the
    // hull's POSITION that ground truth can pin without resolving either
    // ambiguity. At tilt 20 the nadir is well inside a 37 deg half-FOV.
    assert.ok(g.hull.xRowMin < 0 && g.hull.xRowMax > 0 && g.hull.xColMin < 0 && g.hull.xColMax > 0,
      `the nadir is outside the hull: xRow [${g.hull.xRowMin}, ${g.hull.xRowMax}], ` +
      `xCol [${g.hull.xColMin}, ${g.hull.xColMax}]`);
  });
});

test('gpp: the family array is TOTAL -- a frame cannot inherit the last one\'s families', async () => {
  await withDevice(async (device) => {
    // Run 1 must flag slots run 2's dispatch could not reach. §9's version of
    // this test was green under its own mutation because both runs dispatched
    // ONE 64-thread workgroup and wrote the same 64 slots -- so the stale tail
    // was never touched by either. A rendered board gives well over 64 lines,
    // and the blank frame gives zero, which under an indirect dispatch is zero
    // workgroups and therefore no writes at all.
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const busy = await gppOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
    assert.ok(busy.lineCount > 64,
      `run 1 produced ${busy.lineCount} lines -- it must exceed one workgroup or this test is blind`);

    const quiet = await gppOn(device, new Float64Array(FRAME.w * FRAME.h), FRAME, FRAME_DIMS);
    assert.equal(quiet.lineCount, 0, 'the blank frame produced lines');
    assert.equal(quiet.rowCount, 0, `${quiet.rowCount} row lines inherited from the previous frame`);
    assert.equal(quiet.colCount, 0, `${quiet.colCount} column lines inherited from the previous frame`);
    // And the no-samples bit is set rather than the frame quietly carrying on
    // with a zero spread.
    const status = (await readU32(device, quiet.bufs.status!, 1))[0]!;
    assert.equal(status & 64, 64, `gppNoSamples was not set on an empty frame (status ${status})`);
  });
});

// ── S7 gpp, the search ────────────────────────────────────────────────────

test('gpp: the recovered period and height are the true ones', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const truth = truthFor(TEST_WORLD, pose);
    const g = await gppOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);

    const periodErr = g.period / truth.period - 1;
    const heightErr = g.height / truth.height - 1;
    const report =
      `period ${g.period.toFixed(5)} vs ${truth.period} (${(periodErr * 100).toFixed(2)}%), ` +
      `height ${g.height.toFixed(3)} vs ${truth.height} (${(heightErr * 100).toFixed(2)}%)`;

    // THE GATE, and what sets it. Measured 2026-08-13 across six poses at two
    // frame sizes, period error against truth:
    //
    //                        96x128   192x256
    //   nadir                 0.29%     0.14%
    //   tilt20 yaw15          0.49%     0.70%
    //   tilt30 yaw25          4.42%     1.55%
    //   tilt35 yaw40          0.21%     0.95%
    //   tilt20 yaw15 h=16     1.37%     0.99%
    //   tilt10 yaw60 h=6      0.69%     0.13%
    //
    // Every one of those equals the error of the PEAK in the sample values --
    // measured independently in the lattice test above -- to the digit. So the
    // search is finding the true maximum and what is left is entirely the fit's
    // axis error rescaling the gnomonic coordinates upstream. tilt30 yaw25 is
    // the outlier in both tables for the same reason: 0.88 deg of axis error at
    // 96x128, falling to 0.25 deg at 192x256, and the period error falls with it.
    //
    // 2% on the fixture pose is ~4x headroom over its 0.49%. It is deliberately
    // NOT set to cover tilt30 yaw25 at this frame size -- that pose's error is
    // upstream and a gate wide enough to admit it would stop testing this stage.
    assert.ok(Math.abs(periodErr) < 0.02, `the period is wrong: ${report}`);
    // height = cellPitch / period, so this is the same claim in the units Act III
    // consumes. Asserted separately because that division is where a cellPitch
    // that never reached the shader would show up, and it would show up nowhere
    // else in this stage.
    assert.ok(Math.abs(heightErr) < 0.02, `the height is wrong: ${report}`);

    // ── THE REFINEMENT HAS TO BE OBSERVED SEPARATELY ──
    //
    // Deleting the golden-section step entirely passes both gates above, and
    // that is not a hole in the gates -- it is that the refinement's benefit is
    // smaller than the upstream axis error the gates have to leave room for.
    // Measured 2026-08-13, integer-count candidate against refined:
    //
    //   nadir            0.61% -> 0.29%
    //   tilt20 yaw15     0.76% -> 0.49%
    //   tilt35 yaw40    -0.32% -> 0.21%
    //   tilt20 yaw15 h16 1.59% -> 1.37%
    //
    // It roughly halves the error, every time, and no accuracy threshold wide
    // enough to pass this frame can see that. So the assertion is structural
    // instead: the answer MOVED OFF the candidate grid and stayed inside the
    // winner's own one-count neighbourhood.
    const nearest = g.candidates.reduce((a, b) =>
      (Math.abs(b.period - g.period) < Math.abs(a.period - g.period) ? b : a));
    assert.notEqual(g.period, nearest.period,
      'the period is exactly a candidate -- the golden-section refinement did not run');
    // The bracket is [spread/(n+1), spread/(n-1)], which is about 2/n wide. n is
    // 14 to 30 on these frames, so 10% is the loosest that bracket ever is.
    assert.ok(Math.abs(g.period / nearest.period - 1) < 0.1,
      `the refinement left its own bracket: ${g.period} from candidate ${nearest.period}`);
  });
});

test('gpp: the sub-multiples are present, score as well, and lose anyway', async () => {
  await withDevice(async (device) => {
    // The harmonic problem is the whole reason gpp.distinct exists, and a test
    // that merely checks the winner cannot tell "the tie was broken correctly"
    // from "there was no tie". So this asserts the TIE EXISTS first.
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const truth = truthFor(TEST_WORLD, pose);
    const g = await gppOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
    const shown = g.candidates.map((c) =>
      `P=${c.period.toFixed(4)} res=${c.score.toFixed(2)} d=${c.distinctness.toFixed(1)}`).join(' | ');

    assert.ok(g.candidates.length >= 2, `only ${g.candidates.length} candidates: ${shown}`);
    const fundamental = g.candidates.reduce((a, b) => (a.period > b.period ? a : b));

    // A sub-multiple of the true period IS among the candidates. It has to be:
    // a lattice at spacing P0 lies on every finer sub-lattice, so P0/2 folds
    // just as coherently and the resultant has no way to reject it.
    const half = g.candidates.find((c) => Math.abs(c.period / (truth.period / 2) - 1) < 0.05);
    assert.ok(half, `no half-period candidate to disambiguate against: ${shown}`);

    // AND ITS RESULTANT IS EFFECTIVELY TIED. Observed 2026-08-13: 1.99 against
    // 1.95, a 2% gap. This is the assertion that proves the test can see the
    // bug -- deleting the distinctness test entirely would leave the winner
    // decided by that 2%, which is noise.
    assert.ok(half.score > fundamental.score * 0.9,
      `the resultant already separates these, so this fixture is not testing ` +
      `harmonic disambiguation at all: ${shown}`);

    // Image content is what actually decides, and it is not close: the
    // fundamental's cell centres are distinct De Bruijn cells while the
    // half-period's repeat.
    //
    // THE GATE IS SET FROM BOTH SIDES, and a first version at 1.5 was a near
    // miss rather than a gate. Measured 2026-08-13 on this frame:
    //
    //   correct (centres, phase + half a cell)   118.3 / 58.2 = 2.03
    //   mutated (sampled at the phase itself)     98.2 / 64.0 = 1.53
    //
    // Sampling ON the cell boundaries does not destroy distinctness -- a
    // boundary pixel is the average of two cells, and adjacent boundaries still
    // differ when the cells do -- it COMPRESSES the margin, which is why a
    // threshold picked by eye slipped underneath it by 0.03. 1.8 sits between
    // the two measurements with headroom on each side.
    assert.ok(fundamental.distinctness > half.distinctness * 1.8,
      `distinctness does not separate the fundamental from its half: ${shown}`);
    // And the winner is the fundamental, not the half.
    assert.ok(Math.abs(g.period / truth.period - 1) < 0.02,
      `a sub-multiple won: recovered ${g.period}, truth ${truth.period}: ${shown}`);
  });
});

test('gpp: a frame with no lines reports no period, not the last frame\'s', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const busy = await gppOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
    assert.ok(busy.period > 0, 'the fixture frame did not recover a period to inherit');

    const quiet = await gppOn(device, new Float64Array(FRAME.w * FRAME.h), FRAME, FRAME_DIMS);
    // Every one of these is a buffer whose previous contents were a valid
    // answer, which is the shape §16 says a single run structurally cannot see.
    assert.equal(quiet.period, 0, `period inherited from the previous frame: ${quiet.period}`);
    assert.equal(quiet.height, 0, `height inherited from the previous frame: ${quiet.height}`);
    assert.equal(quiet.candidates.length, 0,
      `${quiet.candidates.length} candidates inherited from the previous frame`);
    const status = (await readU32(device, quiet.bufs.status!, 1))[0]!;
    assert.equal(status & 128, 128, `gppNoCandidates was not set (status ${status})`);
  });
});

// ── S8 decode.layout: the sampling lattice ────────────────────────────────
//
// Act III's first stage, and the one §12 restructured: the lattice is bounded by
// the detected LINES rather than by the view quadrilateral, so this reads gpp's
// hull instead of casting 2,401 rays of its own.
//
// Ground truth knows the answer here. The lattice is a set of floor cells, the
// pose that generated the image is known, so "does cell (i, j) land on the pixel
// the true geometry says it should" is checkable directly rather than against a
// second implementation.

const LAYOUT = { cellPitch: TEST_CELL_PITCH, minGrazingCos: 0.1 };

/**
 * The 128-byte layout block, read both ways because it is a mixed struct.
 *
 * The offsets are hand-derived from DECODE_LAYOUT_WGSL's `Layout`, which is
 * exactly the trap that shader's header records: the three axes are vec4 and
 * only xyz is used SO THAT these offsets are the obvious ones. As vec3 they
 * would be 12 bytes with align 16, and every scalar after them would pack four
 * bytes earlier than a reader written from the field order expects.
 */
async function readLayout(device: GPUDevice, buf: GPUBuffer) {
  const f = await readF32(device, buf, 32);
  const i = await readU32(device, buf, 32);
  const ints = new Int32Array(i.buffer, i.byteOffset);
  const v3 = (at: number) => ({ x: f[at]!, y: f[at + 1]!, z: f[at + 2]! });
  return {
    Drow: v3(0), Dcol: v3(4), normal: v3(8),
    distance: f[12]!, tanHalf: f[13]!, aspect: f[14]!, minGrazingCos: f[15]!,
    uPhase: f[16]!, vPhase: f[17]!, cellPitch: f[18]!, binThreshold: f[19]!,
    rows: i[20]!, cols: i[21]!, imageW: i[22]!, imageH: i[23]!,
    kMinU: ints[24]!, kMinV: ints[25]!, zeroI: i[26]!, zeroJ: i[27]!, valid: i[28]!,
  };
}

/** The full chain through decode.layout, plus the layout block decoded. */
async function layoutOn(device: GPUDevice, gray: Float64Array, dims: Dims, dd: typeof FRAME_DIMS) {
  const bufs = await run(device, (ctx) => {
    device.queue.writeBuffer(ctx.bufs.gray!, 0, Float32Array.from(gray));
    encodeGradient(ctx);
    encodeGrow(ctx, { rhoLow: 0.132, toleranceDeg: 9.5 });
    encodeCollect(ctx, { rhoHigh: 0, minRegionSize: 2 });
    encodeLsdFit(ctx, { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 });
    encodeLines(ctx, { minLengthPx: 3 });
    encodeVotes(ctx, { vFovRad: vFovRadOf(dd) });
    encodeFit(ctx);
    encodeGpp(ctx, { ...LAYOUT, vFovRad: vFovRadOf(dd) });
    encodeDecodeLayout(ctx, { ...LAYOUT, vFovRad: vFovRadOf(dd) });
  }, dims);

  return {
    bufs,
    ...await readLayout(device, bufs.layout!),
    buildArgs: await readU32(device, bufs.buildArgs!, 3),
    tallyArgs: await readU32(device, bufs.tallyArgs!, 12),
    status: (await readU32(device, bufs.status!, 1))[0]!,
    binThreshRef: gray.reduce((a, b) => a + b, 0) / gray.length,
  };
}

test('decode.layout: the lattice IS the true floor lattice, in pitch and in phase', async () => {
  await withDevice(async (device) => {
    // ── THE CAMERA IS DELIBERATELY OFF A CELL CENTRE IN BOTH AXES ──
    //
    // `phiRow` is the ROW family's phase and lives in xCol, so it becomes v;
    // `phiCol` becomes u. That crossover is easy to get backwards and it was
    // INVISIBLE to every fixture in this suite, because every one of them sat at
    // overRow/overCol = *.5 -- a cell centre in both axes, where the two phases
    // are equal and swapping them is exactly a no-op. Found by mutation
    // 2026-08-14. Measured, clean against swapped, on `refOff` below:
    //
    //   over(40.5, 40.5)   0.127 -> 0.042   (the swap IMPROVES it -- pure symmetry)
    //   over(40.25, 40.75) 0.147 -> 0.172   still not separated
    //   over(40.5, 40.2)   0.156 -> 0.229
    //   over(40.1, 40.6)   0.123 -> 0.335   <- this fixture
    //   over(40, 40.5)     0.126 -> 0.491   (a boundary-sitting pose)
    //
    // 0.1 and 0.6 are generic fractions rather than the largest separator, and
    // the gate at 0.2 sits with ~1.6x headroom on each side of it.
    const pose = { height: 10, overRow: 40.1, overCol: 40.6, tiltDeg: 20, yawDeg: 15 };
    const truth = truthFor(TEST_WORLD, pose);
    const L = await layoutOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);

    assert.equal(L.valid, 1, `layout reported invalid (status ${L.status})`);
    assert.equal(L.status & 256, 0, 'layoutInvalid is set');

    // ── THE MEAN GREY LEVEL ──
    //
    // A hierarchical f32 reduction against the f64 host sum. The relative gate
    // is what says the hierarchy is doing its job: a flat serial f32
    // accumulation over 12,288 pixels of ~128 reaches ~1.6e6, where the ULP is
    // 0.125 and the drift is real.
    assert.ok(Math.abs(L.binThreshold / L.binThreshRef - 1) < 1e-5,
      `binThreshold ${L.binThreshold} vs ${L.binThreshRef}`);

    // ── WHERE A LATTICE CELL REALLY LOOKS, PER THE POSE THAT RENDERED THE FRAME ──
    //
    // Project a cell exactly as decode.build will, then cast that pixel back
    // through the TRUE camera and intersect the true floor. The result is in
    // world units where a cell is one cellPitch and centres sit at the half-integers,
    // so both of this stage's own claims are readable off it directly.
    const tanHalf = Math.tan(vFovRadOf(FRAME_DIMS) / 2);
    const aspect = FRAME.w / FRAME.h;
    const cam = truth.camPos;
    const d = { x: 0, y: 0, z: 0 };
    const trueHitOf = (i: number, j: number): { hx: number; hz: number } | null => {
      const v = L.vPhase + (L.kMinV + i) * L.cellPitch;
      const uu = L.uPhase + (L.kMinU + j) * L.cellPitch;
      const p = {
        x: L.Drow.x * uu + L.Dcol.x * v - L.normal.x * L.distance,
        y: L.Drow.y * uu + L.Dcol.y * v - L.normal.y * L.distance,
        z: L.Drow.z * uu + L.Dcol.z * v - L.normal.z * L.distance,
      };
      const len = Math.hypot(p.x, p.y, p.z);
      if (!(L.distance / len > L.minGrazingCos) || p.z >= 0) return null;
      const px = ((-p.x / (p.z * tanHalf * aspect) + 1) / 2) * FRAME.w;
      const py = ((1 - -p.y / (p.z * tanHalf)) / 2) * FRAME.h;
      if (!(px >= 0 && px < FRAME.w && py >= 0 && py < FRAME.h)) return null;
      rayDirInto(d, (px / FRAME.w) * 2 - 1, 1 - (py / FRAME.h) * 2, truth.camQuat, tanHalf, aspect);
      if (d.y >= -1e-12) return null;
      const t = -cam.y / d.y;
      return { hx: cam.x + t * d.x, hz: cam.z + t * d.z };
    };
    const fromCentre = (x: number) => Math.abs(x - Math.floor(x) - 0.5);

    // ── 1. THE PITCH -- this stage's own claim, isolated ──
    //
    // One lattice step must be one board cell on the real floor. Measured over a
    // span rather than between neighbours, so the ~1e-3 cell of reprojection
    // noise divides down. This is what separates decode.layout's arithmetic from
    // the drift below: it is a SCALE, so accumulated position error cancels.
    const ref = trueHitOf(L.zeroI, L.zeroJ);
    assert.ok(ref, 'the reference cell does not project on screen');
    // The longest span from the reference that stays on screen, up to 10 cells.
    // Fixed offsets do not survive: the reference sits at the NADIR, which an
    // oblique view puts near one edge of a lattice that reaches away from it.
    const pitchAlong = (di: number, dj: number): { pitch: number; span: number } => {
      let best = { pitch: NaN, span: 0 };
      for (let k = 1; k <= 10; k++) {
        const h = trueHitOf(L.zeroI + di * k, L.zeroJ + dj * k);
        if (h) best = { pitch: Math.hypot(h.hx - ref.hx, h.hz - ref.hz) / k, span: k };
      }
      return best;
    };
    // Either direction along the axis -- whichever reaches further on screen.
    const alongJ = [pitchAlong(0, 1), pitchAlong(0, -1)].sort((a, b) => b.span - a.span)[0]!;
    const alongI = [pitchAlong(1, 0), pitchAlong(-1, 0)].sort((a, b) => b.span - a.span)[0]!;
    assert.ok(alongJ.span >= 4 && alongI.span >= 4,
      `only ${alongJ.span}/${alongI.span} cells of on-screen lattice either side of the reference`);
    const pitchJ = alongJ.pitch, pitchI = alongI.pitch;

    // ── 2. THE PHASE -- also this stage's own, and also isolated ──
    //
    // The reference cell sits under the camera, where accumulated scale error is
    // zero by construction, so its distance from a true cell CENTRE is the phase
    // arithmetic on its own. Half a cell out is the failure this invites, and it
    // is the one that still produces a perfectly regular, perfectly plausible
    // grid -- every sample would land on a boundary, where the grey is the
    // average of two cells and the bit is a coin toss.
    const refOff = Math.max(fromCentre(ref.hx), fromCentre(ref.hz));

    // ── 3. THE JOINT CLAIM, reported and gated loosely ──
    //
    // How much of the lattice lands near a true centre is NOT this stage's claim
    // alone -- it is this stage plus every upstream error, accumulated over the
    // lattice's radius. Measured 2026-08-14 at this pose: 100% inside radius 8,
    // ~80% beyond it, worst 0.356 cells at the far corner. That profile IS the
    // upstream signature (zero at the reference, growing with radius); a defect
    // here would be flat across the lattice instead. The pitch above says the
    // same thing independently at 1.006 against a period recovered to 0.5%.
    let onScreen = 0, centred = 0, worst = 0;
    const bands = [0, 4, 8, 12, 1e9];
    const hit = bands.slice(0, -1).map(() => [0, 0]);
    for (let i = 0; i < L.rows; i++) {
      for (let j = 0; j < L.cols; j++) {
        const h = trueHitOf(i, j);
        if (!h) continue;
        onScreen++;
        const off = Math.max(fromCentre(h.hx), fromCentre(h.hz));
        worst = Math.max(worst, off);
        const r = Math.max(Math.abs(i - L.zeroI), Math.abs(j - L.zeroJ));
        const b = bands.findIndex((lo, k) => r >= lo && r < bands[k + 1]!);
        hit[b]![0] += off < 0.25 ? 1 : 0;
        hit[b]![1] += 1;
        if (off < 0.25) centred++;
      }
    }
    const frac = centred / Math.max(1, onScreen);
    const profile = hit.map((c, k) =>
      `r${bands[k]}+: ${c[1] ? ((c[0]! / c[1]!) * 100).toFixed(0) : '--'}%`).join(' ');
    console.log(`    lattice ${L.rows}x${L.cols}, ${onScreen} on screen, pitch ${pitchJ.toFixed(4)}/` +
      `${pitchI.toFixed(4)}, reference off-centre ${refOff.toFixed(3)}, ` +
      `${(frac * 100).toFixed(1)}% within a quarter cell (worst ${worst.toFixed(3)}) -- ${profile}`);

    assert.ok(onScreen > 100, `only ${onScreen} lattice cells project on screen`);
    // 2% is the same gate gpp's period test uses, and for the same reason: this
    // pitch IS that period, expressed in world units through the layout.
    assert.ok(Math.abs(pitchJ - TEST_CELL_PITCH) < 0.02 && Math.abs(pitchI - TEST_CELL_PITCH) < 0.02,
      `a lattice step is ${pitchJ.toFixed(4)} / ${pitchI.toFixed(4)} board cells, not one`);
    assert.ok(refOff < 0.2,
      `the reference cell sits ${refOff.toFixed(3)} cells from a true centre -- ` +
      `the phase is wrong (0.5 would be exactly on the boundaries)`);
    // Deliberately loose, because the profile above says what it is measuring.
    assert.ok(frac > 0.75,
      `only ${(frac * 100).toFixed(1)}% of the lattice lands near a true centre -- ${profile}`);
  });
});

test('decode.layout: the indirect args cover the lattice, in every rotation', async () => {
  await withDevice(async (device) => {
    // ── THE POSE IS CHOSEN FOR ITS WORKGROUP COUNTS, NOT ITS DIMENSIONS ──
    //
    // An args pair is a CEILING over the 8x8 workgroup, so what has to differ
    // between rows and cols is ceil(n/8) -- not n. The fixture used to be tilt
    // 20, whose lattice is 19x22: genuinely non-square, and ceil(19/8) ==
    // ceil(22/8) == 3, so a swapped pair is bit-identical to a correct one.
    // A mutation run caught that 2026-08-14, and it caught the ASSERTION below
    // too: `rows != cols` was written specifically as the decisiveness check and
    // is the wrong predicate for the quantity under test.
    //
    // Tilt 40 gives 23x37 -> (3, 5), where a swap is visible.
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 40, yawDeg: 15 };
    const L = await layoutOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
    assert.equal(L.valid, 1);
    assert.notEqual(Math.ceil(L.rows / 8), Math.ceil(L.cols / 8),
      `this lattice is ${L.rows}x${L.cols}, whose workgroup counts are equal -- ` +
      `it cannot tell a swapped args pair from a correct one`);

    // decode.build is gid.x = i = row, so args.x covers ROWS. Getting this the
    // wrong way round truncates the grid on one axis and runs empty threads on
    // the other, and every cell it DOES build is correct -- which is why it is
    // asserted here rather than left to a downstream count.
    assert.deepEqual([...L.buildArgs],
      [Math.ceil(L.rows / 8), Math.ceil(L.cols / 8), 1], 'buildArgs');

    for (let o = 0; o < 4; o++) {
      const swap = o === 1 || o === 3;
      const rr = swap ? L.cols : L.rows;
      const cc = swap ? L.rows : L.cols;
      assert.deepEqual([...L.tallyArgs.slice(o * 3, o * 3 + 3)],
        [Math.ceil(rr / 8), Math.ceil(cc / 8), 1], `tallyArgs[${o}]`);
    }
  });
});

test('decode.layout: a frame with no period dispatches nothing, and says why', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const busy = await layoutOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
    assert.equal(busy.valid, 1, 'the fixture frame produced no lattice to inherit');
    assert.ok(busy.rows > 4 && busy.cols > 4);

    // Every one of these buffers now holds a valid answer, which is exactly the
    // state §16 says a single run structurally cannot test against.
    const quiet = await layoutOn(device, new Float64Array(FRAME.w * FRAME.h), FRAME, FRAME_DIMS);
    assert.equal(quiet.valid, 0, 'the blank frame inherited the previous lattice');
    assert.equal(quiet.rows, 0, `rows ${quiet.rows} inherited`);
    assert.equal(quiet.cols, 0, `cols ${quiet.cols} inherited`);
    assert.equal(quiet.status & 256, 256, `layoutInvalid was not set (status ${quiet.status})`);
    // The propagation half: no workgroups, so every later decode pass runs no
    // threads at all rather than reading a stale lattice.
    assert.deepEqual([...quiet.buildArgs], [0, 0, 0], 'buildArgs did not zero');
    assert.deepEqual([...quiet.tallyArgs], new Array(12).fill(0), 'tallyArgs did not zero');
  });
});

test('decode.layout: the lattice bounds ARE the hull, converted', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const gray = renderPose(TEST_WORLD, pose, FRAME_DIMS, 4);
    const L = await layoutOn(device, gray, FRAME, FRAME_DIMS);
    const extent = await readF32(device, L.bufs.extent!, 12);

    // u = distance * xRow, v = distance * xCol -- one scalar, and it has NO SIGN
    // here. src/pose's projectedUVScale returns +/-distance because gnomonic() is
    // handed a raw eigenvector while projectedUVBounds flips it toward the floor;
    // §10 decided that sign once, at the point it was created, so the two frames
    // agree and the conversion is a bare multiply. A version that kept the flip
    // would mirror the lattice about the nadir -- which at a symmetric pose is
    // nearly invisible, so it is asserted here rather than left to decode.
    const cp = L.cellPitch;
    const expect = (lo: number, hi: number, phase: number) => [
      Math.floor((L.distance * lo - phase) / cp),
      Math.ceil((L.distance * hi - phase) / cp),
    ];
    const [kMinU, kMaxU] = expect(extent[5]!, extent[6]!, L.uPhase);
    const [kMinV, kMaxV] = expect(extent[7]!, extent[8]!, L.vPhase);
    assert.equal(L.kMinU, kMinU, 'kMinU is not floor((distance * hullXRowMin - uPhase) / cellPitch)');
    assert.equal(L.kMinV, kMinV, 'kMinV');
    assert.equal(L.cols, kMaxU - kMinU + 1, 'cols');
    assert.equal(L.rows, kMaxV - kMinV + 1, 'rows');
    assert.equal(L.status & 512, 0, 'gridOverflow fired on a lattice well inside one board period');
  });
});

test('decode.layout: a hull past one board period is CLAMPED and keeps decoding', async () => {
  await withDevice(async (device) => {
    // §12's correction to its own proposal, and the reason the word "by
    // construction" came out: the raw hull genuinely reaches 169 cells on an edge
    // at h=24 tilt 45-55, where the camera really does see 200+ cells and the
    // recovered period is right to ~1.4%. So the clamp FIRES on frames that
    // decode correctly and must not be a failure.
    //
    // Reaching that regime at 96x128 is not practical, so the board is shrunk
    // instead of the camera moved: a 12x12 torus makes an ordinary 19x22 lattice
    // exceed one period on both axes. It is the same arithmetic under test --
    // MAX_CELLS is torusR * torusC either way -- and it needs no grazing pose,
    // which would be testing the detector rather than this.
    const SMALL: Dims = { ...FRAME, torusR: 12, torusC: 12, maxCells: 12 * 12 };
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const gray = renderPose(TEST_WORLD, pose, FRAME_DIMS, 4);
    const big = await layoutOn(device, gray, FRAME, FRAME_DIMS);
    const L = await layoutOn(device, gray, SMALL, FRAME_DIMS);

    assert.ok(big.rows > 12 && big.cols > 12,
      `the unclamped lattice is ${big.rows}x${big.cols} -- it must exceed 12 on both axes ` +
      `or this fixture cannot tell the clamp from its absence`);

    // IT KEEPS DECODING. This is the assertion §12 exists to make.
    assert.equal(L.valid, 1, 'a clamped lattice was reported invalid');
    assert.equal(L.status & 256, 0, 'layoutInvalid was set by a clamp');
    assert.equal(L.status & 512, 512, 'gridOverflow was not reported');
    assert.deepEqual([...L.buildArgs], [Math.ceil(12 / 8), Math.ceil(12 / 8), 1],
      'the build dispatch was zeroed by a clamp');

    assert.equal(L.rows, 12, `rows ${L.rows}`);
    assert.equal(L.cols, 12, `cols ${L.cols}`);
    assert.ok(L.rows * L.cols <= SMALL.maxCells, 'MAX_CELLS = torusR * torusC is not a bound');

    // CENTRED on the raw hull. Any contiguous window works for the anchor, since
    // the arithmetic is mod R/C -- so this pins the choice that was made rather
    // than a necessity, and keep-from-min is untested (§12).
    assert.equal(L.kMinU, big.kMinU + Math.floor((big.cols - 12) / 2), 'the kept window is not centred in u');
    assert.equal(L.kMinV, big.kMinV + Math.floor((big.rows - 12) / 2), 'the kept window is not centred in v');
    // And the phase is untouched by clamping -- it is a property of the floor,
    // not of how much of it is being sampled.
    assert.ok(Math.abs(L.uPhase - big.uPhase) < 1e-6 && Math.abs(L.vPhase - big.vPhase) < 1e-6,
      'clamping moved the phase');
  });
});

// ── S9 decode.build: the lattice, sampled ─────────────────────────────────

/** layoutOn, plus the packed grid decode.build wrote. */
async function buildOn(device: GPUDevice, gray: Float64Array, dims: Dims, dd: typeof FRAME_DIMS) {
  const bufs = await run(device, (ctx) => {
    device.queue.writeBuffer(ctx.bufs.gray!, 0, Float32Array.from(gray));
    encodeGradient(ctx);
    encodeGrow(ctx, { rhoLow: 0.132, toleranceDeg: 9.5 });
    encodeCollect(ctx, { rhoHigh: 0, minRegionSize: 2 });
    encodeLsdFit(ctx, { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 });
    encodeLines(ctx, { minLengthPx: 3 });
    encodeVotes(ctx, { vFovRad: vFovRadOf(dd) });
    encodeFit(ctx);
    encodeGpp(ctx, { ...LAYOUT, vFovRad: vFovRadOf(dd) });
    encodeDecodeLayout(ctx, { ...LAYOUT, vFovRad: vFovRadOf(dd) });
    encodeDecodeBuild(ctx);
  }, dims);
  const lay = await readLayout(device, bufs.layout!);
  return {
    bufs, ...lay,
    packed: await readU32(device, bufs.packed!, Math.max(1, lay.rows * lay.cols)),
  };
}

test('decode.build: every sampled bit is the bit the true board has there', async () => {
  await withDevice(async (device) => {
    // THE claim of Act III's first pass, and ground truth can answer it cell by
    // cell: the lattice cell at (i, j) lands on some real point of the printed
    // floor, and the bit read out of the image must be the bit PRINTED there.
    // Nothing about this is an aggregate or a tolerance -- each cell is right or
    // it is wrong.
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const truth = truthFor(TEST_WORLD, pose);
    const B = await buildOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
    assert.equal(B.valid, 1, 'no lattice');

    const tanHalf = Math.tan(vFovRadOf(FRAME_DIMS) / 2);
    const aspect = FRAME.w / FRAME.h;
    const cam = truth.camPos;
    const d = { x: 0, y: 0, z: 0 };
    let valid = 0, agree = 0, boundary = 0;
    for (let i = 0; i < B.rows; i++) {
      for (let j = 0; j < B.cols; j++) {
        const p = B.packed[i * B.cols + j]!;
        if ((p & 1) === 0) continue;
        valid++;
        // Reproject this cell's OWN pixel through the true camera onto the true
        // floor -- the same route decode.build took, run backwards through
        // geometry it did not have.
        const uu = B.uPhase + (B.kMinU + j) * B.cellPitch;
        const vv = B.vPhase + (B.kMinV + i) * B.cellPitch;
        const q = {
          x: B.Drow.x * uu + B.Dcol.x * vv - B.normal.x * B.distance,
          y: B.Drow.y * uu + B.Dcol.y * vv - B.normal.y * B.distance,
          z: B.Drow.z * uu + B.Dcol.z * vv - B.normal.z * B.distance,
        };
        const px = ((-q.x / (q.z * tanHalf * aspect) + 1) / 2) * FRAME.w;
        const py = ((1 - -q.y / (q.z * tanHalf)) / 2) * FRAME.h;
        rayDirInto(d, (px / FRAME.w) * 2 - 1, 1 - (py / FRAME.h) * 2, truth.camQuat, tanHalf, aspect);
        if (d.y >= -1e-12) continue;
        const t = -cam.y / d.y;
        const hx = cam.x + t * d.x, hz = cam.z + t * d.z;
        // A sample within a tenth of a cell of a boundary is reading a pixel
        // that is genuinely between two cells -- the renderer supersamples, so
        // that pixel is a blend and its thresholded bit belongs to neither. Those
        // are counted separately rather than scored, because they measure the
        // upstream phase error and not this pass.
        const nearEdge = Math.min(
          Math.abs(hx / TEST_CELL_PITCH - Math.round(hx / TEST_CELL_PITCH)),
          Math.abs(hz / TEST_CELL_PITCH - Math.round(hz / TEST_CELL_PITCH)));
        if (nearEdge < 0.1) { boundary++; continue; }
        const col = ((Math.floor(hx / TEST_CELL_PITCH + C / 2) % C) + C) % C;
        const row = ((Math.floor(hz / TEST_CELL_PITCH + R / 2) % R) + R) % R;
        if (((p >> 1) & 1) === torus[row]![col]) agree++;
      }
    }
    const scored = valid - boundary;
    const frac = agree / Math.max(1, scored);
    console.log(`    ${valid} valid cells, ${boundary} on a cell boundary, ` +
      `${(frac * 100).toFixed(1)}% of the rest read the printed bit`);
    assert.ok(scored > 100, `only ${scored} scorable cells`);
    // A wrong threshold SENSE inverts every bit, so this lands near 0 rather
    // than merely dropping; a wrong pixel mapping decorrelates it to ~50%. There
    // is no tolerance regime in between that a real defect could hide in.
    assert.ok(frac > 0.97,
      `${(frac * 100).toFixed(1)}% agreement -- 50% is a decorrelated sample, ` +
      `0% is an inverted threshold`);

    // ── WHAT THIS FIXTURE CANNOT SEE, MEASURED RATHER THAN ASSUMED ──
    //
    // The PER-CELL GRAZING TEST. Deleting it changes nothing here, and the
    // reason is not that it does not matter -- it is that no cell of this
    // lattice reaches the band where it fires. Counted directly at three poses
    // at this frame size, including h=24 tilt 45 and tilt 55, which is the exact
    // regime §12 measured it in: rejectedByGrazingAlone = 0 at every one, and
    // behindCamera = 0 too.
    //
    // The cells it rejects are ones the HULL has already clipped at 96x128 --
    // the detector finds no lines that close to the horizon here, so the lattice
    // never reaches out to them. At 480x640 it does: scripts/hull-measure.ts
    // counts 9,811 hull cells across 24 grazing poses that this test rejects and
    // the screen-bounds test would have accepted. That is the evidence for the
    // guard; nothing in this suite is.
  });
});


// ── S10-S12 decode tally / argmax / correctness ───────────────────────────
//
// Ground truth knows the anchor: the board cell directly under the camera is a
// property of the pose that rendered the frame, so `truthFor` states it exactly
// and there is nothing to approximate.

const BOARD = boardDims(TEST_BOARD);

/** The whole pipeline through decode.correctness, plus the result block. */
async function decodeOn(device: GPUDevice, gray: Float64Array, dims: Dims, dd: typeof FRAME_DIMS) {
  const bufs = await run(device, (ctx) => {
    uploadBoard(ctx.device, ctx.bufs, buildBoard(TEST_BOARD, dims));
    device.queue.writeBuffer(ctx.bufs.gray!, 0, Float32Array.from(gray));
    encodeGradient(ctx);
    encodeGrow(ctx, { rhoLow: 0.132, toleranceDeg: 9.5 });
    encodeCollect(ctx, { rhoHigh: 0, minRegionSize: 2 });
    encodeLsdFit(ctx, { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 });
    encodeLines(ctx, { minLengthPx: 3 });
    encodeVotes(ctx, { vFovRad: vFovRadOf(dd) });
    encodeFit(ctx);
    encodeGpp(ctx, { ...LAYOUT, vFovRad: vFovRadOf(dd) });
    encodeDecodeLayout(ctx, { ...LAYOUT, vFovRad: vFovRadOf(dd) });
    encodeDecodeBuild(ctx);
    encodeDecodeTally(ctx, { order: ORDER });
  }, dims);
  const r = await readU32(device, bufs.result!, 8);
  const lay = await readLayout(device, bufs.layout!);
  return {
    bufs, ...lay,
    found: r[0]!, orientation: r[1]!, anchorRow: r[2]!, anchorCol: r[3]!,
    votes: r[4]!, totalWindows: r[5]!, correct: r[6]!, wrong: r[7]!,
    consistency: r[6]! + r[7]! > 0 ? r[6]! / (r[6]! + r[7]!) : 0,
    status: (await readU32(device, bufs.status!, 1))[0]!,
  };
}

test('decode: the winning anchor is the true one, and the grid agrees with the board', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const truth = truthFor(TEST_WORLD, pose);
    const D = await decodeOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);

    console.log(`    ${D.rows}x${D.cols} lattice, orientation ${D.orientation}, ` +
      `anchor (${D.anchorRow}, ${D.anchorCol}), ${D.votes}/${D.totalWindows} windows voted for it, ` +
      `consistency ${D.consistency.toFixed(4)}`);

    assert.equal(D.found, 1, `no anchor found (status ${D.status})`);
    assert.ok(D.totalWindows > 50, `only ${D.totalWindows} complete windows`);

    // ── THE ANCHOR IS THE TRUE ONE ──
    //
    // The anchor names which board cell the lattice's (0, 0) sits over, so it is
    // exactly checkable: it is the cell under the camera, shifted by where the
    // reference cell sits in the lattice. A near miss is not a thing here -- the
    // De Bruijn window is unique, so a wrong anchor is wrong by a whole cell or
    // more and its vote count collapses.
    const [rzI, rzJ] = rotatedZeroIndex(D.rows, D.cols, D.zeroI, D.zeroJ, D.orientation);
    const refRow = (((D.anchorRow + rzI) % R) + R) % R;
    const refCol = (((D.anchorCol + rzJ) % C) + C) % C;
    // The reference cell is the lattice cell nearest the nadir, so the board cell
    // it names is the one under the camera. truthFor states that outright.
    const dRow = Math.min(Math.abs(refRow - truth.anchorRow), R - Math.abs(refRow - truth.anchorRow));
    const dCol = Math.min(Math.abs(refCol - truth.anchorCol), C - Math.abs(refCol - truth.anchorCol));
    assert.ok(dRow <= 1 && dCol <= 1,
      `the reference cell decoded to board (${refRow}, ${refCol}), truth says the camera is over ` +
      `(${truth.anchorRow}, ${truth.anchorCol})`);

    // ── AND THE WHOLE GRID AGREES ──
    //
    // consistency is the fraction of resolvable cells whose sampled bit matches
    // what the pattern says is printed there. It is NOT a check on the pose (see
    // §14 -- a completely wrong camera quaternion still reads 1.000), but it is
    // a decisive check on the anchor: a wrong anchor decorrelates every cell,
    // landing near 0.5 rather than near 1.
    assert.ok(D.consistency > 0.97, `consistency ${D.consistency.toFixed(4)}`);
    assert.equal(D.correct + D.wrong > 0, true, 'the correctness pass counted nothing');
    // ── AND THE WINNER IS DECISIVE ──
    //
    // `totalWindows` counts complete windows across ALL FOUR orientations, so a
    // perfect decode votes with about a QUARTER of it -- the three wrong
    // rotations produce keys that are not in the table and cast nothing. Measured
    // here: 116 votes against 464 total, which is 4 x 116 exactly. Asserting
    // against totalWindows directly would be asserting the wrong denominator and
    // would fail on a flawless frame, which is how this gate was first written.
    const perOrientation = D.totalWindows / 4;
    assert.ok(D.votes > perOrientation * 0.9,
      `${D.votes} votes against ${perOrientation} complete windows per orientation`);
  });
});

test('decode: an undecodable frame reports no anchor, and inherits nothing', async () => {
  await withDevice(async (device) => {
    const pose = { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 };
    const busy = await decodeOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
    assert.equal(busy.found, 1, 'the fixture frame decoded nothing to inherit');

    const quiet = await decodeOn(device, new Float64Array(FRAME.w * FRAME.h), FRAME, FRAME_DIMS);
    // `found = 0` is an ORDINARY outcome, not an error: a frame with no board in
    // it. What would be a defect is reporting the previous frame's answer, and
    // every one of these buffers is holding exactly that.
    assert.equal(quiet.found, 0, 'an anchor was found in a blank frame');
    assert.equal(quiet.votes, 0, `votes ${quiet.votes} inherited`);
    assert.equal(quiet.totalWindows, 0, `totalWindows ${quiet.totalWindows} inherited`);
    assert.equal(quiet.correct, 0, `correct ${quiet.correct} inherited`);
    assert.equal(quiet.wrong, 0, `wrong ${quiet.wrong} inherited`);
    assert.equal(quiet.anchorRow, 0, 'anchorRow inherited');
    assert.equal(quiet.anchorCol, 0, 'anchorCol inherited');
  });
});

test('decode: the hash table round-trips every window the board has', () => {
  // Every entry findable by its own probe, which is what an insertion bug
  // breaks: a wrong probe direction or an overwritten slot loses windows
  // silently, and a lost window is just a window that casts no vote.
  const board = buildBoard(TEST_BOARD, BOARD);
  const size = BOARD.hashSlots;
  let checked = 0;
  for (const [key, value] of debruijnLookup) {
    let slot = hashU32(key) % size;
    let hit = -1;
    for (let probe = 0; probe < size; probe++) {
      if (board.hashValues[slot] === 0xffffffff) break;
      if (board.hashKeys[slot] === key) { hit = slot; break; }
      slot = (slot + 1) % size;
    }
    assert.notEqual(hit, -1, `key ${key} is not findable by its own probe`);
    assert.equal(board.hashValues[hit], value, `key ${key} probes to the wrong value`);
    checked++;
  }
  assert.equal(checked, debruijnLookup.size);
  assert.ok(checked > 10000, `only ${checked} entries -- is this the right board?`);

  // WHAT THIS DOES NOT TEST, stated rather than implied: that board.ts's hash
  // and DECODE_TALLY_WGSL's hash agree. It cannot -- it uses board.ts's, so it
  // is checking a function against itself. The only check on that agreement is
  // the end-to-end decode above, and it is a total one: Math.imul and a WGSL u32
  // multiply either wrap identically or every single lookup misses, so a
  // divergence is `found = 0`, never a degraded answer.
});

// ── S13 finish: the whole pipeline, image in and pose out ─────────────────

/** Everything, in one encoder, and the 128 bytes that come back. */
async function poseOn(
  device: GPUDevice, gray: Float64Array, dims: Dims, dd: typeof FRAME_DIMS,
  opts: { alias?: boolean } = {},
) {
  const alias = opts.alias ?? false;
  const bufs = await run(device, (ctx) => {
    uploadBoard(ctx.device, ctx.bufs, buildBoard(TEST_BOARD, dims));
    device.queue.writeBuffer(ctx.bufs.gray!, 0, Float32Array.from(gray));
    encodeGradient(ctx);
    encodeGrow(ctx, { rhoLow: 0.132, toleranceDeg: 9.5 });
    encodeCollect(ctx, { rhoHigh: 0, minRegionSize: 2 });
    encodeLsdFit(ctx, { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 });
    encodeLines(ctx, { minLengthPx: 3 });
    encodeVotes(ctx, { vFovRad: vFovRadOf(dd) });
    encodeFit(ctx);
    encodeGpp(ctx, { ...LAYOUT, vFovRad: vFovRadOf(dd) });
    encodeDecodeLayout(ctx, { ...LAYOUT, vFovRad: vFovRadOf(dd) });
    encodeDecodeBuild(ctx);
    encodeDecodeTally(ctx, { order: ORDER });
    encodeFinish(ctx);
  }, dims, alias);
  const raw = await readU32(device, bufs.pose!, 32);
  const block = new Uint32Array(32);
  block.set(raw);
  return { bufs, plan: buffers(device, dims, alias).plan, ...decodePose(block.buffer) };
}

test('THE PIPELINE: an image goes in and the pose that made it comes out', async () => {
  await withDevice(async (device) => {
    // ── THE ORIENTATION IS WHAT THIS TEST EXISTS FOR ──
    //
    // Four poses, chosen so the winning orientation is not always 0. §14's trap
    // is a missing sign flip in the board-frame-to-world rotation, which is
    // SILENTLY CORRECT at orientation 0 -- most captures -- and wrong at 1, 2 and
    // 3 while still reporting perfect bit consistency. A fixture that only ever
    // won at orientation 0 would pass with the flip deleted, which is the exact
    // shape six earlier mutations in this rewrite hid behind.
    const poses = [
      { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 15 },
      { height: 10, overRow: 40.5, overCol: 40.5, tiltDeg: 20, yawDeg: 105 },
      { height: 10, overRow: 60.5, overCol: 20.5, tiltDeg: 20, yawDeg: 195 },
      { height: 16, overRow: 20.5, overCol: 60.5, tiltDeg: 30, yawDeg: 285 },
      // ── AN ELONGATED LATTICE, AND THAT IS THE WHOLE REASON FOR IT ──
      //
      // Tilt 40 gives a 23x37 lattice. The other four are near-square, and a
      // near-square lattice cannot see a swapped correctness dispatch: the args
      // are a CEILING over the 8x8 workgroup, so 19x22 rounds to 24x24 threads
      // and covers the grid whichever way round the pair goes. At 23x37 the
      // rounded extents are 24 and 40, so a swap truncates 37 rows to 24 and the
      // equality check below fires. Added after a mutation run 2026-08-14 found
      // the swap surviving even the exact cell count.
      { height: 10, overRow: 40.1, overCol: 40.6, tiltDeg: 40, yawDeg: 15 },
    ];
    const seen = new Set<number>();
    // The consistency check above is only decisive on a frame where some cell
    // DISAGREES -- with wrong = 0 the mutated denominator equals the correct one.
    let sawWrong = false;
    for (const pose of poses) {
      const truth = truthFor(TEST_WORLD, pose);
      const P = await poseOn(device, renderPose(TEST_WORLD, pose, FRAME_DIMS, 4), FRAME, FRAME_DIMS);
      const dx = P.position.x - truth.camPos.x;
      const dy = P.position.y - truth.camPos.y;
      const dz = P.position.z - truth.camPos.z;
      const err = Math.hypot(dx, dy, dz) / TEST_CELL_PITCH;
      // A quaternion and its negation are the same rotation, so the comparison
      // is on |dot|, which is cos(half the angle between them).
      const qd = Math.abs(
        P.quaternion.x * truth.camQuat.x + P.quaternion.y * truth.camQuat.y +
        P.quaternion.z * truth.camQuat.z + P.quaternion.w * truth.camQuat.w);
      const angleDeg = 2 * Math.acos(Math.min(1, qd)) * (180 / Math.PI);
      seen.add(P.orientation);
      console.log(`    tilt${pose.tiltDeg} yaw${pose.yawDeg} h${pose.height}: ` +
        `orientation ${P.orientation}, position off by ${err.toFixed(3)} cells, ` +
        `quaternion off by ${angleDeg.toFixed(2)} deg, consistency ${P.consistency.toFixed(4)}, ` +
        `period ${P.period.toFixed(4)} vs ${truth.period.toFixed(4)}, status ${P.status}`);

      assert.ok(P.ok, `no pose at tilt${pose.tiltDeg} yaw${pose.yawDeg} (status ${P.status})`);
      // 0.9 is set from BOTH sides rather than by eye. Measured 2026-08-14
      // across these four poses: 1.0000, 1.0000, 1.0000 and 0.9653 -- the last
      // is h=16, where the lattice is finer and a 1% period error puts more
      // cells near a boundary. A WRONG anchor decorrelates every cell and lands
      // at the chance floor of ~0.5, so the gap this has to straddle is 0.5 to
      // 0.965 and 0.9 sits in it with margin on both sides. A first version at
      // 0.97 failed the h=16 pose on a perfectly good decode.
      assert.ok(P.consistency > 0.9, `consistency ${P.consistency.toFixed(4)} (0.5 is chance)`);
      // EXACT, not one-sided. `consistency` and the two counters come out of the
      // same 128 bytes but are computed in different places, so this is a real
      // check on finish's arithmetic -- and a one-sided gate cannot make it:
      // dividing by `correct` instead of `correct + wrong` RAISES consistency to
      // 1.0, which sails past any lower bound. Found by mutation 2026-08-14.
      const denom = P.correct + P.wrong;
      assert.ok(denom > 0, 'the correctness pass counted nothing');
      // ── THE CORRECTNESS PASS COVERED THE WHOLE GRID ──
      //
      // Every valid cell appears exactly once in the rotated grid, so this is an
      // EQUALITY, not a bound. It is here because the obvious check -- that
      // `correctnessArgs` swaps rows and cols at orientations 1 and 3 -- is
      // invisible whenever ceil(rows/8) == ceil(cols/8), which is most lattices.
      // A truncated dispatch does not corrupt anything; it silently counts FEWER
      // cells, and consistency stays perfectly plausible. Found by mutation
      // 2026-08-14, after the same ceiling coincidence had already hidden the
      // buildArgs and tallyArgs swaps.
      const packed = await readU32(device, P.bufs.packed!, P.gridRows * P.gridCols);
      let validCells = 0;
      for (let k = 0; k < P.gridRows * P.gridCols; k++) if (packed[k]! & 1) validCells++;
      assert.equal(denom, validCells,
        `the correctness pass counted ${denom} of ${validCells} valid cells at ` +
        `orientation ${P.orientation} on a ${P.gridRows}x${P.gridCols} lattice`);
      assert.ok(Math.abs(P.consistency - P.correct / denom) < 1e-6,
        `consistency ${P.consistency} is not correct/(correct+wrong) = ${P.correct}/${denom}`);
      if (P.wrong > 0) sawWrong = true;
      // ~0.05 cells is the src/pose baseline's median sub-cell error at 480x640
      // (§19). This is a 96x128 frame, so the gate is looser -- what it is
      // actually excluding is a WHOLE-CELL error, which is what a wrong anchor
      // or a missing sign flip produces.
      assert.ok(err < 0.5, `position is off by ${err.toFixed(3)} cells`);
      assert.ok(angleDeg < 3, `the camera quaternion is off by ${angleDeg.toFixed(2)} degrees`);
    }
    // THE FIXTURE HAS TO BE ABLE TO ASK THE QUESTION, asserted rather than
    // assumed: if every pose won at orientation 0, §14's sign flips would be
    // untested by all four.
    assert.ok(seen.size >= 3,
      `only orientations {${[...seen].join(', ')}} were exercised -- the sign flips ` +
      `in the board-to-world rotation are unreachable at orientation 0`);
    assert.ok(sawWrong,
      'every pose decoded perfectly, so the consistency denominator is untested -- ' +
      'correct/(correct+wrong) and correct/correct agree when wrong is 0');
  });
});

// ── decode.build's TWO GUARDS, on a hand-built layout ─────────────────────
//
// Both survived the 2026-08-14 mutation run, and the reason is not that they do
// not matter -- it is that no lattice a RENDERED frame produces at 96x128 ever
// reaches the band where they fire. Counted directly at h=24 tilt 45 and tilt
// 55: zero cells rejected by grazing, zero behind the camera, because the line
// hull has already clipped that band at this frame size.
//
// So the fixture is synthesized instead of rendered. `layout` is a plain buffer
// and decode.build is the only pass that reads it, so writing one by hand and
// dispatching that single pass is both legitimate and exact -- and it is the
// only thing in this suite that exercises either guard.
//
// THE CAMERA IS HORIZONTAL, which is what puts floor cells on both sides of it:
// normal = +y, so the floor is at y = -distance; Drow = +x and Dcol = -z, so a
// cell at lattice coordinate (u, v) sits at p = (u, -distance, -v). Then v > 0
// is in front of the camera (which looks down -z) and v <= 0 is BEHIND it.
//
// Three regimes, all reachable and all hand-computable at distance 10:
//   v <= 0          behind the camera -- and it projects ON SCREEN, mirrored
//   0 < v < ~13     in front, but below the bottom edge of the image
//   ~13 < v < ~99   valid
//   v > ~100        grazing: distance/|p| falls under minGrazingCos
const GUARD_DIST = 10;

/** Writes a `Layout` block by hand. Offsets are readLayout's, in bytes. */
function writeLayoutBlock(device: GPUDevice, buf: GPUBuffer, L: {
  Drow: number[]; Dcol: number[]; normal: number[];
  distance: number; tanHalf: number; aspect: number; minGrazingCos: number;
  uPhase: number; vPhase: number; cellPitch: number; binThreshold: number;
  rows: number; cols: number; imageW: number; imageH: number;
  kMinU: number; kMinV: number;
}): void {
  const b = new ArrayBuffer(128);
  const f = new Float32Array(b);
  const u = new Uint32Array(b);
  const i = new Int32Array(b);
  f.set(L.Drow, 0); f.set(L.Dcol, 4); f.set(L.normal, 8);
  f[12] = L.distance; f[13] = L.tanHalf; f[14] = L.aspect; f[15] = L.minGrazingCos;
  f[16] = L.uPhase; f[17] = L.vPhase; f[18] = L.cellPitch; f[19] = L.binThreshold;
  u[20] = L.rows; u[21] = L.cols; u[22] = L.imageW; u[23] = L.imageH;
  i[24] = L.kMinU; i[25] = L.kMinV;
  u[26] = 0; u[27] = 0; u[28] = 1; // zeroI, zeroJ, valid
  device.queue.writeBuffer(buf, 0, b);
}

test('decode.build: a cell BEHIND the camera is rejected, though it lands on screen', async () => {
  await withDevice(async (device) => {
    const tanHalf = Math.tan(vFovRadOf(FRAME_DIMS) / 2);
    const aspect = FRAME.w / FRAME.h;
    // v runs from -23.5 to +115.5, which is what puts cells in all three bands:
    // behind the camera (v <= 0), valid (~13 < v < ~99), and past the grazing
    // cutoff (|p| > distance / minGrazingCos = 100). A 64-row lattice reached
    // only v = 39 and the grazing band was empty -- the reachability assertions
    // below caught that, which is the job they are there to do.
    const rows = 140, cols = 8, kMinV = -24, kMinU = -4;
    const bufs = await run(device, (ctx) => {
      writeLayoutBlock(device, ctx.bufs.layout!, {
        Drow: [1, 0, 0, 0], Dcol: [0, 0, -1, 0], normal: [0, 1, 0, 0],
        distance: GUARD_DIST, tanHalf, aspect, minGrazingCos: 0.1,
        uPhase: 0.5, vPhase: 0.5, cellPitch: 1, binThreshold: 128,
        rows, cols, imageW: FRAME.w, imageH: FRAME.h, kMinU, kMinV,
      });
      // decode.build dispatches indirectly, and decode.layout is not running.
      device.queue.writeBuffer(ctx.bufs.buildArgs!, 0,
        new Uint32Array([Math.ceil(rows / 8), Math.ceil(cols / 8), 1]));
      encodeDecodeBuild(ctx);
    }, FRAME);
    const packed = await readU32(device, bufs.packed!, rows * cols);

    // Re-derive each cell's fate on the host, from the same geometry.
    let behindAndOnScreen = 0, behindAccepted = 0;
    let grazingOnly = 0, grazingAccepted = 0, valid = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = 0.5 + (kMinV + i);
        const uu = 0.5 + (kMinU + j);
        const p = { x: uu, y: -GUARD_DIST, z: -v };
        const len = Math.hypot(p.x, p.y, p.z);
        const grazingOk = GUARD_DIST / len > 0.1;
        const behind = p.z >= 0;
        const px = ((-p.x / (p.z * tanHalf * aspect) + 1) / 2) * FRAME.w;
        const py = ((1 - -p.y / (p.z * tanHalf)) / 2) * FRAME.h;
        const onScreen = px >= 0 && px < FRAME.w && py >= 0 && py < FRAME.h;
        const got = (packed[i * cols + j]! & 1) === 1;
        if (behind && onScreen) { behindAndOnScreen++; if (got) behindAccepted++; }
        else if (!grazingOk && !behind && onScreen) { grazingOnly++; if (got) grazingAccepted++; }
        else if (grazingOk && !behind && onScreen) { valid++; }
      }
    }
    console.log(`    ${valid} valid, ${behindAndOnScreen} behind-but-on-screen, ` +
      `${grazingOnly} rejected by grazing alone`);

    // THE FIXTURE HAS TO REACH THE BAND, asserted before the behaviour -- this is
    // the whole reason the test exists, since a rendered frame does not.
    assert.ok(behindAndOnScreen > 0,
      'no cell of this lattice is both behind the camera and on screen -- the guard is untested');
    assert.ok(grazingOnly > 0,
      'no cell of this lattice is rejected by grazing alone -- the guard is untested');
    assert.ok(valid > 0, 'the lattice produced no valid cells at all');

    // A cell behind the camera projects through a NEGATED depth to a mirrored
    // image point that passes every bounds test. Without the guard it samples a
    // real pixel and votes, silently.
    assert.equal(behindAccepted, 0,
      `${behindAccepted} of ${behindAndOnScreen} cells behind the camera were accepted`);
    assert.equal(grazingAccepted, 0,
      `${grazingAccepted} of ${grazingOnly} cells past the grazing cutoff were accepted`);
  });
});

// ── §18 buffer pooling: the free self-check ───────────────────────────────
//
// With `alias: true`, buffers whose live ranges do not overlap share one
// allocation. It is not a second code path -- the same interval colouring, over
// a real liveness table instead of a degenerate one -- so the only thing that
// can go wrong is the LIVENESS being wrong, or a clear landing on a slot that
// still belongs to somebody else.
//
// Both failures are invisible by inspection and neither throws. What makes them
// findable is that the pooled and unpooled runs must agree EXACTLY: same
// declaration, same shaders, same input, and nothing in the pipeline knows which
// mode it is in. So any disagreement at all is a pooling or clear-schedule bug.
//
// The failure mode is also better than it sounds. A missed clear under pooling
// hands a stage ANOTHER ARRAY's data -- labels where counts should be -- rather
// than last frame's plausible-looking values. Weird is easier to notice than
// subtly wrong.

test('§18: the POOLED buffer set produces a bit-identical pose', async () => {
  await withDevice(async (device) => {
    // Two poses, because a single frame exercises one lattice size and one
    // region count, and the live ranges that matter are the full-image arrays
    // shared across grow/collect.
    for (const pose of [
      { height: 10, overRow: 40.1, overCol: 40.6, tiltDeg: 20, yawDeg: 15 },
      { height: 16, overRow: 20.5, overCol: 60.5, tiltDeg: 30, yawDeg: 285 },
    ]) {
      const gray = renderPose(TEST_WORLD, pose, FRAME_DIMS, 4);
      const flat = await poseOn(device, gray, FRAME, FRAME_DIMS);
      const pooled = await poseOn(device, gray, FRAME, FRAME_DIMS, { alias: true });

      const slots = pooled.plan.slots.length;
      assert.ok(slots < flat.plan.slots.length,
        `pooling shares nothing (${slots} slots either way) -- this test cannot see a pooling bug`);

      // Every field of the 128 bytes, not just the pose: the diagnostics are
      // where a stage that read the wrong slot shows up FIRST, because a region
      // count or a line count moves long before the pose does.
      for (const k of [
        'ok', 'status', 'consistency', 'orientation', 'boardRow', 'boardCol',
        'votes', 'totalWindows', 'correct', 'wrong', 'regionCount', 'memberCount',
        'lineCount', 'gridRows', 'gridCols', 'growRounds', 'period', 'height',
      ] as const) {
        assert.deepEqual(pooled[k], flat[k],
          `${k} differs under pooling at tilt${pose.tiltDeg} yaw${pose.yawDeg}: ` +
          `${JSON.stringify(flat[k])} unpooled, ${JSON.stringify(pooled[k])} pooled`);
      }
      for (const axis of ['x', 'y', 'z'] as const) {
        assert.equal(pooled.position[axis], flat.position[axis], `position.${axis} differs under pooling`);
      }
      for (const axis of ['x', 'y', 'z', 'w'] as const) {
        assert.equal(pooled.quaternion[axis], flat.quaternion[axis], `quaternion.${axis} differs under pooling`);
      }
    }
  });
});
