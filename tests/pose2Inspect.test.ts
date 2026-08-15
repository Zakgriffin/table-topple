import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { withDevice } from './helpers/gpu.ts';
import { boardDims } from '../src/pose2/board.ts';
import { createPose2Context, destroyPose2Context, runPose2 } from '../src/pose2/run.ts';
import type { Dims } from '../src/pose2/pipeline.ts';
import { renderPose, vFovRadOf } from '../src/pose2/sim.ts';
import { GRID_STEP } from '../src/sphereLab/constants.ts';
import { ORDER } from '../src/sphereLab/floorPattern.ts';

// ── The opt-in intermediate readback ──────────────────────────────────────
//
// §22's display path: the app wants buffers the pose path has no use for, and
// the whole design question was whether it can have them WITHOUT a second
// submit, a second fence, or a hole in the pooling. This file is the check that
// it can.
//
// What is tested here is `run.ts`'s half -- the staging layout and the frame.
// The PLAN's half (that a declared buffer holds its slot, and that without the
// declaration it demonstrably would not) is in pose2Buffers.test.ts, which
// needs no device. The two compose: that file proves the mechanism does
// something, this one proves the bytes that come back are right.

const FRAME: Dims = {
  w: 96, h: 128, maxRegions: 4096, maxLines: 4096, ...boardDims(),
};
const FRAME_DIMS = { w: FRAME.w, h: FRAME.h, horizFovDeg: 60 };
const POSE = { height: 10, overRow: 40.1, overCol: 40.6, tiltDeg: 20, yawDeg: 15 };

// `lines` and `votes` are the two that matter: both are things an overlay draws
// (composite lines, the vote circles) and both are measurably clobbered under
// `alias` without the declaration -- see pose2Buffers.test.ts. `fx` and `layout`
// come along as the two other shapes, a full-image array and a 128-byte struct.
// The region CSR joins them: three buffers that are one fact, and the only
// display request that is megabytes rather than kilobytes. `fy` comes along
// because the CSR's strongest available property is a CROSS-BUFFER one -- every
// member pixel is above grow's edge floor, which needs both components.
const INSPECT = [
  'fx', 'fy', 'lines', 'votes', 'layout', 'members', 'regionOffsets', 'regionSizes', 'counts',
] as const;

const SETTINGS = {
  grow: { rhoLow: 0.132, toleranceDeg: 9.5 },
  collect: { rhoHigh: 0, minRegionSize: 2 },
  lsdFit: { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 },
  lines: { minLengthPx: 3 },
  votes: { vFovRad: vFovRadOf(FRAME_DIMS) },
  gpp: { vFovRad: vFovRadOf(FRAME_DIMS), cellPitch: GRID_STEP, minGrazingCos: 0.15 },
  layout: { vFovRad: vFovRadOf(FRAME_DIMS), cellPitch: GRID_STEP, minGrazingCos: 0.15 },
  order: ORDER,
};

async function frameWith(
  device: GPUDevice, gray: Float64Array,
  opts: { alias: boolean; inspect: readonly string[] },
) {
  const ctx = createPose2Context(device, FRAME, { alias: opts.alias, inspect: opts.inspect });
  try {
    return await runPose2(ctx, Float32Array.from(gray), SETTINGS, opts.inspect);
  } finally {
    destroyPose2Context(ctx);
  }
}

test('inspected bytes are identical with pooling on and off', async () => {
  await withDevice(async (device) => {
    const gray = renderPose(POSE, FRAME_DIMS, 4);

    // `alias: false` is the reference: every buffer holds its own slot, so what
    // comes back is unambiguously that buffer's own bytes. `alias: true` is the
    // one that can go wrong, and the declaration is the only thing stopping it.
    const off = await frameWith(device, gray, { alias: false, inspect: INSPECT });
    const on = await frameWith(device, gray, { alias: true, inspect: INSPECT });

    // The frame has to have actually produced something, or every comparison
    // below is between two empty buffers and passes for the wrong reason.
    assert.equal(off.pose.ok, true, 'the fixture pose did not decode -- nothing here is meaningful');
    assert.ok(off.pose.lineCount > 40, `only ${off.pose.lineCount} lines`);

    assert.deepEqual(on.pose, off.pose, 'pooling changed the pose itself');

    // ── ONE BUFFER IS COMPARED ONLY OVER ITS DEFINED PREFIX ──────────────
    //
    // `members` is sized for the whole image, because regions are a disjoint
    // partition and the worst case is every pixel; a real frame fills the first
    // `counts.y` of it and NOTHING WRITES THE TAIL. So the tail is whatever the
    // slot last held -- under `alias: false` the previous frame's, under
    // `alias: true` its partner's (`next`/`labelCounts`) -- and demanding the two
    // agree there is demanding that two different pieces of garbage match.
    //
    // MEASURED, not assumed, because "compare less until it passes" is how a real
    // aliasing bug would be hidden: at this fixture 314 regions hold 2,137
    // members, `regionOffsets`/`regionSizes` are identical over their FULL
    // length, and all 794 differing `members` elements sit at index 2,144 or
    // above. Not one is inside the data.
    const memberCount = new Uint32Array(off.inspected['counts']!)[1]!;
    assert.ok(memberCount > 500, `only ${memberCount} members -- the prefix below is nearly empty`);
    const defined: Record<string, number> = { members: memberCount * 4 };
    for (const name of INSPECT) {
      const end = defined[name];
      const a = new Uint8Array(off.inspected[name]!);
      const b = new Uint8Array(on.inspected[name]!);
      assert.equal(a.length, b.length, `${name}: different sizes`);
      if (end !== undefined) assert.ok(end < a.length, `${name}: nothing is being skipped`);
      const at = a.subarray(0, end).findIndex((v, i) => v !== b[i]);
      assert.equal(at, -1, `${name}: first differing byte at ${at}`);
    }
  });
});

test('inspected bytes are the BUFFER\'s, not some other stage\'s', async () => {
  await withDevice(async (device) => {
    const gray = renderPose(POSE, FRAME_DIMS, 4);
    const got = await frameWith(device, gray, { alias: true, inspect: INSPECT });

    // Byte-identical is only worth something if the bytes mean what the name
    // says. Each of these is a property the ALIASING PARTNER does not have, so a
    // slot returning the wrong occupant fails them:
    //
    //   lines  -- endpoint pairs in pixel space, inside the frame. colSamples
    //             (its partner) is (value, weight, crossMin, crossMax) in
    //             gnomonic units, which is not bounded by the image at all.
    //   votes  -- unit normals. rowSamples (its partner) is not normalized.
    const lines = new Float32Array(got.inspected['lines']!);
    const n = Math.min(got.pose.lineCount, FRAME.maxLines);
    assert.ok(n > 40, `only ${n} lines`);
    for (let i = 0; i < n; i++) {
      const [x1, y1, x2, y2] = [...lines.subarray(i * 4, i * 4 + 4)];
      for (const [v, lim] of [[x1!, FRAME.w], [x2!, FRAME.w], [y1!, FRAME.h], [y2!, FRAME.h]] as const) {
        assert.ok(v >= -1 && v <= lim + 1, `line ${i} endpoint ${v} is outside a ${FRAME.w}x${FRAME.h} frame`);
      }
      assert.ok(Math.hypot(x2! - x1!, y2! - y1!) >= 3, `line ${i} is shorter than minLengthPx`);
    }

    const votes = new Float32Array(got.inspected['votes']!);
    for (let i = 0; i < n; i++) {
      const [nx, ny, nz, w] = [...votes.subarray(i * 4, i * 4 + 4)];
      if (w! <= 0) continue; // a degenerate line votes with weight 0
      assert.ok(Math.abs(Math.hypot(nx!, ny!, nz!) - 1) < 1e-5, `vote ${i} normal is not unit`);
    }

    // `layout` is 128 bytes and its `distance` IS the camera height, which the
    // pose block reports independently -- so this pins the two against each
    // other, and would fail if the copy landed at the wrong staging offset.
    const layout = new Float32Array(got.inspected['layout']!);
    assert.equal(layout.length, 32);
    assert.ok(Math.abs(layout[12]! - got.pose.height) < 1e-3,
      `layout.distance ${layout[12]} disagrees with pose.height ${got.pose.height}`);
  });
});

// ── The CSR, which is what the two per-pixel rasters draw ────────────────
//
// The rasters walk `members[offsets[r] .. +sizes[r]]` and paint a pixel each, so
// what they need from the readback is not "the bytes match" but "these three
// buffers are a consistent slicing of pixel indices". Every assertion below is a
// property the CSR has and an arbitrary other buffer's bytes do not, which is
// what makes this a check on the staging layout and not just on collect.
//
// The last one is the strongest and is CROSS-BUFFER: a member pixel is by
// construction above grow's edge floor (GROW_SEED_WGSL labels only
// `fx^2+fy^2 > rhoLowSq`), so `members` and `fx`/`fy` have to be the same
// frame's, correctly offset, for it to hold.
test('the region CSR slices members into disjoint, in-frame, above-floor runs', async () => {
  await withDevice(async (device) => {
    const gray = renderPose(POSE, FRAME_DIMS, 4);
    const got = await frameWith(device, gray, { alias: true, inspect: INSPECT });

    const counts = new Uint32Array(got.inspected['counts']!);
    const members = new Uint32Array(got.inspected['members']!);
    const offsets = new Uint32Array(got.inspected['regionOffsets']!);
    const sizes = new Uint32Array(got.inspected['regionSizes']!);
    const [regionCount, memberCount] = [counts[0]!, counts[1]!];

    // Not vacuous: a frame with no regions passes every loop below trivially.
    assert.ok(regionCount > 40, `only ${regionCount} regions -- nothing below is meaningful`);
    // `counts.x` is the UNCLAMPED label count, deliberately (finish compares it
    // against maxRegions to report regionOverflow). This fixture must not
    // overflow, or the per-region arrays are truncated and the slicing below
    // would be a statement about a different set of regions than counts names.
    assert.ok(regionCount <= FRAME.maxRegions, `${regionCount} regions overflowed maxRegions`);
    assert.equal(members.length, FRAME.w * FRAME.h);
    assert.equal(offsets.length, FRAME.maxRegions);

    const n = FRAME.w * FRAME.h;
    const fx = new Float32Array(got.inspected['fx']!);
    const fy = new Float32Array(got.inspected['fy']!);
    const rhoLow = SETTINGS.grow.rhoLow;
    // -1 = not a member of anything, which is most of the image.
    const owner = new Int32Array(n).fill(-1);
    let total = 0;
    for (let r = 0; r < regionCount; r++) {
      const off = offsets[r]!, size = sizes[r]!;
      // Contiguous and in scan order: `offsets` is the exclusive prefix sum of
      // the kept sizes, so a region starts exactly where the previous one ended.
      assert.equal(off, total, `region ${r} starts at ${off}, not ${total}`);
      assert.ok(size >= SETTINGS.collect.minRegionSize, `region ${r} has ${size} members`);
      total += size;
      assert.ok(off + size <= members.length, `region ${r} runs past members`);
      for (let m = off; m < off + size; m++) {
        const p = members[m]!;
        assert.ok(p < n, `region ${r} member ${p} is not a pixel of a ${FRAME.w}x${FRAME.h} frame`);
        // DISJOINT: regions are a partition, which is exactly what lets the app
        // invert this into a hover-to-region map with a single pass.
        assert.equal(owner[p], -1, `pixel ${p} is in both region ${owner[p]} and ${r}`);
        owner[p] = r;
        assert.ok(fx[p]! * fx[p]! + fy[p]! * fy[p]! > rhoLow * rhoLow,
          `region ${r} member ${p} is below the edge floor and could never have been labeled`);
      }
    }
    assert.equal(total, memberCount, 'the region sizes do not sum to counts.y');
  });
});

test('a frame that asks for nothing is byte-for-byte the frame before inspection existed', async () => {
  await withDevice(async (device) => {
    const gray = renderPose(POSE, FRAME_DIMS, 4);
    // Not a performance claim -- a claim that the default path is unchanged.
    // The staging buffer is sized for the catalogue, but a frame requesting
    // nothing copies and maps only the 128-byte pose block.
    const declared = await frameWith(device, gray, { alias: false, inspect: INSPECT });
    const ctx = createPose2Context(device, FRAME, { alias: false, inspect: INSPECT });
    try {
      const quiet = await runPose2(ctx, Float32Array.from(gray), SETTINGS);
      assert.deepEqual(quiet.inspected, {});
      assert.deepEqual(quiet.pose, declared.pose);
    } finally {
      destroyPose2Context(ctx);
    }
  });
});

test('asking for something the context did not declare throws', async () => {
  await withDevice(async (device) => {
    const gray = renderPose(POSE, FRAME_DIMS, 4);
    const ctx = createPose2Context(device, FRAME, { alias: true, inspect: ['fx'] });
    try {
      // The hazard this closes is silent: under `alias` an undeclared buffer's
      // slot may hold a later occupant, so the copy would succeed and return
      // some other stage's plausible-looking bytes.
      await assert.rejects(
        () => runPose2(ctx, Float32Array.from(gray), SETTINGS, ['lines']),
        /did not declare/,
      );
    } finally {
      destroyPose2Context(ctx);
    }
  });
});
