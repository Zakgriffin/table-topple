import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { readF32, readU32, withDevice } from './helpers/gpu.ts';
import { createBuffers, planPool } from '../src/pose2/buffers.ts';
import { cpuCollect, cpuGradient, cpuGrow, cpuLsdFit, logBinomialTail } from '../src/pose2/cpu.ts';
import { encodeCollect, encodeGradient, encodeGrow, encodeLsdFit, makeCtx } from '../src/pose2/pose.ts';
import { renderPose } from '../src/pose2/sim.ts';
import { TEST_WORLD } from './helpers/board.ts';
import type { Dims } from '../src/pose2/pipeline.ts';

// ── GPU against the CPU twin ──────────────────────────────────────────────
//
// The stage tests in pose2Stages.test.ts check grow against hand-derived truth
// on inputs simple enough to reason about: straight stripes, one polarity flip.
// Those catch a broken predicate. They cannot catch a wrong ANSWER on a real
// image, because on a real image nobody can write down the expected partition.
//
// This is the other half. Same stage, inputs it will actually see, scored by an
// independent formulation of the same definition. See src/pose2/cpu.ts for why
// the twin is a BFS rather than a transcription of the shader.

const GROW = { rhoLow: 0.05, toleranceDeg: 22.5 };

function dimsFor(w: number, h: number): Dims {
  return { w, h, maxRegions: 4096, maxLines: 2048, maxCells: 144 * 144, torusR: 144, torusC: 144, hashSlots: 65536 };
}

/** Runs gradient + grow on device and hands back what both stages produced. */
async function gpuGrow(device: GPUDevice, gray: Float32Array, dims: Dims): Promise<{
  fx: Float32Array; fy: Float32Array; label: Int32Array; converged: boolean;
}> {
  const n = dims.w * dims.h;
  const plan = planPool(dims, { alias: false });
  const bufs = createBuffers(device, plan);
  const ctx = makeCtx(device, plan, bufs, dims);
  device.queue.writeBuffer(bufs.gray!, 0, gray);
  encodeGradient(ctx);
  encodeGrow(ctx, GROW);
  device.queue.submit([ctx.enc.finish()]);

  const fx = await readF32(device, bufs.fx!, n);
  const fy = await readF32(device, bufs.fy!, n);
  const label = new Int32Array((await readU32(device, bufs.label!, n)).buffer);
  const args = await readU32(device, bufs.growArgs!, 4);
  return { fx, fy, label, converged: args[0] === 0 };
}

/** Where two label arrays disagree, and on how many pixels. */
function disagreement(a: Int32Array, b: Int32Array): { count: number; first: number } {
  let count = 0, first = -1;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) { count++; if (first < 0) first = i; }
  }
  return { count, first };
}

test('gradient: GPU and the twin agree bit-for-bit on a diagonal edge', async () => {
  await withDevice(async (device) => {
    const dims = dimsFor(64, 64);
    const gray = new Float32Array(dims.w * dims.h);
    // A diagonal half-plane: every 2x2 block along the boundary sees a different
    // corner pattern, so fx and fy take several distinct values rather than the
    // single 0/1 an axis-aligned step produces.
    for (let y = 0; y < dims.h; y++) {
      for (let x = 0; x < dims.w; x++) gray[y * dims.w + x] = x > y + 10 ? 255 : 0;
    }

    const gpu = await gpuGrow(device, gray, dims);
    const cpu = cpuGradient(gray, dims.w, dims.h);
    // Both sides do the same three adds and one divide in f32, so this is
    // equality, not closeness. A tolerance here would hide an operand-order bug.
    assert.deepEqual([...gpu.fx], [...cpu.fx]);
    assert.deepEqual([...gpu.fy], [...cpu.fy]);
  });
});

test('grow: GPU and the twin agree exactly on a diagonal bar', async () => {
  await withDevice(async (device) => {
    const dims = dimsFor(64, 64);
    const gray = new Float32Array(dims.w * dims.h).fill(255);
    // A ONE-PIXEL dark diagonal line, and the width is the whole point. The 2x2
    // block gradient turns it into two edges of opposite polarity in ADJACENT
    // columns, so the directed predicate is actually asked about an antiparallel
    // 8-neighbour pair. At 4px wide the two edges sit two columns apart, never
    // become neighbours, and the test cannot tell a signed dot from an unsigned
    // one -- which is exactly what a mutation run showed.
    for (let y = 0; y < dims.h; y++) {
      for (let x = 0; x < dims.w; x++) {
        if (x - y === 0) gray[y * dims.w + x] = 0;
      }
    }

    const gpu = await gpuGrow(device, gray, dims);
    assert.ok(gpu.converged, 'grow did not converge within the encoded rounds');
    const cpu = cpuGrow(gpu.fx, gpu.fy, dims.w, dims.h, GROW);

    // Establish the comparison is decisive BEFORE asserting on it. If any
    // neighbour pair sat on the tolerance boundary the two precisions could
    // legitimately split it, and an exact-equality failure would mean nothing.
    assert.equal(cpu.marginalPairs, 0, 'a neighbour pair sat on the tolerance boundary');

    const d = disagreement(gpu.label, cpu.label);
    assert.equal(d.count, 0,
      `${d.count} pixels disagree, first at ${d.first} (gpu ${gpu.label[d.first]}, cpu ${cpu.label[d.first]})`);

    // And the answer is the one the geometry demands, not merely a shared one.
    const components = new Set([...cpu.label].filter((l) => l >= 0));
    assert.equal(components.size, 2, 'a bar has two edges of opposite polarity');
  });
});

test('grow: GPU and the twin agree on a rendered board frame', async () => {
  // The real input distribution: anti-aliased edges, hundreds of components,
  // gradient directions varying continuously. No hand-derived truth exists for
  // this and ground truth cannot see it, which is precisely why the twin does.
  await withDevice(async (device) => {
    const dims = dimsFor(96, 128);
    const rendered = renderPose(TEST_WORLD, 
      { height: 10, tiltDeg: 20, yawDeg: 35, overRow: 70.5, overCol: 70.5 },
      { w: dims.w, h: dims.h, horizFovDeg: 65 }, 4,
    );
    const gray = Float32Array.from(rendered);

    const gpu = await gpuGrow(device, gray, dims);
    assert.ok(gpu.converged, 'grow did not converge within the encoded rounds');
    const cpu = cpuGrow(gpu.fx, gpu.fy, dims.w, dims.h, GROW);

    const labelled = [...cpu.label].filter((l) => l >= 0).length;
    assert.ok(labelled > 500, `only ${labelled} pixels were eligible -- the frame is not exercising grow`);
    const components = new Set([...cpu.label].filter((l) => l >= 0));
    assert.ok(components.size > 20, `only ${components.size} components -- expected the grid's many edges`);

    const d = disagreement(gpu.label, cpu.label);
    assert.equal(d.count, 0,
      `${d.count} of ${labelled} labelled pixels disagree, first at ${d.first} ` +
      `(gpu ${gpu.label[d.first]}, cpu ${cpu.label[d.first]}); ` +
      `${cpu.marginalPairs} neighbour pairs were on the tolerance boundary`);
  });
});

// ── S3 collect ────────────────────────────────────────────────────────────

const COLLECT = { rhoHigh: 0.2, minRegionSize: 4 };

/** gradient + grow + collect on device, with everything collect produced. */
async function gpuCollect(device: GPUDevice, gray: Float32Array, dims: Dims): Promise<{
  fx: Float32Array; fy: Float32Array; label: Int32Array;
  regionCount: number; memberCount: number;
  regionOffsets: Uint32Array; regionSizes: Uint32Array;
  members: Uint32Array; meanDirs: Float32Array;
}> {
  const n = dims.w * dims.h;
  const plan = planPool(dims, { alias: false });
  const bufs = createBuffers(device, plan);
  const ctx = makeCtx(device, plan, bufs, dims);
  device.queue.writeBuffer(bufs.gray!, 0, gray);
  encodeGradient(ctx);
  encodeGrow(ctx, GROW);
  encodeCollect(ctx, COLLECT);
  device.queue.submit([ctx.enc.finish()]);

  const counts = await readU32(device, bufs.counts!, 2);
  return {
    fx: await readF32(device, bufs.fx!, n),
    fy: await readF32(device, bufs.fy!, n),
    label: new Int32Array((await readU32(device, bufs.label!, n)).buffer),
    regionCount: counts[0]!,
    memberCount: counts[1]!,
    regionOffsets: await readU32(device, bufs.regionOffsets!, dims.maxRegions),
    regionSizes: await readU32(device, bufs.regionSizes!, dims.maxRegions),
    members: await readU32(device, bufs.members!, n),
    meanDirs: await readF32(device, bufs.meanDirs!, dims.maxRegions * 2),
  };
}

test('collect: GPU and the twin agree region-for-region on a rendered frame', async () => {
  await withDevice(async (device) => {
    const dims = dimsFor(96, 128);
    const rendered = renderPose(TEST_WORLD, 
      { height: 10, tiltDeg: 20, yawDeg: 35, overRow: 70.5, overCol: 70.5 },
      { w: dims.w, h: dims.h, horizFovDeg: 65 }, 4,
    );
    const gpu = await gpuCollect(device, Float32Array.from(rendered), dims);
    const cpu = cpuCollect(gpu.fx, gpu.fy, gpu.label, dims.w, dims.h,
      { ...COLLECT, maxRegions: dims.maxRegions });

    // The counts the device computed with no host involvement at all.
    assert.equal(gpu.regionCount, cpu.regionCount, 'region count');
    assert.equal(gpu.memberCount, cpu.memberCount, 'member count');
    assert.ok(cpu.regionCount > 10, `only ${cpu.regionCount} regions -- not exercising collect`);
    assert.ok(cpu.regionCount < dims.maxRegions, 'this test must not be running the overflow path');

    // REGION-FOR-REGION, with no canonicalization. That the ids line up at all
    // is the property the scan exists to provide: an atomic append would give
    // the same SET of regions under a scheduling-dependent permutation.
    for (let r = 0; r < cpu.regionCount; r++) {
      assert.equal(gpu.regionSizes[r], cpu.regionSizes[r], `region ${r} size`);
      assert.equal(gpu.regionOffsets[r], cpu.regionOffsets[r], `region ${r} offset`);
    }

    // Members must match exactly, INCLUDING order -- the atomic cursor hands out
    // slots in arrival order, so this is what proves finalize's sort ran.
    const off = cpu.regionOffsets[cpu.regionCount - 1]! + cpu.regionSizes[cpu.regionCount - 1]!;
    assert.deepEqual([...gpu.members.subarray(0, off)], [...cpu.members.subarray(0, off)]);

    // f32 vs f64 summation over up to a few hundred members, so closeness
    // rather than equality -- but tight, because these are unit vectors.
    for (let r = 0; r < cpu.regionCount * 2; r++) {
      assert.ok(Math.abs(gpu.meanDirs[r]! - cpu.meanDirs[r]!) < 1e-5,
        `meanDir[${r}]: gpu ${gpu.meanDirs[r]}, cpu ${cpu.meanDirs[r]}`);
    }
  });
});

// ── S4 lsdFit ─────────────────────────────────────────────────────────────

const LSDFIT = { rho: 0.132, toleranceDeg: 9.5, nfaTestExponent: 5, nfaEpsilon: 1 };

/**
 * An angle difference, into (-pi, pi].
 *
 * The two implementations pick different REPRESENTATIVES of the same angle and
 * that is not a defect in either. The shader's half-angle form lands theta in
 * [-pi/2, pi/2] before the 180-degree flip, so its range is [-pi/2, 3pi/2]; the
 * twin takes atan2 of an eigenvector whose sign is arbitrary, so its range is
 * [-pi, 2pi]. The same direction can come back 2pi apart.
 *
 * Wrapping is the right comparison rather than a lenient one: everything
 * downstream uses theta through cos and sin, so it is an angle, and a genuine
 * disambiguation failure -- the case this exists to catch -- is pi away and
 * survives the wrap at full size.
 */
function wrapAngle(d: number): number {
  return d - 2 * Math.PI * Math.round(d / (2 * Math.PI));
}

test('logBinomialTail: the twin\'s own tail, against an exact sum', async () => {
  // MUTATION-TEST THE ORACLE BEFORE BELIEVING IT. The twin's tail is the thing
  // the GPU's online rescale gets compared against, so it needs a check that
  // does not involve either log-sum-exp formulation.
  //
  // At small n the tail is computable EXACTLY in f64 -- plain binomial terms,
  // no cancellation, nothing near the underflow floor -- so the comparison is
  // against arithmetic rather than against a third implementation.
  const choose = (n: number, k: number): number => {
    let c = 1;
    for (let i = 1; i <= k; i++) c = (c * (n - k + i)) / i;
    return c;
  };
  for (const [n, k, p] of [[12, 3, 0.05], [20, 20, 0.0528], [30, 1, 0.25], [7, 0, 0.1]] as const) {
    let exact = 0;
    for (let j = k; j <= n; j++) exact += choose(n, j) * p ** j * (1 - p) ** (n - j);
    const got = logBinomialTail(n, k, p);
    assert.ok(Math.abs(got - Math.log(exact)) < 1e-9,
      `tail(${n}, ${k}, ${p}): ${got} vs exact ${Math.log(exact)}`);
  }
  // k > n is impossible by construction (k counts a subset of the n footprint
  // pixels), but the guard is what stops a caller bug becoming a NaN.
  assert.equal(logBinomialTail(5, 9, 0.1), -Infinity);
});

// The rendered-frame comparison is the ONLY thing that catches two of the
// mutations tried against this stage, and both for the same reason -- the
// synthetic fixtures are symmetric and land on exact arithmetic:
//
//   dropping BOUNDARY_EPS      the stripe fixtures put members at |proj|
//                              EXACTLY hl, and in f32 `13.0 > 13.0` is false,
//                              so they are included with or without the
//                              epsilon. Only irrational angles decide it.
//   centring on the weighted   symmetric fixtures put the centroid exactly at
//   centroid                   the midpoint of the extent, so the two agree
//
// The reverse also holds: the twin frame catches neither the unsigned-dot bug
// nor the sub-rho gate, which is what pose2Stages' hand-built block is for.
// Neither test is redundant, and neither alone is enough.
test('lsdFit: GPU and the twin agree rectangle-for-rectangle on a rendered frame', async () => {
  await withDevice(async (device) => {
    const dims = dimsFor(96, 128);
    const n = dims.w * dims.h;
    const rendered = renderPose(TEST_WORLD, 
      { height: 10, tiltDeg: 20, yawDeg: 35, overRow: 70.5, overCol: 70.5 },
      { w: dims.w, h: dims.h, horizFovDeg: 65 }, 4,
    );

    const plan = planPool(dims, { alias: false });
    const bufs = createBuffers(device, plan);
    const ctx = makeCtx(device, plan, bufs, dims);
    device.queue.writeBuffer(bufs.gray!, 0, Float32Array.from(rendered));
    encodeGradient(ctx);
    encodeGrow(ctx, GROW);
    encodeCollect(ctx, COLLECT);
    encodeLsdFit(ctx, LSDFIT);
    device.queue.submit([ctx.enc.finish()]);

    const counts = await readU32(device, bufs.counts!, 2);
    const regionCount = counts[0]!;
    const fx = await readF32(device, bufs.fx!, n);
    const fy = await readF32(device, bufs.fy!, n);
    const rects = await readF32(device, bufs.rects!, dims.maxRegions * 10);

    // The twin runs against the DEVICE's regions, so a disagreement here is
    // lsdFit's and not collect's. Chaining twin-collect into twin-fit would
    // report one stage's error as the other's.
    const cpu = cpuLsdFit(fx, fy, dims.w, dims.h, {
      regionCount,
      regionOffsets: await readU32(device, bufs.regionOffsets!, dims.maxRegions),
      regionSizes: await readU32(device, bufs.regionSizes!, dims.maxRegions),
      members: await readU32(device, bufs.members!, n),
      meanDirs: await readF32(device, bufs.meanDirs!, dims.maxRegions * 2),
    }, LSDFIT);

    assert.ok(regionCount > 10, `only ${regionCount} regions -- not exercising the fitter`);

    // ── Establish the comparison is decisive BEFORE asserting on it ──
    //
    // n and k are integers: they agree exactly or they do not. That is only
    // worth asserting if no pixel sat where the two sides could legitimately
    // put it on opposite sides -- so each margin gets compared against the
    // DISAGREEMENT THAT ACTUALLY EXISTS between them, measured here rather than
    // against a tolerance picked until the test passed.
    let dTheta = 0, dGeom = 0;
    for (let r = 0; r < regionCount; r++) {
      const o = r * 10, c = cpu.rects[r]!;
      dTheta = Math.max(dTheta, Math.abs(wrapAngle(rects[o + 2]! - c.theta)));
      dGeom = Math.max(dGeom, Math.abs(rects[o]! - c.cx), Math.abs(rects[o + 1]! - c.cy),
        Math.abs(rects[o + 3]! - c.length), Math.abs(rects[o + 4]! - c.width));
    }

    // theta's conditioning goes as 1 / anisotropy, so a blob with no axis lets
    // two correct implementations disagree by any amount at all. Check the
    // axis is determined before using dTheta as a bound on anything.
    assert.ok(cpu.minAnisotropy > 1e-3,
      `least anisotropic region is ${cpu.minAnisotropy} -- its principal axis is not determined well enough to compare`);

    // Inclusion is decided on proj and perp, which differ between the two sides
    // by at most dGeom plus the centre offset -- bounded by 2 * dGeom.
    assert.ok(cpu.closestInclusion > 2 * dGeom,
      `the two sides' geometry differs by ${dGeom} and an inclusion call came within ` +
      `${cpu.closestInclusion} of flipping -- exact n agreement is not licensed`);
    // alignDot is cos(levelLine - theta), so it moves by at most dTheta when
    // the axis does. This is the tight one: nothing protects the alignment
    // boundary the way BOUNDARY_EPS protects the inclusion one, because
    // level-line direction varies continuously and pixels land wherever.
    assert.ok(cpu.closestAlignment > dTheta,
      `the two axes differ by ${dTheta} rad and an alignment call came within ` +
      `${cpu.closestAlignment} of flipping -- exact k agreement is not licensed`);
    assert.ok(cpu.closestAccept > 1e-3,
      `a region sat within ${cpu.closestAccept} of the NFA accept threshold`);

    let accepted = 0;
    for (let r = 0; r < regionCount; r++) {
      const o = r * 10;
      const c = cpu.rects[r]!;
      const where = `region ${r} (${c.n} px, ${c.k} aligned)`;
      // Geometry: f32 against f64 over a few hundred members, at image
      // coordinates, so ~1e-4 is the floor rather than equality.
      assert.ok(Math.abs(rects[o]! - c.cx) < 2e-3, `${where} cx: ${rects[o]} vs ${c.cx}`);
      assert.ok(Math.abs(rects[o + 1]! - c.cy) < 2e-3, `${where} cy: ${rects[o + 1]} vs ${c.cy}`);
      assert.ok(Math.abs(wrapAngle(rects[o + 2]! - c.theta)) < 2e-3,
        `${where} theta: ${rects[o + 2]} vs ${c.theta}`);
      assert.ok(Math.abs(rects[o + 3]! - c.length) < 2e-3, `${where} length: ${rects[o + 3]} vs ${c.length}`);
      assert.ok(Math.abs(rects[o + 4]! - c.width) < 2e-3, `${where} width: ${rects[o + 4]} vs ${c.width}`);
      // The counts, exactly -- licensed by marginalPixels above. This is the
      // assertion that separates a geometry disagreement from a tail-arithmetic
      // one, which is the only reason n and k are carried at all.
      assert.equal(rects[o + 8], c.n, `${where} footprint count`);
      assert.equal(rects[o + 9], c.k, `${where} aligned count`);
      // Same counts in, so this is purely the two tail formulations against
      // each other -- and nfaLog10 runs to tens of decades, so relative.
      assert.ok(Math.abs(rects[o + 5]! - c.nfaLog10) < 1e-3 * Math.max(1, Math.abs(c.nfaLog10)),
        `${where} nfaLog10: ${rects[o + 5]} vs ${c.nfaLog10}`);
      assert.equal(rects[o + 6] !== 0, c.accepted, `${where} accepted`);
      if (c.accepted) accepted++;
    }

    // A frame of a printed grid must yield real lines. Without this the whole
    // comparison above is satisfiable by both sides rejecting everything.
    assert.ok(accepted > 10, `only ${accepted} of ${regionCount} rectangles were accepted`);
  });
});

test('collect: hysteresis drops a region with no strong pixel, and clears between runs', async () => {
  // Two synthetic edges of very different contrast. The faint one is above
  // grow's rhoLow (so it becomes a component) and below collect's rhoHigh (so
  // it must not become a region). Running the SAME buffers twice is what covers
  // the zero-init failure: labelSurvives is only ever written with 1, so
  // without its clear the faint label inherits run 1's survival flag.
  await withDevice(async (device) => {
    const dims = dimsFor(64, 64);
    const gray = new Float32Array(dims.w * dims.h);
    for (let y = 0; y < dims.h; y++) {
      for (let x = 20; x < dims.w; x++) gray[y * dims.w + x] = 255;   // strong step
      for (let x = 8; x < 12; x++) gray[y * dims.w + x] = 20;          // faint bar
    }

    const plan = planPool(dims, { alias: false });
    const bufs = createBuffers(device, plan);
    const runOnce = async () => {
      const ctx = makeCtx(device, plan, bufs, dims);
      device.queue.writeBuffer(bufs.gray!, 0, gray);
      encodeGradient(ctx);
      encodeGrow(ctx, GROW);
      encodeCollect(ctx, COLLECT);
      device.queue.submit([ctx.enc.finish()]);
      const c = await readU32(device, bufs.counts!, 2);
      return { regions: c[0]!, members: c[1]! };
    };

    const first = await runOnce();
    const second = await runOnce();

    const fx = await readF32(device, bufs.fx!, dims.w * dims.h);
    const fy = await readF32(device, bufs.fy!, dims.w * dims.h);
    const label = new Int32Array((await readU32(device, bufs.label!, dims.w * dims.h)).buffer);
    const cpu = cpuCollect(fx, fy, label, dims.w, dims.h, { ...COLLECT, maxRegions: dims.maxRegions });

    // The faint bar really is present as a component, or this proves nothing.
    const components = new Set([...label].filter((l) => l >= 0));
    assert.ok(components.size > cpu.regionCount,
      `hysteresis must drop something: ${components.size} components, ${cpu.regionCount} regions`);

    assert.equal(first.regions, cpu.regionCount);
    assert.deepEqual(second, first, 'run 2 disagreed with run 1 -- a buffer carried state across runs');
  });
});
