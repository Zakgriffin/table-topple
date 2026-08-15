import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { create, globals } from 'webgpu';
import { validateFixture } from '../src/sphereLab/fixture.ts';
import { inputFromFixture } from '../src/sphereLab/harness/input.ts';
import { type SimDims, type SimPose } from '../src/pose2/sim.ts';
import { type PoseObservation, type SweepSpec, runSweep, summarize } from '../src/pose2/sweep.ts';
import { boardDims } from '../src/pose2/board.ts';
import { type Pose2Context, createPose2Context, runPose2 } from '../src/pose2/run.ts';
import { ORDER } from '../src/sphereLab/floorPattern.ts';
import { GRID_STEP } from '../src/sphereLab/constants.ts';
import { vFovRadOf } from '../src/pose2/sim.ts';
import { DAWN_NODE_FLAGS, requestDeviceWithOptionalTimestamps } from '../src/gpu/device.ts';
import { getRecords, profilerReset, spanEnd, spanStart } from '../src/sphereLab/profiling/profiler.ts';
import { ingestGpuFrame } from '../src/sphereLab/profiling/clocks.ts';

// ── The pose sweep, runnable ──────────────────────────────────────────────
//
//   node scripts/sweep.ts [--quick] [--res 480x640] [--ss 2]
//
// Renders a grid of known camera poses, runs a pipeline over each, and reports
// accuracy and timing against ground truth.
//
//   --pipeline pose2  (the only one left)  src/pose2, all GPU, one readback
//
// THE `pose` AND `both` ARMS ARE GONE, with src/pose itself. They ran the old
// pipeline over the same poses and the same renders, which is what produced the
// two directly comparable columns in section 19 of full_system_breakdown.md --
// identical accuracy to every printed digit, 27.9 -> 13.5 ms median. That
// comparison is RECORDED but can no longer be RE-RUN.
//
// The harness underneath is still deliberately pipeline-agnostic: a Runner takes
// a grayscale image and returns what it recovered. Nothing about the surviving
// arm depends on there being only one.

const argv = process.argv.slice(2);
const has = (f: string) => argv.includes(f);
const val = (f: string, d: string) => {
  const i = argv.indexOf(f);
  return i >= 0 && argv[i + 1] ? argv[i + 1]! : d;
};

const [wStr, hStr] = val('--res', '480x640').split('x');
const dims: SimDims = { w: Number(wStr), h: Number(hStr), horizFovDeg: 65 };
const supersample = Number(val('--ss', '4'));
const quick = has('--quick');

// The fixture supplies detector tuning that is known to work on a real capture.
// Inventing thresholds here would make the sweep measure the thresholds.
const base = inputFromFixture(
  validateFixture(JSON.parse(readFileSync('fixtures/default.json', 'utf8')), 'fixtures/default.json'),
);

const spec: SweepSpec = quick
  ? {
    heights: [10], tilts: [0, 20], yaws: [0], dims, supersample,
    offsets: [{ row: 70.5, col: 70.5 }],
  }
  : {
    // Heights span "close enough to resolve cells" to "high enough that cells
    // approach the sampling limit". Tilts run from nadir to strongly oblique,
    // which is where the grazing cutoff and line dropout start to bite.
    heights: [6, 10, 16, 24],
    tilts: [0, 10, 20, 30, 40],
    yaws: [0, 35, 90],
    // Three neighbourhoods of the torus, because the pattern is only LOCALLY
    // unique -- sweeping one spot tests one decode neighbourhood.
    offsets: [{ row: 70.5, col: 70.5 }, { row: 20.5, col: 110.5 }, { row: 100.5, col: 30.5 }],
    dims, supersample,
  };

const total = spec.heights.length * spec.tilts.length * spec.yaws.length * spec.offsets.length;
let done = 0;

// ── The per-stage GPU breakdown, accumulated across the sweep ─────────────
//
// One entry per stage id, holding that stage's TOTAL device time in each pose --
// per pose, not per pass, because grow's stages run 32 times a frame and the
// question a sweep answers is "what does this stage cost me on a frame", not
// "what does one round cost".
const gpuByStage = new Map<string, number[]>();
let posesTimed = 0;

function foldGpuRecords(): void {
  const perStage = new Map<string, number>();
  for (const r of getRecords()) {
    if (r.clock !== 'gpu') continue;
    const stage = r.id.slice('gpu:'.length);
    perStage.set(stage, (perStage.get(stage) ?? 0) + (r.end - r.start));
  }
  if (perStage.size > 0) posesTimed++;
  for (const [stage, ms] of perStage) {
    const list = gpuByStage.get(stage);
    if (list) list.push(ms); else gpuByStage.set(stage, [ms]);
  }
  // Cleared per pose so the 4096-record cap is never reached. profilerReset
  // also clears the User Timing buffer, which is empty here -- nothing in a
  // headless run mirrors to DevTools.
  profilerReset();
}

function quantile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const i = Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)));
  return sorted[i]!;
}

function gpuBreakdown(): string {
  if (posesTimed === 0) {
    return 'GPU pass breakdown: unavailable -- this device has no timestamp-query.';
  }
  const rows = [...gpuByStage.entries()]
    .map(([stage, xs]) => {
      const s = [...xs].sort((a, b) => a - b);
      return { stage, n: xs.length, med: quantile(s, 0.5), p90: quantile(s, 0.9), tot: xs.reduce((a, b) => a + b, 0) };
    })
    .sort((a, b) => b.med - a.med);
  const grand = rows.reduce((a, r) => a + r.med, 0);
  const lines = [
    `GPU pass breakdown over ${posesTimed} timed pose(s) -- per-pose totals per stage,`,
    `sorted by median. Lengths are measured; see profiling_rewrite.md on placement.`,
    `   ${'stage'.padEnd(24)} ${'med'.padStart(8)} ${'p90'.padStart(8)}  ${'%'.padStart(5)}`,
  ];
  for (const r of rows) {
    lines.push(`   ${r.stage.padEnd(24)} ${r.med.toFixed(3).padStart(8)} ${r.p90.toFixed(3).padStart(8)}`
      + `  ${((r.med / grand) * 100).toFixed(1).padStart(5)}`);
  }
  lines.push(`   ${'SUM OF MEDIANS'.padEnd(24)} ${grand.toFixed(3).padStart(8)}  ms on device per pose`);
  return lines.join('\n');
}
// ── src/pose2: all GPU, one readback ──────────────────────────────────────
//
// The settings come from the SAME fixture the baseline runner uses. src/pose2
// deliberately does not import the config -- it takes a settings object per
// stage -- so this is where the two pipelines are held to one set of thresholds.
// Sweeping them under different tuning would measure the tuning.
async function makePose2Runner() {
  Object.assign(globalThis, globals);
  const adapter = await gpuInstance.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  // Timestamps requested here for the same reason they are requested in the
  // app: this is the only moment they can be. The sweep is where timing
  // analysis actually happens -- 180 poses, medians, p90 -- so it is the caller
  // that would most quietly suffer from a device that cannot be timed.
  const device = await requestDeviceWithOptionalTimestamps(adapter);
  const st = base.settings;
  let ctx: Pose2Context | null = null;
  const runner = async (gray: Float64Array, d: SimDims, p: SimPose): Promise<PoseObservation> => {
    if (!ctx) {
      ctx = createPose2Context(device, {
        w: d.w, h: d.h, maxRegions: 16384, maxLines: 16384, ...boardDims(),
      }, { alias });
    }
    console.error(`  [${++done}/${total}] h=${p.height} tilt=${p.tiltDeg} yaw=${p.yawDeg} at (${p.overRow},${p.overCol})`);
    // A SPAN, not a stopwatch of its own. This is Phase 7's payoff: the sweep is
    // where timing analysis actually happens -- 180 poses, medians, p90 -- and
    // recording into the one store is what earns it the per-pass GPU breakdown
    // below for free, off an instrument that already exists.
    const span = spanStart('sweep.pose');
    const frame = await runPose2(ctx, Float32Array.from(gray), {
      grow: { rhoLow: st.lsdRhoNoiseThreshold, toleranceDeg: st.lsdToleranceDeg },
      collect: { rhoHigh: st.lsdRhoHighThreshold, minRegionSize: st.lsdMinRegionSize },
      lsdFit: {
        rho: st.lsdRhoNoiseThreshold, toleranceDeg: st.lsdToleranceDeg,
        nfaTestExponent: st.lsdNfaTestExponent, nfaEpsilon: st.lsdNfaEpsilon,
      },
      lines: { minLengthPx: st.lsdMinLengthPx },
      votes: { vFovRad: vFovRadOf(d) },
      gpp: { vFovRad: vFovRadOf(d), cellPitch: GRID_STEP, minGrazingCos: st.minGrazingCos },
      layout: { vFovRad: vFovRadOf(d), cellPitch: GRID_STEP, minGrazingCos: st.minGrazingCos },
      order: ORDER,
    });
    spanEnd(span);
    if (frame.gpu) ingestGpuFrame(frame.gpu, 'sweep.pose');
    const out = frame.pose;
    // Folded per pose and the store cleared, because 180 poses x ~140 passes is
    // ~25k records against a 4096 cap -- letting it trim would silently drop the
    // early poses out of the medians. The store is the transport; this is the
    // report, the same relationship the renderer has to it.
    foldGpuRecords();
    const ms = span.end - span.start;
    return {
      camPos: out.ok ? new THREE.Vector3(out.position.x, out.position.y, out.position.z) : null,
      height: out.height > 0 ? out.height : null,
      period: out.period > 0 ? out.period : null,
      consistency: out.ok ? out.consistency : null,
      ms,
    };
  };
  return runner;
}

// ── THE GPU INSTANCE MUST BE RETAINED AT MODULE SCOPE ──
//
// `create()` returns the object that owns the adapter and device. Holding it in
// a local inside makePose2Runner is NOT enough: the runner closure captures
// `device` and never mentions the instance, so V8 collects it and the native
// side is freed underneath a still-live device. The next mapAsync segfaults.
//
// Confirmed here, not inherited: the first version of this had it local and the
// sweep died with SIGSEGV part-way through pose 1. Same failure tests/helpers/
// gpu.ts records, reached by a different route -- which is the argument for the
// comment being in both places.
// Timestamps are quantized to a 65536ns grid without this -- see DAWN_NODE_FLAGS.
const gpuInstance = create(DAWN_NODE_FLAGS);

/**
 * `--alias` runs src/pose2 over the POOLED buffer set (§18): buffers whose live
 * ranges do not overlap share one allocation. Not a second code path -- the same
 * interval colouring over a real liveness table instead of a degenerate one.
 *
 * Running the sweep both ways and comparing is the only check on it that exists.
 * A missed clear under pooling hands a stage ANOTHER ARRAY's data rather than
 * last frame's plausible-looking values, so a difference is loud when it comes.
 */
const alias = has('--alias');

// `--pipeline` is still accepted so an old invocation says what happened rather
// than silently sweeping something else. There is only one pipeline now.
const which = val('--pipeline', 'pose2');
if (which !== 'pose2') {
  console.error(
    `--pipeline ${which} is gone: src/pose was deleted, so only pose2 can be swept.\n` +
    `The two-pipeline comparison it produced is recorded in full_system_breakdown.md §19.`,
  );
  process.exit(1);
}

done = 0;
console.error(`rendering + running ${total} poses at ${dims.w}x${dims.h}, supersample ${supersample} through src/pose2...`);
const runner = await makePose2Runner();
const rows = await runSweep(spec, runner);
console.error('');
console.error(summarize(rows, `src/pose2${alias ? ' (gpu, POOLED)' : ' (gpu)'} @ ${dims.w}x${dims.h}`));
console.error('');
console.error(gpuBreakdown());
console.error('');
