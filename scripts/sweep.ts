import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { create, globals } from 'webgpu';
import { validateFixture } from '../src/sphereLab/fixture.ts';
import { inputFromFixture } from '../src/sphereLab/harness/input.ts';
import { computePoseFromCapture } from '../src/pose/poseCompute.ts';
import { type SimDims, type SimPose } from '../src/pose2/sim.ts';
import { type PoseObservation, type SweepSpec, runSweep, summarize } from '../src/pose2/sweep.ts';
import { boardDims } from '../src/pose2/board.ts';
import { type Pose2Context, createPose2Context, runPose2 } from '../src/pose2/run.ts';
import { ORDER } from '../src/sphereLab/floorPattern.ts';
import { GRID_STEP } from '../src/sphereLab/constants.ts';
import { vFovRadOf } from '../src/pose2/sim.ts';

// ── The pose sweep, runnable ──────────────────────────────────────────────
//
//   node scripts/sweep.ts [--quick] [--res 480x640] [--ss 2]
//
// Renders a grid of known camera poses, runs a pipeline over each, and reports
// accuracy and timing against ground truth.
//
//   --pipeline pose  (default)  src/pose on the CPU backend -- THE BASELINE
//   --pipeline pose2            src/pose2, all GPU, one readback
//   --pipeline both             one after the other, same poses, same renders
//
// The two columns are directly comparable because the harness is
// PIPELINE-AGNOSTIC: a Runner takes a grayscale image and returns what it
// recovered. That is the whole reason "is the rewrite at least as accurate"
// gets an answer rather than an opinion.

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
// computePoseFromCapture, NOT runPoseOn: the harness wrapper drains fx, fy,
// regionId, the region CSR and the decode grid into host arrays, and this sweep
// discards every one of them. At 180 poses that churn exhausted the V8 heap.
const cpuRunner = async (gray: Float64Array, d: SimDims, p: SimPose): Promise<PoseObservation> => {
  const input = { aspect: d.w / d.h, settings: base.settings };
  // Logged BEFORE the run, so a pose that kills the process names itself. A
  // progress counter printed afterwards tells you nothing about the one that
  // died.
  const mb = (process.memoryUsage().heapUsed / 1048576).toFixed(0);
  console.error(`  [${++done}/${total}] h=${p.height} tilt=${p.tiltDeg} yaw=${p.yawDeg} at (${p.overRow},${p.overCol})  heap ${mb}MB`);
  const t0 = performance.now();
  const pose = await computePoseFromCapture(input, gray, d.w, d.h, 'cpu');
  const ms = performance.now() - t0;
  return {
    camPos: pose.positionDecode?.camPos ?? null,
    height: pose.recoveredAxes?.distance ?? null,
    period: pose.gridPeriodPhase?.period ?? null,
    consistency: pose.positionDecode?.consistency ?? null,
    ms,
  };
};

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
  const device = await adapter.requestDevice();
  const st = base.settings;
  let ctx: Pose2Context | null = null;
  const runner = async (gray: Float64Array, d: SimDims, p: SimPose): Promise<PoseObservation> => {
    if (!ctx) {
      ctx = createPose2Context(device, {
        w: d.w, h: d.h, maxRegions: 16384, maxLines: 16384, ...boardDims(),
      }, { alias });
    }
    console.error(`  [${++done}/${total}] h=${p.height} tilt=${p.tiltDeg} yaw=${p.yawDeg} at (${p.overRow},${p.overCol})`);
    const t0 = performance.now();
    const out = await runPose2(ctx, Float32Array.from(gray), {
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
    const ms = performance.now() - t0;
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
const gpuInstance = create([]);

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

const which = val('--pipeline', 'pose');
const reports: string[] = [];
for (const name of which === 'both' ? ['pose', 'pose2'] : [which]) {
  done = 0;
  console.error(`rendering + running ${total} poses at ${dims.w}x${dims.h}, supersample ${supersample} through src/${name}...`);
  const runner = name === 'pose2' ? await makePose2Runner() : cpuRunner;
  const rows = await runSweep(spec, runner);
  const tag = name === 'pose' ? ' (cpu)' : alias ? ' (gpu, POOLED)' : ' (gpu)';
  reports.push(summarize(rows, `src/${name}${tag} @ ${dims.w}x${dims.h}`));
}
console.error('');
for (const r of reports) { console.error(r); console.error(''); }
