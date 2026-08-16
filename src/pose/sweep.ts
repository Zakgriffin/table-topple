import * as THREE from 'three';
import { type SimDims, type SimPose, type SimWorld, renderPose, truthFor } from './sim.ts';

// ── The pose sweep ────────────────────────────────────────────────────────
//
// The acceptance harness for the rewrite: render many known poses, run a
// pipeline over each, and score the result against truth that is known by
// construction rather than supplied by a second implementation.
//
// ── Deliberately pipeline-agnostic ──
//
// A `Runner` takes a grayscale image and returns what it recovered. Nothing
// here imports a pipeline, so the SAME sweep scores the old pipeline and `src/pose`
// and the two sets of numbers are directly comparable -- which is the only way
// "is the rewrite at least as accurate" gets an answer rather than an opinion.
//
// ── THE ERROR DECOMPOSITION, which is the point ──
//
// Position error is not one quantity. It is:
//
//   DISCRETE   whole-cell jumps, from the decode locking onto a neighbouring
//              anchor. Moves the answer by exactly one cell at a time and
//              leaves the scale untouched.
//   CONTINUOUS the sub-cell remainder, plus height and period. This is where a
//              geometry, calibration or renderer error lives.
//
// Reporting them separately is what made the first sweep legible: dx stayed
// under 0.09 while dz moved in integer steps, which says "wrong anchor", not
// "wrong projection". A single RMS number would have hidden that completely.

export interface SweepSpec {
  /** Camera heights, in board units (1 unit = 1 cell). */
  heights: readonly number[];
  /** Degrees from straight down. */
  tilts: readonly number[];
  /** Board-relative yaw, degrees. */
  yaws: readonly number[];
  /**
   * Where the camera sits over the board, fractional cells. Several offsets
   * matter because the De Bruijn pattern is only locally unique -- a sweep that
   * never moves is testing one neighbourhood of the torus.
   */
  offsets: readonly { row: number; col: number }[];
  dims: SimDims;
  supersample?: number;
  /** The board being rendered and its cell pitch. Must be the same two the
   *  runner's pipeline is configured with -- see SimWorld. */
  world: SimWorld;
}

export function generatePoses(spec: SweepSpec): SimPose[] {
  const out: SimPose[] = [];
  for (const height of spec.heights) {
    for (const tiltDeg of spec.tilts) {
      for (const yawDeg of spec.yaws) {
        for (const o of spec.offsets) {
          out.push({ height, tiltDeg, yawDeg, overRow: o.row, overCol: o.col });
        }
      }
    }
  }
  return out;
}

/** What a pipeline hands back. All nullable: failing to recover is an outcome. */
export interface PoseObservation {
  camPos: THREE.Vector3 | null;
  /** recoveredAxes.distance -- the camera height. */
  height: number | null;
  /** gridPeriodPhase.period. */
  period: number | null;
  consistency: number | null;
  /** Wall-clock for the reconstruction alone, excluding rendering. */
  ms: number;
}

export type Runner = (gray: Float64Array, dims: SimDims, pose: SimPose) => Promise<PoseObservation>;

export interface SweepRow {
  pose: SimPose;
  /** False when the pipeline returned no pose at all. */
  recovered: boolean;
  /** Whole-cell anchor offset, in board row/col. Zero when the anchor is right. */
  cellRow: number;
  cellCol: number;
  /** Magnitude of the discrete part, in cells. */
  anchorErr: number;
  /** Sub-cell horizontal error, in cells. The continuous part. */
  subCellErr: number;
  /** Total horizontal position error, in cells. */
  posErr: number;
  heightErrRel: number;
  periodErrRel: number;
  consistency: number;
  ms: number;
}

export async function runSweep(spec: SweepSpec, runner: Runner): Promise<SweepRow[]> {
  const poses = generatePoses(spec);
  const rows: SweepRow[] = [];
  for (const pose of poses) {
    const truth = truthFor(spec.world, pose);
    const gray = renderPose(spec.world, pose, spec.dims, spec.supersample ?? 4);
    const obs = await runner(gray, spec.dims, pose);

    if (!obs.camPos) {
      rows.push({
        pose, recovered: false,
        cellRow: NaN, cellCol: NaN, anchorErr: NaN, subCellErr: NaN, posErr: NaN,
        heightErrRel: NaN, periodErrRel: NaN, consistency: obs.consistency ?? 0, ms: obs.ms,
      });
      continue;
    }

    // Board coordinates, so the discrete part is literally an integer count of
    // cells rather than a distance that happens to look like one.
    const dCol = (obs.camPos.x - truth.camPos.x) / spec.world.cellPitch;
    const dRow = (obs.camPos.z - truth.camPos.z) / spec.world.cellPitch;
    const cellCol = Math.round(dCol);
    const cellRow = Math.round(dRow);
    const subCol = dCol - cellCol;
    const subRow = dRow - cellRow;

    rows.push({
      pose,
      recovered: true,
      cellRow, cellCol,
      anchorErr: Math.hypot(cellRow, cellCol),
      subCellErr: Math.hypot(subRow, subCol),
      posErr: Math.hypot(dCol, dRow),
      heightErrRel: obs.height === null ? NaN : (obs.height - truth.height) / truth.height,
      periodErrRel: obs.period === null ? NaN : (obs.period - truth.period) / truth.period,
      consistency: obs.consistency ?? 0,
      ms: obs.ms,
    });
  }
  return rows;
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i]!;
}

function stats(values: number[]): { median: number; p90: number; max: number } {
  const s = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  return { median: quantile(s, 0.5), p90: quantile(s, 0.9), max: quantile(s, 1) };
}

const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
const f = (v: number, n = 3) => (Number.isFinite(v) ? v.toFixed(n) : '--');

/**
 * A report meant to be read, not parsed. Grouped by tilt because that is the
 * axis the first sweep showed structure along; regroup when a different one
 * starts mattering.
 */
export function summarize(rows: SweepRow[], world: SimWorld, label = 'sweep'): string {
  const lines: string[] = [];
  const ok = rows.filter((r) => r.recovered);
  const anchorOk = ok.filter((r) => r.anchorErr === 0);

  lines.push(`── ${label} ── ${rows.length} poses, board ${world.board.R}x${world.board.C}`);
  lines.push(`   recovered      ${ok.length}/${rows.length}`);
  lines.push(`   anchor exact   ${anchorOk.length}/${ok.length}  (whole-cell offset of zero)`);
  lines.push('');

  const sub = stats(ok.map((r) => r.subCellErr));
  const pos = stats(ok.map((r) => r.posErr));
  const hgt = stats(ok.map((r) => Math.abs(r.heightErrRel)));
  const per = stats(ok.map((r) => Math.abs(r.periodErrRel)));
  const ms = stats(ok.map((r) => r.ms));
  lines.push('                    median      p90       max');
  lines.push(`   sub-cell err   ${f(sub.median).padStart(8)}  ${f(sub.p90).padStart(8)}  ${f(sub.max).padStart(8)}   cells (CONTINUOUS)`);
  lines.push(`   total pos err  ${f(pos.median).padStart(8)}  ${f(pos.p90).padStart(8)}  ${f(pos.max).padStart(8)}   cells`);
  lines.push(`   height err     ${pct(hgt.median).padStart(8)}  ${pct(hgt.p90).padStart(8)}  ${pct(hgt.max).padStart(8)}`);
  lines.push(`   period err     ${pct(per.median).padStart(8)}  ${pct(per.p90).padStart(8)}  ${pct(per.max).padStart(8)}`);
  lines.push(`   time           ${f(ms.median, 1).padStart(8)}  ${f(ms.p90, 1).padStart(8)}  ${f(ms.max, 1).padStart(8)}   ms`);
  lines.push('');

  lines.push('   by tilt:  tilt   n   anchorOK   subCell(med)   height(med)   cons(med)    ms(med)');
  const tilts = [...new Set(rows.map((r) => r.pose.tiltDeg))].sort((a, b) => a - b);
  for (const t of tilts) {
    const g = rows.filter((r) => r.pose.tiltDeg === t);
    const gok = g.filter((r) => r.recovered);
    const ga = gok.filter((r) => r.anchorErr === 0);
    lines.push(
      `           ${String(t).padStart(5)} ${String(g.length).padStart(3)}   ` +
      `${String(ga.length).padStart(3)}/${String(gok.length).padEnd(3)}   ` +
      `${f(stats(gok.map((r) => r.subCellErr)).median).padStart(10)}   ` +
      `${pct(stats(gok.map((r) => Math.abs(r.heightErrRel))).median).padStart(10)}   ` +
      `${f(stats(gok.map((r) => r.consistency)).median).padStart(8)}   ` +
      `${f(stats(gok.map((r) => r.ms)).median, 1).padStart(8)}`,
    );
  }

  // Named failures, because an aggregate that hides WHICH pose broke is not
  // actionable -- and the whole reason for the sweep is to find where things
  // break down, not to produce a single score.
  const bad = rows.filter((r) => !r.recovered || r.anchorErr > 0);
  if (bad.length) {
    lines.push('');
    lines.push(`   ${bad.length} pose(s) with no recovery or a wrong anchor:`);
    for (const r of bad.slice(0, 20)) {
      const p = r.pose;
      const what = !r.recovered ? 'NO POSE' : `anchor off (${r.cellRow}, ${r.cellCol})`;
      lines.push(`     h=${p.height} tilt=${p.tiltDeg} yaw=${p.yawDeg} at (${p.overRow}, ${p.overCol}): ${what}, cons ${f(r.consistency)}`);
    }
    if (bad.length > 20) lines.push(`     ... and ${bad.length - 20} more`);
  }
  return lines.join('\n');
}
