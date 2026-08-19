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
  /**
   * The recovered camera's WORLD orientation.
   *
   * Scored on its own because orientation is what the line fit produces
   * DIRECTLY, while position reaches the report through decode, the period
   * search and the lattice. A change that sharpens the fit shows up here first
   * and largest -- see SweepRow's normalErrDeg/yawErrDeg for the split the
   * 90-degree ambiguity forces on it.
   */
  camQuat: THREE.Quaternion | null;
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
  /**
   * Angle between the recovered floor normal and the true one, DEGREES.
   *
   * THE UNAMBIGUOUS HALF of the orientation error, and structurally so: a
   * square grid is only recovered up to a multiple of 90 degrees, and every one
   * of those rotations is ABOUT the normal and therefore leaves it fixed. So
   * this number needs no wrapping, no convention and no tie-break -- it is the
   * cleanest read on the line fit anywhere in the sweep.
   */
  normalErrDeg: number;
  /**
   * In-plane rotation error, DEGREES, minimized over the four 90-degree
   * rotations -- so it lands in [0, 45] and never reports a quadrant choice as
   * a 90-degree miss.
   */
  yawErrDeg: number;
  /**
   * WHICH multiple of 90 degrees minimized `yawErrDeg`. Zero means decode
   * landed in the same quadrant as truth.
   *
   * Kept as its own field rather than folded away because the ambiguity is a
   * real event, not a nuisance to be normalized out: `pose.quaternion` is
   * written AFTER decode applies the winning orientation, so a nonzero value
   * here means the De Bruijn tally chose a different cardinal direction than
   * the truth. That is a decode outcome, and rolling it into a continuous
   * angle would let one discrete miss swamp the metric the fit is judged by.
   */
  cardinalTurns: number;
  /**
   * The pose's SCALE, as the two numbers that actually govern detection: how
   * many board cells span the image's narrow axis, and how many pixels a cell
   * gets. Both at NADIR -- tilt stretches the real footprint, but tilt is
   * already its own axis in the report, and a label that moved with it could
   * not group anything.
   */
  cellsAcross: number;
  pxPerCell: number;
  consistency: number;
  ms: number;
}

const WORLD_UP = new THREE.Vector3(0, 1, 0);
/** Board col is world +X -- and `truthFor` maps THAT onto `DrowMath`. The
 *  naming swap is PoseTruth's, preserved here so the two sides compare. */
const WORLD_COL_DIR = new THREE.Vector3(1, 0, 0);
const DEG = 180 / Math.PI;

/**
 * The recovered orientation against truth, split along the ambiguity.
 *
 * Both quantities are computed in the camera's own frame, by rotating the known
 * world axes through the recovered orientation's inverse -- which is exactly
 * what `truthFor` does to build DrowMath/DnormalMath, so the two sides are
 * constructed the same way and a frame-convention slip cannot hide in one of
 * them.
 */
function orientationErr(
  camQuat: THREE.Quaternion, truth: { DrowMath: THREE.Vector3; DnormalMath: THREE.Vector3 },
): { normalErrDeg: number; yawErrDeg: number; cardinalTurns: number } {
  const inv = camQuat.clone().invert();
  const nRec = WORLD_UP.clone().applyQuaternion(inv);
  const aRec = WORLD_COL_DIR.clone().applyQuaternion(inv);

  // Rotated about the RECOVERED normal rather than the true one, so the search
  // stays self-consistent when the normal itself is off.
  let yawErrDeg = Infinity;
  let cardinalTurns = 0;
  for (let k = 0; k < 4; k++) {
    const turned = aRec.clone().applyAxisAngle(nRec, (k * Math.PI) / 2);
    const d = turned.angleTo(truth.DrowMath) * DEG;
    if (d < yawErrDeg) { yawErrDeg = d; cardinalTurns = k; }
  }
  return { normalErrDeg: nRec.angleTo(truth.DnormalMath) * DEG, yawErrDeg, cardinalTurns };
}

/**
 * How much board a pose can see, at nadir.
 *
 * Derived from the HORIZONTAL fov because `SimDims` declares that one and
 * `vFovRadOf` derives the other from the aspect -- taking the narrow axis keeps
 * this the pessimistic number, which is the one that decides whether ORDER
 * cells fit in frame at all.
 */
export function scaleOf(
  p: SimPose, dims: SimDims, world: SimWorld,
): { cellsAcross: number; pxPerCell: number } {
  const tanHalf = Math.tan((dims.horizFovDeg * Math.PI) / 360);
  const cellsAcross = (2 * p.height * tanHalf) / world.cellPitch;
  return { cellsAcross, pxPerCell: dims.w / cellsAcross };
}

export async function runSweep(spec: SweepSpec, runner: Runner): Promise<SweepRow[]> {
  const poses = generatePoses(spec);
  const rows: SweepRow[] = [];
  for (const pose of poses) {
    const truth = truthFor(spec.world, pose);
    const gray = renderPose(spec.world, pose, spec.dims, spec.supersample ?? 4);
    const obs = await runner(gray, spec.dims, pose);

    const scale = scaleOf(pose, spec.dims, spec.world);

    if (!obs.camPos) {
      rows.push({
        pose, recovered: false,
        cellRow: NaN, cellCol: NaN, anchorErr: NaN, subCellErr: NaN, posErr: NaN,
        heightErrRel: NaN, periodErrRel: NaN,
        normalErrDeg: NaN, yawErrDeg: NaN, cardinalTurns: NaN,
        ...scale, consistency: obs.consistency ?? 0, ms: obs.ms,
      });
      continue;
    }

    // NaN rather than 0 when a runner reports a position but no orientation:
    // `stats` filters non-finite values, so an absent measurement is dropped
    // from the quantiles instead of being averaged in as a perfect score.
    const orient = obs.camQuat
      ? orientationErr(obs.camQuat, truth)
      : { normalErrDeg: NaN, yawErrDeg: NaN, cardinalTurns: NaN };

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
      ...orient,
      ...scale,
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

  const turned = ok.filter((r) => Number.isFinite(r.cardinalTurns) && r.cardinalTurns !== 0);

  lines.push(`── ${label} ── ${rows.length} poses, board ${world.board.R}x${world.board.C}`);
  lines.push(`   recovered      ${ok.length}/${rows.length}`);
  lines.push(`   anchor exact   ${anchorOk.length}/${ok.length}  (whole-cell offset of zero)`);
  lines.push(`   cardinal exact ${ok.length - turned.length}/${ok.length}  (decode's 90-degree quadrant matches truth)`);
  lines.push('');

  const sub = stats(ok.map((r) => r.subCellErr));
  const pos = stats(ok.map((r) => r.posErr));
  const hgt = stats(ok.map((r) => Math.abs(r.heightErrRel)));
  const per = stats(ok.map((r) => Math.abs(r.periodErrRel)));
  const ms = stats(ok.map((r) => r.ms));
  const nrm = stats(ok.map((r) => r.normalErrDeg));
  const yaw = stats(ok.map((r) => r.yawErrDeg));
  lines.push('                    median      p90       max');
  lines.push(`   sub-cell err   ${f(sub.median).padStart(8)}  ${f(sub.p90).padStart(8)}  ${f(sub.max).padStart(8)}   cells (CONTINUOUS)`);
  lines.push(`   total pos err  ${f(pos.median).padStart(8)}  ${f(pos.p90).padStart(8)}  ${f(pos.max).padStart(8)}   cells`);
  lines.push(`   normal err     ${f(nrm.median).padStart(8)}  ${f(nrm.p90).padStart(8)}  ${f(nrm.max).padStart(8)}   deg (AMBIGUITY-FREE)`);
  lines.push(`   yaw err        ${f(yaw.median).padStart(8)}  ${f(yaw.p90).padStart(8)}  ${f(yaw.max).padStart(8)}   deg (mod 90)`);
  lines.push(`   height err     ${pct(hgt.median).padStart(8)}  ${pct(hgt.p90).padStart(8)}  ${pct(hgt.max).padStart(8)}`);
  lines.push(`   period err     ${pct(per.median).padStart(8)}  ${pct(per.p90).padStart(8)}  ${pct(per.max).padStart(8)}`);
  lines.push(`   time           ${f(ms.median, 1).padStart(8)}  ${f(ms.p90, 1).padStart(8)}  ${f(ms.max, 1).padStart(8)}   ms`);
  lines.push('');

  // ── Grouped two ways, because the two axes fail for unrelated reasons ──
  //
  // SCALE decides whether a cell survives sampling at all: past a few pixels
  // per cell the detector stops finding edges, and no amount of good geometry
  // downstream recovers that. TILT decides how much the projection stretches,
  // which is where grazing cutoffs and line dropout bite. A single grouping
  // would show a height column that is really a tilt effect, or the reverse.
  const heights = [...new Set(rows.map((r) => r.pose.height))].sort((a, b) => a - b);
  const tilts = [...new Set(rows.map((r) => r.pose.tiltDeg))].sort((a, b) => a - b);

  // `rec` and `ok` are both counts OUT OF n, not out of each other. The two
  // denominators are the readability defect the first expanded run had: a group
  // reading "7/21" next to a grid cell reading "7/45" is the same group twice.
  const group = (g: SweepRow[], lead: string): string => {
    const gok = g.filter((r) => r.recovered);
    const ga = gok.filter((r) => r.anchorErr === 0);
    return lead +
      `${String(g.length).padStart(4)} ` +
      `${String(gok.length).padStart(5)} ` +
      `${String(ga.length).padStart(5)}   ` +
      `${f(stats(gok.map((r) => r.subCellErr)).median).padStart(10)}   ` +
      `${f(stats(gok.map((r) => r.normalErrDeg)).median).padStart(10)}   ` +
      `${pct(stats(gok.map((r) => Math.abs(r.heightErrRel))).median).padStart(10)}   ` +
      `${f(stats(gok.map((r) => r.consistency)).median).padStart(8)}   ` +
      `${f(stats(gok.map((r) => r.ms)).median, 1).padStart(8)}`;
  };

  const HEAD = '     n   rec    ok   subCell(med)   normal(med)   height(med)   cons(med)    ms(med)';
  lines.push(`   by scale:   h   cells  px/cell${HEAD}`);
  for (const h of heights) {
    const g = rows.filter((r) => r.pose.height === h);
    const s = g[0]!;
    lines.push(group(g,
      `             ${String(h).padStart(5)} ${f(s.cellsAcross, 1).padStart(7)} ${f(s.pxPerCell, 1).padStart(8)}`));
  }
  lines.push('');

  lines.push(`   by tilt: tilt${' '.repeat(17)}${HEAD}`);
  for (const t of tilts) {
    lines.push(group(rows.filter((r) => r.pose.tiltDeg === t), `          ${String(t).padStart(6)}${' '.repeat(19)}`));
  }
  lines.push('');

  // ── The interaction, as the one number that is pass/fail ──
  //
  // Medians per axis cannot show that a scale is fine until it is ALSO tilted.
  // anchorOK is the right cell value: it is the binary "did this pose come out
  // right at all", so a hole in this grid is a region of pose space that does
  // not work rather than one that works less well.
  lines.push('   anchorOK, scale x tilt:');
  lines.push(`        h \\ tilt   ${tilts.map((t) => String(t).padStart(9)).join('')}`);
  for (const h of heights) {
    const cells = tilts.map((t) => {
      const g = rows.filter((r) => r.pose.height === h && r.pose.tiltDeg === t);
      const gok = g.filter((r) => r.recovered);
      return `${gok.filter((r) => r.anchorErr === 0).length}/${g.length}`.padStart(9);
    });
    lines.push(`        ${String(h).padStart(9)}  ${cells.join('')}`);
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
