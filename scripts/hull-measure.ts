import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { validateFixture } from '../src/sphereLab/fixture.ts';
import { inputFromFixture } from '../src/sphereLab/harness/input.ts';
import { computePoseFromCapture } from '../src/pose/poseCompute.ts';
import {
  type DecodeGridLayout, type DecodeInput,
  buildCorrectnessArray, buildDecodeSampleGrid, decodeGridLayout, projectedUVScale,
  rotateGrid, rotatedZeroIndex, tallyPositionVotes,
} from '../src/pose/stages/decode/decodeGrid.ts';
import { C, ORDER, R, debruijnLookup } from '../src/sphereLab/floorPattern.ts';
import { GRID_STEP, MATH_QUAT } from '../src/sphereLab/constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../src/sphereLab/math/geometry.ts';
import { type DecodeSampleGrid, type DecodeSamplePoint, type VoteResult } from '../src/pose/results.ts';
import { type SimDims, type SimPose, renderPose, truthFor } from '../src/pose2/sim.ts';
import { type SweepSpec, generatePoses } from '../src/pose2/sweep.ts';

// ── THE LINE-HULL MEASUREMENT (full_system_breakdown.md §12, open decision 2) ──
//
//   node scripts/hull-measure.ts [--quick] [--res 480x640] [--ss 4]
//
// §12 proposes bounding the decode lattice by the gnomonically projected LINE
// ENDPOINTS instead of by the 49x49 ray grid over the view quadrilateral. That
// would delete decode.bounds/decode.boundsInit and make MAX_CELLS a derived
// quantity (torusR * torusC) rather than a chosen one.
//
// The premise is right about DECODABILITY and not automatically right about
// SAMPLING: sampling a cell centre needs one pixel on the right side of a
// threshold, while detecting the grid line that bounds it needs a whole
// NFA-significant region. There is a band where sampling still works and
// detection has already failed. §12 predicts the loss there is ~zero because
// those cells are sub-pixel and sample to noise -- and says explicitly that
// this is a prediction, not a measurement.
//
// So this script reports, per pose:
//
//   1. the QUAD-derived lattice dims        (settles MAX_CELLS for the current design)
//   2. the HULL-derived lattice dims        (settles it for the proposed one)
//   3. how many CORRECT decode votes come from cells outside the hull,
//      and what clipping to the hull does to the recovered POSE
//
// (3) is the decisive number. Nothing here is built in src/pose2 -- both
// quantities already exist in src/pose, which is what breaks the circularity
// open decision 2 used to have.
//
// ── THE TWO REIMPLEMENTATIONS, AND WHY THEY ARE NOT TRUSTED ──
//
// Measuring a second lattice needs two things src/pose only exposes bound to
// the FIRST one: building a grid from a layout that decodeGridLayout did not
// produce, and finishing a pose from a winner. Both are duplicated below, and
// neither is believed on inspection -- `selfCheck` runs both against the real
// functions on the unmodified layout and throws on any disagreement. A silently
// divergent instrument would report a hull cost that is really its own.

const argv = process.argv.slice(2);
const has = (f: string) => argv.includes(f);
const val = (f: string, d: string) => {
  const i = argv.indexOf(f);
  return i >= 0 && argv[i + 1] ? argv[i + 1]! : d;
};

const [wStr, hStr] = val('--res', '480x640').split('x');
const dims: SimDims = { w: Number(wStr), h: Number(hStr), horizFovDeg: 65 };
const supersample = Number(val('--ss', '4'));

const base = inputFromFixture(
  validateFixture(JSON.parse(readFileSync('fixtures/default.json', 'utf8')), 'fixtures/default.json'),
);

// --grazing is where the premise is actually AT RISK, and it is deliberately
// not folded into the main spec: the main one mirrors the §19 baseline's pose
// set so the lattice dims are comparable to it. §11's known limit is heavy line
// dropout at extreme grazing (~70 lines against ~400), which is exactly the
// band where "the lines bound the decodable region" could stop being true.
const spec: SweepSpec = has('--grazing')
  ? {
    heights: [10, 16, 24], tilts: [40, 45, 50, 55], yaws: [0, 35],
    offsets: [{ row: 70.5, col: 70.5 }], dims, supersample,
  }
  : has('--quick')
  ? { heights: [10], tilts: [0, 20, 40], yaws: [35], dims, supersample, offsets: [{ row: 70.5, col: 70.5 }] }
  : {
    heights: [6, 10, 16, 24],
    tilts: [0, 10, 20, 30, 40],
    yaws: [0, 35, 90],
    offsets: [{ row: 70.5, col: 70.5 }, { row: 20.5, col: 110.5 }, { row: 100.5, col: 30.5 }],
    dims, supersample,
  };

// ── The two duplications ──────────────────────────────────────────────────

/**
 * Why a cell was rejected, which is NOT bookkeeping: the hull can reach past the
 * grazing cutoff (a projected line endpoint is not gated the way every quad ray
 * is), and a cell beyond the horizon projects through a negated depth to a
 * MIRRORED point that can land on screen. So if the screen-bounds test were
 * doing all the work, dropping the per-cell grazing test would be safe -- and if
 * the grazing test is, dropping it samples the wrong pixel silently.
 */
const rejects = { grazing: 0, offscreen: 0, valid: 0 };

/** buildDecodeSampleGrid's body, taking a layout instead of deriving one. */
function buildGridFromLayout(
  layout: DecodeGridLayout, gray: Float64Array, w: number, h: number,
): DecodeSampleGrid {
  const {
    Drow, Dcol, normal, invQuat, distance, halfV, aspect, binThreshold, minGrazingCos,
    uPhase, vPhase, kMinU, kMinV, rows, cols, zeroI, zeroJ,
  } = layout;
  const p = new THREE.Vector3(), local = new THREE.Vector3(), rayDir = new THREE.Vector3();
  const points: DecodeSamplePoint[][] = [];
  for (let i = 0; i < rows; i++) {
    const v = vPhase + (kMinV + i) * GRID_STEP;
    const rowPoints: DecodeSamplePoint[] = [];
    for (let j = 0; j < cols; j++) {
      const u = uPhase + (kMinU + j) * GRID_STEP;
      p.copy(Drow).multiplyScalar(u).addScaledVector(Dcol, v).addScaledVector(normal, -distance);
      rayDir.copy(p).normalize();
      const grazingOk = rayDir.dot(normal) < -minGrazingCos;
      local.copy(p).applyQuaternion(invQuat);
      const ndcU = -local.x / (local.z * Math.tan(halfV) * aspect);
      const ndcV = -local.y / (local.z * Math.tan(halfV));
      const px = ((ndcU + 1) / 2) * w, py = ((1 - ndcV) / 2) * h;
      const onScreen = Number.isFinite(px) && Number.isFinite(py) && px >= 0 && px < w && py >= 0 && py < h;
      const valid = grazingOk && onScreen;
      // The cells that decide it: rejected by grazing while the screen-bounds
      // test would have ACCEPTED them. Those are the mirrored ones.
      if (!grazingOk && onScreen) rejects.grazing++;
      else if (!onScreen) rejects.offscreen++;
      else rejects.valid++;
      if (!valid) { rowPoints.push({ u, v, px, py, valid: false, bit: 0 }); continue; }
      const xx = Math.min(w - 1, Math.max(0, Math.round(px)));
      const yy = Math.min(h - 1, Math.max(0, Math.round(py)));
      rowPoints.push({ u, v, px, py, valid: true, bit: gray[yy * w + xx] < binThreshold ? 1 : 0 });
    }
    points.push(rowPoints);
  }
  return { rows, cols, zeroI, zeroJ, points };
}

/** finishPositionDecode's world position, which decodeGrid.ts does not export. */
function camPosFor(
  input: DecodeInput, vFovRad: number, winner: VoteResult,
  refPtU: number, refPtV: number, rotZeroI: number, rotZeroJ: number,
): THREE.Vector3 {
  const { Drow, Dcol, Dnormal, distance } = input.recoveredAxes!;
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, input.aspect).dot(normal) > 0) normal.negate();
  const refTorusRow = ((winner.anchorRow + rotZeroI) % R + R) % R;
  const refTorusCol = ((winner.anchorCol + rotZeroJ) % C + C) % C;

  let rowMath = Dcol.clone(), colMath = Drow.clone();
  for (let step = 0; step < winner.orientation; step++) {
    const nextRow = colMath, nextCol = rowMath.negate();
    rowMath = nextRow; colMath = nextCol;
  }
  const thirdMath = new THREE.Vector3().crossVectors(rowMath, colMath).normalize();
  const mathBasis = new THREE.Matrix4().makeBasis(rowMath, colMath, thirdMath);
  const rowWorld = new THREE.Vector3(0, 0, 1), colWorld = new THREE.Vector3(1, 0, 0);
  const thirdWorld = new THREE.Vector3().crossVectors(rowWorld, colWorld).normalize();
  const worldBasis = new THREE.Matrix4().makeBasis(rowWorld, colWorld, thirdWorld);
  const quat = new THREE.Quaternion().setFromRotationMatrix(worldBasis.multiply(mathBasis.invert()));

  const hitRelWorld = new THREE.Vector3()
    .addScaledVector(Drow.clone().applyQuaternion(quat), refPtU)
    .addScaledVector(Dcol.clone().applyQuaternion(quat), refPtV)
    .addScaledVector(normal.clone().applyQuaternion(quat), -distance);
  const worldPosTrue = new THREE.Vector3(
    (refTorusCol + 0.5 - C / 2) * GRID_STEP, 0, (refTorusRow + 0.5 - R / 2) * GRID_STEP);
  return worldPosTrue.sub(hitRelWorld);
}

/** One lattice, decoded end to end. */
function decodeWith(
  layout: DecodeGridLayout, input: DecodeInput, gray: Float64Array,
  w: number, h: number, vFovRad: number,
) {
  const grid = buildGridFromLayout(layout, gray, w, h);
  const winner = tallyPositionVotes(grid);
  if (!winner) return { grid, winner: null, camPos: null, consistency: 0, cells: layout.rows * layout.cols };
  const rotated = rotateGrid(grid, winner.orientation);
  const { correctCount, wrongCount } = buildCorrectnessArray(rotated, winner.anchorRow, winner.anchorCol);
  const [rzI, rzJ] = rotatedZeroIndex(layout.rows, layout.cols, layout.zeroI, layout.zeroJ, winner.orientation);
  const refU = layout.uPhase + (layout.kMinU + layout.zeroJ) * GRID_STEP;
  const refV = layout.vPhase + (layout.kMinV + layout.zeroI) * GRID_STEP;
  return {
    grid, winner,
    camPos: camPosFor(input, vFovRad, winner, refU, refV, rzI, rzJ),
    consistency: correctCount + wrongCount > 0 ? correctCount / (correctCount + wrongCount) : 0,
    cells: layout.rows * layout.cols,
  };
}

/**
 * THE THIRD NUMBER: of the votes that agree with the winner, how many come from
 * a window that hull-clipping would destroy?
 *
 * A window survives clipping exactly when all ORDER x ORDER of its cells lie
 * inside the hull's lattice-index rectangle -- tested on the integer index, not
 * on (u,v) with an epsilon, because the two lattices share uPhase/vPhase and so
 * share their index arithmetic exactly.
 */
function voteLoss(
  grid: DecodeSampleGrid, winner: VoteResult, quad: DecodeGridLayout,
  hull: { kMinU: number; kMaxU: number; kMinV: number; kMaxV: number },
) {
  const rotated = rotateGrid(grid, winner.orientation);
  const inHull = (pt: DecodeSamplePoint) => {
    const kU = Math.round((pt.u - quad.uPhase) / GRID_STEP);
    const kV = Math.round((pt.v - quad.vPhase) / GRID_STEP);
    return kU >= hull.kMinU && kU <= hull.kMaxU && kV >= hull.kMinV && kV <= hull.kMaxV;
  };
  let agree = 0, agreeLost = 0, disagree = 0, disagreeLost = 0;
  let windows = 0, windowsLost = 0;
  for (let i0 = 0; i0 + ORDER <= rotated.rows; i0++) {
    for (let j0 = 0; j0 + ORDER <= rotated.cols; j0++) {
      let complete = true, key = 0, whole = true;
      for (let di = 0; di < ORDER && complete; di++) {
        for (let dj = 0; dj < ORDER; dj++) {
          const pt = rotated.points[i0 + di][j0 + dj];
          if (!pt.valid) { complete = false; break; }
          if (!inHull(pt)) whole = false;
          key = ((key << 1) | pt.bit) >>> 0;
        }
      }
      if (!complete) continue;
      // Counted BEFORE the lookup: a window the clip destroys is a window
      // destroyed whether or not its key happens to be in the table. Without
      // this, "0 correct votes lost" cannot be told apart from "the clip never
      // reached a single complete window", which is a different claim.
      windows++;
      if (!whole) windowsLost++;
      const packed = debruijnLookup.get(key);
      if (packed === undefined) continue;
      const anchorRow = ((Math.floor(packed / C) - i0) % R + R) % R;
      const anchorCol = ((packed % C - j0) % C + C) % C;
      const agrees = anchorRow === winner.anchorRow && anchorCol === winner.anchorCol;
      if (agrees) { agree++; if (!whole) agreeLost++; } else { disagree++; if (!whole) disagreeLost++; }
    }
  }
  return { agree, agreeLost, disagree, disagreeLost, windows, windowsLost };
}

// ── The hull ──────────────────────────────────────────────────────────────

/**
 * min/max over every detected line's two gnomonically projected endpoints,
 * converted into the SAME (u,v) space the lattice lives in. §12's conversion:
 * gnomonic gives xRow = -(r.Drow)/(r.Dnormal) while projectedUVBounds gives
 * u = (r.Drow) * t with t = -distance/denom, so u = uvScale * xRow -- and
 * projectedUVScale is where src/pose already writes that scalar down, sign
 * convention included.
 */
/**
 * The same (u,v) extent the quad version bounds, computed from GROUND TRUTH
 * instead of from a recovered pose -- so "the lattice wanted 169 cells" can be
 * told apart from "the camera genuinely sees 169 cells".
 *
 * This is the question a hull edge past 144 actually raises. Either the camera
 * really does see more than one board period (legitimate: the floor tiles, and
 * a high oblique view covers a lot of it), or the hull is inflated by a line
 * detected near the horizon -- whose gnomonic projection has no minGrazingCos
 * gate, unlike every ray this function casts. The first wants a clip; the
 * second wants the endpoints gated.
 */
function truthExtentCells(pose: SimPose, aspect: number, vFovRad: number, minGrazingCos: number) {
  const t = truthFor(pose);
  const normal = t.DnormalMath.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, aspect).dot(normal) > 0) normal.negate();
  const N = 48;
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (let iy = 0; iy <= N; iy++) {
    for (let ix = 0; ix <= N; ix++) {
      const rayDir = cornerDir((ix / N) * 2 - 1, (iy / N) * 2 - 1, MATH_QUAT, vFovRad, aspect);
      const denom = rayDir.dot(normal);
      if (denom >= -minGrazingCos) continue;
      const s = -t.height / denom;
      const u = rayDir.dot(t.DrowMath) * s, v = rayDir.dot(t.DcolMath) * s;
      if (u < minU) minU = u; if (u > maxU) maxU = u;
      if (v < minV) minV = v; if (v > maxV) maxV = v;
    }
  }
  if (!isFinite(minU)) return { cols: 0, rows: 0 };
  return {
    cols: Math.round((maxU - minU) / GRID_STEP) + 1,
    rows: Math.round((maxV - minV) / GRID_STEP) + 1,
  };
}

function hullBoundsFor(input: DecodeInput): { minU: number; maxU: number; minV: number; maxV: number; endpoints: number } | null {
  const gpp = input.gridPeriodPhase;
  const scale = projectedUVScale(input);
  if (!gpp || scale === null) return null;
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity, endpoints = 0;
  for (const line of [...gpp.rowLines, ...gpp.colLines]) {
    for (const p of [line.p1, line.p2]) {
      const u = scale * p.xRow, v = scale * p.xCol;
      if (u < minU) minU = u; if (u > maxU) maxU = u;
      if (v < minV) minV = v; if (v > maxV) maxV = v;
      endpoints++;
    }
  }
  return endpoints ? { minU, maxU, minV, maxV, endpoints } : null;
}

/**
 * decodeGridLayout's extent arithmetic, re-run against different bounds.
 *
 * `--inset N` shrinks the hull by N cells on every side. It is not a design
 * option -- it is the DECISIVENESS CHECK on the loss counter. "0 correct votes
 * lost" is only worth reading if the counter can be nonzero, and this is what
 * demonstrates that on the same poses rather than by argument.
 */
const inset = Number(val('--inset', '0'));

/**
 * `--clip N` caps each lattice edge at N cells (default 144 = one board period,
 * 0 disables). This is the thing that makes MAX_CELLS = torusR * torusC a real
 * bound rather than a hope, and it is applied HERE so its cost is measured
 * rather than argued: the tiling argument says a second board period votes for
 * the same anchor and so carries no information, and this is what checks that.
 *
 * The kept window is CENTRED on the hull. The choice is not forced -- any
 * contiguous 144 works for the anchor, since the arithmetic is mod R/C -- and
 * centring is the least arbitrary of the options rather than the best one.
 */
const clip = Number(val('--clip', '144'));
function clampSpan(kMin: number, kMax: number): [number, number] {
  if (clip <= 0 || kMax - kMin + 1 <= clip) return [kMin, kMax];
  const drop = (kMax - kMin + 1) - clip;
  const lo = kMin + Math.floor(drop / 2);
  return [lo, lo + clip - 1];
}

function relayout(
  quad: DecodeGridLayout, b: { minU: number; maxU: number; minV: number; maxV: number },
): DecodeGridLayout & { kMaxU: number; kMaxV: number } {
  const { uPhase, vPhase } = quad;
  const [kMinU, kMaxU] = clampSpan(
    Math.floor((b.minU - uPhase) / GRID_STEP) + inset, Math.ceil((b.maxU - uPhase) / GRID_STEP) - inset);
  const [kMinV, kMaxV] = clampSpan(
    Math.floor((b.minV - vPhase) / GRID_STEP) + inset, Math.ceil((b.maxV - vPhase) / GRID_STEP) - inset);
  const cols = kMaxU - kMinU + 1, rows = kMaxV - kMinV + 1;
  const zeroI = Math.min(rows - 1, Math.max(0, Math.round(-vPhase / GRID_STEP) - kMinV));
  const zeroJ = Math.min(cols - 1, Math.max(0, Math.round(-uPhase / GRID_STEP) - kMinU));
  return { ...quad, kMinU, kMinV, kMaxU, kMaxV, rows, cols, zeroI, zeroJ };
}

// ── Self-check: the instrument, before any number it produces ──────────────

function selfCheck(
  layout: DecodeGridLayout, input: DecodeInput, gray: Float64Array, w: number, h: number,
  vFovRad: number, truthGrid: DecodeSampleGrid, truthCamPos: THREE.Vector3,
) {
  const mine = buildGridFromLayout(layout, gray, w, h);
  if (mine.rows !== truthGrid.rows || mine.cols !== truthGrid.cols) {
    throw new Error(`selfCheck: dims ${mine.rows}x${mine.cols} vs ${truthGrid.rows}x${truthGrid.cols}`);
  }
  for (let i = 0; i < mine.rows; i++) {
    for (let j = 0; j < mine.cols; j++) {
      const a = mine.points[i][j], b = truthGrid.points[i][j];
      if (a.valid !== b.valid || a.bit !== b.bit || a.u !== b.u || a.v !== b.v) {
        throw new Error(`selfCheck: cell (${i},${j}) differs`);
      }
    }
  }
  const got = decodeWith(layout, input, gray, w, h, vFovRad).camPos;
  if (!got) throw new Error('selfCheck: no winner on the quad lattice');
  const d = got.distanceTo(truthCamPos);
  if (!(d < 1e-9)) throw new Error(`selfCheck: camPos off by ${d} from the pipeline's own`);
}

// ── Run ───────────────────────────────────────────────────────────────────

interface Row {
  pose: SimPose;
  quadRows: number; quadCols: number; quadCells: number;
  hullRows: number; hullCols: number; hullCells: number;
  lines: number;
  agree: number; agreeLost: number; disagree: number; disagreeLost: number;
  windows: number; windowsLost: number;
  quadErr: number; hullErr: number;
  /** Ground-truth visible extent, and the recovered period's error. Together
   *  they say whether a hull edge past 144 is real geometry or a bad line. */
  trueRows: number; trueCols: number; periodErrPct: number;
  /** Hull lattice only: cells the grazing test rejected that the screen-bounds
   *  test would have accepted, versus cells that were simply off screen. */
  grazingOnly: number; offscreen: number; validCells: number;
  quadCons: number; hullCons: number;
  quadVotes: number; hullVotes: number;
  sameAnchor: boolean;
  note: string;
}

const poses = generatePoses(spec);
const vFovRad = getAnalysisVFovRad({ aspect: dims.w / dims.h, settings: base.settings });
const rows: Row[] = [];
let checked = false;

console.error(`${poses.length} poses at ${dims.w}x${dims.h}, supersample ${supersample}`);
for (const pose of poses) {
  const truth = truthFor(pose);
  const gray = renderPose(pose, dims, supersample);
  console.error(`  [${rows.length + 1}/${poses.length}] h=${pose.height} tilt=${pose.tiltDeg} yaw=${pose.yawDeg} at (${pose.overRow},${pose.overCol})`);

  const poseInput = { aspect: dims.w / dims.h, settings: base.settings };
  const result = await computePoseFromCapture(poseInput, gray, dims.w, dims.h, 'cpu');
  const input: DecodeInput = {
    aspect: dims.w / dims.h, settings: base.settings,
    recoveredAxes: result.recoveredAxes, gridPeriodPhase: result.gridPeriodPhase,
  };
  const quad = decodeGridLayout(input, gray, vFovRad);
  const hullB = hullBoundsFor(input);
  const blank = { quadRows: 0, quadCols: 0, quadCells: 0, hullRows: 0, hullCols: 0, hullCells: 0, lines: 0, agree: 0, agreeLost: 0, disagree: 0, disagreeLost: 0, windows: 0, windowsLost: 0, trueRows: 0, trueCols: 0, periodErrPct: NaN, grazingOnly: 0, offscreen: 0, validCells: 0, quadErr: NaN, hullErr: NaN, quadCons: 0, hullCons: 0, quadVotes: 0, hullVotes: 0, sameAnchor: false };
  if (!quad || !hullB || !result.positionDecode) {
    rows.push({ pose, ...blank, note: !quad ? 'NO LAYOUT' : !hullB ? 'NO HULL' : 'NO DECODE' });
    continue;
  }

  if (!checked) {
    selfCheck(quad, input, gray, dims.w, dims.h, vFovRad,
      buildDecodeSampleGrid(input, gray, dims.w, dims.h, vFovRad)!, result.positionDecode.camPos);
    console.error('  selfCheck: grid and camPos reproduce src/pose exactly');
    checked = true;
  }

  const hull = relayout(quad, hullB);
  const q = decodeWith(quad, input, gray, dims.w, dims.h, vFovRad);
  rejects.grazing = 0; rejects.offscreen = 0; rejects.valid = 0; // count the HULL lattice only
  const hOut = decodeWith(hull, input, gray, dims.w, dims.h, vFovRad);
  const loss = q.winner ? voteLoss(q.grid, q.winner, quad, hull)
    : { agree: 0, agreeLost: 0, disagree: 0, disagreeLost: 0, windows: 0, windowsLost: 0 };

  const trueExtent = truthExtentCells(pose, dims.w / dims.h, vFovRad, base.settings.minGrazingCos);
  const hullRejects = { ...rejects };
  const errOf = (p: THREE.Vector3 | null) => p === null ? NaN
    : Math.hypot((p.x - truth.camPos.x) / GRID_STEP, (p.z - truth.camPos.z) / GRID_STEP);
  rows.push({
    pose,
    quadRows: quad.rows, quadCols: quad.cols, quadCells: quad.rows * quad.cols,
    hullRows: hull.rows, hullCols: hull.cols, hullCells: hull.rows * hull.cols,
    lines: hullB.endpoints / 2,
    ...loss,
    trueRows: trueExtent.rows, trueCols: trueExtent.cols,
    grazingOnly: hullRejects.grazing, offscreen: hullRejects.offscreen, validCells: hullRejects.valid,
    periodErrPct: result.gridPeriodPhase
      ? 100 * (result.gridPeriodPhase.period - truth.period) / truth.period : NaN,
    quadErr: errOf(q.camPos), hullErr: errOf(hOut.camPos),
    quadCons: q.consistency, hullCons: hOut.consistency,
    quadVotes: q.winner?.votes ?? 0, hullVotes: hOut.winner?.votes ?? 0,
    sameAnchor: !!q.camPos && !!hOut.camPos && q.camPos.distanceTo(hOut.camPos) < 0.5 * GRID_STEP,
    note: hOut.winner ? '' : 'HULL: NO WINNER',
  });
}

// ── Report ────────────────────────────────────────────────────────────────

const f = (v: number, n = 3) => (Number.isFinite(v) ? v.toFixed(n) : '--');
const max = (xs: number[]) => xs.reduce((a, b) => (b > a ? b : a), -Infinity);

console.log('');
console.log(`── line hull vs view quad ── ${rows.length} poses at ${dims.w}x${dims.h}, ss ${supersample}`);
console.log('');
console.log(`  inset ${inset} cell(s), edge clip ${clip || "off"}`);
console.log('  pose                    lines   quad lattice     hull lattice    TRUE extent  per%   votes q/h   windows clipped  correct votes lost   pos err q/h (cells)   cons q/h');
for (const r of rows) {
  const p = r.pose;
  const lossPct = r.agree ? (100 * r.agreeLost / r.agree) : 0;
  console.log(
    `  h=${String(p.height).padStart(2)} t=${String(p.tiltDeg).padStart(2)} y=${String(p.yawDeg).padStart(2)} @(${p.overRow},${p.overCol})`.padEnd(26) +
    `${String(r.lines).padStart(5)}   ` +
    `${`${r.quadRows}x${r.quadCols}`.padStart(9)} ${String(r.quadCells).padStart(6)}  ` +
    `${`${r.hullRows}x${r.hullCols}`.padStart(9)} ${String(r.hullCells).padStart(6)}  ` +
    `${`${r.trueRows}x${r.trueCols}`.padStart(9)} ${f(r.periodErrPct, 2).padStart(6)}  ` +
    `${String(r.quadVotes).padStart(5)}/${String(r.hullVotes).padEnd(5)} ` +
    `${String(r.windowsLost).padStart(5)}/${String(r.windows).padEnd(5)} ` +
    `${String(r.agreeLost).padStart(5)}/${String(r.agree).padEnd(5)} ${`${lossPct.toFixed(1)}%`.padStart(6)}   ` +
    `${f(r.quadErr).padStart(6)}/${f(r.hullErr).padEnd(6)}  ` +
    `${f(r.quadCons, 2)}/${f(r.hullCons, 2)}` +
    (r.note ? `  ${r.note}` : '') + (r.sameAnchor ? '' : '  ANCHOR DIFFERS'),
  );
}

const ok = rows.filter((r) => !r.note);
console.log('');
console.log(`  poses decoded on both lattices   ${ok.filter((r) => r.hullVotes > 0).length}/${rows.length}`);
console.log(`  same recovered position          ${ok.filter((r) => r.sameAnchor).length}/${ok.length}`);
console.log(`  MAX_CELLS needed:  quad ${max(rows.map((r) => r.quadCells))}   hull ${max(rows.map((r) => r.hullCells))}   (torusR*torusC = ${R * C})`);
console.log(`  worst hull lattice edge          ${max(rows.map((r) => Math.max(r.hullRows, r.hullCols)))}  (144 is the one-board-period cap §12 proposes)`);
const totWin = ok.reduce((a, r) => a + r.windows, 0), totWinLost = ok.reduce((a, r) => a + r.windowsLost, 0);
console.log(`  complete windows clipped         ${totWinLost}/${totWin}  (${(100 * totWinLost / Math.max(1, totWin)).toFixed(2)}%)  <- if this is 0 the loss number below is vacuous`);
const totAgree = ok.reduce((a, r) => a + r.agree, 0), totLost = ok.reduce((a, r) => a + r.agreeLost, 0);
console.log(`  correct votes lost to clipping   ${totLost}/${totAgree}  (${(100 * totLost / Math.max(1, totAgree)).toFixed(2)}%)`);
console.log(`  worst single pose                ${max(ok.map((r) => (r.agree ? 100 * r.agreeLost / r.agree : 0))).toFixed(1)}%`);
console.log(`  worst position change            ${max(ok.map((r) => Math.abs(r.hullErr - r.quadErr))).toFixed(4)} cells`);
console.log(`  worst consistency change         ${max(ok.map((r) => Math.abs(r.hullCons - r.quadCons))).toFixed(4)}`);
// THE PER-CELL GRAZING TEST, and whether the hull makes it load-bearing.
const gz = ok.reduce((a, r) => a + r.grazingOnly, 0);
const off = ok.reduce((a, r) => a + r.offscreen, 0);
const vc = ok.reduce((a, r) => a + r.validCells, 0);
console.log('');
console.log(`  hull cells, by outcome:  valid ${vc}   off screen ${off}   REJECTED BY GRAZING ALONE ${gz}`);
console.log(`    (the last column is cells the screen-bounds test would have ACCEPTED -- delete the`);
console.log(`     per-cell grazing test and every one of them samples a mirrored pixel silently)`);
console.log(`    worst pose: ${max(ok.map((r) => r.grazingOnly))} such cells`);
