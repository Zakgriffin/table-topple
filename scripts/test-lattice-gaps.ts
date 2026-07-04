// Validates the actual fix from this session: src/lattice.ts's
// recoverIndicesFromTransversal (used inside indexFamilyLines) should
// recover each line's TRUE relative integer index even when some lines are
// MISSING — exactly what real De Bruijn content does (confirmed via
// scripts/test-lines-decode.ts: ~50% edge density means a real grid line can
// have too little visible edge to survive Level 1's peak threshold). The old
// naive "consecutive in sorted order == consecutive integers" assignment
// broke the instant a line was missing — this test constructs exactly that
// scenario directly (known true indices, some deliberately dropped) and
// checks recovery is still correct, not just "doesn't crash".
//
// Usage: node scripts/test-lattice-gaps.ts

import type { LineCandidate } from '../src/lines.ts';
import type { LineFamily } from '../src/vp.ts';
import { estimateVanishingPoint } from '../src/vp.ts';
import { indexFamilyLines, buildLatticeCorrespondences } from '../src/lattice.ts';
import { fitHomographyRobust, applyHomography } from '../src/homography.ts';
import { projectToImage, type CameraPose } from './lib/synth-camera.ts';

const W = 640, H = 480;
const PITCH = 30;
let pass = 0, fail = 0;
function check(name: string, ok: boolean, detail: string) {
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: ${detail}`);
  if (ok) pass++; else fail++;
}

function projectLine(pose: CameraPose, worldA: [number, number], worldB: [number, number]): LineCandidate | null {
  const pa = projectToImage(pose, W, H, worldA[0], worldA[1]);
  const pb = projectToImage(pose, W, H, worldB[0], worldB[1]);
  if (!pa || !pb) return null;
  const cx = W / 2, cy = H / 2;
  const dx = pb[0] - pa[0], dy = pb[1] - pa[1];
  let nx = -dy, ny = dx;
  let theta = Math.atan2(ny, nx);
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  const a = Math.cos(theta), b = Math.sin(theta);
  const rho = a * (pa[0] - cx) + b * (pa[1] - cy);
  return { theta, rho, weight: 1 };
}

function runScenario(name: string, pose: CameraPose, droppedRowIndices: Set<number>, droppedColIndices: Set<number>, tolerancePx: number) {
  const rowTrueIdx = new Map<LineCandidate, number>();
  const colTrueIdx = new Map<LineCandidate, number>();
  const rowLines: LineCandidate[] = [];
  const colLines: LineCandidate[] = [];

  for (let i = -6; i <= 6; i++) {
    if (droppedRowIndices.has(i)) continue; // simulates a real grid line Level 1 failed to detect
    const line = projectLine(pose, [-300, i * PITCH], [300, i * PITCH]);
    if (!line) continue;
    rowTrueIdx.set(line, i);
    rowLines.push(line);
  }
  for (let j = -6; j <= 6; j++) {
    if (droppedColIndices.has(j)) continue;
    const line = projectLine(pose, [j * PITCH, -300], [j * PITCH, 300]);
    if (!line) continue;
    colTrueIdx.set(line, j);
    colLines.push(line);
  }

  const vpRow = estimateVanishingPoint(rowLines, W, H);
  const vpCol = estimateVanishingPoint(colLines, W, H);
  const rowFamily: LineFamily = { vp: vpRow, lines: rowLines };
  const colFamily: LineFamily = { vp: vpCol, lines: colLines };

  const rowIndexed = indexFamilyLines(rowFamily, vpCol, W, H);
  const colIndexed = indexFamilyLines(colFamily, vpRow, W, H);

  if (rowIndexed.length !== rowLines.length || colIndexed.length !== colLines.length) {
    check(name, false, `dropped lines during recovery: rows ${rowIndexed.length}/${rowLines.length}, cols ${colIndexed.length}/${colLines.length}`);
    return;
  }

  // The critical check: recovered indices must reproduce the TRUE gap
  // pattern, not silently renumber to consecutive. Verify via the affine
  // relationship between recovered and true index (recovered = m*true + k):
  // if gaps were preserved correctly, EVERY line satisfies this with the
  // SAME (m,k), not just the two used to derive it.
  function checkGapsPreserved(indexed: { index: number; line: LineCandidate }[], trueIdx: Map<LineCandidate, number>): { ok: boolean; detail: string } {
    const pts = indexed.map(x => ({ rec: x.index, truth: trueIdx.get(x.line)! }));
    const sorted = pts.slice().sort((a, b) => a.truth - b.truth);
    const first = sorted[0], last = sorted[sorted.length - 1];
    const m = (last.rec - first.rec) / (last.truth - first.truth);
    const k = first.rec - m * first.truth;
    let maxErr = 0;
    for (const p of pts) maxErr = Math.max(maxErr, Math.abs((m * p.truth + k) - p.rec));
    return { ok: maxErr < 0.01, detail: `m=${m.toFixed(3)} k=${k.toFixed(3)} maxAffineErr=${maxErr.toFixed(3)}` };
  }
  const rowCheck = checkGapsPreserved(rowIndexed, rowTrueIdx);
  const colCheck = checkGapsPreserved(colIndexed, colTrueIdx);

  // End-to-end: does the resulting (gappy) correspondence set still fit a
  // homography that extrapolates correctly to a held-out point?
  const correspondences = buildLatticeCorrespondences(rowIndexed, colIndexed, W, H);
  const H_fit = fitHomographyRobust(correspondences);
  let extrapOk = false, extrapDetail = 'no fit';
  if (H_fit) {
    const rowMap = (() => {
      const pts = rowIndexed.map(x => ({ rec: x.index, truth: rowTrueIdx.get(x.line)! })).sort((a, b) => a.truth - b.truth);
      const f = pts[0], l = pts[pts.length - 1];
      const m = (l.rec - f.rec) / (l.truth - f.truth);
      return { m, k: f.rec - m * f.truth };
    })();
    const colMap = (() => {
      const pts = colIndexed.map(x => ({ rec: x.index, truth: colTrueIdx.get(x.line)! })).sort((a, b) => a.truth - b.truth);
      const f = pts[0], l = pts[pts.length - 1];
      const m = (l.rec - f.rec) / (l.truth - f.truth);
      return { m, k: f.rec - m * f.truth };
    })();
    const heldOutI = 8, heldOutJ = -8;
    const truePixel = projectToImage(pose, W, H, heldOutJ * PITCH, heldOutI * PITCH);
    if (truePixel) {
      const predRowIdx = Math.round(rowMap.m * heldOutI + rowMap.k);
      const predColIdx = Math.round(colMap.m * heldOutJ + colMap.k);
      const predicted = applyHomography(H_fit, predRowIdx, predColIdx);
      if (predicted) {
        const err = Math.hypot(predicted[0] - truePixel[0], predicted[1] - truePixel[1]);
        extrapOk = err < tolerancePx;
        extrapDetail = `held-out err=${err.toFixed(2)}px`;
      }
    }
  }

  const ok = rowCheck.ok && colCheck.ok && extrapOk;
  check(name, ok, `rows:${rowCheck.detail} cols:${colCheck.detail} ${extrapDetail}`);
}

const pose: CameraPose = { targetX: 0, targetY: 0, dist: 400, tilt: 0.4, azimuth: 0.5, roll: 0.3, focal: 500 };

runScenario('no gaps (sanity)', pose, new Set(), new Set(), 1);
runScenario('one row gap', pose, new Set([1]), new Set(), 1);
runScenario('one row + one col gap', pose, new Set([1]), new Set([-2]), 1);
runScenario('two adjacent row gaps (2-cell jump)', pose, new Set([0, 1]), new Set(), 1);
runScenario('multiple scattered gaps both families', pose, new Set([-4, 1, 3]), new Set([-3, 0, 4]), 1);
runScenario('gap near the edge of the detected range', pose, new Set([-6]), new Set([6]), 1);

console.log(`\n${pass}/${pass + fail} correct`);
if (fail > 0) process.exit(1);
