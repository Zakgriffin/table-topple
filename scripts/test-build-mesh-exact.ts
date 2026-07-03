// Isolates buildMesh's BFS row/col assignment from corner detection (which
// test-sample-from-mesh.ts already ruled out) by feeding it EXACT,
// noise-free junction positions computed directly from known ground truth.
// If buildMesh assigns internally-INCONSISTENT (row,col) labels here — same
// real lattice position reachable via two paths gets different labels, or a
// step doesn't correspond to a consistent real distance — sampleFromMesh's
// bit sampling breaks even though sampleFromMesh itself is correct, which
// would explain scripts/test-mesh-decode.ts's bug without either piece
// being wrong in isolation.
//
// Tests theta values including near 90 degrees specifically, since that's
// what estimateRotationRad commonly returns on real captures (see the
// mesh-perspective debugging history) and scripts/test-mesh.ts's rotation
// sweep didn't happen to cover it.
//
// Usage: node scripts/test-build-mesh-exact.ts

import { buildMesh, pruneInconsistentNodes } from '../src/mesh.ts';
import type { RawJunction, Mesh } from '../src/mesh.ts';

const CELL = 20;
const N = 16; // NxN lattice points (0..N-1), all present — omission isn't
// the thing under test here, BFS's own bookkeeping is.

function omit(i: number, j: number): boolean {
  return (i * 7 + j * 13) % 5 === 0; // same ~20% omission pattern as test-sample-from-mesh.ts
}

function runTrial(thetaDeg: number): { mismatches: number; total: number; maxPosErr: number } {
  const theta = thetaDeg * Math.PI / 180;
  const cosT = Math.cos(theta), sinT = Math.sin(theta);
  const cx0 = (N * CELL) / 2, cy0 = (N * CELL) / 2;

  const junctions: RawJunction[] = [];
  const trueIndexByJunctionIdx: { i: number; j: number }[] = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (omit(i, j)) continue;
      // Exact position: lattice point (i,j) rotated by theta around the
      // grid's own center, same convention as scripts/test-mesh.ts.
      const rawX = j * CELL, rawY = i * CELL;
      const dx = rawX - cx0, dy = rawY - cy0;
      const x = dx * cosT - dy * sinT + cx0, y = dx * sinT + dy * cosT + cy0;
      junctions.push({ x, y, type: 'lcorner' });
      trueIndexByJunctionIdx.push({ i, j });
    }
  }

  const seedX = cx0, seedY = cy0;
  const mesh = buildMesh(junctions, seedX, seedY, CELL, CELL, theta);

  // Map each mesh node back to which junction (and thus true lattice index)
  // it came from, by nearest position — exact positions mean this is
  // unambiguous.
  let seedTrue: { i: number; j: number } | null = null;
  for (const node of mesh.nodes) {
    if (node.row === 0 && node.col === 0) {
      let best = Infinity, bi = -1;
      for (let k = 0; k < junctions.length; k++) {
        const d = (junctions[k].x - node.x) ** 2 + (junctions[k].y - node.y) ** 2;
        if (d < best) { best = d; bi = k; }
      }
      seedTrue = trueIndexByJunctionIdx[bi];
    }
  }
  if (!seedTrue) return { mismatches: -1, total: 0, maxPosErr: NaN };

  let mismatches = 0, total = 0, maxPosErr = 0;
  for (const node of mesh.nodes) {
    let best = Infinity, bi = -1;
    for (let k = 0; k < junctions.length; k++) {
      const d = (junctions[k].x - node.x) ** 2 + (junctions[k].y - node.y) ** 2;
      if (d < best) { best = d; bi = k; }
    }
    maxPosErr = Math.max(maxPosErr, Math.sqrt(best));
    const trueIdx = trueIndexByJunctionIdx[bi];
    const expectedRow = trueIdx.i - seedTrue.i, expectedCol = trueIdx.j - seedTrue.j;
    total++;
    if (node.row !== expectedRow || node.col !== expectedCol) mismatches++;
  }
  return { mismatches, total, maxPosErr };
}

let failures = 0;
for (const thetaDeg of [0, 5, 15, 30, 45, 60, 75, 85, 88, 89, 90, 91, 92, 95, 105, 135, 179]) {
  const r = runTrial(thetaDeg);
  const status = r.mismatches === 0 && r.maxPosErr < 0.01 ? 'PASS' : 'FAIL';
  if (status === 'FAIL') failures++;
  console.log(`theta=${String(thetaDeg).padStart(3)}deg: ${status} mismatches=${r.mismatches}/${r.total} maxPosErr=${r.maxPosErr.toFixed(4)}px`);
}
console.log(`\n${failures === 0 ? 'ALL PASS' : `${failures} FAILURES`}`);
if (failures > 0) process.exit(1);

console.log("\n=== with realistic position noise added ===");

function checkConsistency(mesh: Mesh, junctions: RawJunction[], trueIndexByJunctionIdx: { i: number; j: number }[]): { mismatches: number; total: number } {
  let seedTrue: { i: number; j: number } | null = null;
  for (const node of mesh.nodes) {
    if (node.row === 0 && node.col === 0) {
      let best = Infinity, bi = -1;
      for (let k = 0; k < junctions.length; k++) { const d = (junctions[k].x - node.x) ** 2 + (junctions[k].y - node.y) ** 2; if (d < best) { best = d; bi = k; } }
      seedTrue = trueIndexByJunctionIdx[bi];
    }
  }
  if (!seedTrue) return { mismatches: -1, total: 0 };

  let mismatches = 0, total = 0;
  for (const node of mesh.nodes) {
    let best = Infinity, bi = -1;
    for (let k = 0; k < junctions.length; k++) { const d = (junctions[k].x - node.x) ** 2 + (junctions[k].y - node.y) ** 2; if (d < best) { best = d; bi = k; } }
    const trueIdx = trueIndexByJunctionIdx[bi];
    const expectedRow = trueIdx.i - seedTrue.i, expectedCol = trueIdx.j - seedTrue.j;
    total++;
    if (node.row !== expectedRow || node.col !== expectedCol) mismatches++;
  }
  return { mismatches, total };
}

function buildNoisyMesh(thetaDeg: number, noisePx: number): { mesh: Mesh; junctions: RawJunction[]; trueIndexByJunctionIdx: { i: number; j: number }[] } {
  const theta = thetaDeg * Math.PI / 180;
  const cosT = Math.cos(theta), sinT = Math.sin(theta);
  const cx0 = (N * CELL) / 2, cy0 = (N * CELL) / 2;

  const junctions: RawJunction[] = [];
  const trueIndexByJunctionIdx: { i: number; j: number }[] = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (omit(i, j)) continue;
      const rawX = j * CELL, rawY = i * CELL;
      const dx = rawX - cx0, dy = rawY - cy0;
      const x = dx * cosT - dy * sinT + cx0 + (Math.random() - 0.5) * 2 * noisePx;
      const y = dx * sinT + dy * cosT + cy0 + (Math.random() - 0.5) * 2 * noisePx;
      junctions.push({ x, y, type: 'lcorner' });
      trueIndexByJunctionIdx.push({ i, j });
    }
  }
  const mesh = buildMesh(junctions, cx0, cy0, CELL, CELL, theta);
  return { mesh, junctions, trueIndexByJunctionIdx };
}

for (const noisePx of [0.5, 1, 1.5, 2, 3]) {
  for (const thetaDeg of [0, 45, 90]) {
    const { mesh, junctions, trueIndexByJunctionIdx } = buildNoisyMesh(thetaDeg, noisePx);
    const before = checkConsistency(mesh, junctions, trueIndexByJunctionIdx);
    console.log(`noise=${noisePx}px theta=${thetaDeg}deg: mismatches=${before.mismatches}/${before.total}`);
  }
}

console.log("\n=== after pruneInconsistentNodes ===");
for (const noisePx of [0.5, 1, 1.5, 2, 3]) {
  for (const thetaDeg of [0, 45, 90]) {
    const { mesh, junctions, trueIndexByJunctionIdx } = buildNoisyMesh(thetaDeg, noisePx);
    const before = checkConsistency(mesh, junctions, trueIndexByJunctionIdx);
    const pruned = pruneInconsistentNodes(mesh, CELL);
    const after = checkConsistency(pruned, junctions, trueIndexByJunctionIdx);
    const removed = before.total - after.total;
    // false positives: nodes pruned that were ACTUALLY correct (not among the mismatched ones)
    const beforeMismatchCount = before.mismatches;
    console.log(`noise=${noisePx}px theta=${thetaDeg}deg: before mismatches=${beforeMismatchCount}/${before.total} -> after mismatches=${after.mismatches}/${after.total} (removed ${removed} nodes)`);
  }
}
