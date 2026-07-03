// Isolates sampleFromMesh from everything else (corner detection, mesh BFS
// construction, camera/homography) by hand-constructing a Mesh with EXACT,
// noise-free node positions directly from known ground truth, then checking
// that every sampled bit matches the source grid exactly. If this passes,
// the bug found in scripts/test-mesh-decode.ts (decoded bits not matching
// ground truth at any alignment) is upstream of sampleFromMesh — in corner
// detection or mesh construction — not in the sampling/diagonal-completion
// logic itself.
//
// Usage: node scripts/test-sample-from-mesh.ts

import { binarize, sampleFromMesh } from '../src/decode.ts';
import type { Mesh, MeshNode } from '../src/mesh.ts';

const CELL = 20;
const N = 12; // NxN cells -> (N+1)x(N+1) lattice points, indices 0..N
const W = N * CELL, H = W;

const cells: number[][] = Array.from({ length: N }, () => Array.from({ length: N }, () => (Math.random() < 0.5 ? 1 : 0)));

const gray = new Float64Array(W * H);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const ci = Math.floor(y / CELL), cj = Math.floor(x / CELL);
    gray[y * W + x] = cells[ci][cj] ? 0 : 255;
  }
}
const bin = binarize(gray);

// Every lattice point (i,j) for i,j in 0..N gets an EXACT node position —
// no detection noise, no BFS. Some are deliberately omitted to exercise the
// diagonal-completion logic (4-known, exactly-one-missing, and genuinely
// invalid 2-adjacent-known cases).
function omit(i: number, j: number): boolean {
  // Deterministic pattern, not random, so failures are reproducible:
  // omit every node where (i*7+j*13) % 5 === 0 (~20% omission rate).
  return (i * 7 + j * 13) % 5 === 0;
}

const nodes: MeshNode[] = [];
const byCoord = new Map<string, MeshNode>();
for (let i = 0; i <= N; i++) {
  for (let j = 0; j <= N; j++) {
    if (omit(i, j)) continue;
    const node: MeshNode = { x: j * CELL, y: i * CELL, type: 'lcorner', row: i, col: j };
    nodes.push(node);
    byCoord.set(`${i},${j}`, node);
  }
}
const mesh: Mesh = { nodes, byCoord };

const sg = sampleFromMesh(bin, W, H, mesh);
console.log(`Grid: ${N}x${N} cells, ${nodes.length}/${(N + 1) * (N + 1)} lattice points present.`);
console.log(`Sampled grid: ${sg.rows}x${sg.cols}, originRow=${sg.originRow} originCol=${sg.originCol}`);

let validCount = 0, correct = 0, wrong = 0;
const wrongExamples: string[] = [];
for (let i = 0; i < sg.rows; i++) {
  for (let j = 0; j < sg.cols; j++) {
    const cell = sg.cells[i][j];
    if (!cell.valid) continue;
    validCount++;
    // sg cell (i,j) should be TRUE grid cell (i - originRow, j - originCol)
    // — i.e. mesh (row,col) = (i-originRow, j-originCol) is this cell's
    // top-left corner, matching sampleFromMesh's own convention directly
    // (no orientation ambiguity to resolve here, since we built the mesh
    // ourselves with known absolute lattice indices).
    const trueRow = i - sg.originRow, trueCol = j - sg.originCol;
    if (trueRow < 0 || trueRow >= N || trueCol < 0 || trueCol >= N) {
      wrongExamples.push(`sg(${i},${j}) -> true(${trueRow},${trueCol}) OUT OF BOUNDS`);
      wrong++;
      continue;
    }
    if (cell.bit === cells[trueRow][trueCol]) correct++;
    else { wrong++; if (wrongExamples.length < 10) wrongExamples.push(`sg(${i},${j}) -> true(${trueRow},${trueCol}) expected=${cells[trueRow][trueCol]} got=${cell.bit} pos=(${cell.x.toFixed(1)},${cell.y.toFixed(1)})`); }
  }
}
console.log(`\nvalid cells: ${validCount}/${sg.rows * sg.cols}`);
console.log(`correct: ${correct}, wrong: ${wrong}`);
wrongExamples.forEach(e => console.log('  ', e));
console.log(`\n${wrong === 0 ? 'PASS' : 'FAIL'}`);
if (wrong > 0) process.exit(1);
