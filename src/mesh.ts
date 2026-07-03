// Builds a connected mesh of junction points into mesh-relative (row, col)
// coordinates, via local-basis-vector propagation outward from a seed —
// see the architecture discussion this supersedes PuzzleBoard's from-scratch
// topology inference with: our junctions already carry a type label and a
// sub-pixel position (src/cornerdetect.ts), and a coarse pitch/rotation
// estimate (src/decode.ts's detectGrid) is available to seed the search, so
// we don't need to reconstruct adjacency from an unordered point cloud with
// zero prior the way PuzzleBoard does.
//
// Only 5/8 of grid lattice points are directly detectable (L-corner + saddle
// — flat and straight-edge points have no 2D-localizable signal, see
// cornerdetect.ts), so roughly a third of lattice positions are invisible
// gaps. The search tries skipping 1, 2, or 3 steps ahead in each direction
// to jump over them rather than requiring every lattice point to be found.
//
// Each node inherits its parent's basis vectors, but overwrites the ONE
// vector for the direction just traveled in with the actually-observed step
// (divided by however many lattice steps were skipped) — this is what makes
// the mesh locally adaptive to perspective/curvature without any global
// parametric model, unlike the old regional-gradient approach (see the
// removed detectLocalGrid).

import type { JunctionType } from './cornerdetect.ts';

export interface RawJunction { x: number; y: number; type: JunctionType; }

export interface MeshNode {
  x: number; y: number;
  type: JunctionType;
  row: number; col: number;
}

export interface Mesh {
  nodes: MeshNode[];
  byCoord: Map<string, MeshNode>; // key `${row},${col}`
}

type Vec = [number, number];
const rotate = (v: Vec, theta: number): Vec => {
  const c = Math.cos(theta), s = Math.sin(theta);
  return [v[0] * c - v[1] * s, v[0] * s + v[1] * c];
};
const key = (row: number, col: number) => `${row},${col}`;

// direction: dRow/dCol in mesh space, and whether it's the "primary" (right/
// down) or "negated" (left/up) sense of that axis — used to decide which
// basis vector a step updates.
const DIRECTIONS = [
  { dRow: 0, dCol: 1, axis: 'col' as const, sign: 1 },
  { dRow: 0, dCol: -1, axis: 'col' as const, sign: -1 },
  { dRow: 1, dCol: 0, axis: 'row' as const, sign: 1 },
  { dRow: -1, dCol: 0, axis: 'row' as const, sign: -1 },
];

interface BuildNode {
  junctionIndex: number;
  row: number; col: number;
  colVec: Vec; // pixel offset for +1 col step, local to this node
  rowVec: Vec; // pixel offset for +1 row step, local to this node
}

// tolerance is a fraction of the predicted step's own magnitude — how far a
// real detected junction is allowed to sit from the naive prediction and
// still count as a match (perspective + estimation error both eat into
// this; empirically tuned in scripts/test-mesh.ts).
export function buildMesh(
  junctions: RawJunction[],
  seedX: number, seedY: number,
  pitchX: number, pitchY: number, theta: number,
  maxSkip = 3, tolerance = 0.35,
): Mesh {
  if (junctions.length === 0) return { nodes: [], byCoord: new Map() };

  let seedIdx = 0, bestDist = Infinity;
  for (let i = 0; i < junctions.length; i++) {
    const d = (junctions[i].x - seedX) ** 2 + (junctions[i].y - seedY) ** 2;
    if (d < bestDist) { bestDist = d; seedIdx = i; }
  }

  const colVec0 = rotate([pitchX, 0], theta);
  const rowVec0 = rotate([0, pitchY], theta);

  const byCoord = new Map<string, BuildNode>();
  const usedJunctions = new Set<number>([seedIdx]);
  const seed: BuildNode = { junctionIndex: seedIdx, row: 0, col: 0, colVec: colVec0, rowVec: rowVec0 };
  byCoord.set(key(0, 0), seed);
  const queue: BuildNode[] = [seed];

  while (queue.length > 0) {
    const node = queue.shift()!;
    const nx = junctions[node.junctionIndex].x, ny = junctions[node.junctionIndex].y;

    for (const dir of DIRECTIONS) {
      const targetRow = node.row + dir.dRow, targetCol = node.col + dir.dCol;
      if (byCoord.has(key(targetRow, targetCol))) continue;
      const stepVec: Vec = dir.axis === 'col'
        ? [node.colVec[0] * dir.sign, node.colVec[1] * dir.sign]
        : [node.rowVec[0] * dir.sign, node.rowVec[1] * dir.sign];

      // Decompose each unclaimed junction's offset from `node` into
      // along-axis (parallel) and cross-axis (perpendicular) components
      // relative to the search direction, rather than testing candidate
      // skip distances against isotropic tolerance windows that grow with
      // skip — those windows overlap once they exceed half a cell width,
      // letting a real skip-4 neighbor get matched (and mislabeled) as
      // skip-3 whenever it happens to land inside skip 3's generous window
      // (see scripts/test-mesh.ts's coordMismatches, caught by exactly
      // this). Instead: perpendicular drift gets a FIXED bound (shouldn't
      // grow with distance for a locally-consistent grid), and each
      // candidate's own skip count is read off directly by rounding its
      // parallel distance to the nearest whole step — with tolerance kept
      // under 0.5 steps, this can never ambiguously straddle two skip
      // counts, unlike growing search windows.
      const stepUnitLen = Math.hypot(stepVec[0], stepVec[1]);
      const ux = stepVec[0] / stepUnitLen, uy = stepVec[1] / stepUnitLen; // unit vector along search direction
      const perpTol = tolerance * stepUnitLen;

      let found = -1, foundSkip = 1, bestScore = Infinity;
      for (let i = 0; i < junctions.length; i++) {
        if (usedJunctions.has(i)) continue;
        const dx = junctions[i].x - nx, dy = junctions[i].y - ny;
        const parallel = dx * ux + dy * uy;
        const perp = -dx * uy + dy * ux;
        if (Math.abs(perp) >= perpTol) continue;
        const skipFloat = parallel / stepUnitLen;
        const skip = Math.round(skipFloat);
        if (skip < 1 || skip > maxSkip) continue;
        const parallelErr = Math.abs(skipFloat - skip); // fraction of one step
        if (parallelErr >= tolerance) continue;
        const score = skip * 1000 + parallelErr; // smallest skip wins outright; error only breaks ties within the same skip
        if (score < bestScore) { bestScore = score; found = i; foundSkip = skip; }
      }
      if (found === -1) continue;

      const jx = junctions[found].x, jy = junctions[found].y;
      const observed: Vec = [(jx - nx) / foundSkip, (jy - ny) / foundSkip];
      const targetRowFinal = node.row + dir.dRow * foundSkip, targetColFinal = node.col + dir.dCol * foundSkip;
      if (byCoord.has(key(targetRowFinal, targetColFinal))) { continue; } // loop closure landed on an existing node — leave it, don't overwrite

      const newNode: BuildNode = {
        junctionIndex: found,
        row: targetRowFinal, col: targetColFinal,
        colVec: dir.axis === 'col' ? [observed[0] * dir.sign, observed[1] * dir.sign] : node.colVec,
        rowVec: dir.axis === 'row' ? [observed[0] * dir.sign, observed[1] * dir.sign] : node.rowVec,
      };
      byCoord.set(key(newNode.row, newNode.col), newNode);
      usedJunctions.add(found);
      queue.push(newNode);
    }
  }

  const nodes: MeshNode[] = [];
  const outByCoord = new Map<string, MeshNode>();
  for (const bn of byCoord.values()) {
    const j = junctions[bn.junctionIndex];
    const node: MeshNode = { x: j.x, y: j.y, type: j.type, row: bn.row, col: bn.col };
    nodes.push(node);
    outByCoord.set(key(bn.row, bn.col), node);
  }
  return { nodes, byCoord: outByCoord };
}

// Post-hoc consistency check: buildMesh's BFS commits a node's (row,col) the
// moment one search from one parent succeeds, with no cross-check against
// other paths that might reach the same physical point — fine at low
// position noise (real detection is ~0.7-1.2px in isolated testing, see
// scripts/test-subpixel.ts), but scripts/test-build-mesh-exact.ts's noise
// sweep found row/col mislabeling starts appearing above ~2px, which real
// captures can plausibly exceed.
//
// For each node, predicts its position from whichever of its neighbors are
// available — the midpoint of an opposite pair (up+down, or left+right),
// or parallelogram completion from a diagonal-adjacent triple (the same
// principle sampleFromMesh's diagonal-pair cell sampling already relies
// on) — and prunes it if it disagrees with the MEDIAN of those predictions
// by more than toleranceFraction of the expected cell size. A node with no
// available predictions (too few neighbors to check) is kept rather than
// pruned — no positive evidence of being wrong, and pruning purely for low
// connectivity would unfairly target legitimate mesh-edge nodes.
export function pruneInconsistentNodes(mesh: Mesh, cellSize: number, toleranceFraction = 0.3): Mesh {
  const { byCoord } = mesh;
  const tolerance = cellSize * toleranceFraction;
  const suspects = new Set<MeshNode>();

  for (const node of mesh.nodes) {
    const up = byCoord.get(key(node.row - 1, node.col));
    const down = byCoord.get(key(node.row + 1, node.col));
    const left = byCoord.get(key(node.row, node.col - 1));
    const right = byCoord.get(key(node.row, node.col + 1));

    const predictions: Vec[] = [];
    if (up && down) predictions.push([(up.x + down.x) / 2, (up.y + down.y) / 2]);
    if (left && right) predictions.push([(left.x + right.x) / 2, (left.y + right.y) / 2]);
    for (const [a, b, diag] of [
      [up, left, byCoord.get(key(node.row - 1, node.col - 1))],
      [up, right, byCoord.get(key(node.row - 1, node.col + 1))],
      [down, left, byCoord.get(key(node.row + 1, node.col - 1))],
      [down, right, byCoord.get(key(node.row + 1, node.col + 1))],
    ] as const) {
      if (a && b && diag) predictions.push([a.x + b.x - diag.x, a.y + b.y - diag.y]);
    }

    if (predictions.length === 0) continue; // unverifiable — keep, not evidence of a problem
    const errors = predictions.map(([px, py]) => Math.hypot(px - node.x, py - node.y)).sort((a, b) => a - b);
    const medianError = errors[Math.floor(errors.length / 2)];
    if (medianError > tolerance) suspects.add(node);
  }

  if (suspects.size === 0) return mesh;
  const nodes = mesh.nodes.filter(n => !suspects.has(n));
  const outByCoord = new Map<string, MeshNode>();
  for (const n of nodes) outByCoord.set(key(n.row, n.col), n);
  return { nodes, byCoord: outByCoord };
}
