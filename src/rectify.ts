// Global rectify-and-assign: given ANY working homography (however roughly
// it was seeded — see homography.ts's fitHomographyFromVPAndPatch), assigns
// EVERY detected junction a (row,col) address independently by mapping it
// into lattice space via H^-1 and rounding, instead of by walking/
// propagating from neighbor to neighbor the way mesh.ts's buildMesh does.
//
// This is what actually fixes the "coherent region drift" failure mode
// buildMesh's local-only consistency checking could never fully solve (see
// mesh.ts's pruneInconsistentNodes docs): here, a node's label comes from
// ONE global model checked independently per-node, so a single bad junction
// can never corrupt its neighbors' labels the way one bad BFS hop could —
// there's no neighbor chain for the error to propagate through anymore.
//
// The seed homography doesn't need to be very accurate: each round keeps
// only junctions whose rectified position lands close to an integer, then
// refits H from that (now much larger and spatially spread) inlier set via
// full unconstrained DLT. scripts/test-homography.ts proved spread-out
// correspondences give a well-conditioned, accurate fit (unlike a small
// clustered local patch) — so this loop bootstraps a rough local-patch-
// quality H into a much better whole-image H within a few rounds, PROVIDED
// the seed is good enough to produce a first round of real inliers at all.
//
// KNOWN LIMITATION (found via scripts/test-homography-decode.ts, not yet
// root-caused): end-to-end decode is still 0/8 correct at every tilt tested.
// In a controlled axis-aligned case (tilt=0, azimuth=0, roll=0) the fitted
// H's implied scale matched truth closely (~7.9-8px/cell vs the real 8px)
// and the mesh's row/col span matched the true field of view. But across
// random azimuth/roll trials, the final mesh sometimes spans a MUCH wider
// (row,col) range than the true field of view allows (e.g. 76-83 columns
// for a capture that can only see ~37 cells) while inlier COUNT stays
// stable round-over-round (not a runaway-growing refit) — meaning a chunk
// of junctions converge on a self-consistent but wrong absolute labeling
// that survives the per-round residual check. This is a different flavor
// of the same "coherent drift" problem buildMesh had, likely related to the
// pattern's own periodicity making a slightly-wrong homography scale/shear
// still satisfy the round-to-nearest-integer tolerance for many points at
// once. Roll appears to be a factor (the one clean case tested had
// roll=azimuth=0) but this isn't confirmed. Time-boxed and left unresolved
// per the decision to move to live overlay testing instead of continuing to
// debug against synthetic data.

import type { Mat3, VPLike, PointCorrespondence } from './homography.ts';
import { fitHomographyDLT, fitHomographyFromVPAndPatch, applyHomography, invertHomography } from './homography.ts';
import type { RawJunction, Mesh, MeshNode } from './mesh.ts';
import { buildMesh } from './mesh.ts';

export interface RectifyResult { mesh: Mesh; H: Mat3; rounds: number; }

export function rectifyAndAssign(
  junctions: RawJunction[],
  initialH: Mat3,
  toleranceFraction = 0.35,
  maxRounds = 4,
): RectifyResult | null {
  let H = initialH;
  let lastGood: RectifyResult | null = null;

  for (let round = 0; round < maxRounds; round++) {
    const Hinv = invertHomography(H);
    if (!Hinv) break;

    // Rectify every junction; dedupe by keeping whichever candidate lands
    // closest to its assigned integer lattice coordinate (two junctions
    // can round to the same coord under a still-rough H).
    const best = new Map<string, { junction: RawJunction; u: number; v: number; residual: number }>();
    for (const j of junctions) {
      const uv = applyHomography(Hinv, j.x, j.y);
      if (!uv) continue;
      const [u, v] = uv;
      const ru = Math.round(u), rv = Math.round(v);
      const residual = Math.hypot(u - ru, v - rv);
      if (residual >= toleranceFraction) continue;
      const k = `${ru},${rv}`;
      const existing = best.get(k);
      if (!existing || residual < existing.residual) best.set(k, { junction: j, u: ru, v: rv, residual });
    }
    if (best.size < 4) break;

    const nodes: MeshNode[] = [];
    const byCoord = new Map<string, MeshNode>();
    for (const { junction, u, v } of best.values()) {
      const node: MeshNode = { x: junction.x, y: junction.y, type: junction.type, row: u, col: v };
      nodes.push(node);
      byCoord.set(`${u},${v}`, node);
    }
    lastGood = { mesh: { nodes, byCoord }, H, rounds: round + 1 };

    if (round < maxRounds - 1) {
      const correspondences = Array.from(best.values()).map(({ junction, u, v }) => ({ u, v, x: junction.x, y: junction.y }));
      const refit = fitHomographyDLT(correspondences);
      if (!refit) break; // keep lastGood rather than fail outright
      H = refit;
    }
  }

  return lastGood;
}

// Top-level entry point tying the whole homography-based strategy together
// (parallel role to mesh.ts's buildMesh+pruneInconsistentNodes for the old
// walk-based strategy):
//   1. A small LOCAL patch via the existing buildMesh (short-range walking
//      is exactly what it's good at — see homography.ts's docs on why a
//      small clustered patch alone can't extrapolate, but CAN reliably
//      supply the "scale near the seed" info the VP columns can't).
//   2. Decide which of the two vanishing points is the row axis vs. the
//      column axis by comparing each VP's line direction to the local
//      patch's own observed step directions — no ground truth available at
//      runtime, so this has to be inferred, not looked up.
//   3. Seed a homography from VP directions + local patch (fitHomographyFromVPAndPatch).
//   4. Globally rectify-and-assign every detected junction against it,
//      refitting a few rounds (rectifyAndAssign).
export interface HomographyMeshResult {
  mesh: Mesh;
  H: Mat3;
  localPatchSize: number;
  seedRounds: number;
}

function vpLineDirection(vp: VPLike, origin: { x: number; y: number }): [number, number] {
  if (vp.finite) return [vp.x - origin.x, vp.y - origin.y];
  return [Math.cos(vp.angle), Math.sin(vp.angle)];
}

// |cos| of the angle between two directions — absolute value because both a
// VP's line direction (mod PI) and our empirical step-direction hint have
// no meaningful sign, only an axis.
function axisSimilarity(dir: [number, number], hint: [number, number]): number {
  const nd = Math.hypot(dir[0], dir[1]) || 1, nh = Math.hypot(hint[0], hint[1]) || 1;
  return Math.abs(dir[0] * hint[0] + dir[1] * hint[1]) / (nd * nh);
}

export function buildMeshViaHomography(
  junctions: RawJunction[],
  seedX: number, seedY: number,
  pitchX: number, pitchY: number, theta: number,
  vp1: VPLike, vp2: VPLike,
  rectifyTolerance = 0.35, rectifyRounds = 4,
): HomographyMeshResult | null {
  const localMesh = buildMesh(junctions, seedX, seedY, pitchX, pitchY, theta, 3, 0.35);
  const origin = localMesh.byCoord.get('0,0');
  if (!origin || localMesh.nodes.length < 4) return null;

  const rotate = (x: number, y: number): [number, number] => {
    const c = Math.cos(theta), s = Math.sin(theta);
    return [x * c - y * s, x * s + y * c];
  };
  const rowNeighbor = localMesh.byCoord.get('1,0');
  const colNeighbor = localMesh.byCoord.get('0,1');
  const rowHint: [number, number] = rowNeighbor
    ? [rowNeighbor.x - origin.x, rowNeighbor.y - origin.y]
    : rotate(0, pitchY);
  const colHint: [number, number] = colNeighbor
    ? [colNeighbor.x - origin.x, colNeighbor.y - origin.y]
    : rotate(pitchX, 0);

  const dir1 = vpLineDirection(vp1, origin), dir2 = vpLineDirection(vp2, origin);
  const scoreAsIs = axisSimilarity(dir1, rowHint) + axisSimilarity(dir2, colHint);
  const scoreSwapped = axisSimilarity(dir1, colHint) + axisSimilarity(dir2, rowHint);
  const [vpRow, vpCol] = scoreAsIs >= scoreSwapped ? [vp1, vp2] : [vp2, vp1];

  const patch: PointCorrespondence[] = localMesh.nodes.map(n => ({ u: n.row, v: n.col, x: n.x, y: n.y }));
  const originCorr: PointCorrespondence = { u: 0, v: 0, x: origin.x, y: origin.y };
  const seedH = fitHomographyFromVPAndPatch(vpRow, vpCol, originCorr, patch);
  if (!seedH) return null;

  const rectified = rectifyAndAssign(junctions, seedH, rectifyTolerance, rectifyRounds);
  if (!rectified) return null;

  return { mesh: rectified.mesh, H: rectified.H, localPatchSize: localMesh.nodes.length, seedRounds: rectified.rounds };
}
