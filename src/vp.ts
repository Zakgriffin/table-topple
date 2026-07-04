// Level 2 of the line-based rectification redesign: given a set of detected
// lines (src/lines.ts), finds the point where they all (approximately) meet
// — their vanishing point — via the point<->line duality: a line l=(a,b,c)
// (a=cos theta, b=sin theta, c=-rho, so a point p=(x,y,w) lies ON the line
// exactly when a*x+b*y+c*w=0) and a PENCIL of lines through a common point p
// satisfies that same equation for every line in the pencil, with p now
// playing the role individual pixel coordinates played in Level 1. Finding
// the common point of many noisy lines is exactly the same "solve a bunch of
// noisy linear constraints" problem as finding one line from many noisy
// pixels — just one level up the same duality.
//
// Representing p as a homogeneous 3-vector (rather than plain (x,y)) is what
// lets a genuinely-at-infinity vanishing point (near-fronto-parallel view,
// truly parallel lines) fall out with no special case: the least-squares
// solution just returns a vector with w~0, which is still a perfectly good
// direction (x,y) — no separate "did this converge to infinity" branch is
// needed HERE (branching only becomes necessary later, when a caller wants
// to actually use the point's finite pixel coordinates).

import type { LineCandidate } from './lines.ts';
import { smallestEigenvector } from './linalg.ts';

export interface VanishingPoint {
  x: number; y: number; w: number; // homogeneous, ABSOLUTE image pixel coords (not center-relative)
}

export function vpIsFinite(vp: VanishingPoint, eps = 1e-6): boolean {
  return Math.abs(vp.w) > eps;
}

export function vpToPoint(vp: VanishingPoint): { x: number; y: number } {
  return { x: vp.x / vp.w, y: vp.y / vp.w };
}

// LineCandidate.rho is measured relative to the image CENTER (see lines.ts),
// so a line's ABSOLUTE-coordinate equation a*X + b*Y + c = 0 needs c shifted
// back by the center offset.
export function toAbsoluteLine(line: LineCandidate, cx: number, cy: number): [number, number, number] {
  const a = Math.cos(line.theta), b = Math.sin(line.theta);
  const c = -(line.rho + a * cx + b * cy);
  return [a, b, c];
}

// How far a line is from actually passing through (or being parallel to,
// for a far/at-infinity candidate) a hypothesized vanishing point, in
// PIXEL-equivalent units — used to count RANSAC inliers when splitting lines
// into families (see splitIntoTwoFamilies). Two regimes, because a single
// algebraic homogeneous residual isn't uniformly meaningful in pixel units
// across wildly different vp distances:
//   - finite (and not absurdly far outside the image): true perpendicular
//     pixel distance from the point to the line.
//   - at/near infinity: compare the line's own direction to the VP's
//     implied direction instead (a distance doesn't mean anything for a
//     point at infinity), then scale the angular error by the image
//     diagonal to get a roughly comparable "equivalent pixel deviation at
//     the edge of the frame" — a calibrated approximation, not an exact
//     metric, but good enough to threshold consistently against the finite
//     case's pixel threshold.
export function lineResidualPx(line: LineCandidate, vp: VanishingPoint, w: number, h: number): number {
  const cx = w / 2, cy = h / 2;
  const [a, b, c] = toAbsoluteLine(line, cx, cy);
  const diag = Math.hypot(w, h);
  const farThreshold = diag * 50;
  if (Math.abs(vp.w) > 1e-9 && Math.abs(vp.x / vp.w) < farThreshold && Math.abs(vp.y / vp.w) < farThreshold) {
    const px = vp.x / vp.w, py = vp.y / vp.w;
    return Math.abs(a * px + b * py + c);
  }
  const dirLen = Math.hypot(vp.x, vp.y);
  const dx = vp.x / dirLen, dy = vp.y / dirLen;
  const tx = -b, ty = a; // line tangent, perpendicular to its normal (a,b)
  const dot = Math.min(1, Math.abs(tx * dx + ty * dy));
  const angle = Math.acos(dot); // undirected angle between tangent and vp direction
  return angle * diag;
}

// Finds the weighted-least-squares common point of a set of lines: builds
// M = sum_j weight_j * l_j * l_j^T (each line's outer product with itself,
// weighted by its Hough vote mass) and returns the eigenvector of M's
// SMALLEST eigenvalue — the direction in which the lines, as a whole, are
// least inconsistent with passing through a shared point. This is exactly
// the same shape of computation as the structure tensor's smallest
// eigenvalue in cornerdetect.ts, just with "lines" standing in for
// "gradients" — errors are measured as each line's algebraic residual
// a*x+b*y+c*w at the candidate point, not a spatial distance, which is what
// keeps this well-defined even when the true point is at infinity.
export function estimateVanishingPoint(lines: LineCandidate[], w: number, h: number): VanishingPoint {
  const cx = w / 2, cy = h / 2;
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const line of lines) {
    const l = toAbsoluteLine(line, cx, cy);
    const wgt = line.weight;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[i][j] += wgt * l[i] * l[j];
  }
  const v = smallestEigenvector(M);
  return { x: v[0], y: v[1], w: v[2] };
}

export interface LineFamily { vp: VanishingPoint; lines: LineCandidate[]; }

// Intersection of two lines (each in absolute-coordinate a*X+b*Y+c=0 form,
// see toAbsoluteLine) via the homogeneous cross product — gives the
// point-at-infinity case for free when the lines are parallel (the
// w-component comes out exactly 0, no special case needed), same theme as
// estimateVanishingPoint.
export function crossLines(l1: [number, number, number], l2: [number, number, number]): VanishingPoint {
  const [a1, b1, c1] = l1, [a2, b2, c2] = l2;
  return { x: b1 * c2 - c1 * b2, y: c1 * a2 - a1 * c2, w: a1 * b2 - b1 * a2 };
}

// Splits a mixed set of lines (e.g. every peak Level 1 found across a whole
// grid image, rows and columns together) into the two families that share a
// vanishing point, WITHOUT needing to know either VP up front — this is what
// replaces guessing the split from gradient angle alone (which breaks under
// perspective, see the discussion this design is based on: same-family
// lines fan out toward their VP, they don't share one angle).
//
// Two lines are enough to fully determine a candidate intersection point (a
// minimal RANSAC sample). With Level 1 typically producing only a few dozen
// line peaks on a clean synthetic image, trying EVERY pair exhaustively
// (rather than random RANSAC sampling) is cheap and more reliable at that
// size — but real camera video is textured/noisy in ways a clean synthetic
// render never is, and Level 1 can return hundreds of peaks on it. Exhaustive
// pairing is O(lines^2) candidates each scored in O(lines), i.e. O(lines^3)
// overall — fine at N~30, catastrophic (and exactly what froze the live app)
// at N~300+. maxCandidates caps which lines are used to GENERATE pairing
// hypotheses (the top-weighted ones — genuine grid lines should be among the
// strongest responses; weak/spurious peaks are unlikely to be needed as a
// pairing seed) without discarding any line from the final scoring/
// assignment pass below, which stays O(lines) per hypothesis regardless.
// extraLines is an optional second, lower-confidence candidate pool (e.g. a
// second findLinePeaks pass at a much lower vote threshold) that's checked
// against the two VPs during FINAL assignment only -- never used to seed or
// score the RANSAC hypothesis search above, which stays exactly as robust
// and cheap as before. This exists because a real camera's two line
// families are NOT always comparably strong: lighting, focus, or a camera's
// own directional sharpening can make one family's edges systematically
// weaker without there being anything wrong with the grid or the algorithm
// (confirmed via live-device testing to sometimes be a near-total imbalance,
// yet not reproducible from a clean synthetic capture at any roll angle —
// pointing at a real-capture artifact, not a detection bug). Once a
// systematically weak family's VP is known (even from just its strongest
// few peaks), MOST of its true members likely never even cleared Level 1's
// single global vote threshold in the first place — extraLines lets them
// back in without lowering that threshold for everyone (which would just
// flood the RANSAC seeding step with more noise instead).
export function splitIntoTwoFamilies(
  lines: LineCandidate[], w: number, h: number, inlierPx = 6, maxCandidates = 60,
  extraLines: LineCandidate[] = [],
): { familyA: LineFamily; familyB: LineFamily; unassigned: LineCandidate[] } {
  function scoreHypothesis(vp: VanishingPoint, exclude: Set<number>): { support: number; inliers: number[] } {
    let support = 0;
    const inliers: number[] = [];
    for (let i = 0; i < lines.length; i++) {
      if (exclude.has(i)) continue;
      if (lineResidualPx(lines[i], vp, w, h) < inlierPx) { support += lines[i].weight; inliers.push(i); }
    }
    return { support, inliers };
  }
  function refit(indices: number[]): VanishingPoint {
    return estimateVanishingPoint(indices.map(i => lines[i]), w, h);
  }

  const cx = w / 2, cy = h / 2;
  const abs = lines.map(l => toAbsoluteLine(l, cx, cy));
  const none = new Set<number>();
  const seedIndices = lines
    .map((l, i) => ({ i, weight: l.weight }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, maxCandidates)
    .map(s => s.i);

  let bestA: { vp: VanishingPoint; inliers: number[]; support: number } | null = null;
  for (const i of seedIndices) {
    for (const j of seedIndices) {
      if (j <= i) continue;
      const cand = crossLines(abs[i], abs[j]);
      if (Math.hypot(cand.x, cand.y, cand.w) < 1e-12) continue; // degenerate (near-identical lines)
      const { support, inliers } = scoreHypothesis(cand, none);
      if (!bestA || support > bestA.support) bestA = { vp: cand, inliers, support };
    }
  }
  if (!bestA) throw new Error('splitIntoTwoFamilies: fewer than 2 usable lines');

  const excludeA = new Set(bestA.inliers);
  let bestB: { vp: VanishingPoint; inliers: number[]; support: number } | null = null;
  for (const i of seedIndices) {
    if (excludeA.has(i)) continue;
    for (const j of seedIndices) {
      if (j <= i || excludeA.has(j)) continue;
      const cand = crossLines(abs[i], abs[j]);
      if (Math.hypot(cand.x, cand.y, cand.w) < 1e-12) continue;
      const { support, inliers } = scoreHypothesis(cand, excludeA);
      if (!bestB || support > bestB.support) bestB = { vp: cand, inliers, support };
    }
  }
  if (!bestB) throw new Error('splitIntoTwoFamilies: could not find a second family distinct from the first');

  // Refit both from their full inlier sets (weighted least squares beats a
  // single supporting pair), then do one reassignment pass in case
  // refitting shifted a VP enough for borderline lines to switch sides —
  // same 1-iteration Lloyd's-algorithm shape as the old per-pixel-gradient
  // vanishing.ts attempted, but on a much smaller and cleaner set of inputs
  // (dozens of line peaks instead of millions of noisy pixel gradients), so
  // it's worth re-trying here even though it didn't help there.
  let vpA = refit(bestA.inliers);
  let vpB = refit(bestB.inliers);

  // Final assignment scans `lines` PLUS extraLines -- the low-confidence
  // rescue pool only ever gets a chance here, against VPs already
  // established from the strong pool, never influencing which VPs get
  // proposed in the first place.
  const assignedA: number[] = [], assignedB: number[] = [], unassigned: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    const dA = lineResidualPx(lines[i], vpA, w, h);
    const dB = lineResidualPx(lines[i], vpB, w, h);
    if (dA >= inlierPx * 2 && dB >= inlierPx * 2) { unassigned.push(i); continue; }
    if (dA <= dB) assignedA.push(i); else assignedB.push(i);
  }
  const rescuedA: LineCandidate[] = [], rescuedB: LineCandidate[] = [];
  for (const line of extraLines) {
    const dA = lineResidualPx(line, vpA, w, h);
    const dB = lineResidualPx(line, vpB, w, h);
    if (dA >= inlierPx * 2 && dB >= inlierPx * 2) continue; // still doesn't belong to either -- drop, not "unassigned" (it never cleared Level 1 as a confident detection at all)
    if (dA <= dB) rescuedA.push(line); else rescuedB.push(line);
  }

  const finalA = [...assignedA.map(i => lines[i]), ...rescuedA];
  const finalB = [...assignedB.map(i => lines[i]), ...rescuedB];
  if (finalA.length >= 2) vpA = estimateVanishingPoint(finalA, w, h);
  if (finalB.length >= 2) vpB = estimateVanishingPoint(finalB, w, h);

  return {
    familyA: { vp: vpA, lines: finalA },
    familyB: { vp: vpB, lines: finalB },
    unassigned: unassigned.map(i => lines[i]),
  };
}
