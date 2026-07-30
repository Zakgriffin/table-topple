import * as THREE from 'three';
import { GradientField } from '../types.ts';

// ── "Guided bucket fill" line-segment join (pure) ────────────────────────
//
// A from-scratch, simplified take on LSD (Line Segment Detector)'s region-
// growing step: flood-fill outward from strong seed pixels, absorbing any
// connected pixel whose gradient direction stays within toleranceDeg of the
// region's own running-average direction. Deliberately skips LSD's
// rectangle-fit + NFA statistical validation -- pipeline/lsdSegments.ts is
// the separate, full from-scratch pipeline that includes those, and (via
// its own lsdRectanglesToBucketFillShape) is what pipeline/votes.ts's
// computeGradient2x2Composites and pipeline/bucketFillJoin.ts's join walk
// actually run in production now. This function is NOT currently called
// from anywhere live -- kept as a reference/comparison implementation.
//
// Orientation comparisons use the same double-angle (mod PI) convention as
// pipeline/gradientField.ts's computeGradientAgreementField and
// pipeline/tangentWalk.ts's guided walk -- gradient direction is undirected
// (a black-to-white edge and a white-to-black edge along the same line have
// opposite raw angles but represent the same line), so folding by 2*theta
// before averaging/comparing treats them as identical, matching every other
// direction-averaging step already in this codebase.

export interface BucketFillSegment {
  count: number; // pixel "mass" -- every member pixel counts equally, see cx/cy
  cx: number; cy: number; // center of mass: plain mean of member pixel (x,y), UNweighted by magnitude -- kept around, just not currently visualized
  avgFx: number; avgFy: number; // average gradient vector, sign-resolved (see below) -- NOT normalized, its length reflects how tightly the region's directions actually agree

  // The segment's two "endpoints" -- projected along the TANGENT (avgFx/
  // avgFy rotated 90 degrees), not the gradient itself: the gradient axis
  // only spans the edge's 1-2px WIDTH, so its extremes wouldn't be
  // meaningful endpoints of a LINE; the tangent is the axis the segment
  // actually extends along. Computed in a post-pass AFTER the region has
  // finished growing, against the region's FINAL avgFx/avgFy -- every member
  // pixel is projected onto that one fixed, stable axis and compared, rather
  // than each pixel being judged against whatever the running axis happened
  // to be at the moment it was absorbed (which could drift slightly as the
  // region grew).
  endAlongX: number; endAlongY: number; // farthest pixel in the +tangent direction
  endAgainstX: number; endAgainstY: number; // farthest pixel in the -tangent direction

  // "How line-y is this" confidence in [0, 1], the SAME scale and meaning
  // lsdSegments.ts's own lineScore carries (see that file's
  // nfaLog10ToLineScore) -- bucketFillJoin.ts's join walk reads this
  // uniformly regardless of which producer built the segment, so a region's
  // trust going INTO a merge decision reflects how well-formed it actually
  // was, not just its raw geometry. This producer has no NFA statistic to
  // draw on, so it uses the region's own circular-mean resultant length
  // (sumCos/sumSin's magnitude, already tracked while growing) divided by
  // member count -- 1.0 means every member pixel's direction agreed
  // exactly, 0 means they averaged out to nothing.
  lineScore: number;
}

export function computeBucketFillRegions(
  field: GradientField, toleranceDeg: number, magnitudeThreshold: number,
  // Caps each region's growth to at most this many BFS hops from its own
  // seed (0 = unbounded, i.e. today's existing behavior -- unlike
  // pipeline/bucketFillJoin.ts's computeJoinWalk, where 0 steps IS the
  // sane default/off state, "0 growth allowed" would make this a no-op by
  // default, so the sentinel direction is deliberately flipped here). Lets
  // a debug overlay scrub through the flood fill's growth ring by ring,
  // same idea as computeJoinWalk's own numSteps.
  maxSteps: number = 0,
): { regionId: Int32Array; segments: BucketFillSegment[] } {
  const { fx, fy, w, h } = field;
  const n = w * h;
  const mag = new Float64Array(n);
  for (let i = 0; i < n; i++) mag[i] = Math.hypot(fx[i], fy[i]);

  // magnitudeThreshold is a hard floor on which pixels ever participate at
  // ALL -- anything at or below this threshold is excluded from both
  // seeding AND absorption, i.e. it can never be part of any segment, full
  // stop. Every pixel above it is eligible to FOUND a new region (used to
  // also require passing a seedEligible top-N% magnitude band, but every
  // caller always passed (0, 100) -- no actual cutoff -- so that parameter
  // was pure dead weight, removed); once a region exists, growth/absorption
  // is open to any connected pixel above magnitudeThreshold with a
  // consistent orientation, same as LSD itself (which only prioritizes
  // strong pixels as seeds via magnitude-descending order, and never
  // additionally restricts absorption by magnitude).
  const order: number[] = [];
  for (let i = 0; i < n; i++) if (mag[i] > magnitudeThreshold) order.push(i);
  order.sort((a, b) => mag[b] - mag[a]);

  const cosTol = Math.cos(2 * THREE.MathUtils.degToRad(toleranceDeg));
  const regionId = new Int32Array(n).fill(-1);
  // BFS depth (hops from this pixel's own seed) -- only meaningful once
  // regionId[i] !== -1, only ever read/written when maxSteps > 0.
  const depth = new Int32Array(n);
  // One shared queue, reused (never reset) across every region -- each pixel
  // is ever pushed at most once in total (it's marked claimed the instant
  // it's pushed), so a monotonically increasing tail pointer across the
  // WHOLE call is safe and avoids either per-region allocation or an O(n)
  // Array.shift() per pop.
  const queue = new Int32Array(n);
  let qTail = 0;
  const segments: BucketFillSegment[] = [];

  for (const seed of order) {
    if (regionId[seed] !== -1) continue; // already absorbed by an earlier (stronger) region
    const id = segments.length;
    regionId[seed] = id;
    depth[seed] = 0;
    const seedTheta = Math.atan2(fy[seed], fx[seed]);
    let sumCos = Math.cos(2 * seedTheta), sumSin = Math.sin(2 * seedTheta);
    const seedPx = seed % w, seedPy = (seed / w) | 0;
    let sumX = seedPx, sumY = seedPy;
    let sumFx = fx[seed], sumFy = fy[seed];
    let count = 1;

    const regionQueueStart = qTail; // queue[regionQueueStart..qTail) is exactly this region's member pixels, once the loop below finishes
    let qHead = qTail;
    queue[qTail++] = seed;
    while (qHead < qTail) {
      const p = queue[qHead++];
      const pDepth = depth[p];
      if (maxSteps > 0 && pDepth >= maxSteps) continue; // this ring is already at the cap -- absorbed, but doesn't expand further
      const px = p % w, py = (p / w) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = px + dx, ny = py + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const ni = ny * w + nx;
          if (regionId[ni] !== -1 || mag[ni] <= magnitudeThreshold) continue;
          const theta = Math.atan2(fy[ni], fx[ni]);
          const c2 = Math.cos(2 * theta), s2 = Math.sin(2 * theta);
          const avgLen = Math.hypot(sumCos, sumSin);
          const cosDeviation = avgLen > 0 ? (c2 * sumCos + s2 * sumSin) / avgLen : 1;
          if (cosDeviation < cosTol) continue;
          regionId[ni] = id;
          depth[ni] = pDepth + 1;
          sumCos += c2; sumSin += s2;

          // Gradient direction is axial (theta and theta+PI are the SAME
          // edge orientation, see this file's header) -- a plain sum of raw
          // fx/fy would let opposite-signed-but-parallel edges (e.g. the two
          // sides of one grid-line stripe) cancel each other out. Resolve
          // each pixel's sign against the region's running RAW vector sum
          // (not the double-angle one above, which is only for the
          // tolerance check) before adding it -- the standard trick for
          // averaging axial data online.
          const nfx = fx[ni], nfy = fy[ni];
          if (nfx * sumFx + nfy * sumFy < 0) { sumFx -= nfx; sumFy -= nfy; } else { sumFx += nfx; sumFy += nfy; }
          sumX += nx; sumY += ny;
          count++;

          queue[qTail++] = ni;
        }
      }
    }

    // Post-pass: now that the region is finished (sumFx/sumFy are final),
    // project every member pixel onto the ONE stable tangent axis and take
    // the true min/max -- see BucketFillSegment's own comment.
    let endAlongX = seedPx, endAlongY = seedPy, maxProj = 0;
    let endAgainstX = seedPx, endAgainstY = seedPy, minProj = 0;
    const tanLen = Math.hypot(sumFx, sumFy);
    if (tanLen > 0) {
      const tanX = -sumFy / tanLen, tanY = sumFx / tanLen;
      for (let qi = regionQueueStart; qi < qTail; qi++) {
        const p = queue[qi];
        const px = p % w, py = (p / w) | 0;
        const proj = px * tanX + py * tanY;
        if (qi === regionQueueStart || proj > maxProj) { maxProj = proj; endAlongX = px; endAlongY = py; }
        if (qi === regionQueueStart || proj < minProj) { minProj = proj; endAgainstX = px; endAgainstY = py; }
      }
    }

    segments.push({
      count, cx: sumX / count, cy: sumY / count, avgFx: sumFx / count, avgFy: sumFy / count,
      endAlongX, endAlongY, endAgainstX, endAgainstY,
      lineScore: Math.hypot(sumCos, sumSin) / count,
    });
  }
  return { regionId, segments };
}

// Pixel-space distance between a segment's two tracked endpoints -- the
// shared length metric every downstream consumer (the base raster below,
// the join walk, and the experimental segment-vote generator) filters
// short/unreliable segments by. Plain Euclidean pixel distance, not the
// projected arc length used for vote weighting elsewhere (pipeline/
// votes.ts's computeSegmentVotes) -- arc length needs a ray-cast (camera
// quat/FOV/aspect), which the join walk has no reason to know about
// otherwise, so this stays in the one space every consumer already has.
export function segmentLength(seg: BucketFillSegment): number {
  return Math.hypot(seg.endAlongX - seg.endAgainstX, seg.endAlongY - seg.endAgainstY);
}

