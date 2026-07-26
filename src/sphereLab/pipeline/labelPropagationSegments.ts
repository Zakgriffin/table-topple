import * as THREE from 'three';
import { GradientField } from '../types.ts';
import { BucketFillSegment } from './bucketFillSegments.ts';
import { computeGradientLocalMaxima } from './gradientField.ts';

// ── "Label propagation" flood fill (pure) ────────────────────────────────
//
// An alternative to bucketFillSegments.ts's computeBucketFillRegions, built
// to eventually run as a GPU compute shader: instead of one global magnitude-
// sorted queue processed serially (each region claiming pixels in strict
// priority order), this runs a fixed number of SYNCHRONOUS ROUNDS. Every
// round reads one shared "current labels" buffer (frozen for the whole
// round) and produces a brand-new "next labels" buffer -- every pixel's
// update depends only on its own 3x3 neighborhood of that ONE frozen input,
// never on another pixel's update from the SAME round, so a round is
// trivially parallel (one compute dispatch = one round) with no cross-thread
// ordering dependency, unlike the BFS version's shared queue.
//
// Seeds are exactly the 'gradient2x2LocalMax' field-view pixels (see
// gradientField.ts's computeGradientLocalMaxima) -- sparse, one-per-ridge,
// and (like that field view) already gated by magnitudeThreshold, so no
// separate seed-eligibility pass is needed here.
//
// Each seed's label IS its own pixel index (not a dense 0..k id) -- avoids
// needing a seed->id table during the parallel rounds; segments/regionId
// remap down to dense ids only in the final (inherently serial, same as
// computeBucketFillRegions's own post-pass) collection step.
//
// Compatibility test: once a round starts, every currently-labeled pixel's
// raw gradient direction is reduced into a per-label running average (a
// scatter-add over the WHOLE frozen buffer -- a segmented reduction, the one
// other GPU-friendly primitive this needs besides the propagate step
// itself), and an unlabeled pixel may adopt a neighbor's label only if its
// OWN raw direction stays within toleranceDeg of that neighbor's label's
// just-reduced average -- mirroring computeBucketFillRegions's own
// runningAverage-vs-candidate test, just recomputed once per round from
// scratch (a reduction) instead of updated incrementally as regions grow
// online. When more than one neighbor's label qualifies, the pixel joins
// whichever label currently has the most member pixels (ties broken by
// lower label id) -- a small deterministic bias toward "big regions keep
// growing" that stands in for the BFS version's magnitude-descending seed
// order, without needing any global sort.

export function computeLabelPropagationRegions(
  field: GradientField, toleranceDeg: number, magnitudeThreshold: number,
  // Doubles as "number of rounds" -- since a label can only ever spread one
  // hop (one ring) per round, this is the exact same "max hops from seed"
  // quantity as computeBucketFillRegions's own maxSteps, just enforced by
  // running fewer rounds instead of an early BFS cutoff. 0 = run to
  // convergence (capped at w+h rounds, the most hops a label could ever need
  // to cross the whole image).
  maxSteps: number = 0,
): { regionId: Int32Array; segments: BucketFillSegment[] } {
  const { fx, fy, w, h } = field;
  const n = w * h;
  const mag = new Float64Array(n);
  for (let i = 0; i < n; i++) mag[i] = Math.hypot(fx[i], fy[i]);

  let label = new Int32Array(n).fill(-1);
  const seeds = computeGradientLocalMaxima(field, magnitudeThreshold);
  for (let i = 0; i < n; i++) if (seeds[i] === 1) label[i] = i;

  const cosTol = Math.cos(2 * THREE.MathUtils.degToRad(toleranceDeg));
  const maxRounds = maxSteps > 0 ? maxSteps : w + h;

  // Reused across rounds, indexed directly by label value (a pixel index, so
  // bounded by n) -- reset with .fill(0) each round rather than
  // reallocated, since the reset itself is already the same O(n) order as
  // the rest of a round.
  const sumCos = new Float64Array(n), sumSin = new Float64Array(n), labelCount = new Int32Array(n);

  for (let round = 0; round < maxRounds; round++) {
    sumCos.fill(0); sumSin.fill(0); labelCount.fill(0);
    for (let i = 0; i < n; i++) {
      const lab = label[i];
      if (lab === -1) continue;
      const theta = Math.atan2(fy[i], fx[i]);
      sumCos[lab] += Math.cos(2 * theta); sumSin[lab] += Math.sin(2 * theta);
      labelCount[lab]++;
    }

    const nextLabel = label.slice(); // read ONLY from `label` (this round's frozen input) below -- never from nextLabel
    let changed = false;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (label[i] !== -1) continue; // already labeled -- stays, same monotonic-absorption rule as the BFS version
        const m = mag[i];
        if (m <= magnitudeThreshold) continue; // never eligible at all, same hard floor as computeBucketFillRegions
        const theta = Math.atan2(fy[i], fx[i]);
        const c2 = Math.cos(2 * theta), s2 = Math.sin(2 * theta);

        let bestLabel = -1, bestCount = -1;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            const nlab = label[ny * w + nx];
            if (nlab === -1) continue;
            const avgLen = Math.hypot(sumCos[nlab], sumSin[nlab]);
            const cosDeviation = avgLen > 0 ? (c2 * sumCos[nlab] + s2 * sumSin[nlab]) / avgLen : 1;
            if (cosDeviation < cosTol) continue;
            if (bestLabel === -1 || labelCount[nlab] > bestCount || (labelCount[nlab] === bestCount && nlab < bestLabel)) {
              bestLabel = nlab; bestCount = labelCount[nlab];
            }
          }
        }
        if (bestLabel !== -1) { nextLabel[i] = bestLabel; changed = true; }
      }
    }
    label = nextLabel;
    if (!changed) break; // converged -- no pixel anywhere adopted a label this round
  }

  // ── Collect: group by final (raw pixel-index) label, remap to dense ids ──
  // Inherently serial, same role as computeBucketFillRegions's own post-pass
  // over its finished queue -- this runs once after propagation settles, not
  // once per round, so it isn't part of what needs to parallelize.
  const membersByLabel = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const lab = label[i];
    if (lab === -1) continue;
    let list = membersByLabel.get(lab);
    if (!list) { list = []; membersByLabel.set(lab, list); }
    list.push(i);
  }
  const sortedLabels = Array.from(membersByLabel.keys()).sort((a, b) => a - b);

  const regionId = new Int32Array(n).fill(-1);
  const segments: BucketFillSegment[] = [];
  for (const lab of sortedLabels) {
    const members = membersByLabel.get(lab)!;
    const id = segments.length;
    for (const p of members) regionId[p] = id;

    // Sign-resolve each member's raw (fx,fy) against the SEED pixel's own
    // raw vector (a fixed, order-independent reference point) -- same
    // "axial data" reasoning as computeBucketFillRegions's online version,
    // just resolved in one flat pass instead of incrementally.
    const refFx = fx[lab], refFy = fy[lab];
    let sumFx = 0, sumFy = 0, sumX = 0, sumY = 0;
    for (const p of members) {
      const pfx = fx[p], pfy = fy[p];
      if (pfx * refFx + pfy * refFy < 0) { sumFx -= pfx; sumFy -= pfy; } else { sumFx += pfx; sumFy += pfy; }
      sumX += p % w; sumY += (p / w) | 0;
    }
    const count = members.length;

    // Post-pass endpoints: identical projection-onto-final-tangent approach
    // as computeBucketFillRegions's own post-pass, see BucketFillSegment's
    // own comment for why the tangent (not the gradient axis) is what a
    // line's endpoints should be measured along.
    const labX = lab % w, labY = (lab / w) | 0;
    let endAlongX = labX, endAlongY = labY, maxProj = 0;
    let endAgainstX = labX, endAgainstY = labY, minProj = 0;
    const tanLen = Math.hypot(sumFx, sumFy);
    if (tanLen > 0) {
      const tanX = -sumFy / tanLen, tanY = sumFx / tanLen;
      let first = true;
      for (const p of members) {
        const px = p % w, py = (p / w) | 0;
        const proj = px * tanX + py * tanY;
        if (first || proj > maxProj) { maxProj = proj; endAlongX = px; endAlongY = py; }
        if (first || proj < minProj) { minProj = proj; endAgainstX = px; endAgainstY = py; }
        first = false;
      }
    }

    segments.push({
      seedIndex: lab,
      count, cx: sumX / count, cy: sumY / count, avgFx: sumFx / count, avgFy: sumFy / count,
      endAlongX, endAlongY, endAgainstX, endAgainstY,
    });
  }

  return { regionId, segments };
}
