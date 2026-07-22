import * as THREE from 'three';
import { CameraSettingsCommon } from '../camera/settings.ts';
import { jacobiEigenSymmetric, smallestEigenvector } from '../../linalg.ts';
import { cornerDir } from '../math/geometry.ts';
import { spanEnd, spanStart } from '../profiling/profiler.ts';
import { Vote } from '../types.ts';
import { compositeLineLength, computeCompositeLines, computeJoinWalk, computeMergeGroups } from './bucketFillJoin.ts';
import { computeBucketFillRegions } from './bucketFillSegments.ts';
import { computeEffectiveGradientField, computeGradientAgreementField, computeGradientField } from './gradientField.ts';
import { computeTopGradientAlpha } from './gradientHighlight.ts';
import { guidedTangentDirectionForWalk } from './tangentWalk.ts';

// gray is expected to already be captureDistortedGrayscale's output.
export function computeWorldVotes(
  settings: CameraSettingsCommon,
  gray: Float64Array, w: number, h: number,
  gradientRadius: number, agreementRadius: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Vote[] {
  const votes: Vote[] = [];
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  const gradSpan = spanStart('gradientField');
  const field = computeGradientField(gray, w, h, gradientRadius);
  spanEnd(gradSpan);
  const agreeSpan = spanStart('agreementField');
  const agreement = computeGradientAgreementField(field, agreementRadius);
  spanEnd(agreeSpan);
  const effSpan = spanStart('effectiveField');
  const effective = computeEffectiveGradientField(field, agreement);
  spanEnd(effSpan);
  const { fx, fy, r } = effective;
  const walkSpan = spanStart('walk+vote loop');
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      if (fx[i] === 0 && fy[i] === 0) continue;
      const walked = guidedTangentDirectionForWalk(settings, fx, fy, w, h, x, y, fx[i], fy[i]);
      let theta = Math.atan2(walked.fy, walked.fx);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      const tdx = -Math.sin(theta), tdy = Math.cos(theta);
      const [u1, v1] = toNDC(x, y);
      const [u2, v2] = toNDC(x + tdx, y + tdy);
      const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
      const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
      const n = ray1.clone().cross(ray2);
      if (n.lengthSq() < 1e-12) continue;
      n.normalize();
      votes.push({ n, weight: Math.hypot(walked.fx, walked.fy) });
    }
  }
  spanEnd(walkSpan);
  return votes;
}

// Experimental alternative to computeWorldVotes: one vote per bucket-fill-
// and-JOIN-WALK composite LINE (pipeline/bucketFillJoin.ts) instead of one
// per pixel. Builds the exact same "effective" field (raw gradient x local
// agreement) computeWorldVotes builds internally, flood-fills it into
// segments, then runs the full join walk (merge groups + composite lines,
// same as the join-walk overlay) and casts a ray to each GROUP's composite
// endpoints -- not each raw segment's own endpoints -- so a chain of
// several short segments joined into one long line votes as ONE long,
// well-conditioned ray instead of several separate short, noisier ones.
// Reuses the SAME percentile-band, tolerance/magnitude/length, and join-
// walk settings (steps, merge similarity) the bucket-fill/join overlays
// already expose, so this is tunable from the existing sliders without new
// UI just for this path.
export function computeSegmentVotes(
  settings: CameraSettingsCommon,
  gray: Float64Array, w: number, h: number,
  gradientRadius: number, agreementRadius: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Vote[] {
  const field = computeGradientField(gray, w, h, gradientRadius);
  const agreement = computeGradientAgreementField(field, agreementRadius);
  const effective = computeEffectiveGradientField(field, agreement);
  const seedEligible = computeTopGradientAlpha(effective, settings.circleSamplePercentMin, settings.circleSamplePercentMax);
  const { regionId, segments } = computeBucketFillRegions(effective, settings.bucketFillToleranceDeg, seedEligible, settings.bucketFillMagnitudeThreshold);

  const { merges } = computeJoinWalk(
    segments, regionId, w, h, settings.bucketFillMergeMinSimilarity, settings.bucketFillJoinSteps, settings.bucketFillMinLengthPx,
  );
  const groupOf = computeMergeGroups(segments.length, merges);
  const composites = computeCompositeLines(segments, groupOf);

  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  const votes: Vote[] = [];
  for (const line of composites.values()) {
    if (compositeLineLength(line) < settings.bucketFillMinLengthPx) continue; // an unmerged singleton whose own segment was already too short
    const [u1, v1] = toNDC(line.x1, line.y1);
    const [u2, v2] = toNDC(line.x2, line.y2);
    const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
    const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
    const n = ray1.clone().cross(ray2);
    const arcLen = n.length(); // sin(angle between the two endpoint rays) -- the composite line's own PROJECTED ARC LENGTH on the unit sphere
    if (arcLen < 1e-12) continue;
    n.divideScalar(arcLen); // normalize using the length already computed, instead of a second hypot
    // Weighted by that same projected arc length, not pixel count or average
    // magnitude -- a short/close-together line gives a small, easily-
    // corrupted cross product; a long one gives a large, well-conditioned
    // one, which is exactly the property we want a fit-confidence weight to
    // track (same reasoning computeWorldVotes's own per-pixel cross product
    // relies on implicitly, just now visible as an explicit per-composite-
    // line quantity instead of buried in a 1-pixel step).
    votes.push({ n, weight: arcLen });
  }
  return votes;
}

// The TRUE [minPercent, maxPercent) band by magnitude rank, out of every vote.
export function votesInMagnitudeBand(votes: Vote[], minPercent: number, maxPercent: number): Vote[] {
  const sorted = Array.from(votes).sort((a, b) => b.weight - a.weight);
  const lo = Math.round(sorted.length * (minPercent / 100));
  const hi = Math.round(sorted.length * (maxPercent / 100));
  if (hi <= lo) return [];
  return sorted.slice(lo, hi);
}

// Fits the degenerate quadric ("pair of planes through the origin") that
// best explains "every vote lies on one plane or the other" -- see
// pre-Stage-A history for the full derivation. `power` is the caller's
// current weightSharpenPower setting.
export function fitPairOfPlanes(votes: Vote[], power: number): { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null {
  let maxW = 0;
  for (const { weight } of votes) if (weight > maxW) maxW = weight;
  const ATA: number[][] = Array.from({ length: 6 }, () => new Array(6).fill(0));
  for (const { n, weight } of votes) {
    const sharpened = maxW > 0 ? Math.pow(weight / maxW, power) : 0;
    const row = [n.x * n.x, n.y * n.y, n.z * n.z, n.x * n.y, n.x * n.z, n.y * n.z];
    for (let a = 0; a < 6; a++) {
      const wra = sharpened * row[a];
      for (let b = 0; b < 6; b++) ATA[a][b] += wra * row[b];
    }
  }
  const m = smallestEigenvector(ATA);
  const M = [
    [m[0], m[3] / 2, m[4] / 2],
    [m[3] / 2, m[1], m[5] / 2],
    [m[4] / 2, m[5] / 2, m[2]],
  ];
  const { values, vectors } = jacobiEigenSymmetric(M);
  let zeroIdx = 0;
  for (let i = 1; i < 3; i++) if (Math.abs(values[i]) < Math.abs(values[zeroIdx])) zeroIdx = i;
  const others = [0, 1, 2].filter((i) => i !== zeroIdx);
  const b1 = new THREE.Vector3(vectors[others[0]][0], vectors[others[0]][1], vectors[others[0]][2]);
  const b2 = new THREE.Vector3(vectors[others[1]][0], vectors[others[1]][1], vectors[others[1]][2]);
  const Dnormal = new THREE.Vector3(vectors[zeroIdx][0], vectors[zeroIdx][1], vectors[zeroIdx][2]).normalize();
  const Drow = b1.clone().add(b2);
  const Dcol = b1.clone().sub(b2);
  if (Drow.lengthSq() < 1e-9 || Dcol.lengthSq() < 1e-9) return null;
  return { Drow: Drow.normalize(), Dcol: Dcol.normalize(), Dnormal };
}

