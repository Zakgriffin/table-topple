import * as THREE from 'three';
import { CameraSettingsCommon } from '../camera/settings.ts';
import { GradientField, JacobianField } from '../types.ts';

// STATUS (tabled, not abandoned): this is a visualization-only prototype,
// wired into a debug field-view ('jacobian') + hover arrows/spoked-walk
// outline, purely so the eigen-structure could be SEEN on real captures
// before deciding whether to build on it. Nothing here feeds into
// computeWorldVotes/fitPairOfPlanes yet.
//
// What's next if this gets picked back up, roughly in order:
//  1. Noise robustness: computeLocalJacobianField differences the ALREADY-
//     computed gradient field a second time (a discrete Hessian) -- fine on
//     clean simulated captures, but likely too noise-sensitive on real
//     photos without first moving to a windowed/aggregated second-moment
//     estimate (aggregate outer products of nearby raw gradients) rather
//     than a bare finite difference. See the chat discussion this session
//     for the full reasoning on why that's the likely failure mode.
//  2. paintJacobianFieldAsColor's dark pixels are ambiguous between "corner"
//     and "no gradient signal at all here" (both give low anisotropy) --
//     needs a third channel/threshold to disambiguate before the coloring
//     is trustworthy to read at a glance.
//  3. computeSpokedWalkIncludedPixels uses FIXED axes seeded once from the
//     hover pixel's own e1/e2 -- no adaptive re-steering per step (unlike
//     guidedTangentDirectionAdaptive) and no recomputed-Jacobian-as-the-
//     kernel-grows checkpointing (the "watch the eigenvectors diverge from
//     perpendicular as you expand" idea) -- both were explicitly deferred
//     for this first pass.
//  4. The actual payoff, once the above are trusted: cast TWO votes from a
//     confidently-corner pixel (one per eigenvector) into fitPairOfPlanes
//     instead of muting it via the agreement field -- see this session's
//     chat for the full "corners as signal, not contamination" reframe.
//
// ── Local Jacobian (structure) field ──────────────────────────────────────
//
// For each pixel, the 2x2 spatial Jacobian of the gradient FIELD itself
// (i.e. a discrete Hessian of the image): Jxx=d(fx)/dx, Jxy=d(fx)/dy,
// Jyx=d(fy)/dx, Jyy=d(fy)/dy. For an ideal locally-1D edge (I(x,y) ~
// f(n.(x,y))), this works out to J = f''(s)*(n n^T) -- rank 1, symmetric,
// with eigenvectors EXACTLY along the gradient (n) and EXACTLY perpendicular
// to it (the edge tangent). At a genuine corner/junction the local 1-edge
// model breaks down and J stops looking rank-1.
//
// We split J into its symmetric part (always has 2 real, exactly orthogonal
// eigenvectors -- e1/lambda1 the dominant one, e2/lambda2 the subordinate
// one) and its antisymmetric part (a single scalar `asym`, exactly zero for
// the ideal symmetric case above, nonzero only where discretization/noise
// reveal the local model breaking down) -- see JacobianField's own comment
// in types.ts for the full reasoning.
function eigSym2x2(a: number, b: number, d: number): { l1: number; l2: number; e1x: number; e1y: number } {
  const half = (a - d) / 2;
  const rad = Math.hypot(half, b);
  const l1 = (a + d) / 2 + rad, l2 = (a + d) / 2 - rad;
  let e1x: number, e1y: number;
  if (rad < 1e-12) {
    e1x = 1; e1y = 0; // isotropic -- no preferred direction, pick arbitrarily
  } else if (Math.abs(b) > 1e-12) {
    e1x = b; e1y = l1 - a;
  } else {
    e1x = a >= d ? 1 : 0; e1y = a >= d ? 0 : 1;
  }
  const len = Math.hypot(e1x, e1y);
  if (len > 1e-12) { e1x /= len; e1y /= len; } else { e1x = 1; e1y = 0; }
  return { l1, l2, e1x, e1y };
}

export function computeLocalJacobianField(field: GradientField, jacRadius: number): JacobianField {
  const { fx, fy, w, h } = field;
  const r = Math.max(1, Math.round(jacRadius));
  const margin = field.r + r; // fx/fy are only valid outside field.r of the image edge
  const n = w * h;
  const e1x = new Float64Array(n), e1y = new Float64Array(n), lambda1 = new Float64Array(n);
  const e2x = new Float64Array(n), e2y = new Float64Array(n), lambda2 = new Float64Array(n);
  const asym = new Float64Array(n);
  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) {
      const i = y * w + x;
      const Jxx = (fx[i + r] - fx[i - r]) / (2 * r);
      const Jxy = (fx[i + r * w] - fx[i - r * w]) / (2 * r);
      const Jyx = (fy[i + r] - fy[i - r]) / (2 * r);
      const Jyy = (fy[i + r * w] - fy[i - r * w]) / (2 * r);
      const { l1, l2, e1x: v1x, e1y: v1y } = eigSym2x2(Jxx, (Jxy + Jyx) / 2, Jyy);
      e1x[i] = v1x; e1y[i] = v1y; lambda1[i] = l1;
      e2x[i] = -v1y; e2y[i] = v1x; lambda2[i] = l2; // exactly perpendicular to e1, guaranteed by symmetry
      asym[i] = (Jxy - Jyx) / 2;
    }
  }
  return { e1x, e1y, lambda1, e2x, e2y, lambda2, asym, w, h, r: margin };
}

// bright gray = clean, strongly single-direction (anisotropic) local edge;
// blue tint = antisymmetric "orthogonality defect", i.e. the local 1-edge
// model is breaking down (corner/junction candidate) -- independent signal
// from the anisotropy one, can be bright-and-blue (confident corner) or
// dark-and-blue (weak but asymmetric, e.g. noise) at the same time.
export function paintJacobianFieldAsColor(field: JacobianField, out: Uint8Array) {
  const { lambda1, lambda2, asym, w, h } = field;
  const n = w * h;
  let maxL1 = 0;
  for (let i = 0; i < n; i++) { const v = Math.abs(lambda1[i]); if (v > maxL1) maxL1 = v; }
  for (let i = 0; i < n; i++) {
    const l1 = Math.abs(lambda1[i]), l2 = Math.abs(lambda2[i]);
    const sum = l1 + l2;
    const anisotropy = sum > 0 ? (l1 - l2) / sum : 0; // 1 = clean edge, 0 = isotropic/corner
    const normAsym = maxL1 > 0 ? Math.min(1, Math.abs(asym[i]) / maxL1) : 0;
    const bright = Math.round(THREE.MathUtils.clamp(anisotropy, 0, 1) * 255);
    const blue = Math.round(THREE.MathUtils.clamp(normAsym, 0, 1) * 255);
    const o = i * 4;
    out[o] = bright; out[o + 1] = bright; out[o + 2] = Math.max(bright, blue); out[o + 3] = 255;
  }
}

// ── Spoked walk: the 4-direction generalization of guidedTangentDirection ──
//
// Walks BOTH eigenvector axes instead of just the tangent. e2 (subordinate,
// "along the edge") keeps today's tangent-walk semantics unchanged: a low-
// magnitude or direction-deviating sample is a violation, grace-limited.
// e1 (dominant, "across the edge") is walked with different semantics:
// stepping across a real edge is EXPECTED to pass through a near-zero-
// magnitude cell interior, so a low-magnitude sample there is tolerated, not
// a violation -- only a STRONG sample whose direction has drifted off the
// seeded e1 axis counts against it, which is what actually validates "the
// next edge I hit is a parallel one from the same family, at the pattern's
// period" rather than some unrelated feature.
export interface SpokedWalkPixel { x: number; y: number; axis: 1 | 2 }
export function computeSpokedWalkIncludedPixels(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number,
  e1x: number, e1y: number, e2x: number, e2y: number,
): SpokedWalkPixel[] {
  const included: SpokedWalkPixel[] = [{ x, y, axis: 1 }];
  const maxSteps = settings.tangentWalkMaxSteps;
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(settings.tangentWalkDeviationDeg));
  const magFraction = settings.tangentWalkMagFraction;
  const grace = settings.tangentWalkGraceSamples;
  const seedMag = Math.hypot(fx[y * w + x], fy[y * w + x]) || 1e-9;

  function walkAxis(axisDx: number, axisDy: number, axis: 1 | 2, toleratesGap: boolean) {
    const axisTheta = Math.atan2(axisDy, axisDx);
    let sumCos = Math.cos(2 * axisTheta), sumSin = Math.sin(2 * axisTheta);
    let runningMag = seedMag;
    let sampleCount = 1;
    for (const sign of [1, -1]) {
      let violations = 0;
      for (let k = 1; k <= maxSteps; k++) {
        const sx = Math.round(x + sign * k * axisDx), sy = Math.round(y + sign * k * axisDy);
        if (sx < 0 || sx >= w || sy < 0 || sy >= h) break;
        const si = sy * w + sx;
        const sfx = fx[si], sfy = fy[si];
        const mag = Math.hypot(sfx, sfy);
        if (mag === 0 || mag < runningMag * magFraction) {
          if (toleratesGap) { included.push({ x: sx, y: sy, axis }); continue; }
          violations++;
          if (violations >= grace) break;
          continue;
        }
        const theta = Math.atan2(sfy, sfx);
        const c2 = Math.cos(2 * theta), s2 = Math.sin(2 * theta);
        const avgLen = Math.hypot(sumCos, sumSin);
        const cosDeviation = avgLen > 0 ? (c2 * sumCos + s2 * sumSin) / avgLen : 1;
        if (cosDeviation < devCos) {
          violations++;
          if (violations >= grace) break;
          continue;
        }
        violations = 0;
        sumCos += c2 * mag; sumSin += s2 * mag;
        runningMag = (runningMag * sampleCount + mag) / (sampleCount + 1);
        sampleCount++;
        included.push({ x: sx, y: sy, axis });
      }
    }
  }

  walkAxis(e2x, e2y, 2, false);
  walkAxis(e1x, e1y, 1, true);
  return included;
}
