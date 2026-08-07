import * as THREE from 'three';
import { eligibilityThresholdSq } from './levelLine.ts';
import { type GrownRegion, type LsdRectangle, type LsdSettings } from './types.ts';

// ── Stage 4: magnitude-weighted PCA rectangle fit ─────────────────────────

interface RectangleCandidate { cx: number; cy: number; theta: number; length: number; width: number }

// The rectangle's angle comes from PCA on the region's own PIXEL
// COORDINATES (weighted by gradient magnitude) -- the standard image-
// moments technique for "what's this region's own axis of elongation,"
// which is exactly what a thin, long region wants for its rectangle's long
// axis. PCA only determines this axis up to a 180-degree ambiguity (an
// inertia matrix has no notion of "which way" along its own principal
// axis) -- resolved using the region's OWN directed growth-mean direction
// (meanUx/meanUy, already computed in stage 3): whichever of the PCA axis's two
// directions agrees with it is the one kept. This also matters for
// correctness, not just cosmetics -- stage 5's NFA alignment test compares
// pixel directions against the rectangle's theta directly, so a 180-degree-
// flipped theta would make every genuinely-aligned pixel look
// anti-aligned instead.
//
// NOTE this function deliberately KEEPS its atan2/cos/sin, even though the
// vector-space pass removed them everywhere else. The reason is that all of it
// is PER REGION (~1800 calls) rather than per pixel, so replacing the PCA angle
// with a direct half-angle eigenvector would buy nothing measurable while
// perturbing the one quantity that genuinely escapes this file as an angle.
// The per-pixel transcendentals were the whole cost, and those are gone.
export function fitRectangle(
  members: Int32Array, fx: Float64Array, fy: Float64Array, w: number,
  meanUx: number, meanUy: number,
): RectangleCandidate {
  let sumW = 0, sumX = 0, sumY = 0;
  for (let mi = 0; mi < members.length; mi++) {
    const i = members[mi];
    const m = Math.hypot(fx[i], fy[i]);
    sumW += m; sumX += m * (i % w); sumY += m * ((i / w) | 0);
  }
  const wcx = sumX / sumW, wcy = sumY / sumW;

  let Ixx = 0, Iyy = 0, Ixy = 0;
  for (let mi = 0; mi < members.length; mi++) {
    const i = members[mi];
    const m = Math.hypot(fx[i], fy[i]);
    const x = (i % w) - wcx, y = ((i / w) | 0) - wcy;
    Ixx += m * x * x; Iyy += m * y * y; Ixy += m * x * y;
  }
  let theta = 0.5 * Math.atan2(2 * Ixy, Ixx - Iyy);
  if (Math.cos(theta) * meanUx + Math.sin(theta) * meanUy < 0) theta += Math.PI;

  const ax = Math.cos(theta), ay = Math.sin(theta);
  const px = -ay, py = ax;
  let minProj = Infinity, maxProj = -Infinity, minPerp = Infinity, maxPerp = -Infinity;
  for (let mi = 0; mi < members.length; mi++) {
    const i = members[mi];
    const x = (i % w) - wcx, y = ((i / w) | 0) - wcy;
    const proj = x * ax + y * ay, perp = x * px + y * py;
    if (proj < minProj) minProj = proj; if (proj > maxProj) maxProj = proj;
    if (perp < minPerp) minPerp = perp; if (perp > maxPerp) maxPerp = perp;
  }
  // Rectangle center = midpoint of the projected extent, NOT the weighted
  // centroid directly -- the centroid isn't generally at the geometric
  // middle of the region's own extent along either axis, so placing the
  // rectangle there could clip real member pixels on one side while
  // overshooting empty space on the other.
  const midProj = (minProj + maxProj) / 2, midPerp = (minPerp + maxPerp) / 2;
  return {
    cx: wcx + midProj * ax + midPerp * px,
    cy: wcy + midProj * ay + midPerp * py,
    theta, length: maxProj - minProj, width: maxPerp - minPerp,
  };
}

// ── Stage 5: NFA validation ────────────────────────────────────────────────

// log(sum_{j=k}^{n} C(n,j) p^j (1-p)^(n-j)), natural log -- the standard
// stable approach: each term as log-choose + j*log(p) + (n-j)*log(1-p),
// with log-choose built INCREMENTALLY (C(n,j) = C(n,j-1) * (n-j+1)/j)
// rather than restarting an O(j) computation per term, then combined via
// log-sum-exp (subtract the running max before exponentiating) since terms
// can span many orders of magnitude and would overflow/underflow computed
// directly.
export function logBinomialTail(n: number, k: number, p: number): number {
  if (k <= 0) return 0; // log(1) -- any n pixels trivially have >=0 aligned
  if (k > n) return -Infinity;
  const logP = Math.log(p), log1mP = Math.log(1 - p);
  let logChoose = 0;
  for (let i = 1; i <= k; i++) logChoose += Math.log((n - k + i) / i);
  const logTerms: number[] = [];
  let logTerm = logChoose + k * logP + (n - k) * log1mP;
  logTerms.push(logTerm);
  let maxLog = logTerm;
  for (let j = k + 1; j <= n; j++) {
    logChoose += Math.log((n - j + 1) / j);
    logTerm = logChoose + j * logP + (n - j) * log1mP;
    logTerms.push(logTerm);
    if (logTerm > maxLog) maxLog = logTerm;
  }
  let sumExp = 0;
  for (const t of logTerms) sumExp += Math.exp(t - maxLog);
  return maxLog + Math.log(sumExp);
}

// Squashes nfaLog10 (unbounded, more negative = more confident) into the
// shared [0, 1] "how line-y is this" scale the retired join walk
// reads uniformly off every segment, regardless of which producer built it
// (see BucketFillSegment's own lineScore comment). A logistic curve centered
// exactly on the accept/reject threshold: a rectangle that JUST clears NFA
// scores 0.5, one an order of magnitude past it scores ~0.91, and it
// saturates smoothly toward 1 from there -- no separate span constant to
// tune, since the threshold itself (already a real per-settings quantity) is
// the only anchor this needs.
export function nfaLog10ToLineScore(nfaLog10: number, logEpsilon: number): number {
  const thresholdLog10 = logEpsilon / Math.LN10;
  return 1 / (1 + Math.pow(10, nfaLog10 - thresholdLog10));
}

// n = pixels whose center falls inside the rectangle's actual rotated
// footprint (scanned via its axis-aligned bounding box, not just the
// region's original flood-fill members -- the fitted rectangle's shape can
// differ slightly from the grown blob). k = of those, how many have a
// level-line angle within toleranceRad of the rectangle's own theta,
// DIRECTED: a plain signed cos-dot >= cos(toleranceRad), matching
// levelLinesCompatible's growth test exactly.
//
// This briefly used abs(cos-dot) instead, to match a mod-π growth experiment
// -- and that combination was quietly WRONG in a way worth recording, because
// it is easy to reintroduce. The NFA null model below uses p = τ/π, which is
// the probability that a uniformly-random DIRECTED angle lands within ±τ of a
// fixed direction (measure 2τ out of 2π). An abs() acceptance test admits TWO
// such windows (±τ of theta and of theta+π), i.e. measure 4τ out of 2π = 2τ/π
// -- twice as permissive as the model it was scored against. Every region
// therefore looked more statistically significant than it was, and the
// effective accept threshold was looser than nfaEpsilon claimed. Going
// directed here makes p = τ/π correct as-written rather than needing to be
// doubled. If mod-π counting is ever reintroduced, p MUST double with it.
//
// Sub-rho pixels count toward n (they were geometrically tested) but never
// toward k (too weak to trust their angle at all).
export function countRectanglePixels(
  rect: RectangleCandidate, fx: Float64Array, fy: Float64Array, w: number, h: number, rho: number, toleranceRad: number,
): { n: number; k: number } {
  const { cx, cy, theta: rectTheta, length } = rect;
  const ax = Math.cos(rectTheta), ay = Math.sin(rectTheta);
  const px = -ay, py = ax;
  const cosTol = Math.cos(toleranceRad);
  const rhoSq = eligibilityThresholdSq(rho);
  // Inclusion is tested with a small tolerance because the rectangle's own
  // extent is DEFINED by its extreme members: fitRectangle sets length =
  // maxProj - minProj, so those members land at |proj| exactly == hl, and on a
  // thin region every member lands at |perp| exactly == hw (which pins to the
  // 0.5 floor). Exact-boundary pixels are therefore guaranteed to exist in
  // every region, not a rare edge case -- and `Math.abs(proj) > hl` decides
  // them on whichever side the last ulp of rounding falls. That made the count
  // non-deterministic across arithmetic: the GPU port (f32) and this path
  // (f64) disagreed on n for 180 of 2931 regions and on k for 277, by up to 4
  // pixels each, which on a 12-pixel region moved nfaLog10 by 4.7 decades and
  // flipped 6 accept/reject decisions. The epsilon is far above f32 geometry
  // error (~1e-4 at image coordinates) and far below a pixel, so it includes
  // the boundary deterministically without admitting anything that isn't
  // genuinely inside.
  const BOUNDARY_EPS = 1e-3;
  const hl = length / 2;
  // Floor the tested half-width so a near-zero-width fit (a nearly
  // 1-pixel-wide ridge, common on a clean synthetic edge) still tests a
  // real strip of pixels instead of degenerating to nothing.
  const hw = Math.max(rect.width / 2, 0.5);

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [a, b] of [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]]) {
    const x = cx + a * ax + b * px, y = cy + a * ay + b * py;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const x0 = Math.max(0, Math.floor(minX)), x1 = Math.min(w - 1, Math.ceil(maxX));
  const y0 = Math.max(0, Math.floor(minY)), y1 = Math.min(h - 1, Math.ceil(maxY));

  let n = 0, k = 0;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx, dy = y - cy;
      const proj = dx * ax + dy * ay, perp = dx * px + dy * py;
      if (Math.abs(proj) > hl + BOUNDARY_EPS || Math.abs(perp) > hw + BOUNDARY_EPS) continue;
      n++;
      const i = y * w + x;
      // The hot loop of the whole fitter -- every pixel of every candidate
      // rectangle's footprint passes through here. The squared test rejects a
      // sub-rho pixel with no sqrt at all, and a pixel that survives pays
      // exactly one: alignDot is the level-line unit vector dotted with the
      // rectangle axis, and the unit vector is (-fy, fx)/‖g‖, so the whole
      // thing is one divide by the magnitude. This used to be a cos and a sin
      // per pixel, on an angle that had itself cost an atan2 to build.
      const gx = fx[i], gy = fy[i];
      const m2 = gx * gx + gy * gy;
      if (m2 <= rhoSq) continue;
      const alignDot = (-gy * ax + gx * ay) / Math.sqrt(m2); // = cos(levelLine(i) - rectTheta)
      if (alignDot >= cosTol) k++;
    }
  }
  return { n, k };
}

// RETIRED-NOTE: computeMagTheta used to live here, turning the gradient field
// into per-pixel (magnitude, angle) arrays for every stage below to consume.
// It is gone rather than moved. Its output was a lossy re-encoding of the
// gradient the previous stage already produces -- every consumer immediately
// converted the angle back into the vector it came from -- so stages 2 through
// 5 now read fx/fy directly and the pass does not exist. See the level-line
// vector block near the top of this file for the identity that makes that
// work, and the notes on fitRectangle for the one angle that survives.

// Stage 4 + stage 5 for ONE region -- the fitter, used by
// the CPU path for every region and by harness/lsdFitVerify.ts as the
// per-region reference the GPU kernel is compared against. Exactly the scope
// lsdFit.wgsl.ts implements, which is the point: there is nothing left for the
// two paths to disagree about structurally.
//
// Returns null only for a region below the 2-member floor. In practice that
// never happens now -- collectRegionsFromLabels' minRegionSize prefilter
// already dropped those -- but the guard stays because fitRectangle genuinely
// has no axis to fit with fewer than 2 points, and the floor is a slider.
export function fitRegionOnce(
  region: GrownRegion, fx: Float64Array, fy: Float64Array, w: number, h: number,
  settings: LsdSettings, logNTests: number, logEpsilon: number,
): LsdRectangle | null {
  const members = region.members;
  if (members.length < 2) return null;
  const rect = fitRectangle(members, fx, fy, w, region.meanUx, region.meanUy);
  const toleranceRad = THREE.MathUtils.degToRad(settings.toleranceDeg);
  const { n: rn, k: rk } = countRectanglePixels(rect, fx, fy, w, h, settings.rhoNoiseThreshold, toleranceRad);
  const logNfa = logNTests + logBinomialTail(rn, rk, toleranceRad / Math.PI);
  const nfaLog10 = logNfa / Math.LN10;
  return {
    cx: rect.cx, cy: rect.cy, theta: rect.theta, length: rect.length, width: rect.width,
    accepted: logNfa < logEpsilon, nfaLog10,
    lineScore: nfaLog10ToLineScore(nfaLog10, logEpsilon), rawMembers: region.members,
  };
}


// The two NFA log terms, derived once per call from the settings + image size
// so the stage-4 helpers below can't drift in how they compute them.
export function nfaLogTerms(w: number, h: number, settings: LsdSettings): { logNTests: number; logEpsilon: number } {
  return {
    logNTests: settings.nfaTestExponent * Math.log(Math.max(w, h)),
    logEpsilon: Math.log(settings.nfaEpsilon),
  };
}

// Stage 4+5 on CPU, for an ALREADY-GROWN region set. Split out from
// computeLsdRectangles so the grower and the fitter can be dispatched to
// CPU/GPU independently of each other (see computeLsdRectanglesAuto).
export function fitRegionsCPU(
  regions: readonly GrownRegion[], fx: Float64Array, fy: Float64Array,
  w: number, h: number, settings: LsdSettings,
): LsdRectangle[] {
  const { logNTests, logEpsilon } = nfaLogTerms(w, h, settings);
  const results: LsdRectangle[] = [];
  for (const region of regions) {
    const r = fitRegionOnce(region, fx, fy, w, h, settings, logNTests, logEpsilon);
    if (r) results.push(r);
  }
  return results;
}
