import * as THREE from 'three';
import { BucketFillSegment } from './bucketFillSegments.ts';
import { fitAndTestRegionsGPU } from '../pipelineGPU/lsdFit.ts';
import { globalState } from '../state.ts';
import { GradientField } from '../types.ts';

// ── LSD (Line Segment Detector, von Gioi/Jakubowicz/Morel/Randall 2010) ───
// ── from scratch ────────────────────────────────────────────────────────
//
// A genuinely traditional reimplementation. This is now the PRODUCTION
// composite-line source -- pipeline/votes.ts's computeGradient2x2Composites
// calls lsdRectanglesToBucketFillShape (bottom of this file) to feed
// pipeline/bucketFillJoin.ts's join walk, replacing pipeline/
// bucketFillSegments.ts's own BFS growing, which stays defined and correct
// but is no longer invoked from anywhere in the live pipeline -- kept in
// the repo as a reference/comparison implementation, not deleted.
//
// Pipeline: order pixels by gradient magnitude (stage 2) -> serial region
// growing by directed level-line-angle agreement (stage 3) -> magnitude-
// weighted PCA rectangle fit per region (stage 4) -> NFA statistical
// validation with a bounded tighten-then-shrink retry loop (stage 5).
//
// GPU-friendliness note: stages 2, 4, and 5 all map onto standard parallel
// primitives (bucket/counting sort, segmented reduction, per-object bounded
// scan) -- but only stages 4+5's FIRST NFA pass actually got a GPU port
// (pipelineGPU/lsdFit.ts, gated by globalState.useGPULsdFit, see
// computeLsdRectanglesGPU below). Stage 2 (the counting sort) did NOT,
// despite being an equally standard GPU primitive in the abstract: its
// output (`order`) and inputs (mag/theta) all have to be CPU-resident
// regardless, since stage 3 (next) is CPU-only and needs them as plain JS
// arrays -- so a GPU stage 2 would upload fx/fy (again -- they're already
// CPU-resident by the time this function is called, see GradientField's own
// contract), dispatch, and read the SAME data straight back, for a linear-
// time operation that's already cheap on CPU. That round-trip overhead is
// exactly the failure mode pipelineGPU/decodeTally.ts's own header
// documents (measured net-negative at realistic sizes) -- not worth
// repeating here without evidence stage 2 is an actual bottleneck.
//
// Stage 3 (region growing) isn't GPU-friendly AT ALL, round-trip cost
// aside -- LSD's own real behavior depends on growing one region all the
// way to completion, strictly in magnitude-priority order, before even
// looking at the next seed; that serial commitment is WHY a strong seed's
// region absorbs an entire ridge before a weaker pixel on the same ridge
// ever gets a chance to start its own competing region. A synchronous/
// parallel restructuring of this stage reintroduces exactly that
// fragmentation, changing WHICH regions form, not just how fast. Left
// serial (CPU) unconditionally.
//
// Stage 5's retry loop (tighten-then-shrink on NFA rejection) also stayed
// CPU-only -- retry 1 (angle refilter) would be easy, but retry 2+ needs a
// per-region partial sort/selection (drop the farthest-from-center
// fraction), a genuinely harder GPU problem than anything else here. Any
// region the GPU's first pass rejects falls through to fitRegionWithRetries
// below (the exact same CPU code the pure-CPU path uses for every region),
// re-attempting from scratch rather than resuming GPU state -- simpler and
// risk-free, at the minor cost of redoing that region's attempt 0 on CPU
// too.

// The level-line angle (LSD's own convention): the gradient rotated -90
// degrees, a DIRECTED angle (encodes which side is darker) -- unlike this
// codebase's usual undirected/axial gradient-angle convention used
// elsewhere (bucketFillSegments.ts etc.), so comparisons here use a plain
// cosine test, no double-angle fold.
function levelLineAngle(fx: number, fy: number): number {
  return Math.atan2(fx, -fy);
}

function wrappedAngleDiff(a: number, b: number): number {
  let d = Math.abs(a - b) % (2 * Math.PI);
  if (d > Math.PI) d = 2 * Math.PI - d;
  return d;
}

// ── Stage 2: approximate magnitude-descending pixel order ────────────────
//
// A bucket/counting sort, not an exact sort -- LSD's own reference
// implementation does the same (its NOTUSED/USED pixel list is similarly
// bucket-ordered), since "roughly strongest first" is all region growing
// actually needs. Also the natural GPU primitive for this stage (parallel
// bucket counts + prefix sum), unlike an exact comparison sort.
function orderPixelsByMagnitudeDescending(mag: Float64Array, rho: number, numBuckets: number): Int32Array {
  const n = mag.length;
  let maxMag = 0;
  for (let i = 0; i < n; i++) if (mag[i] > maxMag) maxMag = mag[i];
  if (maxMag <= rho) return new Int32Array(0);
  const buckets = Math.max(1, Math.round(numBuckets));
  const counts = new Int32Array(buckets);
  const bucketOf = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const m = mag[i];
    if (m <= rho) continue;
    let b = Math.floor((m / maxMag) * buckets);
    if (b >= buckets) b = buckets - 1;
    bucketOf[i] = b;
    counts[b]++;
  }
  let total = 0;
  for (let b = 0; b < buckets; b++) total += counts[b];
  // Cumulative offsets, HIGHEST bucket first (descending magnitude).
  const offsets = new Int32Array(buckets);
  let running = 0;
  for (let b = buckets - 1; b >= 0; b--) { offsets[b] = running; running += counts[b]; }
  const order = new Int32Array(total);
  const cursor = offsets.slice();
  for (let i = 0; i < n; i++) {
    const b = bucketOf[i];
    if (b < 0) continue;
    order[cursor[b]++] = i;
  }
  return order;
}

// ── Stage 3: serial region growing ────────────────────────────────────────
//
// Same angle-tolerance-agreement shape as bucketFillSegments.ts's own
// growth loop, but: (a) directed level-line angle, plain cosine test, no
// double-angle fold; (b) the region's running mean is a REAL circular mean
// (sum of unit vectors), not weighted by magnitude -- LSD's own region
// angle is unweighted, unlike the rectangle fit's centroid/axis in stage 4;
// (c) keeps each region's full member-pixel LIST (not just running sums),
// since stage 4's weighted PCA needs the actual pixel positions, not just a
// running vector sum.
interface GrownRegion { members: Int32Array; meanAngle: number }

function growRegions(
  mag: Float64Array, theta: Float64Array, w: number, h: number,
  order: Int32Array, toleranceDeg: number, rho: number,
): { regionId: Int32Array; regions: GrownRegion[] } {
  const n = w * h;
  const regionId = new Int32Array(n).fill(-1);
  const toleranceRad = THREE.MathUtils.degToRad(toleranceDeg);
  const cosTol = Math.cos(toleranceRad);
  const regions: GrownRegion[] = [];
  const queue = new Int32Array(n);
  const memberBuf = new Int32Array(n);

  for (let oi = 0; oi < order.length; oi++) {
    const seed = order[oi];
    if (regionId[seed] !== -1) continue;
    const id = regions.length;
    regionId[seed] = id;
    let sumCos = Math.cos(theta[seed]), sumSin = Math.sin(theta[seed]);
    let qHead = 0, qTail = 0, memberCount = 0;
    queue[qTail++] = seed; memberBuf[memberCount++] = seed;

    while (qHead < qTail) {
      const p = queue[qHead++];
      const px = p % w, py = (p / w) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = px + dx, ny = py + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const ni = ny * w + nx;
          if (regionId[ni] !== -1 || mag[ni] <= rho) continue;
          const avgLen = Math.hypot(sumCos, sumSin);
          const cosDeviation = avgLen > 0 ? (Math.cos(theta[ni]) * sumCos + Math.sin(theta[ni]) * sumSin) / avgLen : 1;
          if (cosDeviation < cosTol) continue;
          regionId[ni] = id;
          sumCos += Math.cos(theta[ni]); sumSin += Math.sin(theta[ni]);
          queue[qTail++] = ni;
          memberBuf[memberCount++] = ni;
        }
      }
    }

    regions.push({ members: memberBuf.slice(0, memberCount), meanAngle: Math.atan2(sumSin, sumCos) });
  }
  return { regionId, regions };
}

// ── Stage 4: magnitude-weighted PCA rectangle fit ─────────────────────────

interface RectangleCandidate { cx: number; cy: number; theta: number; length: number; width: number }

// The rectangle's angle comes from PCA on the region's own PIXEL
// COORDINATES (weighted by gradient magnitude) -- the standard image-
// moments technique for "what's this region's own axis of elongation,"
// which is exactly what a thin, long region wants for its rectangle's long
// axis. PCA only determines this axis up to a 180-degree ambiguity (an
// inertia matrix has no notion of "which way" along its own principal
// axis) -- resolved using the region's OWN directed growth-mean angle
// (meanAngle, already computed in stage 3): whichever of the PCA axis's two
// directions agrees with meanAngle is the one kept. This also matters for
// correctness, not just cosmetics -- stage 5's NFA alignment test compares
// pixel angles against the rectangle's theta directly, so a 180-degree-
// flipped theta would make every genuinely-aligned pixel look
// anti-aligned instead.
function fitRectangle(members: Int32Array, mag: Float64Array, w: number, meanAngle: number): RectangleCandidate {
  let sumW = 0, sumX = 0, sumY = 0;
  for (let mi = 0; mi < members.length; mi++) {
    const i = members[mi];
    const m = mag[i];
    sumW += m; sumX += m * (i % w); sumY += m * ((i / w) | 0);
  }
  const wcx = sumX / sumW, wcy = sumY / sumW;

  let Ixx = 0, Iyy = 0, Ixy = 0;
  for (let mi = 0; mi < members.length; mi++) {
    const i = members[mi];
    const m = mag[i];
    const x = (i % w) - wcx, y = ((i / w) | 0) - wcy;
    Ixx += m * x * x; Iyy += m * y * y; Ixy += m * x * y;
  }
  let theta = 0.5 * Math.atan2(2 * Ixy, Ixx - Iyy);
  const meanAx = Math.cos(meanAngle), meanAy = Math.sin(meanAngle);
  if (Math.cos(theta) * meanAx + Math.sin(theta) * meanAy < 0) theta += Math.PI;

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
function logBinomialTail(n: number, k: number, p: number): number {
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
// shared [0, 1] "how line-y is this" scale bucketFillJoin.ts's join walk
// reads uniformly off every segment, regardless of which producer built it
// (see BucketFillSegment's own lineScore comment). A logistic curve centered
// exactly on the accept/reject threshold: a rectangle that JUST clears NFA
// scores 0.5, one an order of magnitude past it scores ~0.91, and it
// saturates smoothly toward 1 from there -- no separate span constant to
// tune, since the threshold itself (already a real per-settings quantity) is
// the only anchor this needs.
function nfaLog10ToLineScore(nfaLog10: number, logEpsilon: number): number {
  const thresholdLog10 = logEpsilon / Math.LN10;
  return 1 / (1 + Math.pow(10, nfaLog10 - thresholdLog10));
}

// n = pixels whose center falls inside the rectangle's actual rotated
// footprint (scanned via its axis-aligned bounding box, not just the
// region's original flood-fill members -- the fitted rectangle's shape can
// differ slightly from the grown blob). k = of those, how many have a
// level-line angle within toleranceRad of the rectangle's own theta.
// Sub-rho pixels count toward n (they were geometrically tested) but never
// toward k (too weak to trust their angle at all).
function countRectanglePixels(
  rect: RectangleCandidate, mag: Float64Array, theta: Float64Array, w: number, h: number, rho: number, toleranceRad: number,
): { n: number; k: number } {
  const { cx, cy, theta: rectTheta, length } = rect;
  const ax = Math.cos(rectTheta), ay = Math.sin(rectTheta);
  const px = -ay, py = ax;
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
      if (Math.abs(proj) > hl || Math.abs(perp) > hw) continue;
      n++;
      const i = y * w + x;
      if (mag[i] <= rho) continue;
      if (wrappedAngleDiff(theta[i], rectTheta) <= toleranceRad) k++;
    }
  }
  return { n, k };
}

export interface LsdRectangle {
  cx: number; cy: number; theta: number; length: number; width: number;
  accepted: boolean;
  retries: number; // how many tighten/shrink attempts were taken before the final accept/reject
  nfaLog10: number; // log10(NFA) -- more negative = more statistically confident
  lineScore: number; // nfaLog10 squashed to [0, 1] via nfaLog10ToLineScore -- see that function's own comment
  // Stage 3's ORIGINAL grown-region membership (pixel indices into the
  // field), before any retry loop tightened/shrank it -- region.members
  // itself is never mutated by the retry loop below (only the local
  // `members` variable gets reassigned), so this is always the true,
  // complete flood-fill result this rectangle came from. For debug display
  // only (overlays/hoverDebugOverlays.ts's raw-region-pixels toggle) -- not
  // read anywhere in the accept/reject decision itself.
  rawMembers: Int32Array;
}

export interface LsdSettings {
  toleranceDeg: number; rhoNoiseThreshold: number; magnitudeBuckets: number;
  nfaEpsilon: number; nfaTestExponent: number;
  maxRetries: number; retryToleranceFactor: number; retryShrinkFraction: number;
}

function computeMagTheta(field: GradientField): { mag: Float64Array; theta: Float64Array } {
  const { fx, fy, w, h } = field;
  const n = w * h;
  const mag = new Float64Array(n), theta = new Float64Array(n);
  for (let i = 0; i < n; i++) { mag[i] = Math.hypot(fx[i], fy[i]); theta[i] = levelLineAngle(fx[i], fy[i]); }
  return { mag, theta };
}

// Stage 5's full accept/retry loop for ONE region -- shared by the pure-CPU
// path (every region) and the GPU path (only regions lsdFit.wgsl.ts's first
// pass rejected, re-attempted from scratch rather than resuming GPU state,
// see this file's header). Returns null only for a degenerate region (< 2
// members even before any retry).
function fitRegionWithRetries(
  region: GrownRegion, mag: Float64Array, theta: Float64Array, w: number, h: number,
  settings: LsdSettings, logNTests: number, logEpsilon: number,
): LsdRectangle | null {
  let members = region.members;
  let toleranceDeg = settings.toleranceDeg;
  let retries = 0;
  let accepted = false;
  let rect: RectangleCandidate | null = null;
  let nfaLog10 = Infinity;

  for (;;) {
    if (members.length < 2) break; // degenerate -- no meaningful axis, leave as rejected
    rect = fitRectangle(members, mag, w, region.meanAngle);
    const toleranceRad = THREE.MathUtils.degToRad(toleranceDeg);
    const p = toleranceRad / Math.PI;
    const { n: rn, k: rk } = countRectanglePixels(rect, mag, theta, w, h, settings.rhoNoiseThreshold, toleranceRad);
    const logNfa = logNTests + logBinomialTail(rn, rk, p);
    nfaLog10 = logNfa / Math.LN10;
    if (logNfa < logEpsilon) { accepted = true; break; }
    if (retries >= settings.maxRetries) break;
    retries++;
    if (retries === 1) {
      // Retighten first (LSD's own first move): a simplified stand-in for
      // re-growing the region from its seed under a stricter tolerance --
      // re-filters the CURRENT members down to those still within the
      // new, tighter tolerance of the region's own (fixed) mean angle,
      // rather than a full re-grow from scratch. Same intent (shrink to
      // the self-consistent "core" of the region) with far less
      // machinery than repeating stage 3 per retry.
      toleranceDeg *= settings.retryToleranceFactor;
      const cosTol = Math.cos(THREE.MathUtils.degToRad(toleranceDeg));
      const meanAx = Math.cos(region.meanAngle), meanAy = Math.sin(region.meanAngle);
      const kept: number[] = [];
      for (let mi = 0; mi < members.length; mi++) {
        const i = members[mi];
        if (Math.cos(theta[i]) * meanAx + Math.sin(theta[i]) * meanAy >= cosTol) kept.push(i);
      }
      members = Int32Array.from(kept);
    } else {
      // Subsequent retries: drop the farthest-from-center fraction of
      // members -- LSD's own "reduce region radius," aimed at cutting
      // away a spuriously-attached side branch (e.g. two near-parallel
      // lines the growth pass glued together) rather than tightening
      // angle further.
      const { cx, cy } = rect;
      const withDist: { i: number; d: number }[] = [];
      for (let mi = 0; mi < members.length; mi++) {
        const i = members[mi];
        const x = (i % w) - cx, y = ((i / w) | 0) - cy;
        withDist.push({ i, d: x * x + y * y });
      }
      withDist.sort((a, b) => a.d - b.d);
      const keep = Math.max(2, Math.round(members.length * (1 - settings.retryShrinkFraction)));
      members = Int32Array.from(withDist.slice(0, keep).map((e) => e.i));
    }
  }

  if (!rect) return null;
  const lineScore = nfaLog10ToLineScore(nfaLog10, logEpsilon);
  return { cx: rect.cx, cy: rect.cy, theta: rect.theta, length: rect.length, width: rect.width, accepted, retries, nfaLog10, lineScore, rawMembers: region.members };
}

export function computeLsdRectangles(field: GradientField, settings: LsdSettings): LsdRectangle[] {
  const { w, h } = field;
  const { mag, theta } = computeMagTheta(field);

  const order = orderPixelsByMagnitudeDescending(mag, settings.rhoNoiseThreshold, settings.magnitudeBuckets);
  const { regions } = growRegions(mag, theta, w, h, order, settings.toleranceDeg, settings.rhoNoiseThreshold);

  const maxDim = Math.max(w, h);
  const logNTests = settings.nfaTestExponent * Math.log(maxDim);
  const logEpsilon = Math.log(settings.nfaEpsilon);

  const results: LsdRectangle[] = [];
  for (const region of regions) {
    const r = fitRegionWithRetries(region, mag, theta, w, h, settings, logNTests, logEpsilon);
    if (r) results.push(r);
  }
  return results;
}

// GPU-resident counterpart to computeLsdRectangles -- stages 0-3 (mag/theta,
// counting sort, region growing) stay identical/CPU (see this file's own
// header for why), only stage 4 + stage 5's first pass move to GPU
// (pipelineGPU/lsdFit.ts). Any region that pass rejects falls through to
// fitRegionWithRetries on CPU, same as the pure-CPU path. Returns null only
// if WebGPU itself is unavailable -- computeLsdRectanglesAuto below is what
// callers actually use, and handles that fallback.
export async function computeLsdRectanglesGPU(field: GradientField, settings: LsdSettings): Promise<LsdRectangle[] | null> {
  const { w, h } = field;
  const { mag, theta } = computeMagTheta(field);

  const order = orderPixelsByMagnitudeDescending(mag, settings.rhoNoiseThreshold, settings.magnitudeBuckets);
  const { regions } = growRegions(mag, theta, w, h, order, settings.toleranceDeg, settings.rhoNoiseThreshold);

  const maxDim = Math.max(w, h);
  const logNTests = settings.nfaTestExponent * Math.log(maxDim);
  const logEpsilon = Math.log(settings.nfaEpsilon);

  const gpuResults = await fitAndTestRegionsGPU(
    mag, theta, w, h, regions, settings.rhoNoiseThreshold, settings.toleranceDeg, logNTests, logEpsilon,
  );
  if (!gpuResults) return null;

  const results: LsdRectangle[] = [];
  for (let i = 0; i < regions.length; i++) {
    const region = regions[i];
    const g = gpuResults[i];
    if (g.accepted) {
      results.push({
        cx: g.cx, cy: g.cy, theta: g.theta, length: g.length, width: g.width,
        accepted: true, retries: 0, nfaLog10: g.nfaLog10, lineScore: nfaLog10ToLineScore(g.nfaLog10, logEpsilon),
        rawMembers: region.members,
      });
    } else {
      const r = fitRegionWithRetries(region, mag, theta, w, h, settings, logNTests, logEpsilon);
      if (r) results.push(r);
    }
  }
  return results;
}

// Single dispatch point both production callers (pipeline/votes.ts,
// overlays/lsdOverlay.ts) use -- centralizes the globalState.useGPULsdFit
// check once instead of duplicating it at each call site.
export async function computeLsdRectanglesAuto(field: GradientField, settings: LsdSettings): Promise<LsdRectangle[]> {
  if (!globalState.useGPULsdFit) return computeLsdRectangles(field, settings);
  return (await computeLsdRectanglesGPU(field, settings)) ?? computeLsdRectangles(field, settings);
}

// ── Adapter: LSD rectangles -> bucketFillJoin.ts's expected input shape ──
//
// pipeline/bucketFillJoin.ts's computeJoinWalk was built against
// bucketFillSegments.ts's own BucketFillSegment[] + regionId output, but
// only reads FIVE things off a segment: endAlongX/Y, endAgainstX/Y (via
// segmentLength and spawnPair) and lineScore (seeds each segment's merge
// confidence, see computeJoinWalk's own header) -- count/cx/cy/avgFx/avgFy
// are present in the type but unused by the join walk or
// computeCompositeLines. That makes this a thin, honest adapter rather than
// a real behavioral bridge:
// only ACCEPTED rectangles become segments (a rejected candidate isn't a
// real detection, shouldn't be treated as one to merge); a rectangle's own
// two tangent-axis ends (its long axis, already what LSD fits the line
// ALONG) become endAlong/endAgainst directly, no re-derivation needed the
// way bucketFillSegments.ts's own post-pass needed one; and regionId is
// built from EACH rectangle's rawMembers (stage 3's actual flood-filled
// pixels, not the rectangle's geometric footprint) -- this is what the join
// walk seeds its buffer with, so a front reacts the moment it enters
// another segment's real grown blob, matching exactly what the join walk's
// own header comment describes bucketFillSegments.ts's regionId as
// providing.
//
// avgFx/avgFy: approximated as the PERPENDICULAR of the rectangle's own
// tangent axis (LSD's theta is a line's long axis; bucketFillSegments.ts's
// avgFx/avgFy was specifically the GRADIENT direction, perpendicular to
// that), scaled by member count so its magnitude still roughly tracks
// region "mass" the way the original did -- populated for type completeness
// and in case a future consumer reads it, not because anything currently
// downstream (computeJoinWalk, computeCompositeLines) actually does.
export function lsdRectanglesToBucketFillShape(
  rects: readonly LsdRectangle[], w: number, h: number,
): { regionId: Int32Array; segments: BucketFillSegment[] } {
  const regionId = new Int32Array(w * h).fill(-1);
  const segments: BucketFillSegment[] = [];
  for (const r of rects) {
    if (!r.accepted) continue;
    const id = segments.length;
    for (let mi = 0; mi < r.rawMembers.length; mi++) regionId[r.rawMembers[mi]] = id;

    const ax = Math.cos(r.theta), ay = Math.sin(r.theta);
    const hl = r.length / 2;
    const count = r.rawMembers.length;
    segments.push({
      count, cx: r.cx, cy: r.cy,
      avgFx: -ay * count, avgFy: ax * count,
      endAlongX: r.cx + hl * ax, endAlongY: r.cy + hl * ay,
      endAgainstX: r.cx - hl * ax, endAgainstY: r.cy - hl * ay,
      lineScore: r.lineScore,
    });
  }
  return { regionId, segments };
}
