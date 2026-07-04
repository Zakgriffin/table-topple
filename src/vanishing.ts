// Estimates the two vanishing points (or, for weak perspective, pure
// directions) that the De Bruijn grid's row-lines and column-lines converge
// to, directly from the dense image gradient field. A flat grid under
// perspective projects to exactly two pencils of straight lines, each
// converging to one vanishing point (or staying parallel at true fronto
// parallel incidence) — a much stronger global constraint on corner
// positions than a single averaged rotation angle, which only captures the
// *mean* direction and is wrong as soon as the two axes have visibly
// different apparent angles (i.e. as soon as there's real tilt). This
// replaces estimateRotationRad's role as the geometry prior feeding mesh
// construction.
//
// Approach:
//   1. Build a dense gradient field; every edge pixel above a magnitude
//      threshold is treated as one sample of an implicit grid-line, tangent
//      to the local gradient's perpendicular.
//   2. Split edge pixels into the two axis families via a coarse orientation
//      histogram (allowed to be imperfect — step 3's RANSAC cleans it up,
//      it only needs to be right on average).
//   3. Within each family, RANSAC over random line pairs to find the
//      dominant intersection, expressed in homogeneous coordinates so a true
//      vanishing point AT INFINITY (the grid axis stays parallel in image
//      space, e.g. near-fronto-parallel views) falls out of the same math
//      rather than needing a separate detection path — only the *reporting*
//      (finite point vs. direction) branches at the end.
//
// All internal math is done in coordinates normalized by roughly the image's
// half-diagonal, so "close to infinity" is a scale-appropriate,
// image-size-independent threshold rather than an arbitrary pixel count.
//
// KNOWN LIMITATION (partially addressed, still real): the family split used
// to assume a single static histogram split was enough, which breaks down
// at steep tilt (a family converging toward a nearby VP has lines whose
// LOCAL angle varies a lot across the frame by construction). EM-style
// reassignment was added (fit each family's VP, reassign every pooled
// sample to whichever family's CURRENT fit it agrees with better via the
// unified homogeneous residual, refit, repeat a few rounds) — but measured
// via scripts/test-homography-vp.ts's Part 2 (the metric that actually
// matters: downstream homography held-out prediction error using ESTIMATED
// vs ground-truth VPs), it did not meaningfully improve accuracy: ~20-140px
// error at 20-30deg tilt and complete failure at 45-60deg, essentially
// unchanged from before EM was added. Also tried widening INF_THRESH to
// bias ambiguous finite/infinite calls toward the numerically stable
// "infinite" answer — also no meaningful change on the downstream metric.
// scripts/test-homography-vp.ts's Part 1 proves the homography math itself
// is near-exact given accurate VPs, so the remaining gap is squarely in
// this module's estimation accuracy at moderate-to-steep tilt, and remains
// unresolved. Time-boxed and deliberately left as-is rather than continuing
// to grind on it — see conversation for the decision to move on to live
// testing instead.

export interface VanishingPoint {
  finite: boolean;
  x: number; y: number;   // meaningful only if finite (pixel coords)
  angle: number;           // meaningful only if !finite — line direction, radians, mod PI
  inliers: number;
  total: number;
}

interface Line { a: number; b: number; c: number; } // a*x + b*y - c = 0, (a,b) unit length, in normalized coords

function computeGradient(gray: Float64Array, w: number, h: number, blurRadius = 1): { fx: Float64Array; fy: Float64Array } {
  const blurred = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let dy = -blurRadius; dy <= blurRadius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -blurRadius; dx <= blurRadius; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          sum += gray[yy * w + xx];
          count++;
        }
      }
      blurred[y * w + x] = sum / count;
    }
  }
  const fx = new Float64Array(w * h);
  const fy = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      fx[i] = blurred[i + 1] - blurred[i - 1];
      fy[i] = blurred[i + w] - blurred[i - w];
    }
  }
  return { fx, fy };
}

function circularMeanModPi(angles: number[], weights: number[]): number {
  // Doubling the angle turns the mod-PI ambiguity (a line has no direction)
  // into a proper mod-2PI quantity that's safe to average, then halve back.
  let sx = 0, sy = 0;
  for (let i = 0; i < angles.length; i++) {
    sx += weights[i] * Math.cos(2 * angles[i]);
    sy += weights[i] * Math.sin(2 * angles[i]);
  }
  return Math.atan2(sy, sx) / 2;
}

interface FamilyFit { finite: boolean; X: number; Y: number; angle: number; inliers: number; Vx: number; Vy: number; Vw: number; }

function fitFamily(lines: Line[], rng: () => number): FamilyFit | null {
  if (lines.length < 8) return null;

  const ITERS = Math.min(400, lines.length * lines.length);
  const INF_THRESH = 0.02;   // |Vw| below this (on a unit-norm homogeneous point) => infinite
  const DIST_TOL = 3 / SCALE_HINT; // ~3px worth, patched below via closure
  const ANGLE_TOL = 0.05;    // ~2.9deg, for infinite-candidate inlier test

  let bestInliers = -1;
  let bestIsFinite = true;
  let bestX = 0, bestY = 0, bestAngle = 0;

  for (let iter = 0; iter < ITERS; iter++) {
    const i = Math.floor(rng() * lines.length);
    let j = Math.floor(rng() * lines.length);
    if (j === i) j = (j + 1) % lines.length;
    const Li = lines[i], Lj = lines[j];

    // Homogeneous line coefficients are (a,b,-c); intersection = cross product.
    let Vx = Li.c * Lj.b - Li.b * Lj.c;
    let Vy = Li.a * Lj.c - Li.c * Lj.a;
    let Vw = Li.a * Lj.b - Lj.a * Li.b;
    const norm = Math.hypot(Vx, Vy, Vw);
    if (norm < 1e-12) continue; // near-identical lines, unstable
    Vx /= norm; Vy /= norm; Vw /= norm;

    let inliers = 0;
    if (Math.abs(Vw) < INF_THRESH) {
      // Direction-at-infinity candidate.
      let dx = Vx, dy = Vy;
      const dnorm = Math.hypot(dx, dy);
      if (dnorm < 1e-9) continue;
      dx /= dnorm; dy /= dnorm;
      for (const L of lines) {
        if (Math.abs(L.a * dx + L.b * dy) < ANGLE_TOL) inliers++;
      }
      if (inliers > bestInliers) {
        bestInliers = inliers; bestIsFinite = false;
        bestAngle = Math.atan2(dx, -dy); // line direction angle (perp to normal), mod handled by caller
      }
    } else {
      const X = Vx / Vw, Y = Vy / Vw;
      for (const L of lines) {
        if (Math.abs(L.a * X + L.b * Y - L.c) < DIST_TOL) inliers++;
      }
      if (inliers > bestInliers) {
        bestInliers = inliers; bestIsFinite = true;
        bestX = X; bestY = Y;
      }
    }
  }

  if (bestInliers < 0) return null;

  // Gather the winning candidate's inlier set (its own finite/infinite guess
  // only matters for picking WHICH lines belong together — a single random
  // pair is far too noisy a basis for the finite-vs-infinite decision
  // itself, since near-parallel-but-not-quite families routinely produce
  // one-off pair intersections that swing wildly between "clearly finite"
  // and "clearly infinite" purely from which two lines got picked).
  const inlierLines: Line[] = bestIsFinite
    ? lines.filter(L => Math.abs(L.a * bestX + L.b * bestY - L.c) < DIST_TOL)
    : (() => {
        const dx0 = Math.sin(bestAngle), dy0 = -Math.cos(bestAngle);
        return lines.filter(L => Math.abs(L.a * dx0 + L.b * dy0) < ANGLE_TOL);
      })();
  if (inlierLines.length < 4) return null;

  // Authoritative fit: total-least-squares over ALL inliers at once, solved
  // as the smallest-eigenvalue eigenvector of M = sum(L_i L_i^T) in
  // homogeneous coordinates (L_i = (a,b,-c), V=(Vx,Vy,Vw) with L_i . V = 0
  // for a perfect common intersection). This is what naturally and stably
  // distinguishes finite from infinite from the FULL inlier set: Vw comes
  // out small only when the whole family is genuinely close to parallel,
  // not as an artifact of which single pair was sampled.
  let m00 = 0, m01 = 0, m02 = 0, m11 = 0, m12 = 0, m22 = 0;
  for (const L of inlierLines) {
    const nc = -L.c;
    m00 += L.a * L.a; m01 += L.a * L.b; m02 += L.a * nc;
    m11 += L.b * L.b; m12 += L.b * nc; m22 += nc * nc;
  }
  const [Vx, Vy, Vw] = smallestEigenvector3x3(m00, m01, m02, m11, m12, m22);

  if (Math.abs(Vw) < INF_THRESH) {
    let dx = Vx, dy = Vy;
    const dnorm = Math.hypot(dx, dy);
    if (dnorm < 1e-9) return null;
    dx /= dnorm; dy /= dnorm;
    const angles: number[] = [], weights: number[] = [];
    for (const L of inlierLines) {
      angles.push(Math.atan2(L.a, -L.b));
      weights.push(1);
    }
    return { finite: false, X: 0, Y: 0, angle: circularMeanModPi(angles, weights), inliers: inlierLines.length, Vx: dx, Vy: dy, Vw: 0 };
  }
  const vnorm = Math.hypot(Vx, Vy, Vw);
  return { finite: true, X: Vx / Vw, Y: Vy / Vw, angle: 0, inliers: inlierLines.length, Vx: Vx / vnorm, Vy: Vy / vnorm, Vw: Vw / vnorm };
}

// Unit eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix,
// given by its upper-triangular entries. Closed-form (Smith, 1961) rather
// than iterative — exact and fast for this fixed-size case.
function smallestEigenvector3x3(m00: number, m01: number, m02: number, m11: number, m12: number, m22: number): [number, number, number] {
  const p1 = m01 * m01 + m02 * m02 + m12 * m12;
  if (p1 < 1e-18) {
    if (m00 <= m11 && m00 <= m22) return [1, 0, 0];
    if (m11 <= m00 && m11 <= m22) return [0, 1, 0];
    return [0, 0, 1];
  }
  const q = (m00 + m11 + m22) / 3;
  const p2 = (m00 - q) ** 2 + (m11 - q) ** 2 + (m22 - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  const b00 = (m00 - q) / p, b01 = m01 / p, b02 = m02 / p, b11 = (m11 - q) / p, b12 = m12 / p, b22 = (m22 - q) / p;
  const detB = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
  const r = Math.max(-1, Math.min(1, detB / 2));
  const phi = Math.acos(r) / 3;
  const eig1 = q + 2 * p * Math.cos(phi);
  const eig3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  const eig2 = 3 * q - eig1 - eig3;
  const lambdaMin = Math.min(eig1, eig2, eig3);

  const a00 = m00 - lambdaMin, a11 = m11 - lambdaMin, a22 = m22 - lambdaMin;
  const cross = (u: [number, number, number], v: [number, number, number]): [number, number, number] =>
    [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const r0: [number, number, number] = [a00, m01, m02];
  const r1: [number, number, number] = [m01, a11, m12];
  const r2: [number, number, number] = [m02, m12, a22];
  let best: [number, number, number] = [0, 0, 1], bestNorm = -1;
  for (const c of [cross(r0, r1), cross(r0, r2), cross(r1, r2)]) {
    const n = Math.hypot(c[0], c[1], c[2]);
    if (n > bestNorm) { best = c; bestNorm = n; }
  }
  if (bestNorm < 1e-12) return [0, 0, 1];
  return [best[0] / bestNorm, best[1] / bestNorm, best[2] / bestNorm];
}

// Set by estimateVanishingPoints before calling fitFamily, since DIST_TOL
// needs the image scale but fitFamily is defined above its only caller.
let SCALE_HINT = 100;

// Simple deterministic LCG so tests are reproducible without relying on
// Math.random's non-seedable global state.
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function estimateVanishingPoints(
  gray: Float64Array, w: number, h: number,
  magThresholdFrac = 0.1, maxSamplesPerFamily = 2000, seed = 12345,
): { vp1: VanishingPoint; vp2: VanishingPoint } | null {
  const { fx, fy } = computeGradient(gray, w, h, 1);
  const cx = w / 2, cy = h / 2;
  const scale = (w + h) / 4;
  SCALE_HINT = scale;

  let maxMag2 = 0;
  for (let i = 0; i < w * h; i++) {
    const m = fx[i] * fx[i] + fy[i] * fy[i];
    if (m > maxMag2) maxMag2 = m;
  }
  if (maxMag2 <= 0) return null;
  const magCut2 = maxMag2 * magThresholdFrac * magThresholdFrac;

  // Collect edge samples: position (for family split via angle only, no
  // position needed there) + implicit line coefficients in normalized coords.
  const HIST_BINS = 180;
  const hist = new Float64Array(HIST_BINS);
  type Sample = { line: Line; binAngle: number };
  const samples: Sample[] = [];

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = fx[i], gy = fy[i];
      const mag2 = gx * gx + gy * gy;
      if (mag2 < magCut2) continue;
      const mag = Math.sqrt(mag2);
      const a = gx / mag, b = gy / mag;
      const nx = (x - cx) / scale, ny = (y - cy) / scale;
      const c = a * nx + b * ny;
      let binAngle = Math.atan2(gy, gx) % Math.PI;
      if (binAngle < 0) binAngle += Math.PI;
      const bin = Math.min(HIST_BINS - 1, Math.floor((binAngle / Math.PI) * HIST_BINS));
      hist[bin] += mag;
      samples.push({ line: { a, b, c }, binAngle });
    }
  }
  if (samples.length < 16) return null;

  // Find the two dominant orientation peaks (allowed to be a rough split —
  // RANSAC per-family cleans up misclassified pixels).
  const peak1Bin = argmax(hist);
  const suppressed = hist.slice();
  const suppressRadius = 10;
  for (let d = -suppressRadius; d <= suppressRadius; d++) {
    const b = ((peak1Bin + d) % HIST_BINS + HIST_BINS) % HIST_BINS;
    suppressed[b] = 0;
  }
  const peak2Bin = argmax(suppressed);
  const peak1Angle = ((peak1Bin + 0.5) / HIST_BINS) * Math.PI;
  const peak2Angle = ((peak2Bin + 0.5) / HIST_BINS) * Math.PI;

  const family1: Line[] = [], family2: Line[] = [];
  for (const s of samples) {
    const d1 = angleDistModPi(s.binAngle, peak1Angle);
    const d2 = angleDistModPi(s.binAngle, peak2Angle);
    (d1 <= d2 ? family1 : family2).push(s.line);
  }

  const rng = makeRng(seed);
  let fam1 = subsample(family1, maxSamplesPerFamily, rng);
  let fam2 = subsample(family2, maxSamplesPerFamily, rng);

  let fit1 = fitFamily(fam1, rng);
  let fit2 = fitFamily(fam2, rng);
  if (!fit1 || !fit2) return null;

  // EM-style reassignment (see module doc's KNOWN LIMITATION note): the
  // initial split above assumes each family is angularly tight, which fails
  // at steep tilt where a genuinely-converging family's local angle varies a
  // lot across the frame. Now that we have an actual fitted VP per family,
  // reassign every pooled sample (not just the initial split) to whichever
  // family's CURRENT fit it agrees with better, and refit — a few rounds of
  // this corrects the initial split's tilt-dependent misclassifications
  // instead of being stuck with them. Residual is the homogeneous line-point
  // incidence |a*Vx+b*Vy-c*Vw| with both L and V unit-normalized, which is
  // directly comparable whether the candidate is finite or at infinity, so
  // reassignment doesn't need separate finite/infinite logic either.
  const EM_ROUNDS = 3;
  for (let round = 0; round < EM_ROUNDS; round++) {
    const newFam1: Line[] = [], newFam2: Line[] = [];
    for (const s of samples) {
      const L = s.line;
      const r1 = Math.abs(L.a * fit1.Vx + L.b * fit1.Vy - L.c * fit1.Vw);
      const r2 = Math.abs(L.a * fit2.Vx + L.b * fit2.Vy - L.c * fit2.Vw);
      (r1 <= r2 ? newFam1 : newFam2).push(L);
    }
    fam1 = subsample(newFam1, maxSamplesPerFamily, rng);
    fam2 = subsample(newFam2, maxSamplesPerFamily, rng);
    const refit1 = fitFamily(fam1, rng);
    const refit2 = fitFamily(fam2, rng);
    if (!refit1 || !refit2) break; // keep the last good fit rather than fail outright
    fit1 = refit1; fit2 = refit2;
  }

  const toVP = (fit: FamilyFit, total: number): VanishingPoint =>
    fit.finite
      ? { finite: true, x: fit.X * scale + cx, y: fit.Y * scale + cy, angle: 0, inliers: fit.inliers, total }
      : { finite: false, x: 0, y: 0, angle: fit.angle, inliers: fit.inliers, total };

  return { vp1: toVP(fit1, fam1.length), vp2: toVP(fit2, fam2.length) };
}

function argmax(arr: Float64Array): number {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function angleDistModPi(a: number, b: number): number {
  let d = Math.abs(a - b) % Math.PI;
  if (d > Math.PI / 2) d = Math.PI - d;
  return d;
}

function subsample<T>(arr: T[], max: number, rng: () => number): T[] {
  if (arr.length <= max) return arr;
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out.slice(0, max);
}
