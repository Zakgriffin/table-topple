// Detects and classifies grid-lattice junction points for the corner-mesh
// geometry pipeline (Option B). A random black/white cell pattern's grid
// intersections fall into exactly 4 topological types depending on the
// surrounding 2x2 cell colors (16 possible colorings): FLAT (all 4 cells
// match, 2/16), STRAIGHT_EDGE (one line passes straight through, no turn,
// 4/16), L_CORNER (a single 90-degree turn, 8/16 — the most common), and
// SADDLE (a true 4-way "X" like a checkerboard corner, 2/16).
//
// Only L_CORNER and SADDLE are well-localized in both dimensions — a
// genuine 2D "cornerness" peak exists at those points. STRAIGHT_EDGE points
// have only one well-determined coordinate (perpendicular to the edge) and
// none along it (the aperture problem), so they never produce a discrete
// detectable peak; this module classifies every pixel for debug display, but
// only returns discrete Junction points for L_CORNER/SADDLE.
//
// Uses two complementary per-pixel measures, both computed on a lightly
// blurred grayscale image (same discipline as estimateRotationRad — raw
// finite differences on a hard/aliased image are unreliable, see decode.ts):
//   - A windowed gradient structure tensor (Shi-Tomasi style) to find where
//     TWO independent edge directions cross nearby — this is what separates
//     "corner-like" (L_CORNER or SADDLE) points from straight edges (one
//     direction only) and flat regions (no signal), regardless of the
//     junction's rotation relative to the camera.
//   - The raw Hessian determinant's SIGN at that same point to tell
//     L_CORNER from SADDLE apart: a true saddle has one positive and one
//     negative principal curvature (det < 0), while a single blob-like
//     corner does not (det >= 0).

export type JunctionType = 'lcorner' | 'saddle';

export interface Junction {
  x: number; y: number;
  type: JunctionType;
  strength: number; // the (normalized) cornerness value at this point
  // The two local edge-line directions (radians, mod PI — a line has no
  // inherent direction). For SADDLE points these are recovered as two
  // INDEPENDENT (not-necessarily-orthogonal) angles via the generalized
  // two-crossing-lines Hessian factorization (see detectJunctions) — this
  // correctly reflects real perspective shear, where the true row/col edges
  // usually aren't exactly perpendicular. For L_CORNER points these are
  // still read off the structure tensor's eigenvectors, which are
  // orthogonal by construction (a symmetric matrix's eigenvectors always
  // are) — an approximation under shear, not yet fixed (see detectJunctions'
  // comment on why the saddle technique can't extend to L-corners, and
  // scripts/test-corner-axes-sheared.ts for what's actually validated).
  axis1: number; axis2: number;
}

export interface JunctionField {
  w: number; h: number;
  cornerness: Float64Array; // Shi-Tomasi min-eigenvalue response, normalized to [0,1]
  hessianDet: Float64Array; // raw (unnormalized) Hessian determinant, sign is what matters
  Sxx: Float64Array; Syy: Float64Array; Sxy: Float64Array; // structure tensor, for reading an L-corner's local axis directions (see Junction.axis1/axis2)
  Hxx: Float64Array; Hyy: Float64Array; Hxy: Float64Array; // blurred Hessian entries, for reading a SADDLE's local axis directions instead (structure tensor is degenerate there)
}

function boxBlur(src: Float64Array, w: number, h: number, radius: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          sum += src[yy * w + xx];
          count++;
        }
      }
      out[y * w + x] = sum / count;
    }
  }
  return out;
}

// Computes the dense per-pixel cornerness + Hessian-determinant fields.
// blurRadius softens the input before any derivatives are taken (stabilizes
// finite differences against pixel-grid aliasing); tensorRadius is the
// window the gradient structure tensor is averaged over (must be large
// enough to "see" both edge directions at a real corner, small enough not to
// blur adjacent junctions together — empirically, roughly a TENTH of the
// expected cell pitch works well; a third (the initial guess) over-widens
// the response into a near-flat plateau spanning several junctions' worth
// of pixels, which is what caused the misclassifications this was tuned
// against, see scripts/test-junctions.ts). Exposed rather than hardcoded
// since it's still genuinely scale-dependent and worth real-pattern tuning
// later, not just this synthetic-junction validation.
export function computeJunctionField(gray: Float64Array, w: number, h: number, blurRadius = 1, tensorRadius = 4): JunctionField {
  const blurred = boxBlur(gray, w, h, blurRadius);

  const fx = new Float64Array(w * h);
  const fy = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      fx[i] = blurred[i + 1] - blurred[i - 1];
      fy[i] = blurred[i + w] - blurred[i - w];
    }
  }

  // Structure tensor components, pre-averaging.
  const fxx_raw = new Float64Array(w * h);
  const fyy_raw = new Float64Array(w * h);
  const fxy_raw = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    fxx_raw[i] = fx[i] * fx[i];
    fyy_raw[i] = fy[i] * fy[i];
    fxy_raw[i] = fx[i] * fy[i];
  }
  const Sxx = boxBlur(fxx_raw, w, h, tensorRadius);
  const Syy = boxBlur(fyy_raw, w, h, tensorRadius);
  const Sxy = boxBlur(fxy_raw, w, h, tensorRadius);

  // Hessian (second derivatives) of the same blurred image. The raw,
  // per-pixel determinant's sign is only reliably negative very close to a
  // true saddle's exact center — a few pixels off (e.g. wherever
  // non-max-suppression's tie-breaking happens to land within a plateau) it
  // can flip positive even at a genuine saddle, misclassifying it as an
  // L-corner. Averaging the determinant over a window fixed this
  // empirically, but the margin is narrow: blurring by tensorRadius itself
  // is already enough to flip the sign back (the positive lobe surrounding
  // a saddle's negative core isn't far away) — one pixel less avoids it, see
  // scripts/test-junctions.ts.
  const hessianDetRaw = new Float64Array(w * h);
  const HxxRaw = new Float64Array(w * h), HyyRaw = new Float64Array(w * h), HxyRaw = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const fxx = blurred[i + 1] - 2 * blurred[i] + blurred[i - 1];
      const fyy = blurred[i + w] - 2 * blurred[i] + blurred[i - w];
      const fxy = (blurred[i + w + 1] - blurred[i + w - 1] - blurred[i - w + 1] + blurred[i - w - 1]) / 4;
      hessianDetRaw[i] = fxx * fyy - fxy * fxy;
      HxxRaw[i] = fxx; HyyRaw[i] = fyy; HxyRaw[i] = fxy;
    }
  }
  const hessianDet = boxBlur(hessianDetRaw, w, h, Math.max(1, tensorRadius - 1));
  // Individually-blurred Hessian entries, same radius as hessianDet — used
  // to read a SADDLE's local axis directions (see Junction.axis1/axis2):
  // the structure tensor (Sxx/Syy/Sxy) is exactly isotropic at a true saddle
  // (its 4-fold symmetry forces Sxx==Syy and Sxy==0 there — confirmed
  // numerically, not a tuning issue), so its eigenvector angle is
  // meaningless noise for this type. The Hessian doesn't have that
  // degeneracy (hessianDet<0 there means real, distinct eigenvalues), and
  // is exactly what "opposite-side gradients" resolves to mathematically:
  // near a saddle, g(p) is locally linear in p (g(p) ~ H*p), so it directly
  // encodes how the gradient differs across the point instead of averaging
  // opposite-signed contributions into cancellation the way the sign-blind
  // structure tensor does.
  const Hxx = boxBlur(HxxRaw, w, h, Math.max(1, tensorRadius - 1));
  const Hyy = boxBlur(HyyRaw, w, h, Math.max(1, tensorRadius - 1));
  const Hxy = boxBlur(HxyRaw, w, h, Math.max(1, tensorRadius - 1));

  // Shi-Tomasi cornerness: the SMALLER eigenvalue of the structure tensor —
  // high only when BOTH principal directions carry real gradient energy.
  const cornerness = new Float64Array(w * h);
  let maxCornerness = 0;
  for (let i = 0; i < w * h; i++) {
    const trace = Sxx[i] + Syy[i];
    const det = Sxx[i] * Syy[i] - Sxy[i] * Sxy[i];
    const disc = Math.max(0, (trace / 2) * (trace / 2) - det);
    const lambdaMin = trace / 2 - Math.sqrt(disc);
    cornerness[i] = lambdaMin;
    if (lambdaMin > maxCornerness) maxCornerness = lambdaMin;
  }
  if (maxCornerness > 0) {
    for (let i = 0; i < w * h; i++) cornerness[i] /= maxCornerness;
  }

  return { w, h, cornerness, hessianDet, Sxx, Syy, Sxy, Hxx, Hyy, Hxy };
}

// Non-max suppression over the cornerness field, then classifies each
// surviving peak as lcorner/saddle via the Hessian determinant's sign.
// threshold is a fraction of the field's own max (scale-independent, same
// normalization approach as PuzzleBoard's detector). minDistance needs to be
// meaningfully larger than tensorRadius — the response near a true corner is
// a near-flat plateau roughly tensorRadius-wide (not a sharp single-pixel
// peak), so a too-small minDistance lets multiple redundant points survive
// from the same plateau instead of collapsing to one (empirically ~2.5x
// tensorRadius avoids that while still keeping genuinely adjacent grid
// junctions separate — see scripts/test-junctions.ts).
// Reads the local row/col axis directions at (x,y) — rounded to the nearest
// pixel, since Sxx/Syy/Sxy/Hxx/Hyy/Hxy are dense per-pixel arrays. Exposed
// standalone (not just inlined in detectJunctions) because the coarse peak
// location detectJunctions works from is often several pixels off the true
// corner (same known issue as L-corner's coarse-vs-refined position, see
// refineJunctionSubPixel's docs) — under real shear this is enough to bias
// the saddle formula noticeably (confirmed via
// scripts/test-corner-axes-sheared.ts: two of its worst cases traced back
// to exactly this). Callers that have already run refineJunctionSubPixel
// should call this AGAIN at the refined position for a more accurate
// result rather than trusting detectJunctions' coarse-position estimate.
export function computeAxisDirections(field: JunctionField, type: JunctionType, x: number, y: number): { axis1: number; axis2: number } {
  const { w, h, Sxx, Syy, Sxy, Hxx, Hyy, Hxy } = field;
  const px = Math.max(0, Math.min(w - 1, Math.round(x)));
  const py = Math.max(0, Math.min(h - 1, Math.round(y)));
  const idx = py * w + px;
  // Both branches read a symmetric 2x2 matrix's eigenvector angle (closed
  // form: 0.5*atan2(2*b, a-d)) then apply a fixed 45deg correction — see
  // Junction.axis1/axis2's doc and JunctionField.Hxx/Hyy/Hxy's doc for why
  // each type needs a DIFFERENT source matrix (structure tensor is
  // degenerate at a saddle) and why the correction is exactly 45deg either
  // way (an L-corner's shape has a diagonal mirror symmetry; the Hessian's
  // principal-curvature axes for a saddle are the diagonals of its edges)
  // — confirmed numerically in scripts/test-corner-axes.ts, not a guess.
  let axis1: number, axis2: number;
  if (type === 'saddle') {
    // Generalized two-crossing-lines factorization: a saddle's intensity
    // locally looks like f = A*L1*L2, the product of two linear forms
    // whose zero sets are the two edges — L_k = cos(n_k)x + sin(n_k)y,
    // where n_k is edge k's NORMAL angle (not necessarily 90deg apart,
    // unlike a plain eigenvector pair). Matching fxx/fyy/fxy against the
    // expansion of that product gives n1+n2 directly (from fxy and
    // fxx-fyy) and n1-n2 via fxx+fyy — this is exactly the discriminant
    // that also defines hessianDet (R^2-A^2 = fxx*fyy-fxy^2 = hessianDet
    // exactly), which is why this factorization only has a real solution
    // when hessianDet<0 — i.e. this is provably a saddle-only technique,
    // not something that extends to L-corners by any scaling (see
    // Junction.axis1/axis2's doc). +90deg converts each recovered normal
    // to the edge's own tangent direction (what we actually want to draw).
    const fxx = Hxx[idx], fyy = Hyy[idx], fxy = Hxy[idx];
    const P = (fxx - fyy) / 2, Q = fxy;
    const sumAngle = Math.atan2(Q, P); // n1 + n2
    const A = Math.hypot(P, Q);
    const R = (fxx + fyy) / 2;
    const ratio = A > 1e-9 ? Math.max(-1, Math.min(1, R / A)) : 0;
    const diffAngle = Math.acos(ratio); // |n1 - n2|
    axis1 = (sumAngle + diffAngle) / 2 + Math.PI / 2;
    axis2 = (sumAngle - diffAngle) / 2 + Math.PI / 2;
  } else {
    const theta = 0.5 * Math.atan2(2 * Sxy[idx], Sxx[idx] - Syy[idx]) - Math.PI / 4;
    axis1 = theta; axis2 = theta + Math.PI / 2;
  }
  return { axis1, axis2 };
}

export function detectJunctions(field: JunctionField, threshold = 0.15, minDistance = 10): Junction[] {
  const { w, h, cornerness, hessianDet } = field;
  const candidates: { x: number; y: number; strength: number }[] = [];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const v = cornerness[i];
      if (v < threshold) continue;
      let isPeak = true;
      for (let dy = -1; dy <= 1 && isPeak; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (cornerness[(y + dy) * w + (x + dx)] > v) { isPeak = false; break; }
        }
      }
      if (isPeak) candidates.push({ x, y, strength: v });
    }
  }
  candidates.sort((a, b) => b.strength - a.strength);

  const kept: Junction[] = [];
  const minDistSq = minDistance * minDistance;
  for (const c of candidates) {
    if (kept.some(k => (k.x - c.x) ** 2 + (k.y - c.y) ** 2 < minDistSq)) continue;
    const type: JunctionType = hessianDet[c.y * w + c.x] < 0 ? 'saddle' : 'lcorner';
    // Coarse-position estimate — see computeAxisDirections' doc: callers
    // that go on to sub-pixel refine this junction should recompute axes at
    // the refined position for a meaningfully better result under shear.
    const { axis1, axis2 } = computeAxisDirections(field, type, c.x, c.y);
    kept.push({ x: c.x, y: c.y, type, strength: c.strength, axis1, axis2 });
  }
  return kept;
}

// Refines a coarse junction position to sub-pixel accuracy — needed for
// L_CORNER points in particular, whose coarse cornerness peak sits a few
// pixels off the true lattice point (see detectJunctions's caller-facing
// docs and scripts/test-junctions.ts's positional offset numbers): unlike a
// saddle, a plain corner lacks the 4-fold symmetry that keeps the coarse
// peak centered.
//
// Uses the same iterative technique OpenCV's cornerSubPix (originally
// Förstner's method) is built on, rather than a hand-tuned offset
// correction: the true corner is the point q that best satisfies, for every
// nearby edge pixel p with gradient g(p), that g(p) is perpendicular to
// (p - q) — i.e. q lies along the tangent line implied by every edge pixel
// in the window. That's a weighted least-squares linear system in q, so it
// generalizes to L-corners (2 edge tangents) and saddles (4) alike without
// needing separate per-type logic, unlike a fitted offset would.
export function refineJunctionSubPixel(gray: Float64Array, w: number, h: number, cx: number, cy: number, windowRadius = 6, iterations = 3): { x: number; y: number } {
  const blurred = boxBlur(gray, w, h, 1);
  let qx = cx, qy = cy;
  for (let iter = 0; iter < iterations; iter++) {
    let Mxx = 0, Mxy = 0, Myy = 0, bx = 0, by = 0;
    const x0 = Math.max(1, Math.round(qx - windowRadius)), x1 = Math.min(w - 2, Math.round(qx + windowRadius));
    const y0 = Math.max(1, Math.round(qy - windowRadius)), y1 = Math.min(h - 2, Math.round(qy + windowRadius));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const gx = blurred[y * w + x + 1] - blurred[y * w + x - 1];
        const gy = blurred[(y + 1) * w + x] - blurred[(y - 1) * w + x];
        const mag2 = gx * gx + gy * gy;
        if (mag2 < 1) continue; // skip near-flat pixels — uninformative, only adds noise
        const dot = gx * x + gy * y;
        Mxx += mag2 * gx * gx; Mxy += mag2 * gx * gy; Myy += mag2 * gy * gy;
        bx += mag2 * gx * dot; by += mag2 * gy * dot;
      }
    }
    const det = Mxx * Myy - Mxy * Mxy;
    if (Math.abs(det) < 1e-9) break; // degenerate window (e.g. all gradients parallel) — keep current estimate
    qx = (Myy * bx - Mxy * by) / det;
    qy = (Mxx * by - Mxy * bx) / det;
  }
  return { x: qx, y: qy };
}
