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
}

export interface JunctionField {
  w: number; h: number;
  cornerness: Float64Array; // Shi-Tomasi min-eigenvalue response, normalized to [0,1]
  hessianDet: Float64Array; // raw (unnormalized) Hessian determinant, sign is what matters
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
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const fxx = blurred[i + 1] - 2 * blurred[i] + blurred[i - 1];
      const fyy = blurred[i + w] - 2 * blurred[i] + blurred[i - w];
      const fxy = (blurred[i + w + 1] - blurred[i + w - 1] - blurred[i - w + 1] + blurred[i - w - 1]) / 4;
      hessianDetRaw[i] = fxx * fyy - fxy * fxy;
    }
  }
  const hessianDet = boxBlur(hessianDetRaw, w, h, Math.max(1, tensorRadius - 1));

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

  return { w, h, cornerness, hessianDet };
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
    kept.push({ x: c.x, y: c.y, type, strength: c.strength });
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
