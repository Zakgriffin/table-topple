import type { CollectSettings, GrowSettings, LsdFitSettings } from './pose.ts';

// ── The CPU twin ──────────────────────────────────────────────────────────
//
// A reference implementation for the stages that synthetic ground truth cannot
// score. Ground truth knows where the camera was, so it can check the POSE; it
// has nothing to say about whether pixel 8,412 ended up in the right connected
// component. For those stages the only available oracle is a second
// implementation.
//
// ── IT IS A DIFFERENT ALGORITHM ON PURPOSE ──
//
// The shader computes components by iterated hook-and-compress: a parallel
// fixpoint reached in O(log n) rounds. This computes them by breadth-first
// search from each unvisited seed. Same answer, unrelated mechanism.
//
// That is the whole value. A twin transcribed line-by-line from the shader
// shares the shader's mistakes and will happily agree with a wrong answer -- it
// tests the port, not the algorithm. Rewriting from the DEFINITION ("components
// of the symmetric compatibility relation, each labelled by its smallest member
// index") is what makes disagreement mean something.
//
// ── WHY THE TWO CAN BE COMPARED EXACTLY ──
//
// hook takes the minimum neighbouring label, so at the fixpoint every member of
// a component carries the smallest pixel index in it. BFS labelling by component
// minimum lands on the same convention, so the two produce identical label
// ARRAYS, not merely equivalent partitions. Nothing has to canonicalize.
//
// The one place they can legitimately differ is a neighbour pair whose dot sits
// within float slop of cos(tolerance): the GPU decides in f32, this decides in
// f64, and they may fall on opposite sides. `marginalPairs` counts those, so a
// test can establish the comparison was decisive before trusting equality.
//
// ── WHAT THIS TWIN STRUCTURALLY CANNOT SEE ──
//
// Grow's only use of (ux, uy) is a dot product between two of them, and a dot is
// invariant under rotating BOTH operands. So every global rotation of the
// level-line field -- swapping the components, negating both, using the gradient
// itself instead of its perpendicular -- leaves the labelling bit-identical.
// Verified by mutation: flipping init to `ux = gy*inv, uy = -gx*inv` changes
// nothing here and every test still passes.
//
// That is not a gap to plug, it is a true property of the stage, and it is the
// same one the shader's own comment records from when the bug was real. The
// convention is only observable downstream, where lsdFit uses the direction
// itself rather than a pairwise comparison. Do not add a test here that claims
// to check it.

/** The 2x2 block gradient. Mirrors GRADIENT_WGSL's definition, not its code. */
export function cpuGradient(gray: Float32Array | Float64Array, w: number, h: number): {
  fx: Float32Array; fy: Float32Array;
} {
  const fx = new Float32Array(w * h), fy = new Float32Array(w * h);
  for (let y = 0; y + 1 < h; y++) {
    for (let x = 0; x + 1 < w; x++) {
      const i = y * w + x;
      const g00 = gray[i]!, g10 = gray[i + 1]!, g01 = gray[i + w]!, g11 = gray[i + 1 + w]!;
      fx[i] = ((g10 + g11) - (g00 + g01)) / 510;
      fy[i] = ((g01 + g11) - (g00 + g10)) / 510;
    }
  }
  // Last row and column stay zero -- the block has no bottom-right neighbour.
  return { fx, fy };
}

const NDX = [1, 1, 0, -1, -1, -1, 0, 1];
const NDY = [0, 1, 1, 1, 0, -1, -1, -1];

export interface CpuGrow {
  ux: Float32Array;
  uy: Float32Array;
  /** Component id = smallest pixel index in the component; -1 = ineligible. */
  label: Int32Array;
  /**
   * Neighbour pairs whose compatibility dot lands within 1e-4 of the threshold.
   * Zero means every edge decision was decisive in both precisions, which is
   * what licenses asserting the two label arrays are exactly equal.
   */
  marginalPairs: number;
}

export function cpuGrow(
  fx: Float32Array, fy: Float32Array, w: number, h: number, s: GrowSettings,
): CpuGrow {
  const n = w * h;
  const ux = new Float32Array(n), uy = new Float32Array(n);
  const label = new Int32Array(n).fill(-1);
  const eligible = new Uint8Array(n);

  // Same eligibility test and same level-line convention as init: the level line
  // runs PERPENDICULAR to the gradient, so (ux, uy) = (-fy, fx).
  const rhoLowSq = s.rhoLow >= 0 ? s.rhoLow * s.rhoLow : -Infinity;
  for (let i = 0; i < n; i++) {
    const gx = fx[i]!, gy = fy[i]!;
    const m2 = gx * gx + gy * gy;
    if (m2 > rhoLowSq) {
      const inv = 1 / Math.sqrt(m2);
      ux[i] = -gy * inv;
      uy[i] = gx * inv;
      eligible[i] = 1;
    }
  }

  const cosTol = Math.cos((s.toleranceDeg * Math.PI) / 180);
  let marginalPairs = 0;

  // BFS from each unvisited eligible pixel. Because the compatibility relation
  // is symmetric -- it is a dot product, and swapping the operands cannot change
  // it -- the reachable set from any seed IS the component, and the seed order
  // cannot affect the partition.
  const seen = new Uint8Array(n);
  const queue = new Int32Array(n);
  const members: number[] = [];
  for (let seed = 0; seed < n; seed++) {
    if (!eligible[seed] || seen[seed]) continue;
    let head = 0, tail = 0;
    queue[tail++] = seed;
    seen[seed] = 1;
    members.length = 0;

    while (head < tail) {
      const i = queue[head++]!;
      members.push(i);
      const x = i % w, y = (i / w) | 0;
      const ci = ux[i]!, si = uy[i]!;
      for (let k = 0; k < 8; k++) {
        const nx = x + NDX[k]!, ny = y + NDY[k]!;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const j = ny * w + nx;
        if (!eligible[j]) continue;
        const dot = ci * ux[j]! + si * uy[j]!;
        if (Math.abs(dot - cosTol) < 1e-4) marginalPairs++;
        if (dot < cosTol) continue;
        if (seen[j]) continue;
        seen[j] = 1;
        queue[tail++] = j;
      }
    }

    // The seed is the smallest index in the component: the outer loop ascends,
    // and anything smaller would already have claimed this component.
    for (const i of members) label[i] = seed;
  }

  return { ux, uy, label, marginalPairs };
}

// ── Collect ───────────────────────────────────────────────────────────────

export interface CpuCollect {
  regionCount: number;
  memberCount: number;
  /** Per region: start index into `members`. */
  regionOffsets: Uint32Array;
  regionSizes: Uint32Array;
  /** CSR payload: pixel indices, grouped by region, ascending within a region. */
  members: Uint32Array;
  /** Per region: mean level-line direction, (x, y) interleaved. */
  meanDirs: Float32Array;
}

/**
 * Regions, straight from the definition.
 *
 * The GPU reaches this answer through a tally, a vec2 prefix scan, an atomic
 * scatter and a per-region sort. This bucket-sorts by label in one pass. The
 * only thing the two share is the SPECIFICATION -- ascending label order,
 * hysteresis then size, members ascending within a region -- which is what makes
 * the comparison worth running.
 *
 * Region ordering is deterministic on both sides (ascending label), so the two
 * can be compared region-for-region with no canonicalization. That is the whole
 * reason collect scans rather than atomically appending.
 */
export function cpuCollect(
  fx: Float32Array, fy: Float32Array, label: Int32Array,
  w: number, h: number, s: CollectSettings & { maxRegions: number },
): CpuCollect {
  const n = w * h;
  const strong = new Set<number>();
  const counts = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const l = label[i]!;
    if (l < 0) continue;
    counts.set(l, (counts.get(l) ?? 0) + 1);
    const gx = fx[i]!, gy = fy[i]!;
    if (gx * gx + gy * gy > (s.rhoHigh >= 0 ? s.rhoHigh * s.rhoHigh : -Infinity)) strong.add(l);
  }

  // Ascending label order IS the region order.
  const keptLabels = [...counts.keys()]
    .filter((l) => strong.has(l) && counts.get(l)! >= s.minRegionSize)
    .sort((a, b) => a - b);

  const regionCount = keptLabels.length;
  const kept = keptLabels.slice(0, s.maxRegions);
  const regionOf = new Map<number, number>();
  kept.forEach((l, r) => regionOf.set(l, r));

  const regionOffsets = new Uint32Array(kept.length);
  const regionSizes = new Uint32Array(kept.length);
  let off = 0;
  for (let r = 0; r < kept.length; r++) {
    regionOffsets[r] = off;
    regionSizes[r] = counts.get(kept[r]!)!;
    off += regionSizes[r]!;
  }
  // The member TOTAL counts every kept region, including any past maxRegions --
  // it is the scan's grand total, which the overflow clamp happens after.
  let memberCount = 0;
  for (const l of keptLabels) memberCount += counts.get(l)!;

  // One ascending pass over the image, so each region's members land ascending
  // without a sort.
  const members = new Uint32Array(n);
  const cursor = new Uint32Array(kept.length);
  for (let i = 0; i < n; i++) {
    const l = label[i]!;
    if (l < 0) continue;
    const r = regionOf.get(l);
    if (r === undefined) continue;
    members[regionOffsets[r]! + cursor[r]!++] = i;
  }

  const meanDirs = new Float32Array(kept.length * 2);
  for (let r = 0; r < kept.length; r++) {
    let sx = 0, sy = 0;
    for (let k = 0; k < regionSizes[r]!; k++) {
      const p = members[regionOffsets[r]! + k]!;
      const gx = fx[p]!, gy = fy[p]!;
      const inv = 1 / Math.sqrt(gx * gx + gy * gy);
      sx += -gy * inv;
      sy += gx * inv;
    }
    const m = Math.hypot(sx, sy);
    if (m > 0) { meanDirs[r * 2] = sx / m; meanDirs[r * 2 + 1] = sy / m; }
  }

  return { regionCount, memberCount, regionOffsets, regionSizes, members, meanDirs };
}

// ── lsdFit ────────────────────────────────────────────────────────────────
//
// The third twin, and the last stage a twin is the right oracle for. From here
// on the sweep's ground truth knows the answer -- a vote normal is perpendicular
// to a true floor direction, the period is the one the renderer used -- so a
// second implementation stops being the best available check and starts being
// merely a second opinion.
//
// ── WHERE THIS IS A DIFFERENT ALGORITHM, AND WHERE IT CANNOT BE ──
//
// Two of the three steps genuinely re-derive:
//
//   THE AXIS. The shader uses the half-angle identity, theta = atan2(2Ixy,
//   Ixx-Iyy)/2. This solves the 2x2 eigenproblem directly and takes atan2 of the
//   major eigenvector. Same angle, no shared line of reasoning -- a wrong factor
//   of two or a swapped argument in the half-angle form shows up immediately.
//
//   THE TAIL. The shader rescales an online running maximum because WGSL has no
//   dynamic-length local array. Here every term is buffered and reduced against
//   their maximum afterwards, which is the textbook form.
//
// The FOOTPRINT count cannot. "Pixels whose centre lies inside this rectangle"
// admits one formulation, and inventing a second one (rasterizing the edges,
// say) would test the rasterizer rather than the fit. So it is written from the
// definition and carries the same two constants, which are correctness
// constants rather than implementation choices -- see LSD_FIT_WGSL's header for
// why BOUNDARY_EPS and the signed alignment test are both load-bearing.
//
// ── THE DIAGNOSTICS EXIST TO LICENSE THE ASSERTIONS ──
//
// Each one answers "was this comparison decisive?", the same job cpuGrow's
// marginalPairs does. An exact match on n and k means nothing if half the
// footprint sat on the inclusion boundary, and an angle comparison means nothing
// on a region with no axis to speak of.

export interface CpuRect {
  cx: number; cy: number; theta: number; length: number; width: number;
  nfaLog10: number; accepted: boolean;
  /** Footprint pixels, and how many of those were aligned. */
  n: number; k: number;
}

export interface CpuLsdFit {
  rects: CpuRect[];
  /**
   * The least anisotropic region's (lam1 - lam2) / (lam1 + lam2): 1 for a
   * perfectly straight run of pixels, 0 for a blob with no axis at all. The
   * principal angle's conditioning goes as 1/anisotropy, so this is what says
   * whether comparing two implementations' theta is meaningful.
   */
  minAnisotropy: number;
  /**
   * How close the closest INCLUSION call came to flipping: the least
   * | |proj| - (hl + BOUNDARY_EPS) | (or the same in perp) over every pixel of
   * every footprint bounding box.
   *
   * A margin is reported rather than a count inside a hand-picked band, because
   * the band would be the thing under test. Compare it against the f32 error of
   * the quantity itself -- proj is a dot of image coordinates, so ~1e-5 -- and
   * the comparison is decisive when the margin clears that by an order or two.
   *
   * BOUNDARY_EPS is what makes this large rather than zero: a rectangle's
   * extent is DEFINED by its extreme members, so members sit exactly on
   * |proj| == hl, and the epsilon moves the decision off that pile-up.
   */
  closestInclusion: number;
  /**
   * The same, for the ALIGNMENT call: the least |alignDot - cos(tolerance)|
   * over every pixel that was tested.
   *
   * Nothing protects this one and nothing can. Level-line direction varies
   * continuously across an image, so pixels land arbitrarily close to the
   * tolerance boundary and the margin shrinks as the frame gets bigger. The
   * saving grace is that alignDot is O(1) rather than an image coordinate, so
   * its f32 error is ~1e-7 and there is a lot of room between the two.
   */
  closestAlignment: number;
  /** The least |logNfa - log(epsilon)| -- how close an accept came to flipping. */
  closestAccept: number;
}

/** The region CSR, however it was produced -- by cpuCollect or read off the device. */
export interface RegionSet {
  regionCount: number;
  regionOffsets: Uint32Array;
  regionSizes: Uint32Array;
  members: Uint32Array;
  meanDirs: Float32Array;
}

/** log(sum_{j=k}^{n} C(n,j) p^j (1-p)^(n-j)), by buffered log-sum-exp. */
export function logBinomialTail(n: number, k: number, p: number): number {
  if (k <= 0) return 0;      // P(X >= 0) == 1
  if (k > n) return -Infinity;
  const logP = Math.log(p), log1mP = Math.log(1 - p);
  // log C(n,k) built up once, then carried forward by C(n,j)/C(n,j-1) =
  // (n-j+1)/j rather than recomputed per term.
  let logChoose = 0;
  for (let i = 1; i <= k; i++) logChoose += Math.log((n - k + i) / i);

  const terms = [logChoose + k * logP + (n - k) * log1mP];
  for (let j = k + 1; j <= n; j++) {
    logChoose += Math.log((n - j + 1) / j);
    terms.push(logChoose + j * logP + (n - j) * log1mP);
  }
  const max = Math.max(...terms);
  let sum = 0;
  for (const t of terms) sum += Math.exp(t - max);
  return max + Math.log(sum);
}

export function cpuLsdFit(
  fx: Float32Array, fy: Float32Array, w: number, h: number,
  regions: RegionSet, s: LsdFitSettings,
): CpuLsdFit {
  const BOUNDARY_EPS = 1e-3;

  const toleranceRad = (s.toleranceDeg * Math.PI) / 180;
  const cosTol = Math.cos(toleranceRad);
  const rhoSq = s.rho >= 0 ? s.rho * s.rho : -Infinity;
  const logNTests = s.nfaTestExponent * Math.log(Math.max(w, h));
  const logEpsilon = Math.log(s.nfaEpsilon);

  const rects: CpuRect[] = [];
  let minAnisotropy = Infinity;
  let closestInclusion = Infinity;
  let closestAlignment = Infinity;
  let closestAccept = Infinity;
  const degenerate: CpuRect = {
    cx: 0, cy: 0, theta: 0, length: 0, width: 0, nfaLog10: 0, accepted: false, n: 0, k: 0,
  };

  for (let r = 0; r < regions.regionCount; r++) {
    const off = regions.regionOffsets[r]!, sz = regions.regionSizes[r]!;
    if (sz < 2) { rects.push({ ...degenerate }); continue; }

    let sumW = 0, sumX = 0, sumY = 0;
    for (let a = 0; a < sz; a++) {
      const i = regions.members[off + a]!;
      const m = Math.hypot(fx[i]!, fy[i]!);
      sumW += m; sumX += m * (i % w); sumY += m * ((i / w) | 0);
    }
    if (sumW <= 0) { rects.push({ ...degenerate }); continue; }
    const wcx = sumX / sumW, wcy = sumY / sumW;

    let Ixx = 0, Iyy = 0, Ixy = 0;
    for (let a = 0; a < sz; a++) {
      const i = regions.members[off + a]!;
      const m = Math.hypot(fx[i]!, fy[i]!);
      const x = (i % w) - wcx, y = ((i / w) | 0) - wcy;
      Ixx += m * x * x; Iyy += m * y * y; Ixy += m * x * y;
    }

    // The eigenproblem, solved rather than pattern-matched. The two eigenvalues
    // of [[Ixx, Ixy], [Ixy, Iyy]] are (tr +/- sqrt((Ixx-Iyy)^2 + 4Ixy^2)) / 2,
    // and the major eigenvector is whichever of the two rows of (M - lam2*I)
    // has the larger norm -- taking the fixed one collapses when the region
    // happens to lie along that axis.
    const disc = Math.hypot(Ixx - Iyy, 2 * Ixy);
    const lam1 = (Ixx + Iyy + disc) / 2, lam2 = (Ixx + Iyy - disc) / 2;
    if (lam1 + lam2 > 0) minAnisotropy = Math.min(minAnisotropy, disc / (lam1 + lam2));
    let vx = Ixy, vy = lam1 - Ixx;
    if (Math.hypot(vx, vy) < Math.hypot(lam1 - Iyy, Ixy)) { vx = lam1 - Iyy; vy = Ixy; }
    let theta = Math.atan2(vy, vx);
    // The 180-degree ambiguity, resolved against the region's own mean level
    // line exactly as the shader does. Not an implementation choice: it is what
    // makes the alignment test below mean "aligned" rather than "anti-aligned".
    const mdx = regions.meanDirs[r * 2]!, mdy = regions.meanDirs[r * 2 + 1]!;
    if (Math.cos(theta) * mdx + Math.sin(theta) * mdy < 0) theta += Math.PI;

    const ax = Math.cos(theta), ay = Math.sin(theta);
    const px = -ay, py = ax;
    let minProj = Infinity, maxProj = -Infinity, minPerp = Infinity, maxPerp = -Infinity;
    for (let a = 0; a < sz; a++) {
      const i = regions.members[off + a]!;
      const x = (i % w) - wcx, y = ((i / w) | 0) - wcy;
      const proj = x * ax + y * ay, perp = x * px + y * py;
      minProj = Math.min(minProj, proj); maxProj = Math.max(maxProj, proj);
      minPerp = Math.min(minPerp, perp); maxPerp = Math.max(maxPerp, perp);
    }
    const midProj = (minProj + maxProj) / 2, midPerp = (minPerp + maxPerp) / 2;
    const cx = wcx + midProj * ax + midPerp * px;
    const cy = wcy + midProj * ay + midPerp * py;
    const length = maxProj - minProj, width = maxPerp - minPerp;

    const hl = length / 2, hw = Math.max(width / 2, 0.5);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [a, b] of [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]] as const) {
      const x = cx + a * ax + b * px, y = cy + a * ay + b * py;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    const x0 = Math.max(0, Math.floor(minX)), x1 = Math.min(w - 1, Math.ceil(maxX));
    const y0 = Math.max(0, Math.floor(minY)), y1 = Math.min(h - 1, Math.ceil(maxY));

    let n = 0, k = 0;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const dx = x - cx, dy = y - cy;
        const proj = dx * ax + dy * ay, perp = dx * px + dy * py;
        closestInclusion = Math.min(closestInclusion,
          Math.abs(Math.abs(proj) - (hl + BOUNDARY_EPS)),
          Math.abs(Math.abs(perp) - (hw + BOUNDARY_EPS)));
        if (Math.abs(proj) > hl + BOUNDARY_EPS || Math.abs(perp) > hw + BOUNDARY_EPS) continue;
        n++;
        const i = y * w + x;
        const gx = fx[i]!, gy = fy[i]!;
        const m2 = gx * gx + gy * gy;
        if (m2 <= rhoSq) continue;
        const alignDot = (-gy * ax + gx * ay) / Math.sqrt(m2);
        closestAlignment = Math.min(closestAlignment, Math.abs(alignDot - cosTol));
        if (alignDot >= cosTol) k++;
      }
    }

    const logNfa = logNTests + logBinomialTail(n, k, toleranceRad / Math.PI);
    closestAccept = Math.min(closestAccept, Math.abs(logNfa - logEpsilon));
    rects.push({
      cx, cy, theta, length, width,
      nfaLog10: logNfa / Math.LN10, accepted: logNfa < logEpsilon, n, k,
    });
  }

  return { rects, minAnisotropy, closestInclusion, closestAlignment, closestAccept };
}
