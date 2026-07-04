// Level 1 of the line-based rectification redesign: turns raw pixel
// gradients directly into a set of candidate straight lines, via a
// gradient-ORIENTED Hough transform.
//
// Classic Hough transform doesn't know a point's orientation, so every edge
// pixel has to vote for every possible theta that could pass through it (an
// O(N * numThetaBins) sweep). We already have the gradient at each pixel,
// which directly IS the line's normal direction there — so each pixel casts
// exactly one weighted vote instead of a whole fan of them (O(N)).
//
// A line is parameterized as (theta, rho): theta in [0, PI) is the line's
// NORMAL angle (folded mod PI, since a line has no inherent direction/arrow —
// unlike the gradient vector itself, which does), and rho is the signed
// perpendicular distance from the image CENTER (not the corner — keeps rho
// centered near 0 and avoids a large constant offset dominating the
// numerics) to the line: rho = dx*cos(theta) + dy*sin(theta) for any point
// (dx,dy) on the line, relative to that center.

export interface LineCandidate {
  theta: number; // radians, [0, PI)
  rho: number;   // pixels, relative to image center
  weight: number; // accumulated vote mass supporting this line
}

export interface HoughField {
  w: number; h: number;
  thetaBins: number; rhoBins: number;
  rhoMin: number; rhoBinSize: number;
  acc: Float64Array; // thetaBins x rhoBins, row-major (theta-major)
}

export function boxBlur(src: Float64Array, w: number, h: number, radius: number): Float64Array {
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

// Builds the (theta, rho) vote accumulator. thetaBins controls angular
// resolution (PI/thetaBins radians per bin); rhoBinSize is in pixels.
// minMag is a noise floor below which a pixel's gradient is too weak/noisy
// to trust its orientation — skip it rather than let it cast a vote at a
// near-arbitrary angle. gradientRadius is the central-difference offset (in
// pixels) used to estimate fx/fy — 1 (the default, matching all prior
// behavior exactly: fx = blurred[i+1]-blurred[i-1], unnormalized) is a
// standard 3-tap derivative. A larger radius widens the baseline, trading
// sensitivity to fine per-pixel noise for sensitivity to genuine edges
// spread over more than 1px (e.g. from capture blur or downsampling) — NOT
// normalized by the radius, so raw magnitude (and therefore minMag's
// effective threshold) shifts with it; that's an expected side effect of
// changing the kernel, not a bug, and worth re-tuning minMag/peak threshold
// alongside it.
export function buildLineAccumulator(
  gray: Float64Array, w: number, h: number,
  thetaBins = 180, rhoBinSize = 2, blurRadius = 1, minMag = 4, gradientRadius = 1,
): HoughField {
  const blurred = boxBlur(gray, w, h, blurRadius);
  const cx = w / 2, cy = h / 2;
  const rhoMax = Math.hypot(w, h) / 2 + 1;
  const rhoBins = Math.ceil((2 * rhoMax) / rhoBinSize) + 1;
  const rhoMin = -rhoMax;
  const acc = new Float64Array(thetaBins * rhoBins);
  const r = gradientRadius;

  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const fx = blurred[i + r] - blurred[i - r];
      const fy = blurred[i + r * w] - blurred[i - r * w];
      const mag = Math.hypot(fx, fy);
      if (mag < minMag) continue;

      let theta = Math.atan2(fy, fx);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI; // exactly PI (rare, from atan2's +PI edge) folds to 0

      const dx = x - cx, dy = y - cy;
      const rho = dx * Math.cos(theta) + dy * Math.sin(theta);

      let tb = Math.floor((theta / Math.PI) * thetaBins);
      if (tb >= thetaBins) tb = 0; // theta wrapped exactly to PI
      const rb = Math.floor((rho - rhoMin) / rhoBinSize);
      if (rb < 0 || rb >= rhoBins) continue;

      acc[tb * rhoBins + rb] += mag;
    }
  }

  return { w, h, thetaBins, rhoBins, rhoMin, rhoBinSize, acc };
}

// Non-max suppression over the accumulator down to `minVal` (an ABSOLUTE
// vote value, not a threshold fraction — callers needing a rescue tier want
// the search to reach below their real reporting threshold without running
// NMS twice), returning deduped bin-level candidates with their raw vote
// value still attached (refinement into an actual LineCandidate happens in
// the two exported wrappers below, since which candidates end up in which
// output tier needs to see `v` first).
function findPeakBins(
  field: HoughField, minVal: number, nmsThetaRadius: number, nmsRhoRadius: number,
): { tb: number; rb: number; v: number }[] {
  const { thetaBins, rhoBins, acc } = field;
  const wrapTheta = (t: number) => ((t % thetaBins) + thetaBins) % thetaBins;

  const candidates: { tb: number; rb: number; v: number }[] = [];
  for (let tb = 0; tb < thetaBins; tb++) {
    for (let rb = 0; rb < rhoBins; rb++) {
      const v = acc[tb * rhoBins + rb];
      if (v < minVal) continue;
      let isPeak = true;
      for (let dt = -nmsThetaRadius; dt <= nmsThetaRadius && isPeak; dt++) {
        const ttb = wrapTheta(tb + dt);
        for (let dr = -nmsRhoRadius; dr <= nmsRhoRadius; dr++) {
          if (dt === 0 && dr === 0) continue;
          const rrb = rb + dr;
          if (rrb < 0 || rrb >= rhoBins) continue;
          if (acc[ttb * rhoBins + rrb] > v) { isPeak = false; break; }
        }
      }
      if (isPeak) candidates.push({ tb, rb, v });
    }
  }
  candidates.sort((a, b) => b.v - a.v);

  const minDistSq = (nmsThetaRadius * 1.5) ** 2 + (nmsRhoRadius * 1.5) ** 2;
  const kept: { tb: number; rb: number; v: number }[] = [];
  for (const c of candidates) {
    if (kept.some(k => {
      let dt = Math.abs(k.tb - c.tb);
      dt = Math.min(dt, thetaBins - dt);
      const dr = k.rb - c.rb;
      return dt * dt + dr * dr < minDistSq;
    })) continue;
    kept.push(c);
  }
  return kept;
}

// Sub-bin refinement (weighted centroid within the peak's local window) for
// better-than-bin-width accuracy. theta is treated as CIRCULAR with period
// thetaBins (folding by PI already makes it a genuine circle: theta=0 and
// theta=PI-epsilon are adjacent lines, not opposite ends of a flat range) —
// the centroid window wraps across that seam.
function refinePeak(field: HoughField, tb: number, rb: number, nmsThetaRadius: number, nmsRhoRadius: number): LineCandidate {
  const { thetaBins, rhoBins, rhoMin, rhoBinSize, acc } = field;
  const wrapTheta = (t: number) => ((t % thetaBins) + thetaBins) % thetaBins;
  let sumW = 0, sumDT = 0, sumR = 0;
  for (let dt = -nmsThetaRadius; dt <= nmsThetaRadius; dt++) {
    const ttb = wrapTheta(tb + dt);
    for (let dr = -nmsRhoRadius; dr <= nmsRhoRadius; dr++) {
      const rrb = rb + dr;
      if (rrb < 0 || rrb >= rhoBins) continue;
      const v = acc[ttb * rhoBins + rrb];
      sumW += v;
      sumDT += v * dt;
      sumR += v * (rrb + 0.5);
    }
  }
  const tbRefined = wrapTheta(tb + (sumW > 0 ? sumDT / sumW : 0) + 0.5);
  const rbRefined = sumW > 0 ? sumR / sumW : rb + 0.5;
  return {
    theta: (tbRefined / thetaBins) * Math.PI,
    rho: rhoMin + rbRefined * rhoBinSize,
    weight: acc[tb * rhoBins + rb],
  };
}

export function findLinePeaks(
  field: HoughField, threshold = 0.15, nmsThetaRadius = 4, nmsRhoRadius = 3,
): LineCandidate[] {
  let maxVal = 0;
  for (let i = 0; i < field.acc.length; i++) if (field.acc[i] > maxVal) maxVal = field.acc[i];
  if (maxVal === 0) return [];
  const bins = findPeakBins(field, threshold * maxVal, nmsThetaRadius, nmsRhoRadius);
  return bins.map(({ tb, rb }) => refinePeak(field, tb, rb, nmsThetaRadius, nmsRhoRadius));
}

// Same NMS search as findLinePeaks, but run ONCE down to the lower of the
// two thresholds and partitioned by vote strength into two tiers, rather
// than calling findLinePeaks twice (which would find the strong peaks
// TWICE — once in each call — needing a fuzzy dedup step to avoid double-
// counting the same physical line in both outputs). `weak` is for
// src/vp.ts's splitIntoTwoFamilies extraLines rescue mechanism: a real
// camera's two grid-line families are not always comparably strong
// (lighting, focus, or a camera's own directional sharpening can make one
// family's edges systematically weaker for reasons that have nothing to do
// with the grid or the algorithm), and that family's true members may
// mostly sit below `threshold` without being pure noise — `rescueThreshold`
// gives them a second, lower bar, still checked against an actual estimated
// vanishing point before being trusted, not just taken on faith.
export function findLinePeaksTiered(
  field: HoughField, threshold: number, rescueThreshold: number, nmsThetaRadius = 4, nmsRhoRadius = 3,
): { strong: LineCandidate[]; weak: LineCandidate[] } {
  let maxVal = 0;
  for (let i = 0; i < field.acc.length; i++) if (field.acc[i] > maxVal) maxVal = field.acc[i];
  if (maxVal === 0) return { strong: [], weak: [] };
  const lowVal = Math.min(threshold, rescueThreshold) * maxVal;
  const highVal = Math.max(threshold, rescueThreshold) * maxVal;
  const bins = findPeakBins(field, lowVal, nmsThetaRadius, nmsRhoRadius);
  const strong: LineCandidate[] = [], weak: LineCandidate[] = [];
  for (const b of bins) (b.v >= highVal ? strong : weak).push(refinePeak(field, b.tb, b.rb, nmsThetaRadius, nmsRhoRadius));
  return { strong, weak };
}
