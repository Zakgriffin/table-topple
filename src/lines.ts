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

// Builds the (theta, rho) vote accumulator. thetaBins controls angular
// resolution (PI/thetaBins radians per bin); rhoBinSize is in pixels.
// minMag is a noise floor below which a pixel's gradient is too weak/noisy
// to trust its orientation — skip it rather than let it cast a vote at a
// near-arbitrary angle.
export function buildLineAccumulator(
  gray: Float64Array, w: number, h: number,
  thetaBins = 180, rhoBinSize = 2, blurRadius = 1, minMag = 4,
): HoughField {
  const blurred = boxBlur(gray, w, h, blurRadius);
  const cx = w / 2, cy = h / 2;
  const rhoMax = Math.hypot(w, h) / 2 + 1;
  const rhoBins = Math.ceil((2 * rhoMax) / rhoBinSize) + 1;
  const rhoMin = -rhoMax;
  const acc = new Float64Array(thetaBins * rhoBins);

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const fx = blurred[i + 1] - blurred[i - 1];
      const fy = blurred[i + w] - blurred[i - w];
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

// Non-max suppression over the accumulator, then sub-bin refinement (weighted
// centroid within the peak's local window) for better-than-bin-width
// accuracy. theta is treated as CIRCULAR with period thetaBins (folding by PI
// already makes it a genuine circle: theta=0 and theta=PI-epsilon are
// adjacent lines, not opposite ends of a flat range) — NMS and the centroid
// window both wrap across that seam.
export function findLinePeaks(
  field: HoughField, threshold = 0.15, nmsThetaRadius = 4, nmsRhoRadius = 3,
): LineCandidate[] {
  const { thetaBins, rhoBins, rhoMin, rhoBinSize, acc } = field;
  let maxVal = 0;
  for (let i = 0; i < acc.length; i++) if (acc[i] > maxVal) maxVal = acc[i];
  if (maxVal === 0) return [];
  const minVal = threshold * maxVal;

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

  return kept.map(({ tb, rb }) => {
    // Weighted centroid over a small window around the peak bin, using
    // circular differences on theta so the centroid doesn't get pulled the
    // wrong way for a peak that sits right at the wrap seam.
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
      theta: (tbRefined / field.thetaBins) * Math.PI,
      rho: rhoMin + rbRefined * rhoBinSize,
      weight: acc[tb * rhoBins + rb],
    };
  });
}
