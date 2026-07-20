// ── Spherical-Hough prototype: noise/blur/downsample primitives ─────────

// Tiny seeded PRNG (mulberry32) so noise is reproducible rather than
// Math.random()-fresh on every capture.
export function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export const NOISE_SEED = 1337;

export function addGaussianNoise(gray: Float64Array, std: number) {
  if (std <= 0) return;
  const rng = mulberry32(NOISE_SEED);
  for (let i = 0; i < gray.length; i++) {
    const u1 = Math.max(1e-9, rng()), u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    gray[i] = Math.min(255, Math.max(0, gray[i] + z * std));
  }
}

export function downsampleBoxAverage(src: Float64Array, srcW: number, srcH: number, scale: number, dstW: number, dstH: number): Float64Array {
  const dst = new Float64Array(dstW * dstH);
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      let sum = 0, count = 0;
      for (let dy = 0; dy < scale; dy++) {
        const sy = y * scale + dy;
        if (sy >= srcH) continue;
        for (let dx = 0; dx < scale; dx++) {
          const sx = x * scale + dx;
          if (sx >= srcW) continue;
          sum += src[sy * srcW + sx];
          count++;
        }
      }
      dst[y * dstW + x] = count > 0 ? sum / count : 0;
    }
  }
  return dst;
}

// Two-pass (horizontal then vertical) box blur -- O(w*h) total, independent
// of radius, via a running sum per row/column slid one pixel at a time.
export function separableBoxBlur(src: Float64Array, w: number, h: number, radius: number): Float64Array {
  if (radius <= 0) return src.slice();
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let lo = 0, hi = Math.min(w - 1, radius);
    let sum = 0;
    for (let i = lo; i <= hi; i++) sum += src[row + i];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum / (hi - lo + 1);
      if (x + 1 >= w) continue;
      const nextHi = Math.min(w - 1, x + 1 + radius);
      if (nextHi > hi) { hi = nextHi; sum += src[row + hi]; }
      const nextLo = Math.max(0, x + 1 - radius);
      if (nextLo > lo) { sum -= src[row + lo]; lo = nextLo; }
    }
  }
  for (let x = 0; x < w; x++) {
    let lo = 0, hi = Math.min(h - 1, radius);
    let sum = 0;
    for (let i = lo; i <= hi; i++) sum += tmp[i * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / (hi - lo + 1);
      if (y + 1 >= h) continue;
      const nextHi = Math.min(h - 1, y + 1 + radius);
      if (nextHi > hi) { hi = nextHi; sum += tmp[hi * w + x]; }
      const nextLo = Math.max(0, y + 1 - radius);
      if (nextLo > lo) { sum -= tmp[lo * w + x]; lo = nextLo; }
    }
  }
  return out;
}

export function flipRowsF64(src: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    const srcRow = h - 1 - y;
    out.set(src.subarray(srcRow * w, (srcRow + 1) * w), y * w);
  }
  return out;
}

export function applyAntialiasFilter(gray: Float64Array, w: number, h: number, supersample: number): Float64Array {
  return separableBoxBlur(gray, w, h, Math.max(1, Math.round(supersample / 2)));
}

export function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; } else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; } else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; } else { r = c; b = x; }
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}
