// Deterministic value noise, for breaking up flat surfaces.
//
// Value noise rather than Perlin/simplex: the floor only needs low-frequency
// mottling, where the difference between the two is invisible, and this is a
// dozen lines with no gradient table to get wrong. Hash-based and stateless --
// noise2(x, y) is the same number for the same inputs forever, so a board looks
// identical across reloads without anything having to be stored.
//
// Pure: no three, no DOM, so it stays runnable under node like frame.ts and
// motion.ts.

/**
 * Small deterministic PRNG (mulberry32). Math.random can't be used for
 * anything the board is built from: a world that differs per reload can't be
 * checked headlessly, and terrain or scenery that moved under a saved game
 * later would be worse.
 *
 * Lives here with the noise because both are "reproducible randomness" and
 * because two callers were otherwise going to carry their own copy.
 */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Integer hash -> [0, 1). The constants are arbitrary large odds; what
 *  matters is that the two multiplies and shifts mix every input bit into the
 *  high bits, so lattice points that are neighbours don't get similar values
 *  (which would show up as visible diagonal banding). */
function hash(ix: number, iy: number, seed: number): number {
  let h = (seed ^ Math.imul(ix | 0, 0x27d4eb2d) ^ Math.imul(iy | 0, 0x165667b1)) | 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d);
  h = Math.imul(h ^ (h >>> 12), 0x297a2d39);
  h ^= h >>> 15;
  return (h >>> 0) / 4294967296;
}

/** Smoothstep. Interpolating the lattice linearly instead would leave a
 *  visible crease along every lattice line, since the slope jumps there. */
function fade(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Value noise in [0, 1), with features one unit across. */
export function noise2(x: number, y: number, seed = 0): number {
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = fade(x - x0), fy = fade(y - y0);

  const a = hash(x0, y0, seed), b = hash(x0 + 1, y0, seed);
  const c = hash(x0, y0 + 1, seed), d = hash(x0 + 1, y0 + 1, seed);

  return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy;
}

/**
 * Several octaves of noise2 summed, each half the amplitude and twice the
 * frequency of the last -- one big shape with smaller ones riding on it, which
 * is what keeps the result from looking like evenly sized blobs.
 *
 * @param scale world units per feature of the FIRST octave.
 * @returns roughly [0, 1), normalized so octave count doesn't change the range.
 */
export function fbm(x: number, y: number, scale: number, octaves = 3, seed = 0): number {
  let sum = 0, amplitude = 1, total = 0, frequency = 1 / scale;
  for (let o = 0; o < octaves; o++) {
    // Each octave gets its own seed, so they're independent fields rather than
    // the same pattern at three sizes stacked on itself.
    sum += amplitude * noise2(x * frequency, y * frequency, seed + o * 0x9e37);
    total += amplitude;
    amplitude *= 0.5;
    frequency *= 2;
  }
  return sum / total;
}
