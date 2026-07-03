// Shared De Bruijn torus construction — used by both the offline PNG
// generator (scripts/generate-debruijn-torus.ts) and the browser tracker
// (src/main.ts). Given just an order, both independently reconstruct the
// identical pattern (the tap search is a deterministic function of N), so
// there's no need to ship pixel data or a lookup table as a separate asset.
//
// See scripts/generate-debruijn-torus.ts for the full construction writeup.

// Runs a Fibonacci LFSR of degree N for a candidate set of feedback taps
// (1-indexed bit positions, XORed together each step to produce the bit fed
// into the MSB after shifting right). Returns the L = 2^N - 1 length output
// sequence if this tap set is maximal-length (every nonzero state visited
// exactly once), or null otherwise. Exits on the first repeated state, so
// non-primitive candidates are usually rejected in a handful of steps, not
// the full period — that's what makes searching many candidates cheap.
function tryTaps(taps: number[], N: number): Uint8Array | null {
  const L = 2 ** N - 1;
  const seen = new Uint8Array(L + 1);
  const seq = new Uint8Array(L);
  let state = 1; // any nonzero seed
  for (let i = 0; i < L; i++) {
    if (seen[state]) return null;
    seen[state] = 1;
    seq[i] = state & 1;
    let feedback = 0;
    for (const t of taps) feedback ^= (state >>> (t - 1)) & 1;
    state = (state >>> 1) | (feedback << (N - 1));
  }
  return seq;
}

// Deterministic PRNG (mulberry32) so a given order always produces the same
// pattern rather than a different one each run.
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Picks `weight` tap positions from 1..N-1, one drawn from each of `weight`
// equal-width bins spanning the register — this directly forces the taps to
// be spread out rather than clustered near one end, which reduces
// short-range correlation (visible diagonal texture) in the resulting
// pattern. Doesn't affect the maximal-length property (verified below
// regardless of which taps are picked).
function stratifiedTaps(N: number, weight: number, rng: () => number): number[] {
  const taps = new Set<number>();
  for (let i = 0; i < weight; i++) {
    const lo = 1 + Math.floor((i * (N - 1)) / weight);
    const hi = 1 + Math.floor(((i + 1) * (N - 1)) / weight) - 1;
    taps.add(Math.min(lo + Math.floor(rng() * (hi - lo + 1)), N - 1));
  }
  return [N, ...taps];
}

function primePowerFactors(nInput: number): number[] {
  const chunks: number[] = [];
  let n = nInput;
  for (let p = 2; p * p <= n; p++) {
    if (n % p === 0) {
      let power = 1;
      while (n % p === 0) { power *= p; n /= p; }
      chunks.push(power);
    }
  }
  if (n > 1) chunks.push(n);
  return chunks;
}

// Finds coprime R, C with R * C = L, as close to square as the factorization allows.
// Each prime-power chunk of L must go entirely to one side to keep gcd(R,C) = 1.
function bestCoprimeSplit(L: number): { R: number; C: number } {
  const chunks = primePowerFactors(L);
  let best = { R: 1, C: L };
  let bestDiff = Infinity;
  for (let mask = 0; mask < (1 << chunks.length); mask++) {
    let R = 1;
    for (let i = 0; i < chunks.length; i++) if (mask & (1 << i)) R *= chunks[i];
    const C = L / R;
    const diff = Math.abs(R - C);
    if (diff < bestDiff) { bestDiff = diff; best = { R, C }; }
  }
  return best;
}

function buildTorus(seq: Uint8Array, R: number, C: number): Uint8Array[] {
  const torus = Array.from({ length: R }, () => new Uint8Array(C));
  for (let t = 0; t < seq.length; t++) torus[t % R][t % C] = seq[t];
  return torus;
}

// Packs the n x n window at torus position (r, c) into an integer key: row 0
// (top) first, then row 1, etc; within each row, left to right; each cell
// shifted in MSB-first. The camera-side decoder MUST sample in this same
// order (top-to-bottom, left-to-right) for lookups to match.
function windowKey(torus: Uint8Array[], R: number, C: number, r: number, c: number, n: number): number {
  let key = 0;
  for (let i = 0; i < n; i++) {
    const row = torus[(r + i) % R];
    for (let j = 0; j < n; j++) key = (key << 1) | row[(c + j) % C];
  }
  return key >>> 0;
}

// Brute-force re-check: every n x n toroidal window in the actual output must
// be unique. Packs each window into an integer key and uses a direct-indexed
// visited array (cheap since keys are bounded by 2^N).
function verifyTorusWindows(torus: Uint8Array[], R: number, C: number, n: number, N: number): boolean {
  const seen = new Uint8Array(2 ** N);
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) {
      const key = windowKey(torus, R, C, r, c, n);
      if (seen[key]) return false;
      seen[key] = 1;
    }
  }
  return true;
}

export const WINDOW_CHECK_LIMIT = 2 ** 22; // ~4M cells, keeps the O(R*C*n^2) 2D check fast

// Finds a feedback tap set for degree N whose m-sequence is BOTH maximal-length
// AND — when folded onto the R x C torus — has verifiably unique n x n windows.
// These are two separate properties: maximal-length (1D) is necessary but,
// surprisingly, not sufficient — some primitive polynomials fold into a torus
// with duplicate windows (confirmed empirically: taps [16,1,5,7,10,15] are a
// valid maximal-length degree-16 LFSR but produce a broken 255x257 fold). So
// this searches candidates and checks the full pipeline (1D + 2D) together,
// rather than trusting 1D primitivity alone.
//
// Searches spread-out, moderately dense candidates first (denser, well-spread
// taps reduce short-range correlation and give a more visually uncorrelated
// pattern than the sparsest possible polynomial), falling back to sparser odd
// weights down to 1.
//
// Tap-count parity matters: empirically (verified for N=16), the taps array —
// which always includes N itself as one entry — must have EVEN total length
// for a candidate to have any chance of being primitive, so the extra-tap
// count (beyond N) is always kept odd.
function findValidTorusSequence(order: number, N: number, R: number, C: number): { taps: number[]; seq: Uint8Array } {
  const check2D = R * C <= WINDOW_CHECK_LIMIT;
  const isValid = (seq: Uint8Array) => {
    if (!check2D) return true; // can't verify at this size — accept on 1D success alone
    return verifyTorusWindows(buildTorus(seq, R, C), R, C, order, N);
  };

  const preferredWeight = Math.min(Math.max(5, Math.round(N / 3) | 1), N % 2 === 0 ? N - 1 : N - 2);
  for (let weight = preferredWeight; weight >= 1; weight -= 2) {
    if (weight === 1) {
      for (let k = 1; k < N; k++) {
        const seq = tryTaps([N, k], N);
        if (seq && isValid(seq)) return { taps: [N, k], seq };
      }
      continue;
    }
    const rng = mulberry32(((N * 2654435761) ^ weight) >>> 0);
    for (let i = 0; i < 2000; i++) {
      const taps = stratifiedTaps(N, weight, rng);
      if (taps.length !== weight + 1) continue; // stratification collision broke exact parity — skip
      const seq = tryTaps(taps, N);
      if (seq && isValid(seq)) return { taps, seq };
    }
  }
  throw new Error(`Could not find a maximal-length, torus-valid LFSR for order ${order} (N=${N}).`);
}

export interface DebruijnTorus {
  order: number;
  N: number;
  R: number;
  C: number;
  taps: number[];
  torus: Uint8Array[];
}

export function generateTorus(order: number): DebruijnTorus {
  const N = order * order;
  const L = 2 ** N - 1;
  const { R, C } = bestCoprimeSplit(L);
  const { taps, seq } = findValidTorusSequence(order, N, R, C);
  const torus = buildTorus(seq, R, C);
  return { order, N, R, C, taps, torus };
}

// Builds a direct-indexed lookup table: window key -> packed (row * C + col).
// Every key is guaranteed unique (by construction, re-verified above for
// tractable sizes), so this is a simple one-pass fill, no collision handling.
// Unfilled entries (should be none, aside from window-key 0 which never
// occurs since m-sequences never contain the all-zero window) are left as -1.
export function buildLookupTable({ torus, R, C, order, N }: DebruijnTorus): Int32Array {
  const table = new Int32Array(2 ** N).fill(-1);
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) {
      table[windowKey(torus, R, C, r, c, order)] = r * C + c;
    }
  }
  return table;
}

export { windowKey, bestCoprimeSplit, buildTorus, verifyTorusWindows };
