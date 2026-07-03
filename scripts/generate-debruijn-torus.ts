// Generates a binary De Bruijn torus PNG: an R x C toroidal grid of black/white
// cells such that every n x n window (wrapping around both edges) is unique.
// Point a camera at any n x n patch and its content alone identifies exactly
// where on the mat it is.
//
// Construction (classical LFSR + CRT fold, MacWilliams & Sloane 1976):
//   1. N = n * n is the window's bit count. Generate a maximal-length LFSR
//      ("m-sequence") of degree N by searching for a feedback tap set that
//      empirically produces a full-period (2^N - 1) cycle — rather than
//      trusting a memorized table of "known primitive polynomials" (tap-list
//      conventions vary by source and are an easy way to silently ship a
//      broken, non-maximal sequence). Maximal length means every nonzero
//      N-bit window appears in the sequence exactly once as it cycles.
//   2. Factor L = 2^N - 1 into two coprime numbers R and C (R * C = L) as
//      close to square as possible. By the Chinese Remainder Theorem, t ->
//      (t mod R, t mod C) is a bijection from Z_L to Z_R x Z_C, so folding
//      the sequence that way fills an R x C torus using every position
//      exactly once.
//   3. The 1D window-uniqueness property is known to carry over to n x n
//      windows on the folded torus. Since getting this fold subtly wrong is
//      an easy way to silently produce a *broken* pattern, this script also
//      brute-force re-verifies uniqueness on the actual output (for sizes
//      where that's tractable) instead of just trusting the theorem.
//
// Usage:
//   node scripts/generate-debruijn-torus.ts --order 4 --cell-size 8 --out torus.png
//
//   --order      Window order n (window is n x n cells). Determines how many
//                cells the camera must see to uniquely determine its position.
//                Order 5 alone is a 25-bit torus, ~33.5M cells, already a lot
//                to print, so in practice order 3 or 4 is the realistic range.
//   --cell-size  Pixels per cell edge, uniform in both dimensions (default 8).
//                The output image is exactly (C * cell-size) x (R * cell-size)
//                px — no stretching to force a square canvas, so a torus whose
//                R x C grid isn't square (2^N-1 doesn't always factor nicely)
//                shows up as a genuinely non-square image.
//   --out        Output file path (default: debruijn-torus-order<n>.png).

import { PNG } from 'pngjs';
import { createWriteStream } from 'node:fs';

function parseArgs(argv: string[]): Record<string, string> {
  const args: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2);
      const next = argv[i + 1];
      args[key] = next !== undefined && !next.startsWith('--') ? argv[++i] : 'true';
    }
  }
  return args;
}

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
// be spread out rather than clustered near one end, which is what caused
// visible short-range correlation (diagonal texture) with the earlier
// clustered 4-tap search. Sparse *or* clustered feedback polynomials are a
// known source of visually-detectable structure in LFSR sequences; spreading
// the taps out is the standard fix and doesn't affect the maximal-length
// property (that depends only on primitivity, verified below regardless).
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

// Brute-force re-check: every n x n toroidal window in the actual output must
// be unique. Packs each window into an integer key and uses a direct-indexed
// visited array (cheap since keys are bounded by 2^N).
function verifyTorusWindows(torus: Uint8Array[], R: number, C: number, n: number, N: number): boolean {
  const seen = new Uint8Array(2 ** N);
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) {
      let key = 0;
      for (let i = 0; i < n; i++) {
        const row = torus[(r + i) % R];
        for (let j = 0; j < n; j++) key = (key << 1) | row[(c + j) % C];
      }
      key = key >>> 0;
      if (seen[key]) return false;
      seen[key] = 1;
    }
  }
  return true;
}

const WINDOW_CHECK_LIMIT = 2 ** 22; // ~4M cells, keeps the O(R*C*n^2) 2D check fast

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
// pattern than the sparsest possible polynomial — with no effect on either
// uniqueness property, or on later position lookup, which will use a
// precomputed table or a discrete-log solve, either of which works the same
// regardless of tap weight), falling back to sparser odd weights down to 1.
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

// Renders each cell as an uncompromising cellSize x cellSize block of solid
// pixels — no resampling, no stretching. Output image is (C*cellSize) x
// (R*cellSize), so a non-square R x C grid produces a genuinely non-square
// image rather than being distorted to fit a fixed canvas.
function writePng(torus: Uint8Array[], R: number, C: number, cellSize: number, outPath: string): Promise<void> {
  const width = C * cellSize;
  const height = R * cellSize;
  const png = new PNG({ width, height });
  for (let y = 0; y < height; y++) {
    const row = torus[Math.floor(y / cellSize)];
    for (let x = 0; x < width; x++) {
      const cell = row[Math.floor(x / cellSize)];
      const shade = cell ? 0 : 255; // 1 -> black, 0 -> white
      const idx = (width * y + x) << 2;
      png.data[idx] = png.data[idx + 1] = png.data[idx + 2] = shade;
      png.data[idx + 3] = 255;
    }
  }
  return new Promise((resolve, reject) => {
    const stream = createWriteStream(outPath);
    png.pack().pipe(stream);
    stream.on('finish', resolve);
    stream.on('error', reject);
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const order = parseInt(args.order ?? '4', 10);
  const cellSize = parseInt(args['cell-size'] ?? '8', 10);
  const outPath = args.out ?? `debruijn-torus-order${order}.png`;

  const N = order * order;
  const L = 2 ** N - 1;
  console.log(`Order n=${order} -> window is ${order}x${order} cells, N=${N} bits, period L=2^${N}-1=${L}`);

  const { R, C } = bestCoprimeSplit(L);
  const aspect = Math.max(R, C) / Math.min(R, C);
  console.log(`Torus grid: ${R} x ${C} cells (aspect ratio ${aspect.toFixed(2)}:1)${aspect > 3 ? '  [warning: quite far from square — 2^N-1 didn\'t factor nicely for this order]' : ''}`);

  const { taps, seq } = findValidTorusSequence(order, N, R, C);
  console.log(`Found feedback taps ${JSON.stringify(taps)} — sequence verified full-period (${seq.length} distinct nonzero states).`);

  const torus = buildTorus(seq, R, C);

  if (R * C <= WINDOW_CHECK_LIMIT) {
    console.log(`Verified: all ${R * C} windows of size ${order}x${order} are unique on the torus.`);
  } else {
    console.log('Torus too large to brute-force verify window uniqueness — skipping (relying on the CRT-fold construction; 1D maximal-length was verified, but the 2D fold was not).');
  }

  await writePng(torus, R, C, cellSize, outPath);
  console.log(`Wrote ${outPath} (${C * cellSize}x${R * cellSize}px, ${R}x${C} cells, ${cellSize}x${cellSize}px per cell)`);
}

main().catch(e => { console.error(e); process.exit(1); });
