// Generates a binary De Bruijn torus PNG: an R x C toroidal grid of black/white
// cells such that every n x n window (wrapping around both edges) is unique.
// Point a camera at any n x n patch and its content alone identifies exactly
// where on the mat it is.
//
// Construction (classical LFSR + CRT fold, MacWilliams & Sloane 1976):
//   1. N = n * n is the window's bit count. Generate a maximal-length LFSR
//      ("m-sequence") of degree N using a known primitive polynomial. Its
//      period is L = 2^N - 1, and — by definition of "maximal length" — every
//      nonzero N-bit window appears in it exactly once as it cycles.
//   2. Factor L into two coprime numbers R and C (R * C = L) as close to
//      square as possible. By the Chinese Remainder Theorem, t -> (t mod R,
//      t mod C) is a bijection from Z_L to Z_R x Z_C, so folding the sequence
//      that way fills an R x C torus using every position exactly once.
//   3. The 1D window-uniqueness property is known to carry over to n x n
//      windows on the folded torus. Since getting this fold subtly wrong is
//      an easy way to silently produce a *broken* pattern, this script also
//      brute-force re-verifies uniqueness on the actual output (for sizes
//      where that's tractable) instead of just trusting the theorem.
//
// Usage:
//   node scripts/generate-debruijn-torus.ts --order 4 --side 2048 --out torus.png
//
//   --order  Window order n (window is n x n cells). Determines how many
//            cells the camera must see to uniquely determine its position.
//            Supported: 2-5, bounded by the primitive-polynomial table below
//            (needs an entry for N = order^2, table goes up to degree 32) —
//            order 5 alone is a 25-bit torus, ~33.5M cells, already a lot to
//            print, so in practice order 3 or 4 is the realistic range.
//   --side   Output PNG width & height in pixels (square image).
//   --out    Output file path (default: debruijn-torus-order<n>.png).

import { PNG } from 'pngjs';
import { createWriteStream } from 'node:fs';

// Standard maximal-length (primitive polynomial) Fibonacci LFSR taps, degree -> tap
// positions (1-indexed from the LSB). Widely published table (e.g. Xilinx XAPP052 /
// Ward & Molteno). Correctness for each entry is re-verified at runtime below rather
// than assumed, since a wrong tap silently produces a non-primitive (broken) sequence.
const LFSR_TAPS: Record<number, number[]> = {
  2: [2, 1], 3: [3, 2], 4: [4, 3], 5: [5, 3], 6: [6, 5], 7: [7, 6],
  8: [8, 6, 5, 4], 9: [9, 5], 10: [10, 7], 11: [11, 9], 12: [12, 6, 4, 1],
  13: [13, 4, 3, 1], 14: [14, 5, 3, 1], 15: [15, 14], 16: [16, 15, 13, 4],
  17: [17, 14], 18: [18, 11], 19: [19, 6, 2, 1], 20: [20, 17], 21: [21, 19],
  22: [22, 21], 23: [23, 18], 24: [24, 23, 22, 17], 25: [25, 22],
  26: [26, 6, 2, 1], 27: [27, 5, 2, 1], 28: [28, 25], 29: [29, 27],
  30: [30, 6, 4, 1], 31: [31, 28], 32: [32, 22, 2, 1],
};

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

// Generates the m-sequence: L = 2^N - 1 bits, every nonzero N-bit window
// (read cyclically) appears exactly once.
function generateMSequence(N: number): Uint8Array {
  const taps = LFSR_TAPS[N];
  if (!taps) throw new Error(`No primitive polynomial on file for degree ${N} (order^2). Supported N: 2-32.`);
  const L = 2 ** N - 1;
  const seq = new Uint8Array(L);
  let state = 1; // any nonzero seed
  for (let i = 0; i < L; i++) {
    seq[i] = state & 1;
    let feedback = 0;
    for (const t of taps) feedback ^= (state >>> (t - 1)) & 1;
    state = (state >>> 1) | (feedback << (N - 1));
  }
  return seq;
}

// Verifies the LFSR actually is maximal-length: all L generated N-bit states
// are distinct and nonzero. Catches a wrong/non-primitive table entry.
function verifyMSequence(seq: Uint8Array, N: number): boolean {
  const L = seq.length;
  const seen = new Uint8Array(L + 1);
  let state = 0;
  for (let i = 0; i < N; i++) state = (state << 1) | seq[i]; // seed with first N bits
  for (let i = 0; i < L; i++) {
    if (state === 0 || seen[state]) return false;
    seen[state] = 1;
    const nextBit = seq[(i + N) % L];
    state = ((state << 1) | nextBit) & ((1 << N) - 1);
  }
  return true;
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

function writePng(torus: Uint8Array[], R: number, C: number, side: number, outPath: string): Promise<void> {
  const png = new PNG({ width: side, height: side });
  for (let y = 0; y < side; y++) {
    const row = torus[Math.floor((y * R) / side)];
    for (let x = 0; x < side; x++) {
      const cell = row[Math.floor((x * C) / side)];
      const shade = cell ? 0 : 255; // 1 -> black, 0 -> white
      const idx = (side * y + x) << 2;
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
  const side = parseInt(args.side ?? '2048', 10);
  const outPath = args.out ?? `debruijn-torus-order${order}.png`;

  const N = order * order;
  console.log(`Order n=${order} -> window is ${order}x${order} cells, N=${N} bits, period L=2^${N}-1=${2 ** N - 1}`);

  const seq = generateMSequence(N);

  const SEQ_CHECK_LIMIT = 2 ** 26; // ~64M, keeps the visited array under ~64MB
  if (seq.length <= SEQ_CHECK_LIMIT) {
    if (!verifyMSequence(seq, N)) {
      throw new Error(`LFSR for order ${order} (N=${N}) is not maximal-length — the tap table entry is wrong. Aborting rather than emit a broken pattern.`);
    }
    console.log('m-sequence verified maximal-length (all nonzero N-bit windows distinct).');
  } else {
    console.log('m-sequence too large to verify exhaustively — skipping (order is likely impractically large anyway).');
  }

  const { R, C } = bestCoprimeSplit(seq.length);
  const aspect = Math.max(R, C) / Math.min(R, C);
  console.log(`Torus grid: ${R} x ${C} cells (aspect ratio ${aspect.toFixed(2)}:1)${aspect > 3 ? '  [warning: quite far from square — 2^N-1 didn\'t factor nicely for this order]' : ''}`);

  const torus = buildTorus(seq, R, C);

  const WINDOW_CHECK_LIMIT = 2 ** 22; // ~4M cells, keeps this O(R*C*n^2) check fast
  if (R * C <= WINDOW_CHECK_LIMIT) {
    if (!verifyTorusWindows(torus, R, C, order, N)) {
      throw new Error('Torus fold produced duplicate windows — construction is broken for this order. Aborting rather than emit a broken pattern.');
    }
    console.log(`Verified: all ${R * C} windows of size ${order}x${order} are unique on the torus.`);
  } else {
    console.log('Torus too large to brute-force verify window uniqueness — skipping (relying on the CRT-fold construction).');
  }

  await writePng(torus, R, C, side, outPath);
  console.log(`Wrote ${outPath} (${side}x${side}px, ${R}x${C} cells, ~${(side / C).toFixed(1)}x${(side / R).toFixed(1)}px per cell)`);
}

main().catch(e => { console.error(e); process.exit(1); });
