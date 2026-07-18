// Worker thread for search-order5-torus.ts -- see that file for the overall
// picture. Each worker independently and indefinitely: picks a random
// (weight, prime-derived seed) pair, searches for a maximal-length degree-25
// LFSR tap-set under it, builds the resulting order-5 torus, then samples
// many random 256x256 sub-region offsets within it, scoring each by how many
// of its 5x5 windows collide with another window under rotation/reflection
// (the D4 symmetry group) -- see the main script and the conversation for
// why this number matters (it's the residual ambiguity a single-window
// decode would have to fall back to multi-window voting for).
//
// Reimplements debruijn.ts's tap search rather than importing it, since the
// real findValidTorusSequence is unexported AND hardcodes its own seed
// derivation (deterministic per weight, not externally randomizable) -- this
// version takes an explicit seed so many workers can explore independently
// instead of all finding the same first candidate.
import { parentPort, workerData } from 'node:worker_threads';

const ORDER = 5;
const N = ORDER * ORDER; // 25
const R = 18631, C = 1801; // fixed coprime split of 2^25-1, see debruijn.ts's bestCoprimeSplit
const BOARD = 256;
const OFFSETS_PER_TORUS = 25; // how many random sub-regions to sample before generating a fresh torus

function tryTaps(taps: number[], N: number): Uint8Array | null {
  const L = 2 ** N - 1;
  const seen = new Uint8Array(L + 1);
  const seq = new Uint8Array(L);
  let state = 1;
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
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function stratifiedTaps(N: number, weight: number, rng: () => number): number[] {
  const taps = new Set<number>();
  for (let i = 0; i < weight; i++) {
    const lo = 1 + Math.floor((i * (N - 1)) / weight);
    const hi = 1 + Math.floor(((i + 1) * (N - 1)) / weight) - 1;
    taps.add(Math.min(lo + Math.floor(rng() * (hi - lo + 1)), N - 1));
  }
  return [N, ...taps];
}
function buildTorus(seq: Uint8Array, R: number, C: number): Uint8Array[] {
  const torus = Array.from({ length: R }, () => new Uint8Array(C));
  for (let t = 0; t < seq.length; t++) torus[t % R][t % C] = seq[t];
  return torus;
}

function packRotated(cells: number[][], order: number, orientation: number): number {
  let key = 0;
  for (let i = 0; i < order; i++) for (let j = 0; j < order; j++) {
    let bit: number;
    if (orientation === 1) bit = cells[order - 1 - j][i];
    else if (orientation === 2) bit = cells[order - 1 - i][order - 1 - j];
    else if (orientation === 3) bit = cells[j][order - 1 - i];
    else bit = cells[i][j];
    key = (key << 1) | bit;
  }
  return key >>> 0;
}
function mirrorCells(cells: number[][], order: number): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < order; i++) out.push(cells[order - 1 - i].slice());
  return out;
}

function analyzeRegion(torus: Uint8Array[], r0: number, c0: number) {
  const win = BOARD - ORDER + 1;
  const cellsAt = (r: number, c: number): number[][] => {
    const block: number[][] = [];
    for (let i = 0; i < ORDER; i++) {
      const row: number[] = [];
      for (let j = 0; j < ORDER; j++) row.push(torus[(r0 + r + i) % R][(c0 + c + j) % C]);
      block.push(row);
    }
    return block;
  };
  const canonicalKeyToPos = new Map<number, string>();
  for (let r = 0; r < win; r++) for (let c = 0; c < win; c++) canonicalKeyToPos.set(packRotated(cellsAt(r, c), ORDER, 0), `${r},${c}`);

  let reflCount = 0, rotCount = 0, bothCount = 0, anyCount = 0;
  for (let r = 0; r < win; r++) {
    for (let c = 0; c < win; c++) {
      const cells = cellsAt(r, c);
      const selfPos = `${r},${c}`;
      let refl = false, rot = false;
      for (let o = 1; o <= 3; o++) {
        const other = canonicalKeyToPos.get(packRotated(cells, ORDER, o));
        if (other !== undefined && other !== selfPos) { rot = true; break; }
      }
      const mirrored = mirrorCells(cells, ORDER);
      for (let o = 0; o <= 3; o++) {
        const other = canonicalKeyToPos.get(packRotated(mirrored, ORDER, o));
        if (other !== undefined && other !== selfPos) { refl = true; break; }
      }
      if (refl) reflCount++;
      if (rot) rotCount++;
      if (refl && rot) bothCount++;
      if (refl || rot) anyCount++;
    }
  }
  return { reflCount, rotCount, bothCount, anyCount, totalWindows: win * win };
}

// Sieve of Eratosthenes -- a decent-sized pool of primes to draw seeds from
// (the user's instinct to try primes for seed/parameter variety, rather than
// sequential or Date.now()-based ones, which could correlate across workers
// in ways primes are less likely to).
function sievePrimes(limit: number): number[] {
  const isComposite = new Uint8Array(limit + 1);
  const primes: number[] = [];
  for (let i = 2; i <= limit; i++) {
    if (!isComposite[i]) {
      primes.push(i);
      for (let j = i * i; j <= limit; j += i) isComposite[j] = 1;
    }
  }
  return primes;
}
const PRIMES = sievePrimes(1_000_003);
const ODD_WEIGHTS = [5, 7, 9, 11, 13, 15, 17, 19]; // extra-tap count beyond N itself; codebase's own convention keeps this odd

const workerId: number = workerData.workerId;
let localBestAny = Infinity;
let evaluatedTori = 0, evaluatedOffsets = 0;
const t0 = Date.now();

function randPrime(): number { return PRIMES[Math.floor(Math.random() * PRIMES.length)]; }

function loop() {
  const weight = ODD_WEIGHTS[Math.floor(Math.random() * ODD_WEIGHTS.length)];
  const primeA = randPrime(), primeB = randPrime();
  // Combine two independent primes with the worker id and a large odd
  // multiplier (same style debruijn.ts's own seed derivation uses) so
  // different workers, even if they happen to draw the same primes, don't
  // collide on the same seed.
  const seed = (Math.imul(primeA, 2654435761) ^ Math.imul(primeB, 40503) ^ Math.imul(workerId + 1, 97)) >>> 0;
  const rng = mulberry32(seed);

  let found: { taps: number[]; seq: Uint8Array } | null = null;
  for (let attempt = 0; attempt < 500 && !found; attempt++) {
    const taps = stratifiedTaps(N, weight, rng);
    if (taps.length !== weight + 1) continue;
    const seq = tryTaps(taps, N);
    if (seq) found = { taps, seq };
  }
  if (!found) { setImmediate(loop); return; }
  evaluatedTori++;
  const torus = buildTorus(found.seq, R, C);

  for (let k = 0; k < OFFSETS_PER_TORUS; k++) {
    const r0 = Math.floor(Math.random() * (R - BOARD));
    const c0 = Math.floor(Math.random() * (C - BOARD));
    const result = analyzeRegion(torus, r0, c0);
    evaluatedOffsets++;
    if (result.anyCount < localBestAny) {
      localBestAny = result.anyCount;
      parentPort!.postMessage({
        type: 'candidate', workerId, weight, seed, primeA, primeB, taps: found.taps, r0, c0, ...result,
      });
    }
  }
  parentPort!.postMessage({ type: 'progress', workerId, evaluatedTori, evaluatedOffsets, elapsedMs: Date.now() - t0 });
  setImmediate(loop); // yield between rounds rather than blocking the event loop forever in one synchronous call
}
loop();