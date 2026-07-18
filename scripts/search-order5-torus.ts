// Multi-core search for a low-symmetry-collision order-5 De Bruijn torus
// sub-region -- see the conversation this came out of for the full
// background. Short version: an order-4 torus (the current floor pattern)
// has ~98% of its 5x5 windows collide with some OTHER window under
// rotation/reflection (mostly reflection), which is why decoding needs
// multi-window voting to disambiguate orientation today. An order-5 torus
// is astronomically larger (~33.5M cells) than the 256x256 (65536-cell)
// board we'd actually use, so sparsely sampling a sub-region from it drops
// that collision rate to roughly 1.2-1.4% -- confirmed empirically, and
// confirmed NOT improvable by brute-force alone (varying sub-region offset
// or trying different tap-sets both plateau in that same range, consistent
// with published research: there's no known efficient construction for
// torus patterns unique under the full rotation+reflection symmetry group,
// only expensive genetic-algorithm search at supercomputer scale for the
// one known working system, Uniform Marker Fields).
//
// This is a cheaper, narrower version of that same idea: instead of a full
// genetic-algorithm search, brute-force sample many (tap-set, sub-region
// offset) combinations in parallel across all CPU cores, keeping whichever
// candidate has the fewest colliding 5x5 windows found so far. Won't reach
// zero (see above), but can likely beat the ~1.24% best found by a single-
// threaded, shorter search.
//
// Usage:
//   node scripts/search-order5-torus.ts [--workers N]
//
// Runs indefinitely -- Ctrl+C to stop. Prints a line every time a new best
// (fewest total colliding windows) is found, with every parameter needed to
// reproduce it, and overwrites best-order5-candidate.json in this directory
// on each improvement so progress survives a crash or an accidental Ctrl+C.
import { Worker } from 'node:worker_threads';
import os from 'node:os';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

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

const args = parseArgs(process.argv.slice(2));
const numWorkers = parseInt(args.workers ?? '', 10) || os.cpus().length;
const workerPath = fileURLToPath(new URL('./search-order5-torus-worker.ts', import.meta.url));
const outPath = fileURLToPath(new URL('./best-order5-candidate.json', import.meta.url));

let globalBestAny = Infinity;
let totalTori = 0, totalOffsets = 0;
const startTime = Date.now();

console.log(`Starting ${numWorkers} workers (order 5, 256x256 board, R=18631 C=1801).`);
console.log(`Best candidate so far will be written to ${outPath} on every improvement.`);
console.log(`Ctrl+C to stop.\n`);

const perWorkerProgress = new Map<number, { tori: number; offsets: number }>();

const workers: Worker[] = [];
for (let i = 0; i < numWorkers; i++) {
  const worker = new Worker(workerPath, { workerData: { workerId: i } });
  workers.push(worker);
  worker.on('message', (msg: any) => {
    if (msg.type === 'candidate') {
      if (msg.anyCount < globalBestAny) {
        globalBestAny = msg.anyCount;
        const pct = (100 * msg.anyCount / msg.totalWindows).toFixed(3);
        const reflPct = (100 * msg.reflCount / msg.totalWindows).toFixed(3);
        const rotPct = (100 * msg.rotCount / msg.totalWindows).toFixed(3);
        console.log(
          `[NEW BEST] any=${msg.anyCount} (${pct}%)  refl=${msg.reflCount} (${reflPct}%)  rot=${msg.rotCount} (${rotPct}%)  both=${msg.bothCount}\n` +
          `           weight=${msg.weight}  seed=${msg.seed}  primeA=${msg.primeA}  primeB=${msg.primeB}  taps=${JSON.stringify(msg.taps)}\n` +
          `           r0=${msg.r0}  c0=${msg.c0}  worker=${msg.workerId}\n`
        );
        writeFileSync(outPath, JSON.stringify(msg, null, 2));
      }
    } else if (msg.type === 'progress') {
      perWorkerProgress.set(msg.workerId, { tori: msg.evaluatedTori, offsets: msg.evaluatedOffsets });
    }
  });
  worker.on('error', (e) => console.error(`worker ${i} error:`, e));
}

// Periodic heartbeat so it's clear the search is still alive even between
// improvements, which get rarer as the best-so-far tightens.
setInterval(() => {
  totalTori = 0; totalOffsets = 0;
  for (const { tori, offsets } of perWorkerProgress.values()) { totalTori += tori; totalOffsets += offsets; }
  const elapsedS = ((Date.now() - startTime) / 1000).toFixed(0);
  console.log(`... ${elapsedS}s elapsed, ${totalTori} tori generated, ${totalOffsets} offsets checked, best any=${globalBestAny === Infinity ? 'none yet' : globalBestAny}`);
}, 15000);

process.on('SIGINT', () => {
  console.log('\nStopping workers...');
  for (const w of workers) w.terminate();
  process.exit(0);
});
