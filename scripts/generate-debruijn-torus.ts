// Generates a binary De Bruijn torus PNG: an R x C toroidal grid of black/white
// cells such that every n x n window (wrapping around both edges) is unique.
// Point a camera at any n x n patch and its content alone identifies exactly
// where on the mat it is.
//
// Construction lives in src/debruijn.ts, shared with the browser tracker app
// (src/main.ts) — both independently rebuild the identical pattern from just
// the order number, so there's no pixel/table asset to keep in sync.
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
import { generateTorus, WINDOW_CHECK_LIMIT } from '../src/debruijn.ts';

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
  console.log(`Order n=${order} -> window is ${order}x${order} cells, N=${N} bits, period L=2^${N}-1=${2 ** N - 1}`);

  const { R, C, taps, torus } = generateTorus(order);
  const aspect = Math.max(R, C) / Math.min(R, C);
  console.log(`Torus grid: ${R} x ${C} cells (aspect ratio ${aspect.toFixed(2)}:1)${aspect > 3 ? '  [warning: quite far from square — 2^N-1 didn\'t factor nicely for this order]' : ''}`);
  console.log(`Feedback taps ${JSON.stringify(taps)} — sequence verified full-period and (where tractable) torus-window-unique.`);

  if (R * C <= WINDOW_CHECK_LIMIT) {
    console.log(`Verified: all ${R * C} windows of size ${order}x${order} are unique on the torus.`);
  } else {
    console.log('Torus too large to brute-force verify window uniqueness — skipping (relying on the CRT-fold construction; 1D maximal-length was verified, but the 2D fold was not).');
  }

  await writePng(torus, R, C, cellSize, outPath);
  console.log(`Wrote ${outPath} (${C * cellSize}x${R * cellSize}px, ${R}x${C} cells, ${cellSize}x${cellSize}px per cell)`);
}

main().catch(e => { console.error(e); process.exit(1); });
