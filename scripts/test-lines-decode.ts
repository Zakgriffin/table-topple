// End-to-end validation of the NEW line-based rectification pipeline
// (buildLineAccumulator -> findLinePeaksTiered -> splitIntoTwoFamilies ->
// indexFamilyLines -> buildLatticeCorrespondences -> fitHomographyRobust ->
// sample -> pickBestCandidate) against real perspective tilt — directly
// comparable to scripts/test-homography-decode.ts (the OLD corner+VP+mesh
// pipeline, which measured 0/8 correct at every tilt) and
// scripts/test-mesh-decode.ts (the original BFS mesh, same result). This is
// the actual test of whether the redesign discussed at length beats either
// prior architecture, not just whether its pieces are accurate in isolation
// (already validated separately in test-hough-lines.ts, test-vp.ts,
// test-vp-split.ts, test-lattice-homography.ts).
//
// Usage: node scripts/test-lines-decode.ts [order]

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { generateTorus, buildLookupTable } from '../src/debruijn.ts';
import { toGrayscale, binarize, pickBestCandidate, rotateShift, type SampledGrid, type SampledCell } from '../src/decode.ts';
import { buildLineAccumulator, findLinePeaksTiered } from '../src/lines.ts';
import { splitIntoTwoFamilies } from '../src/vp.ts';
import { indexFamilyLines, buildLatticeCorrespondences } from '../src/lattice.ts';
import { fitHomographyRobust, applyHomography, invertHomography, type Mat3 } from '../src/homography.ts';
import { captureHomography } from './lib/synth-camera.ts';
import type { CameraPose } from './lib/synth-camera.ts';

const order = parseInt(process.argv[2] ?? '4', 10);
const debruijn = generateTorus(order);
const lookup = buildLookupTable(debruijn);
const { R, C } = debruijn;

const png = PNG.sync.read(readFileSync(`samples/order${order}.png`));
const cellPx = png.width / C;
console.log(`Loaded samples/order${order}.png: ${cellPx}px/cell, torus ${R}x${C}`);

const RAW = 300;
const DIST = 300, FOCAL = 300;
const CONFIDENCE_THRESHOLD = 0.85;
const TRIALS = 8;

function sampleFromHomography(bin: Uint8Array, w: number, h: number, H: Mat3, rowCount: number, colCount: number): SampledGrid {
  const cells: SampledCell[][] = [];
  for (let i = 0; i < rowCount; i++) {
    const rowCells: SampledCell[] = [];
    for (let j = 0; j < colCount; j++) {
      const p = applyHomography(H, i + 0.5, j + 0.5); // cell center = midpoint of its 4 lattice corners
      if (!p) { rowCells.push({ x: NaN, y: NaN, bit: 0, valid: false, cornerCount: 0 }); continue; }
      const [px, py] = p;
      const xx = Math.round(px), yy = Math.round(py);
      if (xx < 0 || xx >= w || yy < 0 || yy >= h) { rowCells.push({ x: px, y: py, bit: 0, valid: false, cornerCount: 0 }); continue; }
      rowCells.push({ x: px, y: py, bit: bin[yy * w + xx], valid: true, cornerCount: 4 });
    }
    cells.push(rowCells);
  }
  return { rows: rowCount, cols: colCount, cells, originRow: 0, originCol: 0 };
}

function mirrorRows(sg: SampledGrid): SampledGrid {
  return { ...sg, cells: sg.cells.slice().reverse() };
}

// Fixed rather than adaptive, matching src/main.ts's HOUGH_RHO_BIN_PX: no
// lines exist yet at this point to measure a real local pitch from, and an
// earlier adaptive version (derotate a small patch near the image center,
// autocorrelate) was only ever a GLOBAL proxy that's systematically wrong
// elsewhere in the frame under perspective. A small fixed size favors
// occasional harmless duplicate peaks (absorbed by Level 3's gap-tolerant
// indexing) over merged real lines (an unrecoverable information loss).
const HOUGH_RHO_BIN_PX = 1.5;
const HOUGH_THETA_BINS = Math.round(360 / HOUGH_RHO_BIN_PX);
const RESCUE_THRESHOLD_FRACTION = 0.3; // matches src/main.ts's -- see its comment for why

interface DecodeOutcome { match: { row: number; col: number } | null; consistency: number; }

function decodeViaLines(pose: CameraPose): DecodeOutcome | 'nolines' | 'nosplit' | 'nogrid' | 'nohomography' {
  const rgba = captureHomography(png, pose, RAW, RAW, 4);
  const gray = toGrayscale(rgba, RAW, RAW);
  const bin = binarize(gray);

  const field = buildLineAccumulator(gray, RAW, RAW, HOUGH_THETA_BINS, HOUGH_RHO_BIN_PX);
  const { strong: peaks, weak: rescuePeaks } = findLinePeaksTiered(field, 0.15, 0.15 * RESCUE_THRESHOLD_FRACTION, 4, 3);
  if (peaks.length < 8) return 'nolines';

  let split;
  try { split = splitIntoTwoFamilies(peaks, RAW, RAW, 6, 60, rescuePeaks); } catch { return 'nosplit'; }
  const { familyA, familyB } = split;
  if (familyA.lines.length < 3 || familyB.lines.length < 3) return 'nosplit';

  const rowIndexed = indexFamilyLines(familyA, familyB.vp, RAW, RAW);
  const colIndexed = indexFamilyLines(familyB, familyA.vp, RAW, RAW);
  const correspondences = buildLatticeCorrespondences(rowIndexed, colIndexed, RAW, RAW);
  const H = fitHomographyRobust(correspondences);
  if (!H) return 'nohomography';

  const rows = rowIndexed.length - 1, cols = colIndexed.length - 1;
  if (rows < order || cols < order) return 'nogrid';

  const sg = sampleFromHomography(bin, RAW, RAW, H, rows, cols);
  // indexFamilyLines' sort direction per family is arbitrary (no reference
  // to which way is "increasing" in the real pattern), so besides the
  // 0/90/180/270 rotation ambiguity pickBestCandidate already resolves, a
  // single-axis MIRROR is also possible (row order flipped, col order not,
  // or vice versa) — not a rotation, so not covered by that search. Feeding
  // both the direct and row-mirrored grid as separate candidates covers all
  // 8 dihedral symmetries (mirror x 4 rotations = the other 4 dihedral cases).
  const result = pickBestCandidate([sg, mirrorRows(sg)], order, lookup, debruijn.torus, R, C);
  if (!result.match) return result;

  // pickBestCandidate's match is the SAMPLED GRID's CENTER cell's torus
  // position — that only equals the camera's actual target here if grid
  // index (0,0) happens to sit at a fixed, known offset from image center,
  // which indexFamilyLines does NOT guarantee (its index origin is just
  // whichever detected line ended up smallest after normalization, with no
  // relation to the image center — unlike the old mesh pipeline, which
  // explicitly seeded coordinates from rawW/2, rawH/2). Comparing match
  // directly against the pose's target under that assumption is WRONG, not
  // the decode itself — confirmed the hard way: a "wrong, dr=7 dc=3" case
  // resolved to dr=1 dc=0 once corrected this way. Fix: map the camera's
  // true target (which always projects to the image center by construction
  // of makeHomographySampler/projectToImage) through H's INVERSE to find
  // which grid cell it actually falls in, then read the torus position AT
  // THAT cell instead of at the grid's arbitrary center.
  const Hinv = invertHomography(H);
  if (!Hinv) return result;
  const gridPos = applyHomography(Hinv, RAW / 2, RAW / 2);
  if (!gridPos) return result;
  let targetGridI = gridPos[0], targetGridJ = gridPos[1];
  // mirrorRows is .slice().reverse() on a length-`rows` array, which maps
  // index k -> (rows-1)-k, NOT rows-k -- the same off-by-one already found
  // and fixed elsewhere this session, just never applied to this spot.
  if (result.candidateIndex === 1) targetGridI = (rows - 1) - targetGridI;
  // Must subtract the SAME reference pickBestCandidate itself used
  // (Math.floor(sg.rows/2), see its centerI/centerJ) -- subtracting the raw
  // rows/2 instead is off by 0.5 whenever rows is odd, which can round to a
  // different integer than intended depending on targetGridI's fractional
  // part, silently mislabeling an otherwise-correct decode as "wrong".
  const [dI, dJ] = rotateShift(Math.round(targetGridI - Math.floor(rows / 2)), Math.round(targetGridJ - Math.floor(cols / 2)), result.orientation ?? 0);
  return {
    ...result,
    match: { row: ((result.match.row + dI) % R + R) % R, col: ((result.match.col + dJ) % C + C) % C },
  };
}

function within(target: number, start: number, span: number, mod: number): boolean {
  const rel = ((target - start) % mod + mod) % mod;
  return rel <= span || rel >= mod - 1;
}

// Seeded (not Math.random()) so runs are reproducible -- this is THE test
// that measures whether a pipeline change actually helps or hurts on real
// end-to-end behavior, which is meaningless to compare across runs if every
// invocation samples different random camera poses.
function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

console.log(`\nEnd-to-end line-based decode vs perspective tilt (${TRIALS} trials/tilt):`);
for (const tiltDeg of [0, 10, 20, 30, 40, 50, 60]) {
  let hits = 0, misses = 0, wrong = 0, nolines = 0, nosplit = 0, nogrid = 0, nohomography = 0;
  const scores: number[] = [];
  const rnd = mulberry32(tiltDeg * 1000 + 7);
  for (let t = 0; t < TRIALS; t++) {
    const testRow = Math.floor(rnd() * R);
    const testCol = Math.floor(rnd() * C);
    const pose: CameraPose = {
      targetX: testCol * cellPx + cellPx / 2, targetY: testRow * cellPx + cellPx / 2,
      dist: DIST, focal: FOCAL, tilt: tiltDeg * Math.PI / 180,
      azimuth: rnd() * 2 * Math.PI, roll: rnd() * 2 * Math.PI,
    };
    const result = decodeViaLines(pose);
    if (result === 'nolines') { nolines++; continue; }
    if (result === 'nosplit') { nosplit++; continue; }
    if (result === 'nogrid') { nogrid++; continue; }
    if (result === 'nohomography') { nohomography++; continue; }
    if (!result.match || result.consistency < CONFIDENCE_THRESHOLD) { misses++; continue; }
    scores.push(result.consistency);
    if (within(testRow, result.match.row, order, R) && within(testCol, result.match.col, order, C)) hits++;
    else wrong++;
  }
  const meanScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3) : 'n/a';
  console.log(
    `  tilt ${String(tiltDeg).padStart(2)}deg: ${hits}/${TRIALS} correct, ${misses} no-lock, ${wrong} wrong, ` +
    `nolines=${nolines} nosplit=${nosplit} nogrid=${nogrid} nohomography=${nohomography}, mean score ${meanScore}`
  );
}
