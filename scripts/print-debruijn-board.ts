// Takes a JSON config (see scripts/debruijn-print-config.json for an
// example) describing the De Bruijn board's seed parameters and physical
// print dimensions, and produces a folder of print-ready PNG tiles sized to
// real-world paper -- one tile per page, each labeled (R,C), with
// single-sided registration overlap between neighbors, upscaled to a real
// dpi so physical scaling is preserved without relying on any viewer
// reading PNG's pHYs metadata correctly. Also combines every tile into a
// single PDF (one full-paper-size page per tile, image placed at its
// natural/unscaled size inside the page's margin) for one-shot printing.
//
// Construction reuses src/debruijn.ts's buildTorusFromCandidate (shared
// with the browser tracker app and scripts/generate-debruijn-torus.ts) --
// see this session's "Print-ready De Bruijn board PNG tiler" plan for the
// full tiling-algorithm derivation.
//
// Usage:
//   node scripts/print-debruijn-board.ts <config.json>
//
// No field has a default -- every value below must be present in the JSON
// or the script fails with a specific message naming what's missing.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { deflateSync } from 'node:zlib';
import { bestCoprimeSplit, buildTorusFromCandidate } from '../src/pose/debruijn.ts';
import { PNG } from 'pngjs';

// ── Config schema + validation ───────────────────────────────────────────

interface Length { value: number; unit: 'cm' | 'in' }
interface Config {
  debruijn: { order: number; taps: number[]; cropOrigin: { row: number; col: number } };
  boardSize: Length; cellSize: Length; tileOverlap: Length;
  paper: { width: Length; height: Length; margin: Length };
  dpi: number;
  outputDir: string;
}

const IN_TO_CM = 2.54;

function toCm(len: unknown, fieldName: string): number {
  const l = len as Partial<Length> | undefined;
  if (!l || typeof l.value !== 'number' || (l.unit !== 'cm' && l.unit !== 'in')) {
    throw new Error(`${fieldName} must be { value: number, unit: "cm" | "in" }, got ${JSON.stringify(len)}`);
  }
  return l.unit === 'in' ? l.value * IN_TO_CM : l.value;
}

// Every physical/count field must land on a whole number of cells -- a
// fractional cell would mean the pattern data and the printed pixel grid
// have silently drifted out of the 1-pixel-per-cell correspondence the rest
// of this script assumes throughout.
function toWholeCells(cm: number, cellCm: number, fieldName: string): number {
  const cells = cm / cellCm;
  const rounded = Math.round(cells);
  if (Math.abs(cells - rounded) > 1e-6) {
    throw new Error(`${fieldName} (${cm}cm) is not a whole number of cells at ${cellCm}cm/cell (computed ${cells})`);
  }
  return rounded;
}

function need<T>(obj: any, key: string, ownerPath: string): T {
  if (obj == null || obj[key] === undefined) throw new Error(`Missing required config field: ${ownerPath}${key}`);
  return obj[key];
}

function loadConfig(configPath: string): Config {
  const raw = JSON.parse(readFileSync(configPath, 'utf8'));

  const debruijn = need<any>(raw, 'debruijn', '');
  const order = need<number>(debruijn, 'order', 'debruijn.');
  const taps = need<number[]>(debruijn, 'taps', 'debruijn.');
  const cropOrigin = need<any>(debruijn, 'cropOrigin', 'debruijn.');
  const cropRow = need<number>(cropOrigin, 'row', 'debruijn.cropOrigin.');
  const cropCol = need<number>(cropOrigin, 'col', 'debruijn.cropOrigin.');
  if (typeof order !== 'number' || order <= 0) throw new Error('debruijn.order must be a positive number');
  if (!Array.isArray(taps) || taps.length === 0 || !taps.every((t) => typeof t === 'number')) {
    throw new Error('debruijn.taps must be a non-empty array of numbers');
  }
  if (typeof cropRow !== 'number' || typeof cropCol !== 'number') {
    throw new Error('debruijn.cropOrigin.row/col must both be numbers');
  }

  const boardSize = need<Length>(raw, 'boardSize', '');
  const cellSize = need<Length>(raw, 'cellSize', '');
  const tileOverlap = need<Length>(raw, 'tileOverlap', '');
  const paper = need<any>(raw, 'paper', '');
  const paperWidth = need<Length>(paper, 'width', 'paper.');
  const paperHeight = need<Length>(paper, 'height', 'paper.');
  const paperMargin = need<Length>(paper, 'margin', 'paper.');
  const dpi = need<number>(raw, 'dpi', '');
  if (typeof dpi !== 'number' || dpi <= 0) throw new Error('dpi must be a positive number');
  const outputDir = need<string>(raw, 'outputDir', '');
  if (typeof outputDir !== 'string' || outputDir.length === 0) throw new Error('outputDir must be a non-empty string');

  return {
    debruijn: { order, taps, cropOrigin: { row: cropRow, col: cropCol } },
    boardSize, cellSize, tileOverlap,
    paper: { width: paperWidth, height: paperHeight, margin: paperMargin },
    dpi,
    outputDir,
  };
}

// ── PNG pHYs chunk (physical pixel resolution) -- pngjs has no built-in
// support for writing this chunk, so it's spliced into the encoded buffer
// by hand: a standard 21-byte PNG chunk (length + type + 9 data bytes +
// CRC32), inserted right after IHDR (always the first chunk, at a fixed
// offset) and before IDAT. This is what makes "print at actual size" in any
// PNG-aware print pipeline reproduce the true physical cell size. ──

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(buf: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) crc = CRC_TABLE[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function buildChunk(type: string, data: Buffer): Buffer {
  const typeBuf = Buffer.from(type, 'ascii');
  const lenBuf = Buffer.alloc(4);
  lenBuf.writeUInt32BE(data.length, 0);
  const crcInput = Buffer.concat([typeBuf, data]);
  const crcBuf = Buffer.alloc(4);
  crcBuf.writeUInt32BE(crc32(crcInput), 0);
  return Buffer.concat([lenBuf, typeBuf, data, crcBuf]);
}

// unit specifier 1 = meter (the only other defined value, 0, means
// "unknown/aspect-ratio-only" -- not what we want, since the whole point
// here is a real physical size).
function buildPhysChunk(pixelsPerMeter: number): Buffer {
  const data = Buffer.alloc(9);
  data.writeUInt32BE(pixelsPerMeter, 0);
  data.writeUInt32BE(pixelsPerMeter, 4);
  data.writeUInt8(1, 8);
  return buildChunk('pHYs', data);
}

function insertPhysChunk(pngBuf: Buffer, pixelsPerMeter: number): Buffer {
  const SIGNATURE_AND_IHDR_LENGTH = 8 + (4 + 4 + 13 + 4); // 8-byte PNG signature + IHDR's fixed 25 bytes
  const physChunk = buildPhysChunk(pixelsPerMeter);
  return Buffer.concat([
    pngBuf.subarray(0, SIGNATURE_AND_IHDR_LENGTH),
    physChunk,
    pngBuf.subarray(SIGNATURE_AND_IHDR_LENGTH),
  ]);
}

// ── Cell -> pixel upscale ────────────────────────────────────────────────
//
// Every drawing routine below (pattern content + label) operates on a
// simple 1-pixel-per-cell RGBA buffer, exactly as before. Rather than fight
// Preview's apparent assumption that every PNG is 72dpi (see this
// session's chat -- our own correctly-written pHYs chunk wasn't reflected
// there), this makes that assumption TRUE instead: the final buffer is
// nearest-neighbor upscaled by `pxPerCell` (computed from the config's
// `dpi` field) before writing, so the image genuinely IS 72dpi (or
// whatever `dpi` is set to), and any tool that just assumes 72dpi without
// reading pHYs at all gets the correct physical size anyway. The pHYs
// chunk is still written (for tools that do read it), now simply
// confirming rather than fighting the assumed default.
function upscaleNearestNeighbor(src: Buffer, srcW: number, srcH: number, scale: number): { data: Buffer; w: number; h: number } {
  const w = srcW * scale, h = srcH * scale;
  const data = Buffer.alloc(w * h * 4);
  for (let y = 0; y < h; y++) {
    const sy = Math.floor(y / scale);
    const srcRowOffset = srcW * sy;
    const dstRowOffset = w * y;
    for (let x = 0; x < w; x++) {
      const sx = Math.floor(x / scale);
      const srcIdx = (srcRowOffset + sx) << 2;
      const dstIdx = (dstRowOffset + x) << 2;
      data[dstIdx] = src[srcIdx];
      data[dstIdx + 1] = src[srcIdx + 1];
      data[dstIdx + 2] = src[srcIdx + 2];
      data[dstIdx + 3] = src[srcIdx + 3];
    }
  }
  return { data, w, h };
}

// ── Minimal embedded 5x7 bitmap font -- just enough glyphs for an "(R,C)"
// label ('#' = ink, anything else = blank). Self-contained rather than
// pulling in a font/canvas dependency, matching this codebase's existing
// preference for small hand-rolled algorithms (see debruijn.ts's own
// mulberry32/LFSR) over new dependencies for a one-off need. ──

const GLYPH_W = 5, GLYPH_H = 7, GLYPH_SPACING = 1;

const FONT: Record<string, string[]> = {
  '0': ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
  '1': ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
  '2': ['01110', '10001', '00001', '00010', '00100', '01000', '11111'],
  '3': ['11111', '00010', '00100', '00010', '00001', '10001', '01110'],
  '4': ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
  '5': ['11111', '10000', '11110', '00001', '00001', '10001', '01110'],
  '6': ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
  '7': ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
  '8': ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
  '9': ['01110', '10001', '10001', '01111', '00001', '00010', '01100'],
  '(': ['00010', '00100', '01000', '01000', '01000', '00100', '00010'],
  ')': ['01000', '00100', '00010', '00010', '00010', '00100', '01000'],
  ',': ['00000', '00000', '00000', '00000', '00110', '00100', '01000'],
};

function textPixelWidth(text: string): number {
  return text.length * GLYPH_W + (text.length - 1) * GLYPH_SPACING;
}

// Centers `text` within `rect` (a whitespace strip in cell/pixel units) and
// blits it directly into the RGBA buffer as black ink. Throws rather than
// silently clipping/omitting the label if it doesn't actually fit -- the
// whole reason this tool computes a guaranteed label strip is so this
// should never happen; if it does, something upstream miscalculated.
function drawText(
  data: Buffer, canvasWCells: number, text: string,
  rect: { x: number; y: number; w: number; h: number },
): void {
  const textW = textPixelWidth(text);
  if (textW > rect.w || GLYPH_H > rect.h) {
    throw new Error(`Label "${text}" (${textW}x${GLYPH_H}px) doesn't fit its ${rect.w}x${rect.h}px whitespace strip`);
  }
  const startX = rect.x + Math.floor((rect.w - textW) / 2);
  const startY = rect.y + Math.floor((rect.h - GLYPH_H) / 2);
  let cx = startX;
  for (const ch of text) {
    const glyph = FONT[ch];
    if (!glyph) throw new Error(`No glyph for character ${JSON.stringify(ch)} in label "${text}"`);
    for (let gy = 0; gy < GLYPH_H; gy++) {
      for (let gx = 0; gx < GLYPH_W; gx++) {
        if (glyph[gy][gx] !== '#' && glyph[gy][gx] !== '1') continue;
        const px = cx + gx, py = startY + gy;
        const idx = (canvasWCells * py + px) << 2;
        data[idx] = data[idx + 1] = data[idx + 2] = 0;
      }
    }
    cx += GLYPH_W + GLYPH_SPACING;
  }
}

// ── Combined PDF (one full-paper-size page per tile) ────────────────────
//
// A minimal, hand-written classic (non-xref-stream) PDF -- one object each
// for the Catalog and the Pages tree, then per page: a Page object, an
// Image XObject (the tile's raw RGB pixels, zlib/FlateDecode-compressed --
// Node's built-in zlib, no PDF library needed), and a one-line content
// stream that maps that image onto the full MediaBox via the `cm` operator.
// Object numbers are predictable (no forward-reference bookkeeping needed):
// 1=Catalog, 2=Pages, then for page i (0-indexed): Page=3+3i, Image=4+3i,
// Content=5+3i. Every string is written as latin1 so byte offsets in the
// xref table line up exactly with what's on disk (matters because PDF's
// xref table is a literal byte-offset index, unlike PNG's chunk-length
// self-describing format).
interface PdfPage {
  widthPx: number; heightPx: number; rgb: Buffer; // rgb: 3 bytes/pixel, row-major, no padding
  pageWidthPt: number; pageHeightPt: number; // full physical paper size, in PDF points (1/72in)
  xPt: number; yPt: number; wPt: number; hPt: number; // where the image is placed on the page
}

function rgbaToRgb(rgba: Buffer, pixelCount: number): Buffer {
  const rgb = Buffer.alloc(pixelCount * 3);
  for (let i = 0; i < pixelCount; i++) {
    rgb[i * 3] = rgba[i * 4];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return rgb;
}

function buildPdf(pages: PdfPage[]): Buffer {
  const chunks: Buffer[] = [];
  let length = 0;
  const offsets: number[] = []; // offsets[objNum] = byte offset of "objNum 0 obj"

  function push(s: string | Buffer): void {
    const b = typeof s === 'string' ? Buffer.from(s, 'latin1') : s;
    chunks.push(b);
    length += b.length;
  }
  function beginObj(num: number): void {
    offsets[num] = length;
    push(`${num} 0 obj\n`);
  }
  function endObj(): void {
    push('\nendobj\n');
  }

  // %PDF header + the conventional 4-high-bit-byte comment line that tells
  // FTP/mail transports "this is binary, don't touch line endings."
  push('%PDF-1.4\n%\xe2\xe3\xcf\xd3\n');

  const CATALOG = 1, PAGES = 2;
  const numPages = pages.length;
  const totalObjs = 2 + numPages * 3;

  beginObj(CATALOG);
  push(`<< /Type /Catalog /Pages ${PAGES} 0 R >>`);
  endObj();

  const kids = Array.from({ length: numPages }, (_, i) => `${3 + 3 * i} 0 R`).join(' ');
  beginObj(PAGES);
  push(`<< /Type /Pages /Kids [${kids}] /Count ${numPages} >>`);
  endObj();

  for (let i = 0; i < numPages; i++) {
    const p = pages[i];
    const PAGE = 3 + 3 * i, IMAGE = 4 + 3 * i, CONTENT = 5 + 3 * i;

    beginObj(PAGE);
    push(
      `<< /Type /Page /Parent ${PAGES} 0 R /MediaBox [0 0 ${p.pageWidthPt.toFixed(3)} ${p.pageHeightPt.toFixed(3)}] ` +
      `/Resources << /XObject << /Im0 ${IMAGE} 0 R >> >> /Contents ${CONTENT} 0 R >>`,
    );
    endObj();

    const compressed = deflateSync(p.rgb);
    beginObj(IMAGE);
    push(
      `<< /Type /XObject /Subtype /Image /Width ${p.widthPx} /Height ${p.heightPx} /ColorSpace /DeviceRGB ` +
      // Explicit /Interpolate false -- the spec default, but stated
      // outright rather than relied on: this is the standard PDF hint
      // telling a renderer not to smooth/anti-alias the image when it has
      // to resample it to the actual print engine's native resolution
      // (rarely an exact multiple of our dpi), which is exactly where a
      // hard cell-edge pattern like this would otherwise blur. A hint, not
      // an enforced guarantee -- most PDF renderers and print drivers
      // honor it, but it isn't binding on every one.
      `/BitsPerComponent 8 /Filter /FlateDecode /Interpolate false /Length ${compressed.length} >>\nstream\n`,
    );
    push(compressed);
    push('\nendstream');
    endObj();

    // `cm` sets the CTM so the image's default unit square [0,1]x[0,1] maps
    // to a wPt x hPt rectangle at (xPt, yPt) -- the standard way to place a
    // raster image at an exact physical size/position on a PDF page.
    const content = `q\n${p.wPt.toFixed(3)} 0 0 ${p.hPt.toFixed(3)} ${p.xPt.toFixed(3)} ${p.yPt.toFixed(3)} cm\n/Im0 Do\nQ`;
    const contentBuf = Buffer.from(content, 'latin1');
    beginObj(CONTENT);
    push(`<< /Length ${contentBuf.length} >>\nstream\n`);
    push(contentBuf);
    push('\nendstream');
    endObj();
  }

  const xrefOffset = length;
  push(`xref\n0 ${totalObjs + 1}\n`);
  push('0000000000 65535 f\r\n'); // object 0 is always the free-list head
  for (let n = 1; n <= totalObjs; n++) {
    push(`${offsets[n].toString().padStart(10, '0')} 00000 n\r\n`);
  }
  push(`trailer\n<< /Size ${totalObjs + 1} /Root ${CATALOG} 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`);

  return Buffer.concat(chunks);
}

// ── Main ─────────────────────────────────────────────────────────────────

function main() {
  const configPath = process.argv[2];
  if (!configPath) {
    console.error('Usage: node scripts/print-debruijn-board.ts <config.json>');
    process.exit(1);
  }
  const config = loadConfig(configPath);

  const cellCm = toCm(config.cellSize, 'cellSize');
  const boardCm = toCm(config.boardSize, 'boardSize');
  const overlapCm = toCm(config.tileOverlap, 'tileOverlap');
  const paperWCm = toCm(config.paper.width, 'paper.width');
  const paperHCm = toCm(config.paper.height, 'paper.height');
  const marginCm = toCm(config.paper.margin, 'paper.margin');

  const cropCells = toWholeCells(boardCm, cellCm, 'boardSize');
  const overlapCells = toWholeCells(overlapCm, cellCm, 'tileOverlap');
  // Split symmetrically across each internal tile boundary -- see the
  // per-tile placement code further down for why. Arbitrary tie-break when
  // overlapCells is odd (one more cell going to the "after" side); doesn't
  // matter which, per this session's own "don't care left/right."
  const overlapBefore = Math.floor(overlapCells / 2);
  const overlapAfter = overlapCells - overlapBefore;

  // Pixels per cell at the target dpi -- e.g. dpi=72, cellSize=0.5cm gives
  // 72 * (0.5/2.54) = 14.17, rounded to 14px/cell. Rounding here is
  // unavoidable (72dpi rarely divides a "nice" metric cell size exactly),
  // so the effective real-world cell size is logged below for visibility
  // rather than silently accepted.
  const pxPerCell = Math.round(config.dpi * (cellCm / IN_TO_CM));
  if (pxPerCell < 1) {
    throw new Error(`dpi (${config.dpi}) and cellSize (${cellCm}cm) combine to less than 1px/cell -- increase dpi`);
  }
  const effectiveCellCm = (pxPerCell / config.dpi) * IN_TO_CM;
  if (Math.abs(effectiveCellCm - cellCm) > 1e-3) {
    console.log(
      `Note: at ${config.dpi}dpi, ${pxPerCell}px/cell actually prints as ${effectiveCellCm.toFixed(4)}cm ` +
      `(requested ${cellCm}cm) -- rounding to a whole pixel count.`,
    );
  }

  const usableWCm = paperWCm - 2 * marginCm;
  const usableHCm = paperHCm - 2 * marginCm;
  if (usableWCm <= 0 || usableHCm <= 0) {
    throw new Error(`paper.margin (${marginCm}cm on each side) leaves no usable area on a ${paperWCm}x${paperHCm}cm page`);
  }
  const usableWCells = Math.floor(usableWCm / cellCm);
  const usableHCells = Math.floor(usableHCm / cellCm);

  // Tiles must be square, so the max content size is bounded by the
  // SHORTER usable page dimension -- used ONLY to decide the minimum
  // number of tiles needed (numTiles) below, not as the actual canvas size
  // (see patternCells further down: once numTiles is fixed, the real
  // content size is usually smaller than this ceiling, and the canvas
  // must shrink to match it exactly -- otherwise leftover page capacity
  // shows up as unwanted blank space on every tile, not just the label axis).
  const shortAxisCells = Math.min(usableWCells, usableHCells);
  const longAxisCells = Math.max(usableWCells, usableHCells);
  const isPortrait = usableHCells >= usableWCells; // ties: label strip appended below, arbitrarily

  if (shortAxisCells <= overlapCells) {
    throw new Error(
      `tileOverlap (${overlapCells} cells) leaves no room for actual content on a ${shortAxisCells}-cell-wide ` +
      `usable page -- reduce tileOverlap or use larger paper / a smaller margin.`,
    );
  }
  // Overlap is split symmetrically across each internal boundary -- 1 cell
  // contributed by each of the two neighboring tiles (overlapBefore +
  // overlapAfter below), rather than one tile extending the full amount
  // into its neighbor. This still shares exactly `overlapCells` of
  // duplicated content per boundary (not double -- see the per-tile
  // placement code further down for the proof), but crucially treats both
  // ends of an axis the SAME way: an edge tile (only one real neighbor)
  // gets just its one side's worth of overlap plus a blank margin on its
  // true outward-facing side (nothing to overlap with there), instead of a
  // lopsided "first tile is full-size, last tile is undersized" split.
  //
  // `coreCellsMax` is the PAGE's ceiling on core size, used only to decide
  // the minimum number of tiles needed (numTiles) -- it is NOT the actual
  // per-tile core size. Using coreCellsMax directly as every tile's core
  // and dumping 100% of the remainder onto just the last tile would make
  // that tile potentially much smaller than the rest (e.g. 27 cells vs 39
  // here), not "equally sized squares" with an odd pixel trimmed off.
  // Instead: numTiles is fixed by the page constraint, then cropCells is
  // evenly redistributed across exactly that many tiles (ceil-divide), so
  // every tile differs from every other by at most ONE cell -- the actual
  // "odd pixel" trim, applied to however many of the outermost tiles are
  // needed to make the total add up exactly (0 tiles needed whenever
  // cropCells happens to divide evenly, as in the 144/4 example below).
  const coreCellsMax = shortAxisCells - overlapCells;

  const numTiles = Math.ceil(cropCells / coreCellsMax);
  const baseCoreCells = Math.ceil(cropCells / numTiles);
  // How many of the LAST `overflowCells` tiles (per axis) need their core
  // trimmed down by exactly 1 cell so the total exactly covers cropCells.
  // Always < numTiles by construction, and baseCoreCells <= coreCellsMax
  // (both proven in the plan's derivation), so this can never overflow the
  // page's own physical capacity.
  const overflowCells = baseCoreCells * numTiles - cropCells;
  const coreSizes: number[] = [];
  for (let i = 0; i < numTiles; i++) {
    coreSizes.push(i >= numTiles - overflowCells ? baseCoreCells - 1 : baseCoreCells);
  }
  const coreStarts: number[] = [];
  { let acc = 0; for (const size of coreSizes) { coreStarts.push(acc); acc += size; } }

  // The actual max content size any tile needs -- a full (non-trimmed)
  // interior tile's core plus the full overlap split across both its
  // sides. This is <=
  // coreCellsMax (usually strictly less, whenever fewer core cells than
  // the page's raw ceiling were actually needed to hit numTiles), and is
  // the real, tight canvas width -- NOT shortAxisCells/coreCellsMax, which
  // are just the page's capacity ceiling used to derive numTiles above.
  // Using the page ceiling here instead would leave every tile with
  // pointless blank space on its content axis, not just on the label axis.
  const patternCells = baseCoreCells + overlapCells;

  const MIN_LABEL_STRIP_CELLS = GLYPH_H + 2; // glyph height + 1px padding top and bottom
  const labelStripCells = longAxisCells - patternCells;
  if (labelStripCells < MIN_LABEL_STRIP_CELLS) {
    throw new Error(
      `Not enough leftover page space for the (R,C) label: the paper's longer usable axis (${longAxisCells} ` +
      `cells) is only ${labelStripCells} cells bigger than the actual tile content size (${patternCells} cells) ` +
      `(need at least ${MIN_LABEL_STRIP_CELLS}). Try a paper size that isn't nearly square, a larger paper size, ` +
      `or a smaller margin.`,
    );
  }

  console.log(
    `Board: ${cropCells}x${cropCells} cells (${boardCm}cm at ${cellCm}cm/cell). Page allows up to ${coreCellsMax} ` +
    `core cells -> ${numTiles}x${numTiles} = ${numTiles * numTiles} pages, core sizes [${coreSizes.join(', ')}] ` +
    `cells, overlap ${overlapCells} cells, tile content ${patternCells} cells wide.`,
  );

  // Rebuild the exact pattern (throws if `taps` isn't actually a valid
  // maximal-length LFSR for `order`). Also confirm the requested crop
  // doesn't wrap past the full torus -- buildTorusFromCandidate itself
  // wraps silently via modulo, which would reuse rows/cols from the far
  // side of the torus and can break the window-uniqueness guarantee.
  const { order, taps, cropOrigin } = config.debruijn;
  const N = order * order;
  const { R: fullR, C: fullC } = bestCoprimeSplit(2 ** N - 1);
  if (cropOrigin.row + cropCells > fullR || cropOrigin.col + cropCells > fullC) {
    throw new Error(
      `Crop (row=${cropOrigin.row}, col=${cropOrigin.col}, size=${cropCells}) would wrap past the full ` +
      `${fullR}x${fullC} torus -- pick a smaller boardSize/cellSize ratio or a different cropOrigin.`,
    );
  }
  const { torus } = buildTorusFromCandidate(order, {
    taps, r0: cropOrigin.row, c0: cropOrigin.col, cropSize: cropCells,
  });

  // Canvas dimensions -- identical for every tile ("equally sized
  // squares"): the pattern area is always patternCells x patternCells (the
  // tight content size, not the page's raw capacity ceiling), plus the
  // fixed label strip appended on whichever physical axis is longer.
  const canvasWCells = isPortrait ? patternCells : patternCells + labelStripCells;
  const canvasHCells = isPortrait ? patternCells + labelStripCells : patternCells;
  const stripRect = isPortrait
    ? { x: 0, y: patternCells, w: canvasWCells, h: labelStripCells }
    : { x: patternCells, y: 0, w: labelStripCells, h: canvasHCells };
  // Pixels-per-meter at the TRUE post-upscale resolution (dpi / 0.0254) --
  // for dpi=72 this rounds to 2835, the same value most encoders default
  // to for "72dpi", i.e. this now agrees with what tools already assume
  // rather than contradicting it.
  const pixelsPerMeter = Math.round(config.dpi / 0.0254);

  // PDF page geometry: full physical paper size per page (not just the
  // tile image's own bounds) -- per this session's decision, this is safer
  // for real printing, since most print dialogs won't try to auto-scale-
  // to-fit a page that already matches the selected physical media, while
  // a custom/odd page size risks a print driver silently rescaling it and
  // defeating the whole point of preserving physical scale. The tile image
  // itself is centered on the page (see below).
  const PT_PER_IN = 72;
  const pageWidthPt = (paperWCm / IN_TO_CM) * PT_PER_IN;
  const pageHeightPt = (paperHCm / IN_TO_CM) * PT_PER_IN;
  const pdfPages: PdfPage[] = [];

  mkdirSync(config.outputDir, { recursive: true });

  for (let tr = 0; tr < numTiles; tr++) {
    for (let tc = 0; tc < numTiles; tc++) {
      const coreSizeRow = coreSizes[tr];
      const coreSizeCol = coreSizes[tc];
      const hasPrevRow = tr > 0, hasNextRow = tr < numTiles - 1;
      const hasPrevCol = tc > 0, hasNextCol = tc < numTiles - 1;
      const contentRows = coreSizeRow + (hasPrevRow ? overlapBefore : 0) + (hasNextRow ? overlapAfter : 0);
      const contentCols = coreSizeCol + (hasPrevCol ? overlapBefore : 0) + (hasNextCol ? overlapAfter : 0);
      const boardRowStart = coreStarts[tr] - (hasPrevRow ? overlapBefore : 0);
      const boardColStart = coreStarts[tc] - (hasPrevCol ? overlapBefore : 0);
      // Where within the patternCells x patternCells pattern area this
      // tile's content is drawn. Anchored at 0 (its own overlap-with-
      // previous cells landing at the low end) whenever there IS a
      // previous neighbor -- true for every interior tile (fills the whole
      // area, no gap) and the LAST tile (whose only neighbor is behind
      // it). Pushed to the high end (patternCells - contentSize) only for
      // the FIRST tile (no previous neighbor) so the resulting blank gap
      // falls on its true outward-facing (low) side instead of wherever
      // the anchor happens to land.
      const rowAnchor = hasPrevRow ? 0 : (patternCells - contentRows);
      const colAnchor = hasPrevCol ? 0 : (patternCells - contentCols);

      // Built at 1px/cell first (same as every drawing routine above
      // expects), then upscaled to the true dpi-driven resolution below --
      // keeps the pattern/label drawing code simple and cell-addressed.
      const cellBuf = Buffer.alloc(canvasWCells * canvasHCells * 4);
      cellBuf.fill(255); // white, opaque background; content/label drawn on top below

      for (let i = 0; i < contentRows; i++) {
        const boardRow = boardRowStart + i;
        const srcRow = torus[boardRow];
        const rowOffset = canvasWCells * (rowAnchor + i);
        for (let j = 0; j < contentCols; j++) {
          const boardCol = boardColStart + j;
          const shade = srcRow[boardCol] ? 0 : 255; // 1 -> black, 0 -> white, matching scene/floor.ts's convention
          const idx = (rowOffset + colAnchor + j) << 2;
          cellBuf[idx] = cellBuf[idx + 1] = cellBuf[idx + 2] = shade;
        }
      }

      drawText(cellBuf, canvasWCells, `(${tr},${tc})`, stripRect);

      const { data: pixelData, w: pixelW, h: pixelH } = upscaleNearestNeighbor(cellBuf, canvasWCells, canvasHCells, pxPerCell);
      const png = new PNG({ width: pixelW, height: pixelH });
      pixelData.copy(png.data);

      const rawBuf = PNG.sync.write(png);
      const finalBuf = insertPhysChunk(rawBuf, pixelsPerMeter);
      writeFileSync(join(config.outputDir, `(${tr},${tc}).png`), finalBuf);

      const imageWidthPt = (pixelW / config.dpi) * PT_PER_IN;
      const imageHeightPt = (pixelH / config.dpi) * PT_PER_IN;
      // Centered on the full page (not just margin-anchored) -- the image
      // is already sized to fit within the usable/margin area (see
      // patternCells/canvasHCells above), so centering never pushes it
      // outside the margin, it just balances the leftover slack evenly on
      // all four sides instead of dumping it all to the right/bottom.
      pdfPages.push({
        widthPx: pixelW, heightPx: pixelH, rgb: rgbaToRgb(pixelData, pixelW * pixelH),
        pageWidthPt, pageHeightPt,
        xPt: (pageWidthPt - imageWidthPt) / 2, yPt: (pageHeightPt - imageHeightPt) / 2,
        wPt: imageWidthPt, hPt: imageHeightPt,
      });
    }
  }

  const pdfPath = join(config.outputDir, 'board.pdf');
  writeFileSync(pdfPath, buildPdf(pdfPages));

  console.log(
    `Wrote ${numTiles * numTiles} tiles to ${config.outputDir}/ ` +
    `(${canvasWCells * pxPerCell}x${canvasHCells * pxPerCell}px each at ${config.dpi}dpi, ${pxPerCell}px/cell, ` +
    `${pixelsPerMeter}px/m), plus a combined ${pdfPath} (${numTiles * numTiles} pages, ` +
    `${(pageWidthPt / PT_PER_IN).toFixed(2)}x${(pageHeightPt / PT_PER_IN).toFixed(2)}in each).`,
  );
}

main();
