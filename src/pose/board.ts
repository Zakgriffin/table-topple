import { ORDER5_CANDIDATE, buildLookupTableSparse, buildTorusFromCandidate, generateTorus } from './debruijn.ts';
import type { Dims } from './pipeline.ts';

// ── The printed board ─────────────────────────────────────────────────────
//
// Decode needs two things the rest of the pipeline does not: the De Bruijn
// pattern itself (to score consistency against) and a lookup from a 25-bit
// window key to a torus position (to turn a window into a vote). Both are
// constants of the physical floor, built once and kept.
//
// ── THE LIBRARY OWNS THE PATTERN, AND CALLERS CONSTRUCT ONE ──
//
// This used to import `poseViewer/floorPattern.ts`, which made the pose library
// depend on ONE OF THE APPS -- and worse, on ambient mutable module state
// (`export let R, C, torus, ...`, rewritten in place by a rebuild function).
// That shape allowed exactly ONE BOARD PER PROCESS, which is why two apps could
// not each pick their own De Bruijn parameters.
//
// So the data lives here and is CONSTRUCTED: `createBoard(params)` returns a
// `Board`, and every entry point takes one. Two callers wanting different board
// sizes is now an ordinary case rather than an impossible one, and changing a
// board size is "build a new one and swap it" rather than a mutation nothing
// downstream can observe.
//
// The library still OWNS the pattern rather than accepting one, and that part
// was never in question -- this is DATA about the world rather than an
// implementation of anything:
//
//  - Rule 2 deletes a second implementation of the ALGORITHM. The torus is not
//    an algorithm; there is no CPU-versus-GPU version of it to choose between.
//  - A duplicated copy could DISAGREE WITH THE PRINTED FLOOR, and nothing in
//    this pipeline could detect that. Every consistency number would still look
//    fine -- decode would simply report a confident, wrong position. That is a
//    strictly worse failure than the coupling it avoids.
//  - sim.ts renders the board the decoder decodes, so a second copy would make
//    the sweep score nothing.
//
// What the caller supplies is PARAMETERS (order, board size), not data. What
// stays app-side is anything with a LENGTH in it: `cellPitch` is a per-call
// setting on the reconstruction, and world-space half-extents belong to whoever
// is drawing a floor. The pipeline is deliberately scale-invariant and talks in
// cells throughout.
//
// What is NOT here is the old pipeline's hash-table builder, even though one existed
// and worked. That module was the thing this rewrite replaced, so importing it
// would have made pose un-shippable without it. The table is re-derived below;
// the HASH FUNCTION is copied verbatim, which is the one thing here that is a
// correctness constant rather than a structure.

const NOT_FOUND = 0xffffffff;

/**
 * What an app picks. `cropSize` is the board's edge in cells; `order` is the
 * De Bruijn window size, and the pattern is built around it.
 */
export interface BoardParams {
  order: number;
  cropSize: number;
}

/**
 * One printed board. Immutable by convention -- to change the size, build
 * another and swap it, which is what makes a board-size control safe (the
 * predecessor mutated seven module-level bindings in place and every reader
 * silently changed underneath).
 */
export interface Board {
  order: number;
  R: number;
  C: number;
  /** [R][C], 0 or 1. A set bit is a DARK cell. */
  torus: Uint8Array[];
  /** order x order window key -> packed `row * C + col`. */
  lookup: Map<number, number>;
}

// Below this many cells a board starts hitting degenerate cases -- a zero-size
// tally buffer here, an empty canvas in whatever draws the floor. Clamped
// rather than rejected so a size control's low end needs no special case.
const MIN_BOARD_SIZE = 8;

/**
 * Builds the board an app asked for.
 *
 * Order 5's full R x C torus (~33.5M cells) has no known efficient construction
 * free of D4 rotation/reflection collisions, so it is not used directly --
 * ORDER5_CANDIDATE is a searched 256x256 sub-region with a low (1.027%)
 * residual collision rate, and `cropSize` re-crops it. Only order 5 has a size
 * at all; every other order is the one full torus for that order, so `cropSize`
 * is ignored there.
 *
 * The taps/r0/c0 search that found this particular low-collision crop was NOT
 * re-verified at every size, so the collision rate away from 256 is unmeasured
 * (not expected to be dramatically worse, just untested).
 */
export function createBoard({ order, cropSize }: BoardParams): Board {
  const debruijn = order === 5
    ? buildTorusFromCandidate(5, { ...ORDER5_CANDIDATE, cropSize: Math.max(MIN_BOARD_SIZE, Math.round(cropSize)) })
    : generateTorus(order);
  return {
    order,
    R: debruijn.R,
    C: debruijn.C,
    torus: debruijn.torus,
    lookup: buildLookupTableSparse(debruijn),
  };
}

/**
 * murmur3's fmix32, and it MUST stay byte-identical to DECODE_TALLY_WGSL's
 * `hashU32`. `Math.imul` wraps mod 2^32 exactly as a WGSL u32 multiply does by
 * spec, which is what makes a table built here probeable there.
 *
 * A drift between the two does not fail loudly: every lookup simply misses, the
 * histogram stays empty, and the frame reports `decodeNoAnchor` -- which is an
 * ORDINARY outcome, indistinguishable from a frame with no board in it. Hence
 * the round-trip test.
 */
export function hashU32(xIn: number): number {
  let x = xIn >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

/**
 * Open addressing at load factor 0.5, rounded up to a power of two so the probe
 * can mask rather than divide.
 *
 * Sized from the entry count rather than chosen, which is why it is exported:
 * `Dims.hashSlots` has to agree with it, and a table that is too small does not
 * overflow -- it fills, and the insert loop below never terminates.
 */
export function hashSlotsFor(entries: number): number {
  let p = 1;
  while (p < entries * 2) p *= 2;
  return p;
}

/** A board, as the dimensions the declaration wants. */
export function boardDims(board: Board): Pick<Dims, 'torusR' | 'torusC' | 'maxCells' | 'hashSlots'> {
  return {
    torusR: board.R,
    torusC: board.C,
    // DERIVED, not chosen (§12): decode.layout clamps each lattice edge to one
    // board period, so this is a bound rather than a guess.
    maxCells: board.R * board.C,
    hashSlots: hashSlotsFor(board.lookup.size),
  };
}

export interface BoardData {
  /** [R * C], 0 or 1, row-major. A set bit is a DARK cell. */
  torus: Uint32Array;
  hashKeys: Uint32Array;
  /** NOT_FOUND in an empty slot, else the packed `row * C + col`. */
  hashValues: Uint32Array;
}

export function buildBoard(board: Board, dims: Pick<Dims, 'torusR' | 'torusC' | 'hashSlots'>): BoardData {
  const { R, C, torus, lookup } = board;
  if (dims.torusR !== R || dims.torusC !== C) {
    throw new Error(`board is ${R}x${C} but the pipeline is configured for ${dims.torusR}x${dims.torusC}`);
  }
  const flat = new Uint32Array(R * C);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) flat[r * C + c] = torus[r]![c]!;

  const size = dims.hashSlots;
  if (size < lookup.size * 2) {
    throw new Error(
      `hashSlots ${size} cannot hold ${lookup.size} entries at load factor 0.5 ` +
      `-- a full open-addressing table makes the insert probe below non-terminating`);
  }
  const hashKeys = new Uint32Array(size);
  const hashValues = new Uint32Array(size).fill(NOT_FOUND);
  for (const [key, value] of lookup) {
    let slot = hashU32(key) % size;
    while (hashValues[slot] !== NOT_FOUND) slot = (slot + 1) % size;
    hashKeys[slot] = key;
    hashValues[slot] = value;
  }
  return { torus: flat, hashKeys, hashValues };
}

/**
 * The three persistent buffers, uploaded once. Persistent is a `kind` in the
 * declaration precisely so `planPool` never pools them -- they are the only
 * buffers here whose contents outlive a frame.
 */
export function uploadBoard(device: GPUDevice, bufs: Record<string, GPUBuffer>, board: BoardData): void {
  device.queue.writeBuffer(bufs.torus!, 0, board.torus);
  device.queue.writeBuffer(bufs.hashKeys!, 0, board.hashKeys);
  device.queue.writeBuffer(bufs.hashValues!, 0, board.hashValues);
}
