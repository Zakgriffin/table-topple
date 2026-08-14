import { C, R, debruijnLookup, torus } from '../sphereLab/floorPattern.ts';
import type { Dims } from './pipeline.ts';

// ── The printed board, as device buffers ──────────────────────────────────
//
// Decode needs two things the rest of the pipeline does not: the De Bruijn
// pattern itself (to score consistency against) and a lookup from a 25-bit
// window key to a torus position (to turn a window into a vote). Both are
// constants of the physical floor, built once per device and kept.
//
// ── OPEN DECISION 5b, SETTLED: sphereLab/floorPattern IS A SHARED LEAF ──
//
// The question was whether src/pose2 should own the pattern or import it. It
// imports it, and the argument is that this is DATA about the world rather than
// an implementation of anything:
//
//  - Rule 2 deletes a second implementation of the ALGORITHM. The torus is not
//    an algorithm; there is no CPU-versus-GPU version of it to choose between.
//  - A duplicated copy could DISAGREE WITH THE PRINTED FLOOR, and nothing in
//    this pipeline could detect that. Every consistency number would still look
//    fine -- decode would simply report a confident, wrong position. That is a
//    strictly worse failure than the coupling it avoids.
//  - sim.ts and sweep.ts already import it, and the simulator has to render the
//    same board the decoder decodes or the sweep is scoring nothing.
//
// What is NOT imported is `src/pose`'s hash-table builder, even though one
// exists and works. That module is the thing this rewrite replaces and is
// deleted in Phase 3, so importing it would make pose2 un-shippable without it.
// The table is re-derived below; the HASH FUNCTION is copied verbatim, which is
// the one thing here that is a correctness constant rather than a structure.
//
// This file is the only place src/pose2 reaches outside itself, and it stays
// separate from pose.ts so the pipeline proper keeps importing nothing.

const NOT_FOUND = 0xffffffff;

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

/** The board this pipeline decodes, as the dimensions the declaration wants. */
export function boardDims(): Pick<Dims, 'torusR' | 'torusC' | 'maxCells' | 'hashSlots'> {
  return {
    torusR: R,
    torusC: C,
    // DERIVED, not chosen (§12): decode.layout clamps each lattice edge to one
    // board period, so this is a bound rather than a guess.
    maxCells: R * C,
    hashSlots: hashSlotsFor(debruijnLookup.size),
  };
}

export interface BoardData {
  /** [R * C], 0 or 1, row-major. A set bit is a DARK cell. */
  torus: Uint32Array;
  hashKeys: Uint32Array;
  /** NOT_FOUND in an empty slot, else the packed `row * C + col`. */
  hashValues: Uint32Array;
}

export function buildBoard(dims: Pick<Dims, 'torusR' | 'torusC' | 'hashSlots'>): BoardData {
  if (dims.torusR !== R || dims.torusC !== C) {
    throw new Error(`board is ${R}x${C} but the pipeline is configured for ${dims.torusR}x${dims.torusC}`);
  }
  const flat = new Uint32Array(R * C);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) flat[r * C + c] = torus[r]![c]!;

  const size = dims.hashSlots;
  if (size < debruijnLookup.size * 2) {
    throw new Error(
      `hashSlots ${size} cannot hold ${debruijnLookup.size} entries at load factor 0.5 ` +
      `-- a full open-addressing table makes the insert probe below non-terminating`);
  }
  const hashKeys = new Uint32Array(size);
  const hashValues = new Uint32Array(size).fill(NOT_FOUND);
  for (const [key, value] of debruijnLookup) {
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
