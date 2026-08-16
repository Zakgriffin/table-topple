import { type Board, createBoard } from '../../src/pose2/board.ts';
import { ORDER5_CANDIDATE } from '../../src/pose2/debruijn.ts';
import type { SimWorld } from '../../src/pose2/sim.ts';

// ── The board the library's own tests decode against ─────────────────────
//
// Constructed here rather than imported from `sphereLab/floorPattern.ts`, which
// is where every one of these tests used to get it. That import was the same
// layering inversion the library itself had: a test of the pose pipeline should
// not need one of the apps to exist, let alone that app's ambient board.
//
// These are the values floorPattern produced before it was rewritten -- order
// 5, the searched candidate's own crop size -- so every fixture, oracle and
// golden in the suite is decoding against a byte-identical torus.
//
// `tests/decodeLattice.test.ts` deliberately does NOT use this: it tests Sphere
// Lab's own host-side lattice against the device, so it has to read the board
// the app's `buildDecodeLattice` reads.
export const TEST_BOARD: Board = createBoard({ order: 5, cropSize: ORDER5_CANDIDATE.cropSize });

// 1 world unit per cell, which is GRID_STEP's value. Written as a literal
// rather than imported because the pipeline is scale-invariant and a length is
// an app concern -- see pose2/board.ts. Any positive value would do; this one
// keeps world coordinates and cell coordinates numerically equal, which is what
// makes the ground-truth arithmetic in these tests readable.
export const TEST_CELL_PITCH = 1;

/** What the simulator renders: this board, at this pitch. */
export const TEST_WORLD: SimWorld = { board: TEST_BOARD, cellPitch: TEST_CELL_PITCH };
