import { ORDER5_CANDIDATE } from '../pose/debruijn.ts';
import { type Board, createBoard } from '../pose/board.ts';
import { GRID_STEP } from './constants.ts';

// ── This app's board, and where it sits in the world ─────────────────────
//
// The De Bruijn pattern itself is the pose library's (see pose/board.ts): it
// owns the construction, because a second copy could disagree with the printed
// floor and nothing downstream could detect that. What lives HERE is the two
// things that are this app's own choices -- WHICH board it wants, and how big a
// cell is in world units.
//
// ── IT USED TO BE AMBIENT MUTABLE STATE, AND THAT WAS THE PROBLEM ──
//
// This module used to export `let debruijn, R, C, torus, debruijnLookup` and
// rewrite all five in place from `rebuildFloorPatternData`. Two consequences,
// and the second is why it had to go:
//
//  1. The pose library IMPORTED THIS FILE -- so the library depended on one of
//     the apps, and any other app importing the library transitively imported
//     Sphere Lab.
//  2. There was exactly ONE BOARD PER PROCESS. Two apps could not each pick
//     their own De Bruijn parameters, because the parameters were not passed
//     anywhere -- they were read off module scope by whoever needed them.
//
// So the board is now CONSTRUCTED and handed to the pipeline. It is still a
// module-level singleton, which is fine: this is one app, and one app has one
// floor. What changed is that it is a single binding swapped WHOLESALE rather
// than five bindings mutated in place, so a rebuild cannot be observed
// half-applied.
//
// **Read `board.R`, never destructure at module scope.** `board` is a live
// binding and a rebuild replaces it; a `const { R } = board` at the top of some
// other module snapshots the old one and silently decodes against a board that
// is no longer on the floor.
//
// No THREE and no DOM here, which is what keeps this importable on a page with
// no #gl canvas -- scene/floor.ts derives its own texture/mesh/grid-line
// geometry from this data rather than owning it.

// The `?order=` override is a BROWSER affordance, so it is read only when there
// is a browser. This module has to stay importable with no DOM (headless tests
// import math/geometry.ts, which reads HALF_C/HALF_R from here), and an
// unguarded `location` at module scope broke that for every importer.
//
// It is deliberately app-side: the library takes an `order` as a parameter and
// has no opinion about URLs. That guard existing at all was the tell that the
// read was in the wrong layer.
export const ORDER = parseInt(
  (typeof location === 'undefined' ? null : new URLSearchParams(location.search).get('order')) ?? '5',
  10,
);

// The STARTING size only. config.global.boardSize is the real source of truth
// (see sphere-lab.config.json), but config is FETCHED rather than imported, so
// it is not available at module init -- the board is rebuilt at the configured
// size once it lands. ORDER5_CANDIDATE's own cropSize is the searched crop's
// natural size and so the right thing to start from.
export let board: Board = createBoard({ order: ORDER, cropSize: ORDER5_CANDIDATE.cropSize });

// One instance of the torus, sized in world units at GRID_STEP per cell —
// NOT tiled. Half-extents, since grid lines/great circles are indexed out from
// the pattern's center.
//
// These are the reason GRID_STEP is an APP concern: the pipeline is
// scale-invariant and talks in cells throughout, so nothing in the library
// needs a length. Only the things that DRAW a floor do.
export let HALF_C = (board.C * GRID_STEP) / 2;
export let HALF_R = (board.R * GRID_STEP) / 2;

/**
 * Swaps in a board of a different size.
 *
 * Only meaningful for ORDER === 5, the searched-crop path -- every other order
 * has no "size" concept, it is always the one full R x C torus for that order.
 *
 * scene/floor.ts's rebuildFloorPattern calls this first, then does its own
 * THREE-side rebuild (texture/mesh/grid-line geometry) on top. Anything holding
 * a pose pipeline built against the old board has to rebuild that too: the
 * board's dimensions are baked into a PoseContext's buffer sizes.
 */
export function rebuildFloorPatternData(size: number) {
  if (ORDER !== 5) return;
  board = createBoard({ order: ORDER, cropSize: size });
  HALF_C = (board.C * GRID_STEP) / 2;
  HALF_R = (board.R * GRID_STEP) / 2;
}
