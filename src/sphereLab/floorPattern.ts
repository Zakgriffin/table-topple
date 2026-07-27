import { ORDER5_CANDIDATE, buildLookupTableSparse, buildTorusFromCandidate, generateTorus } from '../debruijn.ts';
import { GRID_STEP } from './constants.ts';

// ── Pure De Bruijn floor-pattern data ────────────────────────────────────
//
// Extracted from scene/floor.ts (scene/floor.ts:10-31, 130-137) so it can be
// imported without pulling in THREE-scene construction -- no THREE/DOM
// imports here at all. scene/floor.ts re-derives its own THREE-side state
// (texture/mesh/grid-line geometry) from this module instead of owning the
// data itself; everything else that only ever needed the DATA (math/
// geometry.ts's HALF_C/HALF_R, decodeGrid.ts's/decodeTally.ts's/
// positionLM.ts's ORDER/R/C/torus/debruijnLookup) now imports it straight
// from here, which is what actually makes those modules safe to import on a
// page with no #gl canvas (e.g. mobile-capture.html) -- see this session's
// on-device-pose-recovery plan.
export const ORDER = parseInt(new URLSearchParams(location.search).get('order') ?? '5', 10);
// Order 5's full R x C torus (~33.5M cells) has no known efficient
// construction free of D4 rotation/reflection collisions, so it isn't used
// directly -- ORDER5_CANDIDATE is a searched 256x256 sub-region with a low
// (1.027%) residual collision rate instead (see buildTorusFromCandidate's
// header comment in debruijn.ts). cropSize itself is now just the STARTING
// board size -- see globalState.boardSize / rebuildFloorPatternData below,
// which is what actually drives it from here on (the "board size" slider in
// the global settings section). Only cropSize is runtime-adjustable; the
// taps/r0/c0 search that found this particular low-collision crop was NOT
// re-verified at every possible size, so collision rate at sizes other than
// 256 is unmeasured (not expected to be dramatically worse, just untested).
export let debruijn = ORDER === 5 ? buildTorusFromCandidate(5, ORDER5_CANDIDATE) : generateTorus(ORDER);
export let R = debruijn.R, C = debruijn.C, torus = debruijn.torus;
// For decoding an ORDER x ORDER sampled bit window back into an absolute
// torus (row,col) position -- see runPositionDecode.
export let debruijnLookup = buildLookupTableSparse(debruijn);
// One instance of the torus, sized in world units at GRID_STEP per cell —
// NOT tiled. Half-extents, since grid lines/great circles below are indexed
// out from the origin at the pattern's center.
export let HALF_C = (C * GRID_STEP) / 2;
export let HALF_R = (R * GRID_STEP) / 2;
// Floor below this many cells starts hitting degenerate cases (an empty
// canvas for rebuildFloorTexture, a zero-size GPU tally buffer, etc) --
// clamps the board-size slider's low end without needing to special-case
// every consumer for a board nobody would actually want to decode against.
const MIN_BOARD_SIZE = 8;

// Rebuilds just the DATA half of the board at a new size (the non-THREE
// lines of what used to be scene/floor.ts's own rebuildFloorPattern) --
// re-crops the torus, rebuilds the decode lookup table, and recomputes
// HALF_R/HALF_C. scene/floor.ts's rebuildFloorPattern calls this first, then
// does its own THREE-side rebuild (texture/mesh/grid-line geometry) on top.
// Only meaningful for ORDER === 5 (the searched-crop path) -- the other
// orders' generateTorus has no "size" concept, it's always the one full
// R x C torus for that order, so this is a no-op there.
export function rebuildFloorPatternData(size: number) {
  if (ORDER !== 5) return;
  const cropSize = Math.max(MIN_BOARD_SIZE, Math.round(size));
  debruijn = buildTorusFromCandidate(5, { ...ORDER5_CANDIDATE, cropSize });
  R = debruijn.R; C = debruijn.C; torus = debruijn.torus;
  debruijnLookup = buildLookupTableSparse(debruijn);
  HALF_C = (C * GRID_STEP) / 2;
  HALF_R = (R * GRID_STEP) / 2;
}
