import { type Board, createBoard } from '../../pose/board.ts';
import { BOARD_CELLS } from './constants.ts';

// ── Table Topple's printed board ─────────────────────────────────────────
//
// The De Bruijn pattern this project's clients reconstruct their pose against.
// It is CONSTRUCTED here, from this project's own numbers, which is the whole
// point of the isolation rule: `overlay.ts` used to import Pose Viewer's
// `floorPattern.ts` and its `GRID_STEP`, so Table Topple silently inherited
// whatever board size that app's slider happened to be on.
//
// Note the deliberate name: `board.ts` in this directory is the GAME's board
// (terrain, landmarks, the playing field). This is the printed pattern the
// camera decodes. They are the same physical surface and two different things.
//
// ── ONE CELL IS ONE GAME CELL, AND THAT IS THE WHOLE CHOICE ──
//
// `cropSize` is `BOARD_CELLS`, so one De Bruijn cell IS one board-game cell and
// a soldier is exactly one cell tall. The AR overlay then lands on the physical
// board 1:1 rather than at some scale factor, and `overlay.ts`'s fit is the
// identity in the ordinary case.
//
// The searched crop's own default is also 144 (see `debruijn.ts`'s
// ORDER5_CANDIDATE), so this agrees with the library's default rather than
// fighting it -- but it is written out because it is a Table Topple decision,
// not something to inherit silently.
export const board: Board = createBoard({ order: 5, cropSize: BOARD_CELLS });

// World units per pattern cell. The pose pipeline is scale-invariant and talks
// in cells, so this is purely a property of how this project draws its world --
// and at 1 it makes world coordinates and cell coordinates the same numbers,
// which is what lets `overlay.ts` scale the scene by a plain ratio.
export const CELL_PITCH = 1;
