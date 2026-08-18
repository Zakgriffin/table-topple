import { type Board, createBoard } from '../../pose/board.ts';
import { PRINTED_PATTERN_CELLS } from './constants.ts';

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
// ── THE PRINTED CELL COUNT IS ITS OWN NUMBER, NOT THE GAME BOARD'S ──
//
// `cropSize` is `PRINTED_PATTERN_CELLS` (constants.ts), not `BOARD_CELLS` --
// the physical printed surface and the game's own playing-field grid are
// deliberately allowed to disagree (constants.ts's own comment on
// BOARD_CELLS explains why: 140 divides evenly into 35 game tiles, a rounder
// number than the printed pattern's 144 would give). `overlay.ts`'s
// fitBoardToPattern is what reconciles the two on the AR page, scaling the
// whole game world so it fills the physical board exactly regardless of
// whether the two counts happen to match.
//
// The searched crop's own default is also 144 (see `debruijn.ts`'s
// ORDER5_CANDIDATE), so this agrees with the library's default rather than
// fighting it -- but it is written out because it is a Table Topple decision,
// not something to inherit silently.
export const board: Board = createBoard({ order: 5, cropSize: PRINTED_PATTERN_CELLS });

// World units per pattern cell. The pose pipeline is scale-invariant and talks
// in cells, so this is purely a property of how this project draws its world --
// and at 1 it makes world coordinates and cell coordinates the same numbers,
// which is what lets `overlay.ts` scale the scene by a plain ratio.
export const CELL_PITCH = 1;
