// Board dimensions, in world units. One world unit == one board cell, so a
// piece's integer (col, row) doubles as its world (x, z) -- worth keeping
// true as pieces/movement land, since it makes every board<->world
// conversion a floor() instead of a scale factor nobody remembers.
import type { Rank } from './ranks.ts';

export const BOARD_CELLS = 144;
export const BOARD_SIZE = BOARD_CELLS;

// Starting layout: one court per edge, each with their castle behind them.
// Here rather than in denizens.ts so the whole board layout can be checked
// against BOARD_SIZE without loading anything that touches the DOM.
//
// Note that denizens, their speed, and their castles are all in absolute world
// units and do NOT scale with the board -- growing the board buys open ground
// between the castles, nothing else. Crossing the full board at WALK_SPEED
// takes about BOARD_SIZE/WALK_SPEED = 36s.
/** Space left between a court's line and the board edge behind it. Sized to
 *  hold that court's castle: the setback denizens.ts uses is one castle
 *  half-extent plus CASTLE_GAP, and the castle then reaches a further half
 *  extent back, which comes to ~11.65 for the default square plan. 12 puts the
 *  back wall a third of a unit off the edge. Not imported from castle.ts on
 *  purpose -- castle.ts reads SOLDIER_HEIGHT from here, and taking the
 *  dependency the other way would close the cycle. */
export const COURT_EDGE_MARGIN = 12;
/** How far a court's line forms up from the board's center at the start.
 *  Derived, so the courts stay backed up against their own edge at any board
 *  size instead of huddling in the middle of a bigger one. */
export const START_RADIUS = BOARD_SIZE / 2 - COURT_EDGE_MARGIN;
/** Gap between the line's back and their castle's front wall. */
export const CASTLE_GAP = 1.5;

/** The line each court forms up in, read left to right. Symmetric about the
 *  king, with rank descending outward from him -- so the formation itself
 *  shows the hierarchy, and the tallest figure stands dead center. */
export const FORMATION: Rank[] = [
  'soldier', 'soldier', 'captain', 'constable',
  'king',
  'constable', 'captain', 'soldier', 'soldier',
];

/** Spacing between neighbours in the line. Comfortably wider than a king's
 *  shoulders (the widest figure) so nobody overlaps. */
export const RANK_SPACING = 2.2;

// Palette lifted from Pose Viewer (scene/renderer.ts's 0x0a0a0f clear color,
// floor.ts's red/blue row/column line families) so the two pages read as one
// project. Nothing here is shared code -- this page is deliberately isolated
// from the pose pipeline -- just the same colors.
export const COLOR_BG = 0x0a0a0f;

// Team colors, one per court. Red and blue are Pose Viewer's row/column line
// families exactly (floor.ts's 0xff5555 / 0x5599ff); green and yellow are
// chosen to sit at the same weight rather than at their pure hues -- a raw
// 0x00ff00 or 0xffff00 is far brighter than either original and would make
// those two courts dominate the board. Each denizen's own color is its team's,
// deepened by its rank (ranks.ts). You are the red king (denizens.ts).
export const COLOR_TEAM_RED = 0xff5555;
export const COLOR_TEAM_BLUE = 0x5599ff;
export const COLOR_TEAM_GREEN = 0x5ecf6a;
export const COLOR_TEAM_YELLOW = 0xf0c74e;

// Terrain (terrain.ts). These replace the old single COLOR_FLOOR -- the board
// surface is now forest everywhere except the patches.
//
// All four are deliberately dark and low-saturation, for two reasons. The
// board is the largest thing on screen, so a colour that would be pleasant on
// a small swatch dominates everything standing on it; and the courts are
// identified BY colour, so a floor that competes with them makes a green
// denizen on a field, or a yellow one on hallowed ground, hard to pick out.
// Compare COLOR_TEAM_GREEN (0x5ecf6a) with COLOR_FIELD below: same hue family,
// much less light. The floor is the ground the game is read against, not a
// participant in it.
export const COLOR_FOREST = 0x142317;
export const COLOR_FIELD = 0x4a7a43;
export const COLOR_CAVE = 0x44454f;
export const COLOR_HALLOWED = 0x8e7a35;

// Height of a SOLDIER, feet to top of head -- the base every other rank is a
// multiple of (ranks.ts), and the unit castle dimensions are quoted in too.
// Every part dimension in character.ts is a fraction of this, so the whole
// scene's scale follows this one number.
export const SOLDIER_HEIGHT = 1.0;
// World units per second. Rank does not change it: a king covers ground at a
// soldier's pace, he just does it in half as many, much longer strides.
export const WALK_SPEED = 4;

// Movement is velocity-based rather than position-based, so stopping coasts
// instead of cutting out. These are exponential rates (per second): higher
// converges faster.
//
// Deliberately asymmetric. Starting should feel immediate -- input lag on
// acceleration reads as unresponsive controls -- while stopping wants visible
// weight. At these values a release from full speed coasts about
// WALK_SPEED/WALK_DECEL_RATE = 0.57 units, a bit over half a cell.
export const WALK_ACCEL_RATE = 25;
export const WALK_DECEL_RATE = 7;

// ── Health and death ──────────────────────────────────────────────────────
/** Hit points every denizen starts with, regardless of rank. Rank is currently
 *  size and colour only -- a king is no tougher than a soldier. */
export const MAX_HP = 10;
/** How long a health bar stays up after the last hit it took. */
export const HEALTH_BAR_LINGER = 3;
/** ...and how long it spends fading out at the end of that. */
export const HEALTH_BAR_FADE = 0.6;
/** The stagger on death, before the body starts to fade. */
export const DEATH_SHAKE_TIME = 0.4;
/** How long the body takes to fade away once the shake is done. */
export const DEATH_FADE_TIME = 0.9;
