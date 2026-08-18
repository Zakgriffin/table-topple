import { setSpent } from './character.ts';
import { findDenizen, type Denizen } from './denizens.ts';
import {
  COLOR_TEAM_BLUE, COLOR_TEAM_GREEN, COLOR_TEAM_RED, COLOR_TEAM_YELLOW,
} from './constants.ts';

// Whose turn it is. There's no separate human per court -- the server plays
// every court -- so "taking a turn" doesn't drive any AI of its own; it only
// decides which court's denizens act.ts's picking will currently accept, and
// which court's colour the UI that used to be permanently red now wears.
//
// Order is COUNTER-CLOCKWISE as viewed from above the board, derived from
// each court's own facing (denizens.ts): increasing facing sweeps
// red(0) -> green(PI/2) -> blue(PI) -> yellow(-PI/2), which is CLOCKWISE (a
// compass's own N -> E -> S -> W). Counter-clockwise is that reversed,
// starting from red.
const TURN_ORDER = [COLOR_TEAM_RED, COLOR_TEAM_YELLOW, COLOR_TEAM_BLUE, COLOR_TEAM_GREEN];

let activeIndex = 0;

/** This turn's team colour -- constants.ts's COLOR_TEAM_*. */
export function activeTeam(): number {
  return TURN_ORDER[activeIndex];
}

export function isActiveTeam(d: Denizen): boolean {
  return d.team === activeTeam();
}

/** Ids that have already spent this turn's one act -- cleared whenever the
 *  turn advances (see nextTurn), never by anything else. */
const spent = new Set<number>();

export function hasActed(d: Denizen): boolean {
  return spent.has(d.id);
}

/** Marks a denizen as having used this turn's one act, and grays it out --
 *  called by act.ts the instant a walk command commits. */
export function markActed(d: Denizen) {
  if (spent.has(d.id)) return;
  spent.add(d.id);
  setSpent(d.character, true);
}

type Listener = (team: number) => void;
const listeners: Listener[] = [];
/** act.ts subscribes to clear its own selection when the turn moves on;
 *  wireTurnButton (below) subscribes to repaint the button. turns.ts knows
 *  about neither -- same one-way-dependency reasoning mode.ts's own
 *  onModeChange gives for not reaching into the game itself. */
export function onTurnChange(fn: Listener) { listeners.push(fn); }

/**
 * Advances to the next court, counter-clockwise, and ungrays everyone the
 * outgoing turn had marked spent -- a denizen who acted last turn should read
 * as fresh again the instant it stops being relevant which turn they acted
 * on, not only once their own turn comes back around.
 */
export function nextTurn() {
  for (const id of spent) {
    const d = findDenizen(id);
    if (d) setSpent(d.character, false);
  }
  spent.clear();
  activeIndex = (activeIndex + 1) % TURN_ORDER.length;
  for (const fn of listeners) fn(activeTeam());
}

// ── UI ────────────────────────────────────────────────────────────────────

/** Exported: inventory.ts's own panel titles itself off the same names
 *  ("red's inventory"), rather than keeping a second copy. */
export const TEAM_LABEL: Record<number, string> = {
  [COLOR_TEAM_RED]: 'red',
  [COLOR_TEAM_BLUE]: 'blue',
  [COLOR_TEAM_GREEN]: 'green',
  [COLOR_TEAM_YELLOW]: 'yellow',
};

function hexToRgb(hex: number): string {
  return `${(hex >> 16) & 0xff}, ${(hex >> 8) & 0xff}, ${hex & 0xff}`;
}

/**
 * Repaints whatever in the page reads the active team's colour, through ONE
 * CSS custom property on the document root -- `--team-rgb`, an "r, g, b"
 * triple so `rgba(var(--team-rgb), alpha)` can still pick its own opacity per
 * rule. table-topple-server.html reads it from two places: `#modeToggle
 * button.active` (which used to be hardcoded to COLOR_TEAM_RED -- "the
 * player's own red", its own old comment said) and `#turnToggle` itself, so
 * the button that changes the turn visibly wears the turn it's currently on.
 */
function paintTeamColor(team: number) {
  document.documentElement.style.setProperty('--team-rgb', hexToRgb(team));
}

/** Adopts the "next turn" button, if the host page has one -- same guarded-
 *  optional idiom ai.ts's wireBattleButton uses. A host without it can still
 *  advance turns; nextTurn() above is the actual control. The root colour
 *  property is painted unconditionally, though, since the mode bar reads it
 *  regardless of whether this page has a turn button of its own. */
export function wireTurnButton() {
  paintTeamColor(activeTeam());
  const button = document.getElementById('turnToggle') as HTMLButtonElement | null;
  if (!button) return;

  const paint = () => { button.textContent = `${TEAM_LABEL[activeTeam()]}'s turn`; };
  paint();

  button.addEventListener('click', () => {
    nextTurn();
    button.blur();
  });
  onTurnChange(() => { paint(); paintTeamColor(activeTeam()); });
}
