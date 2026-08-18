import { WEAPONS, WEAPON_ORDER, type WeaponKey } from './weapons.ts';

// What the mouse currently does. Exactly one of these owns the pointer at a
// time -- the alternative (modifier chords) means remembering which chord does
// what while also playing.
//
//   camera  orbit/pan the view          (Escape)
//   path    draw a region on the floor  (P)
//   road    drag out road blueprints,   (R)
//           confirmed separately -- see roads.ts
//   fight   aim and use the weapon      (E, or any of 1/2/3)
//   act     pick a denizen, then send   (A)
//           it walking to a nearby
//           structure or tree -- see act.ts
//
// Which weapon is in hand is a SEPARATE axis from the mode: 1/2/3 pick it, and
// picking one also draws it. That keeps the mode bar to one entry per MODE
// instead of growing one per weapon, and means swapping mid-fight doesn't pass
// through any other state.
export type InputMode = 'camera' | 'path' | 'road' | 'fight' | 'act';

export const MODES: InputMode[] = ['camera', 'path', 'road', 'fight', 'act'];

const LABELS: Record<InputMode, string> = {
  camera: 'camera',
  path: 'path',
  road: 'road',
  fight: 'fight',
  act: 'act',
};

/** Shown under each label, so the hotkey and the button that does the same
 *  thing are never out of sync in the UI. */
const KEY_HINTS: Record<InputMode, string> = {
  camera: 'esc',
  path: 'p',
  road: 'r',
  fight: 'e',
  act: 'a',
};

export let mode: InputMode = 'camera';

let currentWeapon: WeaponKey = 'sword';
/** The weapon that will be (or already is) in hand. */
export function equipped(): WeaponKey { return currentWeapon; }

// Subscribers rather than mode.ts reaching into the game itself: the weapon
// and drawing code care about mode changes, but mode.ts importing THEM would
// make a cycle out of what should be a one-way dependency.
type Listener = (next: InputMode, prev: InputMode) => void;
const listeners: Listener[] = [];
export function onModeChange(fn: Listener) { listeners.push(fn); }

type WeaponListener = (key: WeaponKey) => void;
const weaponListeners: WeaponListener[] = [];
export function onWeaponChange(fn: WeaponListener) { weaponListeners.push(fn); }

// Populated by wireModeUI(), and empty until then. A host with no mode bar --
// the AR overlay -- simply never calls it, and every loop over these maps
// below turns into a no-op rather than needing a guard of its own.
const buttons = new Map<InputMode, HTMLButtonElement>();
const weaponButtons = new Map<WeaponKey, HTMLButtonElement>();

/**
 * Adopts the mode bar, the weapon rack and the hotkeys that duplicate them.
 *
 * Called only by the standalone page's boot. It has to be a call rather than
 * import-time work: this module is on the simulation's own import path (the
 * weapon and combat code read `mode`), so wiring a keydown handler at import
 * would silently install one on any page that runs the game at all -- the
 * capture page included, where the keyboard belongs to something else.
 */
export function wireModeUI() {
  for (const m of MODES) {
    const el = document.querySelector<HTMLButtonElement>(`#modeToggle button[data-mode="${m}"]`);
    if (!el) continue;
    el.textContent = LABELS[m];
    const key = document.createElement('span');
    key.className = 'key';
    key.textContent = KEY_HINTS[m];
    el.appendChild(key);
    el.addEventListener('click', () => {
      setMode(m);
      // A clicked button keeps keyboard focus, and a focused button treats
      // Space/Enter as another click -- drop focus so later keys are the
      // game's, not the button's.
      el.blur();
    });
    buttons.set(m, el);
  }

  for (const w of WEAPON_ORDER) {
    const el = document.querySelector<HTMLButtonElement>(`#weaponToggle button[data-weapon="${w}"]`);
    if (!el) continue;
    el.textContent = WEAPONS[w].label;
    const key = document.createElement('span');
    key.className = 'key';
    key.textContent = WEAPONS[w].key;
    el.appendChild(key);
    el.addEventListener('click', () => { selectWeapon(w); el.blur(); });
    weaponButtons.set(w, el);
  }

  addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
    // Note: while the pointer is locked, the browser eats Escape to exit the
    // lock and never delivers it here. aim.ts handles that case by watching for
    // the lock being released -- which lands in the same place, camera mode.
    if (e.code === 'Escape') setMode('camera');
    else if (e.code === 'KeyP') setMode('path');
    else if (e.code === 'KeyR') setMode('road');
    else if (e.code === 'KeyE') setMode('fight');
    else if (e.code === 'KeyA') setMode('act');
    else {
      const weapon = WEAPON_ORDER.find((w) => WEAPONS[w].key === e.key);
      if (!weapon) return;
      selectWeapon(weapon);
    }
    e.preventDefault();
  });

  // Reflect the state the module already booted in, so the bar agrees with it
  // from the first frame.
  buttons.get('camera')?.classList.add('active');
  weaponButtons.get(currentWeapon)?.classList.add('active');
}

export function setMode(next: InputMode) {
  const prev = mode;
  if (next === prev) return;
  mode = next;

  // What the camera controls do about this is view.ts's business, subscribed
  // through onModeChange below -- this module has no renderer to reach into.
  for (const [m, el] of buttons) el.classList.toggle('active', m === next);
  // Crosshair while drawing, or picking. Fight mode hides the cursor outright
  // via pointer lock (aim.ts) and draws its own reticle, so it needs nothing
  // here.
  document.body.style.cursor = next === 'path' || next === 'road' || next === 'act' ? 'crosshair' : '';

  for (const fn of listeners) fn(next, prev);
}

/** Picks a weapon, and draws it: reaching for a weapon IS entering the fight,
 *  so 1/2/3 work as a single gesture from any mode. */
export function selectWeapon(key: WeaponKey) {
  currentWeapon = key;
  for (const [w, el] of weaponButtons) el.classList.toggle('active', w === key);
  if (mode !== 'fight') setMode('fight');
  else for (const fn of weaponListeners) fn(key);
}
