import * as THREE from 'three';
import {
  COLOR_TEAM_BLUE, COLOR_TEAM_GREEN, COLOR_TEAM_RED, COLOR_TEAM_YELLOW,
} from './constants.ts';
import { activeTeam, onTurnChange, TEAM_LABEL } from './turns.ts';

// Each court's own stockpile. Pure state plus a change listener, same shape
// as turns.ts -- the UI (below) shows whichever team's turn it currently is,
// from turns.ts's activeTeam(), not a fixed one.

export type Resource = 'wood' | 'metal' | 'string' | 'aether';
export const RESOURCES: Resource[] = ['wood', 'metal', 'string', 'aether'];

function empty(): Record<Resource, number> {
  return { wood: 0, metal: 0, string: 0, aether: 0 };
}

const inventories = new Map<number, Record<Resource, number>>(
  [COLOR_TEAM_RED, COLOR_TEAM_BLUE, COLOR_TEAM_GREEN, COLOR_TEAM_YELLOW].map((team) => [team, empty()]),
);

export function inventoryOf(team: number): Readonly<Record<Resource, number>> {
  return inventories.get(team)!;
}

type Listener = (team: number) => void;
const listeners: Listener[] = [];
export function onInventoryChange(fn: Listener) { listeners.push(fn); }

/** Adds to a court's stockpile -- actions.ts calls this once a chop completes.
 *  No source subtracts yet; there's nothing to spend on today. */
export function addResource(team: number, resource: Resource, amount: number) {
  inventories.get(team)![resource] += amount;
  for (const fn of listeners) fn(team);
}

// ── The panel's own tiny 3D icons ───────────────────────────────────────────
// One real WebGL scene per resource, rendered into its own small canvas
// inside the panel -- not a CSS fake-3D trick. Four extra contexts is nothing
// next to the one WebGL limit that matters (browsers allow well over a dozen),
// and only ONE of them (#gl, view.ts) is ever the big one. Items are built in
// the same "boxes and simple primitives, flat-lit" spirit as weapons.ts's
// axe/sword, sized to sit inside roughly a unit cube so one fixed camera
// frames all four without per-item tuning.

const ICON_SIZE = 56; // CSS pixels, square -- matches .inv-icon in the page's own CSS
const ICON_SPIN_RATE = 0.6; // radians/sec, just enough to read as 3D rather than a flat sprite

function buildWoodIcon(): THREE.Object3D {
  const group = new THREE.Group();
  const bark = new THREE.MeshStandardMaterial({ color: 0x5a3d24, roughness: 0.85 });
  const rings = new THREE.MeshStandardMaterial({ color: 0xc9a06b, roughness: 0.7 });
  const log = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.9, 10), bark);
  log.rotation.z = Math.PI / 2;
  group.add(log);
  for (const s of [-1, 1]) {
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.43, 0.43, 0.03, 10), rings);
    cap.rotation.z = Math.PI / 2;
    cap.position.x = s * 0.45;
    group.add(cap);
  }
  return group;
}

function buildMetalIcon(): THREE.Object3D {
  const mat = new THREE.MeshStandardMaterial({ color: 0xb9bec8, roughness: 0.35, metalness: 0.8 });
  return new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.4, 0.55), mat);
}

function buildStringIcon(): THREE.Object3D {
  const mat = new THREE.MeshStandardMaterial({ color: 0xe6dcc4, roughness: 0.75 });
  return new THREE.Mesh(new THREE.TorusGeometry(0.4, 0.15, 10, 18), mat);
}

/** Same crystal treatment weapons.ts's staff gives its own -- aether is the
 *  magic resource, so it borrows that visual language rather than inventing
 *  a second one. */
function buildAetherIcon(): THREE.Object3D {
  const mat = new THREE.MeshStandardMaterial({
    color: 0x8fd8ff, emissive: 0x2b6fa8, emissiveIntensity: 1.1, roughness: 0.2, metalness: 0.1,
  });
  const gem = new THREE.Mesh(new THREE.OctahedronGeometry(0.55, 0), mat);
  gem.rotation.z = Math.PI / 6;
  return gem;
}

/** Exported so anything else that wants to show "this resource" in 3D --
 *  act.ts's hover popup -- builds the exact same object this panel does,
 *  rather than a second visual for the same resource. */
export const RESOURCE_ICON_BUILDERS: Record<Resource, () => THREE.Object3D> = {
  wood: buildWoodIcon, metal: buildMetalIcon, string: buildStringIcon, aether: buildAetherIcon,
};

interface Icon {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  item: THREE.Object3D;
}

function buildIcon(canvas: HTMLCanvasElement, resource: Resource): Icon {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 10);
  camera.position.set(1.05, 0.95, 1.5);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(2, 3, 2);
  scene.add(key);

  const item = RESOURCE_ICON_BUILDERS[resource]();
  scene.add(item);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  // false: only the drawing buffer, matching view.ts's own reasoning in
  // reverse -- THIS canvas already has a CSS size from the stylesheet
  // (.inv-icon), so writing a second one here would fight it.
  renderer.setSize(ICON_SIZE, ICON_SIZE, false);

  return { renderer, scene, camera, item };
}

// ── UI ────────────────────────────────────────────────────────────────────

let panelOpen = false;
const icons: Icon[] = [];

/** Spins and re-renders every icon, but only while the panel is actually
 *  open -- called unconditionally every frame from server/main.ts's
 *  animate(), same as aim.ts's updateCrosshair, and just as cheap to skip
 *  when there's nothing to show. */
export function updateInventoryIcons(dt: number) {
  if (!panelOpen) return;
  for (const icon of icons) {
    icon.item.rotation.y += dt * ICON_SPIN_RATE;
    icon.renderer.render(icon.scene, icon.camera);
  }
}

/**
 * Adopts the inventory button, its overlay and its panel, if the host page
 * has all three -- same guarded-optional idiom ai.ts's wireBattleButton
 * uses. Always shows the CURRENT TURN's stockpile, never a fixed one --
 * repainted on both an inventory change and a turn change, since either can
 * make the numbers on screen stale.
 */
export function wireInventoryPanel() {
  const button = document.getElementById('inventoryToggle') as HTMLButtonElement | null;
  const overlay = document.getElementById('inventoryOverlay') as HTMLDivElement | null;
  const panel = document.getElementById('inventoryPanel') as HTMLDivElement | null;
  const title = panel?.querySelector<HTMLElement>('.inv-title') ?? null;
  if (!button || !overlay || !panel) return;

  const counts = new Map<Resource, HTMLElement>();
  for (const r of RESOURCES) {
    const el = panel.querySelector<HTMLElement>(`[data-resource="${r}"]`);
    if (el) counts.set(r, el);
    const canvas = panel.querySelector<HTMLCanvasElement>(`canvas[data-icon="${r}"]`);
    if (canvas) icons.push(buildIcon(canvas, r));
  }

  const paint = () => {
    const team = activeTeam();
    const inv = inventoryOf(team);
    for (const [r, el] of counts) el.textContent = String(inv[r]);
    if (title) title.textContent = `${TEAM_LABEL[team]}'s inventory`;
  };
  paint();

  const setOpen = (next: boolean) => {
    panelOpen = next;
    overlay.style.display = next ? 'block' : 'none';
    panel.style.display = next ? 'block' : 'none';
    button.classList.toggle('active', next);
  };

  button.addEventListener('click', () => { setOpen(!panelOpen); button.blur(); });
  // Click-outside-to-close, the usual modal convention -- the panel itself
  // doesn't relay clicks up to the overlay since it isn't a child of it.
  overlay.addEventListener('click', () => setOpen(false));

  onInventoryChange(paint);
  onTurnChange(paint);
}
