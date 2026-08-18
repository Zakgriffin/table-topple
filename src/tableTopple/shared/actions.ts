import * as THREE from 'three';
import { camera, scene } from './scene.ts';
import { SOLDIER_HEIGHT } from './constants.ts';
import { CHARACTER_UNIT as U, HAND_LOCAL_Y } from './character.ts';
import { denizens, type Denizen } from './denizens.ts';
import { equipWeapon, type WeaponKey } from './weapons.ts';
import { hideTree } from './forest.ts';
import { harvestLandmark } from './landmarks.ts';
import { addResource, type Resource } from './inventory.ts';
import { createBar, type HealthBar } from './health.ts';
import { markActed } from './turns.ts';

// What a denizen does once it arrives at an act-mode moveTarget (act.ts) --
// a plain walk just marks the mover spent the instant it gets there; walking
// to a tree or terrain structure harvests it first, with a tool matched to
// whatever that target pays out. Driven once a frame from sim.ts's step(),
// the same way ai.ts's updateAI is: a per-denizen decision loop, not input
// wiring, so it belongs beside the rest of the simulation rather than in
// act.ts itself.
//
// This is also WHY the mover doesn't grey out until it's done (turns.ts's
// markActed is only ever called from in here, never from act.ts's own
// commit): "spent" means "used its turn", and a walk that hasn't arrived, or
// a harvest still swinging, hasn't used anything yet.

export type HarvestTarget =
  | { kind: 'tree'; id: number }
  | { kind: 'landmark'; id: number };

export type PendingAction =
  | { kind: 'arrive' }
  | {
    kind: 'harvest';
    target: HarvestTarget;
    resource: Resource;
    timer: number;
    previousWeapon: WeaponKey | null;
    /** The loading bar shown over the denizen's head while it works, built
     *  the instant the harvest actually starts (same moment as the tool
     *  equip, timer 0 -> nonzero) and torn down the instant it stops, one way
     *  or another -- see startHarvest/endHarvest below. Null in between times
     *  is never observed by anything outside this file. */
    bar: HealthBar | null;
  };

/** Seconds the harvest animation runs once the denizen arrives at the target. */
const HARVEST_DURATION = 2.4;
/** How many up-down strikes fit in that span. Doubled along with
 *  HARVEST_DURATION so the swing keeps its original pace rather than slowing
 *  down -- there are just twice as many of them now. */
const HARVEST_SWINGS = 6;

// ── The tools: temporary visuals, not real weapons ──────────────────────────
// Built in the same blocky-boxes idiom weapons.ts uses (handle + head, no
// bevels), but deliberately NOT added to weapons.ts's WEAPONS registry: none
// has combat stats, none is ever selected with 1/2/3, and none should ever be
// reachable from ai.ts/combat.ts, which a real WeaponKey would make possible
// by construction. Each still rides the real equip/dispose machinery below
// (equipWeapon(d, null) and `heldObject`), just without ever being one.

const WOOD = 0x6b4a2a;
const METAL = 0xb8bcc4;
const DARK_METAL = 0x7d818c;
const HANDLE_L = 15 * U, HANDLE_W = 1.1 * U;

function buildAxe(): THREE.Object3D {
  const axe = new THREE.Group();
  const woodMat = new THREE.MeshStandardMaterial({ color: WOOD, roughness: 0.85 });
  const metalMat = new THREE.MeshStandardMaterial({ color: METAL, roughness: 0.4, metalness: 0.6 });

  const handle = new THREE.Mesh(new THREE.BoxGeometry(HANDLE_W, HANDLE_L, HANDLE_W), woodMat);
  handle.position.y = -HANDLE_L / 2;
  axe.add(handle);

  // Offset to one side of the handle's axis so the silhouette reads as a
  // blade rather than a mallet.
  const HEAD_W = 6 * U, HEAD_H = 3.5 * U, HEAD_T = 1.2 * U;
  const head = new THREE.Mesh(new THREE.BoxGeometry(HEAD_W, HEAD_H, HEAD_T), metalMat);
  head.position.set(HEAD_W * 0.3, -HANDLE_L * 0.85, 0);
  axe.add(head);

  return axe;
}

/** A pick: the axe's own handle, but the head is two thin spikes angled apart
 *  from the grip -- a mattock's silhouette rather than a blade's. */
function buildPickaxe(): THREE.Object3D {
  const pick = new THREE.Group();
  const woodMat = new THREE.MeshStandardMaterial({ color: WOOD, roughness: 0.85 });
  const metalMat = new THREE.MeshStandardMaterial({ color: DARK_METAL, roughness: 0.35, metalness: 0.7 });

  const handle = new THREE.Mesh(new THREE.BoxGeometry(HANDLE_W, HANDLE_L, HANDLE_W), woodMat);
  handle.position.y = -HANDLE_L / 2;
  pick.add(handle);

  const SPIKE_L = 5 * U, SPIKE_T = 1 * U;
  const headY = -HANDLE_L * 0.85;
  for (const s of [-1, 1]) {
    const spike = new THREE.Mesh(new THREE.BoxGeometry(SPIKE_L, SPIKE_T, SPIKE_T), metalMat);
    spike.position.set(s * SPIKE_L * 0.42, headY, 0);
    spike.rotation.z = s * 0.5;
    pick.add(spike);
  }

  return pick;
}

/** A scythe: a taller shaft and one long blade set at a shallow angle off its
 *  top -- the closest a straight box gets to a curved edge. */
function buildScythe(): THREE.Object3D {
  const scythe = new THREE.Group();
  const woodMat = new THREE.MeshStandardMaterial({ color: WOOD, roughness: 0.85 });
  const metalMat = new THREE.MeshStandardMaterial({ color: METAL, roughness: 0.3, metalness: 0.7 });

  const shaftL = HANDLE_L * 1.2;
  const handle = new THREE.Mesh(new THREE.BoxGeometry(HANDLE_W, shaftL, HANDLE_W), woodMat);
  handle.position.y = -shaftL / 2;
  scythe.add(handle);

  const BLADE_L = 8 * U, BLADE_W = 1.4 * U, BLADE_T = 0.5 * U;
  const blade = new THREE.Mesh(new THREE.BoxGeometry(BLADE_L, BLADE_W, BLADE_T), metalMat);
  blade.position.set(BLADE_L * 0.4, -shaftL * 0.92, 0);
  blade.rotation.z = -0.9;
  scythe.add(blade);

  return scythe;
}

/** A wand: a thin shaft with a small glowing crystal tip -- weapons.ts's
 *  staff, shrunk to a one-hand tool. Borrows its exact crystal colours
 *  (weapons.ts's buildStaff / inventory.ts's buildAetherIcon) rather than
 *  inventing a third look for the same resource. */
function buildWand(): THREE.Object3D {
  const wand = new THREE.Group();
  const shaftMat = new THREE.MeshStandardMaterial({ color: 0x4b3b2c, roughness: 0.9 });
  const crystalMat = new THREE.MeshStandardMaterial({
    color: 0x8fd8ff, emissive: 0x2b6fa8, emissiveIntensity: 1.1, roughness: 0.2, metalness: 0.1,
  });

  const SHAFT_L = HANDLE_L * 0.85, SHAFT_T = HANDLE_W * 0.7;
  const shaft = new THREE.Mesh(new THREE.BoxGeometry(SHAFT_T, SHAFT_L, SHAFT_T), shaftMat);
  shaft.position.y = -SHAFT_L / 2;
  wand.add(shaft);

  const TIP = 2.2 * U;
  const crystal = new THREE.Mesh(new THREE.OctahedronGeometry(TIP, 0), crystalMat);
  crystal.position.y = -SHAFT_L - TIP * 0.4;
  wand.add(crystal);

  return wand;
}

const TOOL_BUILDERS: Record<Resource, () => THREE.Object3D> = {
  wood: buildAxe, metal: buildPickaxe, string: buildScythe, aether: buildWand,
};

/** Swaps whatever's in the right hand for the tool matched to `resource`.
 *  equipWeapon(d, null) is what actually clears and disposes the old one,
 *  through its own tested path -- the tool is then attached by hand since it
 *  isn't a WeaponKey equipWeapon could build itself. */
function equipTool(d: Denizen, resource: Resource) {
  equipWeapon(d, null);
  const tool = TOOL_BUILDERS[resource]();
  tool.position.y = HAND_LOCAL_Y;
  d.character.rightArm.add(tool);
  d.heldObject = tool;
}

/** Puts the tool away and restores whatever was equipped before the harvest.
 *  equipWeapon finds the tool sitting in `heldObject` (weapon is still null,
 *  so holdingArm() still resolves the same right arm the tool was put in),
 *  disposes it exactly like it would any weapon, and re-arms
 *  `previousWeapon`, or nothing. */
function unequipTool(d: Denizen, previousWeapon: WeaponKey | null) {
  equipWeapon(d, previousWeapon);
}

/** Repeating up-down swing, drawn straight onto the arm's rotation rather
 *  than through aim.ts: `d.weapon` is null for the whole harvest (see
 *  equipTool), so applyAimPose already leaves this arm alone. One motion for
 *  all four tools -- axe, pick, scythe and wand all read as "working at
 *  something" with the same swing, and only the tool in hand changes. Must
 *  run AFTER renderDenizen's own updateWalk call for this denizen THIS frame,
 *  or the idle pose would overwrite it straight back -- see the call site in
 *  sim.ts's step(). */
function poseHarvest(d: Denizen, t: number) {
  const swing = Math.sin((t / HARVEST_DURATION) * HARVEST_SWINGS * Math.PI * 2);
  d.character.rightArm.rotation.x = -1.1 + swing * 0.9;
}

// ── The loading bar ──────────────────────────────────────────────────────
// Same style as health.ts's own damage bar -- built by that module's exact
// createBar (dark backing, coloured fill that grows from the left edge), just
// with a different colour so it doesn't read as a hurt denizen, and filling
// UP with progress rather than draining down with damage. A little higher
// than the health bar's own clearance so the two never overlap if a denizen
// takes a hit mid-harvest.

const HARVEST_BAR_COLOR = 0xbfe8ff; // act.ts's own hitbox-cylinder blue
const HARVEST_BAR_CLEARANCE = 0.55;

function positionHarvestBar(bar: HealthBar, d: Denizen) {
  bar.group.position.set(d.pos.x, SOLDIER_HEIGHT * d.scale + HARVEST_BAR_CLEARANCE * d.scale, d.pos.y);
  // Billboard, same as health.ts's renderVitals -- copied every frame the bar
  // is up, not just once, since an orbiting camera can turn with nothing else
  // about this denizen changing.
  bar.group.quaternion.copy(camera.quaternion);
}

function startHarvestBar(d: Denizen): HealthBar {
  const bar = createBar(HARVEST_BAR_COLOR);
  bar.fill.scale.x = 0.0001; // starts empty; never exactly 0 (degenerate matrix)
  bar.group.visible = true;
  positionHarvestBar(bar, d);
  return bar;
}

/** Torn down outright rather than hidden -- unlike a denizen's own health
 *  bar, which lives for that denizen's whole life, this one exists only for
 *  the span of one harvest and would otherwise leak a group into the scene
 *  every time someone chops a tree. */
function endHarvestBar(bar: HealthBar) {
  scene.remove(bar.group);
  bar.backMat.dispose();
  bar.fillMat.dispose();
}

/**
 * Abandons whatever `d` was about to do or is in the middle of doing --
 * unequipping the tool first if a harvest had actually started (timer > 0 is
 * exactly that: updateActions only ever equips it the instant a harvest's
 * timer leaves 0). Used by regionDraw.ts: a fresh path-mode walk overwrites
 * `you.moveTarget` directly rather than going through act.ts, which would
 * otherwise leave a stale pendingAction pointed at wherever the OLD target
 * was -- and, worse, a denizen stuck holding a tool forever with nothing left
 * to ever put it away.
 */
export function cancelPendingAction(d: Denizen) {
  const action = d.pendingAction;
  if (action?.kind === 'harvest' && action.timer > 0) {
    unequipTool(d, action.previousWeapon);
    if (action.bar) endHarvestBar(action.bar);
  }
  d.pendingAction = null;
}

/**
 * Advances every denizen's pending action by one frame. A denizen still
 * walking there (moveTarget non-null) is left alone -- sim.ts's own movement
 * code owns that part. Called once a frame, after the render loop (see
 * sim.ts's step()) so poseHarvest's write is the last thing touching that arm
 * this frame.
 */
export function updateActions(dt: number) {
  for (const d of denizens) {
    const action = d.pendingAction;
    if (!action || d.moveTarget) continue;

    if (action.kind === 'arrive') {
      d.pendingAction = null;
      markActed(d);
      continue;
    }

    // 'harvest'
    if (action.timer === 0) {
      equipTool(d, action.resource);
      action.bar = startHarvestBar(d);
    }
    action.timer += dt;
    if (action.timer < HARVEST_DURATION) {
      poseHarvest(d, action.timer);
      action.bar!.fill.scale.x = Math.max(action.timer / HARVEST_DURATION, 0.0001);
      positionHarvestBar(action.bar!, d);
      continue;
    }

    unequipTool(d, action.previousWeapon);
    endHarvestBar(action.bar!);
    // A tree is gone after one harvest; a structure just gives up one unit of
    // stock and stays standing. Either can come back false if another
    // denizen got there first (hideTree) or already emptied it
    // (harvestLandmark) -- that guards the PAYOUT, not the race itself: both
    // still finish their swing and both still spend their turn.
    const granted = action.target.kind === 'tree'
      ? hideTree(action.target.id)
      : harvestLandmark(action.target.id);
    if (granted) addResource(d.team, action.resource, 1);
    d.pendingAction = null;
    markActed(d);
  }
}
