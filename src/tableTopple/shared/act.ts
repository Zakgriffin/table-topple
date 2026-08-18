import * as THREE from 'three';
import { camera, scene } from './scene.ts';
import { mode, onModeChange } from './mode.ts';
import { pointerToGround } from './groundRay.ts';
import { denizenInteractables, findDenizen, type Denizen } from './denizens.ts';
import { setHighlighted } from './character.ts';
import { isTargetable } from './health.ts';
import { landmarkInteractables } from './landmarks.ts';
import { treeInteractables } from './forest.ts';
import { nearestWithin, type Interactable } from './interactable.ts';
import { hasActed, isActiveTeam, onTurnChange } from './turns.ts';
import { RESOURCE_ICON_BUILDERS, type Resource } from './inventory.ts';
import {
  buildTileHighlightGeometry, type GameTile, sameTile, tileAt, tileManhattan, tilesInCircle, tilesWithinReach,
} from './gameTile.ts';
import type { PendingAction } from './actions.ts';

// Act mode: click a denizen to select it (the same emissive-boost glow
// road-mode's own hover gives a snap target -- see character.ts's
// setHighlighted), which lights up the floor in a radius around it; a second
// click on a terrain structure or tree inside that radius sends the selected
// denizen walking there. Two clicks because one point can't say both WHO and
// WHERE -- the same reason a road is a drag with a start and an end rather
// than a single point (roads.ts's own header).
//
// Picking goes through interactable.ts's generic query for all three kinds
// involved (denizens to select FROM, landmarks/trees to walk TO) rather than
// three bespoke distance loops -- see that module's header for why the three
// were worth unifying.
//
// One act per denizen per turn (turns.ts): only the active court's denizens
// are pickable at all. A commit here only sets moveTarget and a pendingAction
// (below) -- it does NOT grey the mover out. actions.ts's own per-frame
// update owns that, once the walk (and, for a tree or structure, the harvest
// after it) actually finishes; see its own header for why.
//
// ── THE HITBOX CYLINDER ──
//
// A landmark or tree's clickable footprint IS a circle -- roads.ts already
// treats them that way for snapping (closestOnLandmark), and interactable.ts's
// own `radius` (added for this) is that same circle read back out. Hovering
// one traces its exact footprint on the floor as a real, short 3D cylinder --
// same "real vertical extent so it isn't fully occluded" reasoning roads.ts's
// own hoverIndicator gives for using one instead of a flat decal -- and a walk
// command aimed at it stops on that boundary rather than at the center; see
// edgePoint below.
//
// Only shown once a denizen is SELECTED, and only for a target that denizen
// could actually reach (findTargetAt's own ACT_REACH_SECTORS filter) -- highlighting
// something you can't yet act on would show a hitbox that's a lie the moment
// you click it and nothing happens.
//
// DESKTOP-ONLY, same as wireRegionDraw/wireRoadBuild/wireAim: called only by
// server/main.ts. Never wired means `act` mode is simply unreachable, same
// consequence those three describe for their own modes.
//
// ── TILE OCCUPANCY ──
//
// The same hover that shows the exact circular hitbox ALSO highlights every
// game tile (gameTile.ts) that footprint touches -- the cylinder answers
// "can I click this", the tile highlight answers "which of the grid squares
// a road's own 6-tile span (roads.ts) would have to reach does this actually
// sit in". Same lifetime as the cylinder, built from the exact same
// tilesInCircle query roads.ts and this file both now share.
//
// ── THE HARVEST POPUP ──
//
// The same hover that shows the hitbox cylinder also shows a floating
// "+1 <icon>" over whatever's hovered, when it's something acting on pays
// out (interactable.ts's own `resource`) -- same lifetime as the cylinder,
// on together and off together, since both are answering the same question
// ("what happens if I click here"). Billboarded like health.ts's own bar
// (copy the camera's quaternion outright), and built from the exact same
// icon object the inventory panel shows for that resource
// (inventory.ts's RESOURCE_ICON_BUILDERS) rather than a second visual for
// the same thing.

/** How many game tiles (gameTile.ts) a selected denizen can reach a target
 *  from -- MANHATTAN distance between TILES, not world-unit distance between
 *  points. Reach is decided at the grid's own granularity: standing anywhere
 *  in a tile reaches exactly the same set of other tiles as standing
 *  anywhere else in it. */
const ACT_REACH_SECTORS = 2;
/** How close an actual click has to land to a candidate's own boundary to
 *  count as picking it -- unrelated to ACT_REACH_SECTORS, which only decides
 *  what's REACHABLE, not what the cursor is currently over. */
const CLICK_TOLERANCE = 2.2;

/** Height the radius disc floats at -- distinct from path's 0.02 and road's
 *  0.025 (both regionDraw.ts/roads.ts), same z-fighting reasoning both give:
 *  any two of path/road/act could legitimately be visible at once. */
const DRAW_Y = 0.03;
/** The hitbox cylinder's own height, world units -- tall enough to read as a
 *  real volume standing on the floor (and to clear radiusIndicator's flat
 *  disc without z-fighting it, the same reasoning roads.ts's own
 *  HOVER_MARKER_HEIGHT gives), short enough to still read as a footprint
 *  rather than a wall. */
const HITBOX_HEIGHT = 0.14;

let selected: Denizen | null = null;

// A flat translucent overlay rather than tinting the floor's own per-cell
// vertex colours (board.ts): that scheme is baked once at load and flat per
// tile on purpose (see board-game notes on why), and this needs to move live
// under one denizen. An overlay reads as "the ground brightens here" without
// touching that.
//
// Geometry is built directly in WORLD space by gameTile.ts's own
// buildTileHighlightGeometry (absolute tile coordinates baked in, not a
// shape-around-the-origin repositioned per frame), so the mesh itself just
// sits at a fixed (0, DRAW_Y, 0) forever -- only its GEOMETRY is rebuilt, and
// only when the selected denizen's own tile actually changes (reachTile,
// tracked in updateActIndicator below), not every frame.
const radiusIndicator = new THREE.Mesh(
  new THREE.BufferGeometry(),
  new THREE.MeshBasicMaterial({
    color: 0xfff4c2, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide,
  }),
);
radiusIndicator.rotation.x = -Math.PI / 2;
radiusIndicator.position.y = DRAW_Y;
radiusIndicator.renderOrder = 1;
radiusIndicator.visible = false;
scene.add(radiusIndicator);
let reachTile: GameTile | null = null;

// The hovered target's own hitbox, traced exactly -- built at radius 1 and
// scaled per hover (X/Z only, never Y: the cylinder's HEIGHT is constant
// regardless of which target's footprint it's tracing, only its footprint
// should change size). CylinderGeometry is centered on its own origin, so its
// position is set with height/2 as the Y each time (below) -- same "plants
// its foot at y=0" reasoning roads.ts's own hoverIndicator gives.
const hitboxIndicator = new THREE.Mesh(
  new THREE.CylinderGeometry(1, 1, HITBOX_HEIGHT, 48),
  new THREE.MeshBasicMaterial({
    color: 0xbfe8ff, transparent: true, opacity: 0.32, depthWrite: false,
  }),
);
hitboxIndicator.renderOrder = 2;
hitboxIndicator.visible = false;
scene.add(hitboxIndicator);

// Every game tile the hovered target's own circular footprint touches --
// see this file's header ("TILE OCCUPANCY"). Same geometry-in-world-space,
// rebuild-only-on-change idiom as radiusIndicator above, keyed by which
// target is hovered (occupancyTargetKey) rather than by tile, since a whole
// new set of tiles has to be recomputed every time the hovered target itself
// changes.
const occupancyHighlight = new THREE.Mesh(
  new THREE.BufferGeometry(),
  new THREE.MeshBasicMaterial({
    color: 0xbfe8ff, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide,
  }),
);
occupancyHighlight.rotation.x = -Math.PI / 2;
occupancyHighlight.position.y = DRAW_Y + 0.002; // clear of radiusIndicator's own DRAW_Y, which can be visible at once
occupancyHighlight.renderOrder = 1;
occupancyHighlight.visible = false;
scene.add(occupancyHighlight);
let occupancyTargetKey: string | null = null;

/** Rebuilds occupancyHighlight's geometry for `hovered`, but only if it isn't
 *  already showing this exact target -- pointermove fires far more often
 *  than the hovered target actually changes, and a BufferGeometry rebuild
 *  isn't free enough to redo on every one of those events. */
function setOccupancyTarget(hovered: Interactable) {
  const key = `${hovered.kind}:${hovered.id}`;
  if (key === occupancyTargetKey) return;
  occupancyTargetKey = key;
  occupancyHighlight.geometry.dispose();
  occupancyHighlight.geometry = buildTileHighlightGeometry(tilesInCircle(hovered.pos.x, hovered.pos.y, hovered.radius!));
}

// ── The harvest popup ────────────────────────────────────────────────────

/** How far above a target's own footprint the popup floats -- scaled off the
 *  target's own radius (landmarks.ts/forest.ts), same "read it back rather
 *  than hand-maintain a number per kind" reasoning those modules give for
 *  their own footprint, floored so even a small tree still clears its own
 *  canopy. */
const POPUP_HEIGHT_SCALE = 1.5;
const POPUP_MIN_HEIGHT = 2.4;
/** The icon's own scale inside the popup -- inventory.ts's builders model at
 *  roughly a unit cube, sized for that panel's own fixed camera, which is
 *  much too big standing on a board where a soldier is one unit tall. */
const POPUP_ICON_SCALE = 0.3;
/** World distance from the camera at which the popup renders at its
 *  authored size (the geometry sizes above) -- roughly the opening view's own
 *  distance to the action. Below is unrelated to how far a hovered target
 *  itself sits; see updateActIndicator's per-frame rescale. */
const POPUP_REFERENCE_DISTANCE = 15;

/** Drawn once, since the payout is always exactly one unit -- only the icon
 *  beside it ever changes. Same backing colour/opacity as health.ts's own bar
 *  (0x11131a at 0.75), so a "+1" reads as the same family of floating UI. */
function buildPlusOneTexture(): THREE.CanvasTexture {
  const size = 128;
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = 'rgba(17, 19, 26, 0.75)';
  const r = 26;
  ctx.beginPath();
  ctx.moveTo(r, 4);
  ctx.arcTo(size - 4, 4, size - 4, size - 4, r);
  ctx.arcTo(size - 4, size - 4, 4, size - 4, r);
  ctx.arcTo(4, size - 4, 4, 4, r);
  ctx.arcTo(4, 4, size - 4, 4, r);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 58px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('+1', size * 0.4, size / 2 + 4);
  return new THREE.CanvasTexture(c);
}

const popupTextMat = new THREE.MeshBasicMaterial({
  map: buildPlusOneTexture(), transparent: true, depthTest: false, depthWrite: false,
});
const popupTextMesh = new THREE.Mesh(new THREE.PlaneGeometry(0.55, 0.55), popupTextMat);
popupTextMesh.position.x = -0.08;

// Where the live icon mesh gets attached and swapped -- its own group so the
// icon's fixed viewing tilt (below) doesn't have to be redone every time the
// resource changes.
const popupIconSlot = new THREE.Group();
popupIconSlot.position.set(0.32, 0.02, 0);
popupIconSlot.scale.setScalar(POPUP_ICON_SCALE);
// A fixed tilt, not spinning (unlike inventory.ts's own icons) -- close to
// that panel's own fixed camera angle, so the icon reads the same way at a
// glance without needing to actually match the panel's per-frame render.
popupIconSlot.rotation.set(-0.3, 0.6, 0);

const popupGroup = new THREE.Group();
popupGroup.add(popupTextMesh, popupIconSlot);
popupGroup.renderOrder = 21; // above health bars' own 20
popupGroup.visible = false;
scene.add(popupGroup);

let popupResource: Resource | null = null;
let popupIcon: THREE.Object3D | null = null;

/** Swaps the icon mesh in, disposing the old one -- cheap enough to do on
 *  every resource change (there are only ever four), and simpler than trying
 *  to keep all four alive and just toggling visibility. */
function setPopupResource(resource: Resource) {
  if (resource === popupResource) return;
  popupResource = resource;
  if (popupIcon) {
    popupIconSlot.remove(popupIcon);
    popupIcon.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.geometry.dispose();
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const m of mats) m.dispose();
    });
  }
  popupIcon = RESOURCE_ICON_BUILDERS[resource]();
  // Drawn like the "+1" text beside it: on TOP of the world rather than
  // depth-tested against it. Left at inventory.ts's own defaults, the icon is
  // an ordinary lit mesh and vanishes into whatever's actually between it and
  // the camera -- a tree's own canopy, a mine's hillside -- the instant the
  // popup floats behind one from this angle.
  popupIcon.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.renderOrder = popupGroup.renderOrder;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const m of mats) { m.depthTest = false; m.depthWrite = false; }
  });
  popupIconSlot.add(popupIcon);
}

function select(d: Denizen | null) {
  if (selected) setHighlighted(selected.character, false);
  selected = d;
  radiusIndicator.visible = !!d;
  // A new (or cleared) selection invalidates whatever was hovered under the
  // OLD one -- findTargetAt's own ACT_REACH_SECTORS filter is relative to whoever's
  // selected, so the hitbox (and the popup/occupancy highlight riding along
  // with it) has to wait for the next pointermove to decide whether it's
  // still valid rather than staying visible on stale grounds.
  hitboxIndicator.visible = false;
  occupancyHighlight.visible = false;
  popupGroup.visible = false;
  if (d) setHighlighted(d.character, true);
}

/** Keeps the radius disc pinned to the selected denizen even while it moves
 *  on its own (WASD, if it's `you`) or dies mid-selection -- called every
 *  frame from server/main.ts's animate(), same as aim.ts's updateCrosshair.
 *  Also keeps the popup facing the camera every frame it's up, the same
 *  reason health.ts's own renderVitals re-copies the camera's quaternion
 *  every frame rather than once on hover: an orbiting camera can turn with
 *  no pointermove to react to. And keeps it a CONSTANT SCREEN size -- a
 *  world-space object this small would shrink to nothing at the board's own
 *  max zoom-out (view.ts's controls.maxDistance) and swallow the structure
 *  under it at max zoom-in, so its scale is rescaled every frame in
 *  proportion to its own distance from the camera, cancelling perspective
 *  out the same way a screen-space HUD element would read.
 *
 *  The radius disc's own GEOMETRY only rebuilds when the selected denizen
 *  crosses into a different game tile (reachTile) -- gameTile.ts's
 *  buildTileHighlightGeometry bakes absolute tile coordinates straight into
 *  the geometry, so there's no per-frame reposition left to do at all, unlike
 *  the old build-once-and-translate scheme this replaced. */
export function updateActIndicator() {
  if (popupGroup.visible) {
    popupGroup.quaternion.copy(camera.quaternion);
    const dist = camera.position.distanceTo(popupGroup.position);
    popupGroup.scale.setScalar(dist / POPUP_REFERENCE_DISTANCE);
  }
  if (!selected) return;
  if (!isTargetable(selected)) { select(null); return; }

  const centerTile = tileAt(selected.pos.x, selected.pos.y);
  if (!reachTile || !sameTile(reachTile, centerTile)) {
    reachTile = centerTile;
    radiusIndicator.geometry.dispose();
    radiusIndicator.geometry = buildTileHighlightGeometry(tilesWithinReach(centerTile, ACT_REACH_SECTORS));
  }
}

function findDenizenAt(pt: THREE.Vector2): Denizen | null {
  const pickable = denizenInteractables((d) => isActiveTeam(d) && !hasActed(d));
  const hit = nearestWithin(pt, pickable, CLICK_TOLERANCE);
  return hit ? findDenizen(hit.item.id) ?? null : null;
}

function findTargetAt(pt: THREE.Vector2, from: THREE.Vector2): Interactable | null {
  // Tile to tile (gameTile.ts's tileManhattan), not distanceTo's own
  // edge-of-footprint measure or raw point distance: reach is decided at the
  // grid's own granularity, the same one the (now tile-snapped) radius
  // indicator draws -- a target right at ACT_REACH_SECTORS tiles away is
  // exactly as reachable as one standing at the near edge of that same tile.
  const reachable: Interactable[] = [...landmarkInteractables(), ...treeInteractables()]
    .filter((i) => tileManhattan(from, i.pos) <= ACT_REACH_SECTORS);
  return nearestWithin(pt, reachable, CLICK_TOLERANCE)?.item ?? null;
}

/** The point on `target`'s own hitbox boundary closest to `from` -- where a
 *  walk command aimed at it should actually stop, rather than at its center
 *  (which a denizen can't stand at anyway, having a footprint of its own).
 *  Falls back to the center itself for a kind with no radius (interactable.ts's
 *  own optional field; nothing currently reachable here lacks one). */
function edgePoint(from: THREE.Vector2, target: Interactable): THREE.Vector2 {
  if (!target.radius) return target.pos.clone();
  const toFrom = new THREE.Vector2().subVectors(from, target.pos);
  // Degenerate: `from` sits exactly on the target's own center (never
  // actually reachable, but a direction has to come from somewhere). Any
  // point on the boundary is as good as another.
  if (toFrom.lengthSq() < 1e-6) toFrom.set(0, 1);
  return target.pos.clone().addScaledVector(toFrom.normalize(), target.radius);
}

const scratch = new THREE.Vector2();

/** Starts listening for act-mode picks on the game's own canvas. */
export function wireAct(canvas: HTMLElement) {
  canvas.addEventListener('pointerdown', (e) => {
    if (mode !== 'act' || e.button !== 0) return;
    if (!pointerToGround(e, scratch)) return;

    // A denizen under the click always wins, whether or not one is already
    // selected -- reselecting (or picking a different one) has to work at
    // any time, the same way roads.ts lets a fresh drag start over an old
    // blueprint.
    const clicked = findDenizenAt(scratch);
    if (clicked) { select(clicked); return; }

    if (selected) {
      const target = findTargetAt(scratch, selected.pos);
      if (target) {
        const mover = selected;
        // Stop at the target's own edge, not its center -- see edgePoint's
        // own comment.
        mover.moveTarget = edgePoint(mover.pos, target);
        // A tree or terrain structure gets harvested on arrival (actions.ts);
        // anything else is a plain walk, spent the moment it gets there.
        // `target.kind` is narrowed to the two harvestable kinds here since
        // findTargetAt only ever draws from landmarkInteractables/
        // treeInteractables, both of which always set `resource`.
        const action: PendingAction =
          (target.kind === 'tree' || target.kind === 'landmark') && target.resource
            ? {
              kind: 'harvest',
              target: { kind: target.kind, id: target.id },
              resource: target.resource,
              timer: 0,
              previousWeapon: mover.weapon,
              bar: null,
            }
            : { kind: 'arrive' };
        mover.pendingAction = action;
        select(null);
        return;
      }
    }

    // Missed everything: cancel whatever was selected rather than leave it
    // lit with no obvious way to back out.
    select(null);
  });

  canvas.addEventListener('pointermove', (e) => {
    // Nothing usable to highlight without a mover already picked -- see this
    // file's header on why "usable now" gates the hitbox, not just presence.
    if (mode !== 'act' || !selected) {
      hitboxIndicator.visible = false; occupancyHighlight.visible = false; popupGroup.visible = false;
      return;
    }
    if (!pointerToGround(e, scratch)) return;
    const hovered = findTargetAt(scratch, selected.pos);
    if (!hovered || !hovered.radius) {
      hitboxIndicator.visible = false; occupancyHighlight.visible = false; popupGroup.visible = false;
      return;
    }
    hitboxIndicator.position.set(hovered.pos.x, HITBOX_HEIGHT / 2, hovered.pos.y);
    hitboxIndicator.scale.set(hovered.radius, 1, hovered.radius);
    hitboxIndicator.visible = true;

    // Same lifetime as the hitbox above -- up together, down together.
    setOccupancyTarget(hovered);
    occupancyHighlight.visible = true;

    if (hovered.resource) {
      setPopupResource(hovered.resource);
      const y = Math.max(POPUP_MIN_HEIGHT, hovered.radius * POPUP_HEIGHT_SCALE);
      popupGroup.position.set(hovered.pos.x, y, hovered.pos.y);
      popupGroup.visible = true;
    } else {
      popupGroup.visible = false;
    }
  });

  onModeChange((next, prev) => {
    if (prev === 'act' && next !== 'act') select(null);
    if (next !== 'act') { hitboxIndicator.visible = false; occupancyHighlight.visible = false; popupGroup.visible = false; }
  });
  // A new turn means whatever was selected belonged to the court whose turn
  // just ended -- it can't still be commanded, so there's nothing left for
  // the selection to mean.
  onTurnChange(() => select(null));
}
