// Table Topple Server -- the board game sandbox, deliberately isolated from the De Bruijn/pose
// reconstruction pipeline that the rest of this repo is about. The AR pose
// work exists to eventually put a real board on a real table; this page is
// where the game those poses are FOR gets designed, in a plain virtual scene
// with no camera, no capture, and no CV.
//
// Layout:
//   constants.ts  board dimensions, palette, denizen size/speed
//   scene.ts      the world container: the scene, its lights, and the shared
//                 camera it is seen through -- no DOM, no renderer
//   view.ts       how THIS page presents it: canvas, renderer, orbit controls,
//                 resize. The AR overlay on the capture page has its own
//                 equivalent and never imports this one
//   sim.ts        one frame of the world, with nothing about drawing it
//   board.ts      the playing surface: one plane, coloured cell by cell
//   terrain.ts    what each cell is made of: forest, and patches in it
//   gameTile.ts   the board's "game tiles" -- a coarser grid terrain patches
//                 snap to, the floor checkerboards, act-mode reach and a
//                 road's max span are measured in
//   landmarks.ts  one structure per patch: shrubbery, mine, ruined church --
//                 each with a stock of string/metal/aether to harvest
//   forest.ts     trees scattered over the forest terrain, instanced
//   blocks.ts     the shared box-assembly kit scenery is built from
//   noise.ts      deterministic value noise + the seeded RNG the world uses
//   character.ts  the blocky humanoid model, built from boxes
//   ranks.ts      the denizen hierarchy: size, shading, crown
//   denizens.ts   the four courts: figures with a rank, position, and facing
//   castle.ts     each court's castle, from a tower/wall plan
//   animation.ts  the walk cycle + turn easing, driven by distance walked
//   idle.ts       a standing denizen's own life: head wander, eye blinks
//   mode.ts       who owns the mouse: camera / path / road / fight / act, +
//                 weapon choice
//   regionDraw.ts path-mode drawing: trace a region, walk to its center
//   groundRay.ts  one screen point -> one point on the floor plane
//   ribbon.ts     a polyline with real world-space width, flat on the ground
//   roadMesh.ts   a confirmed road's paved-slab geometry
//   roads.ts      road-mode drawing: blueprints, snapping, confirming
//   interactable.ts  the shared "pick and highlight" shape denizens/
//                 landmarks/trees answer, act.ts's query runs on
//   act.ts        act-mode picking: select a denizen, send it walking to a
//                 nearby structure or tree -- one per denizen per turn.ts,
//                 with a "+1 <resource>" popup on hover
//   actions.ts    what happens on arrival: a plain walk just marks the mover
//                 spent, a tree or structure gets harvested first with a tool
//                 matched to its resource (temporary axe/pickaxe/scythe/wand)
//   inventory.ts  each court's own wood/metal/string/aether, and the panel
//                 that shows the current turn's
//   turns.ts      whose turn it is, counter-clockwise; gates act.ts's pick
//                 and repaints the UI that used to be permanently red
//   weapons.ts    the weapon registry + models; equipping into a hand
//   combat.ts     what attacks do: blade hitbox, arrows, the staff's cone
//   health.ts     hit points, the floating bar, dying
//   aim.ts        the reticle: where you point, and the pose that follows.
//                 Still only ever the red king (see denizens.ts's own `you`)
//   frame.ts      the shared yaw/forward/right convention, in one place
//   motion.ts     velocity integration: accelerate, coast, arrive
//   ai.ts         fighter behaviour: pick a target, close, attack -- `you`
//                 included, since he's an ordinary denizen now
//   protocol.ts   the game messages: denizenState, attackFx, roadState
//   net.ts        the websocket both hosts share, and the send/receive gate
//   main.ts       this page's boot: claim the input, then loop
//
// This file is the STANDALONE page's entry point specifically. Everything it
// does falls into one of two jobs -- claiming the browser's input devices, and
// running a render loop -- and both are things a second host does differently.
// That is the whole reason none of it lives in sim.ts: the AR overlay renders
// the same world, on someone else's canvas, through a camera it doesn't
// control, and with no claim on the keyboard or the pointer at all.

import * as THREE from 'three';
import { camera, scene } from '../shared/scene.ts';
import { canvas, controls, renderer, resize } from './view.ts';
import { snapshotDenizens, step } from '../shared/sim.ts';
import { wireModeUI } from '../shared/mode.ts';
import { wireRegionDraw } from '../shared/regionDraw.ts';
import { snapshotRoads, wireRoadBuild } from '../shared/roads.ts';
import { updateActIndicator, wireAct } from '../shared/act.ts';
import { wireTurnButton } from '../shared/turns.ts';
import { updateInventoryIcons, wireInventoryPanel } from '../shared/inventory.ts';
import { updateCrosshair, wireAim } from '../shared/aim.ts';
import { wireBattleButton } from '../shared/ai.ts';
import { chargeLevel } from '../shared/combat.ts';
import { attachNetHost, connect, send } from '../shared/net.ts';

const clock = new THREE.Clock();

// This page is the ONE authoritative host (see shared/sim.ts's own header on
// the simulate/render split) -- it never receives a game message, only sends
// them, so its NetHost is a no-op except for logging the unexpected.
attachNetHost({
  onMessage(msg) { console.warn('table-topple-server: unexpected inbound game message', msg); },
});
connect('desktop');

// Every input this page owns, claimed in one place. Previously each of these
// modules attached its own listeners the moment it was imported, which made
// "run the simulation" and "take over the keyboard and pointer" the same
// indivisible act -- fine for the only page that existed then, fatal for a
// host that wants the first without the second.
wireModeUI();
wireAim(canvas);
wireRegionDraw(canvas);
wireRoadBuild(canvas);
wireAct(canvas);
wireBattleButton();
wireTurnButton();
wireInventoryPanel();

function animate() {
  requestAnimationFrame(animate);
  // Capped: a backgrounded tab stops rAF, and the first frame after refocusing
  // would otherwise carry the entire away-time as one step and teleport you.
  const dt = Math.min(clock.getDelta(), 0.1);
  step(dt);
  // Broadcast every tick, not throttled -- see shared/net.ts's own header on
  // why that's the lean choice for now: it's one small JSON array over a LAN
  // websocket, and a smarter send rate is a real optimization to make once
  // there's a reason to (a slow phone, a real network), not before.
  send({ type: 'denizenState', denizens: snapshotDenizens() });
  // Roads change rarely, but this rides the same every-tick broadcast for the
  // reason protocol.ts's own comment gives: it's what makes a late-joining
  // phone catch up on every already-built road with no separate mechanism.
  send({ type: 'roadState', roads: snapshotRoads() });
  controls.update(); // required: damping is on
  updateCrosshair(chargeLevel());
  updateActIndicator();
  updateInventoryIcons(dt);
  renderer.render(scene, camera);
}

resize();
animate();
