// Board game sandbox -- deliberately isolated from the De Bruijn/pose
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
//   landmarks.ts  one structure per patch: shrubbery, mine, ruined church
//   forest.ts     trees scattered over the forest terrain, instanced
//   blocks.ts     the shared box-assembly kit scenery is built from
//   noise.ts      deterministic value noise + the seeded RNG the world uses
//   character.ts  the blocky humanoid model, built from boxes
//   ranks.ts      the denizen hierarchy: size, shading, crown
//   denizens.ts   the four courts: figures with a rank, position, and facing
//   castle.ts     each court's castle, from a tower/wall plan
//   animation.ts  the walk cycle + turn easing, driven by distance walked
//   mode.ts       who owns the mouse: camera / path / fight, + weapon choice
//   regionDraw.ts path-mode drawing: trace a region, walk to its center
//   ribbon.ts     a polyline with real world-space width, flat on the ground
//   weapons.ts    the weapon registry + models; equipping into a hand
//   combat.ts     what attacks do: blade hitbox, arrows, the staff's cone
//   health.ts     hit points, the floating bar, dying
//   aim.ts        the reticle: where you point, and the pose that follows
//   frame.ts      the shared yaw/forward/right convention, in one place
//   motion.ts     velocity integration: accelerate, coast, arrive
//   ai.ts         fighter behaviour: pick a target, close, attack
//   input.ts      held-key state -> strafe/forward axes, + the camera's yaw
//   main.ts       this page's boot: claim the input, then loop
//
// This file is the STANDALONE page's entry point specifically. Everything it
// does falls into one of two jobs -- claiming the browser's input devices, and
// running a render loop -- and both are things a second host does differently.
// That is the whole reason none of it lives in sim.ts: the AR overlay renders
// the same world, on someone else's canvas, through a camera it doesn't
// control, and with no claim on the keyboard or the pointer at all.

import * as THREE from 'three';
import { camera, scene } from './scene.ts';
import { canvas, controls, renderer, resize } from './view.ts';
import { step } from './sim.ts';
import { wireKeys } from './input.ts';
import { wireModeUI } from './mode.ts';
import { wireRegionDraw } from './regionDraw.ts';
import { updateCrosshair, wireAim } from './aim.ts';
import { wireBattleButton } from './ai.ts';
import { chargeLevel } from './combat.ts';

const clock = new THREE.Clock();

// Every input this page owns, claimed in one place. Previously each of these
// modules attached its own listeners the moment it was imported, which made
// "run the simulation" and "take over the keyboard and pointer" the same
// indivisible act -- fine for the only page that existed then, fatal for a
// host that wants the first without the second.
wireKeys();
wireModeUI();
wireAim(canvas);
wireRegionDraw(canvas);
wireBattleButton();

function animate() {
  requestAnimationFrame(animate);
  // Capped: a backgrounded tab stops rAF, and the first frame after refocusing
  // would otherwise carry the entire away-time as one step and teleport you.
  step(Math.min(clock.getDelta(), 0.1));
  controls.update(); // required: damping is on
  updateCrosshair(chargeLevel());
  renderer.render(scene, camera);
}

resize();
animate();
