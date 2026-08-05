import * as THREE from 'three';
import {
  BOARD_SIZE, WALK_ACCEL_RATE, WALK_DECEL_RATE, WALK_SPEED,
} from './constants.ts';
import { advance, arrivalSpeed } from './motion.ts';
import { CHARACTER_HALF_WIDTH } from './character.ts';
import { denizens, syncMesh, you } from './denizens.ts';
import { turnToward, updateWalk } from './animation.ts';
import { cameraYaw, moveAxes } from './input.ts';
import { basisToWorld, signedAlongYaw, yawFromDirection } from './frame.ts';
import { clearRegion } from './regionDraw.ts';
import { isEngaged, updateAim } from './aim.ts';
import { retire, updateVitals } from './health.ts';
import { updateAI } from './ai.ts';
import { updateCombat } from './combat.ts';
import './board.ts'; // side effect: adds the floor to the scene

// One frame of the world, and nothing about drawing it.
//
// This is the half of the old main.ts that both hosts share: the standalone
// page calls it from its own rAF loop before rendering, and so does the AR
// overlay on the capture page. Deliberately no renderer, no controls.update(),
// no crosshair -- those are decisions about a viewport, and step() has none.
//
// Note that importing this module is also what BUILDS the world: the floor
// (board.ts) and the four courts (denizens.ts) add themselves to the scene as
// an import side effect, so a host that wants the board without simulating it
// still starts here and simply never calls step().

const axes = new THREE.Vector2();
const dir = new THREE.Vector2();
/** Velocity the character is currently trying to have. The gap between this
 *  and its actual velocity is what acceleration and coasting live in. */
const desired = new THREE.Vector2();

/** How far the controlled denizen's center can get from the board's center
 *  before its edge would overhang. Scales with rank -- a king is twice as wide
 *  as a soldier -- so it can't be a constant. */
const limit = () => BOARD_SIZE / 2 - CHARACTER_HALF_WIDTH * you.scale;

/** Distance at which a walk-to is called done. The approach already tapers to
 *  a crawl by this point (arrivalSpeed), so closing the last sliver instantly
 *  isn't visible -- it just stops the target being chased forever. */
const ARRIVE_EPSILON = 0.03;

export function step(dt: number) {
  moveAxes(axes);

  // Snapshot every position up front. The walk cycle is driven by distance
  // actually covered, and there are now two things that move denizens -- the
  // human's input below and the brains in ai.ts -- so measuring it centrally
  // beats making each mover report its own displacement.
  for (const p of denizens) p.lastPos.copy(p.pos);

  // Holding the button with the sword out is what puts the character in a
  // fighting stance: the body stops steering and holds its heading so the
  // blade stays trained downrange, and WASD becomes footwork in the
  // character's own frame (W/S advance and back up, A/D sidestep). Let go and
  // it's an ordinary walk again -- camera-relative, turning to follow. Sword
  // merely DRAWN is not enough; it takes a held button.
  const stance = you.weapon !== null && isEngaged();

  // Nothing here writes position directly any more; it all goes through a
  // desired velocity, and advance() below does the integrating. That's what
  // makes releasing the keys coast to a halt rather than cutting out.
  desired.set(0, 0);
  // No input means coasting, which is the slow rate. Anything driving the
  // character uses the brisk one.
  let rate = WALK_DECEL_RATE;

  if (axes.lengthSq() > 0) {
    // The whole difference between the two movement styles: which yaw the
    // input axes are read against.
    basisToWorld(axes, stance ? you.facing : cameraYaw(), dir);

    // Grabbing the keys cancels a walk-to. Manual input should always win
    // outright -- fighting the auto-walk for control would feel broken. The
    // region goes with it: he's no longer heading there.
    if (you.moveTarget) { you.moveTarget = null; clearRegion(); }
    desired.copy(dir).multiplyScalar(WALK_SPEED);
    rate = WALK_ACCEL_RATE;
    if (!stance) turnToward(you, yawFromDirection(dir.x, dir.y), dt);
  } else if (you.moveTarget) {
    const dx = you.moveTarget.x - you.pos.x, dz = you.moveTarget.y - you.pos.y;
    const remaining = Math.hypot(dx, dz);
    if (remaining <= ARRIVE_EPSILON) {
      you.pos.copy(you.moveTarget);
      you.velocity.set(0, 0);
      you.moveTarget = null;
      // Arrived -- this is what finally wipes the region off the floor.
      clearRegion();
    } else {
      dir.set(dx / remaining, dz / remaining);
      // Tapered near the destination so the character eases into it instead of
      // running at full tilt until it snaps.
      desired.copy(dir).multiplyScalar(arrivalSpeed(remaining, WALK_SPEED, WALK_DECEL_RATE));
      rate = WALK_ACCEL_RATE;
      if (!stance) turnToward(you, yawFromDirection(dir.x, dir.y), dt);
    }
  }

  advance(you.pos, you.velocity, desired, rate, dt, limit());

  // The brains move everyone else, through the same motion and animation
  // helpers the human goes through.
  updateAI(dt);

  // Every denizen runs the same per-frame pose work -- this loop never names a
  // specific one, which is why the AI needed no seam cut for it.
  //
  // Backwards, because updateVitals can retire a denizen mid-pass and
  // the loop below splices it out of this very array. Iterating forwards would
  // skip whoever moved into the gap.
  for (let i = denizens.length - 1; i >= 0; i--) {
    const p = denizens[i];
    const dying = p.dyingFor !== null;
    // A dying figure is done walking and done aiming; health.ts owns its
    // transform from here (the shake moves it off its logical position, which
    // syncMesh would immediately undo).
    if (!dying) {
      // Signed, not just a magnitude: a denizen can walk backwards or
      // sideways -- the human in stance, an archer backing off a rusher -- and
      // feeding the walk cycle a negative distance runs the legs in reverse,
      // so a backstep reads as a backstep instead of a forward march sliding
      // the wrong way.
      const dx = p.pos.x - p.lastPos.x, dz = p.pos.y - p.lastPos.y;
      let moved = Math.hypot(dx, dz);
      if (signedAlongYaw(dx, dz, p.facing) < 0) moved = -moved;

      syncMesh(p);
      updateWalk(p, moved, dt);
      // Strictly after updateWalk: an armed denizen's weapon arm is aimed, not
      // swung, so the aim pose overwrites what the walk cycle just wrote to
      // that one limb. Everything else keeps walking normally. Reversing these
      // two lines silently hands the arm back to the walk cycle.
      updateAim(p, dt);
    }
    if (updateVitals(p, dt)) {
      // Retiring is split: health.ts frees the scene objects, and the roster
      // is spliced here, where we already hold the index. That split is what
      // keeps health.ts free of a runtime import back into denizens.ts.
      retire(p);
      denizens.splice(i, 1);
    }
  }

  // After the poses, so the blade's hitbox is tested where it is actually
  // being drawn this frame rather than where it was last frame.
  updateCombat(dt);
}
