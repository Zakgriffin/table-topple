import * as THREE from 'three';
import { CHARACTER_LIMB_LENGTH } from './character.ts';
import type { Denizen } from './denizens.ts';

// ── Walk cycle ────────────────────────────────────────────────────────────
// Rigid-limb swing: legs counter-swing, arms counter-swing the legs, body bobs
// twice per cycle. No knees or elbows -- the limbs are single boxes, and a
// blocky character reads fine without them.

/** Peak swing of a leg away from vertical, radians (~34 degrees). */
const MAX_SWING = 0.6;
/** Arms swing less than legs -- matched amplitudes look like marching. */
const ARM_SWING_RATIO = 0.75;

// Ground distance covered by one FULL cycle (left step + right step) BY A
// SOLDIER. Derived from the leg's own geometry rather than picked by eye: a
// leg of length L swinging +/-A plants its foot 2*L*sin(A) apart, and a cycle
// is two such steps. Get this wrong and the feet visibly skate -- the
// character slides along the floor like it's on ice (too long) or scurries
// (too short).
//
// Both this and the bob below are scaled per denizen by its rank's size. A
// king's legs really are twice as long, so he covers twice the ground per
// stride: at the same walking speed he takes half as many, which is exactly
// how a bigger figure should move. Leaving these unscaled would have his feet
// skating at 2x while a soldier's stayed planted.
const SOLDIER_STRIDE = 4 * CHARACTER_LIMB_LENGTH * Math.sin(MAX_SWING);

/** Vertical bob at full stride for a soldier, world units. Small on purpose:
 *  the body rises when a leg is planted and vertical, so this is a fraction of
 *  the "missing" height L*(1-cos(A)) that a real swinging leg would give up. */
const SOLDIER_BOB = CHARACTER_LIMB_LENGTH * (1 - Math.cos(MAX_SWING)) * 0.6;

/** How fast walkBlend chases its target. ~12 gives a couple of frames of
 *  ease-in at 60fps: enough to not snap, fast enough to feel responsive. */
const BLEND_RATE = 12;

/**
 * Advances and applies one denizen's walk cycle.
 *
 * @param distance SIGNED ground distance the denizen actually moved this frame
 *   (after clamping -- walking into a wall covers zero distance, so the legs
 *   correctly stop rather than treadmilling in place). Negative means the
 *   character travelled backwards relative to its own facing, which runs the
 *   cycle in reverse -- that's what makes an armed backstep read as backing
 *   up rather than marching forward while sliding the wrong way.
 */
export function updateWalk(p: Denizen, distance: number, dt: number) {
  // Phase tracks DISTANCE, not time. That's what keeps the feet locked to the
  // ground at any speed: if the character is moved twice as fast, the legs
  // cycle twice as fast for free, with no speed-to-cadence constant to tune.
  p.walkPhase += (distance / (SOLDIER_STRIDE * p.scale)) * Math.PI * 2;

  // Magnitude, since "is it moving at all" doesn't care about direction.
  const target = Math.abs(distance) > 1e-6 ? 1 : 0;
  p.walkBlend = THREE.MathUtils.damp(p.walkBlend, target, BLEND_RATE, dt);

  // Below a pixel's worth of swing, settle exactly to the rest pose. Without
  // this the exponential ease never quite reaches 0 and the character keeps a
  // permanent, imperceptible-but-real twitch in its limbs forever.
  if (p.walkBlend < 0.001) {
    p.walkBlend = 0;
    // Park the phase at 0 too, so the next walk always starts from a neutral
    // stance rather than mid-stride.
    p.walkPhase = 0;
  }

  const swing = Math.sin(p.walkPhase) * MAX_SWING * p.walkBlend;
  const { leftArm, rightArm, leftLeg, rightLeg, group } = p.character;

  leftLeg.rotation.x = swing;
  rightLeg.rotation.x = -swing;
  // Opposite the leg on the same side: left arm forward with RIGHT leg. That
  // contralateral swing is what makes a walk look like a walk.
  leftArm.rotation.x = -swing * ARM_SWING_RATIO;
  rightArm.rotation.x = swing * ARM_SWING_RATIO;

  // Twice per cycle (once per step), and always upward -- abs(), not a raw
  // sine, because the body rises over each planted leg and never sinks below
  // its standing height. Scaled by rank: position.y is in world units (the
  // group's own scale doesn't apply to its position), so a king needs a
  // king-sized bob written here explicitly.
  group.position.y = Math.abs(Math.sin(p.walkPhase)) * SOLDIER_BOB * p.scale * p.walkBlend;
}

// ── Turning ───────────────────────────────────────────────────────────────

/** How fast facing chases the direction of travel. */
const TURN_RATE = 14;

/**
 * Eases a denizen's facing toward `target` (radians about +Y), taking the
 * short way around. Snapping straight to the target made a 180 (tapping S
 * while walking north) flip the character inside out in one frame.
 */
export function turnToward(p: Denizen, target: number, dt: number) {
  // Wrap the difference into (-PI, PI] first: without this, turning from
  // +179deg to -179deg eases the LONG way around, a full 358deg spin for what
  // should be a 2deg correction.
  let delta = target - p.facing;
  delta = Math.atan2(Math.sin(delta), Math.cos(delta));
  p.facing += delta * (1 - Math.exp(-TURN_RATE * dt));
}
