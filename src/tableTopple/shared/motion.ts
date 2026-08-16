import * as THREE from 'three';

// Velocity integration for a character on the board. Pure: takes state in,
// mutates it, touches no scene graph and no DOM -- so the feel of stopping can
// be simulated and measured instead of only eyeballed in the browser.

/** Below this speed (world units/sec) a coasting character is simply stopped.
 *  An exponential decay never actually reaches zero, so without a floor the
 *  character keeps creeping by immeasurable amounts forever -- and the walk
 *  cycle, which is driven by distance travelled, never quite settles. */
const STOP_SPEED = 0.02;

/**
 * Speed to approach a destination at, so that an exponential decel lands ON it
 * rather than overshooting and getting dragged back.
 *
 * Braking from speed v at rate k covers exactly v/k before stopping, so the
 * speed that stops in `remaining` is remaining*k. Capping that at maxSpeed
 * means the character runs flat out until it's within braking distance, then
 * eases in. This is why arriving at a drawn region glides to a halt instead of
 * snapping, with no oscillation around the target.
 */
export function arrivalSpeed(remaining: number, maxSpeed: number, decelRate: number): number {
  return Math.min(maxSpeed, remaining * decelRate);
}

/**
 * Advances velocity toward `desired`, then position by velocity, clamped to a
 * square board of half-extent `limit`.
 *
 * @param rate exponential rate for the velocity change -- the caller passes
 *   the acceleration rate when there's input and the deceleration rate when
 *   coasting, which is what makes starts crisp and stops heavy.
 */
export function advance(
  pos: THREE.Vector2,
  velocity: THREE.Vector2,
  desired: THREE.Vector2,
  rate: number,
  dt: number,
  limit: number,
) {
  velocity.x = THREE.MathUtils.damp(velocity.x, desired.x, rate, dt);
  velocity.y = THREE.MathUtils.damp(velocity.y, desired.y, rate, dt);

  if (desired.lengthSq() === 0 && velocity.lengthSq() < STOP_SPEED * STOP_SPEED) {
    velocity.set(0, 0);
  }

  pos.x = THREE.MathUtils.clamp(pos.x + velocity.x * dt, -limit, limit);
  pos.y = THREE.MathUtils.clamp(pos.y + velocity.y * dt, -limit, limit);

  // Hitting the board edge kills the velocity INTO the edge, rather than
  // letting it keep accumulating against a clamped position -- otherwise
  // holding a key against the wall banks up speed that fires the character off
  // the instant it's free to move again.
  if (Math.abs(pos.x) >= limit) velocity.x = 0;
  if (Math.abs(pos.y) >= limit) velocity.y = 0;
}
