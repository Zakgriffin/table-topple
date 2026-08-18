import * as THREE from 'three';

// The one place the game's heading convention is written down. Everything with
// a facing -- characters, the camera basis, the direction a sword points --
// agrees on this, and nothing else should be re-deriving it inline:
//
//   A yaw of 0 faces +Z. Increasing yaw turns toward +X.
//   At yaw t:  forward = ( sin t,  cos t)      right = (-cos t,  sin t)
//
// Vector2 here is always ground-plane (x, z), never (x, y) -- the game is 2D
// on the floor and the third dimension is only ever the character's height.
//
// The forward definition is what THREE.Object3D.rotation.y already does (it
// sends +Z to (sin, cos)), so a character's `facing` can be assigned straight
// to rotation.y. The right definition follows from the right-hand rule with +Y
// up, which is why a character facing +Z has its right hand toward -X -- the
// same handedness character.ts builds the body with.

/** Direction a yaw faces. */
export function forwardFromYaw(yaw: number, out: THREE.Vector2): THREE.Vector2 {
  return out.set(Math.sin(yaw), Math.cos(yaw));
}

/** The right-hand side of a yaw. */
export function rightFromYaw(yaw: number, out: THREE.Vector2): THREE.Vector2 {
  return out.set(-Math.cos(yaw), Math.sin(yaw));
}

/** Yaw that faces along a ground direction. The inverse of forwardFromYaw --
 *  note the argument order is (x, z), not the usual atan2(y, x). */
export function yawFromDirection(x: number, z: number): number {
  return Math.atan2(x, z);
}

/** Component of a movement along a yaw's forward direction. Negative means it
 *  travelled backwards relative to that heading. */
export function signedAlongYaw(dx: number, dz: number, yaw: number): number {
  return dx * Math.sin(yaw) + dz * Math.cos(yaw);
}
