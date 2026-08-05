import * as THREE from 'three';
import { camera } from './scene.ts';
import { yawFromDirection } from './frame.ts';

// Held-key set rather than acting on the keydown event itself: keydown
// auto-repeats at the OS's repeat rate (and only for the most recent key), so
// reading it directly gives lurchy, un-diagonal movement. Held state sampled
// once per frame gives smooth motion and free diagonals.
const held = new Set<string>();

/**
 * Starts listening for the movement keys. Called by the standalone page's
 * boot, and by nothing else: a host that isn't the game's own page has its own
 * claim on the keyboard, and until this runs the set below simply stays empty,
 * so moveAxes reports no input rather than needing a mode flag.
 */
export function wireKeys() {
  addEventListener('keydown', (e) => {
    // Leave browser/OS chords alone -- Cmd+R must still reload rather than
    // registering a held 'r'.
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    held.add(e.code);
  });
  addEventListener('keyup', (e) => held.delete(e.code));
  // A window that loses focus never delivers the keyup, so the key would stay
  // "held" forever and the player would walk into the wall while you're in
  // another tab.
  addEventListener('blur', () => held.clear());
}

const forward = new THREE.Vector3();

/**
 * WASD as intent, in nobody's frame in particular: `x` is strafe (+ is right,
 * D), `y` is forward/back (+ is W). The caller picks what "forward" means by
 * supplying a basis yaw -- see basisToWorld().
 *
 * Deliberately NOT resolved to a world direction here. There are two different
 * bases in play: unarmed movement is camera-relative (W goes away from the
 * camera), while an armed character walks in its OWN frame so the sword keeps
 * pointing where it points. Both are the same input with a different basis, so
 * the input shouldn't know which one is in force.
 *
 * Returns a unit vector, or zero length if nothing is held. Diagonals are
 * normalized, so holding W+D is not faster than W alone.
 */
export function moveAxes(out: THREE.Vector2): THREE.Vector2 {
  out.set(0, 0);
  if (held.has('KeyW')) out.y += 1;
  if (held.has('KeyS')) out.y -= 1;
  if (held.has('KeyD')) out.x += 1;
  if (held.has('KeyA')) out.x -= 1;
  if (out.lengthSq() > 0) out.normalize();
  return out;
}

/** The camera's heading, flattened onto the ground, in the shared yaw
 *  convention (frame.ts: 0 faces +Z). */
export function cameraYaw(): number {
  camera.getWorldDirection(forward);
  forward.y = 0;
  // Degenerate only if the camera looks straight down, which the orbit
  // controls' maxPolarAngle already prevents.
  if (forward.lengthSq() < 1e-8) return 0;
  return yawFromDirection(forward.x, forward.z);
}
