import * as THREE from 'three';

// The world, and the eye it is seen through -- deliberately free of the DOM,
// the renderer and the controls, which all live in view.ts instead.
//
// The split exists because this game now has TWO hosts. The standalone page
// (game.html -> main.ts -> view.ts) owns a canvas, orbits a camera with the
// mouse and fills the screen. The AR overlay on the capture page owns neither:
// it draws this same world into a transparent canvas laid over the camera
// feed, through a camera whose pose comes from the reconstruction pipeline.
// Everything that BUILDS or ANIMATES the world imports this module, so none of
// it can accidentally take a dependency on being drawn to a page of its own.

export const scene = new THREE.Scene();

// No scene.background here on purpose: an opaque clear colour is a decision
// only a full-screen host gets to make. view.ts sets one; the AR overlay
// leaves it null so the camera feed shows through.
scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.2));

const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(5, 10, 3);
scene.add(sun);

/**
 * The camera the world is currently being seen through -- shared, and MUTATED
 * by whichever host is presenting: view.ts hands it to the orbit controls, the
 * AR overlay writes a recovered pose into it every frame.
 *
 * One shared object rather than a camera per host, because code that isn't
 * about presentation at all still needs "whatever is looking" -- the health
 * bars billboard to it (health.ts), the ground raycasts unproject through it
 * (aim.ts, regionDraw.ts). Reading one object keeps all of them ignorant of
 * which host is driving it.
 *
 * fov/aspect/near/far are only a starting point; both hosts overwrite them
 * (view.ts's resize, the overlay's per-fix intrinsics).
 */
export const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
