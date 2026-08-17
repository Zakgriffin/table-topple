import * as THREE from 'three';
import { camera } from './scene.ts';

// The one place a screen point becomes a point on the board's ground plane.
// regionDraw.ts (path mode) and roads.ts (road mode) both draw by dragging
// across the floor, and both need exactly this -- split out rather than left
// as regionDraw.ts's own private copy once a second caller wanted it.
//
// The floor is raycast as an infinite mathematical plane rather than against
// the floor Mesh: a drag that strays past the board's edge still yields a
// point instead of a gap in the shape, and it costs no triangle tests.

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const hit = new THREE.Vector3();

export function pointerToGround(e: PointerEvent, out: THREE.Vector2): boolean {
  ndc.x = (e.clientX / innerWidth) * 2 - 1;
  ndc.y = -(e.clientY / innerHeight) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  if (!raycaster.ray.intersectPlane(groundPlane, hit)) return false; // ray parallel to the floor
  out.set(hit.x, hit.z);
  return true;
}
