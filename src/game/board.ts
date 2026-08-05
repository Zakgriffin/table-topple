import * as THREE from 'three';
import { scene } from './scene.ts';
import { BOARD_CELLS, BOARD_SIZE, COLOR_FLOOR, COLOR_GRID, COLOR_GRID_CENTER } from './constants.ts';

// The playing surface: a plain lit plane with a THREE.GridHelper sitting just
// above it. No De Bruijn pattern here -- that belongs to the tracking work,
// and this page is a clean room for game mechanics.
export const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(BOARD_SIZE, BOARD_SIZE),
  new THREE.MeshStandardMaterial({ color: COLOR_FLOOR, roughness: 0.95, metalness: 0 }),
);
floor.rotation.x = -Math.PI / 2;
scene.add(floor);

export const grid = new THREE.GridHelper(BOARD_SIZE, BOARD_CELLS, COLOR_GRID_CENTER, COLOR_GRID);
// Lift the lines off the plane by a hair: coplanar geometry z-fights, and the
// grid loses (it's drawn as lines, so the artifacts stipple rather than blotch).
grid.position.y = 0.002;
scene.add(grid);
