import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { BOARD_SIZE, COLOR_BG, SOLDIER_HEIGHT, START_RADIUS } from './constants.ts';

export const canvas = document.getElementById('gl') as HTMLCanvasElement;

export const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

export const scene = new THREE.Scene();
scene.background = new THREE.Color(COLOR_BG);
scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.2));

const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(5, 10, 3);
scene.add(sun);

export const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 500);

// The opening view is over the red king's shoulder, NOT a framing of the
// whole board. The board is 144 units across while a character is 1 unit tall,
// so a full-board framing renders everyone as a speck -- the size of the arena
// is something to discover by walking it, not by being shown all of it at
// once. Red starts at -Z looking toward the middle (denizens.ts), so the camera
// sits further back along -Z and looks along their line of sight.
const RED_SEAT_Z = -START_RADIUS;
camera.position.set(0, 9, RED_SEAT_Z - 13);

export const controls = new OrbitControls(camera, canvas);
controls.target.set(0, SOLDIER_HEIGHT, RED_SEAT_Z);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
// Stop the orbit dipping under the board -- looking up through the floor
// from below is never a view this game wants.
controls.maxPolarAngle = Math.PI / 2 - 0.02;
controls.minDistance = 3;
// Far enough out to take in the whole board when you want to, since the
// default view deliberately doesn't.
controls.maxDistance = BOARD_SIZE * 1.5;
controls.update();

export function resize() {
  // updateStyle left at its default (true) on purpose: the canvas has no CSS
  // size of its own, so it needs setSize to write one. Passing false sets only
  // the drawing buffer (window * devicePixelRatio) and the canvas then lays
  // out at that raw pixel count -- 2x the viewport on a retina display,
  // anchored top-left, with the board pushed off the bottom-right corner.
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}

addEventListener('resize', resize);
