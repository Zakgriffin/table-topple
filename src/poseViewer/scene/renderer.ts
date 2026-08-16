import * as THREE from 'three';
import { canvas } from '../ui/dom.ts';

export const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.autoClear = false;

export const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);
scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.2));
export const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(5, 10, 3);
scene.add(sun);
