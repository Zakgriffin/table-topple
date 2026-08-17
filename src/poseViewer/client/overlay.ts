// ── The AR overlay: colored cubes standing on the printed board ──────────
//
// A transparent THREE canvas laid exactly over the viewfinder, seen through a
// camera at the pose this device's own reconstruction recovered. When the decode
// is good, the cubes land on the printed De Bruijn pattern in the real feed
// underneath them and stay put as the phone moves.
//
// ── WHY CUBES, AND NOT THE BOARD GAME ────────────────────────────────────
//
// This page USED to draw Table Topple's board here. That overlay left with the
// four-page restructure -- it is Table Topple's, it lives on
// table-topple-client.html, and the isolation rule is that these two projects
// import nothing from one another. Reaching back across for it is exactly what
// the restructure exists to prevent, so this is a fresh overlay that owns its
// own scene and shares no code with that one.
//
// Cubes rather than a re-implementation of a game: what Pose Viewer wants from
// an overlay is a POSE-CORRECTNESS READOUT. A lattice of cubes on known cells is
// the most direct one there is -- each cube either sits on its cell or it does
// not, the error is visible as a lateral offset in cells rather than as a vague
// wrongness, and the colour gradient makes a 90-degree orientation error
// obvious at a glance (the decode has four cardinal candidates, and picking the
// wrong one is a real failure mode this shows immediately).
//
// It is deliberately NOT a re-creation of the old blue rectangle + red marker
// cube either. That pair was replaced for a reason: it showed WHERE the camera
// thought the floor was without showing whether the SCALE and the CELL PHASE
// were right, which is most of what can go wrong.
//
// ── WHAT THIS MODULE IS NOT ──────────────────────────────────────────────
//
// It owns no simulation, no input, no clock. It is handed a camera placement and
// told to draw. That keeps it callable from the display loop at screen rate
// while pose fixes arrive at reconstruction rate -- the two are decoupled on
// purpose, and IMU prediction writes the camera between fixes (see main.ts).

import * as THREE from 'three';
import { GRID_STEP } from '../shared/constants.ts';
import { HALF_C, HALF_R } from '../shared/floorPattern.ts';
import { arCanvas } from './dom.ts';

// ── Layout, in cells ─────────────────────────────────────────────────────
//
// Both in CELLS and multiplied by GRID_STEP at use, so the lattice stays aligned
// to cell boundaries whatever GRID_STEP is -- the whole point is that a cube
// lands on an identifiable cell.
//
// The spacing is a compromise the picture makes visible: too fine and the cubes
// occlude the pattern the decode needs a human to be able to check against; too
// coarse and there are not enough of them across the frame to see a rotation.
// 16 cells gives ~9 per axis on a 144-cell board.
const CUBE_SPACING_CELLS = 16;
const CUBE_SIZE_CELLS = 3;

const canvas = arCanvas;

// alpha + a zero clear alpha: the camera feed underneath is the entire point,
// so this renderer must never paint a background. `premultipliedAlpha` is left
// at its default -- the compositor path that matters here is the browser's own
// canvas stacking, not a texture read.
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setClearColor(0x000000, 0);
// Capped at 2: this page is already CPU/GPU constrained by continuous pose
// recovery, and a 3x phone display would triple the fill cost of an overlay
// whose whole job is to be looked at, not admired.
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

const scene = new THREE.Scene();

// FOV and aspect are both overwritten by the first pose (they are properties of
// the lens, and the reconstruction is what knows them) -- these are placeholders
// that only ever matter for the frames before the first fix, which are not
// drawn. `near` is pulled in tight because the phone is held over the board and
// can easily be closer than a default 0.1 to the nearest cube.
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 4000);

// Lit rather than flat-shaded: an unlit cube is a coloured silhouette, and a
// silhouette over a busy black-and-white pattern is genuinely hard to read as a
// solid standing on a surface. Two cheap lights buy the shading that makes the
// cubes look like they are ON the board rather than floating over the picture.
scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(0.4, 1, 0.25);
scene.add(sun);

// ── The lattice ──────────────────────────────────────────────────────────
//
// ONE InstancedMesh rather than N meshes: this is a phone competing with a
// continuous reconstruction, and ~81 individual draw calls per frame is a cost
// with nothing to show for it.
//
// Positions are computed in WORLD SPACE directly, symmetric about the origin,
// rather than by mapping cell (row,col) through the floor texture's own UV
// convention. That is deliberate: the texture's flipY behaviour decides which
// world edge row 0 lands on, and getting it wrong would put every cube half a
// board away while still looking like a plausible lattice. A symmetric lattice
// on cell-aligned world coordinates needs none of that -- it is correct under
// either convention, and the thing being checked (do cubes sit on cells, at the
// right scale, at the right rotation) is unaffected.
let cubes: THREE.InstancedMesh | null = null;

const cubeGeo = new THREE.BoxGeometry(
  CUBE_SIZE_CELLS * GRID_STEP, CUBE_SIZE_CELLS * GRID_STEP, CUBE_SIZE_CELLS * GRID_STEP,
);
const cubeMat = new THREE.MeshLambertMaterial();

// A taller, white post at the world origin -- which is where the pose pipeline
// puts the centre of the board. Worth its own object because "is the origin
// where I think it is" is a different question from "is the scale right", and
// the lattice alone cannot answer it: a lattice offset by exactly one spacing
// looks identical to a correct one.
const originPost = new THREE.Mesh(
  new THREE.BoxGeometry(GRID_STEP, CUBE_SIZE_CELLS * 3 * GRID_STEP, GRID_STEP),
  new THREE.MeshLambertMaterial({ color: 0xffffff }),
);
originPost.position.y = (CUBE_SIZE_CELLS * 3 * GRID_STEP) / 2;
scene.add(originPost);

/**
 * Builds (or rebuilds) the cube lattice for the CURRENT board size.
 *
 * Called at boot and again whenever the board is rebuilt -- the desktop's
 * board-size slider changes HALF_C/HALF_R, and a lattice sized for the old board
 * would run off the edge of the new one. HALF_C/HALF_R are live ESM bindings, so
 * reading them here always sees the current values.
 */
export function rebuildCubes(): void {
  if (cubes) {
    scene.remove(cubes);
    cubes.dispose();
    cubes = null;
  }
  const spacing = CUBE_SPACING_CELLS * GRID_STEP;
  // How many whole spacings fit inside each half-extent. The lattice is then
  // symmetric (-n..n), which is what keeps it centred on the origin regardless
  // of whether the board has an odd or even number of cells.
  const nx = Math.floor(HALF_C / spacing);
  const nz = Math.floor(HALF_R / spacing);
  const count = (2 * nx + 1) * (2 * nz + 1);
  const mesh = new THREE.InstancedMesh(cubeGeo, cubeMat, count);
  const m = new THREE.Matrix4();
  const color = new THREE.Color();
  let i = 0;
  for (let ix = -nx; ix <= nx; ix++) {
    for (let iz = -nz; iz <= nz; iz++) {
      m.makeTranslation(
        ix * spacing,
        // Sitting ON the floor, not centred in it: the board is the y=0 plane,
        // and a cube half-sunk into the table reads as a pose error.
        (CUBE_SIZE_CELLS * GRID_STEP) / 2,
        iz * spacing,
      );
      mesh.setMatrixAt(i, m);
      // A 2D gradient, not one hue per cube: hue runs along X and lightness
      // along Z, so the lattice as a whole encodes its own orientation. Under a
      // 90-degree decode error the gradient visibly runs the wrong way, which a
      // single-hue or randomly-coloured lattice would hide completely.
      const u = nx > 0 ? (ix + nx) / (2 * nx) : 0.5;
      const v = nz > 0 ? (iz + nz) / (2 * nz) : 0.5;
      color.setHSL(u * 0.85, 0.75, 0.35 + v * 0.3);
      mesh.setColorAt(i, color);
      i++;
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  scene.add(mesh);
  cubes = mesh;
}
rebuildCubes();

// ── Visibility ───────────────────────────────────────────────────────────
//
// The canvas' own class, not a scene-graph flag: while the overlay is off this
// page should pay NOTHING for it, and a hidden canvas that is still being
// rendered into costs exactly as much as a visible one. `renderOverlay` returns
// early on the same flag.
let enabled = false;
export function setOverlayEnabled(on: boolean): void {
  enabled = on;
  canvas.classList.toggle('visible', on);
}
export function isOverlayEnabled(): boolean {
  return enabled;
}

// ── Pose ─────────────────────────────────────────────────────────────────

/** A camera placement to draw the cubes from. Produced by client/pose.ts's
 *  toOverlayCamera from a local reconstruction, or by main.ts's IMU prediction
 *  between fixes -- the two are the same shape on purpose, so this module never
 *  learns which one it is looking at. */
export interface ARCameraPose {
  camPos: THREE.Vector3; camQuat: THREE.Quaternion; aspect: number; fovDeg: number;
}

/** No fix means nothing is drawn at all, rather than the cubes being left at a
 *  stale pose. A frozen overlay silently claims to still know where the camera
 *  is; a blank one is the honest failure, and on this page in particular the
 *  overlay IS the measurement. */
let hasPose = false;

/** Only ever moves the CAMERA -- the cubes are bolted to the world origin, where
 *  the pose pipeline's own reconstruction puts the board. */
export function updateOverlayCamera(pose: ARCameraPose | null): void {
  hasPose = !!pose;
  if (!pose) return;
  camera.position.copy(pose.camPos);
  camera.quaternion.copy(pose.camQuat);
  camera.fov = pose.fovDeg;
  camera.aspect = pose.aspect;
  camera.updateProjectionMatrix();
}

// Mirrors the viewfinder's own INTRINSIC (cw,ch), not its rendered CSS box --
// both canvases carry the identical max-width/max-height letterbox fit (see
// pose-viewer-client.html), so matching intrinsic dimensions is what keeps this
// canvas scaled and positioned exactly over the one underneath. `false` skips
// three.js's own inline-style sizing, since the stylesheet already owns display
// sizing for both canvases identically. Only touches the GL backing store when
// (cw,ch) genuinely changed, since this is called every frame.
let sizedCw = 0, sizedCh = 0;
export function syncOverlayRendererSize(cw: number, ch: number): void {
  if (cw === sizedCw && ch === sizedCh) return;
  sizedCw = cw; sizedCh = ch;
  if (cw > 0 && ch > 0) renderer.setSize(cw, ch, false);
}

/**
 * One frame.
 *
 * Skipped entirely while the overlay is off or there is no fix -- `clear()`
 * rather than `render()` on the no-fix path, so a lost decode blanks what is
 * already on the canvas instead of leaving the last good frame painted there.
 */
export function renderOverlay(): void {
  if (!enabled) return;
  if (!hasPose) { renderer.clear(); return; }
  renderer.render(scene, camera);
}
