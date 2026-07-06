// Great Sphere Lab — a visual testbed, not part of the tracking pipeline.
//
// Point of the exercise: a straight floor line, together with a camera's
// optical center, spans a plane through that center. Intersect that plane
// with the unit sphere centered on the camera and you get a great circle.
// A family of parallel floor lines all share one direction, and that shared
// direction is exactly what a vanishing point *is* — so every great circle
// from one family passes through the same antipodal point pair on the
// sphere. That crossing point pair is the vanishing point, made literally
// visible instead of algebraic. The floor grid here is two orthogonal
// families (world X and world Z), so their two pole pairs sit exactly 90°
// apart on the sphere, always — that's the "orthogonal constraint" the real
// pipeline (src/orthogonalVp.ts) searches for, seen from the inside.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { generateTorus } from './debruijn.ts';
import { toGrayscale } from './decode.ts';
import { jacobiEigenSymmetric, smallestEigenvector } from './linalg.ts';

type Mode = 'world' | 'through' | 'inside';

// ── DOM ──────────────────────────────────────────────────────────────────

const canvas = document.getElementById('gl') as HTMLCanvasElement;
const pipFrame = document.getElementById('pipFrame') as HTMLDivElement;
const pipLabel = document.getElementById('pipLabel') as HTMLDivElement;
const insideHint = document.getElementById('insideHint') as HTMLDivElement;
const readout = document.getElementById('readout') as HTMLDivElement;
const axesReadout = document.getElementById('axesReadout') as HTMLDivElement;
const captureAxesBtn = document.getElementById('captureAxesBtn') as HTMLButtonElement;

const modeBtns: Record<Mode, HTMLButtonElement> = {
  world: document.getElementById('modeWorld') as HTMLButtonElement,
  through: document.getElementById('modeThrough') as HTMLButtonElement,
  inside: document.getElementById('modeInside') as HTMLButtonElement,
};

// Persist every slider/checkbox under one localStorage key so a dev-server
// restart or a revisit doesn't reset the scene back to defaults.
const STORAGE_KEY = 'sphereLab.controls';
let savedControls: Record<string, string> = {};
try { savedControls = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}'); } catch { savedControls = {}; }
function persistControl(id: string, value: string) {
  savedControls[id] = value;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(savedControls));
}

function bindSlider(id: string, onChange: (v: number) => void, fmt: (v: number) => string = (v) => v.toFixed(1)) {
  const input = document.getElementById(id) as HTMLInputElement;
  const val = document.getElementById(id + 'Val') as HTMLSpanElement;
  if (savedControls[id] !== undefined) input.value = savedControls[id];
  const apply = () => { const v = parseFloat(input.value); val.textContent = fmt(v); onChange(v); persistControl(id, input.value); };
  input.addEventListener('input', apply);
  apply();
}

function bindCheckbox(id: string, onChange: (v: boolean) => void) {
  const input = document.getElementById(id) as HTMLInputElement;
  if (savedControls[id] !== undefined) input.checked = savedControls[id] === '1';
  const apply = () => { onChange(input.checked); persistControl(id, input.checked ? '1' : '0'); };
  input.addEventListener('change', apply);
  apply();
}

function bindRadioGroup(name: string, onChange: (v: string) => void) {
  const inputs = Array.from(document.getElementsByName(name)) as HTMLInputElement[];
  if (savedControls[name] !== undefined) {
    for (const inp of inputs) inp.checked = inp.value === savedControls[name];
  }
  const apply = () => {
    const checked = inputs.find((inp) => inp.checked);
    if (!checked) return;
    onChange(checked.value);
    persistControl(name, checked.value);
  };
  for (const inp of inputs) inp.addEventListener('change', apply);
  apply();
}

// ── State ────────────────────────────────────────────────────────────────

const state = {
  camX: 0, camY: 4, camZ: 8,
  camYawDeg: 0, camPitchDeg: -20,
  focalMM: 26,
  mode: 'world' as Mode,
  showSphere: true, showCircles: true, showPoles: true, showFrustum: true, showPatch: true, showFloor: true, showGizmoBody: true,
  simNoise: 8, simBlur: 1, simGradRadius: 1, simMinMag: 4, coherenceRadius: 1,
  circleSamplePercentMin: 0, circleSamplePercentMax: 5,
  showRecoveredPoles: true,
  showAxisVectors: false,
  showTopCircles: true,
  weightSharpenPower: 4,
  fieldView: 'normal' as 'normal' | 'gradient' | 'cleaned',
  axesAutoCapture: false, axesCaptureIntervalMs: 500,
  viewportW: 512, viewportH: 384, captureSupersample: 2, aspectLocked: false,
};

const SENSOR_WIDTH_MM = 36; // 35mm-equivalent convention, so "focal (mm eq.)" reads like a familiar lens spec
// Derived from the tunable viewport width/height sliders (see resizeCaptureBuffers)
// rather than a fixed constant -- kept as a mutable top-level binding so
// every existing call site (gizmoCam.aspect, cornerDir's aspect param, PIP
// layout, computeWorldVotes) picks up changes without being rewired.
let RT_ASPECT = state.viewportW / state.viewportH;
const SPHERE_RADIUS = 2.5;
const GRID_STEP = 1; // world units per pattern cell
const VIS_HALF_EXTENT = 20; // cap on how many grid lines get a reference line / great circle drawn (perf + clutter, independent of the floor's true size)
const CIRCLE_SEGMENTS = 96;

// ── Scene ────────────────────────────────────────────────────────────────

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.autoClear = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);
scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(5, 10, 3);
scene.add(sun);

// -- Floor: the actual De Bruijn torus, tiled seamlessly (it IS a torus, so
// repeat-wrapping the texture reproduces the true infinite pattern with no
// seam — the same fact the real tracker relies on to work from any crop).
const ORDER = parseInt(new URLSearchParams(location.search).get('order') ?? '4', 10);
const debruijn = generateTorus(ORDER);
const { R, C, torus } = debruijn;
// One instance of the torus, sized in world units at GRID_STEP per cell —
// NOT tiled. Half-extents, since grid lines/great circles below are indexed
// out from the origin at the pattern's center.
const HALF_C = (C * GRID_STEP) / 2;
const HALF_R = (R * GRID_STEP) / 2;

const patternCanvas = document.createElement('canvas');
patternCanvas.width = C; patternCanvas.height = R;
const pctx = patternCanvas.getContext('2d')!;
const img = pctx.createImageData(C, R);
for (let r = 0; r < R; r++) {
  for (let c = 0; c < C; c++) {
    const v = torus[r][c] ? 235 : 20;
    const i = (r * C + c) * 4;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
}
pctx.putImageData(img, 0, 0);

const floorTex = new THREE.CanvasTexture(patternCanvas);
floorTex.wrapS = THREE.RepeatWrapping;
floorTex.wrapT = THREE.RepeatWrapping;
floorTex.magFilter = THREE.NearestFilter;
floorTex.colorSpace = THREE.SRGBColorSpace;
floorTex.repeat.set(1, 1); // exactly one instance of the torus, not tiled

const floorMat = new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.95 });
const floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(C * GRID_STEP, R * GRID_STEP), floorMat);
floorMesh.rotation.x = -Math.PI / 2;
scene.add(floorMesh);

// Colored reference lines at the same integer cell boundaries the great
// circles below are computed from — row family (world +X direction, red)
// and column family (world +Z direction, blue), matching the sphere colors.
// Bounded to the pattern's actual extent, since it's a single instance now.
function buildGridLines(axis: 'row' | 'col', color: number): THREE.LineSegments {
  const half = Math.min(VIS_HALF_EXTENT, axis === 'row' ? HALF_R : HALF_C);
  const cross = axis === 'row' ? HALF_C : HALF_R;
  const pts: number[] = [];
  for (let k = -half; k <= half; k += GRID_STEP) {
    if (axis === 'row') pts.push(-cross, 0.01, k, cross, 0.01, k);
    else pts.push(k, 0.01, -cross, k, 0.01, cross);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 }));
}
const rowGridLines = buildGridLines('row', 0xff5555);
const colGridLines = buildGridLines('col', 0x5599ff);
scene.add(rowGridLines, colGridLines);

// ── Gizmo camera (the camera being studied) ─────────────────────────────

const gizmoCam = new THREE.PerspectiveCamera(50, RT_ASPECT, 0.05, 500);
scene.add(gizmoCam);

const gizmoBody = new THREE.Mesh(
  new THREE.BoxGeometry(0.3, 0.25, 0.4),
  new THREE.MeshStandardMaterial({ color: 0xffcc44 }),
);
scene.add(gizmoBody);
const gizmoAxes = new THREE.AxesHelper(0.6);
gizmoBody.add(gizmoAxes);

const camHelper = new THREE.CameraHelper(gizmoCam);
scene.add(camHelper);

// ── Great-sphere group: repositioned (not rotated) to the gizmo's origin
// each frame, since every direction it draws is expressed in WORLD axes. ──

const sphereAnchor = new THREE.Object3D();
scene.add(sphereAnchor);

const sphereShell = new THREE.Mesh(
  new THREE.SphereGeometry(SPHERE_RADIUS, 48, 32),
  new THREE.MeshBasicMaterial({ color: 0x88aaff, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }),
);
sphereAnchor.add(sphereShell);

const circlesGroup = new THREE.Group();
sphereAnchor.add(circlesGroup);

const polesGroup = new THREE.Group();
sphereAnchor.add(polesGroup);

function makePoleMarker(color: number): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), new THREE.MeshBasicMaterial({ color }));
  polesGroup.add(m);
  return m;
}
const rowPoleA = makePoleMarker(0xff5555);
const rowPoleB = makePoleMarker(0xff5555);
const colPoleA = makePoleMarker(0x5599ff);
const colPoleB = makePoleMarker(0x5599ff);

// Frustum outline: 4 great-circle arcs connecting the camera's 4 corner-ray
// directions, i.e. the actual boundary of "what this camera sees" traced
// onto the sphere.
const frustumLine = new THREE.LineLoop(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 }),
);
sphereAnchor.add(frustumLine);

// Viewport-image patch: literally the camera's rendered pixels, wrapped onto
// the sphere over exactly the solid angle the camera's frustum subtends.
// This is the direct, literal answer to "what does my viewport look like
// mapped onto the great sphere" — no abstraction, the actual image warped
// onto the actual sphere.
const PATCH_RES = 48;
const patchGeo = new THREE.BufferGeometry();
{
  const verts: number[] = [], uvs: number[] = [], idx: number[] = [];
  // uv.y tracks j directly (NOT flipped): a WebGLRenderTarget's texture has
  // no file-style flipY correction, so NDC v=+1 (top of the camera's view,
  // where the position loop below sends j=PATCH_RES) really does land at
  // uv.y=1 (top of the texture) — pairing them inverted put the rendered
  // image on the patch upside-down relative to the ray directions.
  for (let j = 0; j <= PATCH_RES; j++) for (let i = 0; i <= PATCH_RES; i++) { verts.push(0, 0, 0); uvs.push(i / PATCH_RES, j / PATCH_RES); }
  for (let j = 0; j < PATCH_RES; j++) {
    for (let i = 0; i < PATCH_RES; i++) {
      const a = j * (PATCH_RES + 1) + i, b = a + 1, c = a + PATCH_RES + 1, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  }
  patchGeo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  patchGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  patchGeo.setIndex(idx);
}
// final analysis/display resolution -- driven by the viewportW/H sliders
// (camera pose section) rather than a fixed constant; see resizeCaptureBuffers.
let rtSize = { w: state.viewportW, h: state.viewportH };
// Render+blur happen at this multiple of rtSize, THEN get box-downsampled
// to rtSize -- see captureDistortedGrayscale for why (physical lens blur
// acts on a near-continuous image; only the sensor's final discretization
// should introduce the pixel grid, so blurring after already rendering
// small just smooths an aliased/staircased image instead of preventing it).
// The multiple itself (state.captureSupersample) is the "2x2 block" slider
// in the simulated capture distortion section.
let captureRTSize = { w: rtSize.w * state.captureSupersample, h: rtSize.h * state.captureSupersample };
const camRT = new THREE.WebGLRenderTarget(captureRTSize.w, captureRTSize.h, { colorSpace: THREE.SRGBColorSpace });

// What the patch/PIP/Through-Cam views actually display -- camRT's own
// clean render, PLUS the same simulated noise+blur computeWorldVotes
// applies before the algorithm ever sees it. Without this, every "what does
// the camera see" view in the app was showing an idealized frame that
// doesn't match what's actually being analyzed. flipY=false to match
// camRT.texture's own convention exactly (a WebGLRenderTarget's texture is
// never flipY'd), since this buffer is built directly from camRT's own raw
// (unflipped) pixel readback and needs to stay bit-compatible with it for
// the patch mesh's already-validated UV mapping to keep working unchanged.
let distortedPreviewData = new Uint8Array(rtSize.w * rtSize.h * 4);
const distortedPreviewTex = new THREE.DataTexture(distortedPreviewData, rtSize.w, rtSize.h, THREE.RGBAFormat);
distortedPreviewTex.flipY = false;
distortedPreviewTex.colorSpace = THREE.SRGBColorSpace;

// Declared here (rather than down by layoutPip/renderViewport where it's
// otherwise used) because resizeCaptureBuffers below can call layoutPip()
// as early as page-load-time slider binding -- a `let` declared later in
// the module would still be in its temporal dead zone at that point.
let pipRect = { x: 0, y: 0, w: 0, h: 0 };

// Called once at startup (implicitly, via the viewportW/H/captureSupersample
// slider bindings firing on load) and again whenever those sliders change.
// camRT.setSize() resizes the render target in place; distortedPreviewTex
// keeps its own object identity (so previewQuadMat/patchMat, which hold a
// reference to it, don't need to be touched) by having its .image swapped
// for a new {data,width,height} triple -- DataTexture has no other resize path.
function resizeCaptureBuffers() {
  rtSize = { w: Math.round(state.viewportW), h: Math.round(state.viewportH) };
  RT_ASPECT = rtSize.w / rtSize.h;
  captureRTSize = { w: rtSize.w * state.captureSupersample, h: rtSize.h * state.captureSupersample };
  camRT.setSize(captureRTSize.w, captureRTSize.h);
  distortedPreviewData = new Uint8Array(rtSize.w * rtSize.h * 4);
  distortedPreviewTex.image = { data: distortedPreviewData, width: rtSize.w, height: rtSize.h };
  // WebGL2 typically allocates a texture's GPU storage immutably (texStorage2D)
  // on first upload -- just swapping .image to a different size and setting
  // needsUpdate does NOT reallocate that storage, so the GPU-side texture can
  // stay locked at the OLD dimensions while new, differently-shaped data gets
  // uploaded into it, which reads back as stretched/misaligned content with
  // stale GPU memory showing through at the edges. dispose() forces three.js
  // to drop the old GL texture object entirely so the next upload allocates
  // fresh storage at the new size instead of reusing the old one.
  distortedPreviewTex.dispose();
  distortedPreviewTex.needsUpdate = true;
  gizmoCam.aspect = RT_ASPECT;
  gizmoCam.updateProjectionMatrix();
  layoutPip();
}

// A plain full-screen textured quad, rendered instead of a live gizmoCam
// scene pass for the PIP box and Through-Cam mode -- both should show the
// same distorted preview the patch mesh does, not a second, cleaner render.
const previewQuadScene = new THREE.Scene();
const previewQuadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const previewQuadMat = new THREE.MeshBasicMaterial({ map: distortedPreviewTex });
previewQuadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), previewQuadMat));
function renderPreviewViewport(x: number, y: number, w: number, h: number) {
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(previewQuadScene, previewQuadCam);
}

const patchMat = new THREE.MeshBasicMaterial({ map: distortedPreviewTex, side: THREE.DoubleSide });
const patchMesh = new THREE.Mesh(patchGeo, patchMat);
sphereAnchor.add(patchMesh);

// gizmoCam renders into camRT to capture "what the real camera sees" — but
// gizmoCam sits exactly at the sphere's center, so every debug overlay
// object above (sphere shell, great circles, poles, frustum outline, the
// patch mesh itself, the gizmo's own body/helper, the reference floor grid
// lines) falls directly in its view too. Left unfiltered, that overlay
// geometry gets baked into the captured image — and since the patch mesh's
// own material samples camRT.texture, it would even read the very texture
// it's contaminating (a same-pass feedback loop). Layer 1 is "debug-only":
// gizmoCam never sees it, so its capture is a clean shot of just the floor.
const DEBUG_LAYER = 1;
for (const o of [rowGridLines, colGridLines, camHelper, sphereShell, frustumLine, patchMesh]) o.layers.set(DEBUG_LAYER);
gizmoBody.traverse((o) => o.layers.set(DEBUG_LAYER));
polesGroup.traverse((o) => o.layers.set(DEBUG_LAYER));

// Two pole markers for whatever (Drow,Dcol) fitPairOfPlanes recovers, in a
// color distinct from the ground-truth red/blue poles so the two can be
// compared by eye.
function makeRecoveredPoleMarker(color: number): THREE.Mesh {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 8), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 }));
  m.layers.set(DEBUG_LAYER);
  m.visible = false;
  sphereAnchor.add(m);
  return m;
}
// Same hue family as the ground-truth poles (pure red/blue vs. their softer
// 0xff5555/0x5599ff), slightly bigger, and translucent -- so a good fit
// visually MERGES into the ground-truth marker underneath it (translucent
// pure red over soft red reads as one blended dot), while a bad fit shows
// as two clearly separate dots.
const recoveredRowPoleA = makeRecoveredPoleMarker(0xff0000);
const recoveredRowPoleB = makeRecoveredPoleMarker(0xff0000);
const recoveredColPoleA = makeRecoveredPoleMarker(0x0000ff);
const recoveredColPoleB = makeRecoveredPoleMarker(0x0000ff);

// Debug overlay: the actual physical great circle each of a random sample
// of gradient votes traces on the sphere -- not the pole (a single point),
// the whole circle, low-opacity so overlapping ones read as a density field
// rather than individual lines. One shared LineSegments buffer (not a Line
// per circle) since there can be hundreds of these; each circle contributes
// DEBUG_CIRCLE_SEGMENTS independent 2-point segments so many unrelated
// circles can live in one buffer with no connecting lines between them.
const DEBUG_CIRCLE_SEGMENTS = 48;
const gradientCirclesGeo = new THREE.BufferGeometry();
const gradientCirclesMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.35 });
const gradientCirclesLines = new THREE.LineSegments(gradientCirclesGeo, gradientCirclesMat);
gradientCirclesLines.layers.set(DEBUG_LAYER);
sphereAnchor.add(gradientCirclesLines);

// Debug overlay: each CHOSEN vote's own raw axis (n), drawn as a short
// directed segment from the origin outward -- unlike the ring above (which
// is symmetric and can't show orientation), a segment anchored at one fixed
// end reveals which way each vote's pole actually points, i.e. whether the
// mod-180 gradient-angle fold (see computeWorldVotes) is collapsing what
// would otherwise be a two-winged +-n "butterfly" per family down to a
// single-sided cluster.
const AXIS_VECTOR_LENGTH = 0.7;
const axisVectorsGeo = new THREE.BufferGeometry();
const axisVectorsMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6 });
const axisVectorsLines = new THREE.LineSegments(axisVectorsGeo, axisVectorsMat);
axisVectorsLines.layers.set(DEBUG_LAYER);
axisVectorsLines.visible = false;
sphereAnchor.add(axisVectorsLines);

// ── Math helpers ─────────────────────────────────────────────────────────

function slerpUnit(a: THREE.Vector3, b: THREE.Vector3, t: number): THREE.Vector3 {
  const dot = THREE.MathUtils.clamp(a.dot(b), -1, 1);
  const omega = Math.acos(dot);
  if (omega < 1e-6) return a.clone();
  const s = Math.sin(omega);
  return a.clone().multiplyScalar(Math.sin((1 - t) * omega) / s).addScaledVector(b, Math.sin(t * omega) / s);
}

// Plane through the camera origin containing a line (a point on it + its
// direction) — normal via point-direction cross product, no far-point
// approximation needed since the direction is exact and constant.
function greatCircleNormal(pointOnLine: THREE.Vector3, direction: THREE.Vector3, camPos: THREE.Vector3): THREE.Vector3 | null {
  const toPoint = pointOnLine.clone().sub(camPos);
  const n = toPoint.clone().cross(direction);
  if (n.lengthSq() < 1e-10) return null; // camera sits (nearly) on the line — degenerate
  return n.normalize();
}

// Fixed-size pool of reusable Line objects, one per grid line in a family —
// updated in place each frame (position attribute overwritten) rather than
// allocating a new BufferGeometry every frame, which would leak GPU buffers
// at 60fps.
function buildCirclePool(count: number, color: number): THREE.Line[] {
  const lines: THREE.Line[] = [];
  for (let i = 0; i < count; i++) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array((CIRCLE_SEGMENTS + 1) * 3), 3));
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }));
    line.layers.set(DEBUG_LAYER);
    circlesGroup.add(line);
    lines.push(line);
  }
  return lines;
}
function writeCirclePoints(line: THREE.Line, normal: THREE.Vector3, radius: number) {
  const helper = Math.abs(normal.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const u = helper.clone().cross(normal).normalize();
  const v = normal.clone().cross(u);
  const pos = (line.geometry.attributes.position as THREE.BufferAttribute);
  for (let i = 0; i <= CIRCLE_SEGMENTS; i++) {
    const a = (i / CIRCLE_SEGMENTS) * Math.PI * 2;
    const ca = Math.cos(a) * radius, sa = Math.sin(a) * radius;
    pos.setXYZ(i, u.x * ca + v.x * sa, u.y * ca + v.y * sa, u.z * ca + v.z * sa);
  }
  pos.needsUpdate = true;
}
const rowLineKs: number[] = [];
for (let k = -Math.min(VIS_HALF_EXTENT, HALF_R); k <= Math.min(VIS_HALF_EXTENT, HALF_R); k += GRID_STEP) rowLineKs.push(k);
const colLineKs: number[] = [];
for (let k = -Math.min(VIS_HALF_EXTENT, HALF_C); k <= Math.min(VIS_HALF_EXTENT, HALF_C); k += GRID_STEP) colLineKs.push(k);
const rowCirclePool = buildCirclePool(rowLineKs.length, 0xff5555);
const colCirclePool = buildCirclePool(colLineKs.length, 0x5599ff);

function cornerDir(u: number, v: number, quat: THREE.Quaternion, vFovRad: number, aspect: number): THREE.Vector3 {
  const halfV = vFovRad / 2;
  const yc = Math.tan(halfV) * v;
  const xc = Math.tan(halfV) * aspect * u;
  return new THREE.Vector3(xc, yc, -1).normalize().applyQuaternion(quat);
}

// ── Spherical-Hough prototype ────────────────────────────────────────────
//
// Validated first as an offline experiment (scripts/experiments/
// spherical-gradient-hough.ts): every qualifying pixel's (position,
// gradient angle) already fully determines the (theta,rho) of the line
// through it, so it can be lifted straight to a 3D plane normal with no
// discrete-2D-line-extraction stage at all. Ported here to run on the
// gizmo camera's own rendered capture (with simulated noise/blur), and
// visualized directly on the great sphere instead of a console log.
//
// The lift itself is done differently here than in the offline experiment,
// specifically to avoid re-deriving a local-frame sign convention by hand:
// instead of the closed-form n=[cosT,sinT,-rho/f], two nearby image points
// along the local edge direction are each converted to a WORLD ray via
// cornerDir (already validated elsewhere in this file against THREE's own
// raycaster), and the plane normal is just their cross product — the same
// point+direction construction greatCircleNormal above already uses, so
// there's no separate local-to-world convention to get wrong.

interface Vote { n: THREE.Vector3; weight: number }

// Simulated sensor noise -- Box-Muller Gaussian, in place.
function addGaussianNoise(gray: Float64Array, std: number) {
  if (std <= 0) return;
  for (let i = 0; i < gray.length; i++) {
    const u1 = Math.max(1e-9, Math.random()), u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    gray[i] = Math.min(255, Math.max(0, gray[i] + z * std));
  }
}

// Box-average decimation from a `scale`x-oversized buffer down to its true
// (dstW, dstH) -- a proper box filter, not point-sampling, so it actually
// band-limits before the final grid rather than just picking one of every
// `scale` pixels (which would reintroduce the very aliasing this whole
// capture path exists to avoid).
function downsampleBoxAverage(src: Float64Array, srcW: number, srcH: number, scale: number, dstW: number, dstH: number): Float64Array {
  const dst = new Float64Array(dstW * dstH);
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      let sum = 0, count = 0;
      for (let dy = 0; dy < scale; dy++) {
        const sy = y * scale + dy;
        if (sy >= srcH) continue;
        for (let dx = 0; dx < scale; dx++) {
          const sx = x * scale + dx;
          if (sx >= srcW) continue;
          sum += src[sy * srcW + sx];
          count++;
        }
      }
      dst[y * dstW + x] = count > 0 ? sum / count : 0;
    }
  }
  return dst;
}

// Two-pass (horizontal then vertical) box blur -- O(radius) per pixel per
// pass instead of src/lines.ts's boxBlur, which is a single O(radius^2)
// pass (fine there: it only ever runs at small radius on an
// already-small image). Here it runs on a captureSupersample x buffer at a
// correspondingly larger radius, where the quadratic cost is what pegged
// the main thread at radius 6 on a 1024x768 buffer (~130M ops single-pass
// vs ~27M split two ways) -- this is the actual fix for that freeze, not
// just a smaller cap.
function separableBoxBlur(src: Float64Array, w: number, h: number, radius: number): Float64Array {
  if (radius <= 0) return src.slice();
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const lo = Math.max(0, x - radius), hi = Math.min(w - 1, x + radius);
      let sum = 0;
      for (let xx = lo; xx <= hi; xx++) sum += src[row + xx];
      tmp[row + x] = sum / (hi - lo + 1);
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      const lo = Math.max(0, y - radius), hi = Math.min(h - 1, y + radius);
      let sum = 0;
      for (let yy = lo; yy <= hi; yy++) sum += tmp[yy * w + x];
      out[y * w + x] = sum / (hi - lo + 1);
    }
  }
  return out;
}

// Flips row order top<->bottom on a grayscale buffer (symmetric, so it's
// its own inverse either direction).
function flipRowsF64(src: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    const srcRow = h - 1 - y;
    out.set(src.subarray(srcRow * w, (srcRow + 1) * w), y * w);
  }
  return out;
}

// Replaces the old "render small, blur small" pipeline with the physically
// correct order: render at captureSupersample x, blur THERE (so the blur
// acts on a near-continuous image the way a real lens's defocus/diffraction
// would), THEN box-downsample to the final resolution (the sensor's actual
// discretization step), THEN add sensor noise at that final pixel grid --
// noise is an electronic/shot-noise artifact of the sensor itself, so it
// belongs after discretization, not before it.
//
// Returned in GL's native bottom-up row order, matching camRT.texture's own
// convention (and distortedPreviewTex's, which is built directly from this)
// -- NOT flipped to top-down. computeWorldVotes needs top-down for its NDC
// math, so its caller flips explicitly (flipRowsF64) rather than this
// function flipping internally: flipping here once made computeWorldVotes
// correct but silently turned the display texture upside down, since that
// consumer needs the un-flipped orientation. Two consumers, two different
// needs -- the shared step has to stay neutral (native order) and let each
// caller transform to what IT specifically needs.
function captureDistortedGrayscale(): { gray: Float64Array; w: number; h: number } {
  const { w: cw, h: ch } = captureRTSize;
  const raw = new Uint8Array(cw * ch * 4);
  renderer.readRenderTargetPixels(camRT, 0, 0, cw, ch, raw);
  const hiResGray = toGrayscale(raw, cw, ch);
  const hiResBlurred = separableBoxBlur(hiResGray, cw, ch, Math.round(state.simBlur * state.captureSupersample));
  const gray = downsampleBoxAverage(hiResBlurred, cw, ch, state.captureSupersample, rtSize.w, rtSize.h);
  addGaussianNoise(gray, state.simNoise);
  return { gray, w: rtSize.w, h: rtSize.h };
}

let lastPreviewUpdate = 0;
const PREVIEW_UPDATE_INTERVAL_MS = 100; // ~10fps -- see captureDistortedGrayscale/updateDistortedPreview

// Rebuilds distortedPreviewTex via the same captureDistortedGrayscale used
// for the real computation -- so what's displayed IS what's being
// analyzed, not an idealized stand-in for it. Throttled (see animate())
// since a hi-res readback + blur + downsample every single frame is real
// CPU cost, not something to pay 60x/sec for a preview that only needs to
// look "live", not be frame-perfect.
// Same encoding as the tracker page's own gradient-field overlay (hue =
// direction, saturation = magnitude normalized to this frame's own max) --
// see src/main.ts's hsvToRgb. Duplicated rather than imported since main.ts
// is a page entry point, not a shared module.
function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; } else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; } else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; } else { r = c; b = x; }
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

// Writes the gradient-field visualization into `out` (RGBA, w*h*4) --
// reuses state.simGradRadius so the visualized kernel matches the one
// computeWorldVotes actually analyzes with, not an independently-chosen one.
function paintGradientField(gray: Float64Array, w: number, h: number, out: Uint8Array) {
  const r = Math.round(state.simGradRadius);
  const mags = new Float64Array(w * h);
  let maxMag = 0;
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const fx = gray[i + r] - gray[i - r];
      const fy = gray[i + r * w] - gray[i - r * w];
      const mag = Math.hypot(fx, fy);
      mags[i] = mag;
      if (mag > maxMag) maxMag = mag;
    }
  }
  out.fill(0);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const fx = gray[i + r] - gray[i - r];
      const fy = gray[i + r * w] - gray[i - r * w];
      let theta = Math.atan2(fy, fx);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      const sat = maxMag > 0 ? mags[i] / maxMag : 0;
      const [rr, gg, bb] = hsvToRgb((theta / Math.PI) * 360, sat, 1);
      const o = i * 4;
      out[o] = rr; out[o + 1] = gg; out[o + 2] = bb; out[o + 3] = 255;
    }
  }
}

// Structure-tensor "cornerness" measure: at each pixel, aggregate the local
// gradient outer-product tensor [[Sxx,Sxy],[Sxy,Syy]] over a window --
// separate from the gradient-sampling radius (gradRadius, which only shapes
// the fx/fy stencil itself). Coupling these two originally seemed like the
// natural way to avoid a new tunable, but testing showed it's a real
// confound: widening gradRadius to strengthen corner discrimination also
// changes the base gradient computation (which pixels register as edges at
// all), degrading the fit for reasons unrelated to coherence. aggRadius
// (state.coherenceRadius) is the one tunable this cleanup actually needs.
// A clean single-edge neighborhood has one dominant eigenvalue (anisotropic);
// a junction/corner where two edge families overlap has two comparable
// eigenvalues (more isotropic). coherence = (lambda1-lambda2)/(lambda1+lambda2)
// is bounded to [0,1] by construction -- no threshold or empirical constant
// (e.g. Harris's k) needed. Shared by computeWorldVotes (as an actual
// per-vote weight factor) and paintCleanedGradientField (its visualization),
// so the field view always shows exactly what's driving the fit.
function computeCoherenceField(gray: Float64Array, w: number, h: number, gradRadius: number, aggRadius: number): { fx: Float64Array; fy: Float64Array; coherence: Float64Array } {
  const r = gradRadius;
  const n = w * h;
  const fx = new Float64Array(n), fy = new Float64Array(n);
  const Ixx = new Float64Array(n), Iyy = new Float64Array(n), Ixy = new Float64Array(n);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const gx = gray[i + r] - gray[i - r];
      const gy = gray[i + r * w] - gray[i - r * w];
      fx[i] = gx; fy[i] = gy;
      Ixx[i] = gx * gx; Iyy[i] = gy * gy; Ixy[i] = gx * gy;
    }
  }
  const Sxx = separableBoxBlur(Ixx, w, h, aggRadius);
  const Syy = separableBoxBlur(Iyy, w, h, aggRadius);
  const Sxy = separableBoxBlur(Ixy, w, h, aggRadius);

  const coherence = new Float64Array(n);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const sxx = Sxx[i], syy = Syy[i], sxy = Sxy[i];
      const trace = sxx + syy;
      const lambdaDiff = Math.hypot(sxx - syy, 2 * sxy); // lambda1 - lambda2
      coherence[i] = trace > 1e-9 ? lambdaDiff / trace : 0;
    }
  }
  return { fx, fy, coherence };
}

function paintCleanedGradientField(gray: Float64Array, w: number, h: number, out: Uint8Array) {
  const r = Math.round(state.simGradRadius);
  const n = w * h;
  const { fx, fy, coherence } = computeCoherenceField(gray, w, h, r, Math.round(state.coherenceRadius));

  const mags = new Float64Array(n);
  let maxMag = 0;
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const mag = Math.hypot(fx[i], fy[i]);
      mags[i] = mag;
      if (mag > maxMag) maxMag = mag;
    }
  }

  out.fill(0);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      let theta = Math.atan2(fy[i], fx[i]);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      // Same hue/sat encoding as the plain gradient field, but saturation is
      // now magnitude * coherence -- junction pixels visibly desaturate
      // toward white here even when their raw magnitude is just as strong
      // as a clean edge's, which is exactly the failure mode this is meant
      // to expose (magnitude alone can't tell them apart; coherence can).
      const sat = maxMag > 0 ? (mags[i] / maxMag) * coherence[i] : 0;
      const [rr, gg, bb] = hsvToRgb((theta / Math.PI) * 360, sat, 1);
      const o = i * 4;
      out[o] = rr; out[o + 1] = gg; out[o + 2] = bb; out[o + 3] = 255;
    }
  }
}

function updateDistortedPreview() {
  const { gray, w, h } = captureDistortedGrayscale();
  if (state.fieldView === 'gradient') {
    paintGradientField(gray, w, h, distortedPreviewData);
  } else if (state.fieldView === 'cleaned') {
    paintCleanedGradientField(gray, w, h, distortedPreviewData);
  } else {
    for (let i = 0; i < gray.length; i++) {
      const v = Math.max(0, Math.min(255, gray[i]));
      const o = i * 4;
      distortedPreviewData[o] = v; distortedPreviewData[o + 1] = v; distortedPreviewData[o + 2] = v; distortedPreviewData[o + 3] = 255;
    }
  }
  distortedPreviewTex.needsUpdate = true;
}

// gray is expected to already be captureDistortedGrayscale's output --
// blur is no longer applied here; it happens upstream, at supersampled
// resolution, before that function's own downsample step (see its comment
// for why applying it after an already-small render couldn't remove the
// staircase aliasing an early low-res render bakes in).
function computeWorldVotes(
  gray: Float64Array, w: number, h: number,
  minMag: number, gradientRadius: number, coherenceRadius: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Vote[] {
  const r = gradientRadius;
  const votes: Vote[] = [];
  // top-down pixel (px,py) -> NDC (u,v); v flips since NDC is up-positive
  // but py (top-down) is down-positive -- same relationship the patch
  // mesh's own UV fix (elsewhere in this file) already established.
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  // Each vote is weighted by magnitude * coherence: coherence (structure-
  // tensor eigenvalue ratio -- see computeCoherenceField) is ~1 for a clean
  // single-direction edge and drops toward 0 at junctions where two edge
  // families blend into one gradient sample, i.e. exactly the contamination
  // source identified earlier. coherenceRadius is independent of
  // gradientRadius -- see computeCoherenceField's header comment for why
  // coupling them was a confound.
  const { fx: allFx, fy: allFy, coherence } = computeCoherenceField(gray, w, h, r, coherenceRadius);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const fx = allFx[i], fy = allFy[i];
      const mag = Math.hypot(fx, fy);
      if (mag < minMag) continue;
      let theta = Math.atan2(fy, fx);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      // tangent direction (along the line, perpendicular to the gradient),
      // in the same top-down pixel-delta convention as (x,y) themselves.
      const tdx = -Math.sin(theta), tdy = Math.cos(theta);
      const [u1, v1] = toNDC(x, y);
      const [u2, v2] = toNDC(x + tdx, y + tdy);
      const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
      const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
      const n = ray1.clone().cross(ray2);
      if (n.lengthSq() < 1e-12) continue;
      n.normalize();
      votes.push({ n, weight: mag * coherence[i] });
    }
  }
  return votes;
}

// Cached from the last runAxesReconstruction so the circle-sample slider
// can rebuild its debug view instantly on drag, without re-running gradient
// computation + accumulation just to change how many circles are drawn.
let lastVotes: Vote[] = [];

// The TRUE [minPercent, maxPercent) band by magnitude rank, out of every
// vote -- not a percent of some fixed render cap. An earlier version
// computed the count as a percent of a fixed MAX_DEBUG_CIRCLES=2000 cap
// instead of votes.length, so every setting from ~3% up to 100% rendered
// the identical ~2000 absolute-strongest votes in the whole frame (2000 out
// of a typical 70k-180k), never revealing anything further down the ranking
// -- which is exactly why contamination never showed up regardless of the
// slider position. No cap now: this same slice is also what actually feeds
// fitPairOfPlanes, not just the overlay. The min bound lets the strongest
// (often the most saturated/clipped, or otherwise atypical) votes be
// excluded from the fit while still keeping a wide magnitude window below.
function votesInMagnitudeBand(votes: Vote[], minPercent: number, maxPercent: number): Vote[] {
  const sorted = Array.from(votes).sort((a, b) => b.weight - a.weight);
  const lo = Math.round(sorted.length * (minPercent / 100));
  const hi = Math.round(sorted.length * (maxPercent / 100));
  if (hi <= lo) return [];
  return sorted.slice(lo, hi);
}

function updateGradientCirclesDebug() {
  const chosen = votesInMagnitudeBand(lastVotes, state.circleSamplePercentMin, state.circleSamplePercentMax);
  if (chosen.length === 0) {
    gradientCirclesGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    axisVectorsGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    return;
  }

  // Color by magnitude: lowest weight among the CHOSEN circles is red,
  // highest is blue, straight linear RGB blend between them.
  let minW = Infinity, maxW = -Infinity;
  for (const vote of chosen) {
    if (vote.weight < minW) minW = vote.weight;
    if (vote.weight > maxW) maxW = vote.weight;
  }
  const wRange = maxW - minW;

  const positions = new Float32Array(chosen.length * DEBUG_CIRCLE_SEGMENTS * 2 * 3);
  const colors = new Float32Array(chosen.length * DEBUG_CIRCLE_SEGMENTS * 2 * 3);
  // One 2-point segment per vote: (0,0,0) -> n*AXIS_VECTOR_LENGTH, in the
  // same sphereAnchor-local space as everything else here. Anchoring every
  // segment at the same fixed point (rather than drawing +-n symmetrically)
  // is what makes an asymmetric "one-wing" vs "two-wing" cluster visible at
  // all -- a plain axis line through the origin would look identical either way.
  const axisPositions = new Float32Array(chosen.length * 2 * 3);
  const axisColors = new Float32Array(chosen.length * 2 * 3);
  let p = 0, pc = 0, ap = 0, apc = 0;
  const u = new THREE.Vector3(), v = new THREE.Vector3(), helper = new THREE.Vector3();
  for (const vote of chosen) {
    const normal = vote.n;
    const t = wRange > 0 ? (vote.weight - minW) / wRange : 0;
    const r = 1 - t, b = t;
    helper.set(0, 1, 0);
    if (Math.abs(normal.y) >= 0.9) helper.set(1, 0, 0);
    u.crossVectors(helper, normal).normalize();
    v.crossVectors(normal, u);
    for (let s = 0; s < DEBUG_CIRCLE_SEGMENTS; s++) {
      const a0 = (s / DEBUG_CIRCLE_SEGMENTS) * Math.PI * 2;
      const a1 = ((s + 1) / DEBUG_CIRCLE_SEGMENTS) * Math.PI * 2;
      const c0 = Math.cos(a0) * SPHERE_RADIUS, sn0 = Math.sin(a0) * SPHERE_RADIUS;
      const c1 = Math.cos(a1) * SPHERE_RADIUS, sn1 = Math.sin(a1) * SPHERE_RADIUS;
      positions[p++] = u.x * c0 + v.x * sn0; positions[p++] = u.y * c0 + v.y * sn0; positions[p++] = u.z * c0 + v.z * sn0;
      positions[p++] = u.x * c1 + v.x * sn1; positions[p++] = u.y * c1 + v.y * sn1; positions[p++] = u.z * c1 + v.z * sn1;
      colors[pc++] = r; colors[pc++] = 0; colors[pc++] = b;
      colors[pc++] = r; colors[pc++] = 0; colors[pc++] = b;
    }
    // Length is a power curve of the vote's own weight (magnitude *
    // coherence, same quantity the color already encodes), not a plain
    // linear ratio -- within any single top-X% band the true weight spread
    // is narrow (that's the actual data, not a display bug), so a straight
    // linear map barely shows any variation. state.weightSharpenPower is the
    // SAME exponent fitPairOfPlanes itself now applies to vote weighting
    // (see its header comment) -- this overlay is a live preview of exactly
    // how the fit will weight these same votes, not just a display trick.
    const len = maxW > 0 ? AXIS_VECTOR_LENGTH * Math.pow(vote.weight / maxW, state.weightSharpenPower) : 0;
    axisPositions[ap++] = 0; axisPositions[ap++] = 0; axisPositions[ap++] = 0;
    axisPositions[ap++] = normal.x * len;
    axisPositions[ap++] = normal.y * len;
    axisPositions[ap++] = normal.z * len;
    axisColors[apc++] = r; axisColors[apc++] = 0; axisColors[apc++] = b;
    axisColors[apc++] = r; axisColors[apc++] = 0; axisColors[apc++] = b;
  }
  gradientCirclesGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  gradientCirclesGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  axisVectorsGeo.setAttribute('position', new THREE.Float32BufferAttribute(axisPositions, 3));
  axisVectorsGeo.setAttribute('color', new THREE.Float32BufferAttribute(axisColors, 3));
  axisVectorsGeo.computeBoundingSphere();
  gradientCirclesGeo.computeBoundingSphere();
}

// Fits the degenerate quadric ("pair of planes through the origin") that
// best explains "every vote lies on one plane or the other," directly, with
// no clustering, no discretization, no bucket resolution, and no nonlinear
// search over a joint (Drow, roll) parameterization -- replaced an earlier
// Fibonacci-hemisphere bucket accumulator + peak search that needed all
// three. A pair of planes with unit normals
// n1, n2 is the surface (P.n1)(P.n2) = 0 -- expand that into a quadratic
// form P^T M P for a symmetric 3x3 M, and every vote gives one LINEAR
// equation in M's 6 independent entries, so fitting M is a single smallest-
// eigenvector solve (same tool src/vp.ts and src/lattice.ts already use,
// just a 6x6 system instead of 3x3/4x4). M's own eigenvectors then are NOT
// n1/n2 directly -- for orthogonal n1 ⊥ n2 they come out as the two 45-
// degree BISECTORS of n1/n2 (eigenvalues of equal magnitude, opposite
// sign), plus n1 x n2 itself (the floor normal) at eigenvalue ~0. Weighted
// by gradient magnitude * coherence, then SHARPENED by state.weightSharpenPower
// (each vote's weight normalized to the strongest in this batch, then raised
// to that power) before accumulating -- the same power curve used to draw
// the axis-vector debug overlay, applied here for real: if the strongest
// votes visually stand out as far more trustworthy once sharpened, letting
// them dominate the solve (rather than being diluted by the much larger
// population of merely-okay ones) should help the fit the same way. Still
// no attempt to downweight junction/corner contamination specifically --
// only coherence (independently) and this magnitude-based sharpening do.
function fitPairOfPlanes(votes: Vote[]): { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null {
  let maxW = 0;
  for (const { weight } of votes) if (weight > maxW) maxW = weight;
  const power = state.weightSharpenPower;
  const ATA: number[][] = Array.from({ length: 6 }, () => new Array(6).fill(0));
  for (const { n, weight } of votes) {
    const sharpened = maxW > 0 ? Math.pow(weight / maxW, power) : 0;
    // monomials of the quadratic form P^T M P; the xy/xz/yz entries here
    // are the coefficients of 2*Mxy etc. (see the expansion in the header
    // comment), halved back out below once M is reassembled.
    const row = [n.x * n.x, n.y * n.y, n.z * n.z, n.x * n.y, n.x * n.z, n.y * n.z];
    for (let a = 0; a < 6; a++) {
      const wra = sharpened * row[a];
      for (let b = 0; b < 6; b++) ATA[a][b] += wra * row[b];
    }
  }
  const m = smallestEigenvector(ATA);
  const M = [
    [m[0], m[3] / 2, m[4] / 2],
    [m[3] / 2, m[1], m[5] / 2],
    [m[4] / 2, m[5] / 2, m[2]],
  ];
  const { values, vectors } = jacobiEigenSymmetric(M);
  let zeroIdx = 0;
  for (let i = 1; i < 3; i++) if (Math.abs(values[i]) < Math.abs(values[zeroIdx])) zeroIdx = i;
  const others = [0, 1, 2].filter((i) => i !== zeroIdx);
  const b1 = new THREE.Vector3(vectors[others[0]][0], vectors[others[0]][1], vectors[others[0]][2]);
  const b2 = new THREE.Vector3(vectors[others[1]][0], vectors[others[1]][1], vectors[others[1]][2]);
  const Dnormal = new THREE.Vector3(vectors[zeroIdx][0], vectors[zeroIdx][1], vectors[zeroIdx][2]).normalize();
  const Drow = b1.clone().add(b2);
  const Dcol = b1.clone().sub(b2);
  if (Drow.lengthSq() < 1e-9 || Dcol.lengthSq() < 1e-9) return null; // degenerate: bisectors nearly parallel/antiparallel
  return { Drow: Drow.normalize(), Dcol: Dcol.normalize(), Dnormal };
}

function angleBetweenDegV(a: THREE.Vector3, b: THREE.Vector3): number {
  return THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(Math.abs(a.dot(b)), -1, 1)));
}

// Recovered-pole visibility is handled per-frame in updateSphereOverlays,
// tied to showCircles (the same toggle as the ground-truth great circles).
let axesComputed = false;

let axesCapturing = false;
let lastAxesCapture = 0;

function runAxesReconstruction() {
  if (axesCapturing) return; // don't stack overlapping captures (e.g. auto-recompute firing while a slow one is still in flight)
  axesCapturing = true;
  captureAxesBtn.disabled = true;
  const prevLabel = captureAxesBtn.textContent;
  captureAxesBtn.textContent = '⏳ computing...';
  axesReadout.textContent = 'computing...';
  requestAnimationFrame(() => {
    try {
      const t0 = performance.now();
      // captureDistortedGrayscale already applies noise+blur (in the
      // physically-correct order -- see its comment) but returns GL-native
      // (bottom-up) row order; computeWorldVotes's NDC math needs top-down.
      const { gray: rawGray, w, h } = captureDistortedGrayscale();
      const gray = flipRowsF64(rawGray, w, h);
      const vFovRad = THREE.MathUtils.degToRad(gizmoCam.fov);
      const votes = computeWorldVotes(gray, w, h, state.simMinMag, state.simGradRadius, state.coherenceRadius, camQuat, vFovRad, RT_ASPECT);
      lastVotes = votes;
      updateGradientCirclesDebug();
      const t1 = performance.now();

      // The "top circles" range is the actual fit input, not just a
      // debug-overlay knob: fitPairOfPlanes only ever sees the same
      // magnitude band that's drawn as rings, so the overlay always shows
      // exactly what's driving the result.
      const fitVotes = votesInMagnitudeBand(votes, state.circleSamplePercentMin, state.circleSamplePercentMax);
      // No bucket resolution, no dedup radius: a single pass building a 6x6
      // matrix, then one small eigendecomposition -- see fitPairOfPlanes's
      // header comment for the derivation.
      const quadricPair = fitPairOfPlanes(fitVotes);
      const t2 = performance.now();

      if (quadricPair) {
        recoveredRowPoleA.position.copy(quadricPair.Drow).multiplyScalar(SPHERE_RADIUS);
        recoveredRowPoleB.position.copy(quadricPair.Drow).multiplyScalar(-SPHERE_RADIUS);
        recoveredColPoleA.position.copy(quadricPair.Dcol).multiplyScalar(SPHERE_RADIUS);
        recoveredColPoleB.position.copy(quadricPair.Dcol).multiplyScalar(-SPHERE_RADIUS);
      }
      axesComputed = !!quadricPair;

      const lines = [`${votes.length} votes  (${fitVotes.length} fed to fit)`];
      if (quadricPair) {
        const errRowFromRow = angleBetweenDegV(quadricPair.Drow, ROW_DIR);
        const errColFromRow = angleBetweenDegV(quadricPair.Dcol, ROW_DIR);
        const flipped = errRowFromRow > errColFromRow;
        const rowErr = flipped ? errColFromRow : errRowFromRow;
        const colErr = flipped ? angleBetweenDegV(quadricPair.Drow, COL_DIR) : angleBetweenDegV(quadricPair.Dcol, COL_DIR);
        lines.push(`row err ${rowErr.toFixed(2)}°  col err ${colErr.toFixed(2)}°`);
      } else {
        lines.push(`degenerate fit`);
      }
      lines.push(`votes ${(t1 - t0).toFixed(0)}ms  fit ${(t2 - t1).toFixed(0)}ms`);
      axesReadout.textContent = lines.join('\n');
    } finally {
      captureAxesBtn.disabled = false;
      captureAxesBtn.textContent = prevLabel;
      axesCapturing = false;
    }
  });
}
captureAxesBtn.addEventListener('click', runAxesReconstruction);

// ── Controls: world-orbit (mode A) + free look-around (mode C) ─────────

const viewerCam = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.05, 500);
viewerCam.position.set(6, 6, 14);
viewerCam.layers.enable(DEBUG_LAYER);
const worldOrbit = new OrbitControls(viewerCam, canvas);
worldOrbit.target.set(0, 3, 0);

const insideCam = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.02, 500);
insideCam.layers.enable(DEBUG_LAYER);
let insideYaw = 0, insidePitch = 0;
let dragging = false, lastPX = 0, lastPY = 0;

canvas.addEventListener('pointerdown', (e) => {
  if (state.mode !== 'inside') return;
  dragging = true; lastPX = e.clientX; lastPY = e.clientY;
});
addEventListener('pointerup', () => { dragging = false; });
addEventListener('pointermove', (e) => {
  if (!dragging || state.mode !== 'inside') return;
  const dx = e.clientX - lastPX, dy = e.clientY - lastPY;
  lastPX = e.clientX; lastPY = e.clientY;
  insideYaw -= dx * 0.004;
  insidePitch = THREE.MathUtils.clamp(insidePitch - dy * 0.004, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
});
canvas.addEventListener('wheel', (e) => {
  if (state.mode !== 'inside') return;
  e.preventDefault();
  insideCam.fov = THREE.MathUtils.clamp(insideCam.fov + e.deltaY * 0.02, 20, 110);
  insideCam.updateProjectionMatrix();
}, { passive: false });

// ── Mode switching ───────────────────────────────────────────────────────

function setMode(m: Mode) {
  state.mode = m;
  for (const k of Object.keys(modeBtns) as Mode[]) modeBtns[k].classList.toggle('active', k === m);
  worldOrbit.enabled = m === 'world';
  insideHint.style.display = m === 'inside' ? 'block' : 'none';
  pipFrame.style.display = m === 'through' ? 'none' : 'block';
  pipLabel.style.display = m === 'through' ? 'none' : 'block';
}
modeBtns.world.addEventListener('click', () => setMode('world'));
modeBtns.through.addEventListener('click', () => setMode('through'));
modeBtns.inside.addEventListener('click', () => setMode('inside'));

// ── Slider / checkbox wiring ─────────────────────────────────────────────

bindSlider('camX', (v) => (state.camX = v));
bindSlider('camY', (v) => (state.camY = v));
bindSlider('camZ', (v) => (state.camZ = v));
bindSlider('camYaw', (v) => (state.camYawDeg = v), (v) => `${v.toFixed(0)}°`);
bindSlider('camPitch', (v) => (state.camPitchDeg = v), (v) => `${v.toFixed(0)}°`);
bindSlider('camFocal', (v) => (state.focalMM = v), (v) => `${v.toFixed(0)}mm`);
// Re-entrancy guard: with aspect lock on, dragging one slider programmatically
// drives the other via a dispatched 'input' event, which would otherwise
// trigger ITS OWN lock logic and try to drive the first one back again.
let syncingViewportAspect = false;
function clampViewport(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(v)));
}
bindSlider('viewportW', (v) => {
  // Ratio BEFORE this change -- state.viewportW is still the old value here.
  const oldAspect = state.viewportW / state.viewportH;
  state.viewportW = v;
  if (state.aspectLocked && !syncingViewportAspect) {
    syncingViewportAspect = true;
    const hInput = document.getElementById('viewportH') as HTMLInputElement;
    hInput.value = String(clampViewport(v / oldAspect, 96, 2000));
    hInput.dispatchEvent(new Event('input'));
    syncingViewportAspect = false;
  }
  resizeCaptureBuffers();
}, (v) => v.toFixed(0));
bindSlider('viewportH', (v) => {
  const oldAspect = state.viewportW / state.viewportH;
  state.viewportH = v;
  if (state.aspectLocked && !syncingViewportAspect) {
    syncingViewportAspect = true;
    const wInput = document.getElementById('viewportW') as HTMLInputElement;
    wInput.value = String(clampViewport(v * oldAspect, 128, 2000));
    wInput.dispatchEvent(new Event('input'));
    syncingViewportAspect = false;
  }
  resizeCaptureBuffers();
}, (v) => v.toFixed(0));
bindCheckbox('aspectLocked', (v) => (state.aspectLocked = v));

bindCheckbox('showSphere', (v) => (state.showSphere = v));
bindCheckbox('showCircles', (v) => (state.showCircles = v));
bindCheckbox('showPoles', (v) => (state.showPoles = v));
bindCheckbox('showFrustum', (v) => (state.showFrustum = v));
bindCheckbox('showPatch', (v) => (state.showPatch = v));
bindCheckbox('showFloor', (v) => (state.showFloor = v));
bindCheckbox('showGizmoBody', (v) => (state.showGizmoBody = v));

bindSlider('simNoise', (v) => (state.simNoise = v), (v) => v.toFixed(0));
bindSlider('simBlur', (v) => (state.simBlur = v), (v) => v.toFixed(0));
bindSlider('simGradRadius', (v) => (state.simGradRadius = v), (v) => v.toFixed(0));
bindSlider('simMinMag', (v) => (state.simMinMag = v), (v) => v.toFixed(0));
bindSlider('captureSupersample', (v) => { state.captureSupersample = v; resizeCaptureBuffers(); }, (v) => `${v.toFixed(0)}x`);
bindSlider('coherenceRadius', (v) => (state.coherenceRadius = v), (v) => v.toFixed(0));
bindRadioGroup('fieldView', (v) => (state.fieldView = v as 'normal' | 'gradient' | 'cleaned'));
bindSlider('circleSamplePercentMin', (v) => { state.circleSamplePercentMin = v; updateGradientCirclesDebug(); }, (v) => `${v.toFixed(0)}%`);
bindSlider('circleSamplePercentMax', (v) => { state.circleSamplePercentMax = v; updateGradientCirclesDebug(); }, (v) => `${v.toFixed(0)}%`);
bindCheckbox('showRecoveredPoles', (v) => (state.showRecoveredPoles = v));
bindCheckbox('showAxisVectors', (v) => (state.showAxisVectors = v));
bindCheckbox('showTopCircles', (v) => (state.showTopCircles = v));
bindSlider('weightSharpenPower', (v) => { state.weightSharpenPower = v; updateGradientCirclesDebug(); }, (v) => v.toFixed(1));
bindCheckbox('axesAutoCapture', (v) => (state.axesAutoCapture = v));
bindSlider('axesCaptureInterval', (v) => (state.axesCaptureIntervalMs = v), (v) => `${v.toFixed(0)}`);

// ── Per-frame update ─────────────────────────────────────────────────────

const ROW_DIR = new THREE.Vector3(1, 0, 0); // world +X — direction shared by every "row" floor line
const COL_DIR = new THREE.Vector3(0, 0, 1); // world +Z — direction shared by every "col" floor line
const camPos = new THREE.Vector3();
const camQuat = new THREE.Quaternion();
const euler = new THREE.Euler(0, 0, 0, 'YXZ');

function updateGizmo() {
  camPos.set(state.camX, state.camY, state.camZ);
  euler.set(THREE.MathUtils.degToRad(state.camPitchDeg), THREE.MathUtils.degToRad(state.camYawDeg), 0);
  camQuat.setFromEuler(euler);

  gizmoCam.position.copy(camPos);
  gizmoCam.quaternion.copy(camQuat);
  const hFovRad = 2 * Math.atan(SENSOR_WIDTH_MM / (2 * state.focalMM));
  const vFovRad = 2 * Math.atan(Math.tan(hFovRad / 2) / RT_ASPECT);
  gizmoCam.fov = THREE.MathUtils.radToDeg(vFovRad);
  gizmoCam.aspect = RT_ASPECT;
  gizmoCam.updateProjectionMatrix();

  gizmoBody.position.copy(camPos);
  gizmoBody.quaternion.copy(camQuat);
  camHelper.update();

  sphereAnchor.position.copy(camPos);

  readout.innerHTML =
    `h-fov: ${THREE.MathUtils.radToDeg(hFovRad).toFixed(1)}&deg; &nbsp; v-fov: ${gizmoCam.fov.toFixed(1)}&deg;<br>` +
    `pole separation: ${THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(ROW_DIR.dot(COL_DIR), -1, 1))).toFixed(2)}&deg; (always 90&deg; — the orthogonal constraint)`;

  return { hFovRad, vFovRad };
}

function updateSphereOverlays(vFovRad: number) {
  circlesGroup.visible = state.showCircles;
  polesGroup.visible = state.showPoles;
  frustumLine.visible = state.showFrustum;
  patchMesh.visible = state.showPatch;
  sphereShell.visible = state.showSphere;

  // Recovered (pair-of-planes) poles have their own dedicated toggle in the
  // axes reconstruction section, independent of showCircles -- they're a
  // fit result, not part of the ground-truth great-circle grid.
  const recoveredPolesVisible = state.showRecoveredPoles && axesComputed;
  recoveredRowPoleA.visible = recoveredPolesVisible;
  recoveredRowPoleB.visible = recoveredPolesVisible;
  recoveredColPoleA.visible = recoveredPolesVisible;
  recoveredColPoleB.visible = recoveredPolesVisible;
  axisVectorsLines.visible = state.showAxisVectors;
  gradientCirclesLines.visible = state.showTopCircles;

  if (state.showCircles) {
    const updateFamily = (ks: number[], pool: THREE.Line[], axis: 'row' | 'col', dir: THREE.Vector3) => {
      for (let i = 0; i < ks.length; i++) {
        const k = ks[i];
        const pointOnLine = axis === 'row' ? new THREE.Vector3(0, 0, k) : new THREE.Vector3(k, 0, 0);
        const n = greatCircleNormal(pointOnLine, dir, camPos);
        pool[i].visible = !!n;
        if (n) writeCirclePoints(pool[i], n, SPHERE_RADIUS);
      }
    };
    updateFamily(rowLineKs, rowCirclePool, 'row', ROW_DIR);
    updateFamily(colLineKs, colCirclePool, 'col', COL_DIR);
  }

  if (state.showPoles) {
    rowPoleA.position.copy(ROW_DIR).multiplyScalar(SPHERE_RADIUS);
    rowPoleB.position.copy(ROW_DIR).multiplyScalar(-SPHERE_RADIUS);
    colPoleA.position.copy(COL_DIR).multiplyScalar(SPHERE_RADIUS);
    colPoleB.position.copy(COL_DIR).multiplyScalar(-SPHERE_RADIUS);
  }

  if (state.showFrustum) {
    const corners = [
      cornerDir(-1, -1, camQuat, vFovRad, RT_ASPECT),
      cornerDir(1, -1, camQuat, vFovRad, RT_ASPECT),
      cornerDir(1, 1, camQuat, vFovRad, RT_ASPECT),
      cornerDir(-1, 1, camQuat, vFovRad, RT_ASPECT),
    ];
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < 4; i++) {
      const a = corners[i], b = corners[(i + 1) % 4];
      for (let t = 0; t < 16; t++) pts.push(slerpUnit(a, b, t / 16).multiplyScalar(SPHERE_RADIUS));
    }
    frustumLine.geometry.dispose();
    frustumLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  }

  if (state.showPatch) {
    const pos = patchGeo.attributes.position as THREE.BufferAttribute;
    for (let j = 0; j <= PATCH_RES; j++) {
      const v = (j / PATCH_RES) * 2 - 1;
      for (let i = 0; i <= PATCH_RES; i++) {
        const u = (i / PATCH_RES) * 2 - 1;
        const d = cornerDir(u, v, camQuat, vFovRad, RT_ASPECT).multiplyScalar(SPHERE_RADIUS);
        const idx = j * (PATCH_RES + 1) + i;
        pos.setXYZ(idx, d.x, d.y, d.z);
      }
    }
    pos.needsUpdate = true;
    patchGeo.computeVertexNormals();
  }
}

// ── Render ───────────────────────────────────────────────────────────────

function renderViewport(cam: THREE.Camera, x: number, y: number, w: number, h: number) {
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(scene, cam);
}

function layoutPip() {
  const w = Math.min(320, innerWidth * 0.28);
  const h = w / RT_ASPECT;
  const margin = 20;
  pipRect = { x: innerWidth - w - margin, y: innerHeight - h - margin, w, h };
  pipFrame.style.left = pipRect.x + 'px';
  pipFrame.style.top = pipRect.y + 'px';
  pipFrame.style.width = w + 'px';
  pipFrame.style.height = h + 'px';
  pipLabel.style.left = pipRect.x + 'px';
  pipLabel.style.top = (pipRect.y - 16) + 'px';
}

function resize() {
  renderer.setSize(innerWidth, innerHeight);
  viewerCam.aspect = innerWidth / innerHeight;
  viewerCam.updateProjectionMatrix();
  insideCam.aspect = innerWidth / innerHeight;
  insideCam.updateProjectionMatrix();
  layoutPip();
}
addEventListener('resize', resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  const { vFovRad } = updateGizmo();
  updateSphereOverlays(vFovRad);

  // Toggling helper/body visibility per mode: don't render the camera's own
  // body/frustum-helper from inside its own optical center, and don't render
  // the patch mesh (it would occlude the very view it's built from) while
  // actually looking through the real gizmo camera.
  gizmoBody.visible = state.mode === 'world' && state.showGizmoBody;
  camHelper.visible = state.mode === 'world' && state.showFrustum;
  floorMesh.visible = state.showFloor;

  // Camera render target feeds the PIP preview, the sphere patch, AND
  // Through-Cam mode now (previously skipped in "through" since that mode
  // used to render gizmoCam live instead) -- all three now show the same
  // distorted preview texture rather than gizmoCam directly, so all three
  // need camRT fresh every frame.
  // renderer.setViewport/setScissor multiply whatever they're given by
  // devicePixelRatio internally before the real gl.viewport() call — verified
  // empirically (a call with device-pixel values here produced a raw GL
  // viewport double what was intended, and against camRT specifically, a
  // viewport bigger than the target's actual buffer, silently cropping the
  // capture to one quadrant). Every call below therefore passes plain
  // CSS-pixel values, matching innerWidth/innerHeight — never pre-multiplied.
  const dpr = renderer.getPixelRatio();

  {
    const prevRT = renderer.getRenderTarget();
    renderer.setRenderTarget(camRT);
    renderer.setViewport(0, 0, captureRTSize.w / dpr, captureRTSize.h / dpr);
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.render(scene, gizmoCam);
    renderer.setRenderTarget(prevRT);
  }

  // Throttled: a full readback + per-pixel noise + blur every frame is real
  // CPU cost for a preview that only needs to look live, not be frame-exact.
  const now = performance.now();
  if (now - lastPreviewUpdate >= PREVIEW_UPDATE_INTERVAL_MS) {
    lastPreviewUpdate = now;
    updateDistortedPreview();
  }

  // Auto-recompute: still throttled by an explicit interval, not run every
  // frame -- fitPairOfPlanes itself is cheap (a single 6x6 eigenvector
  // solve), but captureDistortedGrayscale + computeWorldVotes together are
  // not free (a full hi-res readback/blur/downsample plus a per-pixel pass
  // over the analysis frame), so this is still meaningfully slower than
  // 60fps and needs its own pacing, just a much shorter one than the old
  // bucket accumulator ever allowed.
  if (state.axesAutoCapture && !axesCapturing && now - lastAxesCapture >= state.axesCaptureIntervalMs) {
    lastAxesCapture = now;
    runAxesReconstruction();
  }

  // Full-buffer reset before the per-frame clear: leftover viewport state
  // from the previous frame otherwise constrains what clear() actually
  // touches, leaving stale content from an earlier frame visible outside
  // whatever the new frame's viewport happens to be.
  renderer.setViewport(0, 0, innerWidth, innerHeight);
  renderer.setScissorTest(false);
  renderer.setClearColor(0x0a0a0f, 1);
  renderer.clear();

  if (state.mode === 'world') {
    worldOrbit.update();
    renderViewport(viewerCam, 0, 0, innerWidth, innerHeight);
    renderPreviewViewport(pipRect.x, innerHeight - pipRect.y - pipRect.h, pipRect.w, pipRect.h);
  } else if (state.mode === 'through') {
    // Letterbox the fixed-aspect gizmo camera into whatever the window shape is.
    const winAspect = innerWidth / innerHeight;
    let w = innerWidth, h = innerHeight, x = 0, y = 0;
    if (winAspect > RT_ASPECT) { w = innerHeight * RT_ASPECT; x = (innerWidth - w) / 2; }
    else { h = innerWidth / RT_ASPECT; y = (innerHeight - h) / 2; }
    renderPreviewViewport(x, y, w, h);
  } else {
    insideCam.position.copy(camPos);
    euler.set(insidePitch, insideYaw, 0);
    insideCam.quaternion.setFromEuler(euler);
    renderViewport(insideCam, 0, 0, innerWidth, innerHeight);
    renderPreviewViewport(pipRect.x, innerHeight - pipRect.y - pipRect.h, pipRect.w, pipRect.h);
  }
}

setMode('world');
animate();

// ── Dev bridge ───────────────────────────────────────────────────────────
//
// Lets an external tool (scripts/dev-bridge/) send arbitrary JS to run
// directly in THIS module's scope — a literal `eval(code)` call written
// inline below, so it closes over every top-level const/let/function in
// this file (state, scene, camPos, gizmoCam, ...) exactly as if typed into
// this file itself — plus pull PNG snapshots of the canvas. Local-only;
// no-ops silently if scripts/dev-bridge/server.js isn't running.
(function initDevBridge() {
  const BRIDGE_PORT = 8787;
  let ws: WebSocket | null = null;
  let reconnectTimer: number | undefined;

  function scheduleReconnect() {
    ws = null;
    clearTimeout(reconnectTimer);
    reconnectTimer = window.setTimeout(connect, 2000);
  }

  function connect() {
    try { ws = new WebSocket(`ws://localhost:${BRIDGE_PORT}`); }
    catch { scheduleReconnect(); return; }

    ws.addEventListener('open', () => ws!.send(JSON.stringify({ role: 'browser' })));
    ws.addEventListener('close', scheduleReconnect);
    ws.addEventListener('error', () => {});
    ws.addEventListener('message', (ev) => {
      let msg: any;
      try { msg = JSON.parse(ev.data); } catch { return; }

      if (msg.type === 'eval') {
        let ok = true, value: any, error: string | undefined;
        try { value = eval(msg.code); }
        catch (e: any) { ok = false; error = String(e?.stack ?? e); }
        try { value = value === undefined ? undefined : JSON.parse(JSON.stringify(value)); }
        catch { value = String(value); }
        ws!.send(JSON.stringify({ type: 'evalResult', id: msg.id, ok, value, error }));
      } else if (msg.type === 'screenshot') {
        const dataUrl = renderer.domElement.toDataURL('image/png');
        ws!.send(JSON.stringify({ type: 'screenshotResult', id: msg.id, ok: true, dataUrl }));
      }
    });
  }
  connect();

  // Low-rate unsolicited frame push so a reasonably fresh screenshot is
  // always on disk without an explicit request.
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'frame', dataUrl: renderer.domElement.toDataURL('image/jpeg', 0.7) }));
    }
  }, 1000);
})();
