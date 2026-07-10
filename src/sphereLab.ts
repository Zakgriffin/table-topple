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
import { generateTorus, buildLookupTable } from './debruijn.ts';
import { toGrayscale, binarize, pickBestCandidate } from './decode.ts';
import type { SampledGrid, SampledCell } from './decode.ts';
import { jacobiEigenSymmetric, smallestEigenvector } from './linalg.ts';

type Mode = 'world' | 'through' | 'inside' | 'projected';

// ── DOM ──────────────────────────────────────────────────────────────────

const canvas = document.getElementById('gl') as HTMLCanvasElement;
const pipFrame = document.getElementById('pipFrame') as HTMLDivElement;
const pipLabel = document.getElementById('pipLabel') as HTMLDivElement;
const insideHint = document.getElementById('insideHint') as HTMLDivElement;
const readout = document.getElementById('readout') as HTMLDivElement;
const axesReadout = document.getElementById('axesReadout') as HTMLDivElement;
const captureAxesBtn = document.getElementById('captureAxesBtn') as HTMLButtonElement;
const positionReadout = document.getElementById('positionReadout') as HTMLDivElement;
const marginalRightCanvas = document.getElementById('marginalRight') as HTMLCanvasElement;
const marginalBottomCanvas = document.getElementById('marginalBottom') as HTMLCanvasElement;
const marginalRightCtx = marginalRightCanvas.getContext('2d')!;
const marginalBottomCtx = marginalBottomCanvas.getContext('2d')!;
const contamToggles = document.getElementById('contamToggles') as HTMLDivElement;
const toggleHideFieldBtn = document.getElementById('toggleHideField') as HTMLButtonElement;
const toggleTrueContamBtn = document.getElementById('toggleTrueContam') as HTMLButtonElement;
const toggleReconContamBtn = document.getElementById('toggleReconContam') as HTMLButtonElement;

const modeBtns: Record<Mode, HTMLButtonElement> = {
  world: document.getElementById('modeWorld') as HTMLButtonElement,
  through: document.getElementById('modeThrough') as HTMLButtonElement,
  inside: document.getElementById('modeInside') as HTMLButtonElement,
  projected: document.getElementById('modeProjected') as HTMLButtonElement,
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
  // Only honor a saved value if it still matches one of the CURRENT options --
  // otherwise a renamed/removed option value (e.g. an old 'normal' after this
  // group's options changed) would leave every input unchecked instead of
  // falling back to the HTML's own default `checked` attribute.
  if (savedControls[name] !== undefined && inputs.some((inp) => inp.value === savedControls[name])) {
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
  showSphere: true, showCircles: true, showPoles: true, showFrustum: true, showPatch: true, showFloor: true, showGizmoBody: true, showRecoveredFloor: true,
  showTrueContamination: false, showReconstructedContamination: false, hideField: false,
  simNoise: 8, simBlur: 1, simGradRadius: 1, coherenceRadius: 1,
  circleSamplePercentMin: 0, circleSamplePercentMax: 5,
  showRecoveredPoles: true,
  showAxisVectors: false,
  showTopCircles: true,
  weightSharpenPower: 4,
  fieldView: 'noised' as 'raw' | 'antialiased' | 'downsampled' | 'noised' | 'gradient' | 'agreement' | 'effective',
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
// For decoding an ORDER x ORDER sampled bit window back into an absolute
// torus (row,col) position -- see runPositionDecode.
const debruijnLookup = buildLookupTable(debruijn);
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

// Same shape/size as the yellow ground-truth gizmoBody above, in green, at
// the DECODED position from runPositionDecode -- uses the real camQuat for
// orientation (camQuat is a known input to the whole recovery pipeline, not
// something being solved for here) so any visible gap against the yellow box
// is purely positional error. Hidden until a successful decode exists.
const recoveredCamGizmo = new THREE.Mesh(
  new THREE.BoxGeometry(0.3, 0.25, 0.4),
  new THREE.MeshStandardMaterial({ color: 0x33dd55 }),
);
recoveredCamGizmo.visible = false;
scene.add(recoveredCamGizmo);
const recoveredCamAxes = new THREE.AxesHelper(0.6);
recoveredCamGizmo.add(recoveredCamAxes);
function updateRecoveredCamGizmo() {
  if (lastPositionDecode) {
    recoveredCamGizmo.position.copy(lastPositionDecode.camPos);
    recoveredCamGizmo.quaternion.copy(camQuat);
  }
  recoveredCamGizmo.visible = state.mode === 'world' && state.showGizmoBody && !!lastPositionDecode;
}

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

// True whenever gizmoCam's rendered output might have changed since the last
// capture -- camera details or a capture-distortion/filter-pipeline tunable.
// Declared here (not down by its own usage in animate()) for the same
// reason as pipRect above: markCaptureDirty() can fire as early as
// page-load-time slider binding, before a `let` declared later in the
// module would have left its temporal dead zone. Starts true so the very
// first frame always populates the preview.
let captureDirty = true;
function markCaptureDirty() {
  captureDirty = true;
}

// Called once at startup (implicitly, via the viewportW/H/captureSupersample
// slider bindings firing on load) and again whenever those sliders change.
// camRT.setSize() resizes the render target in place; distortedPreviewTex
// keeps its own object identity (so previewQuadMat/patchMat, which hold a
// reference to it, don't need to be touched) by having its .image swapped
// for a new {data,width,height} triple -- DataTexture has no other resize path.
function resizeCaptureBuffers() {
  captureDirty = true;
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
  projectedPreviewData = new Uint8Array(rtSize.w * rtSize.h * 4);
  projectedPreviewTex.image = { data: projectedPreviewData, width: rtSize.w, height: rtSize.h };
  projectedPreviewTex.dispose();
  projectedPreviewTex.needsUpdate = true;
  trueContamData = new Uint8Array(rtSize.w * rtSize.h * 4);
  trueContamTex.image = { data: trueContamData, width: rtSize.w, height: rtSize.h };
  trueContamTex.dispose();
  trueContamTex.needsUpdate = true;
  reconContamData = new Uint8Array(rtSize.w * rtSize.h * 4);
  reconContamTex.image = { data: reconContamData, width: rtSize.w, height: rtSize.h };
  reconContamTex.dispose();
  reconContamTex.needsUpdate = true;
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

// "Projected Cam" mode: a bird's-eye, floor-plane-rectified view of
// whichever field view is currently selected -- built by buildProjectedTexture
// (see its own comment) from distortedPreviewData's own current content, so
// it always shows a rectified version of the SAME thing Through-Cam is
// currently showing, not a separately-computed approximation of it.
let projectedPreviewData = new Uint8Array(rtSize.w * rtSize.h * 4);
const projectedPreviewTex = new THREE.DataTexture(projectedPreviewData, rtSize.w, rtSize.h, THREE.RGBAFormat);
projectedPreviewTex.flipY = false;
projectedPreviewTex.colorSpace = THREE.SRGBColorSpace;
const projectedQuadScene = new THREE.Scene();
const projectedQuadMat = new THREE.MeshBasicMaterial({ map: projectedPreviewTex });
projectedQuadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), projectedQuadMat));
function renderProjectedViewport(x: number, y: number, w: number, h: number) {
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(projectedQuadScene, previewQuadCam);
}

// Through-Cam-only translucent overlays: per-pixel alpha is how "bad" that
// pixel's vote axis vector (see computeWorldVotes) is against a pair of
// reference directions -- true ROW_DIR/COL_DIR for the red overlay, the last
// recovered Drow/Dcol for the orange one -- so contamination (votes that
// don't actually belong to either family) shows up as a translucent wash
// directly on top of whatever field view is already selected, instead of
// only being visible indirectly via the fit's error numbers. Same
// DataTexture-quad architecture as projectedPreviewTex above, rendered as an
// extra alpha-blended pass on top of Through-Cam's own preview quad.
let trueContamData = new Uint8Array(rtSize.w * rtSize.h * 4);
const trueContamTex = new THREE.DataTexture(trueContamData, rtSize.w, rtSize.h, THREE.RGBAFormat);
trueContamTex.flipY = false;
const trueContamMat = new THREE.MeshBasicMaterial({ map: trueContamTex, transparent: true, depthTest: false, depthWrite: false });
const trueContamScene = new THREE.Scene();
trueContamScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), trueContamMat));
function renderTrueContamOverlay(x: number, y: number, w: number, h: number) {
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(trueContamScene, previewQuadCam);
}

let reconContamData = new Uint8Array(rtSize.w * rtSize.h * 4);
const reconContamTex = new THREE.DataTexture(reconContamData, rtSize.w, rtSize.h, THREE.RGBAFormat);
reconContamTex.flipY = false;
const reconContamMat = new THREE.MeshBasicMaterial({ map: reconContamTex, transparent: true, depthTest: false, depthWrite: false });
const reconContamScene = new THREE.Scene();
reconContamScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), reconContamMat));
function renderReconContamOverlay(x: number, y: number, w: number, h: number) {
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(reconContamScene, previewQuadCam);
}

// Per-pixel "badness": 90 - angleBetweenDegV(n, D) is 0 when the vote normal
// n is exactly perpendicular to D (the ideal case -- see computeWorldVotes'
// own n·D=0 constraint) and grows as n drifts toward being PARALLEL to D
// instead (the degenerate/wrong case). Taking the min across both reference
// directions mirrors "take the min badness for the two planes" -- a vote is
// only truly contaminated if it fits NEITHER family well. /45 so 45 degrees
// off reads as fully opaque, matching the user's stated scale; badness can
// go up to 90 (n exactly parallel to D) but stays clamped at full opacity
// past 45 rather than needing a wider scale to "fit".
// agreement is the SAME array computeWorldVotes itself weights votes by
// (mag * agreement[i], see its own comment) -- multiplying it into alpha
// here too means a contaminated-corner pixel that the fit barely listens to
// (tiny effective-gradient weight, even if its raw angle is bad) fades out
// here as well, so the overlay shows contamination AS the fit actually
// experiences it, not as a plain per-pixel angle check would in isolation.
function computeContaminationAlpha(
  field: GradientField, agreement: Float64Array,
  dirA: THREE.Vector3, dirB: THREE.Vector3,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Float64Array {
  const { fx, fy, w, h, r } = field;
  const alpha = new Float64Array(w * h);
  // Bottom-up, NOT the flipped top-down convention computeWorldVotes uses --
  // this reads from lastNoisedPreviewGray (same GL-native bottom-up row
  // order as distortedPreviewData, see buildProjectedTexture's own comment
  // on the same distinction), and the overlay has to align pixel-for-pixel
  // with what's already on screen.
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, (py / h) * 2 - 1];
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const mag = Math.hypot(fx[i], fy[i]);
      if (mag === 0) continue;
      let theta = Math.atan2(fy[i], fx[i]);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      const tdx = -Math.sin(theta), tdy = Math.cos(theta);
      const [u1, v1] = toNDC(x, y);
      const [u2, v2] = toNDC(x + tdx, y + tdy);
      const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
      const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
      const n = ray1.clone().cross(ray2);
      if (n.lengthSq() < 1e-12) continue;
      n.normalize();
      const badnessA = 90 - angleBetweenDegV(n, dirA);
      const badnessB = 90 - angleBetweenDegV(n, dirB);
      const badnessAlpha = THREE.MathUtils.clamp(Math.min(badnessA, badnessB) / 45, 0, 1);
      alpha[i] = badnessAlpha * agreement[i];
    }
  }
  return alpha;
}

function paintContaminationOverlay(alpha: Float64Array, color: readonly [number, number, number], out: Uint8Array) {
  for (let i = 0; i < alpha.length; i++) {
    const o = i * 4;
    out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2];
    out[o + 3] = Math.round(alpha[i] * 255);
  }
}

const TRUE_CONTAM_COLOR = [230, 40, 40] as const;
const RECON_CONTAM_COLOR = [235, 150, 20] as const;

// Recomputes whichever overlay(s) are actually toggled on -- skipped
// entirely otherwise, since each is a full per-pixel gradient+cross-product
// pass, the same cost class as computeWorldVotes itself. lum is derived
// straight from distortedPreviewData (Through-Cam's own current content, any
// field view), not from a fresh capture -- this overlay always describes
// whatever's already on screen.
function updateContaminationOverlays() {
  if (!state.showTrueContamination && !state.showReconstructedContamination) return;
  // Real analysis-equivalent brightness, NOT whatever field view happens to
  // be on screen (see lastNoisedPreviewGray's comment) -- null only if
  // updateDistortedPreview hasn't run at all yet (cold start), in which case
  // there's nothing to score.
  if (!lastNoisedPreviewGray) return;
  const w = rtSize.w, h = rtSize.h;
  const lum = lastNoisedPreviewGray;
  const vFovRad = THREE.MathUtils.degToRad(gizmoCam.fov);
  // Same field, same agreement computeWorldVotes itself uses to weight
  // votes -- shared across both overlays below since neither the true nor
  // the reconstructed reference directions affect this computation at all.
  const field = computeGradientField(lum, w, h, Math.round(state.simGradRadius));
  const agreement = computeGradientAgreementField(field, Math.round(state.coherenceRadius));

  if (state.showTrueContamination) {
    const alpha = computeContaminationAlpha(field, agreement, ROW_DIR, COL_DIR, camQuat, vFovRad, RT_ASPECT);
    paintContaminationOverlay(alpha, TRUE_CONTAM_COLOR, trueContamData);
    trueContamTex.needsUpdate = true;
  }
  if (state.showReconstructedContamination) {
    if (lastRecoveredAxes) {
      const alpha = computeContaminationAlpha(field, agreement, lastRecoveredAxes.Drow, lastRecoveredAxes.Dcol, camQuat, vFovRad, RT_ASPECT);
      paintContaminationOverlay(alpha, RECON_CONTAM_COLOR, reconContamData);
      toggleReconContamBtn.textContent = 'reconstructed contamination overlay (orange)';
    } else {
      // No fit to compare against yet -- fully transparent is correct, but
      // silently invisible reads as broken. Flag it directly on the button
      // instead of leaving the user to guess why nothing's showing.
      reconContamData.fill(0);
      toggleReconContamBtn.textContent = 'reconstructed contamination overlay (orange) — run "capture now" first';
    }
    reconContamTex.needsUpdate = true;
  }
}

// Bin geometry from the most recent buildProjectedTexture call -- the (u,v)
// extent and per-bin width of the floor-plane rectification, in the
// camera-relative Drow/Dcol frame (see buildProjectedTexture's "Relative to
// the camera" comment). Exposed at module scope so the marginal-accumulation
// overlays (drawMarginalLines) and the position decode step can reuse the
// exact same bin layout instead of re-deriving it.
interface ProjectedBins { minU: number; maxU: number; minV: number; maxV: number; binWidthU: number; binWidthV: number; w: number; h: number }
let lastProjectedBins: ProjectedBins | null = null;

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
recoveredCamGizmo.traverse((o) => o.layers.set(DEBUG_LAYER));
polesGroup.traverse((o) => o.layers.set(DEBUG_LAYER));

// Reuses the SAME projectedPreviewTex "Projected Cam" mode already builds
// (not a separate computation) as a decal on a plane placed at the DECODED
// pose in the actual 3D world -- any visible misalignment against the real
// floor pattern underneath is the recovery pipeline's true end-to-end error,
// made visible directly rather than read off a percentage.
const recoveredFloorOverlayMat = new THREE.MeshBasicMaterial({ map: projectedPreviewTex, side: THREE.DoubleSide, transparent: true, opacity: 0.92 });
const recoveredFloorOverlay = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), recoveredFloorOverlayMat);
recoveredFloorOverlay.visible = false;
scene.add(recoveredFloorOverlay);
recoveredFloorOverlay.layers.set(DEBUG_LAYER);

// Rebuilds the overlay's geometry/position/orientation -- called once per
// fresh decode (from runAxesReconstruction), not per frame: unlike the
// gizmo box above, this needs a new PlaneGeometry sized to the current bin
// extent, which is wasteful to reallocate 60x/sec. Per-frame mode/toggle
// visibility is handled separately in animate(), same split as gizmoBody.
function applyRecoveredFloorOverlay() {
  if (!lastPositionDecode || !lastRecoveredAxes || !lastProjectedBins) return;
  const { Drow, Dcol, Dnormal, distance } = lastRecoveredAxes;
  const normal = Dnormal.clone();
  const vFovRad = THREE.MathUtils.degToRad(gizmoCam.fov);
  if (cornerDir(0, 0, camQuat, vFovRad, RT_ASPECT).dot(normal) > 0) normal.negate();
  const { minU, maxU, minV, maxV } = lastProjectedBins;
  const width = maxU - minU, height = maxV - minV;
  if (!(width > 0) || !(height > 0)) return;

  recoveredFloorOverlay.geometry.dispose();
  recoveredFloorOverlay.geometry = new THREE.PlaneGeometry(width, height);

  const centerU = (minU + maxU) / 2, centerV = (minV + maxV) / 2;
  recoveredFloorOverlay.position.copy(lastPositionDecode.camPos)
    .addScaledVector(Drow, centerU)
    .addScaledVector(Dcol, centerV)
    .addScaledVector(normal, -distance);

  // Cross product, NOT the sign-corrected `normal` above -- guarantees a
  // right-handed, purely-rotational basis (setFromRotationMatrix on a
  // reflection matrix, which using a wrongly-signed normal could produce,
  // gives a garbage quaternion). Only affects which face is "front"; with
  // DoubleSide + a paper-thin plane that's invisible, so it's safe to let
  // this differ from the physically-signed normal used for the offset above.
  const zAxis = new THREE.Vector3().crossVectors(Drow, Dcol).normalize();
  const basis = new THREE.Matrix4().makeBasis(Drow, Dcol, zAxis);
  recoveredFloorOverlay.quaternion.setFromRotationMatrix(basis);
}

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
// the whole circle, low-opacity so overlapping ones read as a agreement field
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

// Tiny seeded PRNG (mulberry32) so noise is reproducible rather than
// Math.random()-fresh on every capture -- switching between field views (or
// re-triggering a preview update) would otherwise show a different noise
// realization each time, making it hard to visually compare stages.
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const NOISE_SEED = 1337;

// Simulated sensor noise -- Box-Muller Gaussian, in place. Re-seeds on every
// call (rather than sharing one running generator) so the same pixel index
// always gets the same noise draw for a given buffer size, regardless of
// how many times or in what order this has been called before.
function addGaussianNoise(gray: Float64Array, std: number) {
  if (std <= 0) return;
  const rng = mulberry32(NOISE_SEED);
  for (let i = 0; i < gray.length; i++) {
    const u1 = Math.max(1e-9, rng()), u2 = rng();
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

// Renders gizmoCam's view into camRT -- pulled out into its own function so
// the real analysis path (captureDistortedGrayscale, below) can always force
// a truly fresh capture regardless of the passive preview's dirty/throttle
// gating in animate(): correctness of "capture now" matters more than saving
// a render that's already infrequent (button click / auto-capture interval).
// renderer.setViewport/setScissor multiply whatever they're given by
// devicePixelRatio internally before the real gl.viewport() call — verified
// empirically (a call with device-pixel values here produced a raw GL
// viewport double what was intended, and against camRT specifically, a
// viewport bigger than the target's actual buffer, silently cropping the
// capture to one quadrant). Always pass plain CSS-pixel values.
function renderCamRT() {
  const dpr = renderer.getPixelRatio();
  const prevRT = renderer.getRenderTarget();
  renderer.setRenderTarget(camRT);
  renderer.setViewport(0, 0, captureRTSize.w / dpr, captureRTSize.h / dpr);
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.render(scene, gizmoCam);
  renderer.setRenderTarget(prevRT);
}

// Real cameras put a dedicated anti-aliasing / optical low-pass filter
// directly on the sensor, separate from whatever blur the lens itself adds
// -- specifically to band-limit the image before discretization, since a
// fine repeating pattern (like this scene's checkerboard) aliases into
// moire against the pixel grid otherwise. Its strength isn't independently
// tunable: its whole job is "just enough smoothing to prevent aliasing for
// THIS decimation factor," which is already fully determined by
// captureSupersample (an existing tunable) -- a separate slider here would
// just be redundant with one that already exists for a different reason.
function applyAntialiasFilter(gray: Float64Array, w: number, h: number, supersample: number): Float64Array {
  return separableBoxBlur(gray, w, h, Math.max(1, Math.round(supersample / 2)));
}

// Replaces the old "render small, blur small" pipeline with the physically
// correct order: render at captureSupersample x, apply the sensor's fixed
// AA filter, THEN blur (lens defocus/diffraction, user-controlled, acting
// on a near-continuous image the way a real lens would), THEN box-downsample
// to the final resolution (the sensor's actual discretization step), THEN
// add sensor noise at that final pixel grid -- noise is an electronic/
// shot-noise artifact of the sensor itself, so it belongs after
// discretization, not before it.
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
  renderCamRT();
  const { w: cw, h: ch } = captureRTSize;
  const raw = new Uint8Array(cw * ch * 4);
  renderer.readRenderTargetPixels(camRT, 0, 0, cw, ch, raw);
  const hiResGray = toGrayscale(raw, cw, ch);
  const antialiased = applyAntialiasFilter(hiResGray, cw, ch, state.captureSupersample);
  const hiResBlurred = separableBoxBlur(antialiased, cw, ch, Math.round(state.simBlur * state.captureSupersample));
  const gray = downsampleBoxAverage(hiResBlurred, cw, ch, state.captureSupersample, rtSize.w, rtSize.h);
  addGaussianNoise(gray, state.simNoise);
  return { gray, w: rtSize.w, h: rtSize.h };
}

// For the field-view preview only (not the real analysis path above, which
// stays a fast single-purpose function). "blurred" isn't its own stage
// here: once resampled down to a common display resolution, "blur at
// hi-res, then box-average to rtSize for display" and "blur at hi-res,
// then box-average downsample to rtSize as the real pipeline step" are the
// exact same computation -- so it would be pixel-identical to "downsampled"
// and add no new information.
//
// Stops as soon as it has what the CURRENTLY selected field view needs,
// instead of always computing raw+downsampled+noised regardless of which
// one is displayed -- the earlier always-compute-everything version did a
// second full downsample pass (for "raw") and a full-buffer .slice() copy
// (for "noised") on every single throttled tick even when neither was
// selected, real wasted work on top of what was already the most expensive
// per-frame CPU cost in the app.
// Cached by updateDistortedPreview whenever it computes the "noised"
// analysis-equivalent grayscale, regardless of which field view is actually
// selected for display -- the contamination overlays (updateContaminationOverlays)
// always need to compute gradients from this real brightness signal, never
// from whatever's currently painted into distortedPreviewData (which for
// 'gradient'/'agreement'/'effective' is already hue-encoded, and for 'none'
// is blank -- deriving a gradient back out of either would measure the
// wrong thing, or nothing at all).
let lastNoisedPreviewGray: Float64Array | null = null;

function updateDistortedPreview() {
  const { w: cw, h: ch } = captureRTSize;
  const rawRGBA = new Uint8Array(cw * ch * 4);
  renderer.readRenderTargetPixels(camRT, 0, 0, cw, ch, rawRGBA);
  const hiResGray = toGrayscale(rawRGBA, cw, ch);

  // Contamination overlays need the real noised grayscale even when it isn't
  // otherwise needed for display (hidden, or an early field view like 'raw')
  // -- see lastNoisedPreviewGray's comment. When neither overlay is active
  // this stays false and every early-return below still fires exactly as
  // before, preserving the "stop as soon as the selected view is ready"
  // perf win this function was originally written for.
  const needGrayForOverlay = state.showTrueContamination || state.showReconstructedContamination;

  if (state.hideField) {
    // Solid black -- isolates the contamination overlay(s) with nothing else
    // competing for attention underneath. A toggle alongside the overlay
    // buttons, not a field-view choice, so it overrides whichever field is
    // selected rather than being one itself.
    for (let i = 0; i < distortedPreviewData.length; i += 4) {
      distortedPreviewData[i] = 0; distortedPreviewData[i + 1] = 0; distortedPreviewData[i + 2] = 0; distortedPreviewData[i + 3] = 255;
    }
    distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
    // Still need the real "noised" grayscale below for the overlay(s) --
    // every paint call from here on is gated by !state.hideField so nothing
    // ever gets painted over the blanked buffer.
  }

  if (!state.hideField && state.fieldView === 'raw') {
    const raw = downsampleBoxAverage(hiResGray, cw, ch, state.captureSupersample, rtSize.w, rtSize.h);
    fillGrayscalePreview(raw, distortedPreviewData);
    distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
  }

  const antialiased = applyAntialiasFilter(hiResGray, cw, ch, state.captureSupersample);

  if (!state.hideField && state.fieldView === 'antialiased') {
    const aaDisplay = downsampleBoxAverage(antialiased, cw, ch, state.captureSupersample, rtSize.w, rtSize.h);
    fillGrayscalePreview(aaDisplay, distortedPreviewData);
    distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
  }

  const hiResBlurred = separableBoxBlur(antialiased, cw, ch, Math.round(state.simBlur * state.captureSupersample));
  const downsampled = downsampleBoxAverage(hiResBlurred, cw, ch, state.captureSupersample, rtSize.w, rtSize.h);

  if (!state.hideField && state.fieldView === 'downsampled') {
    fillGrayscalePreview(downsampled, distortedPreviewData);
    distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
  }

  // Safe to noise in place from here -- downsampled's un-noised value is
  // never needed again once we know the view isn't 'downsampled' itself.
  const noised = downsampled;
  addGaussianNoise(noised, state.simNoise);
  if (!state.hideField) {
    if (state.fieldView === 'noised') {
      fillGrayscalePreview(noised, distortedPreviewData);
      distortedPreviewTex.needsUpdate = true;
    } else if (state.fieldView === 'gradient') {
      const field = computeGradientField(noised, rtSize.w, rtSize.h, Math.round(state.simGradRadius));
      paintVectorFieldAsColor(field, distortedPreviewData);
      distortedPreviewTex.needsUpdate = true;
    } else if (state.fieldView === 'agreement') {
      const field = computeGradientField(noised, rtSize.w, rtSize.h, Math.round(state.simGradRadius));
      const agreement = computeGradientAgreementField(field, Math.round(state.coherenceRadius));
      paintScalarFieldAsGray(agreement, distortedPreviewData);
      distortedPreviewTex.needsUpdate = true;
    } else if (state.fieldView === 'effective') {
      const field = computeGradientField(noised, rtSize.w, rtSize.h, Math.round(state.simGradRadius));
      const agreement = computeGradientAgreementField(field, Math.round(state.coherenceRadius));
      const effective = computeEffectiveGradientField(field, agreement);
      paintVectorFieldAsColor(effective, distortedPreviewData);
      distortedPreviewTex.needsUpdate = true;
    }
  }
  lastNoisedPreviewGray = noised;
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

// ── Value fields (no color) ─────────────────────────────────────────────
//
// Every field below is a pure computation over numbers: gradient field is
// derived only from the grayscale image + a radius, agreement only from a
// gradient field + a radius, effective gradient only from a gradient field
// + an agreement field. None of them know about color -- that only happens
// in the paint* functions further down, and only for whichever ONE field is
// actually selected for display (see updateDistortedPreview's dispatch).

interface GradientField { fx: Float64Array; fy: Float64Array; w: number; h: number; r: number }

function computeGradientField(gray: Float64Array, w: number, h: number, gradRadius: number): GradientField {
  const r = gradRadius;
  const fx = new Float64Array(w * h), fy = new Float64Array(w * h);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      fx[i] = gray[i + r] - gray[i - r];
      fy[i] = gray[i + r * w] - gray[i - r * w];
    }
  }
  return { fx, fy, w, h, r };
}

// Magnitude of the local VECTOR SUM of gradients, not the sum of their
// individual magnitudes -- gradients that reinforce (point the same way)
// build agreement; gradients that conflict (point different ways) cancel
// instead of both counting toward "agreeing". Expanding |sum(v_i)|^2 into
// sum_i sum_j (v_i . v_j) is exactly the dot-product structure this is
// going for, just computed via vector-sum-then-magnitude rather than
// explicit pairwise dot products. Turned out to be a real, effective
// answer to junction/corner contamination -- a blended gradient at a
// junction disagrees with its neighbors' directions and cancels toward low
// agreement, doing the job corner-cleaning/coherence used to do, without
// needing the structure-tensor eigendecomposition that approach required.
//
// Raw (fx,fy) flips 180 degrees between a black->white and white->black
// edge along the SAME physical line (the mod-180 polarity fold discussed
// elsewhere, e.g. computeWorldVotes) -- summed naively, alternating-polarity
// edges (which a checkerboard is full of) would falsely cancel and read as
// "disagreeing" in genuinely coherent, edge-dense areas. Fixed the same way
// structure tensors handle it: double the angle (cos(2*theta), sin(2*theta))
// before summing, which maps a direction and its 180-degree-flipped twin to
// the identical point, so polarity can't cause false cancellation.
//
// No magnitude threshold: a pixel's own contribution to the vector sum
// already scales with its magnitude (cx,cy = mag*cos/sin(2theta)), so tiny,
// incoherent noise gradients mostly cancel each other out on aggregation
// rather than needing to be gated out beforehand. Reuses coherenceRadius
// ("agreement window" in the UI) as the "what counts as nearby" aggregation
// window. Normalized to this frame's own max, same convention used
// everywhere else in this file.
function computeGradientAgreementField(field: GradientField, aggRadius: number): Float64Array {
  const { fx, fy, w, h } = field;
  const n = w * h;
  const cx = new Float64Array(n), cy = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const mag = Math.hypot(fx[i], fy[i]);
    if (mag === 0) continue;
    const theta = Math.atan2(fy[i], fx[i]);
    cx[i] = mag * Math.cos(2 * theta);
    cy[i] = mag * Math.sin(2 * theta);
  }
  const sx = separableBoxBlur(cx, w, h, aggRadius);
  const sy = separableBoxBlur(cy, w, h, aggRadius);
  const agreement = new Float64Array(n);
  let maxAgreement = 0;
  for (let i = 0; i < n; i++) {
    agreement[i] = Math.hypot(sx[i], sy[i]);
    if (agreement[i] > maxAgreement) maxAgreement = agreement[i];
  }
  if (maxAgreement > 0) for (let i = 0; i < n; i++) agreement[i] /= maxAgreement;
  return agreement;
}

// The gradient field's own vector, componentwise-scaled by agreement's
// scalar -- literally "the multiple of these two fields", nothing more.
// Direction is unaffected (scaling by a non-negative scalar can't change a
// vector's angle); magnitude shrinks wherever agreement is low. Wired into
// computeWorldVotes as the actual vote weight, not just a visualization.
function computeEffectiveGradientField(field: GradientField, agreement: Float64Array): GradientField {
  const { fx, fy, w, h, r } = field;
  const n = w * h;
  const efx = new Float64Array(n), efy = new Float64Array(n);
  for (let i = 0; i < n; i++) { efx[i] = fx[i] * agreement[i]; efy[i] = fy[i] * agreement[i]; }
  return { fx: efx, fy: efy, w, h, r };
}

// ── Display: colorizes a value field, only for whichever one is on screen ─

// hue = direction, saturation = magnitude normalized to this field's own
// max, value = 1 always.
function paintVectorFieldAsColor(field: GradientField, out: Uint8Array) {
  const { fx, fy, w, h } = field;
  const n = w * h;
  const mags = new Float64Array(n);
  let maxMag = 0;
  for (let i = 0; i < n; i++) {
    const mag = Math.hypot(fx[i], fy[i]);
    mags[i] = mag;
    if (mag > maxMag) maxMag = mag;
  }
  for (let i = 0; i < n; i++) {
    let theta = Math.atan2(fy[i], fx[i]);
    if (theta < 0) theta += Math.PI;
    if (theta >= Math.PI) theta -= Math.PI;
    const sat = maxMag > 0 ? mags[i] / maxMag : 0;
    const [rr, gg, bb] = hsvToRgb((theta / Math.PI) * 360, sat, 1);
    const o = i * 4;
    out[o] = rr; out[o + 1] = gg; out[o + 2] = bb; out[o + 3] = 255;
  }
}

function paintScalarFieldAsGray(field: Float64Array, out: Uint8Array) {
  for (let i = 0; i < field.length; i++) {
    const v = Math.round(THREE.MathUtils.clamp(field[i], 0, 1) * 255);
    const o = i * 4;
    out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
  }
}

// Corner-cleaning (structure-tensor coherence weighting) and its "cleaned"
// field-view visualization were removed here -- coherenceRadius/"corner
// window" is kept (computeGradientAgreementField's aggregation window still
// uses it), but nothing coherence-specific remains.

function fillGrayscalePreview(gray: Float64Array, out: Uint8Array) {
  for (let i = 0; i < gray.length; i++) {
    const v = Math.max(0, Math.min(255, gray[i]));
    const o = i * 4;
    out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
  }
}

// gray is expected to already be captureDistortedGrayscale's output --
// blur is no longer applied here; it happens upstream, at supersampled
// resolution, before that function's own downsample step (see its comment
// for why applying it after an already-small render couldn't remove the
// staircase aliasing an early low-res render bakes in).
function computeWorldVotes(
  gray: Float64Array, w: number, h: number,
  gradientRadius: number, agreementRadius: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Vote[] {
  const votes: Vote[] = [];
  // top-down pixel (px,py) -> NDC (u,v); v flips since NDC is up-positive
  // but py (top-down) is down-positive -- same relationship the patch
  // mesh's own UV fix (elsewhere in this file) already established.
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  // Driving the fit with "effective gradient" (magnitude * agreement) instead
  // of plain magnitude -- agreement is computeGradientAgreementField's
  // dot-product/vector-sum measure (same thing "effective gradient field"
  // visualizes), reused here as the actual vote weight rather than just a
  // display.
  const field = computeGradientField(gray, w, h, gradientRadius);
  const { fx, fy, r } = field;
  const agreement = computeGradientAgreementField(field, agreementRadius);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const mag = Math.hypot(fx[i], fy[i]);
      if (mag === 0) continue; // untouched border / exactly-flat pixel, not a tunable threshold
      let theta = Math.atan2(fy[i], fx[i]);
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
      votes.push({ n, weight: mag * agreement[i] });
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
    // Length is a power curve of the vote's own weight (plain magnitude,
    // same quantity the color already encodes), not a plain
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
// by "effective gradient" (magnitude * local agreement, see
// computeGradientAgreementField -- corner-cleaning/coherence weighting was
// removed and replaced with this), then SHARPENED by
// state.weightSharpenPower (each vote's weight normalized to the strongest
// in this batch, then raised to that power) before accumulating -- the same
// power curve used to draw the axis-vector debug overlay, applied here for
// real: if the strongest votes visually stand out as far more trustworthy
// once sharpened, letting them dominate the solve (rather than being
// diluted by the much larger population of merely-okay ones) should help
// the fit the same way. No attempt to downweight junction/corner
// contamination currently -- only this magnitude-based sharpening does.
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

// De-means the profile, then finds the lag (in bins) of the strongest
// non-trivial autocorrelation peak -- i.e. the period of whatever periodic
// signal dominates it. Searches from a small minimum lag (skipping the
// always-large near-zero-lag correlation, uninformative for period-finding)
// up to half the profile length (beyond that too little overlap remains to
// trust the estimate). O(n * maxLag); cheap at the profile sizes here (a
// handful of ms even at n=2000).
function autocorrelationPeriod(profile: Float64Array): number | null {
  const n = profile.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += profile[i];
  mean /= n;
  const centered = new Float64Array(n);
  for (let i = 0; i < n; i++) centered[i] = profile[i] - mean;

  const minLag = Math.max(2, Math.floor(n * 0.005));
  const maxLag = Math.floor(n / 2);
  let bestLag = -1, bestScore = -Infinity;
  for (let lag = minLag; lag < maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) sum += centered[i] * centered[i + lag];
    if (sum > bestScore) { bestScore = sum; bestLag = lag; }
  }
  return bestLag > 0 ? bestLag : null;
}

interface SpacingEstimate { periodU: number; periodV: number; distanceU: number; distanceV: number }

// Recovers the camera's distance to the floor plane along Dnormal, from
// nothing but the already-recovered axis directions plus the KNOWN true
// grid pitch (GRID_STEP) -- no separate line-indexing/homography machinery
// needed. The trick: ray-cast every above-threshold gradient pixel onto the
// floor plane using an ASSUMED, deliberately-arbitrary distance (see
// ASSUMED_FLOOR_DISTANCE below). Getting that assumed distance wrong doesn't
// warp the projection's shape (similar triangles -- wrong distance only
// ever rescales the projected image uniformly, never introduces keystoning,
// since orientation is already exactly known). So the checkerboard's grid
// lines still project as a perfectly regular grid, just at some apparent
// period that's wrong by exactly the same scale factor as the assumed
// distance. Squish-accumulating gradient MAGNITUDE (not brightness) along
// each axis is what makes that period cleanly measurable: magnitude peaks
// at every cell boundary regardless of which color the cell is, so it's
// blind to the pseudorandom bit content and shows a strictly periodic
// signal at the grid pitch. Squishing along U also isolates the COL-family
// lines specifically (a col line, constant in U, reinforces at one U value
// across every V summed over; a row line, constant in V, exists at every U
// and smears into a flat baseline instead) -- so no separate family
// classification is needed either. Comparing the measured period against
// the known-true GRID_STEP gives the exact correction factor back to the
// real distance.
const ASSUMED_FLOOR_DISTANCE = 1;

function estimateFloorDistance(
  gray: Float64Array, w: number, h: number, gradRadius: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
  origin: THREE.Vector3, Drow: THREE.Vector3, Dcol: THREE.Vector3, Dnormal: THREE.Vector3,
): SpacingEstimate | null {
  // Dnormal's sign out of fitPairOfPlanes is arbitrary (it's just the
  // near-zero eigenvalue's eigenvector) -- orient it so a straight-ahead
  // ray actually hits the floor in front of the camera, not behind it.
  const normal = Dnormal.clone();
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  if (cornerDir(0, 0, quat, vFovRad, aspect).dot(normal) > 0) normal.negate();

  const r = gradRadius;
  const us: number[] = [], vs: number[] = [], mags: number[] = [];
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  const hit = new THREE.Vector3();
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const fx = gray[i + r] - gray[i - r];
      const fy = gray[i + r * w] - gray[i - r * w];
      const mag = Math.hypot(fx, fy);
      if (mag <= 0) continue;
      const [ndcU, ndcV] = toNDC(x, y);
      const rayDir = cornerDir(ndcU, ndcV, quat, vFovRad, aspect);
      const denom = rayDir.dot(normal);
      // Excludes rays grazing near-parallel to the floor (pointed toward the
      // horizon), not just ones heading away from it: as a ray approaches
      // parallel, t = -ASSUMED_FLOOR_DISTANCE/denom blows up, so a tiny
      // angular error there becomes a huge world-position error. Left
      // unfiltered, those few extreme hits dominate minU/maxU or minV/maxV,
      // wrecking the bin width for whichever axis happens to run toward the
      // horizon (empirically: one axis came back at 2% distance error, the
      // other at 55%, before this cutoff was added). Threshold is on the
      // ray/normal angle (scale-invariant -- not an absolute distance
      // cutoff, which would depend on the arbitrary ASSUMED_FLOOR_DISTANCE).
      const MIN_GRAZING_COS = 0.15;
      if (denom >= -MIN_GRAZING_COS) continue;
      const t = -ASSUMED_FLOOR_DISTANCE / denom;
      hit.copy(origin).addScaledVector(rayDir, t);
      const u = hit.dot(Drow), v = hit.dot(Dcol);
      us.push(u); vs.push(v); mags.push(mag);
      if (u < minU) minU = u; if (u > maxU) maxU = u;
      if (v < minV) minV = v; if (v > maxV) maxV = v;
    }
  }
  if (us.length === 0 || !isFinite(minU) || !isFinite(minV)) return null;

  // Bin count derived from the existing capture resolution -- no new
  // tunable: never binning coarser or finer than the actual pixel data
  // can support.
  const numBinsU = w, numBinsV = h;
  const binWidthU = (maxU - minU) / numBinsU || 1;
  const binWidthV = (maxV - minV) / numBinsV || 1;
  const profileU = new Float64Array(numBinsU);
  const profileV = new Float64Array(numBinsV);
  for (let k = 0; k < us.length; k++) {
    const bu = Math.min(numBinsU - 1, Math.floor((us[k] - minU) / binWidthU));
    const bv = Math.min(numBinsV - 1, Math.floor((vs[k] - minV) / binWidthV));
    profileU[bu] += mags[k];
    profileV[bv] += mags[k];
  }

  const periodBinsU = autocorrelationPeriod(profileU);
  const periodBinsV = autocorrelationPeriod(profileV);
  if (periodBinsU === null || periodBinsV === null) return null;

  const periodU = periodBinsU * binWidthU; // apparent grid period in the assumed-distance projection's units
  const periodV = periodBinsV * binWidthV;
  return {
    periodU, periodV,
    distanceU: ASSUMED_FLOOR_DISTANCE * (GRID_STEP / periodU),
    distanceV: ASSUMED_FLOOR_DISTANCE * (GRID_STEP / periodV),
  };
}

// Set by runAxesReconstruction on a successful capture; consumed by
// buildProjectedTexture. distance is the average of the U/V estimates --
// both should agree once the grazing-angle cutoff is in place (see
// estimateFloorDistance's header comment), so averaging is just cheap
// noise reduction, not picking one over the other.
interface RecoveredAxes { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3; distance: number }
let lastRecoveredAxes: RecoveredAxes | null = null;

// Rebuilds projectedPreviewData: a bird's-eye, floor-plane-rectified view of
// whichever field view is currently in distortedPreviewData. Same ray-cast-
// onto-the-floor idea as estimateFloorDistance, but keeping full 2D
// structure and each pixel's own color instead of squishing to a 1D
// magnitude profile -- so this works for ANY field view (raw, noised,
// gradient, agreement, ...), not just the gradient magnitude the distance
// recovery itself needs. Bin grid matches rtSize, same "derive from the
// existing capture resolution, no new tunable" convention used everywhere
// else in this file. Needs a successful "capture now" first (lastRecoveredAxes).
function buildProjectedTexture() {
  if (!lastRecoveredAxes) { projectedPreviewData.fill(0); projectedPreviewTex.needsUpdate = true; lastProjectedBins = null; lastMarginals = null; return; }
  const { Drow, Dcol, Dnormal, distance } = lastRecoveredAxes;
  const w = rtSize.w, h = rtSize.h;
  const vFovRad = THREE.MathUtils.degToRad(gizmoCam.fov);
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, camQuat, vFovRad, RT_ASPECT).dot(normal) > 0) normal.negate();
  // NOT the same toNDC as computeWorldVotes/estimateFloorDistance -- those
  // receive gray flipped to top-down first (flipRowsF64), so row 0 -> NDC
  // v=+1 (top) is correct for them. This function reads colors straight
  // from distortedPreviewData, which is GL-native BOTTOM-UP (row 0 =
  // bottom, matching distortedPreviewTex's flipY=false convention
  // everywhere else it's used) -- so row 0 has to map to NDC v=-1 here, not
  // +1. Using the top-down formula on bottom-up data paired each pixel's
  // color with the ray meant for its vertical mirror row, and since the
  // ray-cast/plane-intersection step is nonlinear, feeding it mirrored
  // input coordinates is NOT the same as mirroring its correct output --
  // that's what actually produced the sheared, not-90-degree result rather
  // than a simple upside-down (but still right-angled) image.
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, (py / h) * 2 - 1];

  const MIN_GRAZING_COS = 0.15; // see estimateFloorDistance's header comment
  const hit = new THREE.Vector3();
  const us: number[] = [], vs: number[] = [], srcIdx: number[] = [];
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [ndcU, ndcV] = toNDC(x, y);
      const rayDir = cornerDir(ndcU, ndcV, camQuat, vFovRad, RT_ASPECT);
      const denom = rayDir.dot(normal);
      if (denom >= -MIN_GRAZING_COS) continue;
      const t = -distance / denom;
      // Relative to the camera (origin at the camera itself), NOT
      // camPos-inclusive -- deliberately so this same ray-cast can double
      // as the non-cheating basis for position decode (runPositionDecode),
      // which needs "where is this floor point relative to ME" without
      // ever reading the ground-truth camPos. Harmless for display/
      // periodicity purposes too: translating every hit point by the same
      // constant (camPos) wouldn't change the picture's shape or the
      // measured period at all.
      hit.copy(rayDir).multiplyScalar(t);
      const u = hit.dot(Drow), v = hit.dot(Dcol);
      us.push(u); vs.push(v); srcIdx.push(y * w + x);
      if (u < minU) minU = u; if (u > maxU) maxU = u;
      if (v < minV) minV = v; if (v > maxV) maxV = v;
    }
  }
  if (!isFinite(minU) || !isFinite(minV)) { projectedPreviewData.fill(0); projectedPreviewTex.needsUpdate = true; lastProjectedBins = null; lastMarginals = null; return; }

  const binWidthU = (maxU - minU) / w || 1;
  const binWidthV = (maxV - minV) / h || 1;
  lastProjectedBins = { minU, maxU, minV, maxV, binWidthU, binWidthV, w, h };
  const sums = new Float64Array(w * h * 3);
  const counts = new Float64Array(w * h);
  for (let k = 0; k < us.length; k++) {
    const bu = Math.min(w - 1, Math.max(0, Math.floor((us[k] - minU) / binWidthU)));
    const bv = Math.min(h - 1, Math.max(0, Math.floor((vs[k] - minV) / binWidthV)));
    const bi = bv * w + bu;
    const srcO = srcIdx[k] * 4;
    sums[bi * 3] += distortedPreviewData[srcO];
    sums[bi * 3 + 1] += distortedPreviewData[srcO + 1];
    sums[bi * 3 + 2] += distortedPreviewData[srcO + 2];
    counts[bi]++;
  }
  for (let bi = 0; bi < w * h; bi++) {
    const c = counts[bi];
    const o = bi * 4;
    if (c > 0) {
      projectedPreviewData[o] = Math.round(sums[bi * 3] / c);
      projectedPreviewData[o + 1] = Math.round(sums[bi * 3 + 1] / c);
      projectedPreviewData[o + 2] = Math.round(sums[bi * 3 + 2] / c);
      projectedPreviewData[o + 3] = 255;
    } else {
      projectedPreviewData[o] = 0; projectedPreviewData[o + 1] = 0; projectedPreviewData[o + 2] = 0; projectedPreviewData[o + 3] = 255;
    }
  }
  projectedPreviewTex.needsUpdate = true;
  lastMarginals = computeProjectedMarginals();
}

// Column sums (U axis, varies with x) and row sums (V axis, varies with y)
// of projectedPreviewData's CELL-BOUNDARY edge strength, not raw brightness.
// The de Bruijn torus's bit content is deliberately pseudorandom (that's the
// whole point -- unique windows everywhere), so adjacent cells' brightness
// values are uncorrelated and a raw-brightness profile has no special
// structure at lag = one cell width. A spike train of edge strength at cell
// BOUNDARIES, however, recurs at exactly the cell period regardless of which
// side of each boundary is bright vs dark -- the same reason
// estimateFloorDistance bins gradient magnitude rather than raw brightness.
// Computed as a simple finite difference directly on the already-rectified
// projectedPreviewData (cheaper and just as valid as re-deriving gradients
// from the source image, since rectification has already made cell
// boundaries axis-aligned here).
interface Marginals { colSum: Float64Array; rowSum: Float64Array; colPeriod: number | null; rowPeriod: number | null; colPhase: number; rowPhase: number }
// Cached by buildProjectedTexture's caller (see below) so drawMarginalLines
// can redraw every frame (cheap: just two line-graph passes) without
// rescanning projectedPreviewData every frame too (not as cheap, and only
// actually changes when buildProjectedTexture itself reruns).
let lastMarginals: Marginals | null = null;
function computeProjectedMarginals(): Marginals {
  const w = rtSize.w, h = rtSize.h;
  const lum = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    lum[i] = (projectedPreviewData[o] + projectedPreviewData[o + 1] + projectedPreviewData[o + 2]) / 3;
  }
  const colSum = new Float64Array(w);
  const rowSum = new Float64Array(h);
  for (let y = 0; y < h; y++) {
    for (let x = 1; x < w; x++) colSum[x] += Math.abs(lum[y * w + x] - lum[y * w + x - 1]);
  }
  for (let x = 0; x < w; x++) {
    for (let y = 1; y < h; y++) rowSum[y] += Math.abs(lum[y * w + x] - lum[(y - 1) * w + x]);
  }
  const colPeriod = autocorrelationPeriod(colSum);
  const rowPeriod = autocorrelationPeriod(rowSum);
  const colPhase = colPeriod ? findPhase(colSum, colPeriod) : 0;
  const rowPhase = rowPeriod ? findPhase(rowSum, rowPeriod) : 0;
  return { colSum, rowSum, colPeriod, rowPeriod, colPhase, rowPhase };
}

// Locates the nearest cell-BOUNDARY peak within one period, as a fractional
// bin index in [0, period) -- a weighted circular mean (de-meaned profile as
// weight, angle = 2*pi*i/period) rather than a literal peak search, so it's
// robust to the profile being noisy/asymmetric rather than a clean spike.
// Needed later so bit-sampling (inverse projection, runPositionDecode) can
// target true cell CENTERS (half a GRID_STEP off whatever boundary this
// finds) instead of guessing an arbitrary phase.
function findPhase(profile: Float64Array, period: number): number {
  let mean = 0;
  for (let i = 0; i < profile.length; i++) mean += profile[i];
  mean /= profile.length;
  let sc = 0, ss = 0;
  for (let i = 0; i < profile.length; i++) {
    const wgt = profile[i] - mean;
    const theta = (2 * Math.PI * i) / period;
    sc += wgt * Math.cos(theta);
    ss += wgt * Math.sin(theta);
  }
  let phase = (Math.atan2(ss, sc) / (2 * Math.PI)) * period;
  if (phase < 0) phase += period;
  return phase;
}

// Draws the two marginal-accumulation line graphs beside the Projected Cam
// viewport: a vertical strip to the right (V/row accumulation, one sample
// per image row) and a horizontal strip below (U/column accumulation, one
// sample per image column) -- (x,y,w,h) is the same letterboxed rect
// animate() just rendered the projected viewport into.
const MARGINAL_THICKNESS = 90;
function drawMarginalLines(x: number, y: number, w: number, h: number) {
  if (!lastMarginals) { hideMarginalLines(); return; }
  const m = lastMarginals;

  marginalRightCanvas.style.display = 'block';
  marginalRightCanvas.style.left = (x + w) + 'px';
  marginalRightCanvas.style.top = y + 'px';
  marginalRightCanvas.width = MARGINAL_THICKNESS;
  marginalRightCanvas.height = Math.round(h);
  marginalRightCanvas.style.width = MARGINAL_THICKNESS + 'px';
  marginalRightCanvas.style.height = h + 'px';
  const rc = marginalRightCtx;
  rc.clearRect(0, 0, marginalRightCanvas.width, marginalRightCanvas.height);
  {
    const n = m.rowSum.length;
    let maxV = 0;
    for (let i = 0; i < n; i++) if (m.rowSum[i] > maxV) maxV = m.rowSum[i];
    rc.strokeStyle = '#6cf';
    rc.lineWidth = 1;
    rc.beginPath();
    for (let i = 0; i < n; i++) {
      const py = (i / n) * marginalRightCanvas.height;
      const px = maxV > 0 ? (m.rowSum[i] / maxV) * (MARGINAL_THICKNESS - 4) : 0;
      if (i === 0) rc.moveTo(px, py); else rc.lineTo(px, py);
    }
    rc.stroke();
    if (m.rowPeriod) {
      rc.strokeStyle = 'rgba(255,80,80,0.6)';
      for (let py = m.rowPhase; py < n; py += m.rowPeriod) {
        const yy = (py / n) * marginalRightCanvas.height;
        rc.beginPath(); rc.moveTo(0, yy); rc.lineTo(MARGINAL_THICKNESS, yy); rc.stroke();
      }
    }
  }

  marginalBottomCanvas.style.display = 'block';
  marginalBottomCanvas.style.left = x + 'px';
  marginalBottomCanvas.style.top = (y + h) + 'px';
  marginalBottomCanvas.width = Math.round(w);
  marginalBottomCanvas.height = MARGINAL_THICKNESS;
  marginalBottomCanvas.style.width = w + 'px';
  marginalBottomCanvas.style.height = MARGINAL_THICKNESS + 'px';
  const bc = marginalBottomCtx;
  bc.clearRect(0, 0, marginalBottomCanvas.width, marginalBottomCanvas.height);
  {
    const n = m.colSum.length;
    let maxV = 0;
    for (let i = 0; i < n; i++) if (m.colSum[i] > maxV) maxV = m.colSum[i];
    bc.strokeStyle = '#6cf';
    bc.lineWidth = 1;
    bc.beginPath();
    for (let i = 0; i < n; i++) {
      const px = (i / n) * marginalBottomCanvas.width;
      const py = maxV > 0 ? (m.colSum[i] / maxV) * (MARGINAL_THICKNESS - 4) : 0;
      if (i === 0) bc.moveTo(px, py); else bc.lineTo(px, py);
    }
    bc.stroke();
    if (m.colPeriod) {
      bc.strokeStyle = 'rgba(255,80,80,0.6)';
      for (let px = m.colPhase; px < n; px += m.colPeriod) {
        const xx = (px / n) * marginalBottomCanvas.width;
        bc.beginPath(); bc.moveTo(xx, 0); bc.lineTo(xx, MARGINAL_THICKNESS); bc.stroke();
      }
    }
  }

  if (positionReadout) {
    const uStep = m.colPeriod && lastProjectedBins ? m.colPeriod * lastProjectedBins.binWidthU : null;
    const vStep = m.rowPeriod && lastProjectedBins ? m.rowPeriod * lastProjectedBins.binWidthV : null;
    positionReadout.textContent =
      `col period: ${m.colPeriod ?? '—'} bins (phase ${m.colPhase.toFixed(1)})\n` +
      `row period: ${m.rowPeriod ?? '—'} bins (phase ${m.rowPhase.toFixed(1)})\n` +
      `implied grid step: U=${uStep?.toFixed(3) ?? '—'}  V=${vStep?.toFixed(3) ?? '—'}\n` +
      `(expect both ≈ ${GRID_STEP})`;
  }
}

function hideMarginalLines() {
  marginalRightCanvas.style.display = 'none';
  marginalBottomCanvas.style.display = 'none';
}

// main.ts's own mirrorRowsGrid isn't exported (it's local to that module) --
// reimplemented here rather than exported for a one-line helper. Covers the
// row-mirror half of the D4 (8-fold) symmetry ambiguity; pickBestCandidate's
// own 4-way rotation search covers the rest -- {identity, row-mirror} x
// {0,90,180,270} is 8 elements, and any single reflection composed with all
// 4 rotations spans the full 8-element dihedral group, so together they
// cover every possible axis swap/sign-flip of our (u,v) sampling grid
// relative to the torus's own (row,col) convention -- no need to separately
// worry about which sign/labeling lastRecoveredAxes's Drow/Dcol happen to be.
function mirrorRowsGrid(sg: SampledGrid): SampledGrid {
  return { ...sg, cells: sg.cells.slice().reverse() };
}

// Odd so that reversing the row (or col) array maps the middle index to
// itself (mirrorRowsGrid(sg).cells[floor(n/2)] is exactly sg.cells[floor(n/2)]
// reversed-in-place, not some OTHER row) -- keeps pickBestCandidate's
// "match is the grid's center cell" guarantee pointing at the exact same
// physical anchor point regardless of which of the 8 candidates wins,
// without needing the homography-inverse correction the real line pipeline
// needs (see scripts/test-lines-decode.ts's decodeViaLines comment) -- that
// correction compensates for the sampled grid's origin floating at an
// arbitrary offset from the camera's target, which doesn't apply here since
// this grid is built centered on the camera's own floor anchor by construction.
const DECODE_WINDOW = ORDER * 4 + 1;

interface PositionDecodeResult {
  row: number; col: number; consistency: number;
  camPos: THREE.Vector3;
}
let lastPositionDecode: PositionDecodeResult | null = null;

// Decodes the camera's absolute world position: samples a DECODE_WINDOW
// square of bits at exact floor-cell centers around the camera (via inverse
// projection straight from the analysis grayscale, NOT the lossy forward-
// binned projectedPreviewData), decodes which torus (row,col) that window
// sits at via the same pickBestCandidate machinery the real line pipeline
// uses, then combines the decoded absolute cell position with its known
// position RELATIVE to the camera to recover the camera's own absolute
// position -- the only ground truth this touches is which torus (row,col) a
// bit pattern implies (a property of the pattern itself, verifiable from the
// image alone), never camPos.
function runPositionDecode(gray: Float64Array, w: number, h: number, vFovRad: number) {
  if (!lastRecoveredAxes || !lastMarginals || lastMarginals.colPeriod === null || lastMarginals.rowPeriod === null || !lastProjectedBins) {
    lastPositionDecode = null;
    return;
  }
  const { Drow, Dcol, Dnormal, distance } = lastRecoveredAxes;
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, camQuat, vFovRad, RT_ASPECT).dot(normal) > 0) normal.negate();
  const invQuat = camQuat.clone().invert();
  const halfV = vFovRad / 2;
  const bin = binarize(gray);

  // Sub-cell phase: findPhase locked onto the nearest cell-BOUNDARY peak (see
  // computeProjectedMarginals) in BIN units of the projected buffer; running
  // that bin position back through lastProjectedBins' own scale into world
  // units, then folding to a single representative offset within one
  // GRID_STEP of zero, gives exactly where a boundary sits relative to the
  // camera along each axis -- a cell CENTER is exactly half a GRID_STEP past
  // that. Every other sampled cell is just that anchor plus an integer
  // GRID_STEP hop -- GRID_STEP itself is a known construction constant of
  // the pattern (not something recovered), so hopping by exactly 1 avoids
  // compounding the period measurement's own noise across many cells.
  const uBoundaryRaw = lastProjectedBins.minU + lastMarginals.colPhase * lastProjectedBins.binWidthU;
  const vBoundaryRaw = lastProjectedBins.minV + lastMarginals.rowPhase * lastProjectedBins.binWidthV;
  const uAnchor = (uBoundaryRaw - Math.round(uBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;
  const vAnchor = (vBoundaryRaw - Math.round(vBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;

  const rows = DECODE_WINDOW, cols = DECODE_WINDOW;
  const centerI = Math.floor(rows / 2), centerJ = Math.floor(cols / 2);
  const p = new THREE.Vector3();
  const local = new THREE.Vector3();
  const cells: SampledCell[][] = [];
  for (let i = 0; i < rows; i++) {
    const v = vAnchor + (i - centerI) * GRID_STEP;
    const rowCells: SampledCell[] = [];
    for (let j = 0; j < cols; j++) {
      const u = uAnchor + (j - centerJ) * GRID_STEP;
      // Relative-to-camera world point at this cell's exact center (same
      // "hit" construction as buildProjectedTexture/estimateFloorDistance:
      // p.dot(normal) == -distance for every floor point), then rotated into
      // camera-local space (inverse of cornerDir's forward rotation) and run
      // through cornerDir's pinhole formula backwards to find the pixel it
      // projects to.
      p.copy(Drow).multiplyScalar(u).addScaledVector(Dcol, v).addScaledVector(normal, -distance);
      local.copy(p).applyQuaternion(invQuat);
      if (local.z >= 0) { rowCells.push({ x: NaN, y: NaN, bit: 0, valid: false, cornerCount: 0 }); continue; }
      const ndcU = -local.x / (local.z * Math.tan(halfV) * RT_ASPECT);
      const ndcV = -local.y / (local.z * Math.tan(halfV));
      const px = ((ndcU + 1) / 2) * w, py = ((1 - ndcV) / 2) * h;
      const xx = Math.round(px), yy = Math.round(py);
      if (xx < 0 || xx >= w || yy < 0 || yy >= h) { rowCells.push({ x: px, y: py, bit: 0, valid: false, cornerCount: 0 }); continue; }
      rowCells.push({ x: px, y: py, bit: bin[yy * w + xx], valid: true, cornerCount: 0 });
    }
    cells.push(rowCells);
  }
  const sg: SampledGrid = { rows, cols, cells, originRow: centerI, originCol: centerJ };
  const result = pickBestCandidate([sg, mirrorRowsGrid(sg)], ORDER, debruijnLookup, torus, R, C);
  if (!result.match) { lastPositionDecode = null; return; }

  // worldX = c - C/2, worldZ = r - R/2 (GRID_STEP=1) -- the floor mesh's own
  // world<->torus coordinate convention, reverse-engineered from its
  // PlaneGeometry + rotation.x=-pi/2 + CanvasTexture setup (see the floor
  // mesh construction above). result.match is the torus position of OUR
  // sampled grid's (centerI,centerJ) cell -- i.e. of the point (uAnchor,vAnchor)
  // relative to the camera -- so camPos = that cell's true world position
  // minus its known camera-relative offset.
  const worldPosTrue = new THREE.Vector3((result.match.col - C / 2) * GRID_STEP, 0, (result.match.row - R / 2) * GRID_STEP);
  const hitRel = new THREE.Vector3().addScaledVector(Drow, uAnchor).addScaledVector(Dcol, vAnchor).addScaledVector(normal, -distance);
  const camPosRecovered = worldPosTrue.sub(hitRel);
  lastPositionDecode = { row: result.match.row, col: result.match.col, consistency: result.consistency, camPos: camPosRecovered };
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
      const votes = computeWorldVotes(gray, w, h, state.simGradRadius, state.coherenceRadius, camQuat, vFovRad, RT_ASPECT);
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

      // fitPairOfPlanes can't tell from the math alone which of its two
      // recovered orthogonal directions is "row" vs "col" (a genuine
      // ambiguity, not a bug in the fit itself -- see its header comment).
      // Resolved here, ONCE, against ground truth (fine for this testbed,
      // which has it available) -- and reused everywhere downstream
      // (distance recovery, lastRecoveredAxes, the readout) so they all
      // agree on the same labeling. Previously this correction only
      // affected the readout's displayed text, while lastRecoveredAxes kept
      // whatever raw (possibly swapped) labels the fit produced --
      // "Projected Cam" then used those unswapped axes, which showed up as
      // a fully swapped u/v projection despite the readout itself looking
      // correct.
      let rowDirRecovered: THREE.Vector3 | null = null, colDirRecovered: THREE.Vector3 | null = null;
      if (quadricPair) {
        const errRowFromRow = angleBetweenDegV(quadricPair.Drow, ROW_DIR);
        const errColFromRow = angleBetweenDegV(quadricPair.Dcol, ROW_DIR);
        const flipped = errRowFromRow > errColFromRow;
        rowDirRecovered = flipped ? quadricPair.Dcol : quadricPair.Drow;
        colDirRecovered = flipped ? quadricPair.Drow : quadricPair.Dcol;
      }

      // Distance-to-floor recovery needs the axes above, so it only runs on
      // a successful orientation fit -- see estimateFloorDistance's header
      // comment for the squish-accumulate + autocorrelation approach.
      const spacing = rowDirRecovered && colDirRecovered && quadricPair
        ? estimateFloorDistance(gray, w, h, state.simGradRadius, camQuat, vFovRad, RT_ASPECT, camPos, rowDirRecovered, colDirRecovered, quadricPair.Dnormal)
        : null;
      const t3 = performance.now();

      // Feeds "Projected Cam" mode (see buildProjectedTexture) -- averaging
      // the U/V distance estimates is just noise reduction, not picking one
      // over the other (both should agree once past the grazing-angle cutoff).
      lastRecoveredAxes = rowDirRecovered && colDirRecovered && quadricPair && spacing
        ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: quadricPair.Dnormal, distance: (spacing.distanceU + spacing.distanceV) / 2 }
        : null;
      // Unconditional now (previously gated to state.mode === 'projected'):
      // position decode below depends on the marginal periodicity/phase this
      // computes, regardless of which mode is currently on screen, not just
      // on the visualization it was originally built for.
      buildProjectedTexture();
      runPositionDecode(gray, w, h, vFovRad);
      updateRecoveredCamGizmo();
      applyRecoveredFloorOverlay();
      // lastRecoveredAxes just changed but captureDirty may not have (this
      // capture didn't necessarily touch any camera/distortion slider), so
      // the reconstructed-contamination overlay needs its own explicit
      // refresh here rather than waiting on the dirty-gated block above.
      if (state.mode === 'through') updateContaminationOverlays();
      const t4 = performance.now();

      const lines = [`${votes.length} votes  (${fitVotes.length} fed to fit)`];
      if (rowDirRecovered && colDirRecovered) {
        const rowErr = angleBetweenDegV(rowDirRecovered, ROW_DIR);
        const colErr = angleBetweenDegV(colDirRecovered, COL_DIR);
        lines.push(`row err ${rowErr.toFixed(2)}°  col err ${colErr.toFixed(2)}°`);
      } else {
        lines.push(`degenerate fit`);
      }
      if (spacing) {
        // Ground truth: the floor sits at world y=0, so the camera's true
        // distance to it along the true normal is just its own height.
        const trueDist = camPos.y;
        const distU = spacing.distanceU, distV = spacing.distanceV;
        const errU = (Math.abs(distU - trueDist) / trueDist) * 100;
        const errV = (Math.abs(distV - trueDist) / trueDist) * 100;
        lines.push(`dist U ${distU.toFixed(2)} (${errU.toFixed(1)}% err)  dist V ${distV.toFixed(2)} (${errV.toFixed(1)}% err)  true ${trueDist.toFixed(2)}`);
      } else if (quadricPair) {
        lines.push(`spacing: no period found`);
      }
      lines.push(`votes ${(t1 - t0).toFixed(0)}ms  fit ${(t2 - t1).toFixed(0)}ms  spacing ${(t3 - t2).toFixed(0)}ms  decode ${(t4 - t3).toFixed(0)}ms`);
      axesReadout.textContent = lines.join('\n');

      if (lastPositionDecode) {
        const rec = lastPositionDecode.camPos;
        const errPos = rec.distanceTo(camPos);
        positionReadout.textContent =
          `torus cell: row ${lastPositionDecode.row}  col ${lastPositionDecode.col}\n` +
          `consistency: ${(lastPositionDecode.consistency * 100).toFixed(1)}%\n` +
          `recovered camPos: (${rec.x.toFixed(2)}, ${rec.y.toFixed(2)}, ${rec.z.toFixed(2)})\n` +
          `true camPos: (${camPos.x.toFixed(2)}, ${camPos.y.toFixed(2)}, ${camPos.z.toFixed(2)})\n` +
          `error: ${errPos.toFixed(3)} world units`;
      } else {
        positionReadout.textContent = 'position decode: no match (need periodicity + a successful orientation/distance fit)';
      }
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
  persistControl('mode', m);
  for (const k of Object.keys(modeBtns) as Mode[]) modeBtns[k].classList.toggle('active', k === m);
  worldOrbit.enabled = m === 'world';
  insideHint.style.display = m === 'inside' ? 'block' : 'none';
  // Both 'through' and 'projected' are dedicated fullscreen 2D displays,
  // same as each other -- no PIP-in-a-corner needed for either.
  pipFrame.style.display = m === 'through' || m === 'projected' ? 'none' : 'block';
  pipLabel.style.display = m === 'through' || m === 'projected' ? 'none' : 'block';
  // distortedPreviewData may already be current (captureDirty could be
  // false if nothing's changed recently) even the first time switching into
  // this mode, so build once on the transition rather than only on the next
  // dirty-gated update in animate().
  if (m === 'projected') buildProjectedTexture();
  else hideMarginalLines();
  contamToggles.style.display = m === 'through' ? 'flex' : 'none';
  // updateDistortedPreview first: lastNoisedPreviewGray may still be null
  // (cold start) or stale (overlay was off while a 'raw'/'none' view's
  // early-return skipped computing it -- see updateDistortedPreview's
  // needGrayForOverlay) at the moment either toggle first turns on.
  if (m === 'through') { updateDistortedPreview(); updateContaminationOverlays(); }
}
modeBtns.world.addEventListener('click', () => setMode('world'));
modeBtns.through.addEventListener('click', () => setMode('through'));
modeBtns.inside.addEventListener('click', () => setMode('inside'));
modeBtns.projected.addEventListener('click', () => setMode('projected'));

toggleHideFieldBtn.addEventListener('click', () => {
  state.hideField = !state.hideField;
  toggleHideFieldBtn.classList.toggle('active', state.hideField);
  updateDistortedPreview();
  updateContaminationOverlays();
});
toggleTrueContamBtn.addEventListener('click', () => {
  state.showTrueContamination = !state.showTrueContamination;
  toggleTrueContamBtn.classList.toggle('active', state.showTrueContamination);
  updateDistortedPreview();
  updateContaminationOverlays();
});
toggleReconContamBtn.addEventListener('click', () => {
  state.showReconstructedContamination = !state.showReconstructedContamination;
  toggleReconContamBtn.classList.toggle('active', state.showReconstructedContamination);
  updateDistortedPreview();
  updateContaminationOverlays();
});

// ── Slider / checkbox wiring ─────────────────────────────────────────────

bindSlider('camX', (v) => { state.camX = v; markCaptureDirty(); });
bindSlider('camY', (v) => { state.camY = v; markCaptureDirty(); });
bindSlider('camZ', (v) => { state.camZ = v; markCaptureDirty(); });
bindSlider('camYaw', (v) => { state.camYawDeg = v; markCaptureDirty(); }, (v) => `${v.toFixed(0)}°`);
bindSlider('camPitch', (v) => { state.camPitchDeg = v; markCaptureDirty(); }, (v) => `${v.toFixed(0)}°`);
bindSlider('camFocal', (v) => { state.focalMM = v; markCaptureDirty(); }, (v) => `${v.toFixed(0)}mm`);
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
bindCheckbox('showRecoveredFloor', (v) => (state.showRecoveredFloor = v));

bindSlider('simNoise', (v) => { state.simNoise = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindSlider('simBlur', (v) => { state.simBlur = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindSlider('simGradRadius', (v) => { state.simGradRadius = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindSlider('captureSupersample', (v) => { state.captureSupersample = v; resizeCaptureBuffers(); }, (v) => `${v.toFixed(0)}x`);
bindSlider('coherenceRadius', (v) => { state.coherenceRadius = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindRadioGroup('fieldView', (v) => { state.fieldView = v as 'raw' | 'antialiased' | 'downsampled' | 'noised' | 'gradient' | 'agreement' | 'effective'; markCaptureDirty(); });
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
  updateRecoveredCamGizmo();
  recoveredFloorOverlay.visible = state.mode === 'world' && state.showRecoveredFloor && !!lastPositionDecode;
  camHelper.visible = state.mode === 'world' && state.showFrustum;
  floorMesh.visible = state.showFloor;

  // Camera render target feeds the PIP preview, the sphere patch, AND
  // Through-Cam mode -- all three show the same distorted preview texture
  // rather than gizmoCam directly. Only worth redoing when something that
  // actually changes gizmoCam's output has changed (camera details, or a
  // capture-distortion/filter-pipeline tunable -- see markCaptureDirty's
  // call sites), NOT every single frame: panning the WORLD-mode orbit
  // camera, for instance, never touches gizmoCam at all, so re-rendering
  // and reading back a potentially large captureRTSize buffer on every one
  // of those frames was pure waste that made panning feel heavy at larger
  // viewport sizes. Still throttled on top (a full readback + blur/downsample
  // is real CPU cost even when it IS needed -- e.g. while a slider is being
  // actively dragged, firing many 'input' events per second).
  const now = performance.now();
  if (captureDirty && now - lastPreviewUpdate >= PREVIEW_UPDATE_INTERVAL_MS) {
    lastPreviewUpdate = now;
    captureDirty = false;
    renderCamRT();
    updateDistortedPreview();
    if (state.mode === 'projected') buildProjectedTexture();
    if (state.mode === 'through') updateContaminationOverlays();
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
    if (state.showTrueContamination) renderTrueContamOverlay(x, y, w, h);
    if (state.showReconstructedContamination) renderReconContamOverlay(x, y, w, h);
  } else if (state.mode === 'projected') {
    // Same letterbox as 'through' -- the projected grid is built at rtSize
    // dimensions, same aspect as the regular capture.
    const winAspect = innerWidth / innerHeight;
    let w = innerWidth, h = innerHeight, x = 0, y = 0;
    if (winAspect > RT_ASPECT) { w = innerHeight * RT_ASPECT; x = (innerWidth - w) / 2; }
    else { h = innerWidth / RT_ASPECT; y = (innerHeight - h) / 2; }
    renderProjectedViewport(x, y, w, h);
    drawMarginalLines(x, y, w, h);
  } else {
    insideCam.position.copy(camPos);
    euler.set(insidePitch, insideYaw, 0);
    insideCam.quaternion.setFromEuler(euler);
    renderViewport(insideCam, 0, 0, innerWidth, innerHeight);
    renderPreviewViewport(pipRect.x, innerHeight - pipRect.y - pipRect.h, pipRect.w, pipRect.h);
  }
}

// Same persistence as every slider/checkbox (see savedControls) -- only
// honor a saved value if it's still a real Mode, same guard bindRadioGroup
// uses for a renamed/removed option.
const VALID_MODES: Mode[] = ['world', 'through', 'inside', 'projected'];
const savedMode = savedControls['mode'];
setMode(VALID_MODES.includes(savedMode as Mode) ? (savedMode as Mode) : 'world');
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
