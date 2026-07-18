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
import { toGrayscale, binarize, rotateShift } from './decode.ts';
import { jacobiEigenSymmetric, smallestEigenvector } from './linalg.ts';

type Mode = 'world' | 'through' | 'inside' | 'projected';

// ── DOM ──────────────────────────────────────────────────────────────────

const canvas = document.getElementById('gl') as HTMLCanvasElement;
const panel = document.getElementById('panel') as HTMLDivElement;
const panelToggle = document.getElementById('panelToggle') as HTMLButtonElement;
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
const sampleLatticeCanvas = document.getElementById('sampleLattice') as HTMLCanvasElement;
const sampleLatticeCtx = sampleLatticeCanvas.getContext('2d')!;
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
  showSphere: true, showCircles: true, showPoles: true, showFrustum: true, showPatch: true, showFloor: true, showGizmoBody: true, showRecoveredFloor: true, showSampleLattice: false,
  showTrueContamination: false, showReconstructedContamination: false, hideField: false,
  simNoise: 8, simBlur: 1, simGradRadius: 1, coherenceRadius: 1,
  // Guided tangent-walk gradient direction (see conversation) -- fixes
  // per-pixel gradient direction being noise-fragile enough that the
  // closed-form orientation fit (fitPairOfPlanes) sometimes lands outside
  // LM's basin of convergence at realistic noise levels (confirmed live:
  // simNoise=8, the default above, produced a stable, reproducible ~75
  // degree orientation error at one pose; simNoise=1 was reliable). First
  // guesses (20deg/0.35/2) were confirmed live to be too strict -- most
  // walks bailed out after 0-1 samples under simNoise=8's actual per-pixel
  // jitter, never getting the chance to average anything out. These looser
  // values fixed the same pose to a fraction of a degree.
  tangentWalkMaxSteps: 12, tangentWalkDeviationDeg: 45, tangentWalkMagFraction: 0.15, tangentWalkGraceSamples: 3,
  // max was 5 -- bumped after finding live that 1% (drifted there via
  // localStorage during testing, not this default) starved fitPairOfPlanes
  // to ~1500 of 150000 votes, which was fine at easy poses but produced a
  // badly wrong orientation fit (70-90 degree error) at harder ones. 10%
  // fixed that cleanly across the whole pitch range tested -- see
  // conversation.
  circleSamplePercentMin: 0, circleSamplePercentMax: 10,
  showRecoveredPoles: true,
  showAxisVectors: false,
  showTopCircles: true,
  weightSharpenPower: 4,
  orientationLM: true,
  positionLM: true,
  fieldView: 'noised' as 'raw' | 'antialiased' | 'downsampled' | 'noised' | 'gradient' | 'walked' | 'agreement' | 'effective',
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
    // 1 -> dark, 0 -> light -- matches scripts/generate-debruijn-torus.ts's
    // canonical convention (cell ? 0 : 255) and binarize's own documented
    // "dark -> 1" intent (src/decode.ts). This was backwards (1 -> bright)
    // until now, which inverted every simulated bit read here relative to
    // the real torus/debruijnLookup content -- the actual root cause of the
    // whole session's poor decode consistency, not any of the axis/mirror
    // bugs fixed earlier: pt.bit came out as the bitwise complement of the
    // true window almost everywhere, so tallyPositionVotes' lookup rarely
    // matched the real anchor.
    const v = torus[r][c] ? 20 : 235;
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
    // alpha can now exceed 1 (computeGradientAgreementField is unnormalized,
    // see its comment) -- clamp before writing into this Uint8Array, which
    // wraps (mod 256) rather than clamping on its own, unlike
    // Uint8ClampedArray.
    out[o + 3] = Math.min(255, Math.round(alpha[i] * 255));
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

  // -Drow, NOT Drow: PlaneGeometry's local +X maps to this texture's U=0
  // edge, which is bu=0 -- and bu runs maxU -> minU (buildProjectedTexture's
  // mirror, see its comment), so local -X/2 (U=0) must land at world offset
  // +Drow*(width/2) (i.e. u=maxU), not -Drow*(width/2). Using -Drow here
  // gets that right.
  //
  // Cross product, NOT the sign-corrected `normal` above -- guarantees a
  // right-handed, purely-rotational basis (setFromRotationMatrix on a
  // reflection matrix, which using a wrongly-signed normal could produce,
  // gives a garbage quaternion). Only affects which face is "front"; with
  // DoubleSide + a paper-thin plane that's invisible, so it's safe to let
  // this differ from the physically-signed normal used for the offset above.
  const drowDisplay = Drow.clone().negate();
  const zAxis = new THREE.Vector3().crossVectors(drowDisplay, Dcol).normalize();
  const basis = new THREE.Matrix4().makeBasis(drowDisplay, Dcol, zAxis);
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

// Two-pass (horizontal then vertical) box blur -- O(w*h) total, independent
// of radius, instead of src/lines.ts's boxBlur, which is a single
// O(radius^2) pass (fine there: it only ever runs at small radius on an
// already-small image). An earlier version of this function resummed each
// pixel's whole clamped window from scratch (O(radius) per pixel per pass,
// O(w*h*radius) total) -- fine at small radius, but this runs on a
// captureSupersample x buffer at a correspondingly larger radius, where
// that cost showed up as the top frame in a profile of "capture now".
// Below instead keeps a running sum per row/column and slides it by one
// pixel at a time: entering pixel added, leaving pixel subtracted, O(1)
// work per step regardless of radius. The clamped-window edge behavior
// (shrunken, correctly-divided average for the first/last `radius`
// pixels) is unchanged -- lo/hi still track the live window bounds, just
// incrementally instead of recomputed from x/y each time.
function separableBoxBlur(src: Float64Array, w: number, h: number, radius: number): Float64Array {
  if (radius <= 0) return src.slice();
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let lo = 0, hi = Math.min(w - 1, radius);
    let sum = 0;
    for (let i = lo; i <= hi; i++) sum += src[row + i];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum / (hi - lo + 1);
      if (x + 1 >= w) continue;
      const nextHi = Math.min(w - 1, x + 1 + radius);
      if (nextHi > hi) { hi = nextHi; sum += src[row + hi]; }
      const nextLo = Math.max(0, x + 1 - radius);
      if (nextLo > lo) { sum -= src[row + lo]; lo = nextLo; }
    }
  }
  for (let x = 0; x < w; x++) {
    let lo = 0, hi = Math.min(h - 1, radius);
    let sum = 0;
    for (let i = lo; i <= hi; i++) sum += tmp[i * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / (hi - lo + 1);
      if (y + 1 >= h) continue;
      const nextHi = Math.min(h - 1, y + 1 + radius);
      if (nextHi > hi) { hi = nextHi; sum += tmp[hi * w + x]; }
      const nextLo = Math.max(0, y + 1 - radius);
      if (nextLo > lo) { sum -= tmp[lo * w + x]; lo = nextLo; }
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
    } else if (state.fieldView === 'walked') {
      // Same per-pixel direction computeWorldVotes actually casts votes
      // with (see guidedTangentDirection's own comment) -- lets 'gradient'
      // and this view be flipped between directly to see what the walk
      // changes, instead of only inferring it from the final orientation
      // error.
      const field = computeGradientField(noised, rtSize.w, rtSize.h, Math.round(state.simGradRadius));
      const { fx, fy, r } = field;
      const walkedFx = new Float64Array(fx.length), walkedFy = new Float64Array(fy.length);
      for (let y = r; y < rtSize.h - r; y++) {
        for (let x = r; x < rtSize.w - r; x++) {
          const i = y * rtSize.w + x;
          if (fx[i] === 0 && fy[i] === 0) continue;
          const walked = guidedTangentDirection(fx, fy, rtSize.w, rtSize.h, x, y, fx[i], fy[i]);
          walkedFx[i] = walked.fx; walkedFy[i] = walked.fy;
        }
      }
      paintVectorFieldAsColor({ fx: walkedFx, fy: walkedFy, w: rtSize.w, h: rtSize.h, r }, distortedPreviewData);
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
// window.
//
// Normalized against this frame's own RAW (unsmoothed) max magnitude, NOT
// the max of the smoothed field itself -- confirmed empirically (see
// conversation) that the smoothed max is not a stable reference: it drops
// sharply as aggRadius grows, since even the single most locally-coherent
// pixel's window picks up some real disagreement once the window is big
// enough. Dividing by THAT shrinking max inflated every other pixel's
// normalized score at the same time real corner contamination was
// (correctly) shrinking -- contamination visibly "spread" onto clean
// running edges as the window grew, not because those edges got any less
// directionally consistent. Tried skipping normalization entirely too
// (also see conversation): that failed the opposite way -- badnessAlpha's
// small residual on a genuinely clean edge (~0.01-0.05, from ordinary
// sensor noise) got multiplied by raw gradient magnitude (tens to hundreds
// in this scene), swamping the actual signal and saturating alpha to
// "opaque" on nearly every edge in the frame, even at aggRadius=0.
// The raw max magnitude is stable across aggRadius (it doesn't depend on
// the smoothing window at all) while still keeping output bounded to
// roughly [0,1] -- hypot(avg(cx),avg(cy)) <= avg(mag) <= this max, by the
// same convexity argument the old max-of-smoothed version relied on.
function computeGradientAgreementField(field: GradientField, aggRadius: number): Float64Array {
  const { fx, fy, w, h } = field;
  const n = w * h;
  const cx = new Float64Array(n), cy = new Float64Array(n);
  let maxRawMag = 0;
  for (let i = 0; i < n; i++) {
    const mag = Math.hypot(fx[i], fy[i]);
    if (mag > maxRawMag) maxRawMag = mag;
    if (mag === 0) continue;
    const theta = Math.atan2(fy[i], fx[i]);
    cx[i] = mag * Math.cos(2 * theta);
    cy[i] = mag * Math.sin(2 * theta);
  }
  const sx = separableBoxBlur(cx, w, h, aggRadius);
  const sy = separableBoxBlur(cy, w, h, aggRadius);
  const agreement = new Float64Array(n);
  for (let i = 0; i < n; i++) agreement[i] = Math.hypot(sx[i], sy[i]);
  if (maxRawMag > 0) for (let i = 0; i < n; i++) agreement[i] /= maxRawMag;
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

// Guided tangent walk: instead of estimating a pixel's edge direction from
// one small isotropic sample (fragile under noise -- confirmed live,
// simNoise=8 produces a stable, reproducible ~75 degree orientation error
// at one pose purely from per-pixel direction noise, not vote count, since
// vote count is identical at simNoise=1 where the same pose is accurate to
// a fraction of a degree), grow outward along the TANGENT direction (the
// edge's own direction, perpendicular to the gradient) on both sides
// independently, averaging each new sample's direction into a running
// estimate. Averaging several independent samples of "which way does this
// edge point," taken at different points along its own length, is the
// direct fix for noise corrupting any ONE sample -- this is local
// structure-tensor estimation (the same eigenvector-of-summed-outer-
// products idea fitPairOfPlanes already does globally over all votes,
// applied here locally per-pixel first); growing preferentially along the
// tangent rather than isotropically is coherence-enhancing anisotropic
// diffusion territory (Weickert) -- see conversation for the fuller
// discussion and why isotropic growth alone fails (it eventually samples
// neighboring grid lines running the OTHER direction too).
//
// Reuses the already-computed whole-image (fx,fy) field for every step
// (same seed radius throughout) rather than re-differencing gray at each
// stepped-to pixel -- cheap, and keeps every sample directly comparable.
// Direction is FIXED for the whole walk, set once from the seed pixel's own
// gradient (not adaptively re-steered sample to sample) -- v1; adaptive
// tracking (following a walk that curves slightly under perspective
// foreshortening) is a deferred follow-up, see conversation.
//
// Each side (+tangent, -tangent) grows and stops independently -- no
// reason to discard good samples on one side just because the other hit a
// wall. Two stop conditions, each needing tangentWalkGraceSamples
// CONSECUTIVE violations (not one bad sample) before actually cutting off:
// direction deviates from the running average by more than
// tangentWalkDeviationDeg (wandered into competing structure -- a
// neighboring grid line), or magnitude drops below tangentWalkMagFraction
// of the running average (wandered into a flat, edge-less interior). The
// grace window exists because a genuinely straight edge crossing a
// PERPENDICULAR edge partway along its length shows a few confused samples
// right at that intersection, then continues cleanly -- a single-sample
// trip-wire would permanently truncate the walk exactly there instead of
// pushing through, the same reasoning as Canny's hysteresis thresholding
// bridging small gaps between genuinely-connected edge segments.
//
// Direction is accumulated via the double-angle fold (cos(2*theta),
// sin(2*theta)) computeGradientAgreementField already uses for the same
// reason: a black->white and a white->black edge along the SAME physical
// line are 180 degrees apart in raw (fx,fy) but describe the same line, so
// a raw vector sum would wrongly let them cancel where they should
// reinforce.
function guidedTangentDirection(
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { fx: number; fy: number } {
  const seedTheta = Math.atan2(seedFy, seedFx);
  const tdx = -Math.sin(seedTheta), tdy = Math.cos(seedTheta);
  const seedMag = Math.hypot(seedFx, seedFy);
  let sumCos = Math.cos(2 * seedTheta) * seedMag;
  let sumSin = Math.sin(2 * seedTheta) * seedMag;
  let runningMag = seedMag;
  let sampleCount = 1;
  const maxSteps = state.tangentWalkMaxSteps;
  // cos(2*threshold) once, so each step's check is a single dot-product
  // comparison instead of an atan2 + subtraction per sample.
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(state.tangentWalkDeviationDeg));
  const magFraction = state.tangentWalkMagFraction;
  const grace = state.tangentWalkGraceSamples;
  for (const sign of [1, -1]) {
    let violations = 0;
    for (let k = 1; k <= maxSteps; k++) {
      const sx = Math.round(x + sign * k * tdx), sy = Math.round(y + sign * k * tdy);
      if (sx < 0 || sx >= w || sy < 0 || sy >= h) break;
      const si = sy * w + sx;
      const sfx = fx[si], sfy = fy[si];
      const mag = Math.hypot(sfx, sfy);
      if (mag === 0 || mag < runningMag * magFraction) {
        violations++;
        if (violations >= grace) break;
        continue; // tolerate a short bad run (e.g. a corner crossing) without accumulating it
      }
      const theta = Math.atan2(sfy, sfx);
      const c2 = Math.cos(2 * theta), s2 = Math.sin(2 * theta);
      const avgLen = Math.hypot(sumCos, sumSin);
      const cosDeviation = avgLen > 0 ? (c2 * sumCos + s2 * sumSin) / avgLen : 1;
      if (cosDeviation < devCos) {
        violations++;
        if (violations >= grace) break;
        continue;
      }
      violations = 0;
      sumCos += c2 * mag; sumSin += s2 * mag;
      runningMag = (runningMag * sampleCount + mag) / (sampleCount + 1);
      sampleCount++;
    }
  }
  const avgTheta = Math.atan2(sumSin, sumCos) / 2; // undo the doubling
  // Magnitude stays the seed's own single-pixel value -- only DIRECTION is
  // walk-refined here; "how strong is this pixel's edge" for vote-weight
  // purposes is a separate question from "which way does it point,"
  // already handled by computeGradientAgreementField's own weighting.
  return { fx: Math.cos(avgTheta) * seedMag, fy: Math.sin(avgTheta) * seedMag };
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
      // Noise-robust direction (see guidedTangentDirection's own comment) --
      // magnitude for vote weight below stays the seed's own single-pixel
      // value, only direction comes from the walk.
      const walked = guidedTangentDirection(fx, fy, w, h, x, y, fx[i], fy[i]);
      let theta = Math.atan2(walked.fy, walked.fx);
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

// ── Orientation refinement (Levenberg-Marquardt) ─────────────────────────
//
// fitPairOfPlanes minimizes an ALGEBRAIC residual (n^T M n, a linear-algebra
// proxy with no direct angular meaning) via one closed-form eigensolve --
// fast, but only an approximation of what actually matters: whether each
// vote's normal ends up truly perpendicular to the recovered Drow or Dcol.
// This refines that closed-form answer against the real GEOMETRIC residual,
// the same "cheap algebraic init + iterative nonlinear geometric refine"
// pattern used throughout classical multi-view geometry (DLT+bundle
// adjustment, EPnP+LM, IPPE, etc. -- see conversation).
//
// Residual, per vote: decompose n onto the (Drow,Dcol) plane as
// (a,b) = (n.Drow, n.Dcol), and let psi = atan2(b,a). n should be
// perpendicular to EITHER Drow (a=0) OR Dcol (b=0) -- both "good" and
// indistinguishable from this objective alone (see below) -- which are
// exactly the points where psi is a multiple of 90 degrees; diagonal
// (a=b, 45 degrees off both) is the worst case. sin(4*psi) folds this into
// one smooth scalar: 0 at all 4 good angles, extremal at all 4 bad ones --
// the same "double the angle to fold a symmetry away" trick used elsewhere
// in this file (computeGradientAgreementField etc.), just QUADRUPLED
// instead of doubled, since there are two separate symmetries to fold here
// (n vs -n, AND Drow-aligned vs Dcol-aligned both counting as "good").
//
// This objective is PROVABLY BLIND to which of the 8 combinations of
// {Drow<->Dcol swap, Drow sign, Dcol sign} is "true": swapping a<->b sends
// psi -> pi/2 - psi, and negating either sends psi -> (const) - psi; both
// map sin(4*psi) -> -sin(4*psi), which leaves its SQUARE (what LM actually
// minimizes) unchanged. So this refinement cannot resolve, and does not
// attempt to resolve, the row/col-swap or axis-sign ambiguity -- that's
// still handled exactly as before (ground truth, testbed-only), just
// applied to the REFINED Drow/Dcol below instead of fitPairOfPlanes' raw
// output.
function fourFoldResidual(n: THREE.Vector3, Drow: THREE.Vector3, Dcol: THREE.Vector3): number {
  const psi = Math.atan2(n.dot(Dcol), n.dot(Drow));
  return Math.sin(4 * psi);
}

interface OrientationFit { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 }

// Weighted sum of squared residuals -- the scalar Levenberg-Marquardt
// actually works to shrink. Same effective-gradient weight fitPairOfPlanes
// itself uses (mag * agreement), unsharpened -- sharpening was
// fitPairOfPlanes' own way of trusting its strongest votes more within a
// single closed-form solve; here weight just scales each vote's
// contribution to the sum, no separate tuning knob needed for a small
// iterative refinement.
function orientationCost(votes: Vote[], Drow: THREE.Vector3, Dcol: THREE.Vector3): number {
  let cost = 0;
  for (const { n, weight } of votes) {
    const r = weight * fourFoldResidual(n, Drow, Dcol);
    cost += r * r;
  }
  return cost;
}

// General NxN linear solve via Gaussian elimination with partial pivoting --
// small, fixed-size systems only (LM normal equations below, at most a
// handful of parameters across Phases 1-3), no need for anything more
// sophisticated. Supersedes an earlier Cramer's-rule version hardcoded to
// 3x3 (Phase 1's rotation-only refinement) -- Phase 2 widens the parameter
// count to 5 (rotation + phase), so this needed to generalize anyway.
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(M[row][col]) > Math.abs(M[pivot][col])) pivot = row;
    if (Math.abs(M[pivot][col]) < 1e-18) return null;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let k = col; k <= n; k++) M[row][k] -= f * M[col][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let s = M[row][n];
    for (let k = row + 1; k < n; k++) s -= M[row][k] * x[k];
    x[row] = s / M[row][row];
  }
  return x;
}

// Refines fitPairOfPlanes' closed-form (Drow,Dcol,Dnormal) against the true
// geometric alignment residual above, via Levenberg-Marquardt over a
// 3-parameter Lie-algebra (axis-angle) perturbation of the current
// orientation -- the standard manifold-optimization pattern for refining a
// rotation: at each step, express the next candidate as a SMALL rotation
// composed onto the current one (qNext = exp(delta) * qCurrent), rather
// than optimizing quaternion components directly under a unit-norm
// constraint. Jacobian is numerical (forward differences) -- simpler to get
// right than the analytic derivative through the vote's own construction,
// at the cost of a few extra residual evaluations per iteration; this is a
// 3-parameter problem evaluated over at most a couple thousand votes, so
// that cost is not a concern yet (see conversation on expected perf).
function refineOrientationLM(votes: Vote[], initial: OrientationFit, maxIterations = 20): OrientationFit & { iterations: number; initialCost: number; finalCost: number } {
  const q = new THREE.Quaternion(); // identity: candidate axes start as initial's, rotated by q as q moves off identity
  const Drow0 = initial.Drow.clone(), Dcol0 = initial.Dcol.clone(), Dnormal0 = initial.Dnormal.clone();
  const candidateDrow = (qq: THREE.Quaternion) => Drow0.clone().applyQuaternion(qq);
  const candidateDcol = (qq: THREE.Quaternion) => Dcol0.clone().applyQuaternion(qq);

  const initialCost = orientationCost(votes, Drow0, Dcol0);
  let cost = initialCost;
  let lambda = 1e-3;
  const EPS = 1e-5; // finite-difference step, radians
  const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];

  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    const Drow = candidateDrow(q), Dcol = candidateDcol(q);
    const n = votes.length;
    const residuals = new Float64Array(n);
    for (let i = 0; i < n; i++) residuals[i] = votes[i].weight * fourFoldResidual(votes[i].n, Drow, Dcol);

    // Numerical Jacobian: d(residual_i)/d(perturbation along world axis k),
    // via a small rotation about each of the 3 world axes LEFT-composed
    // onto q (i.e. applied in the global frame, on top of the current
    // estimate -- qPlus.multiply(q) in THREE's convention applies q first,
    // then the perturbation).
    const J: Float64Array[] = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
    for (let k = 0; k < 3; k++) {
      const qPlus = new THREE.Quaternion().setFromAxisAngle(axes[k], EPS).multiply(q);
      const DrowP = candidateDrow(qPlus), DcolP = candidateDcol(qPlus);
      for (let i = 0; i < n; i++) {
        const rP = votes[i].weight * fourFoldResidual(votes[i].n, DrowP, DcolP);
        J[k][i] = (rP - residuals[i]) / EPS;
      }
    }

    // Normal equations (JtJ + lambda*diag(JtJ)) delta = -Jtr, 3x3.
    const JtJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const Jtr = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += J[a][i] * J[b][i];
        JtJ[a][b] = s;
      }
      let s = 0;
      for (let i = 0; i < n; i++) s += J[a][i] * residuals[i];
      Jtr[a] = s;
    }
    const A = JtJ.map((row, a) => row.map((v, b) => v + (a === b ? lambda * (JtJ[a][a] || 1) : 0)));
    const rhs = Jtr.map((v) => -v);
    const delta = solveLinearSystem(A, rhs);
    if (!delta) break; // singular normal equations -- shouldn't happen for a well-posed, direction-diverse vote set, but bail cleanly if it does

    const deltaVec = new THREE.Vector3(delta[0], delta[1], delta[2]);
    const deltaAngle = deltaVec.length();
    if (deltaAngle < 1e-10) break; // converged: no meaningful step left
    const deltaAxis = deltaVec.normalize(); // mutates deltaVec in place; deltaAngle already captured above
    const qTry = new THREE.Quaternion().setFromAxisAngle(deltaAxis, deltaAngle).multiply(q).normalize();

    const DrowTry = candidateDrow(qTry), DcolTry = candidateDcol(qTry);
    const tryCost = orientationCost(votes, DrowTry, DcolTry);
    if (tryCost < cost) {
      q.copy(qTry);
      cost = tryCost;
      lambda = Math.max(lambda * 0.5, 1e-8);
    } else {
      lambda = Math.min(lambda * 3, 1e8);
    }
  }

  return {
    Drow: candidateDrow(q), Dcol: candidateDcol(q), Dnormal: Dnormal0.clone().applyQuaternion(q),
    iterations, initialCost, finalCost: cost,
  };
}


// ── Phase 3 (Option B): joint orientation + ABSOLUTE position refinement ─
// EXPERIMENTAL / standalone, not wired into the pipeline yet -- same
// discipline as Phase 2. Where Option A's wrapped-distance residual is
// pattern-AGNOSTIC (every grid line looks the same to it, so it has no way
// to distinguish the true line from a neighboring wrong one -- confirmed
// live this is exactly what broke the V axis at a harder pose, converging
// confidently to the wrong local minimum), this compares against the
// KNOWN, globally-unique bit content directly: predicted brightness at a
// candidate ABSOLUTE world position vs the actually-observed brightness.
// That breaks the symmetry a wrapped residual can't -- a wrong cell
// alignment predicts the wrong BIT, not just "some distance off".
//
// The raw predicted bit is a step function (constant except exactly AT a
// cell boundary) -- a naive finite-difference Jacobian against it would be
// zero almost everywhere, giving LM nothing to work with. predictedBilinear
// below smooths this via bilinear interpolation between the 4 nearest
// cells, the simplest thing that actually has a usable gradient everywhere
// -- NOT yet the full coarse-to-fine image pyramid from the original plan
// (see conversation); that's a bigger step, added only if bilinear alone
// turns out to have too narrow a basin of convergence once tested.

// torus[r][c] convention: 1 -> dark (20), 0 -> light (235) -- see the floor
// texture's own comment, matching scripts/generate-debruijn-torus.ts and
// binarize's "dark -> 1" intent (both established earlier this session).
function torusBrightness(row: number, col: number): number {
  const r = ((row % R) + R) % R, c = ((col % C) + C) % C;
  return torus[r][c] ? 20 : 235;
}

// Bilinear blend of the 4 torus cells nearest (worldX,worldZ) -- smooth,
// differentiable prediction of what the sensor should see there. Cell c's
// CENTER sits at integer "cell-index" coordinate xf = worldX + C/2 - 0.5
// (established earlier this session, confirmed via raycast against the
// real floor mesh: an integer xf lands exactly on a cell's UV center).
function predictedBilinear(worldX: number, worldZ: number): number {
  const xf = worldX + C / 2 - 0.5, zf = worldZ + R / 2 - 0.5;
  const c0 = Math.floor(xf), r0 = Math.floor(zf);
  const fx = xf - c0, fz = zf - r0;
  const b00 = torusBrightness(r0, c0), b10 = torusBrightness(r0, c0 + 1);
  const b01 = torusBrightness(r0 + 1, c0), b11 = torusBrightness(r0 + 1, c0 + 1);
  return b00 * (1 - fx) * (1 - fz) + b10 * fx * (1 - fz) + b01 * (1 - fx) * fz + b11 * fx * fz;
}

// Photometric sample: just a screen pixel + its OBSERVED brightness --
// unlike Option A's edge-focused votes, this deliberately does NOT need a
// gradient/tangent at all, and should NOT be edge-weighted the same way:
// edge pixels are exactly where the observed image is blurriest/most
// ambiguous (mid-transition between two cells), while flat, confidently-one-
// cell interior pixels are where a photometric comparison is most reliable.
// Sampled on a regular screen-space stride instead of by gradient magnitude
// for that reason -- roughly uniform coverage across the visible floor,
// not concentrated on the hardest-to-match pixels.
interface PhotometricSample { px: number; py: number; observed: number }

function computePhotometricSamples(gray: Float64Array, w: number, h: number, stride: number): PhotometricSample[] {
  const samples: PhotometricSample[] = [];
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      samples.push({ px: x, py: y, observed: gray[y * w + x] });
    }
  }
  return samples;
}

interface PositionFit extends OrientationFit { worldX0: number; worldZ0: number; distance: number }

// Joint LM over 5 parameters: 3-DOF rotation (same Lie-algebra perturbation
// as Phases 1-2) + worldX0/worldZ0, the ABSOLUTE world position of the
// (u=0,v=0) floor-intersection point (i.e. roughly where the camera sits
// over the floor) -- NOT a wrapped sub-cell phase like Option A, a genuine
// absolute coordinate, since the photometric residual needs to know WHICH
// cell, not just how far from the nearest boundary.
function refineOrientationAndPositionLM(
  samples: PhotometricSample[], w: number, h: number,
  initial: OrientationFit, distance: number, initialWorldX0: number, initialWorldZ0: number,
  camQuat: THREE.Quaternion, vFovRad: number, aspect: number,
  maxIterations = 20,
): PositionFit & { iterations: number; initialCost: number; finalCost: number } {
  const q = new THREE.Quaternion();
  const Drow0 = initial.Drow.clone(), Dcol0 = initial.Dcol.clone(), Dnormal0 = initial.Dnormal.clone();
  let worldX0 = initialWorldX0, worldZ0 = initialWorldZ0;
  const MIN_GRAZING_COS = 0.15;
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, (py / h) * 2 - 1];

  const candidateNormal = (qq: THREE.Quaternion) => {
    const n = Dnormal0.clone().applyQuaternion(qq);
    if (cornerDir(0, 0, camQuat, vFovRad, aspect).dot(n) > 0) n.negate();
    return n;
  };

  function residualsFor(qq: THREE.Quaternion, wx0: number, wz0: number): Float64Array {
    const Drow = Drow0.clone().applyQuaternion(qq), Dcol = Dcol0.clone().applyQuaternion(qq);
    const normal = candidateNormal(qq);
    const out: number[] = [];
    for (const s of samples) {
      const [ndcU, ndcV] = toNDC(s.px, s.py);
      const rayDir = cornerDir(ndcU, ndcV, camQuat, vFovRad, aspect);
      const denom = rayDir.dot(normal);
      if (denom >= -MIN_GRAZING_COS) continue;
      const hit = rayDir.multiplyScalar(-distance / denom);
      const u = hit.dot(Drow), v = hit.dot(Dcol);
      const predicted = predictedBilinear(wx0 + u, wz0 + v);
      out.push(predicted - s.observed);
    }
    return new Float64Array(out);
  }

  const cost = (r: Float64Array) => { let s = 0; for (let i = 0; i < r.length; i++) s += r[i] * r[i]; return s; };
  const initialCost = cost(residualsFor(q, worldX0, worldZ0));
  let curCost = initialCost;
  let lambda = 1e-3;
  const EPS_ROT = 1e-5, EPS_POS = 1e-3;
  const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const P = 5;

  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    const r0 = residualsFor(q, worldX0, worldZ0);
    const n = r0.length;
    if (n === 0) break;

    const J: Float64Array[] = [];
    for (let k = 0; k < 3; k++) {
      const qPlus = new THREE.Quaternion().setFromAxisAngle(axes[k], EPS_ROT).multiply(q);
      const rP = residualsFor(qPlus, worldX0, worldZ0);
      const len = Math.min(n, rP.length);
      const col = new Float64Array(n);
      for (let i = 0; i < len; i++) col[i] = (rP[i] - r0[i]) / EPS_ROT;
      J.push(col);
    }
    for (const [dx, dz] of [[EPS_POS, 0], [0, EPS_POS]]) {
      const rP = residualsFor(q, worldX0 + dx, worldZ0 + dz);
      const len = Math.min(n, rP.length);
      const col = new Float64Array(n);
      const eps = dx || dz;
      for (let i = 0; i < len; i++) col[i] = (rP[i] - r0[i]) / eps;
      J.push(col);
    }

    const JtJ: number[][] = Array.from({ length: P }, () => new Array(P).fill(0));
    const Jtr: number[] = new Array(P).fill(0);
    for (let a = 0; a < P; a++) {
      for (let b = 0; b < P; b++) {
        let s = 0; for (let i = 0; i < n; i++) s += J[a][i] * J[b][i];
        JtJ[a][b] = s;
      }
      let s = 0; for (let i = 0; i < n; i++) s += J[a][i] * r0[i];
      Jtr[a] = s;
    }
    const A = JtJ.map((row, a) => row.map((v, b) => v + (a === b ? lambda * (JtJ[a][a] || 1) : 0)));
    const rhs = Jtr.map((v) => -v);
    const delta = solveLinearSystem(A, rhs);
    if (!delta) break;

    const deltaRotVec = new THREE.Vector3(delta[0], delta[1], delta[2]);
    const deltaRotAngle = deltaRotVec.length();
    const deltaWX = delta[3], deltaWZ = delta[4];
    if (deltaRotAngle < 1e-10 && Math.abs(deltaWX) < 1e-10 && Math.abs(deltaWZ) < 1e-10) break;

    const qTry = deltaRotAngle > 1e-12
      ? new THREE.Quaternion().setFromAxisAngle(deltaRotVec.normalize(), deltaRotAngle).multiply(q).normalize()
      : q.clone();
    const wx0Try = worldX0 + deltaWX, wz0Try = worldZ0 + deltaWZ;

    const tryCost = cost(residualsFor(qTry, wx0Try, wz0Try));
    if (tryCost < curCost) {
      q.copy(qTry); worldX0 = wx0Try; worldZ0 = wz0Try;
      curCost = tryCost;
      lambda = Math.max(lambda * 0.5, 1e-8);
    } else {
      lambda = Math.min(lambda * 3, 1e8);
    }
  }

  return {
    Drow: Drow0.clone().applyQuaternion(q), Dcol: Dcol0.clone().applyQuaternion(q), Dnormal: candidateNormal(q),
    worldX0, worldZ0, distance,
    iterations, initialCost, finalCost: curCost,
  };
}

// De-means the profile, then finds the lag (in bins) of the strongest
// non-trivial autocorrelation peak -- i.e. the period of whatever periodic
// signal dominates it. Searches from a small minimum lag (skipping the
// always-large near-zero-lag correlation, uninformative for period-finding)
// up to half the profile length (beyond that too little overlap remains to
// trust the estimate). O(n * maxLag); cheap at the profile sizes here (a
// handful of ms even at n=2000).
//
// Two extra steps beyond a textbook autocorrelation, both needed to survive
// steep-pitch/off-axis poses where the profile carries a strong SMOOTH,
// BROADBAND component (the frustum wedge's occupancy envelope -- more rows
// contribute gradient energy to buckets near the middle of the visible
// footprint than near its grazing edges) on top of the genuine periodic
// ripple:
//   1. Detrend first (subtract a wide box-smoothed version of the profile).
//      A smooth envelope is, by construction, always most self-similar at
//      the smallest possible lag -- its own autocorrelation is a
//      monotonically DECAYING ramp with no real peak, just an edge sitting
//      at minLag. Left in, that edge can outscore the true periodic peak
//      outright (confirmed live: colMag's raw lag-2 "correlation" tracked a
//      smoothly yaw-varying quantity with no periodic meaning at all, while
//      the genuine grid-period peak at the true period weakened in relative
//      terms at the same poses).
//   2. Require the winning lag to be a genuine LOCAL peak (score strictly
//      greater than both neighbors), not just whatever lag happens to score
//      highest overall -- a monotonic decay ramp's "best" point is always
//      its first sample (minLag), which step 1 usually removes but doesn't
//      always fully suppress (confirmed live: one pose's ramp survived
//      detrending and still out-scored the true period peak). Requiring a
//      local peak rejects that ramp-edge on structural grounds regardless
//      of its remaining amplitude, since a ramp edge only ever falls off to
//      one side. Falls back to the plain global max if no lag qualifies as
//      a local peak (nothing periodic to find either way).
function autocorrelationPeriod(profile: Float64Array): number | null {
  const n = profile.length;

  // Step 1: detrend -- subtract a wide box-smoothed version so only
  // shorter-range (periodic-scale) structure survives into the correlation.
  const detrendWin = 41;
  const half = Math.floor(detrendWin / 2);
  const detrended = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half), hi = Math.min(n - 1, i + half);
    let s = 0, c = 0;
    for (let j = lo; j <= hi; j++) { s += profile[j]; c++; }
    detrended[i] = profile[i] - s / c;
  }

  let mean = 0;
  for (let i = 0; i < n; i++) mean += detrended[i];
  mean /= n;
  const centered = new Float64Array(n);
  for (let i = 0; i < n; i++) centered[i] = detrended[i] - mean;

  const minLag = Math.max(2, Math.floor(n * 0.005));
  const maxLag = Math.floor(n / 2);
  const scores = new Float64Array(maxLag - minLag);
  let bestLagAny = -1, bestScoreAny = -Infinity;
  for (let lag = minLag; lag < maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) sum += centered[i] * centered[i + lag];
    scores[lag - minLag] = sum;
    if (sum > bestScoreAny) { bestScoreAny = sum; bestLagAny = lag; }
  }

  // Step 2: collect every genuine local peak (strictly greater than both
  // neighbors), then prefer the SMALLEST-lag one that's still comparably
  // strong -- not simply the single highest-scoring peak. A true fundamental
  // period's harmonics (2x, 3x, ...) are also local peaks by construction
  // (twice a real period is still a lag where the signal realigns with
  // itself), and their score is frequently close to -- sometimes even a
  // little above -- the fundamental's own, especially once the profile has
  // any asymmetry. Picking the highest-scoring peak outright confirmed live
  // to land on exactly 2x the true period at two poses once step 1's ramp-
  // edge candidate was suppressed (their real local peak was still there,
  // just outscored by its own octave). Threshold is deliberately loose
  // (half the best peak's score) since the goal is only to distinguish
  // "real peak, possibly a harmonic" from "noise floor," not to rank peaks
  // finely -- once something clears that bar, smallest-lag-first is the
  // tiebreaker that picks the fundamental over its harmonics.
  let bestScorePeak = -Infinity;
  const peaks: number[] = [];
  for (let lag = minLag + 1; lag < maxLag - 1; lag++) {
    const s = scores[lag - minLag];
    if (s > scores[lag - minLag - 1] && s > scores[lag - minLag + 1]) {
      peaks.push(lag);
      if (s > bestScorePeak) bestScorePeak = s;
    }
  }
  const peakThreshold = bestScorePeak * 0.5;
  let bestLagPeak = -1;
  for (const lag of peaks) {
    if (scores[lag - minLag] >= peakThreshold) { bestLagPeak = lag; break; } // peaks[] is in ascending-lag order
  }

  const bestLag = bestLagPeak > 0 ? bestLagPeak : bestLagAny;
  if (bestLag <= 0) return null;

  // Step 3: sub-bin refinement -- integer-lag precision is a real limit at
  // high camera altitudes, where the visible floor extent (and so the
  // bucket grid's world-units-per-bin) grows large enough that a whole grid
  // cell only spans a handful of bins. A true period like 4.2 bins has
  // nowhere to go in an integer search except round to 3, 4, or 5 -- a
  // 20%+ error that lands directly in the recovered distance (confirmed
  // live: colPeriod=3 measured against an expected ~4.2 at one high-altitude
  // pose, with distance off by a correlated ~22%). Standard parabolic fit
  // through the winning bin and its two immediate neighbors' scores
  // recovers the peak's true (non-integer) location cheaply, without
  // needing a finer bucket grid -- same trick pitch/frequency estimators
  // use to beat their own bin resolution.
  const i = bestLag - minLag;
  if (i > 0 && i < scores.length - 1) {
    const y0 = scores[i - 1], y1 = scores[i], y2 = scores[i + 1];
    const denom = y0 - 2 * y1 + y2;
    if (denom !== 0) {
      const delta = 0.5 * (y0 - y2) / denom;
      if (Math.abs(delta) < 1) return bestLag + delta; // sanity bound -- a real peak's vertex stays within the sampled interval
    }
  }
  return bestLag;
}

// Distance-to-floor recovery no longer has its own separate implementation
// here -- it used to (a gradient-magnitude ray-cast at an assumed distance,
// squish-accumulated into per-axis profiles, rescaled by the known
// GRID_STEP), but that was a second, parallel implementation of exactly the
// same trick buildProjectedTexture/computeProjectedMarginals already do for
// position decode's periodicity. The ray-cast/bin formula scales linearly
// with whatever distance is assumed (hit(d) = (-d/denom)*rayDir for a fixed
// ray), and bin ASSIGNMENT is completely invariant to that scale (minU,
// maxU, and binWidthU all scale by the same factor, which cancels out of
// floor((u-minU)/binWidthU)) -- so there's no need for a separate ray-cast
// pass at all. runAxesReconstruction now: builds the projected texture once
// at an arbitrary placeholder distance, reads the period computeProjectedMarginals
// already measures, solves for the true distance in closed form against the
// known GRID_STEP (identical rescaling trick, just reusing the already-tuned
// marginal-line pipeline), writes that back into lastRecoveredAxes.distance,
// and rebuilds once more now that it's correct.

// Set by runAxesReconstruction on a successful capture; consumed by
// buildProjectedTexture. distance is the average of the U/V estimates --
// both should agree once the grazing-angle cutoff is in place (see
// buildProjectedTexture's own MIN_GRAZING_COS comment), so averaging is
// just cheap noise reduction, not picking one over the other.
interface RecoveredAxes { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3; distance: number }
let lastRecoveredAxes: RecoveredAxes | null = null;

// Casts one ray per SCREEN pixel (always at rtSize resolution -- that's the
// actual captured pixel grid regardless of how finely the result gets
// bucketed) and bins the hits into a bucketW x bucketH grid, independent of
// the screen resolution. Split out of buildProjectedTexture so the
// period-measurement refinement pass below (see runAxesReconstruction) can
// re-bucket the SAME rays at a finer resolution than the rtSize-sized
// display texture uses -- needed because a fixed bucket count tied to
// display resolution means a grid cell can shrink to just a handful of bins
// at high camera altitudes (confirmed live: colPeriod measured 3 bins
// against an expected ~4.2 at one high-altitude pose, a 20%+ error that
// propagated straight into the recovered distance -- not a peak-picking bug,
// genuinely too few samples per cycle for ANY discrete-lag method to trust).
function castAndBucketProjectedSamples(bucketW: number, bucketH: number): {
  bins: ProjectedBins; sums: Float64Array; counts: Float64Array; gradCxSum: Float64Array; gradCySum: Float64Array;
} | null {
  if (!lastRecoveredAxes) return null;
  const { Drow, Dcol, Dnormal, distance } = lastRecoveredAxes;
  const w = rtSize.w, h = rtSize.h;
  const vFovRad = THREE.MathUtils.degToRad(gizmoCam.fov);
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, camQuat, vFovRad, RT_ASPECT).dot(normal) > 0) normal.negate();
  // NOT the same toNDC as computeWorldVotes uses -- that receives gray
  // flipped to top-down first (flipRowsF64), so row 0 -> NDC
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

  // Excludes rays grazing near-parallel to the floor (pointed toward the
  // horizon): as a ray approaches parallel, t = -distance/denom blows up, so
  // a tiny angular error there becomes a huge world-position error. Left
  // unfiltered, those few extreme hits dominate minU/maxU or minV/maxV,
  // wrecking the bin width for whichever axis happens to run toward the
  // horizon. Threshold is on the ray/normal angle (scale-invariant -- not an
  // absolute distance cutoff, which would depend on whatever distance is
  // currently assumed).
  const MIN_GRAZING_COS = 0.15;
  const hit = new THREE.Vector3();
  const hit2 = new THREE.Vector3();
  const us: number[] = [], vs: number[] = [], srcIdx: number[] = [];
  // Per-sample gradient, ALREADY expressed in the (u,v) frame -- see the big
  // comment below, right where these get consumed, for the full derivation.
  // Computed here (not in the bucket loop below) because it needs the same
  // per-pixel ray-cast this loop is already doing, one extra time for a
  // tangent-shifted screen pixel.
  const gradCxAtSample: number[] = [], gradCyAtSample: number[] = [];
  const srcGrad = lastNoisedPreviewGray ? computeGradientField(lastNoisedPreviewGray, w, h, 1) : null;
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

      // A gradient is a covector: it does NOT carry over unchanged when you
      // relocate it into a different coordinate frame (screen pixels here,
      // (u,v) world-floor coordinates below) -- it transforms by the local
      // Jacobian of the map between them, which perspective makes vary
      // continuously across the image (most anisotropically exactly near
      // grazing angles, i.e. exactly where periodicity/phase were breaking
      // down -- see conversation). computeWorldVotes already handles this
      // correctly for ORIENTATION recovery, by casting a tangent-shifted
      // screen pixel through the real camera model and cross-producting the
      // two resulting 3D rays instead of trusting the raw screen-space
      // angle. This does the analogous thing one step further, all the way
      // to the floor: cast that same tangent-shifted pixel through to its
      // OWN (u,v) hit point, and use the (u,v)-space DISPLACEMENT between
      // the two hits as the tangent direction, instead of the raw
      // screen-space one -- no Jacobian ever needs to be written down
      // explicitly, it falls out of doing the same ray-cast twice.
      let cxAtSample = 0, cyAtSample = 0;
      if (srcGrad) {
        const si = y * w + x;
        const fx = srcGrad.fx[si], fy = srcGrad.fy[si];
        const mag = Math.hypot(fx, fy);
        if (mag > 0) {
          const theta = Math.atan2(fy, fx);
          // Tangent direction (along the edge, perpendicular to the
          // gradient) -- same construction computeWorldVotes uses.
          const tdx = -Math.sin(theta), tdy = Math.cos(theta);
          const [ndcU2, ndcV2] = toNDC(x + tdx, y + tdy);
          const rayDir2 = cornerDir(ndcU2, ndcV2, camQuat, vFovRad, RT_ASPECT);
          const denom2 = rayDir2.dot(normal);
          if (denom2 < -MIN_GRAZING_COS) {
            const t2 = -distance / denom2;
            hit2.copy(rayDir2).multiplyScalar(t2);
            const u2 = hit2.dot(Drow), v2 = hit2.dot(Dcol);
            const du = u2 - u, dv = v2 - v;
            // Direction only for now, NOT also rescaling magnitude by
            // 1/hypot(du,dv) -- tried that (see conversation), but hypot(du,dv)
            // is "how many (u,v) world-units one screen pixel spans here",
            // which can be tiny for heavily-OVERSAMPLED near-camera pixels
            // (confirmed live: magUV spiked to 1076 vs an average of 56 on
            // the working baseline pose), and those few extreme outliers
            // dominated colSum/rowSum and broke periodicity worse than the
            // original bug. A correct version needs to cap that rescaling
            // against the bucket width, which isn't known until AFTER this
            // whole loop finishes (minU/maxU aren't final yet) -- a real
            // two-pass restructure, not a quick clamp here.
            if (Math.hypot(du, dv) > 1e-9) {
              // (du,dv) is a TANGENT vector (the pushforward of a screen
              // direction), which transports validly through ANY invertible
              // map, conformal or not -- straightforward "where does this
              // direction end up". A GRADIENT is a COVECTOR: it transforms
              // by the inverse-transpose of the local Jacobian, not by
              // rotating the transported tangent 90 degrees -- that
              // rotate-the-tangent step is only valid for CONFORMAL
              // (angle-preserving) maps, and perspective projection isn't
              // conformal (that's the entire reason this correction exists
              // -- different axes get squeezed by different amounts). Tried
              // it anyway first: it's not just a sign error, cos(2*)/sin(2*)
              // are pi-periodic so a naive sign flip on the atan2 arguments
              // provably can't fix it (confirmed live: identical, still-
              // broken output before and after flipping the sign).
              //
              // Sidesteps the whole covector-transform question the same
              // way computeWorldVotes does (cross-producting rays instead of
              // rotating a gradient): build the double-angle fold directly
              // from the TANGENT's own angle, algebraically equivalent to
              // the original gradient-angle formula (tangent angle = grad
              // angle + pi/2, so cos(2*tangent) = -cos(2*grad), etc.) but
              // using only the one quantity that's actually valid to
              // transport here.
              const phiUV = Math.atan2(dv, du);
              cxAtSample = -mag * Math.cos(2 * phiUV); // double-angle, same polarity-fold reasoning as computeGradientAgreementField
              cyAtSample = -mag * Math.sin(2 * phiUV);
            }
          }
        }
      }
      gradCxAtSample.push(cxAtSample); gradCyAtSample.push(cyAtSample);
    }
  }
  // Literal min/max, NOT a percentile crop -- tried percentile cropping to
  // fix a pitch-cliff collapse diagnosed earlier in this session (a few
  // outlier grazing rays inflating minU/maxU, wrecking bin resolution), but
  // it turned out not to be the actual mechanism behind either failure mode
  // actually found in testing: one was really an orientation-fit problem
  // (circleSamplePercentMax had drifted too narrow), the other had a
  // perfectly reasonable extent already (10ish units) and still collapsed
  // -- cropping 2% or even 20% of the distribution barely moved the extent
  // in the case it was meant to fix. Reverted rather than carry unproven
  // complexity -- see conversation.
  if (!isFinite(minU) || !isFinite(minV)) return null;

  const binWidthU = (maxU - minU) / bucketW || 1;
  const binWidthV = (maxV - minV) / bucketH || 1;
  const bins: ProjectedBins = { minU, maxU, minV, maxV, binWidthU, binWidthV, w: bucketW, h: bucketH };
  const sums = new Float64Array(bucketW * bucketH * 3);
  const counts = new Float64Array(bucketW * bucketH);
  // Gradient of the SOURCE (un-rectified) analysis-equivalent brightness,
  // NOT the rectified/binned display buffer -- computed once per source
  // pixel (gradCxAtSample/gradCyAtSample above, already re-expressed in the
  // (u,v) frame -- see that big comment), then projected+averaged into
  // buckets the SAME principled way RGB already is below (sum per bucket,
  // divide by that bucket's own count), instead of computeProjectedMarginals
  // previously taking a discrete difference BETWEEN already-averaged
  // neighboring buckets. That old approach faked an edge at every transition
  // into an empty (black) bucket, and made sparser rows/columns look
  // artificially weaker just for having fewer populated buckets summed in,
  // independent of how strong their real data actually was.
  const gradCxSum = new Float64Array(bucketW * bucketH);
  const gradCySum = new Float64Array(bucketW * bucketH);
  // bu runs from maxU down to minU (NOT the naive us[k]-minU), i.e. U
  // increases right-to-left on screen -- deliberately, to cancel out a
  // handedness mismatch that's otherwise baked into this whole axis system.
  // Drow, Dcol, and Dnormal are each independently, correctly sign-fixed
  // against ground truth (ROW_DIR=+X, COL_DIR=+Z, and "faces the camera"
  // respectively -- see runAxesReconstruction and this function's own normal
  // fix above) -- but Drow x Dcol = (+X)x(+Z) = -Y = -Dnormal, always,
  // regardless of camera pose. So (Drow,Dcol,Dnormal), despite every single
  // axis individually being correct, form a LEFT-handed triple -- confirmed
  // empirically too (Jacobian of screen-pixel -> (bu,bv) has a negative
  // determinant without this mirror). That mismatch can't be fixed by
  // re-flipping Drow or Dcol's sign -- both are already pinned to the one
  // sign hitRel/buildDecodeSampleGrid need to reconstruct world position
  // correctly -- so it has to be absorbed here, purely in how U gets turned
  // into a screen column, without touching u itself (still used unmirrored
  // everywhere else: hitRel, the decode grid's world reconstruction, the
  // pole markers). buildDecodeSampleGrid's uBoundaryRaw and
  // drawSampleLattice's bu both invert this exact same formula -- keep all
  // three in sync if this ever changes.
  for (let k = 0; k < us.length; k++) {
    const bu = Math.min(bucketW - 1, Math.max(0, Math.floor((maxU - us[k]) / binWidthU)));
    const bv = Math.min(bucketH - 1, Math.max(0, Math.floor((vs[k] - minV) / binWidthV)));
    const bi = bv * bucketW + bu;
    const si = srcIdx[k];
    const srcO = si * 4;
    sums[bi * 3] += distortedPreviewData[srcO];
    sums[bi * 3 + 1] += distortedPreviewData[srcO + 1];
    sums[bi * 3 + 2] += distortedPreviewData[srcO + 2];
    counts[bi]++;
    gradCxSum[bi] += gradCxAtSample[k];
    gradCySum[bi] += gradCyAtSample[k];
  }
  return { bins, sums, counts, gradCxSum, gradCySum };
}

// Rebuilds projectedPreviewData: a bird's-eye, floor-plane-rectified view of
// whichever field view is currently in distortedPreviewData -- also what
// runAxesReconstruction uses (via computeProjectedMarginals) to recover the
// camera's distance to the floor, since the ray-cast/bin math here is what
// that distance recovery is built on top of. Keeps full 2D structure and
// each pixel's own color instead of squishing to a 1D profile -- so this
// works for ANY field view (raw, noised, gradient, agreement, ...). Bin grid
// matches rtSize, same "derive from the existing capture resolution, no new
// tunable" convention used everywhere else in this file for the DISPLAY
// texture specifically -- the higher-resolution period-measurement pass
// (castAndBucketProjectedSamples, called directly, bypassing this wrapper)
// uses a separate bucket count and never touches projectedPreviewData at
// all. Needs a successful "capture now" first (lastRecoveredAxes) -- called
// TWICE per capture (see runAxesReconstruction): once at a placeholder
// distance just to seed the period-measurement refinement pass, once more
// after distance is corrected so the final projectedPreviewData/
// lastProjectedBins/lastMarginals are all properly scaled.
function buildProjectedTexture() {
  const result = lastRecoveredAxes ? castAndBucketProjectedSamples(rtSize.w, rtSize.h) : null;
  if (!result) { projectedPreviewData.fill(0); projectedPreviewTex.needsUpdate = true; lastProjectedBins = null; lastMarginals = null; return; }
  const { bins, sums, counts, gradCxSum, gradCySum } = result;
  lastProjectedBins = bins;
  for (let bi = 0; bi < bins.w * bins.h; bi++) {
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
  lastMarginals = computeProjectedMarginals(bins.w, bins.h, counts, gradCxSum, gradCySum);
}

// Column sums (U axis, varies with x) and row sums (V axis, varies with y)
// of projectedPreviewData's CELL-BOUNDARY edge strength, not raw brightness.
// The de Bruijn torus's bit content is deliberately pseudorandom (that's the
// whole point -- unique windows everywhere), so adjacent cells' brightness
// values are uncorrelated and a raw-brightness profile has no special
// structure at lag = one cell width. Edge strength (gradient magnitude)
// fixes that -- but plain |magnitude| still throws away
// something real: a genuine vertical grid line's edge pixels all have a
// HORIZONTAL gradient direction (mod 180, cell to cell, regardless of which
// side is bright), while scattered noise/junction/off-axis structure has
// its direction spread more randomly. That's exactly the same distinction
// computeGradientAgreementField's double-angle vector-sum already exploits
// spatially -- reused here along the marginal axis instead: cx = mag*cos(2*
// theta) is +mag for a horizontal gradient (theta=0, i.e. a vertical line),
// -mag for a vertical gradient (theta=90, i.e. a horizontal line), and ~0
// for anything diagonal in between.
//
// colSum/colSumCy (and rowHueCx/rowSumCy) are a genuine 2D vector sum per
// bucket-column/row -- but periodicity/phase detection runs on colMag/rowMag,
// the MAGNITUDE of that summed vector (hypot(colSum,colSumCy)), not the raw
// signed cx component alone. That distinction matters: a large negative cx
// sum (a column dominated by the OTHER axis's lines crossing through it) is,
// in a weighted circular mean, mathematically indistinguishable from a large
// POSITIVE sum at the opposite phase (half a period away) -- confirmed live,
// this was exactly why findPhase locked onto a flat, unremarkable plateau
// instead of the profile's actual dominant feature, which happened to be a
// deep negative trough. Magnitude is always >= 0, so there's no sign left to
// get flipped into the wrong half of the circle. rowSum (the sign-flipped,
// magnitude-irrelevant -cx sum) and rowHueCx (unflipped) both still exist
// only so the display can recover each bin's own dominant gradient direction
// (atan2(cy,cx)/2) for hue -- none of that feeds periodicity/phase anymore.
// Computed on the already-rectified projectedPreviewData's luminance (not
// re-deriving from the source image), since rectification has already made
// cell boundaries axis-aligned here -- works the same regardless of which
// field view happens to be selected for display, same reasoning as
// lastNoisedPreviewGray decoupling the contamination overlays from it.
interface Marginals {
  colSum: Float64Array; rowSum: Float64Array; colSumCy: Float64Array; rowHueCx: Float64Array; rowSumCy: Float64Array;
  colMag: Float64Array; rowMag: Float64Array;
  colPeriod: number | null; rowPeriod: number | null; colPhase: number; rowPhase: number;
}
// Cached by buildProjectedTexture's caller (see below) so drawMarginalLines
// can redraw every frame (cheap: just two line-graph passes) without
// rescanning projectedPreviewData every frame too (not as cheap, and only
// actually changes when buildProjectedTexture itself reruns).
let lastMarginals: Marginals | null = null;
// gradCxSum/gradCySum/counts are buildProjectedTexture's own per-bucket
// accumulators (same source pixels, same bucket assignment, as its RGB
// sum/count) -- see that function's own comment on why this reads the
// SOURCE image's gradient (projected+averaged per bucket, matching how the
// displayed color is already a true per-bucket average) rather than taking
// a discrete difference between already-averaged buckets. A bucket with
// zero samples contributes NOTHING here (skipped outright), not a
// zero-valued sample -- that's what previously faked an edge against every
// empty/black bucket and made sparser rows/columns look weaker just for
// having fewer populated buckets, independent of how strong their real
// data actually was.
function computeProjectedMarginals(w: number, h: number, counts: Float64Array, gradCxSum: Float64Array, gradCySum: Float64Array): Marginals {
  const colSum = new Float64Array(w);
  const colSumCy = new Float64Array(w);
  const rowSum = new Float64Array(h);
  const rowHueCx = new Float64Array(h);
  const rowSumCy = new Float64Array(h);
  for (let bv = 0; bv < h; bv++) {
    for (let bu = 0; bu < w; bu++) {
      const bi = bv * w + bu;
      const c = counts[bi];
      if (c === 0) continue;
      const cx = gradCxSum[bi] / c; // true per-bucket average, not a difference between neighboring bucket averages
      const cy = gradCySum[bi] / c;
      colSum[bu] += cx; colSumCy[bu] += cy;
      rowSum[bv] -= cx; rowHueCx[bv] += cx; rowSumCy[bv] += cy;
    }
  }
  const colMag = new Float64Array(w);
  for (let bu = 0; bu < w; bu++) colMag[bu] = Math.hypot(colSum[bu], colSumCy[bu]);
  const rowMag = new Float64Array(h);
  for (let bv = 0; bv < h; bv++) rowMag[bv] = Math.hypot(rowHueCx[bv], rowSumCy[bv]);

  const colPeriod = autocorrelationPeriod(colMag);
  const rowPeriod = autocorrelationPeriod(rowMag);
  const colPhase = colPeriod ? findPhase(colMag, colPeriod) : 0;
  const rowPhase = rowPeriod ? findPhase(rowMag, rowPeriod) : 0;
  return { colSum, rowSum, colSumCy, rowHueCx, rowSumCy, colMag, rowMag, colPeriod, rowPeriod, colPhase, rowPhase };
}

// Re-buckets castAndBucketProjectedSamples' rays at a resolution sized to
// keep a fixed target of buckets per grid cell -- independent of rtSize
// display resolution -- then applies the same closed-form "true period must
// equal the known GRID_STEP" correction runAxesReconstruction's rough pass
// already uses, against this far more reliable period measurement. Needed
// because a fixed bucket count tied to display resolution leaves only a
// handful of buckets per cell at high camera altitudes (confirmed live:
// colPeriod measured 3 bins against an expected ~4.2 at one high-altitude
// pose, a 20%+ error that landed straight in the recovered distance -- not
// a peak-picking bug, genuinely too few samples per cycle for any
// discrete-lag autocorrelation to trust, see autocorrelationPeriod's own
// comment). Called twice per capture (see runAxesReconstruction): once
// right after the rough placeholder-distance correction, and again after
// Phase 3 improves orientation -- confirmed live that the SAME period
// measurement, redone with Phase 3's ~0.4 degree orientation instead of
// Phase 1's ~2 degree orientation, drops U/V distance error from ~3%/1% to
// ~0.03%/0.36%, since a rotated (Drow,Dcol) basis distorts the ray-cast's
// u,v values (and so the measured extent/period) asymmetrically between
// the two axes -- orientation error was silently leaking into distance
// error this whole time. extentU/extentV must already be in TRUE world
// units at whatever distance is CURRENTLY on lastRecoveredAxes (the caller
// is responsible for that -- see the two call sites for the two different
// ways they satisfy it).
function measurePeriodDistance(currentDistance: number, extentU: number, extentV: number): { distanceU: number; distanceV: number } | null {
  const TARGET_BUCKETS_PER_CELL = 20;
  const MAX_REFINE_BUCKETS = 2048; // memory cap: sums+counts+gradCxSum+gradCySum is ~48 bytes/bucket, so this bounds the transient allocation to a couple hundred MB even at a wildly-wrong distance estimate
  const refineW = Math.min(MAX_REFINE_BUCKETS, Math.max(rtSize.w, Math.ceil(extentU / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const refineH = Math.min(MAX_REFINE_BUCKETS, Math.max(rtSize.h, Math.ceil(extentV / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const refined = castAndBucketProjectedSamples(refineW, refineH); // reads lastRecoveredAxes.distance, which must already equal currentDistance
  const refinedMarginals = refined ? computeProjectedMarginals(refineW, refineH, refined.counts, refined.gradCxSum, refined.gradCySum) : null;
  if (!refined || !refinedMarginals || refinedMarginals.colPeriod === null || refinedMarginals.rowPeriod === null) return null;
  return {
    distanceU: currentDistance * (GRID_STEP / (refinedMarginals.colPeriod * refined.bins.binWidthU)),
    distanceV: currentDistance * (GRID_STEP / (refinedMarginals.rowPeriod * refined.bins.binWidthV)),
  };
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
// Recovers a bin's own dominant gradient direction from its (cx,cy) vector
// sum (undoing the double-angle fold, see computeProjectedMarginals) and
// maps it to a color via the SAME hue convention paintVectorFieldAsColor
// uses (hue=direction), so a bin dominated by a clean axis-aligned edge
// reads as a strong, recognizable color instead of a flat line-graph blue.
function marginalHueColor(cx: number, cy: number): string {
  let theta = Math.atan2(cy, cx) / 2;
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  const [r, g, b] = hsvToRgb((theta / Math.PI) * 360, 1, 1);
  return `rgb(${r},${g},${b})`;
}

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
    const n = m.rowMag.length;
    // Magnitude, always >= 0 (see computeProjectedMarginals) -- plain
    // deflection from the left edge, no centered baseline needed since
    // there's no negative direction to accommodate anymore.
    let maxMag = 0;
    for (let i = 0; i < n; i++) if (m.rowMag[i] > maxMag) maxMag = m.rowMag[i];
    // Per-segment color, not one flat strokeStyle -- each bin's own
    // (rowHueCx,rowSumCy) vector sum picks the hue (see marginalHueColor),
    // so a run of bins dominated by a clean horizontal edge reads as a
    // solid, recognizable color instead of uniform blue.
    rc.lineWidth = 1;
    let prevPx = 0, prevPy = 0;
    for (let i = 0; i < n; i++) {
      // i indexes rowMag the same GL-native bottom-up way bv does
      // everywhere else (see drawSampleLattice's own comment) -- bv=0
      // displays at the visual BOTTOM of the adjacent image
      // (projectedPreviewTex.flipY=false), so this strip needs the same
      // (1 - i/n) flip drawSampleLattice already applies, or its red
      // boundary lines end up drawn in the opposite vertical order from
      // the image and sample dots beside them -- confirmed live: the
      // column (U) strip needed no such flip and lined up correctly, only
      // this row (V) strip was inverted.
      const py = (1 - i / n) * marginalRightCanvas.height;
      const px = maxMag > 0 ? (m.rowMag[i] / maxMag) * (MARGINAL_THICKNESS - 4) : 0;
      if (i > 0) {
        rc.strokeStyle = marginalHueColor(m.rowHueCx[i], m.rowSumCy[i]);
        rc.beginPath(); rc.moveTo(prevPx, prevPy); rc.lineTo(px, py); rc.stroke();
      }
      prevPx = px; prevPy = py;
    }
    if (m.rowPeriod) {
      rc.strokeStyle = 'rgba(255,80,80,0.6)';
      for (let py = m.rowPhase; py < n; py += m.rowPeriod) {
        const yy = (1 - py / n) * marginalRightCanvas.height;
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
    const n = m.colMag.length;
    // Magnitude, always >= 0 -- see the matching comment in the
    // marginalRight block above. Deflection grows DOWN, away from the image
    // (which sits above this strip), matching the right strip's "grows away
    // from the image" (rightward) convention: py=0 (top of strip, adjacent
    // to the image) at zero magnitude, py=THICKNESS (bottom, outer edge) at
    // max.
    let maxMag = 0;
    for (let i = 0; i < n; i++) if (m.colMag[i] > maxMag) maxMag = m.colMag[i];
    // Per-segment color -- see the matching comment in the marginalRight
    // block above; each bin's own (colSum,colSumCy) vector sum picks the hue.
    bc.lineWidth = 1;
    let prevPx = 0, prevPy = 0;
    for (let i = 0; i < n; i++) {
      const px = (i / n) * marginalBottomCanvas.width;
      const py = maxMag > 0 ? (m.colMag[i] / maxMag) * (MARGINAL_THICKNESS - 4) : 0;
      if (i > 0) {
        bc.strokeStyle = marginalHueColor(m.colSum[i], m.colSumCy[i]);
        bc.beginPath(); bc.moveTo(prevPx, prevPy); bc.lineTo(px, py); bc.stroke();
      }
      prevPx = px; prevPy = py;
    }
    if (m.colPeriod) {
      bc.strokeStyle = 'rgba(255,80,80,0.6)';
      for (let px = m.colPhase; px < n; px += m.colPeriod) {
        const xx = (px / n) * marginalBottomCanvas.width;
        bc.beginPath(); bc.moveTo(xx, 0); bc.lineTo(xx, MARGINAL_THICKNESS); bc.stroke();
      }
    }
  }

  updatePositionReadoutText();
}

// Builds positionReadout's full text from lastMarginals + lastPositionDecode.
// Called both every frame from drawMarginalLines (so it stays live while
// tuning in 'projected' mode) AND once at the end of runAxesReconstruction
// (so a capture triggered from a DIFFERENT mode still updates it -- this
// function is the only place that writes positionReadout.textContent;
// two separate assignments used to fight over it, silently clobbering each
// other every frame, which is why the "world-axis swap" diagnostic line
// never actually appeared on screen despite being computed correctly).
function updatePositionReadoutText() {
  if (!positionReadout) return;
  if (!lastMarginals) { positionReadout.textContent = 'not yet computed (switch to Projected Cam or capture now)'; return; }
  const m = lastMarginals;
  const uStep = m.colPeriod && lastProjectedBins ? m.colPeriod * lastProjectedBins.binWidthU : null;
  const vStep = m.rowPeriod && lastProjectedBins ? m.rowPeriod * lastProjectedBins.binWidthV : null;
  const periodicityLines =
    `col period: ${m.colPeriod ?? '—'} bins (phase ${m.colPhase.toFixed(1)})\n` +
    `row period: ${m.rowPeriod ?? '—'} bins (phase ${m.rowPhase.toFixed(1)})\n` +
    `implied grid step: U=${uStep?.toFixed(3) ?? '—'}  V=${vStep?.toFixed(3) ?? '—'}\n` +
    `(expect both ≈ ${GRID_STEP})`;
  let decodeLines: string;
  if (lastPositionDecode) {
    const rec = lastPositionDecode.camPos;
    const errPos = rec.distanceTo(camPos);
    decodeLines =
      `torus cell: row ${lastPositionDecode.row}  col ${lastPositionDecode.col}\n` +
      `consistency: ${(lastPositionDecode.consistency * 100).toFixed(1)}%\n` +
      `recovered camPos: (${rec.x.toFixed(2)}, ${rec.y.toFixed(2)}, ${rec.z.toFixed(2)})\n` +
      `true camPos: (${camPos.x.toFixed(2)}, ${camPos.y.toFixed(2)}, ${camPos.z.toFixed(2)})\n` +
      `error: ${errPos.toFixed(3)} world units\n` +
      `world-axis swap: ${lastPositionDecode.worldAxisSwapped ? 'YES' : 'no'} ` +
      `(unswapped err ${lastPositionDecode.camPosErrUnswapped.toFixed(2)}, swapped err ${lastPositionDecode.camPosErrSwapped.toFixed(2)} -- ground-truth diagnostic, lab-only)`;
  } else {
    decodeLines = 'position decode: no match (need periodicity + a successful orientation/distance fit)';
  }
  positionReadout.textContent = `${periodicityLines}\n\n${decodeLines}`;
}

function hideMarginalLines() {
  marginalRightCanvas.style.display = 'none';
  marginalBottomCanvas.style.display = 'none';
}

function hideSampleLattice() {
  sampleLatticeCanvas.style.display = 'none';
}

// One small filled circle per runPositionDecode sample point (see
// buildDecodeSampleGrid), positioned at wherever that point's world (u,v)
// lands in the CURRENT "Projected Cam" rectification -- a linear map of a
// uniform integer grid, so they land in a uniform rectangular lattice, one
// dot per sampled cell. Each circle's fill is read straight from
// distortedPreviewData at that same point's source pixel, so it shows
// exactly what color/brightness whichever field view is currently selected
// actually has at the spot each decode sample reads from.
function drawSampleLattice(x: number, y: number, w: number, h: number) {
  if (!state.showSampleLattice) { hideSampleLattice(); return; }
  // Reuses the grid runPositionDecode most recently built (needs `gray`,
  // which isn't available in this render-only path) rather than rebuilding
  // it here.
  const grid = lastDecodeGrid;
  if (!grid || !lastProjectedBins) { hideSampleLattice(); return; }
  const { maxU, binWidthU, minV, binWidthV, w: bw, h: bh } = lastProjectedBins;

  sampleLatticeCanvas.style.display = 'block';
  sampleLatticeCanvas.style.left = x + 'px';
  sampleLatticeCanvas.style.top = y + 'px';
  sampleLatticeCanvas.width = Math.round(w);
  sampleLatticeCanvas.height = Math.round(h);
  sampleLatticeCanvas.style.width = w + 'px';
  sampleLatticeCanvas.style.height = h + 'px';
  const ctx = sampleLatticeCtx;
  ctx.clearRect(0, 0, sampleLatticeCanvas.width, sampleLatticeCanvas.height);

  const radius = 3;
  for (let i = 0; i < grid.points.length; i++) {
    const row = grid.points[i];
    for (let j = 0; j < row.length; j++) {
      const pt = row[j];
      if (!pt.valid) continue;
      // bu runs maxU -> minU, matching buildProjectedTexture's mirror -- see
      // its comment.
      const bu = (maxU - pt.u) / binWidthU;
      const bv = (pt.v - minV) / binWidthV;
      if (bu < 0 || bu >= bw || bv < 0 || bv >= bh) continue;
      // X maps directly (bu=0 -> left, matching distortedPreviewTex/
      // projectedPreviewTex's UV convention), but Y is GL-native bottom-up
      // (bv=0 -> bottom of the rendered image, see buildProjectedTexture's
      // own comment on this exact convention) while this canvas is CSS-
      // style top-down -- flip.
      const cx = (bu / bw) * sampleLatticeCanvas.width;
      const cy = (1 - bv / bh) * sampleLatticeCanvas.height;
      // Fill = the actual BINARIZED bit this cell sampled (black/white,
      // same convention as the real tracker's patch dots, src/main.ts:
      // bit=1 is dark), not whichever field view happens to be on screen --
      // this is what the decode itself sees. Stroke = decode correctness,
      // same idea as that tracker's per-cell patch highlighting -- green if
      // this cell's bit matches the real torus content at the position the
      // winning decode implies, red if not, dim gray if there's no decode
      // (or this specific cell) to check yet.
      const debug = lastDecodeCorrectness ? lastDecodeCorrectness[i][j] : null;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = debug ? (debug.bit ? '#000' : '#fff') : '#888';
      ctx.fill();
      ctx.strokeStyle = debug ? (debug.correct ? '#0f0' : '#f00') : 'rgba(0,0,0,0.6)';
      ctx.lineWidth = debug ? 2 : 1;
      ctx.stroke();
    }
  }
}

// Local reimplementation of decode.ts's packPatchCells (not exported there,
// and small enough not to be worth exporting for one caller) -- packs an
// order x order window into a lookup key under one of 4 reading
// orientations, matching src/debruijn.ts's windowKey order (top-to-bottom,
// left-to-right, MSB-first) for orientation 0; the others apply the same
// index rotation packPatchCells itself uses.
function packPatchCellsLocal(cells: number[][], order: number, orientation: number): number {
  let key = 0;
  for (let i = 0; i < order; i++) {
    for (let j = 0; j < order; j++) {
      let bit: number;
      if (orientation === 1) bit = cells[order - 1 - j][i];
      else if (orientation === 2) bit = cells[order - 1 - i][order - 1 - j];
      else if (orientation === 3) bit = cells[j][order - 1 - i];
      else bit = cells[i][j];
      key = (key << 1) | bit;
    }
  }
  return key >>> 0;
}

// u,v are the sample's world position (relative to camera, in Drow/Dcol
// units); px,py are where that point projects to in the CURRENT capture's
// pixel space, in the same TOP-DOWN row convention captureDistortedGrayscale's
// flipped output uses (row 0 = top) -- NOT distortedPreviewData's own
// GL-native bottom-up convention (see flipRowsF64 elsewhere in this file).
// valid is false when the point is behind the camera or projects outside
// the image entirely (px/py meaningless, possibly NaN, in that case) --
// bit is meaningless too when !valid.
interface DecodeSamplePoint { u: number; v: number; px: number; py: number; valid: boolean; bit: number }
interface DecodeSampleGrid { rows: number; cols: number; zeroI: number; zeroJ: number; points: DecodeSamplePoint[][] }

// Builds a sampling grid covering the FULL observed quadrilateral (per
// lastProjectedBins' minU/maxU/minV/maxV -- the same extent "Projected Cam"
// itself renders), not a small fixed window around one anchor. Every
// integer GRID_STEP hop from the phase-anchored sub-cell offset, in both
// directions, until the whole observed extent is covered -- "behind
// camera" and "off-frame" points still get marked invalid (skipped below),
// but those are genuinely outside the visible quadrilateral by
// definition (see this function's header comment on what "behind camera"
// means), not an arbitrary window boundary excluding real, visible floor.
// zeroI/zeroJ mark whichever sampled cell sits closest to the camera
// itself (world u=0,v=0) -- used as the position-recovery reference point,
// since cells far from the camera accumulate more drift error (confirmed
// empirically: correctness degrades with distance from center) than cells
// right next to it.
function buildDecodeSampleGrid(gray: Float64Array, w: number, h: number, vFovRad: number): DecodeSampleGrid | null {
  if (!lastRecoveredAxes || !lastMarginals || lastMarginals.colPeriod === null || lastMarginals.rowPeriod === null || !lastProjectedBins) {
    return null;
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
  // colPhase is a bu bin-index, and bu runs maxU -> minU (see
  // buildProjectedTexture's comment on why) -- so bu=0 is u=maxU, not minU.
  const uBoundaryRaw = lastProjectedBins.maxU - lastMarginals.colPhase * lastProjectedBins.binWidthU;
  const vBoundaryRaw = lastProjectedBins.minV + lastMarginals.rowPhase * lastProjectedBins.binWidthV;
  const uPhase = (uBoundaryRaw - Math.round(uBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;
  const vPhase = (vBoundaryRaw - Math.round(vBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;

  const { minU, maxU, minV, maxV } = lastProjectedBins;
  const kMinU = Math.floor((minU - uPhase) / GRID_STEP), kMaxU = Math.ceil((maxU - uPhase) / GRID_STEP);
  const kMinV = Math.floor((minV - vPhase) / GRID_STEP), kMaxV = Math.ceil((maxV - vPhase) / GRID_STEP);
  const cols = kMaxU - kMinU + 1, rows = kMaxV - kMinV + 1;
  const zeroI = Math.min(rows - 1, Math.max(0, Math.round(-vPhase / GRID_STEP) - kMinV));
  const zeroJ = Math.min(cols - 1, Math.max(0, Math.round(-uPhase / GRID_STEP) - kMinU));

  // No separate "is this behind the camera" check -- every (u,v) here is
  // already constrained to lastProjectedBins' own min/max, which
  // buildProjectedTexture only ever populates from rays it confirmed are
  // in front of the camera AND past its grazing-angle cutoff (see that
  // function's MIN_GRAZING_COS). A point that's actually behind the camera
  // or too grazing simply won't project to a real, in-frame pixel below --
  // that's the only check that actually matters for "can this cell be
  // sampled at all", and we need to compute the pixel position anyway to
  // know where to read the bit from. Number.isFinite guards the one
  // remaining numerical edge case (local.z landing exactly on 0, an
  // infinite/NaN projection) from slipping past the plain range check --
  // NaN fails `< 0` and `>= w` alike, so without this it would read as
  // in-bounds.
  const p = new THREE.Vector3();
  const local = new THREE.Vector3();
  const points: DecodeSamplePoint[][] = [];
  for (let i = 0; i < rows; i++) {
    const v = vPhase + (kMinV + i) * GRID_STEP;
    const rowPoints: DecodeSamplePoint[] = [];
    for (let j = 0; j < cols; j++) {
      const u = uPhase + (kMinU + j) * GRID_STEP;
      // Relative-to-camera world point at this cell's exact center (same
      // "hit" construction as buildProjectedTexture:
      // p.dot(normal) == -distance for every floor point), then rotated into
      // camera-local space (inverse of cornerDir's forward rotation) and run
      // through cornerDir's pinhole formula backwards to find the pixel it
      // projects to.
      p.copy(Drow).multiplyScalar(u).addScaledVector(Dcol, v).addScaledVector(normal, -distance);
      local.copy(p).applyQuaternion(invQuat);
      const ndcU = -local.x / (local.z * Math.tan(halfV) * RT_ASPECT);
      const ndcV = -local.y / (local.z * Math.tan(halfV));
      const px = ((ndcU + 1) / 2) * w, py = ((1 - ndcV) / 2) * h;
      const valid = Number.isFinite(px) && Number.isFinite(py) && px >= 0 && px < w && py >= 0 && py < h;
      if (!valid) { rowPoints.push({ u, v, px, py, valid: false, bit: 0 }); continue; }
      const xx = Math.round(px), yy = Math.round(py);
      rowPoints.push({ u, v, px, py, valid: true, bit: bin[yy * w + xx] });
    }
    points.push(rowPoints);
  }
  return { rows, cols, zeroI, zeroJ, points };
}

interface PositionDecodeResult {
  row: number; col: number; consistency: number; votes: number; totalWindows: number;
  camPos: THREE.Vector3;
  // Diagnostic only (see runPositionDecode's comment on worldAxisSwapped):
  // ground-truth distance each of the two world-axis candidates lands at.
  // Not something a real (non-lab) decode could compute.
  camPosErrUnswapped: number; camPosErrSwapped: number; worldAxisSwapped: boolean;
}
let lastPositionDecode: PositionDecodeResult | null = null;

// Per DecodeSampleGrid cell (ORIGINAL, non-mirrored indexing) -- bit is what
// was actually sampled there, correct is whether it matches the real torus
// content at the position the winning vote implies. null where the sample
// itself was invalid or no decode exists yet. Rebuilt every
// runPositionDecode call; consumed by drawSampleLattice to fill/stroke each
// dot, same idea as the real tracker's per-cell patch highlighting
// (src/main.ts, Patch.correct) -- see that file's togglePatches block.
interface DecodeCellDebug { bit: number; correct: boolean }
let lastDecodeCorrectness: (DecodeCellDebug | null)[][] | null = null;
// The grid runPositionDecode most recently built, cached so drawSampleLattice
// can reuse its (u,v,px,py) geometry without rebuilding it (which needs
// `gray`, not available in the render path) or duplicating this math.
let lastDecodeGrid: DecodeSampleGrid | null = null;

// Every valid order x order window in the grid (NOT just non-overlapping
// tiles -- a sliding window, one step at a time, so a window that just
// misses a clean read by starting one cell over still gets its own
// independent chance), at each of the 4 reading rotations, either finds an
// EXACT match in the de Bruijn lookup table or doesn't (no partial credit --
// a single wrong bit anywhere in the 16 sends the packed key to some
// unrelated, essentially random torus position, or occasionally to nothing
// at all). Every window that DOES find a match casts one vote for the torus
// anchor (grid cell (0,0)'s implied position) that match implies. Unlike
// pickBestCandidate's own correlation-based scoring (checks only the single
// best-looking anchor against the whole grid), this tallies literal vote
// counts across every anchor any window proposed, and the anchor with the
// most votes wins -- unambiguous "most patches agree" rather than "best of
// whichever candidates got an exact hit at all."
interface VoteResult { mirrored: boolean; orientation: number; anchorRow: number; anchorCol: number; votes: number; totalWindows: number }
function tallyPositionVotes(grid: DecodeSampleGrid): VoteResult | null {
  const tally = new Map<string, number>();
  let totalWindows = 0;
  const block: number[][] = Array.from({ length: ORDER }, () => new Array(ORDER).fill(0));
  for (const mirrored of [false, true]) {
    const at = (i: number, j: number) => mirrored ? grid.points[grid.rows - 1 - i][j] : grid.points[i][j];
    for (let i0 = 0; i0 + ORDER <= grid.rows; i0++) {
      for (let j0 = 0; j0 + ORDER <= grid.cols; j0++) {
        let complete = true;
        for (let di = 0; di < ORDER && complete; di++) {
          for (let dj = 0; dj < ORDER; dj++) {
            const pt = at(i0 + di, j0 + dj);
            if (!pt.valid) { complete = false; break; }
            block[di][dj] = pt.bit;
          }
        }
        if (!complete) continue;
        totalWindows++;
        for (let o = 0; o < 4; o++) {
          const key = packPatchCellsLocal(block, ORDER, o);
          const packed = debruijnLookup[key];
          if (packed === -1) continue;
          const matchRow = Math.floor(packed / C), matchCol = packed % C;
          const [dr, dc] = rotateShift(i0, j0, o);
          const anchorRow = ((matchRow - dr) % R + R) % R;
          const anchorCol = ((matchCol - dc) % C + C) % C;
          const voteKey = `${mirrored ? 1 : 0},${o},${anchorRow},${anchorCol}`;
          tally.set(voteKey, (tally.get(voteKey) ?? 0) + 1);
        }
      }
    }
  }
  let best: VoteResult | null = null;
  for (const [key, votes] of tally) {
    if (best && votes <= best.votes) continue;
    const [m, o, ar, ac] = key.split(',').map(Number);
    best = { mirrored: m === 1, orientation: o, anchorRow: ar, anchorCol: ac, votes, totalWindows };
  }
  return best;
}

// Decodes the camera's absolute world position: samples every valid floor
// cell across the ENTIRE observed quadrilateral (via inverse projection
// straight from the analysis grayscale, NOT the lossy forward-binned
// projectedPreviewData), decodes which torus (row,col) it sits at via
// tallyPositionVotes' literal per-window vote count, then combines the
// decoded absolute cell position with its known position RELATIVE to the
// camera to recover the camera's own absolute position -- the only ground
// truth this touches is which torus (row,col) a bit pattern implies (a
// property of the pattern itself, verifiable from the image alone), never
// camPos.
function runPositionDecode(gray: Float64Array, w: number, h: number, vFovRad: number) {
  const grid = buildDecodeSampleGrid(gray, w, h, vFovRad);
  lastDecodeGrid = grid;
  if (!grid) { lastPositionDecode = null; lastDecodeCorrectness = null; return; }
  const winner = tallyPositionVotes(grid);
  if (!winner) { lastPositionDecode = null; lastDecodeCorrectness = null; return; }

  // Per-cell correctness across the FULL grid (not just the cells that
  // happened to be part of a voting window) -- same math scoreCorrelation
  // uses internally: given the winning anchor + orientation, every cell's
  // implied torus position is anchor + rotateShift(i,j,o).
  const { mirrored, orientation: o, anchorRow, anchorCol } = winner;
  const correctness: (DecodeCellDebug | null)[][] = Array.from({ length: grid.rows }, () => new Array(grid.cols).fill(null));
  let correctCount = 0, wrongCount = 0;
  for (let i = 0; i < grid.rows; i++) {
    for (let j = 0; j < grid.cols; j++) {
      // mirrorRowsGrid-equivalent: mirrored candidate's cell (i,j) is
      // ORIGINAL cell (rows-1-i, j).
      const pt = mirrored ? grid.points[grid.rows - 1 - i][j] : grid.points[i][j];
      if (!pt.valid) continue;
      const [dr, dc] = rotateShift(i, j, o);
      const torusRow = ((anchorRow + dr) % R + R) % R;
      const torusCol = ((anchorCol + dc) % C + C) % C;
      const correct = pt.bit === torus[torusRow][torusCol];
      correctness[i][j] = { bit: pt.bit, correct };
      correct ? correctCount++ : wrongCount++;
    }
  }
  lastDecodeCorrectness = correctness;
  const consistency = correctCount + wrongCount > 0 ? correctCount / (correctCount + wrongCount) : 0;

  // worldX = c + 0.5 - C/2, worldZ = r + 0.5 - R/2 (GRID_STEP=1) -- the floor
  // mesh's own world<->torus CELL CENTER convention, reverse-engineered from
  // its PlaneGeometry + rotation.x=-pi/2 + CanvasTexture setup (see the floor
  // mesh construction above) and confirmed empirically via raycast against
  // the real floorMesh (the +0.5-less formula lands exactly on a cell
  // CORNER, uv fraction (0,0); the +0.5 one lands exactly on the cell
  // CENTER, uv fraction (0.5,0.5)). Missing that +0.5 here previously left
  // every recovered camPos off by almost exactly half a GRID_STEP in both
  // X and Z. Uses the cell nearest the camera itself
  // (grid.zeroI/zeroJ) as the reference point, not an arbitrary corner --
  // minimizes the drift error that grows with distance from the camera
  // (confirmed empirically via lastDecodeCorrectness's own distance-vs-
  // correctness breakdown).
  const { Drow, Dcol, Dnormal, distance } = lastRecoveredAxes!; // buildDecodeSampleGrid returning non-null guarantees this
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, camQuat, vFovRad, RT_ASPECT).dot(normal) > 0) normal.negate();
  const refI = mirrored ? grid.rows - 1 - grid.zeroI : grid.zeroI;
  const [drRef, dcRef] = rotateShift(refI, grid.zeroJ, o);
  const refTorusRow = ((anchorRow + drRef) % R + R) % R;
  const refTorusCol = ((anchorCol + dcRef) % C + C) % C;
  const refPt = grid.points[grid.zeroI][grid.zeroJ];
  const hitRel = new THREE.Vector3().addScaledVector(Drow, refPt.u).addScaledVector(Dcol, refPt.v).addScaledVector(normal, -distance);

  // tallyPositionVotes' D4 search (row-mirror x 4 orientations) already
  // resolves any ambiguity in how OUR sampling grid's own (i,j) axes relate
  // to the TORUS's native (row,col) -- refTorusRow/refTorusCol ARE genuinely
  // "torus row" and "torus col" for the reference cell. What it does NOT
  // resolve is whether the FLOOR MESH's own construction actually assigns
  // world X to torus COLUMN and world Z to torus ROW, or the other way
  // around -- that's a fixed fact about the mesh (see the comment above),
  // not a per-frame ambiguity, and vote count can't tell us which: it's
  // computed entirely within the decode's own row/col index space and never
  // touches world X/Z, so it comes out IDENTICAL either way. The only way
  // to tell, in this lab (NOT available to a real, non-cheating tracker),
  // is ground truth -- pick whichever candidate lands closer to the real
  // camPos.
  const worldPosTrueUnswapped = new THREE.Vector3((refTorusCol + 0.5 - C / 2) * GRID_STEP, 0, (refTorusRow + 0.5 - R / 2) * GRID_STEP);
  const worldPosTrueSwapped = new THREE.Vector3((refTorusRow + 0.5 - R / 2) * GRID_STEP, 0, (refTorusCol + 0.5 - C / 2) * GRID_STEP);
  const camPosUnswapped = worldPosTrueUnswapped.sub(hitRel); // .sub() only mutates the receiver, hitRel is safe to reuse below
  const camPosSwapped = worldPosTrueSwapped.sub(hitRel);
  const errUnswapped = camPosUnswapped.distanceTo(camPos);
  const errSwapped = camPosSwapped.distanceTo(camPos);
  const swapped = errSwapped < errUnswapped;
  lastPositionDecode = {
    row: refTorusRow, col: refTorusCol, consistency, votes: winner.votes, totalWindows: winner.totalWindows,
    camPos: swapped ? camPosSwapped : camPosUnswapped,
    camPosErrUnswapped: errUnswapped, camPosErrSwapped: errSwapped, worldAxisSwapped: swapped,
  };
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
      // buildProjectedTexture (called below) now needs lastNoisedPreviewGray
      // for its own per-source-pixel gradient (see its own comment) -- but
      // updateDistortedPreview only ever SETS that when the current field
      // view is 'noised'/'gradient'/'walked'/'agreement'/'effective' or a
      // contamination overlay is on (an early-return skips it otherwise, to
      // avoid the antialiasing/blur/downsample/noise cost when displaying
      // something cheaper like 'raw'). That's fine for the passive per-frame
      // preview it was written for, but buildProjectedTexture needs it
      // EVERY capture regardless of what's currently selected for display --
      // confirmed live: leaving this out silently left lastNoisedPreviewGray
      // null whenever fieldView happened to be 'raw', collapsing distance/
      // period recovery entirely (autocorrelation on an all-zero signal).
      // rawGray is already bottom-up (matching distortedPreviewData's own
      // convention, which is what lastNoisedPreviewGray needs to align with)
      // and freshly captured through the exact same distortion pipeline, so
      // just use it directly rather than depending on the preview path.
      lastNoisedPreviewGray = rawGray;
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

      // Refines fitPairOfPlanes' closed-form (algebraic-residual) answer
      // against the true geometric alignment residual -- see
      // refineOrientationLM's own header comment. Toggle-able (state.
      // orientationLM) specifically so this can be A/B compared live
      // against the unrefined fit, the same way every other change this
      // session has been verified rather than assumed.
      const refinedFit = quadricPair && state.orientationLM ? refineOrientationLM(fitVotes, quadricPair) : null;
      const orientationFit = refinedFit ?? quadricPair;
      const t2b = performance.now();

      axesComputed = !!quadricPair;

      // fitPairOfPlanes can't tell from the math alone which of its two
      // recovered orthogonal directions is "row" vs "col" (a genuine
      // ambiguity, not a bug in the fit itself -- see its header comment).
      // Resolved here, ONCE, against ground truth (fine for this testbed,
      // which has it available) -- and reused everywhere downstream
      // (pole markers, distance recovery, lastRecoveredAxes, the readout) so
      // they all agree on the same labeling. Previously this correction only
      // affected the readout's displayed text, while lastRecoveredAxes AND
      // the red/blue pole markers below kept whatever raw (possibly swapped)
      // labels the fit produced -- "Projected Cam" showed a fully swapped
      // u/v projection despite the readout looking correct, and the pole
      // markers could show up on the wrong axis even once that was fixed,
      // since they were still being set from quadricPair.Drow/Dcol directly
      // further up rather than from this corrected pair.
      //
      // Picks whichever labeling minimizes TOTAL error against both ground-
      // truth axes (Drow-vs-ROW_DIR + Dcol-vs-COL_DIR, or the swapped pairing)
      // rather than just comparing both candidates to ROW_DIR alone. The two
      // are mathematically equivalent here -- Drow/Dcol come out of a
      // symmetric eigendecomposition so they're already exactly orthogonal,
      // same as ROW_DIR/COL_DIR by construction, so agreement on one axis
      // guarantees agreement on the other -- but this version states the
      // actual intent directly instead of leaning on that orthogonality fact.
      //
      // angleBetweenDegV takes Math.abs() of the dot product, so it (by
      // design) can't tell a candidate from its own negation -- it only
      // resolves row-vs-col, not which way each one points. Drow/Dcol's
      // individual signs are otherwise arbitrary (whatever the eigensolver's
      // b1/b2 happened to come out as), unlike Dnormal, whose sign IS fixed
      // elsewhere in this file (against the non-cheating fact that the floor
      // must face the camera). Left uncorrected, this flipped u and/or v's
      // sign in buildProjectedTexture's u = hit.dot(Drow)/v = hit.dot(Dcol)
      // (the "Projected Cam" view coming out mirrored relative to the true
      // top-down pattern), and fed straight into runPositionDecode's hitRel,
      // injecting a large systematic camPos error along whichever axis was
      // flipped -- while per-cell bit correctness stayed unaffected, since
      // tallyPositionVotes' own D4 (mirror x 4 rotations) search already
      // absorbs any axis-sign ambiguity when matching bits to the known
      // pattern. Ground truth again (testbed-only, same as the swap above):
      // flip each axis independently if it landed antiparallel to its true
      // world direction -- negating one axis alone can't break their
      // orthogonality (dot(-a,b) = -dot(a,b), still 0 if it started at 0).
      let rowDirRecovered: THREE.Vector3 | null = null, colDirRecovered: THREE.Vector3 | null = null;
      if (orientationFit) {
        const errUnswapped = angleBetweenDegV(orientationFit.Drow, ROW_DIR) + angleBetweenDegV(orientationFit.Dcol, COL_DIR);
        const errSwapped = angleBetweenDegV(orientationFit.Drow, COL_DIR) + angleBetweenDegV(orientationFit.Dcol, ROW_DIR);
        const flipped = errSwapped < errUnswapped;
        rowDirRecovered = flipped ? orientationFit.Dcol : orientationFit.Drow;
        colDirRecovered = flipped ? orientationFit.Drow : orientationFit.Dcol;
        if (rowDirRecovered.dot(ROW_DIR) < 0) rowDirRecovered.negate();
        if (colDirRecovered.dot(COL_DIR) < 0) colDirRecovered.negate();

        recoveredRowPoleA.position.copy(rowDirRecovered).multiplyScalar(SPHERE_RADIUS);
        recoveredRowPoleB.position.copy(rowDirRecovered).multiplyScalar(-SPHERE_RADIUS);
        recoveredColPoleA.position.copy(colDirRecovered).multiplyScalar(SPHERE_RADIUS);
        recoveredColPoleB.position.copy(colDirRecovered).multiplyScalar(-SPHERE_RADIUS);
      }

      // Distance-to-floor recovery needs the axes above, so it only runs on
      // a successful orientation fit. Builds the projected texture once at
      // an arbitrary PLACEHOLDER distance purely to get computeProjectedMarginals'
      // period measurement -- the ray-cast/bin formula scales linearly with
      // whatever distance is assumed, and bin ASSIGNMENT is completely
      // invariant to that scale (see the big comment above, right before
      // this function, for the full derivation), so any placeholder works
      // and no separate ray-cast implementation is needed just for distance.
      const PLACEHOLDER_DISTANCE = 1;
      lastRecoveredAxes = rowDirRecovered && colDirRecovered && orientationFit
        ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: orientationFit.Dnormal, distance: PLACEHOLDER_DISTANCE }
        : null;
      if (lastRecoveredAxes) buildProjectedTexture();

      // Closed-form correction: the true period must equal the known
      // GRID_STEP, so the ratio between that and what was actually measured
      // (at the placeholder distance) gives the exact rescale factor back to
      // the real distance -- same trick the old estimateFloorDistance used,
      // just reusing this already-tuned marginal-line measurement instead of
      // a second, separate gradient-magnitude implementation.
      const marginals = lastMarginals, bins = lastProjectedBins;
      const spacing = lastRecoveredAxes && marginals && bins && marginals.colPeriod !== null && marginals.rowPeriod !== null
        ? {
          distanceU: PLACEHOLDER_DISTANCE * (GRID_STEP / (marginals.colPeriod * bins.binWidthU)),
          distanceV: PLACEHOLDER_DISTANCE * (GRID_STEP / (marginals.rowPeriod * bins.binWidthV)),
        }
        : null;
      const t3 = performance.now();

      let refinedSpacing: { distanceU: number; distanceV: number } | null = null;
      let finalSpacing: { distanceU: number; distanceV: number } | null = null;
      if (lastRecoveredAxes && spacing) {
        // Feeds "Projected Cam" mode (see buildProjectedTexture) -- averaging
        // the U/V distance estimates is just noise reduction, not picking one
        // over the other (both should agree once past the grazing-angle cutoff).
        lastRecoveredAxes.distance = (spacing.distanceU + spacing.distanceV) / 2;

        // Sub-bin refinement pass -- see measurePeriodDistance's own comment
        // for why this exists. roughBins' extent is still in
        // PLACEHOLDER_DISTANCE-scale units (u,v scale linearly with whatever
        // distance was assumed for the ray-cast that produced it) -- rescale
        // by the correction factor just applied to get the TRUE extent,
        // without wasting a whole extra ray-cast pass just to remeasure it.
        const roughBins = lastProjectedBins;
        if (roughBins) {
          const rescale = lastRecoveredAxes.distance / PLACEHOLDER_DISTANCE;
          const trueExtentU = (roughBins.maxU - roughBins.minU) * rescale;
          const trueExtentV = (roughBins.maxV - roughBins.minV) * rescale;
          const roughDistance = lastRecoveredAxes.distance;
          const measured = measurePeriodDistance(roughDistance, trueExtentU, trueExtentV);
          if (measured) {
            refinedSpacing = measured;
            lastRecoveredAxes.distance = (measured.distanceU + measured.distanceV) / 2;
          }
        }

        // Rebuilds projectedPreviewData/lastProjectedBins/lastMarginals now
        // that distance is correct -- position decode below (and "Projected
        // Cam" itself) need the properly-scaled versions, not the
        // placeholder-distance ones from the first pass above.
        buildProjectedTexture();
      } else {
        lastRecoveredAxes = null;
      }
      const t3b = performance.now();
      runPositionDecode(gray, w, h, vFovRad);

      // Phase 3 (Option B): refines the decode's own coarse camPos AND the
      // already-refined orientation jointly, against the actual known bit
      // content instead of a pattern-agnostic distance-to-boundary signal
      // -- see refineOrientationAndPositionLM's header comment for why this
      // exists (Option A's wrapped residual couldn't tell the true grid
      // line from a neighboring wrong one when samples were sparse on one
      // axis; this can, since a wrong cell alignment predicts the wrong
      // BIT, not just "some distance off"). Only runs if decode produced
      // something to refine FROM -- same "cheap coarse init + iterative
      // geometric refine" relationship as every other LM stage here, not a
      // replacement for the decode step. Toggle-able for direct A/B
      // comparison, same reasoning as orientationLM.
      let lastPositionLMResult: (PositionFit & { iterations: number; initialCost: number; finalCost: number }) | null = null;
      if (state.positionLM && lastRecoveredAxes && lastPositionDecode && lastNoisedPreviewGray) {
        const { Drow, Dcol, Dnormal, distance } = lastRecoveredAxes;
        const normalForInit = Dnormal.clone();
        if (cornerDir(0, 0, camQuat, vFovRad, RT_ASPECT).dot(normalForInit) > 0) normalForInit.negate();
        // worldPos(u=0,v=0) = camPos + normal*(-distance) EXACTLY, not just
        // approximately camPos.x/z -- Drow/Dcol/normal is a complete
        // orthonormal basis and every hit point satisfies hit.normal=
        // -distance by construction (see refineOrientationAndPositionLM's
        // own comment), so this is the correct non-cheating initializer.
        const initialWorldX0 = lastPositionDecode.camPos.x + normalForInit.x * -distance;
        const initialWorldZ0 = lastPositionDecode.camPos.z + normalForInit.z * -distance;
        const photoSamples = computePhotometricSamples(lastNoisedPreviewGray, w, h, 4);
        lastPositionLMResult = refineOrientationAndPositionLM(
          photoSamples, w, h, { Drow, Dcol, Dnormal }, distance, initialWorldX0, initialWorldZ0, camQuat, vFovRad, RT_ASPECT,
        );
        lastRecoveredAxes.Drow = lastPositionLMResult.Drow;
        lastRecoveredAxes.Dcol = lastPositionLMResult.Dcol;
        lastRecoveredAxes.Dnormal = lastPositionLMResult.Dnormal;
        const refinedNormal = lastPositionLMResult.Dnormal.clone();
        if (cornerDir(0, 0, camQuat, vFovRad, RT_ASPECT).dot(refinedNormal) > 0) refinedNormal.negate();
        lastPositionDecode.camPos.x = lastPositionLMResult.worldX0 + refinedNormal.x * distance;
        lastPositionDecode.camPos.z = lastPositionLMResult.worldZ0 + refinedNormal.z * distance;
        // Rebuilds with the now Option-B-refined orientation, so "Projected
        // Cam" and the marginal graphs reflect the final answer, not the
        // pre-refinement one.
        buildProjectedTexture();

        // Second distance refinement pass, now that orientation is Phase
        // 3's much better estimate (~0.4 degree here vs Phase 1's ~2
        // degree) instead of the cruder one the first pass (above,
        // pre-Phase-3) had to use -- see measurePeriodDistance's own
        // comment for why a rotated (Drow,Dcol) basis leaks into distance
        // error asymmetrically between U and V. lastProjectedBins is
        // already at the correct TRUE-world-unit scale here (no
        // placeholder-rescale needed, unlike the first pass), since
        // buildProjectedTexture just rebuilt it at the current distance.
        const postPhase3Bins = lastProjectedBins;
        if (postPhase3Bins) {
          const extentU = postPhase3Bins.maxU - postPhase3Bins.minU;
          const extentV = postPhase3Bins.maxV - postPhase3Bins.minV;
          const currentDistance = lastRecoveredAxes.distance;
          const measured = measurePeriodDistance(currentDistance, extentU, extentV);
          if (measured) {
            finalSpacing = measured;
            lastRecoveredAxes.distance = (measured.distanceU + measured.distanceV) / 2;
            // lastPositionDecode.camPos was set just above from worldX0/
            // worldZ0 + normal*distance using the OLD (pre-this-pass)
            // distance -- worldX0/worldZ0 themselves don't change when
            // distance is corrected, but camPos does, so it has to be
            // recomputed with the same formula or decode ends up reading a
            // camPos inconsistent with the axes/distance buildProjectedTexture
            // now uses (confirmed live: skipping this recompute dropped
            // consistency from a clean 1.0 to chance-level ~0.5 at poses
            // that had nothing wrong with them before this pass existed).
            lastPositionDecode.camPos.x = lastPositionLMResult.worldX0 + refinedNormal.x * lastRecoveredAxes.distance;
            lastPositionDecode.camPos.z = lastPositionLMResult.worldZ0 + refinedNormal.z * lastRecoveredAxes.distance;
            buildProjectedTexture();
          }
        }
      }
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
        lines.push(`row err ${rowErr.toFixed(2)}°  col err ${colErr.toFixed(2)}°  [Phase 1, vote-based]`);
      } else {
        lines.push(`degenerate fit`);
      }
      if (refinedFit) {
        // Ground-truth-free cost values (the algebraic-vs-geometric residual
        // this refinement actually optimizes) -- the row/col err % above is
        // the ground-truth-based check on top, only available in this lab.
        lines.push(`LM: ${refinedFit.iterations} iters, cost ${refinedFit.initialCost.toExponential(2)} -> ${refinedFit.finalCost.toExponential(2)}`);
      }
      if (spacing) {
        // Ground truth: the floor sits at world y=0, so the camera's true
        // distance to it along the true normal is just its own height.
        const trueDist = camPos.y;
        const distU = spacing.distanceU, distV = spacing.distanceV;
        const errU = (Math.abs(distU - trueDist) / trueDist) * 100;
        const errV = (Math.abs(distV - trueDist) / trueDist) * 100;
        lines.push(`dist U ${distU.toFixed(2)} (${errU.toFixed(1)}% err)  dist V ${distV.toFixed(2)} (${errV.toFixed(1)}% err)  true ${trueDist.toFixed(2)}  [rough, rtSize buckets]`);
        if (refinedSpacing) {
          const rDistU = refinedSpacing.distanceU, rDistV = refinedSpacing.distanceV;
          const rErrU = (Math.abs(rDistU - trueDist) / trueDist) * 100;
          const rErrV = (Math.abs(rDistV - trueDist) / trueDist) * 100;
          lines.push(`dist U ${rDistU.toFixed(2)} (${rErrU.toFixed(1)}% err)  dist V ${rDistV.toFixed(2)} (${rErrV.toFixed(1)}% err)  [refined, adaptive buckets]`);
        }
        if (finalSpacing) {
          const fDistU = finalSpacing.distanceU, fDistV = finalSpacing.distanceV;
          const fErrU = (Math.abs(fDistU - trueDist) / trueDist) * 100;
          const fErrV = (Math.abs(fDistV - trueDist) / trueDist) * 100;
          lines.push(`dist U ${fDistU.toFixed(2)} (${fErrU.toFixed(1)}% err)  dist V ${fDistV.toFixed(2)} (${fErrV.toFixed(1)}% err)  [final, post-Phase-3 orientation]`);
        }
      } else if (quadricPair) {
        lines.push(`spacing: no period found`);
      }
      if (lastPositionLMResult) {
        lines.push(`photoLM: ${lastPositionLMResult.iterations} iters, cost ${lastPositionLMResult.initialCost.toExponential(2)} -> ${lastPositionLMResult.finalCost.toExponential(2)}`);
        // The row/col err line above is Phase 1's OUTPUT only -- Phase 3
        // (this block) goes on to overwrite lastRecoveredAxes.Drow/Dcol/
        // Dnormal in place with its own, usually much better, jointly-
        // refined estimate (confirmed live: 2.33/2.01 degrees pre-Phase-3
        // vs 0.41/0.36 post, same capture) -- but nothing downstream ever
        // re-reported that, so every orientation-error number this debug
        // panel ever showed after Phase 3 landed was silently stale. This
        // is the actual final orientation decode uses.
        if (lastRecoveredAxes) {
          const finalRowErr = angleBetweenDegV(lastRecoveredAxes.Drow, ROW_DIR);
          const finalColErr = angleBetweenDegV(lastRecoveredAxes.Dcol, COL_DIR);
          lines.push(`row err ${finalRowErr.toFixed(2)}°  col err ${finalColErr.toFixed(2)}°  [Phase 3, final]`);
        }
      }
      lines.push(`votes ${(t1 - t0).toFixed(0)}ms  fit ${(t2 - t1).toFixed(0)}ms  LM ${(t2b - t2).toFixed(0)}ms  spacing ${(t3 - t2b).toFixed(0)}ms  refine ${(t3b - t3).toFixed(0)}ms  decode ${(t4 - t3b).toFixed(0)}ms`);
      axesReadout.textContent = lines.join('\n');
      // drawMarginalLines also calls this every frame, but ONLY while in
      // 'projected' mode -- needed here too so a capture triggered from a
      // different mode still updates positionReadout instead of leaving it
      // stale until the user happens to switch to Projected Cam.
      updatePositionReadoutText();
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
  else { hideMarginalLines(); hideSampleLattice(); }
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

// Same persistence as every slider/checkbox/mode -- a collapsed panel
// shouldn't pop back open on a dev-server restart or reload.
function setPanelCollapsed(collapsed: boolean) {
  panel.classList.toggle('collapsed', collapsed);
  panelToggle.classList.toggle('collapsed', collapsed);
  panelToggle.textContent = collapsed ? '›' : '‹';
  persistControl('panelCollapsed', collapsed ? '1' : '0');
}
panelToggle.addEventListener('click', () => setPanelCollapsed(!panel.classList.contains('collapsed')));
setPanelCollapsed(savedControls['panelCollapsed'] === '1');

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
bindCheckbox('showSampleLattice', (v) => (state.showSampleLattice = v));
bindCheckbox('orientationLM', (v) => (state.orientationLM = v));
bindCheckbox('positionLM', (v) => (state.positionLM = v));

bindSlider('simNoise', (v) => { state.simNoise = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindSlider('simBlur', (v) => { state.simBlur = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindSlider('simGradRadius', (v) => { state.simGradRadius = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindSlider('captureSupersample', (v) => { state.captureSupersample = v; resizeCaptureBuffers(); }, (v) => `${v.toFixed(0)}x`);
bindSlider('coherenceRadius', (v) => { state.coherenceRadius = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindSlider('tangentWalkMaxSteps', (v) => { state.tangentWalkMaxSteps = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindSlider('tangentWalkDeviationDeg', (v) => { state.tangentWalkDeviationDeg = v; markCaptureDirty(); }, (v) => `${v.toFixed(0)}°`);
bindSlider('tangentWalkMagFraction', (v) => { state.tangentWalkMagFraction = v; markCaptureDirty(); }, (v) => v.toFixed(2));
bindSlider('tangentWalkGraceSamples', (v) => { state.tangentWalkGraceSamples = v; markCaptureDirty(); }, (v) => v.toFixed(0));
bindRadioGroup('fieldView', (v) => { state.fieldView = v as 'raw' | 'antialiased' | 'downsampled' | 'noised' | 'gradient' | 'walked' | 'agreement' | 'effective'; markCaptureDirty(); });
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
    // Reserves MARGINAL_THICKNESS px on the right and bottom BEFORE fitting
    // the letterbox rect, unlike 'through' above -- otherwise, whichever
    // dimension isn't letterboxed (the common case: window wider than
    // RT_ASPECT means h fills innerHeight exactly, y=0) leaves zero room for
    // marginalBottomCanvas below it, pushing that graph fully off-screen.
    const availW = innerWidth - MARGINAL_THICKNESS;
    const availH = innerHeight - MARGINAL_THICKNESS;
    const winAspect = availW / availH;
    let w = availW, h = availH, x = 0, y = 0;
    if (winAspect > RT_ASPECT) { w = availH * RT_ASPECT; x = (availW - w) / 2; }
    else { h = availW / RT_ASPECT; y = (availH - h) / 2; }
    // renderProjectedViewport ultimately calls renderer.setViewport/setScissor,
    // which (like WebGL generally) measure y from the BOTTOM of the canvas --
    // y above is a CSS-style top-down offset (correct as-is for
    // drawMarginalLines/style.top). Same conversion renderPreviewViewport's
    // PIP call already applies (innerHeight - y - h). Harmless no-op in the
    // 'through' branch above (its margin is symmetric top/bottom, so y and
    // its flip are equal there) but NOT a no-op here: the reserved margin is
    // bottom-only, so skipping this shifted the rendered image down into
    // marginalBottomCanvas's strip -- reported live as "the bottom plot
    // still overlaps the quad a bit".
    renderProjectedViewport(x, innerHeight - y - h, w, h);
    drawMarginalLines(x, y, w, h);
    drawSampleLattice(x, y, w, h);
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
