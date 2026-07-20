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
//
// ── Stage A architecture note ────────────────────────────────────────────
// This file models N cameras (exactly one lives at runtime today — see the
// "Camera model" section below) instead of one hardcoded camera. Every
// mutable THREE object / buffer that used to be a single module-level
// binding now lives on a `Camera` object; truly shared things (the scene,
// renderer, floor, De Bruijn pattern, the world-view orbit controls,
// ROW_DIR/COL_DIR, MATH_QUAT) stay module-level. `activeCamera()` is the
// camera whose detail panel (sliders/readouts/Through-Cam/Projected-Cam/
// Inside-Sphere) is currently shown; the cheap per-frame gizmo/overlay
// update loop in animate() runs for every camera in `cameras`, while the
// expensive preview-render/auto-capture work only ever runs for the active
// one — today that's a distinction without a difference (only one camera
// exists), but it's what lets Stage B add more cameras without another
// big-bang rewrite.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { generateTorus, buildLookupTableSparse, buildTorusFromCandidate, ORDER5_CANDIDATE } from './debruijn.ts';
import { toGrayscale, binarize } from './decode.ts';
import { jacobiEigenSymmetric, smallestEigenvector } from './linalg.ts';

type Mode = 'world' | 'through' | 'inside' | 'projected';
type FieldView = 'raw' | 'antialiased' | 'downsampled' | 'noised' | 'gradient' | 'walked' | 'agreement' | 'effective';

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
const gradientArrowCanvas = document.getElementById('gradientArrowOverlay') as HTMLCanvasElement;
const gradientArrowCtx = gradientArrowCanvas.getContext('2d')!;
const toggleGradientArrowBtn = document.getElementById('toggleGradientArrow') as HTMLButtonElement;
const toggleGradientArrowModeBtn = document.getElementById('toggleGradientArrowMode') as HTMLButtonElement;
const toggleTangentWalkPathBtn = document.getElementById('toggleTangentWalkPath') as HTMLButtonElement;
const arrowToggles = document.getElementById('arrowToggles') as HTMLDivElement;
const cameraDetailsSection = document.getElementById('cameraDetailsSection') as HTMLDivElement;
const simDistortionSection = document.getElementById('simDistortionSection') as HTMLDivElement;

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

// ── Global settings ──────────────────────────────────────────────────────
//
// Everything that applies regardless of which camera is active/exists --
// per the N-camera plan's explicit global/per-camera split. Deliberately
// tiny: `mode` because the 3D canvas has exactly one current view regardless
// of camera count/selection, `showFloor`/`floorCellOutlineSubdiv` because
// the floor itself is one shared object, not owned by any camera.
const globalState = {
  mode: 'world' as Mode,
  showFloor: true,
  floorCellOutlineSubdiv: 0,
};

// ── Per-camera settings ──────────────────────────────────────────────────
//
// Everything that used to live in the single module-level `state` object,
// split into what's common to both camera types and what's type-specific.
// See createDefaultCameraSettings below for the actual default values
// (mirrors this file's pre-Stage-A `state` initializer exactly).

interface CameraSettingsCommon {
  showSphere: boolean; showCircles: boolean; showPoles: boolean; showFrustum: boolean; showPatch: boolean;
  showGizmoBody: boolean; showRecoveredFloor: boolean; showSampleLattice: boolean;
  showTrueContamination: boolean; showReconstructedContamination: boolean; hideField: boolean;
  showGradientArrow: boolean; showGradientArrowPerpendicular: boolean; gradientArrowScale: number;
  showTangentWalkPath: boolean;
  simGradRadius: number; coherenceRadius: number;
  tangentWalkMaxSteps: number; tangentWalkDeviationDeg: number; tangentWalkMagFraction: number; tangentWalkGraceSamples: number;
  tangentWalkAdaptive: boolean;
  circleSamplePercentMin: number; circleSamplePercentMax: number;
  showRecoveredPoles: boolean;
  showAxisVectors: boolean;
  showTopCircles: boolean;
  weightSharpenPower: number;
  orientationLM: boolean;
  positionLM: boolean;
  fieldView: FieldView;
  axesAutoCapture: boolean; axesCaptureIntervalMs: number;
  viewportW: number; viewportH: number; aspectLocked: boolean;
}
interface SimulatedCameraSettings extends CameraSettingsCommon {
  camX: number; camY: number; camZ: number;
  camYawDeg: number; camPitchDeg: number;
  focalMM: number;
  simNoise: number; simBlur: number; captureSupersample: number;
}
interface PhysicalCameraSettings extends CameraSettingsCommon {
  realCaptureFovDeg: number;
}

function createDefaultCommonSettings(): CameraSettingsCommon {
  return {
    showSphere: true, showCircles: true, showPoles: true, showFrustum: true, showPatch: true, showGizmoBody: true, showRecoveredFloor: true, showSampleLattice: false,
    showTrueContamination: false, showReconstructedContamination: false, hideField: false,
    showGradientArrow: false, showGradientArrowPerpendicular: false, gradientArrowScale: 2,
    showTangentWalkPath: false,
    simGradRadius: 1, coherenceRadius: 1,
    // See the pre-Stage-A history for the full derivation of these tangent-walk
    // defaults (guided tangent walk, simNoise=8 stability etc.) -- unchanged.
    tangentWalkMaxSteps: 12, tangentWalkDeviationDeg: 45, tangentWalkMagFraction: 0.15, tangentWalkGraceSamples: 3,
    tangentWalkAdaptive: false,
    circleSamplePercentMin: 0, circleSamplePercentMax: 10,
    showRecoveredPoles: true,
    showAxisVectors: false,
    showTopCircles: true,
    weightSharpenPower: 4,
    orientationLM: true,
    positionLM: true,
    fieldView: 'noised',
    axesAutoCapture: false, axesCaptureIntervalMs: 500,
    viewportW: 512, viewportH: 384, aspectLocked: false,
  };
}
function createDefaultSimulatedSettings(): SimulatedCameraSettings {
  return {
    ...createDefaultCommonSettings(),
    camX: 0, camY: 4, camZ: 8,
    camYawDeg: 0, camPitchDeg: -20,
    focalMM: 26,
    simNoise: 8, simBlur: 1, captureSupersample: 2,
  };
}
function createDefaultPhysicalSettings(): PhysicalCameraSettings {
  return {
    ...createDefaultCommonSettings(),
    realCaptureFovDeg: 65,
  };
}

const SENSOR_WIDTH_MM = 36; // 35mm-equivalent convention, so "focal (mm eq.)" reads like a familiar lens spec
const SPHERE_RADIUS = 2.5;
const GRID_STEP = 1; // world units per pattern cell
const VIS_HALF_EXTENT = 20; // cap on how many grid lines get a reference line / great circle drawn (perf + clutter, independent of the floor's true size)
const CIRCLE_SEGMENTS = 96;
const PATCH_RES = 48; // patch-mesh tessellation, shared by every camera's own patch geometry

// ── Scene (shared by every camera) ──────────────────────────────────────

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
const ORDER = parseInt(new URLSearchParams(location.search).get('order') ?? '5', 10);
// Order 5's full R x C torus (~33.5M cells) has no known efficient
// construction free of D4 rotation/reflection collisions, so it isn't used
// directly -- ORDER5_CANDIDATE is a searched 256x256 sub-region with a low
// (1.027%) residual collision rate instead (see buildTorusFromCandidate's
// header comment in debruijn.ts).
const debruijn = ORDER === 5 ? buildTorusFromCandidate(5, ORDER5_CANDIDATE) : generateTorus(ORDER);
const { R, C, torus } = debruijn;
// For decoding an ORDER x ORDER sampled bit window back into an absolute
// torus (row,col) position -- see runPositionDecode.
const debruijnLookup = buildLookupTableSparse(debruijn);
// One instance of the torus, sized in world units at GRID_STEP per cell —
// NOT tiled. Half-extents, since grid lines/great circles below are indexed
// out from the origin at the pattern's center.
const HALF_C = (C * GRID_STEP) / 2;
const HALF_R = (R * GRID_STEP) / 2;

const patternCanvas = document.createElement('canvas');
const pctx = patternCanvas.getContext('2d')!;

// Cell subdivision, directly driven by globalState.floorCellOutlineSubdiv (0:
// off, exactly today's 1-texture-pixel-per-cell flat color) -- BORDER is
// the outermost ring's thickness in subdivided pixels, always the OPPOSITE
// of the cell's own color. At subdiv 1-2, BORDER(1) alone already covers the
// whole cell (no room left for an inner square), so the cell renders as
// solid opposite-color -- a real, continuous endpoint of the same formula.
const FLOOR_OUTLINE_BORDER = 1;

function rebuildFloorTexture() {
  const subdiv = globalState.floorCellOutlineSubdiv;
  const s = subdiv > 0 ? subdiv : 1;
  const width = C * s, height = R * s;
  patternCanvas.width = width; patternCanvas.height = height;
  const img = pctx.createImageData(width, height);
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) {
      // 1 -> dark, 0 -> light -- matches scripts/generate-debruijn-torus.ts's
      // canonical convention and binarize's "dark -> 1" intent (src/decode.ts).
      const inner = torus[r][c] ? 20 : 235;
      const outer = torus[r][c] ? 235 : 20;
      for (let sy = 0; sy < s; sy++) {
        const py = r * s + sy;
        const borderY = subdiv > 0 && (sy < FLOOR_OUTLINE_BORDER || sy >= s - FLOOR_OUTLINE_BORDER);
        for (let sx = 0; sx < s; sx++) {
          const px = c * s + sx;
          const borderX = subdiv > 0 && (sx < FLOOR_OUTLINE_BORDER || sx >= s - FLOOR_OUTLINE_BORDER);
          const v = (borderX || borderY) ? outer : inner;
          const i = (py * width + px) * 4;
          img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
          img.data[i + 3] = 255;
        }
      }
    }
  }
  pctx.putImageData(img, 0, 0);
  floorTex.needsUpdate = true;
}

const floorTex = new THREE.CanvasTexture(patternCanvas);
floorTex.wrapS = THREE.RepeatWrapping;
floorTex.wrapT = THREE.RepeatWrapping;
floorTex.magFilter = THREE.NearestFilter;
floorTex.colorSpace = THREE.SRGBColorSpace;
floorTex.repeat.set(1, 1); // exactly one instance of the torus, not tiled
rebuildFloorTexture(); // paint the initial pattern now that floorTex/patternCanvas both exist

const floorMat = new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.95 });
const floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(C * GRID_STEP, R * GRID_STEP), floorMat);
floorMesh.rotation.x = -Math.PI / 2;
scene.add(floorMesh);

// Colored reference lines at the same integer cell boundaries the great
// circles below are computed from — row family (world +X direction, red)
// and column family (world +Z direction, blue), matching the sphere colors.
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

// Debug layer: gizmoCam (whichever camera is rendering "what the real camera
// sees") never sees layer 1, so its capture is a clean shot of just the
// floor -- every debug/overlay object (grid lines, gizmo bodies, sphere
// shell, poles, frustum outline, patch mesh, camera helper) lives on it.
const DEBUG_LAYER = 1;
for (const o of [rowGridLines, colGridLines]) o.layers.set(DEBUG_LAYER);

// ── Math helpers (pure, shared) ──────────────────────────────────────────

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

function cornerDir(u: number, v: number, quat: THREE.Quaternion, vFovRad: number, aspect: number): THREE.Vector3 {
  const halfV = vFovRad / 2;
  const yc = Math.tan(halfV) * v;
  const xc = Math.tan(halfV) * aspect * u;
  return new THREE.Vector3(xc, yc, -1).normalize().applyQuaternion(quat);
}

function angleBetweenDegV(a: THREE.Vector3, b: THREE.Vector3): number {
  return THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(Math.abs(a.dot(b)), -1, 1)));
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

// Row/col great-circle family "k" values, shared by every camera's own
// circle-pool meshes -- purely derived from the (global, shared) pattern
// extent, not camera state.
const rowLineKs: number[] = [];
for (let k = -Math.min(VIS_HALF_EXTENT, HALF_R); k <= Math.min(VIS_HALF_EXTENT, HALF_R); k += GRID_STEP) rowLineKs.push(k);
const colLineKs: number[] = [];
for (let k = -Math.min(VIS_HALF_EXTENT, HALF_C); k <= Math.min(VIS_HALF_EXTENT, HALF_C); k += GRID_STEP) colLineKs.push(k);

// ── Shared result/field types (referenced by the Camera interfaces below) ─

interface Vote { n: THREE.Vector3; weight: number }
interface GradientField { fx: Float64Array; fy: Float64Array; w: number; h: number; r: number }
interface ProjectedBins { minU: number; maxU: number; minV: number; maxV: number; binWidthU: number; binWidthV: number; w: number; h: number }
interface Marginals {
  colSum: Float64Array; rowSum: Float64Array; colSumCy: Float64Array; rowHueCx: Float64Array; rowSumCy: Float64Array;
  colMag: Float64Array; rowMag: Float64Array;
  colPeriod: number | null; rowPeriod: number | null; colPhase: number; rowPhase: number;
}
// Set by runAxesReconstruction on a successful capture; consumed by
// buildProjectedTexture. distance is the average of the U/V estimates.
interface RecoveredAxes { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3; distance: number }
interface PositionDecodeResult {
  row: number; col: number; consistency: number; votes: number; totalWindows: number;
  camPos: THREE.Vector3;
  // The camera's TRUE world orientation, solved entirely from the pattern --
  // see solveRecoveredCamQuat. Anything placed into the actual 3D scene
  // needs this to convert lastRecoveredAxes' Drow/Dcol/Dnormal (expressed in
  // MATH_QUAT's fixed math frame) into true world space first.
  recoveredCamQuat: THREE.Quaternion;
}
// u,v are the sample's world position (relative to camera, in Drow/Dcol
// units); px,py are where that point projects to in the CURRENT capture's
// pixel space, TOP-DOWN row convention. valid is false when the point is
// behind the camera or projects outside the image entirely.
interface DecodeSamplePoint { u: number; v: number; px: number; py: number; valid: boolean; bit: number }
interface DecodeSampleGrid { rows: number; cols: number; zeroI: number; zeroJ: number; points: DecodeSamplePoint[][] }
interface DecodeCellDebug { bit: number; correct: boolean }
interface VoteResult { orientation: number; anchorRow: number; anchorCol: number; votes: number; totalWindows: number }
interface OrientationFit { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 }
interface PositionFit extends OrientationFit { worldX0: number; worldZ0: number; distance: number }
interface PhotometricSample { px: number; py: number; observed: number }

// ── Camera model ─────────────────────────────────────────────────────────
//
// Exactly one Camera exists at runtime in Stage A (the `useRealCapture`
// checkbox destroys and recreates it as the other type -- see
// switchActiveCameraType below), but every per-camera THREE object/buffer
// already lives on this object rather than as a module-level singleton, so
// Stage B can add more without touching this shape again.

interface CameraBase {
  id: string;
  name: string;
  color: THREE.Color;

  // -- recovered/decoded state -- already fully source-agnostic (MATH_QUAT-
  // frame recovery + solveRecoveredCamQuat), unchanged by this stage.
  lastRecoveredAxes: RecoveredAxes | null;
  lastPositionDecode: PositionDecodeResult | null;
  lastDecodeGrid: DecodeSampleGrid | null;
  lastDecodeRotated: DecodeSampleGrid | null;
  lastDecodeCorrectness: (DecodeCellDebug | null)[][] | null;
  lastProjectedBins: ProjectedBins | null;
  lastMarginals: Marginals | null;
  lastVotes: Vote[];
  axesComputed: boolean;
  axesCapturing: boolean;
  lastAxesCapture: number;

  // -- capture/analysis buffers, shared shape for both camera types --
  rtSize: { w: number; h: number };
  aspect: number; // rtSize.w / rtSize.h -- replaces the old module-level RT_ASPECT
  pipRect: { x: number; y: number; w: number; h: number };
  captureDirty: boolean;
  lastPreviewUpdate: number;
  lastNoisedPreviewGray: Float64Array | null;
  lastDisplayedVectorField: GradientField | null;
  lastEffectiveField: GradientField | null;

  distortedPreviewData: Uint8Array; distortedPreviewTex: THREE.DataTexture;
  projectedPreviewData: Uint8Array; projectedPreviewTex: THREE.DataTexture;
  trueContamData: Uint8Array; trueContamTex: THREE.DataTexture;
  reconContamData: Uint8Array; reconContamTex: THREE.DataTexture;

  // -- THREE objects: recovered side (both camera types have these) --
  recoveredCamGizmo: THREE.Mesh; recoveredCamAxes: THREE.AxesHelper;
  recoveredRowPoleA: THREE.Mesh; recoveredRowPoleB: THREE.Mesh;
  recoveredColPoleA: THREE.Mesh; recoveredColPoleB: THREE.Mesh;
  recoveredFloorOverlayMat: THREE.MeshBasicMaterial; recoveredFloorOverlay: THREE.Mesh;

  // -- Great-sphere group: repositioned (not rotated) to the camera's own
  // origin each frame, since every direction it draws is expressed in WORLD
  // axes. --
  sphereAnchor: THREE.Object3D;
  sphereShell: THREE.Mesh;
  circlesGroup: THREE.Group;
  rowCirclePool: THREE.Line[]; colCirclePool: THREE.Line[];
  frustumLine: THREE.LineLoop;
  patchGeo: THREE.BufferGeometry; patchMat: THREE.MeshBasicMaterial; patchMesh: THREE.Mesh;
  gradientCirclesGeo: THREE.BufferGeometry; gradientCirclesMat: THREE.LineBasicMaterial; gradientCirclesLines: THREE.LineSegments;
  axisVectorsGeo: THREE.BufferGeometry; axisVectorsMat: THREE.LineBasicMaterial; axisVectorsLines: THREE.LineSegments;
}
interface SimulatedCamera extends CameraBase {
  type: 'simulated';
  settings: SimulatedCameraSettings;
  // Ground-truth pose, driven by settings.camX/Y/Z/camYawDeg/camPitchDeg --
  // see updateGizmo.
  camPos: THREE.Vector3; camQuat: THREE.Quaternion;
  gizmoCam: THREE.PerspectiveCamera; gizmoBody: THREE.Mesh; gizmoAxes: THREE.AxesHelper;
  camHelper: THREE.CameraHelper;
  camRT: THREE.WebGLRenderTarget;
  captureRTSize: { w: number; h: number };
  // Ground-truth pole markers (ROW_DIR/COL_DIR comparison) -- no equivalent
  // for a physical camera, since there's no ground truth to compare against.
  polesGroup: THREE.Group;
  rowPoleA: THREE.Mesh; rowPoleB: THREE.Mesh; colPoleA: THREE.Mesh; colPoleB: THREE.Mesh;
}
interface PhysicalCamera extends CameraBase {
  type: 'physical';
  settings: PhysicalCameraSettings;
  lastRealCaptureGray: Float64Array | null;
  lastRealCaptureW: number; lastRealCaptureH: number;
}
type Camera = SimulatedCamera | PhysicalCamera;

const cameras = new Map<string, Camera>();
let activeCameraId = '';
function activeCamera(): Camera | undefined { return cameras.get(activeCameraId); }
function isSimulated(camera: Camera): camera is SimulatedCamera { return camera.type === 'simulated'; }
function isPhysical(camera: Camera): camera is PhysicalCamera { return camera.type === 'physical'; }

let nextCameraSerial = 1;
// Assigned in creation order, keyed off nextCameraSerial (never reused, even
// across deletions -- reusing a color the moment a camera's slot frees up
// would risk two SIMULTANEOUSLY existing cameras sharing a color, which
// defeats the entire point). Falls back to a random, well-saturated HSL hue
// once the fixed palette runs out, rather than capping how many cameras can
// exist.
const CAMERA_COLOR_PALETTE = [0xffcc44, 0x33dd55, 0xff5588, 0x55ccff, 0xcc88ff, 0xff8833, 0x33ffcc, 0xdd4444];
function nextCameraColor(): THREE.Color {
  const idx = nextCameraSerial - 1;
  if (idx < CAMERA_COLOR_PALETTE.length) return new THREE.Color(CAMERA_COLOR_PALETTE[idx]);
  return new THREE.Color().setHSL(Math.random(), 0.65, 0.55);
}

// ── Reusable full-screen quad renderers (shared infra, NOT per-camera) ───
//
// A plain full-screen textured quad, rendered instead of a live gizmoCam
// scene pass for the PIP box / Through-Cam / Projected-Cam / contamination
// overlays -- each camera owns its OWN texture (distortedPreviewTex etc,
// see CameraBase above), but the Scene/Material/Mesh doing the actual blit
// is shared, reusable machinery: swap `.map` to whichever camera's texture
// needs drawing right before each render call.
const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
function makeQuadRenderer(matOpts: THREE.MeshBasicMaterialParameters) {
  const mat = new THREE.MeshBasicMaterial(matOpts);
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
  return { mat, scene };
}
const previewQuad = makeQuadRenderer({});
const projectedQuad = makeQuadRenderer({});
const trueContamQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
const reconContamQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
function renderQuad(q: { mat: THREE.MeshBasicMaterial; scene: THREE.Scene }, tex: THREE.Texture, x: number, y: number, w: number, h: number) {
  q.mat.map = tex;
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(q.scene, quadCam);
}
function renderPreviewViewport(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(previewQuad, camera.distortedPreviewTex, x, y, w, h); }
function renderProjectedViewport(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(projectedQuad, camera.projectedPreviewTex, x, y, w, h); }
function renderTrueContamOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(trueContamQuad, camera.trueContamTex, x, y, w, h); }
function renderReconContamOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(reconContamQuad, camera.reconContamTex, x, y, w, h); }

// ── Camera factories ─────────────────────────────────────────────────────
//
// Build every per-camera THREE object/buffer this file used to allocate
// once at module scope, add them to the shared `scene`, and return a fully
// populated Camera. Real allocate-N-of-them functions -- addSimulatedCamera
// (Stage B's "+" button) and switchActiveCameraType both call these.

function makeCameraBaseParts(rtSize: { w: number; h: number }, color: THREE.Color) {
  const aspect = rtSize.w / rtSize.h;

  // Recovered/decoded pose gizmo: SOLID in the camera's own color -- solid
  // consistently means "recovered" across every camera, ground-truth (only
  // simulated cameras have one, see createSimulatedCamera) is the
  // translucent one instead.
  const recoveredCamGizmo = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 0.25, 0.4),
    new THREE.MeshStandardMaterial({ color }),
  );
  recoveredCamGizmo.visible = false;
  scene.add(recoveredCamGizmo);
  const recoveredCamAxes = new THREE.AxesHelper(0.6);
  recoveredCamGizmo.add(recoveredCamAxes);
  recoveredCamGizmo.traverse((o) => o.layers.set(DEBUG_LAYER));

  const sphereAnchor = new THREE.Object3D();
  scene.add(sphereAnchor);

  const sphereShell = new THREE.Mesh(
    new THREE.SphereGeometry(SPHERE_RADIUS, 48, 32),
    new THREE.MeshBasicMaterial({ color: 0x88aaff, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }),
  );
  sphereAnchor.add(sphereShell);
  sphereShell.layers.set(DEBUG_LAYER);

  const circlesGroup = new THREE.Group();
  sphereAnchor.add(circlesGroup);

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
  const rowCirclePool = buildCirclePool(rowLineKs.length, 0xff5555);
  const colCirclePool = buildCirclePool(colLineKs.length, 0x5599ff);

  const frustumLine = new THREE.LineLoop(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 }),
  );
  sphereAnchor.add(frustumLine);
  frustumLine.layers.set(DEBUG_LAYER);

  // Viewport-image patch: literally the camera's rendered pixels, wrapped
  // onto the sphere over exactly the solid angle the camera's frustum
  // subtends.
  const patchGeo = new THREE.BufferGeometry();
  {
    const verts: number[] = [], uvs: number[] = [], idx: number[] = [];
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

  const distortedPreviewData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const distortedPreviewTex = new THREE.DataTexture(distortedPreviewData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  distortedPreviewTex.flipY = false;
  distortedPreviewTex.colorSpace = THREE.SRGBColorSpace;

  const patchMat = new THREE.MeshBasicMaterial({ map: distortedPreviewTex, side: THREE.DoubleSide });
  const patchMesh = new THREE.Mesh(patchGeo, patchMat);
  sphereAnchor.add(patchMesh);
  patchMesh.layers.set(DEBUG_LAYER);

  const projectedPreviewData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const projectedPreviewTex = new THREE.DataTexture(projectedPreviewData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  projectedPreviewTex.flipY = false;
  projectedPreviewTex.colorSpace = THREE.SRGBColorSpace;

  const trueContamData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const trueContamTex = new THREE.DataTexture(trueContamData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  trueContamTex.flipY = false;

  const reconContamData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const reconContamTex = new THREE.DataTexture(reconContamData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  reconContamTex.flipY = false;

  // Reuses the SAME projectedPreviewTex "Projected Cam" mode already builds
  // (not a separate computation) as a decal on a plane placed at the
  // DECODED pose in the actual 3D world.
  const recoveredFloorOverlayMat = new THREE.MeshBasicMaterial({ map: projectedPreviewTex, side: THREE.DoubleSide, transparent: true, opacity: 0.92 });
  const recoveredFloorOverlay = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), recoveredFloorOverlayMat);
  recoveredFloorOverlay.visible = false;
  scene.add(recoveredFloorOverlay);
  recoveredFloorOverlay.layers.set(DEBUG_LAYER);

  function makeRecoveredPoleMarker(color: number): THREE.Mesh {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 8), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 }));
    m.layers.set(DEBUG_LAYER);
    m.visible = false;
    sphereAnchor.add(m);
    return m;
  }
  const recoveredRowPoleA = makeRecoveredPoleMarker(0xff0000);
  const recoveredRowPoleB = makeRecoveredPoleMarker(0xff0000);
  const recoveredColPoleA = makeRecoveredPoleMarker(0x0000ff);
  const recoveredColPoleB = makeRecoveredPoleMarker(0x0000ff);

  const gradientCirclesGeo = new THREE.BufferGeometry();
  const gradientCirclesMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.35 });
  const gradientCirclesLines = new THREE.LineSegments(gradientCirclesGeo, gradientCirclesMat);
  gradientCirclesLines.layers.set(DEBUG_LAYER);
  sphereAnchor.add(gradientCirclesLines);

  const axisVectorsGeo = new THREE.BufferGeometry();
  const axisVectorsMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6 });
  const axisVectorsLines = new THREE.LineSegments(axisVectorsGeo, axisVectorsMat);
  axisVectorsLines.layers.set(DEBUG_LAYER);
  axisVectorsLines.visible = false;
  sphereAnchor.add(axisVectorsLines);

  const base: Omit<CameraBase, 'id' | 'name' | 'color'> = {
    lastRecoveredAxes: null, lastPositionDecode: null, lastDecodeGrid: null, lastDecodeRotated: null,
    lastDecodeCorrectness: null, lastProjectedBins: null, lastMarginals: null, lastVotes: [],
    axesComputed: false, axesCapturing: false, lastAxesCapture: 0,
    rtSize: { ...rtSize }, aspect, pipRect: { x: 0, y: 0, w: 0, h: 0 }, captureDirty: true, lastPreviewUpdate: 0,
    lastNoisedPreviewGray: null, lastDisplayedVectorField: null, lastEffectiveField: null,
    distortedPreviewData, distortedPreviewTex, projectedPreviewData, projectedPreviewTex,
    trueContamData, trueContamTex, reconContamData, reconContamTex,
    recoveredCamGizmo, recoveredCamAxes,
    recoveredRowPoleA, recoveredRowPoleB, recoveredColPoleA, recoveredColPoleB,
    recoveredFloorOverlayMat, recoveredFloorOverlay,
    sphereAnchor, sphereShell, circlesGroup, rowCirclePool, colCirclePool, frustumLine,
    patchGeo, patchMat, patchMesh, gradientCirclesGeo, gradientCirclesMat, gradientCirclesLines,
    axisVectorsGeo, axisVectorsMat, axisVectorsLines,
  };
  return base;
}

function createSimulatedCamera(color: THREE.Color): SimulatedCamera {
  const settings = createDefaultSimulatedSettings();
  const rtSize = { w: Math.round(settings.viewportW), h: Math.round(settings.viewportH) };
  const base = makeCameraBaseParts(rtSize, color);
  const aspect = rtSize.w / rtSize.h;

  const gizmoCam = new THREE.PerspectiveCamera(50, aspect, 0.05, 500);
  scene.add(gizmoCam);

  // Ground-truth (assumed) pose gizmo: TRANSLUCENT in the same color the
  // recovered gizmo above uses solid -- see makeCameraBaseParts' comment.
  const gizmoBody = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 0.25, 0.4),
    new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.4 }),
  );
  scene.add(gizmoBody);
  const gizmoAxes = new THREE.AxesHelper(0.6);
  gizmoBody.add(gizmoAxes);
  gizmoBody.traverse((o) => o.layers.set(DEBUG_LAYER));

  const camHelper = new THREE.CameraHelper(gizmoCam);
  scene.add(camHelper);
  camHelper.layers.set(DEBUG_LAYER);

  const captureRTSize = { w: rtSize.w * settings.captureSupersample, h: rtSize.h * settings.captureSupersample };
  const camRT = new THREE.WebGLRenderTarget(captureRTSize.w, captureRTSize.h, { colorSpace: THREE.SRGBColorSpace });

  const polesGroup = new THREE.Group();
  base.sphereAnchor.add(polesGroup);
  function makePoleMarker(color: number): THREE.Mesh {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), new THREE.MeshBasicMaterial({ color }));
    polesGroup.add(m);
    return m;
  }
  const rowPoleA = makePoleMarker(0xff5555);
  const rowPoleB = makePoleMarker(0xff5555);
  const colPoleA = makePoleMarker(0x5599ff);
  const colPoleB = makePoleMarker(0x5599ff);
  // traverse (not a single .set on the empty group) -- must run AFTER every
  // marker is added, since layers are per-object and not inherited by
  // children added later. Missing this left the ground-truth pole markers
  // on the default layer, which gizmoCam DOES render -- they'd leak into
  // the simulated capture instead of staying debug-only.
  polesGroup.traverse((o) => o.layers.set(DEBUG_LAYER));

  const camera: SimulatedCamera = {
    ...base,
    id: `sim-${nextCameraSerial}`, name: `Simulated ${nextCameraSerial}`, color,
    type: 'simulated', settings,
    camPos: new THREE.Vector3(), camQuat: new THREE.Quaternion(),
    gizmoCam, gizmoBody, gizmoAxes, camHelper, camRT, captureRTSize,
    polesGroup, rowPoleA, rowPoleB, colPoleA, colPoleB,
  };
  nextCameraSerial++;
  return camera;
}

function createPhysicalCamera(color: THREE.Color): PhysicalCamera {
  const settings = createDefaultPhysicalSettings();
  const rtSize = { w: Math.round(settings.viewportW), h: Math.round(settings.viewportH) };
  const base = makeCameraBaseParts(rtSize, color);
  const camera: PhysicalCamera = {
    ...base,
    id: `phys-${nextCameraSerial}`, name: `Physical ${nextCameraSerial}`, color,
    type: 'physical', settings,
    lastRealCaptureGray: null, lastRealCaptureW: 0, lastRealCaptureH: 0,
  };
  nextCameraSerial++;
  return camera;
}

// Disposes every THREE object/geometry/material/texture/render-target a
// camera owns and removes them from `scene` -- the mirror image of the
// factories above. Stage A only ever calls this from
// switchActiveCameraType (toggling useRealCapture), but it's written as a
// real, general "tear down one camera completely" function since Stage B's
// destroyCamera(id) needs exactly this.
function destroyCamera(camera: Camera) {
  const disposeObj = (o: THREE.Object3D) => {
    scene.remove(o);
    o.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const mat = (child as any).material;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else if (mat) mat.dispose();
    });
  };
  disposeObj(camera.recoveredCamGizmo);
  disposeObj(camera.sphereAnchor); // takes sphereShell/circlesGroup/pools/frustumLine/patchMesh/pole markers/gradientCircles/axisVectors with it
  disposeObj(camera.recoveredFloorOverlay);
  camera.distortedPreviewTex.dispose();
  camera.projectedPreviewTex.dispose();
  camera.trueContamTex.dispose();
  camera.reconContamTex.dispose();
  if (isSimulated(camera)) {
    disposeObj(camera.gizmoBody);
    scene.remove(camera.gizmoCam);
    scene.remove(camera.camHelper);
    camera.camHelper.dispose();
    camera.camRT.dispose();
  }
  cameras.delete(camera.id);
}

// ── Spherical-Hough prototype: noise/blur/downsample primitives ─────────

// Tiny seeded PRNG (mulberry32) so noise is reproducible rather than
// Math.random()-fresh on every capture.
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

function addGaussianNoise(gray: Float64Array, std: number) {
  if (std <= 0) return;
  const rng = mulberry32(NOISE_SEED);
  for (let i = 0; i < gray.length; i++) {
    const u1 = Math.max(1e-9, rng()), u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    gray[i] = Math.min(255, Math.max(0, gray[i] + z * std));
  }
}

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
// of radius, via a running sum per row/column slid one pixel at a time.
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

function flipRowsF64(src: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    const srcRow = h - 1 - y;
    out.set(src.subarray(srcRow * w, (srcRow + 1) * w), y * w);
  }
  return out;
}

function applyAntialiasFilter(gray: Float64Array, w: number, h: number, supersample: number): Float64Array {
  return separableBoxBlur(gray, w, h, Math.max(1, Math.round(supersample / 2)));
}

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

// Magnitude of the local VECTOR SUM of gradients (double-angle folded so
// alternating-polarity edges reinforce instead of cancelling) -- see
// pre-Stage-A history for the full derivation. Normalized against this
// frame's own RAW (unsmoothed) max magnitude.
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

function computeEffectiveGradientField(field: GradientField, agreement: Float64Array): GradientField {
  const { fx, fy, w, h, r } = field;
  const n = w * h;
  const efx = new Float64Array(n), efy = new Float64Array(n);
  for (let i = 0; i < n; i++) { efx[i] = fx[i] * agreement[i]; efy[i] = fy[i] * agreement[i]; }
  return { fx: efx, fy: efy, w, h, r };
}

// ── Display: colorizes a value field, only for whichever one is on screen ─

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

function fillGrayscalePreview(gray: Float64Array, out: Uint8Array) {
  for (let i = 0; i < gray.length; i++) {
    const v = Math.max(0, Math.min(255, gray[i]));
    const o = i * 4;
    out[o] = v; out[o + 1] = v; out[o + 2] = v; out[o + 3] = 255;
  }
}

// ── Guided tangent walk ──────────────────────────────────────────────────
//
// Fixed-direction walk: seeded once from the seed pixel's own gradient, not
// adaptively re-steered. Every tunable comes from `settings` (the active
// camera's own CameraSettingsCommon) now, instead of a module-level `state`.
function guidedTangentDirection(
  settings: CameraSettingsCommon,
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
  const maxSteps = settings.tangentWalkMaxSteps;
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(settings.tangentWalkDeviationDeg));
  const magFraction = settings.tangentWalkMagFraction;
  const grace = settings.tangentWalkGraceSamples;
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
        continue;
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
  const avgTheta = Math.atan2(sumSin, sumCos) / 2;
  return { fx: Math.cos(avgTheta) * seedMag, fy: Math.sin(avgTheta) * seedMag };
}

// Adaptive variant -- re-steers at every step using the CURRENT running-
// average direction instead of always sampling along a fixed straight line
// from the seed. settings.tangentWalkAdaptive toggles between the two.
function guidedTangentDirectionAdaptive(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { fx: number; fy: number } {
  const seedTheta = Math.atan2(seedFy, seedFx);
  const seedMag = Math.hypot(seedFx, seedFy);
  const seedCos = Math.cos(2 * seedTheta) * seedMag, seedSin = Math.sin(2 * seedTheta) * seedMag;
  const maxSteps = settings.tangentWalkMaxSteps;
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(settings.tangentWalkDeviationDeg));
  const magFraction = settings.tangentWalkMagFraction;
  const grace = settings.tangentWalkGraceSamples;

  let totalCos = 0, totalSin = 0;
  for (const sign of [1, -1]) {
    let sumCos = seedCos, sumSin = seedSin, runningMag = seedMag, sampleCount = 1;
    let curX = x, curY = y;
    let violations = 0;
    for (let k = 1; k <= maxSteps; k++) {
      const avgTheta = Math.atan2(sumSin, sumCos) / 2;
      const tdx = -Math.sin(avgTheta), tdy = Math.cos(avgTheta);
      curX += sign * tdx; curY += sign * tdy;
      const sx = Math.round(curX), sy = Math.round(curY);
      if (sx < 0 || sx >= w || sy < 0 || sy >= h) break;
      const si = sy * w + sx;
      const sfx = fx[si], sfy = fy[si];
      const mag = Math.hypot(sfx, sfy);
      if (mag === 0 || mag < runningMag * magFraction) {
        violations++;
        if (violations >= grace) break;
        continue;
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
    totalCos += sumCos - seedCos;
    totalSin += sumSin - seedSin;
  }
  totalCos += seedCos; totalSin += seedSin;
  const avgTheta = Math.atan2(totalSin, totalCos) / 2;
  return { fx: Math.cos(avgTheta) * seedMag, fy: Math.sin(avgTheta) * seedMag };
}

// Single dispatch point used by every REAL (non-diagnostic) caller.
function guidedTangentDirectionForWalk(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { fx: number; fy: number } {
  return settings.tangentWalkAdaptive
    ? guidedTangentDirectionAdaptive(settings, fx, fy, w, h, x, y, seedFx, seedFy)
    : guidedTangentDirection(settings, fx, fy, w, h, x, y, seedFx, seedFy);
}

function computeWalkedGradientField(settings: CameraSettingsCommon, field: GradientField): GradientField {
  const { fx, fy, w, h, r } = field;
  const walkedFx = new Float64Array(fx.length), walkedFy = new Float64Array(fy.length);
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      if (fx[i] === 0 && fy[i] === 0) continue;
      const walked = guidedTangentDirectionForWalk(settings, fx, fy, w, h, x, y, fx[i], fy[i]);
      walkedFx[i] = walked.fx; walkedFy[i] = walked.fy;
    }
  }
  return { fx: walkedFx, fy: walkedFy, w, h, r };
}

// gray is expected to already be captureDistortedGrayscale's output.
function computeWorldVotes(
  settings: CameraSettingsCommon,
  gray: Float64Array, w: number, h: number,
  gradientRadius: number, agreementRadius: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Vote[] {
  const votes: Vote[] = [];
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  const field = computeGradientField(gray, w, h, gradientRadius);
  const agreement = computeGradientAgreementField(field, agreementRadius);
  const effective = computeEffectiveGradientField(field, agreement);
  const { fx, fy, r } = effective;
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      if (fx[i] === 0 && fy[i] === 0) continue;
      const walked = guidedTangentDirectionForWalk(settings, fx, fy, w, h, x, y, fx[i], fy[i]);
      let theta = Math.atan2(walked.fy, walked.fx);
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
      votes.push({ n, weight: Math.hypot(walked.fx, walked.fy) });
    }
  }
  return votes;
}

// The TRUE [minPercent, maxPercent) band by magnitude rank, out of every vote.
function votesInMagnitudeBand(votes: Vote[], minPercent: number, maxPercent: number): Vote[] {
  const sorted = Array.from(votes).sort((a, b) => b.weight - a.weight);
  const lo = Math.round(sorted.length * (minPercent / 100));
  const hi = Math.round(sorted.length * (maxPercent / 100));
  if (hi <= lo) return [];
  return sorted.slice(lo, hi);
}

// Fits the degenerate quadric ("pair of planes through the origin") that
// best explains "every vote lies on one plane or the other" -- see
// pre-Stage-A history for the full derivation. `power` is the caller's
// current weightSharpenPower setting.
function fitPairOfPlanes(votes: Vote[], power: number): { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null {
  let maxW = 0;
  for (const { weight } of votes) if (weight > maxW) maxW = weight;
  const ATA: number[][] = Array.from({ length: 6 }, () => new Array(6).fill(0));
  for (const { n, weight } of votes) {
    const sharpened = maxW > 0 ? Math.pow(weight / maxW, power) : 0;
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
  if (Drow.lengthSq() < 1e-9 || Dcol.lengthSq() < 1e-9) return null;
  return { Drow: Drow.normalize(), Dcol: Dcol.normalize(), Dnormal };
}

// ── Orientation refinement (Levenberg-Marquardt) ─────────────────────────

function fourFoldResidual(n: THREE.Vector3, Drow: THREE.Vector3, Dcol: THREE.Vector3): number {
  const psi = Math.atan2(n.dot(Dcol), n.dot(Drow));
  return Math.sin(4 * psi);
}

function orientationCost(votes: Vote[], Drow: THREE.Vector3, Dcol: THREE.Vector3): number {
  let cost = 0;
  for (const { n, weight } of votes) {
    const r = weight * fourFoldResidual(n, Drow, Dcol);
    cost += r * r;
  }
  return cost;
}

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

function refineOrientationLM(votes: Vote[], initial: OrientationFit, maxIterations = 20): OrientationFit & { iterations: number; initialCost: number; finalCost: number } {
  const q = new THREE.Quaternion();
  const Drow0 = initial.Drow.clone(), Dcol0 = initial.Dcol.clone(), Dnormal0 = initial.Dnormal.clone();
  const candidateDrow = (qq: THREE.Quaternion) => Drow0.clone().applyQuaternion(qq);
  const candidateDcol = (qq: THREE.Quaternion) => Dcol0.clone().applyQuaternion(qq);

  const initialCost = orientationCost(votes, Drow0, Dcol0);
  let cost = initialCost;
  let lambda = 1e-3;
  const EPS = 1e-5;
  const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];

  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    const Drow = candidateDrow(q), Dcol = candidateDcol(q);
    const n = votes.length;
    const residuals = new Float64Array(n);
    for (let i = 0; i < n; i++) residuals[i] = votes[i].weight * fourFoldResidual(votes[i].n, Drow, Dcol);

    const J: Float64Array[] = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
    for (let k = 0; k < 3; k++) {
      const qPlus = new THREE.Quaternion().setFromAxisAngle(axes[k], EPS).multiply(q);
      const DrowP = candidateDrow(qPlus), DcolP = candidateDcol(qPlus);
      for (let i = 0; i < n; i++) {
        const rP = votes[i].weight * fourFoldResidual(votes[i].n, DrowP, DcolP);
        J[k][i] = (rP - residuals[i]) / EPS;
      }
    }

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
    if (!delta) break;

    const deltaVec = new THREE.Vector3(delta[0], delta[1], delta[2]);
    const deltaAngle = deltaVec.length();
    if (deltaAngle < 1e-10) break;
    const deltaAxis = deltaVec.normalize();
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

function torusBrightness(row: number, col: number): number {
  const r = ((row % R) + R) % R, c = ((col % C) + C) % C;
  return torus[r][c] ? 20 : 235;
}

function predictedBilinear(worldX: number, worldZ: number): number {
  const xf = worldX + C / 2 - 0.5, zf = worldZ + R / 2 - 0.5;
  const c0 = Math.floor(xf), r0 = Math.floor(zf);
  const fx = xf - c0, fz = zf - r0;
  const b00 = torusBrightness(r0, c0), b10 = torusBrightness(r0, c0 + 1);
  const b01 = torusBrightness(r0 + 1, c0), b11 = torusBrightness(r0 + 1, c0 + 1);
  return b00 * (1 - fx) * (1 - fz) + b10 * fx * (1 - fz) + b01 * (1 - fx) * fz + b11 * fx * fz;
}

function computePhotometricSamples(gray: Float64Array, w: number, h: number, stride: number): PhotometricSample[] {
  const samples: PhotometricSample[] = [];
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      samples.push({ px: x, py: y, observed: gray[y * w + x] });
    }
  }
  return samples;
}

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
// non-trivial autocorrelation peak -- see pre-Stage-A history for the full
// derivation (detrend, local-peak requirement, sub-bin parabolic refinement).
function autocorrelationPeriod(profile: Float64Array): number | null {
  const n = profile.length;

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
    if (scores[lag - minLag] >= peakThreshold) { bestLagPeak = lag; break; }
  }

  const bestLag = bestLagPeak > 0 ? bestLagPeak : bestLagAny;
  if (bestLag <= 0) return null;

  const i = bestLag - minLag;
  if (i > 0 && i < scores.length - 1) {
    const y0 = scores[i - 1], y1 = scores[i], y2 = scores[i + 1];
    const denom = y0 - 2 * y1 + y2;
    if (denom !== 0) {
      const delta = 0.5 * (y0 - y2) / denom;
      if (Math.abs(delta) < 1) return bestLag + delta;
    }
  }
  return bestLag;
}

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
      const cx = gradCxSum[bi] / c;
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

function marginalHueColor(cx: number, cy: number): string {
  let theta = Math.atan2(cy, cx) / 2;
  if (theta < 0) theta += Math.PI;
  if (theta >= Math.PI) theta -= Math.PI;
  const [r, g, b] = hsvToRgb((theta / Math.PI) * 360, 1, 1);
  return `rgb(${r},${g},${b})`;
}

// ── Grid rotation helpers (pure) ─────────────────────────────────────────

function rotatedDims(rows: number, cols: number, o: number): [number, number] {
  return (o === 1 || o === 3) ? [cols, rows] : [rows, cols];
}
function readRotated(grid: DecodeSampleGrid, o: number, a: number, b: number): DecodeSamplePoint {
  const { rows: gr, cols: gc, points } = grid;
  if (o === 1) return points[gr - 1 - b][a];
  if (o === 2) return points[gr - 1 - a][gc - 1 - b];
  if (o === 3) return points[b][gc - 1 - a];
  return points[a][b];
}
function rotateGrid(grid: DecodeSampleGrid, o: number): DecodeSampleGrid {
  if (o === 0) return grid;
  const [rr, cc] = rotatedDims(grid.rows, grid.cols, o);
  const points: DecodeSamplePoint[][] = Array.from({ length: rr }, (_, a) =>
    Array.from({ length: cc }, (_, b) => readRotated(grid, o, a, b)));
  let zeroI = 0, zeroJ = 0, bestD2 = Infinity;
  for (let a = 0; a < rr; a++) {
    for (let b = 0; b < cc; b++) {
      const pt = points[a][b];
      if (!pt.valid) continue;
      const d2 = pt.u * pt.u + pt.v * pt.v;
      if (d2 < bestD2) { bestD2 = d2; zeroI = a; zeroJ = b; }
    }
  }
  return { rows: rr, cols: cc, zeroI, zeroJ, points };
}

// Every valid order x order window, in EACH of the 4 whole-grid rotations,
// votes for a torus anchor -- see pre-Stage-A history for the full
// derivation. Pure function of the grid + the shared De Bruijn lookup.
function tallyPositionVotes(grid: DecodeSampleGrid): VoteResult | null {
  const tally = new Map<string, number>();
  let totalWindows = 0;
  const block: number[][] = Array.from({ length: ORDER }, () => new Array(ORDER).fill(0));
  for (let o = 0; o < 4; o++) {
    const [rr, cc] = rotatedDims(grid.rows, grid.cols, o);
    for (let i0 = 0; i0 + ORDER <= rr; i0++) {
      for (let j0 = 0; j0 + ORDER <= cc; j0++) {
        let complete = true;
        for (let di = 0; di < ORDER && complete; di++) {
          for (let dj = 0; dj < ORDER; dj++) {
            const pt = readRotated(grid, o, i0 + di, j0 + dj);
            if (!pt.valid) { complete = false; break; }
            block[di][dj] = pt.bit;
          }
        }
        if (!complete) continue;
        totalWindows++;
        let key = 0;
        for (let di = 0; di < ORDER; di++) for (let dj = 0; dj < ORDER; dj++) key = (key << 1) | block[di][dj];
        key = key >>> 0;
        const packed = debruijnLookup.get(key);
        if (packed === undefined) continue;
        const matchRow = Math.floor(packed / C), matchCol = packed % C;
        const anchorRow = ((matchRow - i0) % R + R) % R;
        const anchorCol = ((matchCol - j0) % C + C) % C;
        const voteKey = `${o},${anchorRow},${anchorCol}`;
        tally.set(voteKey, (tally.get(voteKey) ?? 0) + 1);
      }
    }
  }
  let best: VoteResult | null = null;
  for (const [key, votes] of tally) {
    if (best && votes <= best.votes) continue;
    const [o, ar, ac] = key.split(',').map(Number);
    best = { orientation: o, anchorRow: ar, anchorCol: ac, votes, totalWindows };
  }
  return best;
}

// Solves for the camera's ACTUAL world orientation, entirely from the
// pattern -- see pre-Stage-A history (solveRecoveredCamQuat's own comment)
// for the full derivation. Pure function of the (already math-frame)
// geometry plus the shared torus's true world layout.
function solveRecoveredCamQuat(
  rotated: DecodeSampleGrid, anchorRow: number, anchorCol: number,
  Drow: THREE.Vector3, Dcol: THREE.Vector3, normal: THREE.Vector3, distance: number,
): THREE.Quaternion | null {
  const mathPos = (i: number, j: number) => new THREE.Vector3()
    .addScaledVector(Drow, rotated.points[i][j].u)
    .addScaledVector(Dcol, rotated.points[i][j].v)
    .addScaledVector(normal, -distance);
  const worldPos = (i: number, j: number) => {
    const tRow = ((anchorRow + i) % R + R) % R, tCol = ((anchorCol + j) % C + C) % C;
    return new THREE.Vector3((tCol + 0.5 - C / 2) * GRID_STEP, 0, (tRow + 0.5 - R / 2) * GRID_STEP);
  };
  function findStep(i0: number, j0: number, di: number, dj: number): { i: number; j: number } | null {
    const maxSteps = di !== 0 ? rotated.rows : rotated.cols;
    for (let k = 1; k <= maxSteps; k++) {
      const i = i0 + di * k, j = j0 + dj * k;
      if (i < 0 || i >= rotated.rows || j < 0 || j >= rotated.cols) return null;
      if (rotated.points[i][j].valid) return { i, j };
    }
    return null;
  }

  const zi = rotated.zeroI, zj = rotated.zeroJ;
  const rowStep = findStep(zi, zj, 1, 0) ?? findStep(zi, zj, -1, 0);
  const colStep = findStep(zi, zj, 0, 1) ?? findStep(zi, zj, 0, -1);
  if (!rowStep || !colStep) return null;

  const originMath = mathPos(zi, zj), originWorld = worldPos(zi, zj);
  const rowMath = mathPos(rowStep.i, rowStep.j).sub(originMath).normalize();
  const rowWorld = worldPos(rowStep.i, rowStep.j).sub(originWorld).normalize();
  const colMath = mathPos(colStep.i, colStep.j).sub(originMath).normalize();
  const colWorld = worldPos(colStep.i, colStep.j).sub(originWorld).normalize();

  const thirdMath = new THREE.Vector3().crossVectors(rowMath, colMath).normalize();
  const thirdWorld = new THREE.Vector3().crossVectors(rowWorld, colWorld).normalize();
  if (thirdMath.lengthSq() < 1e-9 || thirdWorld.lengthSq() < 1e-9) return null;

  const mathBasis = new THREE.Matrix4().makeBasis(rowMath, colMath, thirdMath);
  const worldBasis = new THREE.Matrix4().makeBasis(rowWorld, colWorld, thirdWorld);
  const mathBasisInv = mathBasis.clone().invert();
  return new THREE.Quaternion().setFromRotationMatrix(worldBasis.clone().multiply(mathBasisInv));
}

// ── Contamination overlay math (pure) ────────────────────────────────────

function computeContaminationAlpha(
  field: GradientField, agreement: Float64Array,
  dirA: THREE.Vector3, dirB: THREE.Vector3,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Float64Array {
  const { fx, fy, w, h, r } = field;
  const alpha = new Float64Array(w * h);
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
    out[o + 3] = Math.min(255, Math.round(alpha[i] * 255));
  }
}

const TRUE_CONTAM_COLOR = [230, 40, 40] as const;
const RECON_CONTAM_COLOR = [235, 150, 20] as const;

// ── Per-camera capture/analysis pipeline ─────────────────────────────────

// Central place for "what FOV should ray-casting assume" -- real-capture
// mode uses the direct FOV input (settings.realCaptureFovDeg) instead of the
// simulated camera's focalMM-derived one. realCaptureFovDeg is HORIZONTAL;
// this function always returns VERTICAL (THREE.js's camera.fov convention),
// via the camera's own current aspect ratio.
function getAnalysisVFovRad(camera: Camera): number {
  if (isPhysical(camera)) {
    const hFovRad = THREE.MathUtils.degToRad(camera.settings.realCaptureFovDeg);
    return 2 * Math.atan(Math.tan(hFovRad / 2) / camera.aspect);
  }
  return THREE.MathUtils.degToRad(camera.gizmoCam.fov);
}

function markCaptureDirty(camera: Camera) {
  camera.captureDirty = true;
}

// Called once at camera creation and again whenever the viewportW/H/
// captureSupersample sliders change -- or, with an explicit override,
// whenever a real capture arrives at a different resolution than whatever's
// currently allocated (see ingestRealCapture).
function resizeCaptureBuffers(camera: Camera, explicitSize?: { w: number; h: number }) {
  camera.captureDirty = true;
  camera.rtSize = explicitSize ?? { w: Math.round(camera.settings.viewportW), h: Math.round(camera.settings.viewportH) };
  camera.aspect = camera.rtSize.w / camera.rtSize.h;
  const { w, h } = camera.rtSize;

  if (isSimulated(camera)) {
    camera.captureRTSize = { w: w * camera.settings.captureSupersample, h: h * camera.settings.captureSupersample };
    camera.camRT.setSize(camera.captureRTSize.w, camera.captureRTSize.h);
    camera.gizmoCam.aspect = camera.aspect;
    camera.gizmoCam.updateProjectionMatrix();
  }

  camera.distortedPreviewData = new Uint8Array(w * h * 4);
  camera.distortedPreviewTex.image = { data: camera.distortedPreviewData, width: w, height: h };
  // WebGL2 typically allocates a texture's GPU storage immutably on first
  // upload -- dispose() forces three.js to drop the old GL texture object so
  // the next upload allocates fresh storage at the new size.
  camera.distortedPreviewTex.dispose();
  camera.distortedPreviewTex.needsUpdate = true;

  camera.projectedPreviewData = new Uint8Array(w * h * 4);
  camera.projectedPreviewTex.image = { data: camera.projectedPreviewData, width: w, height: h };
  camera.projectedPreviewTex.dispose();
  camera.projectedPreviewTex.needsUpdate = true;

  camera.trueContamData = new Uint8Array(w * h * 4);
  camera.trueContamTex.image = { data: camera.trueContamData, width: w, height: h };
  camera.trueContamTex.dispose();
  camera.trueContamTex.needsUpdate = true;

  camera.reconContamData = new Uint8Array(w * h * 4);
  camera.reconContamTex.image = { data: camera.reconContamData, width: w, height: h };
  camera.reconContamTex.dispose();
  camera.reconContamTex.needsUpdate = true;

  if (camera === activeCamera()) layoutPip(camera);
}

// Renders gizmoCam's view into camRT -- pulled out into its own function so
// the real analysis path (captureDistortedGrayscale) can always force a
// truly fresh capture regardless of the passive preview's dirty/throttle
// gating in animate().
function renderCamRT(camera: SimulatedCamera) {
  const dpr = renderer.getPixelRatio();
  const prevRT = renderer.getRenderTarget();
  renderer.setRenderTarget(camera.camRT);
  renderer.setViewport(0, 0, camera.captureRTSize.w / dpr, camera.captureRTSize.h / dpr);
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.render(scene, camera.gizmoCam);
  renderer.setRenderTarget(prevRT);
}

// Render+blur happen at captureSupersample x rtSize, THEN get box-downsampled
// to rtSize -- see pre-Stage-A history for why (physical lens blur acts on a
// near-continuous image; only the sensor's final discretization should
// introduce the pixel grid). Returned in GL's native bottom-up row order.
function captureDistortedGrayscale(camera: SimulatedCamera): { gray: Float64Array; w: number; h: number } {
  renderCamRT(camera);
  const { w: cw, h: ch } = camera.captureRTSize;
  const raw = new Uint8Array(cw * ch * 4);
  renderer.readRenderTargetPixels(camera.camRT, 0, 0, cw, ch, raw);
  const hiResGray = toGrayscale(raw, cw, ch);
  const antialiased = applyAntialiasFilter(hiResGray, cw, ch, camera.settings.captureSupersample);
  const hiResBlurred = separableBoxBlur(antialiased, cw, ch, Math.round(camera.settings.simBlur * camera.settings.captureSupersample));
  const gray = downsampleBoxAverage(hiResBlurred, cw, ch, camera.settings.captureSupersample, camera.rtSize.w, camera.rtSize.h);
  addGaussianNoise(gray, camera.settings.simNoise);
  return { gray, w: camera.rtSize.w, h: camera.rtSize.h };
}

// Decodes an incoming data URL, resamples it to the current analysis
// resolution, converts to grayscale, and flips it to bottom-up.
async function ingestRealCapture(camera: PhysicalCamera, dataUrl: string): Promise<void> {
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error('failed to decode incoming capture image'));
    img.src = dataUrl;
  });
  const w = img.naturalWidth, h = img.naturalHeight;
  if (w !== camera.rtSize.w || h !== camera.rtSize.h) resizeCaptureBuffers(camera, { w, h });

  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = w; tmpCanvas.height = h;
  const tctx = tmpCanvas.getContext('2d')!;
  tctx.drawImage(img, 0, 0);
  const topDown = tctx.getImageData(0, 0, w, h).data;
  const grayTopDown = toGrayscale(topDown, w, h);
  camera.lastRealCaptureGray = flipRowsF64(grayTopDown, w, h);
  camera.lastRealCaptureW = w; camera.lastRealCaptureH = h;

  updateDistortedPreview(camera);
  if (globalState.mode === 'projected' && camera === activeCamera()) buildProjectedTexture(camera);
  runAxesReconstruction(camera);
}

// Shared tail for both capture sources: given a final analysis-resolution
// grayscale, paints whichever of the 4 direction/scalar field views is
// currently selected.
function paintFieldViewFromGray(camera: Camera, gray: Float64Array) {
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const settings = camera.settings;
  if (settings.fieldView === 'gradient') {
    const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
    camera.lastDisplayedVectorField = field;
    paintVectorFieldAsColor(field, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  } else if (settings.fieldView === 'walked') {
    const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
    const agreement = computeGradientAgreementField(field, Math.round(settings.coherenceRadius));
    const effective = computeEffectiveGradientField(field, agreement);
    camera.lastEffectiveField = effective;
    const walked = computeWalkedGradientField(settings, effective);
    camera.lastDisplayedVectorField = walked;
    paintVectorFieldAsColor(walked, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  } else if (settings.fieldView === 'agreement') {
    const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
    const agreement = computeGradientAgreementField(field, Math.round(settings.coherenceRadius));
    paintScalarFieldAsGray(agreement, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  } else if (settings.fieldView === 'effective') {
    const field = computeGradientField(gray, w, h, Math.round(settings.simGradRadius));
    const agreement = computeGradientAgreementField(field, Math.round(settings.coherenceRadius));
    const effective = computeEffectiveGradientField(field, agreement);
    camera.lastEffectiveField = effective;
    camera.lastDisplayedVectorField = effective;
    paintVectorFieldAsColor(effective, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
  }
}

function updateDistortedPreview(camera: Camera) {
  camera.lastDisplayedVectorField = null;
  camera.lastEffectiveField = null;
  const settings = camera.settings;
  if (settings.hideField) {
    for (let i = 0; i < camera.distortedPreviewData.length; i += 4) {
      camera.distortedPreviewData[i] = 0; camera.distortedPreviewData[i + 1] = 0; camera.distortedPreviewData[i + 2] = 0; camera.distortedPreviewData[i + 3] = 255;
    }
    camera.distortedPreviewTex.needsUpdate = true;
  }
  const needGrayForOverlay = settings.showTrueContamination || settings.showReconstructedContamination;
  if (settings.hideField && !needGrayForOverlay) return;

  if (isPhysical(camera)) {
    if (!camera.lastRealCaptureGray) return;
    if (!settings.hideField) {
      if (settings.fieldView === 'raw' || settings.fieldView === 'antialiased' || settings.fieldView === 'downsampled' || settings.fieldView === 'noised') {
        fillGrayscalePreview(camera.lastRealCaptureGray, camera.distortedPreviewData);
        camera.distortedPreviewTex.needsUpdate = true;
      } else {
        paintFieldViewFromGray(camera, camera.lastRealCaptureGray);
      }
    }
    camera.lastNoisedPreviewGray = camera.lastRealCaptureGray;
    return;
  }

  // camera narrowed to SimulatedCamera by the isPhysical() early-return
  // above, but `settings` was captured before that -- re-derive it so
  // TypeScript (and the simNoise/simBlur/captureSupersample accesses below)
  // see the narrower SimulatedCameraSettings type.
  const simSettings = camera.settings;
  const { w: cw, h: ch } = camera.captureRTSize;
  const rawRGBA = new Uint8Array(cw * ch * 4);
  renderer.readRenderTargetPixels(camera.camRT, 0, 0, cw, ch, rawRGBA);
  const hiResGray = toGrayscale(rawRGBA, cw, ch);

  if (!settings.hideField && settings.fieldView === 'raw') {
    const raw = downsampleBoxAverage(hiResGray, cw, ch, simSettings.captureSupersample, camera.rtSize.w, camera.rtSize.h);
    fillGrayscalePreview(raw, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
  }

  const antialiased = applyAntialiasFilter(hiResGray, cw, ch, simSettings.captureSupersample);

  if (!settings.hideField && settings.fieldView === 'antialiased') {
    const aaDisplay = downsampleBoxAverage(antialiased, cw, ch, simSettings.captureSupersample, camera.rtSize.w, camera.rtSize.h);
    fillGrayscalePreview(aaDisplay, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
  }

  const hiResBlurred = separableBoxBlur(antialiased, cw, ch, Math.round(simSettings.simBlur * simSettings.captureSupersample));
  const downsampled = downsampleBoxAverage(hiResBlurred, cw, ch, simSettings.captureSupersample, camera.rtSize.w, camera.rtSize.h);

  if (!settings.hideField && settings.fieldView === 'downsampled') {
    fillGrayscalePreview(downsampled, camera.distortedPreviewData);
    camera.distortedPreviewTex.needsUpdate = true;
    if (!needGrayForOverlay) return;
  }

  const noised = downsampled;
  addGaussianNoise(noised, simSettings.simNoise);
  if (!settings.hideField) {
    if (settings.fieldView === 'noised') {
      fillGrayscalePreview(noised, camera.distortedPreviewData);
      camera.distortedPreviewTex.needsUpdate = true;
    } else {
      paintFieldViewFromGray(camera, noised);
    }
  }
  camera.lastNoisedPreviewGray = noised;
}

const PREVIEW_UPDATE_INTERVAL_MS = 100; // ~10fps

// Recomputes whichever contamination overlay(s) are actually toggled on.
function updateContaminationOverlays(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showTrueContamination && !settings.showReconstructedContamination) return;
  if (settings.fieldView !== 'gradient' && settings.fieldView !== 'effective' && settings.fieldView !== 'walked') return;
  if (!camera.lastNoisedPreviewGray) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const lum = camera.lastNoisedPreviewGray;
  const vFovRad = getAnalysisVFovRad(camera);
  const rawField = computeGradientField(lum, w, h, Math.round(settings.simGradRadius));
  const agreement = computeGradientAgreementField(rawField, Math.round(settings.coherenceRadius));
  const field = settings.fieldView === 'gradient' ? rawField
    : settings.fieldView === 'effective' ? computeEffectiveGradientField(rawField, agreement)
    : computeWalkedGradientField(settings, computeEffectiveGradientField(rawField, agreement));

  if (settings.showTrueContamination && isSimulated(camera)) {
    const alpha = computeContaminationAlpha(field, agreement, ROW_DIR, COL_DIR, camera.camQuat, vFovRad, camera.aspect);
    paintContaminationOverlay(alpha, TRUE_CONTAM_COLOR, camera.trueContamData);
    camera.trueContamTex.needsUpdate = true;
  } else if (settings.showTrueContamination) {
    // No ground truth for a physical camera -- nothing to compare against.
    camera.trueContamData.fill(0);
    camera.trueContamTex.needsUpdate = true;
  }
  if (settings.showReconstructedContamination) {
    if (camera.lastRecoveredAxes) {
      const alpha = computeContaminationAlpha(field, agreement, camera.lastRecoveredAxes.Drow, camera.lastRecoveredAxes.Dcol, MATH_QUAT, vFovRad, camera.aspect);
      paintContaminationOverlay(alpha, RECON_CONTAM_COLOR, camera.reconContamData);
      toggleReconContamBtn.textContent = 'reconstructed contamination overlay (orange)';
    } else {
      camera.reconContamData.fill(0);
      toggleReconContamBtn.textContent = 'reconstructed contamination overlay (orange) — run "capture now" first';
    }
    camera.reconContamTex.needsUpdate = true;
  }
}

const DEBUG_CIRCLE_SEGMENTS = 48;
const AXIS_VECTOR_LENGTH = 0.7;

function updateGradientCirclesDebug(camera: Camera) {
  const chosen = votesInMagnitudeBand(camera.lastVotes, camera.settings.circleSamplePercentMin, camera.settings.circleSamplePercentMax);
  if (chosen.length === 0) {
    camera.gradientCirclesGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    camera.axisVectorsGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    return;
  }

  let minW = Infinity, maxW = -Infinity;
  for (const vote of chosen) {
    if (vote.weight < minW) minW = vote.weight;
    if (vote.weight > maxW) maxW = vote.weight;
  }
  const wRange = maxW - minW;

  const positions = new Float32Array(chosen.length * DEBUG_CIRCLE_SEGMENTS * 2 * 3);
  const colors = new Float32Array(chosen.length * DEBUG_CIRCLE_SEGMENTS * 2 * 3);
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
    const len = maxW > 0 ? AXIS_VECTOR_LENGTH * Math.pow(vote.weight / maxW, camera.settings.weightSharpenPower) : 0;
    axisPositions[ap++] = 0; axisPositions[ap++] = 0; axisPositions[ap++] = 0;
    axisPositions[ap++] = normal.x * len;
    axisPositions[ap++] = normal.y * len;
    axisPositions[ap++] = normal.z * len;
    axisColors[apc++] = r; axisColors[apc++] = 0; axisColors[apc++] = b;
    axisColors[apc++] = r; axisColors[apc++] = 0; axisColors[apc++] = b;
  }
  camera.gradientCirclesGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  camera.gradientCirclesGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  camera.axisVectorsGeo.setAttribute('position', new THREE.Float32BufferAttribute(axisPositions, 3));
  camera.axisVectorsGeo.setAttribute('color', new THREE.Float32BufferAttribute(axisColors, 3));
  camera.axisVectorsGeo.computeBoundingSphere();
  camera.gradientCirclesGeo.computeBoundingSphere();
}

// Casts one ray per SCREEN pixel and bins the hits into a bucketW x bucketH
// grid -- see pre-Stage-A history for the full derivation (grazing-angle
// cutoff, gradient-covector re-expression in the (u,v) frame, the U-mirror
// that cancels a handedness mismatch).
function castAndBucketProjectedSamples(camera: Camera, bucketW: number, bucketH: number): {
  bins: ProjectedBins; sums: Float64Array; counts: Float64Array; gradCxSum: Float64Array; gradCySum: Float64Array;
} | null {
  if (!camera.lastRecoveredAxes) return null;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const vFovRad = getAnalysisVFovRad(camera);
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, (py / h) * 2 - 1];

  const MIN_GRAZING_COS = 0.15;
  const hit = new THREE.Vector3();
  const hit2 = new THREE.Vector3();
  const us: number[] = [], vs: number[] = [], srcIdx: number[] = [];
  const gradCxAtSample: number[] = [], gradCyAtSample: number[] = [];
  const srcGrad = camera.lastNoisedPreviewGray ? computeGradientField(camera.lastNoisedPreviewGray, w, h, 1) : null;
  let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [ndcU, ndcV] = toNDC(x, y);
      const rayDir = cornerDir(ndcU, ndcV, MATH_QUAT, vFovRad, camera.aspect);
      const denom = rayDir.dot(normal);
      if (denom >= -MIN_GRAZING_COS) continue;
      const t = -distance / denom;
      hit.copy(rayDir).multiplyScalar(t);
      const u = hit.dot(Drow), v = hit.dot(Dcol);
      us.push(u); vs.push(v); srcIdx.push(y * w + x);
      if (u < minU) minU = u; if (u > maxU) maxU = u;
      if (v < minV) minV = v; if (v > maxV) maxV = v;

      let cxAtSample = 0, cyAtSample = 0;
      if (srcGrad) {
        const si = y * w + x;
        const fx = srcGrad.fx[si], fy = srcGrad.fy[si];
        const mag = Math.hypot(fx, fy);
        if (mag > 0) {
          const theta = Math.atan2(fy, fx);
          const tdx = -Math.sin(theta), tdy = Math.cos(theta);
          const [ndcU2, ndcV2] = toNDC(x + tdx, y + tdy);
          const rayDir2 = cornerDir(ndcU2, ndcV2, MATH_QUAT, vFovRad, camera.aspect);
          const denom2 = rayDir2.dot(normal);
          if (denom2 < -MIN_GRAZING_COS) {
            const t2 = -distance / denom2;
            hit2.copy(rayDir2).multiplyScalar(t2);
            const u2 = hit2.dot(Drow), v2 = hit2.dot(Dcol);
            const du = u2 - u, dv = v2 - v;
            if (Math.hypot(du, dv) > 1e-9) {
              const phiUV = Math.atan2(dv, du);
              cxAtSample = -mag * Math.cos(2 * phiUV);
              cyAtSample = -mag * Math.sin(2 * phiUV);
            }
          }
        }
      }
      gradCxAtSample.push(cxAtSample); gradCyAtSample.push(cyAtSample);
    }
  }
  if (!isFinite(minU) || !isFinite(minV)) return null;

  const binWidthU = (maxU - minU) / bucketW || 1;
  const binWidthV = (maxV - minV) / bucketH || 1;
  const bins: ProjectedBins = { minU, maxU, minV, maxV, binWidthU, binWidthV, w: bucketW, h: bucketH };
  const sums = new Float64Array(bucketW * bucketH * 3);
  const counts = new Float64Array(bucketW * bucketH);
  const gradCxSum = new Float64Array(bucketW * bucketH);
  const gradCySum = new Float64Array(bucketW * bucketH);
  for (let k = 0; k < us.length; k++) {
    const bu = Math.min(bucketW - 1, Math.max(0, Math.floor((maxU - us[k]) / binWidthU)));
    const bv = Math.min(bucketH - 1, Math.max(0, Math.floor((vs[k] - minV) / binWidthV)));
    const bi = bv * bucketW + bu;
    const si = srcIdx[k];
    const srcO = si * 4;
    sums[bi * 3] += camera.distortedPreviewData[srcO];
    sums[bi * 3 + 1] += camera.distortedPreviewData[srcO + 1];
    sums[bi * 3 + 2] += camera.distortedPreviewData[srcO + 2];
    counts[bi]++;
    gradCxSum[bi] += gradCxAtSample[k];
    gradCySum[bi] += gradCyAtSample[k];
  }
  return { bins, sums, counts, gradCxSum, gradCySum };
}

// Rebuilds projectedPreviewData: a bird's-eye, floor-plane-rectified view of
// whichever field view is currently in distortedPreviewData.
function buildProjectedTexture(camera: Camera) {
  const result = camera.lastRecoveredAxes ? castAndBucketProjectedSamples(camera, camera.rtSize.w, camera.rtSize.h) : null;
  if (!result) { camera.projectedPreviewData.fill(0); camera.projectedPreviewTex.needsUpdate = true; camera.lastProjectedBins = null; camera.lastMarginals = null; return; }
  const { bins, sums, counts, gradCxSum, gradCySum } = result;
  camera.lastProjectedBins = bins;
  for (let bi = 0; bi < bins.w * bins.h; bi++) {
    const c = counts[bi];
    const o = bi * 4;
    if (c > 0) {
      camera.projectedPreviewData[o] = Math.round(sums[bi * 3] / c);
      camera.projectedPreviewData[o + 1] = Math.round(sums[bi * 3 + 1] / c);
      camera.projectedPreviewData[o + 2] = Math.round(sums[bi * 3 + 2] / c);
      camera.projectedPreviewData[o + 3] = 255;
    } else {
      camera.projectedPreviewData[o] = 0; camera.projectedPreviewData[o + 1] = 0; camera.projectedPreviewData[o + 2] = 0; camera.projectedPreviewData[o + 3] = 255;
    }
  }
  camera.projectedPreviewTex.needsUpdate = true;
  camera.lastMarginals = computeProjectedMarginals(bins.w, bins.h, counts, gradCxSum, gradCySum);
}

// Re-buckets castAndBucketProjectedSamples' rays at a resolution sized to
// keep a fixed target of buckets per grid cell -- see pre-Stage-A history.
function measurePeriodDistance(camera: Camera, currentDistance: number, extentU: number, extentV: number): { distanceU: number; distanceV: number } | null {
  const TARGET_BUCKETS_PER_CELL = 20;
  const MAX_REFINE_BUCKETS = 2048;
  const refineW = Math.min(MAX_REFINE_BUCKETS, Math.max(camera.rtSize.w, Math.ceil(extentU / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const refineH = Math.min(MAX_REFINE_BUCKETS, Math.max(camera.rtSize.h, Math.ceil(extentV / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const refined = castAndBucketProjectedSamples(camera, refineW, refineH);
  const refinedMarginals = refined ? computeProjectedMarginals(refineW, refineH, refined.counts, refined.gradCxSum, refined.gradCySum) : null;
  if (!refined || !refinedMarginals || refinedMarginals.colPeriod === null || refinedMarginals.rowPeriod === null) return null;
  return {
    distanceU: currentDistance * (GRID_STEP / (refinedMarginals.colPeriod * refined.bins.binWidthU)),
    distanceV: currentDistance * (GRID_STEP / (refinedMarginals.rowPeriod * refined.bins.binWidthV)),
  };
}

// Own, axis-symmetric-bucket bins/marginals -- deliberately NOT
// lastProjectedBins/lastMarginals (the display pipeline's own state) -- see
// pre-Stage-A history for why sharing that state caused a real bug.
function computeDecodeMarginals(camera: Camera): { bins: ProjectedBins; marginals: Marginals } | null {
  if (!camera.lastRecoveredAxes || !camera.lastProjectedBins) return null;
  const TARGET_BUCKETS_PER_CELL = 20;
  const MAX_REFINE_BUCKETS = 2048;
  const floor = Math.max(camera.rtSize.w, camera.rtSize.h);
  const extentU = camera.lastProjectedBins.maxU - camera.lastProjectedBins.minU;
  const extentV = camera.lastProjectedBins.maxV - camera.lastProjectedBins.minV;
  const bucketW = Math.min(MAX_REFINE_BUCKETS, Math.max(floor, Math.ceil(extentU / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const bucketH = Math.min(MAX_REFINE_BUCKETS, Math.max(floor, Math.ceil(extentV / GRID_STEP * TARGET_BUCKETS_PER_CELL)));
  const result = castAndBucketProjectedSamples(camera, bucketW, bucketH);
  if (!result) return null;
  const marginals = computeProjectedMarginals(bucketW, bucketH, result.counts, result.gradCxSum, result.gradCySum);
  return { bins: result.bins, marginals };
}

// Builds a sampling grid covering the FULL observed quadrilateral -- see
// pre-Stage-A history for the full derivation.
function buildDecodeSampleGrid(camera: Camera, gray: Float64Array, w: number, h: number, vFovRad: number): DecodeSampleGrid | null {
  if (!camera.lastRecoveredAxes) return null;
  const decodeMarginals = computeDecodeMarginals(camera);
  if (!decodeMarginals || decodeMarginals.marginals.colPeriod === null || decodeMarginals.marginals.rowPeriod === null) {
    return null;
  }
  const { bins, marginals } = decodeMarginals;
  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const invQuat = MATH_QUAT.clone().invert();
  const halfV = vFovRad / 2;
  const bin = binarize(gray);

  const uBoundaryRaw = bins.maxU - marginals.colPhase * bins.binWidthU;
  const vBoundaryRaw = bins.minV + marginals.rowPhase * bins.binWidthV;
  const uPhase = (uBoundaryRaw - Math.round(uBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;
  const vPhase = (vBoundaryRaw - Math.round(vBoundaryRaw / GRID_STEP) * GRID_STEP) + GRID_STEP / 2;

  const { minU, maxU, minV, maxV } = bins;
  const kMinU = Math.floor((minU - uPhase) / GRID_STEP), kMaxU = Math.ceil((maxU - uPhase) / GRID_STEP);
  const kMinV = Math.floor((minV - vPhase) / GRID_STEP), kMaxV = Math.ceil((maxV - vPhase) / GRID_STEP);
  const cols = kMaxU - kMinU + 1, rows = kMaxV - kMinV + 1;
  const zeroI = Math.min(rows - 1, Math.max(0, Math.round(-vPhase / GRID_STEP) - kMinV));
  const zeroJ = Math.min(cols - 1, Math.max(0, Math.round(-uPhase / GRID_STEP) - kMinU));

  const p = new THREE.Vector3();
  const local = new THREE.Vector3();
  const points: DecodeSamplePoint[][] = [];
  for (let i = 0; i < rows; i++) {
    const v = vPhase + (kMinV + i) * GRID_STEP;
    const rowPoints: DecodeSamplePoint[] = [];
    for (let j = 0; j < cols; j++) {
      const u = uPhase + (kMinU + j) * GRID_STEP;
      p.copy(Drow).multiplyScalar(u).addScaledVector(Dcol, v).addScaledVector(normal, -distance);
      local.copy(p).applyQuaternion(invQuat);
      const ndcU = -local.x / (local.z * Math.tan(halfV) * camera.aspect);
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

// Decodes the camera's absolute world position -- see pre-Stage-A history
// for the full derivation.
function runPositionDecode(camera: Camera, gray: Float64Array, w: number, h: number, vFovRad: number) {
  const grid = buildDecodeSampleGrid(camera, gray, w, h, vFovRad);
  camera.lastDecodeGrid = grid;
  camera.lastDecodeRotated = null;
  if (!grid) { camera.lastPositionDecode = null; camera.lastDecodeCorrectness = null; return; }
  const winner = tallyPositionVotes(grid);
  if (!winner) { camera.lastPositionDecode = null; camera.lastDecodeCorrectness = null; return; }

  const { anchorRow, anchorCol } = winner;
  const rotated = rotateGrid(grid, winner.orientation);
  camera.lastDecodeRotated = rotated;
  const correctness: (DecodeCellDebug | null)[][] = Array.from({ length: rotated.rows }, () => new Array(rotated.cols).fill(null));
  let correctCount = 0, wrongCount = 0;
  for (let i = 0; i < rotated.rows; i++) {
    for (let j = 0; j < rotated.cols; j++) {
      const pt = rotated.points[i][j];
      if (!pt.valid) continue;
      const torusRow = ((anchorRow + i) % R + R) % R;
      const torusCol = ((anchorCol + j) % C + C) % C;
      const correct = pt.bit === torus[torusRow][torusCol];
      correctness[i][j] = { bit: pt.bit, correct };
      correct ? correctCount++ : wrongCount++;
    }
  }
  camera.lastDecodeCorrectness = correctness;
  const consistency = correctCount + wrongCount > 0 ? correctCount / (correctCount + wrongCount) : 0;

  const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes!; // buildDecodeSampleGrid returning non-null guarantees this
  const normal = Dnormal.clone();
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normal) > 0) normal.negate();
  const refTorusRow = ((anchorRow + rotated.zeroI) % R + R) % R;
  const refTorusCol = ((anchorCol + rotated.zeroJ) % C + C) % C;

  const recoveredCamQuat = solveRecoveredCamQuat(rotated, anchorRow, anchorCol, Drow, Dcol, normal, distance);
  if (!recoveredCamQuat) { camera.lastPositionDecode = null; return; }

  const DrowWorld = Drow.clone().applyQuaternion(recoveredCamQuat);
  const DcolWorld = Dcol.clone().applyQuaternion(recoveredCamQuat);
  const normalWorld = normal.clone().applyQuaternion(recoveredCamQuat);
  const refPt = rotated.points[rotated.zeroI][rotated.zeroJ];
  const hitRelWorld = new THREE.Vector3()
    .addScaledVector(DrowWorld, refPt.u).addScaledVector(DcolWorld, refPt.v).addScaledVector(normalWorld, -distance);

  const worldPosTrue = new THREE.Vector3((refTorusCol + 0.5 - C / 2) * GRID_STEP, 0, (refTorusRow + 0.5 - R / 2) * GRID_STEP);
  camera.lastPositionDecode = {
    row: refTorusRow, col: refTorusCol, consistency, votes: winner.votes, totalWindows: winner.totalWindows,
    camPos: worldPosTrue.sub(hitRelWorld),
    recoveredCamQuat,
  };
}

// ── World-frame constants ────────────────────────────────────────────────

const ROW_DIR = new THREE.Vector3(1, 0, 0); // world +X — direction shared by every "row" floor line
const COL_DIR = new THREE.Vector3(0, 0, 1); // world +Z — direction shared by every "col" floor line
// Scratch, reused sequentially by updateGizmo (each call fully overwrites it
// before use, so sharing across cameras/frames is safe).
const euler = new THREE.Euler(0, 0, 0, 'YXZ');

// Fixed identity, NEVER mutated -- the reference frame every ray-casting call
// inside the recovery pipeline is expressed in. It does NOT need to equal
// the camera's true orientation for any of that math to work -- see
// pre-Stage-A history (solveRecoveredCamQuat's own comment) for how the
// camera's actual world orientation gets recovered afterward, entirely from
// the pattern, with zero dependency on this being "correct". A simulated
// camera's own camQuat (ground truth) must never leak into the recovery
// math itself.
const MATH_QUAT = new THREE.Quaternion();

// ── Axes/position reconstruction (the big orchestrator) ──────────────────

function runAxesReconstruction(camera: Camera) {
  if (camera.axesCapturing) return; // don't stack overlapping captures
  camera.axesCapturing = true;
  const isActive = camera === activeCamera();
  const prevLabel = captureAxesBtn.textContent;
  if (isActive) {
    captureAxesBtn.disabled = true;
    captureAxesBtn.textContent = '⏳ computing...';
    axesReadout.textContent = 'computing...';
  }
  requestAnimationFrame(() => {
    try {
      const t0 = performance.now();
      if (isPhysical(camera) && !camera.lastRealCaptureGray) {
        if (isActive) axesReadout.textContent = 'waiting for a real capture -- take a photo on the phone page';
        return;
      }
      const { gray: rawGray, w, h } = isPhysical(camera)
        ? { gray: camera.lastRealCaptureGray!, w: camera.lastRealCaptureW, h: camera.lastRealCaptureH }
        : captureDistortedGrayscale(camera);
      camera.lastNoisedPreviewGray = rawGray;
      const gray = flipRowsF64(rawGray, w, h);
      const vFovRad = getAnalysisVFovRad(camera);
      const votes = computeWorldVotes(camera.settings, gray, w, h, camera.settings.simGradRadius, camera.settings.coherenceRadius, MATH_QUAT, vFovRad, camera.aspect);
      camera.lastVotes = votes;
      updateGradientCirclesDebug(camera);
      const t1 = performance.now();

      const fitVotes = votesInMagnitudeBand(votes, camera.settings.circleSamplePercentMin, camera.settings.circleSamplePercentMax);
      const quadricPair = fitPairOfPlanes(fitVotes, camera.settings.weightSharpenPower);
      const t2 = performance.now();

      const refinedFit = quadricPair && camera.settings.orientationLM ? refineOrientationLM(fitVotes, quadricPair) : null;
      const orientationFit = refinedFit ?? quadricPair;
      const t2b = performance.now();

      camera.axesComputed = !!quadricPair;

      let rowDirRecovered: THREE.Vector3 | null = null, colDirRecovered: THREE.Vector3 | null = null;
      if (orientationFit) {
        const normalForHandedness = orientationFit.Dnormal.clone();
        if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalForHandedness) > 0) normalForHandedness.negate();
        rowDirRecovered = orientationFit.Drow.clone();
        colDirRecovered = orientationFit.Dcol.clone();
        const handedness = rowDirRecovered.clone().cross(colDirRecovered).dot(normalForHandedness);
        if (handedness > 0) colDirRecovered.negate();
      }

      const PLACEHOLDER_DISTANCE = 1;
      camera.lastRecoveredAxes = rowDirRecovered && colDirRecovered && orientationFit
        ? { Drow: rowDirRecovered, Dcol: colDirRecovered, Dnormal: orientationFit.Dnormal, distance: PLACEHOLDER_DISTANCE }
        : null;
      if (camera.lastRecoveredAxes) buildProjectedTexture(camera);

      const marginals = camera.lastMarginals, bins = camera.lastProjectedBins;
      const spacing = camera.lastRecoveredAxes && marginals && bins && marginals.colPeriod !== null && marginals.rowPeriod !== null
        ? {
          distanceU: PLACEHOLDER_DISTANCE * (GRID_STEP / (marginals.colPeriod * bins.binWidthU)),
          distanceV: PLACEHOLDER_DISTANCE * (GRID_STEP / (marginals.rowPeriod * bins.binWidthV)),
        }
        : null;
      const t3 = performance.now();

      let refinedSpacing: { distanceU: number; distanceV: number } | null = null;
      let finalSpacing: { distanceU: number; distanceV: number } | null = null;
      if (camera.lastRecoveredAxes && spacing) {
        camera.lastRecoveredAxes.distance = (spacing.distanceU + spacing.distanceV) / 2;

        const roughBins = camera.lastProjectedBins;
        if (roughBins) {
          const rescale = camera.lastRecoveredAxes.distance / PLACEHOLDER_DISTANCE;
          const trueExtentU = (roughBins.maxU - roughBins.minU) * rescale;
          const trueExtentV = (roughBins.maxV - roughBins.minV) * rescale;
          const roughDistance = camera.lastRecoveredAxes.distance;
          const measured = measurePeriodDistance(camera, roughDistance, trueExtentU, trueExtentV);
          if (measured) {
            refinedSpacing = measured;
            camera.lastRecoveredAxes.distance = (measured.distanceU + measured.distanceV) / 2;
          }
        }

        buildProjectedTexture(camera);
      } else {
        camera.lastRecoveredAxes = null;
      }
      const t3b = performance.now();
      runPositionDecode(camera, gray, w, h, vFovRad);

      let lastPositionLMResult: (PositionFit & { iterations: number; initialCost: number; finalCost: number }) | null = null;
      if (camera.settings.positionLM && camera.lastRecoveredAxes && camera.lastPositionDecode && camera.lastNoisedPreviewGray) {
        const { Drow, Dcol, Dnormal, distance } = camera.lastRecoveredAxes;
        const normalForInit = Dnormal.clone();
        if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalForInit) > 0) normalForInit.negate();
        const normalForInitWorld = normalForInit.clone().applyQuaternion(camera.lastPositionDecode.recoveredCamQuat);
        const initialWorldX0 = camera.lastPositionDecode.camPos.x + normalForInitWorld.x * -distance;
        const initialWorldZ0 = camera.lastPositionDecode.camPos.z + normalForInitWorld.z * -distance;
        const photoSamples = computePhotometricSamples(camera.lastNoisedPreviewGray, w, h, 4);
        lastPositionLMResult = refineOrientationAndPositionLM(
          photoSamples, w, h, { Drow, Dcol, Dnormal }, distance, initialWorldX0, initialWorldZ0, MATH_QUAT, vFovRad, camera.aspect,
        );
        camera.lastRecoveredAxes.Drow = lastPositionLMResult.Drow;
        camera.lastRecoveredAxes.Dcol = lastPositionLMResult.Dcol;
        camera.lastRecoveredAxes.Dnormal = lastPositionLMResult.Dnormal;
        const refinedNormal = lastPositionLMResult.Dnormal.clone();
        if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(refinedNormal) > 0) refinedNormal.negate();
        if (camera.lastDecodeRotated) {
          const anchorRow = ((camera.lastPositionDecode.row - camera.lastDecodeRotated.zeroI) % R + R) % R;
          const anchorCol = ((camera.lastPositionDecode.col - camera.lastDecodeRotated.zeroJ) % C + C) % C;
          const refinedQuat = solveRecoveredCamQuat(
            camera.lastDecodeRotated, anchorRow, anchorCol, camera.lastRecoveredAxes.Drow, camera.lastRecoveredAxes.Dcol, refinedNormal, distance,
          );
          if (refinedQuat) camera.lastPositionDecode.recoveredCamQuat = refinedQuat;
        }
        const refinedNormalWorld = refinedNormal.clone().applyQuaternion(camera.lastPositionDecode.recoveredCamQuat);
        camera.lastPositionDecode.camPos.x = lastPositionLMResult.worldX0 + refinedNormalWorld.x * distance;
        camera.lastPositionDecode.camPos.z = lastPositionLMResult.worldZ0 + refinedNormalWorld.z * distance;
        buildProjectedTexture(camera);

        const postPhase3Bins = camera.lastProjectedBins;
        if (postPhase3Bins) {
          const extentU = postPhase3Bins.maxU - postPhase3Bins.minU;
          const extentV = postPhase3Bins.maxV - postPhase3Bins.minV;
          const currentDistance = camera.lastRecoveredAxes.distance;
          const measured = measurePeriodDistance(camera, currentDistance, extentU, extentV);
          if (measured) {
            finalSpacing = measured;
            camera.lastRecoveredAxes.distance = (measured.distanceU + measured.distanceV) / 2;
            camera.lastPositionDecode.camPos.x = lastPositionLMResult.worldX0 + refinedNormalWorld.x * camera.lastRecoveredAxes.distance;
            camera.lastPositionDecode.camPos.z = lastPositionLMResult.worldZ0 + refinedNormalWorld.z * camera.lastRecoveredAxes.distance;
            buildProjectedTexture(camera);
          }
        }
      }
      if (camera.lastPositionDecode && rowDirRecovered && colDirRecovered) {
        const { recoveredCamQuat } = camera.lastPositionDecode;
        const rowDirWorld = rowDirRecovered.clone().applyQuaternion(recoveredCamQuat);
        const colDirWorld = colDirRecovered.clone().applyQuaternion(recoveredCamQuat);
        camera.recoveredRowPoleA.position.copy(rowDirWorld).multiplyScalar(SPHERE_RADIUS);
        camera.recoveredRowPoleB.position.copy(rowDirWorld).multiplyScalar(-SPHERE_RADIUS);
        camera.recoveredColPoleA.position.copy(colDirWorld).multiplyScalar(SPHERE_RADIUS);
        camera.recoveredColPoleB.position.copy(colDirWorld).multiplyScalar(-SPHERE_RADIUS);
      }
      updateRecoveredCamGizmo(camera);
      applyRecoveredFloorOverlay(camera);
      if (isActive && globalState.mode === 'through') updateContaminationOverlays(camera);
      const t4 = performance.now();

      if (isActive) {
        const haveGroundTruth = isSimulated(camera);
        const lines = [`${votes.length} votes  (${fitVotes.length} fed to fit)`];
        if (rowDirRecovered && colDirRecovered) {
          if (haveGroundTruth) {
            const errUnswapped = angleBetweenDegV(rowDirRecovered, ROW_DIR) + angleBetweenDegV(colDirRecovered, COL_DIR);
            const errSwapped = angleBetweenDegV(rowDirRecovered, COL_DIR) + angleBetweenDegV(colDirRecovered, ROW_DIR);
            const rowErr = errSwapped < errUnswapped ? angleBetweenDegV(rowDirRecovered, COL_DIR) : angleBetweenDegV(rowDirRecovered, ROW_DIR);
            const colErr = errSwapped < errUnswapped ? angleBetweenDegV(colDirRecovered, ROW_DIR) : angleBetweenDegV(colDirRecovered, COL_DIR);
            lines.push(`row err ${rowErr.toFixed(2)}°  col err ${colErr.toFixed(2)}°  [Phase 1, vote-based${errSwapped < errUnswapped ? ', swapped' : ''}]`);
          }
        } else {
          lines.push(`degenerate fit`);
        }
        if (refinedFit) {
          lines.push(`LM: ${refinedFit.iterations} iters, cost ${refinedFit.initialCost.toExponential(2)} -> ${refinedFit.finalCost.toExponential(2)}`);
        }
        if (spacing) {
          const trueDist = isSimulated(camera) ? camera.camPos.y : NaN;
          const distU = spacing.distanceU, distV = spacing.distanceV;
          if (haveGroundTruth) {
            const errU = (Math.abs(distU - trueDist) / trueDist) * 100;
            const errV = (Math.abs(distV - trueDist) / trueDist) * 100;
            lines.push(`dist U ${distU.toFixed(2)} (${errU.toFixed(1)}% err)  dist V ${distV.toFixed(2)} (${errV.toFixed(1)}% err)  true ${trueDist.toFixed(2)}  [rough, rtSize buckets]`);
          } else {
            lines.push(`dist U ${distU.toFixed(2)}  dist V ${distV.toFixed(2)}  [rough, rtSize buckets]`);
          }
          if (refinedSpacing) {
            const rDistU = refinedSpacing.distanceU, rDistV = refinedSpacing.distanceV;
            if (haveGroundTruth) {
              const rErrU = (Math.abs(rDistU - trueDist) / trueDist) * 100;
              const rErrV = (Math.abs(rDistV - trueDist) / trueDist) * 100;
              lines.push(`dist U ${rDistU.toFixed(2)} (${rErrU.toFixed(1)}% err)  dist V ${rDistV.toFixed(2)} (${rErrV.toFixed(1)}% err)  [refined, adaptive buckets]`);
            } else {
              lines.push(`dist U ${rDistU.toFixed(2)}  dist V ${rDistV.toFixed(2)}  [refined, adaptive buckets]`);
            }
          }
          if (finalSpacing) {
            const fDistU = finalSpacing.distanceU, fDistV = finalSpacing.distanceV;
            if (haveGroundTruth) {
              const fErrU = (Math.abs(fDistU - trueDist) / trueDist) * 100;
              const fErrV = (Math.abs(fDistV - trueDist) / trueDist) * 100;
              lines.push(`dist U ${fDistU.toFixed(2)} (${fErrU.toFixed(1)}% err)  dist V ${fDistV.toFixed(2)} (${fErrV.toFixed(1)}% err)  [final, post-Phase-3 orientation]`);
            } else {
              lines.push(`dist U ${fDistU.toFixed(2)}  dist V ${fDistV.toFixed(2)}  [final, post-Phase-3 orientation]`);
            }
          }
        } else if (quadricPair) {
          lines.push(`spacing: no period found`);
        }
        if (camera.lastPositionDecode) {
          lines.push(`decoded torus (row,col): (${camera.lastPositionDecode.row}, ${camera.lastPositionDecode.col})  consistency ${(camera.lastPositionDecode.consistency * 100).toFixed(1)}%  camPos (${camera.lastPositionDecode.camPos.x.toFixed(2)}, ${camera.lastPositionDecode.camPos.y.toFixed(2)}, ${camera.lastPositionDecode.camPos.z.toFixed(2)})`);
        }
        if (lastPositionLMResult) {
          lines.push(`photoLM: ${lastPositionLMResult.iterations} iters, cost ${lastPositionLMResult.initialCost.toExponential(2)} -> ${lastPositionLMResult.finalCost.toExponential(2)}`);
          if (camera.lastRecoveredAxes && haveGroundTruth) {
            const { Drow: finalDrow, Dcol: finalDcol } = camera.lastRecoveredAxes;
            const errUnswapped = angleBetweenDegV(finalDrow, ROW_DIR) + angleBetweenDegV(finalDcol, COL_DIR);
            const errSwapped = angleBetweenDegV(finalDrow, COL_DIR) + angleBetweenDegV(finalDcol, ROW_DIR);
            const finalRowErr = errSwapped < errUnswapped ? angleBetweenDegV(finalDrow, COL_DIR) : angleBetweenDegV(finalDrow, ROW_DIR);
            const finalColErr = errSwapped < errUnswapped ? angleBetweenDegV(finalDcol, ROW_DIR) : angleBetweenDegV(finalDcol, COL_DIR);
            lines.push(`row err ${finalRowErr.toFixed(2)}°  col err ${finalColErr.toFixed(2)}°  [Phase 3, final${errSwapped < errUnswapped ? ', swapped' : ''}]`);
          }
        }
        lines.push(`votes ${(t1 - t0).toFixed(0)}ms  fit ${(t2 - t1).toFixed(0)}ms  LM ${(t2b - t2).toFixed(0)}ms  spacing ${(t3 - t2b).toFixed(0)}ms  refine ${(t3b - t3).toFixed(0)}ms  decode ${(t4 - t3b).toFixed(0)}ms`);
        axesReadout.textContent = lines.join('\n');
        updatePositionReadoutText(camera);
      }
    } finally {
      if (isActive) {
        captureAxesBtn.disabled = false;
        captureAxesBtn.textContent = prevLabel;
      }
      camera.axesCapturing = false;
    }
  });
}

// ── Projected-Cam marginal graphs / sample lattice ───────────────────────

const MARGINAL_THICKNESS = 90;

function drawMarginalLines(camera: Camera, x: number, y: number, w: number, h: number) {
  if (!camera.lastMarginals) { hideMarginalLines(); return; }
  const m = camera.lastMarginals;

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
    let maxMag = 0;
    for (let i = 0; i < n; i++) if (m.rowMag[i] > maxMag) maxMag = m.rowMag[i];
    rc.lineWidth = 1;
    let prevPx = 0, prevPy = 0;
    for (let i = 0; i < n; i++) {
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
    let maxMag = 0;
    for (let i = 0; i < n; i++) if (m.colMag[i] > maxMag) maxMag = m.colMag[i];
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

  updatePositionReadoutText(camera);
}

function updatePositionReadoutText(camera: Camera) {
  if (!positionReadout) return;
  if (!camera.lastMarginals) { positionReadout.textContent = 'not yet computed (switch to Projected Cam or capture now)'; return; }
  const m = camera.lastMarginals;
  const uStep = m.colPeriod && camera.lastProjectedBins ? m.colPeriod * camera.lastProjectedBins.binWidthU : null;
  const vStep = m.rowPeriod && camera.lastProjectedBins ? m.rowPeriod * camera.lastProjectedBins.binWidthV : null;
  const periodicityLines =
    `col period: ${m.colPeriod ?? '—'} bins (phase ${m.colPhase.toFixed(1)})\n` +
    `row period: ${m.rowPeriod ?? '—'} bins (phase ${m.rowPhase.toFixed(1)})\n` +
    `implied grid step: U=${uStep?.toFixed(3) ?? '—'}  V=${vStep?.toFixed(3) ?? '—'}\n` +
    `(expect both ≈ ${GRID_STEP})`;
  let decodeLines: string;
  if (camera.lastPositionDecode) {
    const rec = camera.lastPositionDecode.camPos;
    if (isPhysical(camera)) {
      decodeLines =
        `torus cell: row ${camera.lastPositionDecode.row}  col ${camera.lastPositionDecode.col}\n` +
        `consistency: ${(camera.lastPositionDecode.consistency * 100).toFixed(1)}%\n` +
        `recovered camPos: (${rec.x.toFixed(2)}, ${rec.y.toFixed(2)}, ${rec.z.toFixed(2)})`;
    } else {
      const errPos = rec.distanceTo(camera.camPos);
      const errOrientationDeg = THREE.MathUtils.radToDeg(camera.camQuat.angleTo(camera.lastPositionDecode.recoveredCamQuat));
      decodeLines =
        `torus cell: row ${camera.lastPositionDecode.row}  col ${camera.lastPositionDecode.col}\n` +
        `consistency: ${(camera.lastPositionDecode.consistency * 100).toFixed(1)}%\n` +
        `recovered camPos: (${rec.x.toFixed(2)}, ${rec.y.toFixed(2)}, ${rec.z.toFixed(2)})\n` +
        `true camPos: (${camera.camPos.x.toFixed(2)}, ${camera.camPos.y.toFixed(2)}, ${camera.camPos.z.toFixed(2)})\n` +
        `error: ${errPos.toFixed(3)} world units\n` +
        `orientation error: ${errOrientationDeg.toFixed(2)}° (recoveredCamQuat vs true camQuat -- ground-truth diagnostic, lab-only)`;
    }
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

function drawSampleLattice(camera: Camera, x: number, y: number, w: number, h: number) {
  if (!camera.settings.showSampleLattice) { hideSampleLattice(); return; }
  const grid = camera.lastDecodeRotated;
  if (!grid || !camera.lastProjectedBins) { hideSampleLattice(); return; }
  const { maxU, binWidthU, minV, binWidthV, w: bw, h: bh } = camera.lastProjectedBins;

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
  for (let i = 0; i < grid.rows; i++) {
    for (let j = 0; j < grid.cols; j++) {
      const pt = grid.points[i][j];
      if (!pt.valid) continue;
      const bu = (maxU - pt.u) / binWidthU;
      const bv = (pt.v - minV) / binWidthV;
      if (bu < 0 || bu >= bw || bv < 0 || bv >= bh) continue;
      const cx = (bu / bw) * sampleLatticeCanvas.width;
      const cy = (1 - bv / bh) * sampleLatticeCanvas.height;
      const debug = camera.lastDecodeCorrectness ? camera.lastDecodeCorrectness[i][j] : null;
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

// Rebuilds the recovered-floor overlay's geometry/position/orientation --
// called once per fresh decode, not per frame.
function applyRecoveredFloorOverlay(camera: Camera) {
  if (!camera.lastPositionDecode || !camera.lastRecoveredAxes || !camera.lastProjectedBins) return;
  const { Drow: DrowMath, Dcol: DcolMath, Dnormal, distance } = camera.lastRecoveredAxes;
  const normalMath = Dnormal.clone();
  const vFovRad = getAnalysisVFovRad(camera);
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalMath) > 0) normalMath.negate();
  const { recoveredCamQuat } = camera.lastPositionDecode;
  const Drow = DrowMath.clone().applyQuaternion(recoveredCamQuat);
  const Dcol = DcolMath.clone().applyQuaternion(recoveredCamQuat);
  const normal = normalMath.clone().applyQuaternion(recoveredCamQuat);
  const { minU, maxU, minV, maxV } = camera.lastProjectedBins;
  const width = maxU - minU, height = maxV - minV;
  if (!(width > 0) || !(height > 0)) return;

  camera.recoveredFloorOverlay.geometry.dispose();
  camera.recoveredFloorOverlay.geometry = new THREE.PlaneGeometry(width, height);

  const centerU = (minU + maxU) / 2, centerV = (minV + maxV) / 2;
  camera.recoveredFloorOverlay.position.copy(camera.lastPositionDecode.camPos)
    .addScaledVector(Drow, centerU)
    .addScaledVector(Dcol, centerV)
    .addScaledVector(normal, -distance);

  const drowDisplay = Drow.clone().negate();
  const zAxis = new THREE.Vector3().crossVectors(drowDisplay, Dcol).normalize();
  const basis = new THREE.Matrix4().makeBasis(drowDisplay, Dcol, zAxis);
  camera.recoveredFloorOverlay.quaternion.setFromRotationMatrix(basis);
}

// Same shape/size as the ground-truth gizmoBody, in green, at the DECODED
// position AND orientation from runPositionDecode.
function updateRecoveredCamGizmo(camera: Camera) {
  if (camera.lastPositionDecode) {
    camera.recoveredCamGizmo.position.copy(camera.lastPositionDecode.camPos);
    camera.recoveredCamGizmo.quaternion.copy(camera.lastPositionDecode.recoveredCamQuat);
  }
  camera.recoveredCamGizmo.visible = globalState.mode === 'world' && camera.settings.showGizmoBody && !!camera.lastPositionDecode;
}

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
  if (globalState.mode !== 'inside') return;
  dragging = true; lastPX = e.clientX; lastPY = e.clientY;
});
addEventListener('pointerup', () => { dragging = false; });
addEventListener('pointermove', (e) => {
  if (!dragging || globalState.mode !== 'inside') return;
  const dx = e.clientX - lastPX, dy = e.clientY - lastPY;
  lastPX = e.clientX; lastPY = e.clientY;
  insideYaw -= dx * 0.004;
  insidePitch = THREE.MathUtils.clamp(insidePitch - dy * 0.004, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
});
canvas.addEventListener('wheel', (e) => {
  if (globalState.mode !== 'inside') return;
  e.preventDefault();
  insideCam.fov = THREE.MathUtils.clamp(insideCam.fov + e.deltaY * 0.02, 20, 110);
  insideCam.updateProjectionMatrix();
}, { passive: false });

// ── Mode switching ───────────────────────────────────────────────────────

function setMode(m: Mode) {
  globalState.mode = m;
  persistControl('mode', m);
  for (const k of Object.keys(modeBtns) as Mode[]) modeBtns[k].classList.toggle('active', k === m);
  worldOrbit.enabled = m === 'world';
  insideHint.style.display = m === 'inside' ? 'block' : 'none';
  pipFrame.style.display = m === 'through' || m === 'projected' ? 'none' : 'block';
  pipLabel.style.display = m === 'through' || m === 'projected' ? 'none' : 'block';
  const cam = activeCamera();
  if (m === 'projected') { if (cam) buildProjectedTexture(cam); }
  else { hideMarginalLines(); hideSampleLattice(); }
  contamToggles.style.display = m === 'through' ? 'flex' : 'none';
  arrowToggles.style.display = m === 'through' ? 'flex' : 'none';
  if (m !== 'through') clearGradientArrowOverlay();
  if (m === 'through' && cam) { updateDistortedPreview(cam); updateContaminationOverlays(cam); }
}
modeBtns.world.addEventListener('click', () => setMode('world'));
modeBtns.through.addEventListener('click', () => setMode('through'));
modeBtns.inside.addEventListener('click', () => setMode('inside'));
modeBtns.projected.addEventListener('click', () => setMode('projected'));

function setPanelCollapsed(collapsed: boolean) {
  panel.classList.toggle('collapsed', collapsed);
  panelToggle.classList.toggle('collapsed', collapsed);
  panelToggle.textContent = collapsed ? '›' : '‹';
  persistControl('panelCollapsed', collapsed ? '1' : '0');
}
panelToggle.addEventListener('click', () => setPanelCollapsed(!panel.classList.contains('collapsed')));
setPanelCollapsed(savedControls['panelCollapsed'] === '1');

// Letterbox rect for the fixed-aspect gizmo camera within whatever shape the
// window currently is.
function computeThroughRect(camera: Camera): { x: number; y: number; w: number; h: number } {
  const winAspect = innerWidth / innerHeight;
  let w = innerWidth, h = innerHeight, x = 0, y = 0;
  if (winAspect > camera.aspect) { w = innerHeight * camera.aspect; x = (innerWidth - w) / 2; }
  else { h = innerWidth / camera.aspect; y = (innerHeight - h) / 2; }
  return { x, y, w, h };
}

function clearGradientArrowOverlay() {
  gradientArrowCtx.clearRect(0, 0, gradientArrowCanvas.width, gradientArrowCanvas.height);
}
function drawOneArrow(px: number, py: number, dirVecX: number, dirVecY: number, color: string, scale: number) {
  const tipX = px + dirVecX * scale, tipY = py + dirVecY * scale;
  const headLen = 8, headAngle = Math.PI / 7;
  const backAngle = Math.atan2(tipY - py, tipX - px);
  const headPath = new Path2D();
  headPath.moveTo(tipX, tipY);
  headPath.lineTo(tipX - headLen * Math.cos(backAngle - headAngle), tipY - headLen * Math.sin(backAngle - headAngle));
  headPath.lineTo(tipX - headLen * Math.cos(backAngle + headAngle), tipY - headLen * Math.sin(backAngle + headAngle));
  headPath.closePath();

  const ctx = gradientArrowCtx;
  ctx.strokeStyle = 'rgba(0,0,0,0.85)'; ctx.fillStyle = 'rgba(0,0,0,0.85)'; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tipX, tipY); ctx.stroke();
  ctx.fill(headPath);
  ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();

  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(tipX, tipY); ctx.stroke();
  ctx.fill(headPath);
  ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
}

// Diagnostic-only re-run of guidedTangentDirection's walk that also records
// which pixels actually got incorporated -- must stay in exact lockstep
// with guidedTangentDirection's own logic.
function computeTangentWalkIncludedPixels(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { x: number; y: number }[] {
  const included: { x: number; y: number }[] = [{ x, y }];
  const seedTheta = Math.atan2(seedFy, seedFx);
  const tdx = -Math.sin(seedTheta), tdy = Math.cos(seedTheta);
  const seedMag = Math.hypot(seedFx, seedFy);
  let sumCos = Math.cos(2 * seedTheta) * seedMag;
  let sumSin = Math.sin(2 * seedTheta) * seedMag;
  let runningMag = seedMag;
  let sampleCount = 1;
  const maxSteps = settings.tangentWalkMaxSteps;
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(settings.tangentWalkDeviationDeg));
  const magFraction = settings.tangentWalkMagFraction;
  const grace = settings.tangentWalkGraceSamples;
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
        continue;
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
      included.push({ x: sx, y: sy });
    }
  }
  return included;
}

function computeTangentWalkIncludedPixelsAdaptive(
  settings: CameraSettingsCommon,
  fx: Float64Array, fy: Float64Array, w: number, h: number,
  x: number, y: number, seedFx: number, seedFy: number,
): { x: number; y: number }[] {
  const included: { x: number; y: number }[] = [{ x, y }];
  const seedTheta = Math.atan2(seedFy, seedFx);
  const seedMag = Math.hypot(seedFx, seedFy);
  const seedCos = Math.cos(2 * seedTheta) * seedMag, seedSin = Math.sin(2 * seedTheta) * seedMag;
  const maxSteps = settings.tangentWalkMaxSteps;
  const devCos = Math.cos(2 * THREE.MathUtils.degToRad(settings.tangentWalkDeviationDeg));
  const magFraction = settings.tangentWalkMagFraction;
  const grace = settings.tangentWalkGraceSamples;
  for (const sign of [1, -1]) {
    let sumCos = seedCos, sumSin = seedSin, runningMag = seedMag, sampleCount = 1;
    let curX = x, curY = y;
    let violations = 0;
    for (let k = 1; k <= maxSteps; k++) {
      const avgTheta = Math.atan2(sumSin, sumCos) / 2;
      const tdx = -Math.sin(avgTheta), tdy = Math.cos(avgTheta);
      curX += sign * tdx; curY += sign * tdy;
      const sx = Math.round(curX), sy = Math.round(curY);
      if (sx < 0 || sx >= w || sy < 0 || sy >= h) break;
      const si = sy * w + sx;
      const sfx = fx[si], sfy = fy[si];
      const mag = Math.hypot(sfx, sfy);
      if (mag === 0 || mag < runningMag * magFraction) {
        violations++;
        if (violations >= grace) break;
        continue;
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
      included.push({ x: sx, y: sy });
    }
  }
  return included;
}

function drawTangentWalkOutline(
  rect: { x: number; y: number; w: number; h: number }, fieldW: number, fieldH: number,
  fx: Float64Array, fy: Float64Array, included: { x: number; y: number }[],
) {
  const cellW = rect.w / fieldW, cellH = rect.h / fieldH;
  const ctx = gradientArrowCtx;
  for (let idx = 0; idx < included.length; idx++) {
    const { x: fc, y: fr } = included[idx];
    const boxLeft = rect.x + fc * cellW;
    const boxTop = rect.y + rect.h - (fr + 1) * cellH;
    const isSeed = idx === 0;
    if (isSeed) {
      const si = fr * fieldW + fc;
      let hueTheta = Math.atan2(fy[si], fx[si]);
      if (hueTheta < 0) hueTheta += Math.PI;
      if (hueTheta >= Math.PI) hueTheta -= Math.PI;
      const [rr, gg, bb] = hsvToRgb((hueTheta / Math.PI) * 360, 1, 1);
      ctx.strokeStyle = `rgb(${255 - rr},${255 - gg},${255 - bb})`;
      ctx.lineWidth = 2.5;
    } else {
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1.5;
    }
    ctx.strokeRect(boxLeft + 0.5, boxTop + 0.5, Math.max(1, cellW - 1), Math.max(1, cellH - 1));
  }
}

// Single per-hover entry point -- operates on the ACTIVE camera, since only
// its Through-Cam view is ever on screen.
function updateHoverOverlays(clientX: number, clientY: number) {
  const camera = activeCamera();
  if (!camera) { clearGradientArrowOverlay(); return; }
  const settings = camera.settings;
  const arrowsOn = settings.showGradientArrow || settings.showGradientArrowPerpendicular;
  const walkOn = settings.showTangentWalkPath;
  if (globalState.mode !== 'through' || (!arrowsOn && !walkOn)) { clearGradientArrowOverlay(); return; }
  const rect = computeThroughRect(camera);
  if (clientX < rect.x || clientX >= rect.x + rect.w || clientY < rect.y || clientY >= rect.y + rect.h) { clearGradientArrowOverlay(); return; }

  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const nx = (clientX - rect.x) / rect.w, ny = (clientY - rect.y) / rect.h;
  const fieldCol = Math.min(fieldW - 1, Math.max(0, Math.floor(nx * fieldW)));
  const fieldRow = Math.min(fieldH - 1, Math.max(0, Math.floor((1 - ny) * fieldH)));
  const i = fieldRow * fieldW + fieldCol;

  clearGradientArrowOverlay();

  if (arrowsOn && camera.lastDisplayedVectorField) {
    const { fx, fy } = camera.lastDisplayedVectorField;
    const gx = fx[i], gy = fy[i];
    const mag = Math.hypot(gx, gy);
    if (mag > 0) {
      const px = rect.x + (fieldCol + 0.5) * (rect.w / fieldW);
      const py = rect.y + rect.h - (fieldRow + 0.5) * (rect.h / fieldH);
      let hueTheta = Math.atan2(gy, gx);
      if (hueTheta < 0) hueTheta += Math.PI;
      if (hueTheta >= Math.PI) hueTheta -= Math.PI;
      const [rr, gg, bb] = hsvToRgb((hueTheta / Math.PI) * 360, 1, 1);
      const color = `rgb(${rr},${gg},${bb})`;

      if (settings.showGradientArrow) {
        const theta = Math.atan2(gy, gx);
        drawOneArrow(px, py, Math.cos(theta) * mag, -Math.sin(theta) * mag, color, settings.gradientArrowScale);
      }
      if (settings.showGradientArrowPerpendicular) {
        const theta = Math.atan2(gx, -gy);
        drawOneArrow(px, py, Math.cos(theta) * mag, -Math.sin(theta) * mag, color, settings.gradientArrowScale);
      }
    }
  }

  if (walkOn && camera.lastEffectiveField) {
    const { fx, fy } = camera.lastEffectiveField;
    const seedFx = fx[i], seedFy = fy[i];
    if (seedFx !== 0 || seedFy !== 0) {
      const included = settings.tangentWalkAdaptive
        ? computeTangentWalkIncludedPixelsAdaptive(settings, fx, fy, fieldW, fieldH, fieldCol, fieldRow, seedFx, seedFy)
        : computeTangentWalkIncludedPixels(settings, fx, fy, fieldW, fieldH, fieldCol, fieldRow, seedFx, seedFy);
      drawTangentWalkOutline(rect, fieldW, fieldH, fx, fy, included);
    }
  }
}
let lastHoverClientX = -1, lastHoverClientY = -1;
canvas.addEventListener('pointermove', (e) => {
  lastHoverClientX = e.clientX; lastHoverClientY = e.clientY;
  updateHoverOverlays(e.clientX, e.clientY);
});
canvas.addEventListener('pointerleave', () => {
  lastHoverClientX = -1; lastHoverClientY = -1;
  clearGradientArrowOverlay();
});

toggleHideFieldBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.hideField = !cam.settings.hideField;
  toggleHideFieldBtn.classList.toggle('active', cam.settings.hideField);
  updateDistortedPreview(cam);
  updateContaminationOverlays(cam);
});
toggleTrueContamBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showTrueContamination = !cam.settings.showTrueContamination;
  toggleTrueContamBtn.classList.toggle('active', cam.settings.showTrueContamination);
  updateDistortedPreview(cam);
  updateContaminationOverlays(cam);
});
toggleReconContamBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showReconstructedContamination = !cam.settings.showReconstructedContamination;
  toggleReconContamBtn.classList.toggle('active', cam.settings.showReconstructedContamination);
  updateDistortedPreview(cam);
  updateContaminationOverlays(cam);
});
toggleGradientArrowBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showGradientArrow = !cam.settings.showGradientArrow;
  toggleGradientArrowBtn.classList.toggle('active', cam.settings.showGradientArrow);
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
toggleGradientArrowModeBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showGradientArrowPerpendicular = !cam.settings.showGradientArrowPerpendicular;
  toggleGradientArrowModeBtn.classList.toggle('active', cam.settings.showGradientArrowPerpendicular);
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
toggleTangentWalkPathBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showTangentWalkPath = !cam.settings.showTangentWalkPath;
  toggleTangentWalkPathBtn.classList.toggle('active', cam.settings.showTangentWalkPath);
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});

// ── Slider / checkbox wiring ─────────────────────────────────────────────

function setSectionDisabled(section: HTMLDivElement, disabled: boolean) {
  section.classList.toggle('disabled', disabled);
  for (const el of section.querySelectorAll('input')) (el as HTMLInputElement).disabled = disabled;
}

// ── Camera lifecycle: tabs, add/remove (Stage B) ─────────────────────────

const cameraTabsEl = document.getElementById('cameraTabs') as HTMLDivElement;

// Rebuilds the tab bar from `cameras` (Map iteration = creation order) --
// called after anything that adds/removes/renames a camera or changes which
// one is active. Cheap enough (a handful of plain DOM nodes) to just rebuild
// wholesale rather than diff.
function renderCameraTabs() {
  cameraTabsEl.innerHTML = '';
  for (const camera of cameras.values()) {
    const tab = document.createElement('button');
    tab.className = 'cameraTab' + (camera.id === activeCameraId ? ' active' : '');
    tab.style.setProperty('--tab-color', `#${camera.color.getHexString()}`);
    tab.title = camera.type === 'simulated' ? 'simulated camera' : 'physical camera';
    const label = document.createElement('span');
    label.textContent = camera.name;
    tab.appendChild(label);
    // Every camera can be removed in Stage B (a plain local teardown) --
    // Stage C adds a distinct "kick" affordance specifically for physical
    // cameras (closes the phone's own connection server-side); this stays
    // the always-available "stop showing/tracking this camera" action
    // regardless of type, so it isn't wasted work once Stage C lands.
    if (cameras.size > 1) {
      const close = document.createElement('span');
      close.className = 'cameraTabClose';
      close.textContent = '×';
      close.title = 'remove this camera';
      close.addEventListener('click', (e) => { e.stopPropagation(); removeCameraTab(camera.id); });
      tab.appendChild(close);
    }
    tab.addEventListener('click', () => {
      if (camera.id === activeCameraId) return;
      activeCameraId = camera.id;
      renderCameraTabs();
      refreshCameraPanel();
    });
    cameraTabsEl.appendChild(tab);
  }
  const addBtn = document.createElement('button');
  addBtn.className = 'cameraTabAdd';
  addBtn.textContent = '+';
  addBtn.title = 'add a simulated camera';
  addBtn.addEventListener('click', () => addSimulatedCamera());
  cameraTabsEl.appendChild(addBtn);
}

// Brings a freshly-created-or-reactivated camera's capture pipeline up to
// date -- the same handful of calls every camera-creation path needs.
function primeCameraForDisplay(camera: Camera) {
  if (isSimulated(camera)) renderCamRT(camera); // populate camRT before reading it back below, so the preview isn't blank for the first frame or two
  updateDistortedPreview(camera);
  if (globalState.mode === 'projected') buildProjectedTexture(camera);
  markCaptureDirty(camera);
  layoutPip(camera);
}

// Adds a new simulated camera ALONGSIDE whatever already exists (unlike
// switchActiveCameraType below, which REPLACES the active one) and makes it
// active. Offsets its default X position a few units per already-existing
// camera so a fresh gizmo doesn't spawn exactly on top of another one.
function addSimulatedCamera() {
  const camera = createSimulatedCamera(nextCameraColor());
  camera.settings.camX += (cameras.size % 6) * 3;
  cameras.set(camera.id, camera);
  activeCameraId = camera.id;
  primeCameraForDisplay(camera);
  renderCameraTabs();
  refreshCameraPanel();
}

// Tears down one camera. If it was the active one, falls back to whichever
// camera is next in the map, or -- if that was the last camera left --
// creates a fresh default simulated camera, so the app is never left with
// zero cameras and no way to add one back.
function removeCameraTab(id: string) {
  const camera = cameras.get(id);
  if (!camera) return;
  const wasActive = id === activeCameraId;
  destroyCamera(camera);
  if (wasActive) {
    const next = cameras.values().next().value;
    const replacement = next ?? createSimulatedCamera(nextCameraColor());
    if (!next) cameras.set(replacement.id, replacement);
    activeCameraId = replacement.id;
    primeCameraForDisplay(replacement);
  }
  renderCameraTabs();
  refreshCameraPanel();
}

// Re-syncs every per-camera control's DISPLAYED value/state to match
// whichever camera just became active. Writes already redirect correctly on
// every tick regardless (bindSlider/bindCheckbox's onChange callbacks all
// look up activeCamera() fresh each time they fire) -- this is purely the
// other direction, camera state -> DOM. Dispatches the SAME 'input'/'change'
// events bindSlider/bindCheckbox already listen for, reusing their existing
// fmt/persist/onChange logic wholesale instead of duplicating it; the
// onChange round-trip this causes (writing the same value straight back to
// the SAME camera it was just read from) is a harmless no-op. useRealCapture
// is the one exception -- it must NOT dispatch (that would immediately
// destroy the camera this function is trying to display), so its checked
// state is set directly and silently instead.
function refreshCameraPanel() {
  const cam = activeCamera();
  if (!cam) return;

  const useRealCaptureInput = document.getElementById('useRealCapture') as HTMLInputElement;
  useRealCaptureInput.checked = isPhysical(cam);
  setSectionDisabled(cameraDetailsSection, isPhysical(cam));
  setSectionDisabled(simDistortionSection, isPhysical(cam));

  const setNum = (id: string, v: number) => {
    const el = document.getElementById(id) as HTMLInputElement | null;
    if (!el) return;
    el.value = String(v);
    el.dispatchEvent(new Event('input'));
  };
  const setBool = (id: string, v: boolean) => {
    const el = document.getElementById(id) as HTMLInputElement | null;
    if (!el) return;
    el.checked = v;
    el.dispatchEvent(new Event('change'));
  };

  if (isSimulated(cam)) {
    setNum('camX', cam.settings.camX); setNum('camY', cam.settings.camY); setNum('camZ', cam.settings.camZ);
    setNum('camYaw', cam.settings.camYawDeg); setNum('camPitch', cam.settings.camPitchDeg); setNum('camFocal', cam.settings.focalMM);
    setNum('simNoise', cam.settings.simNoise); setNum('simBlur', cam.settings.simBlur); setNum('captureSupersample', cam.settings.captureSupersample);
  } else {
    setNum('realCaptureFovDeg', cam.settings.realCaptureFovDeg);
  }
  setNum('viewportW', cam.settings.viewportW); setNum('viewportH', cam.settings.viewportH);
  setBool('aspectLocked', cam.settings.aspectLocked);

  setBool('showSphere', cam.settings.showSphere); setBool('showCircles', cam.settings.showCircles);
  setBool('showPoles', cam.settings.showPoles); setBool('showFrustum', cam.settings.showFrustum);
  setBool('showPatch', cam.settings.showPatch); setBool('showGizmoBody', cam.settings.showGizmoBody);
  setBool('showRecoveredFloor', cam.settings.showRecoveredFloor); setBool('showSampleLattice', cam.settings.showSampleLattice);
  setBool('orientationLM', cam.settings.orientationLM); setBool('positionLM', cam.settings.positionLM);

  setNum('simGradRadius', cam.settings.simGradRadius); setNum('coherenceRadius', cam.settings.coherenceRadius);
  setNum('tangentWalkMaxSteps', cam.settings.tangentWalkMaxSteps); setNum('tangentWalkDeviationDeg', cam.settings.tangentWalkDeviationDeg);
  setNum('tangentWalkMagFraction', cam.settings.tangentWalkMagFraction); setNum('tangentWalkGraceSamples', cam.settings.tangentWalkGraceSamples);
  setBool('tangentWalkAdaptive', cam.settings.tangentWalkAdaptive);

  const fieldViewId = 'fieldView' + cam.settings.fieldView[0].toUpperCase() + cam.settings.fieldView.slice(1);
  const fieldViewInput = document.getElementById(fieldViewId) as HTMLInputElement | null;
  if (fieldViewInput) { fieldViewInput.checked = true; fieldViewInput.dispatchEvent(new Event('change')); }

  setNum('gradientArrowScale', cam.settings.gradientArrowScale);
  setNum('circleSamplePercentMin', cam.settings.circleSamplePercentMin); setNum('circleSamplePercentMax', cam.settings.circleSamplePercentMax);
  setBool('showRecoveredPoles', cam.settings.showRecoveredPoles); setBool('showAxisVectors', cam.settings.showAxisVectors);
  setBool('showTopCircles', cam.settings.showTopCircles);
  setNum('weightSharpenPower', cam.settings.weightSharpenPower);
  setBool('axesAutoCapture', cam.settings.axesAutoCapture);
  setNum('axesCaptureInterval', cam.settings.axesCaptureIntervalMs);

  toggleHideFieldBtn.classList.toggle('active', cam.settings.hideField);
  toggleTrueContamBtn.classList.toggle('active', cam.settings.showTrueContamination);
  toggleReconContamBtn.classList.toggle('active', cam.settings.showReconstructedContamination);
  toggleGradientArrowBtn.classList.toggle('active', cam.settings.showGradientArrow);
  toggleGradientArrowModeBtn.classList.toggle('active', cam.settings.showGradientArrowPerpendicular);
  toggleTangentWalkPathBtn.classList.toggle('active', cam.settings.showTangentWalkPath);
  updateContaminationAvailability();
  updateGradientArrowAvailability();
  updateTangentWalkPathAvailability();

  updateDistortedPreview(cam);
  if (globalState.mode === 'projected') buildProjectedTexture(cam);
  markCaptureDirty(cam);
  layoutPip(cam);
}

// Destroys the current active camera and creates a fresh one of the other
// type -- see this file's header / Stage A plan for why: a camera's THREE
// objects and settings shape genuinely differ by type (SimulatedCamera vs
// PhysicalCamera don't share a settings shape), so there's no meaningful
// way to preserve state across the switch, matching how today's toggle
// already behaves (it swaps data source, not settings). Keeps the SAME
// color the outgoing camera had (or assigns a fresh one if there wasn't
// one yet), rather than resetting it, so this camera's own tab/gizmo/
// overlay color stays stable across a type flip.
function switchActiveCameraType(kind: 'simulated' | 'physical') {
  const current = activeCamera();
  if (current && current.type === kind) return;
  const color = current ? current.color : nextCameraColor();
  if (current) destroyCamera(current);
  const cam = kind === 'simulated' ? createSimulatedCamera(color) : createPhysicalCamera(color);
  cameras.set(cam.id, cam);
  activeCameraId = cam.id;
  setSectionDisabled(cameraDetailsSection, kind === 'physical');
  setSectionDisabled(simDistortionSection, kind === 'physical');
  primeCameraForDisplay(cam);
  renderCameraTabs();
}
bindCheckbox('useRealCapture', (v) => switchActiveCameraType(v ? 'physical' : 'simulated'));

function rerunOnRealCaptureSettingChange() {
  const cam = activeCamera();
  if (cam && isPhysical(cam) && cam.lastRealCaptureGray) runAxesReconstruction(cam);
}
let realCaptureFovRerunTimer: number | undefined;
bindSlider('realCaptureFovDeg', (v) => {
  const cam = activeCamera();
  if (!cam || !isPhysical(cam)) return;
  cam.settings.realCaptureFovDeg = v;
  markCaptureDirty(cam);
  clearTimeout(realCaptureFovRerunTimer);
  realCaptureFovRerunTimer = window.setTimeout(rerunOnRealCaptureSettingChange, 200);
}, (v) => `${v.toFixed(0)}°`);
bindSlider('camX', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camX = v; markCaptureDirty(cam); } });
bindSlider('camY', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camY = v; markCaptureDirty(cam); } });
bindSlider('camZ', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camZ = v; markCaptureDirty(cam); } });
bindSlider('camYaw', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camYawDeg = v; markCaptureDirty(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('camPitch', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camPitchDeg = v; markCaptureDirty(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('camFocal', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.focalMM = v; markCaptureDirty(cam); } }, (v) => `${v.toFixed(0)}mm`);

let syncingViewportAspect = false;
function clampViewport(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(v)));
}
bindSlider('viewportW', (v) => {
  const cam = activeCamera(); if (!cam) return;
  const oldAspect = cam.settings.viewportW / cam.settings.viewportH;
  cam.settings.viewportW = v;
  if (cam.settings.aspectLocked && !syncingViewportAspect) {
    syncingViewportAspect = true;
    const hInput = document.getElementById('viewportH') as HTMLInputElement;
    hInput.value = String(clampViewport(v / oldAspect, 96, 2000));
    hInput.dispatchEvent(new Event('input'));
    syncingViewportAspect = false;
  }
  resizeCaptureBuffers(cam);
}, (v) => v.toFixed(0));
bindSlider('viewportH', (v) => {
  const cam = activeCamera(); if (!cam) return;
  const oldAspect = cam.settings.viewportW / cam.settings.viewportH;
  cam.settings.viewportH = v;
  if (cam.settings.aspectLocked && !syncingViewportAspect) {
    syncingViewportAspect = true;
    const wInput = document.getElementById('viewportW') as HTMLInputElement;
    wInput.value = String(clampViewport(v * oldAspect, 128, 2000));
    wInput.dispatchEvent(new Event('input'));
    syncingViewportAspect = false;
  }
  resizeCaptureBuffers(cam);
}, (v) => v.toFixed(0));
bindCheckbox('aspectLocked', (v) => { const cam = activeCamera(); if (cam) cam.settings.aspectLocked = v; });

bindCheckbox('showSphere', (v) => { const cam = activeCamera(); if (cam) cam.settings.showSphere = v; });
bindCheckbox('showCircles', (v) => { const cam = activeCamera(); if (cam) cam.settings.showCircles = v; });
bindCheckbox('showPoles', (v) => { const cam = activeCamera(); if (cam) cam.settings.showPoles = v; });
bindCheckbox('showFrustum', (v) => { const cam = activeCamera(); if (cam) cam.settings.showFrustum = v; });
bindCheckbox('showPatch', (v) => { const cam = activeCamera(); if (cam) cam.settings.showPatch = v; });
bindCheckbox('showFloor', (v) => { globalState.showFloor = v; });
bindSlider('floorCellOutlineSubdiv', (v) => {
  globalState.floorCellOutlineSubdiv = v;
  rebuildFloorTexture();
  for (const cam of cameras.values()) markCaptureDirty(cam); // this IS the real rendered floor, so every camera's capture path needs to re-render too
}, (v) => v.toFixed(0));
bindCheckbox('showGizmoBody', (v) => { const cam = activeCamera(); if (cam) cam.settings.showGizmoBody = v; });
bindCheckbox('showRecoveredFloor', (v) => { const cam = activeCamera(); if (cam) cam.settings.showRecoveredFloor = v; });
bindCheckbox('showSampleLattice', (v) => { const cam = activeCamera(); if (cam) cam.settings.showSampleLattice = v; });
bindCheckbox('orientationLM', (v) => { const cam = activeCamera(); if (cam) cam.settings.orientationLM = v; });
bindCheckbox('positionLM', (v) => { const cam = activeCamera(); if (cam) cam.settings.positionLM = v; });

bindSlider('simNoise', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.simNoise = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('simBlur', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.simBlur = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('simGradRadius', (v) => { const cam = activeCamera(); if (cam) { cam.settings.simGradRadius = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('captureSupersample', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.captureSupersample = v; resizeCaptureBuffers(cam); } }, (v) => `${v.toFixed(0)}x`);
bindSlider('coherenceRadius', (v) => { const cam = activeCamera(); if (cam) { cam.settings.coherenceRadius = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('tangentWalkMaxSteps', (v) => { const cam = activeCamera(); if (cam) { cam.settings.tangentWalkMaxSteps = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('tangentWalkDeviationDeg', (v) => { const cam = activeCamera(); if (cam) { cam.settings.tangentWalkDeviationDeg = v; markCaptureDirty(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('tangentWalkMagFraction', (v) => { const cam = activeCamera(); if (cam) { cam.settings.tangentWalkMagFraction = v; markCaptureDirty(cam); } }, (v) => v.toFixed(2));
bindSlider('tangentWalkGraceSamples', (v) => { const cam = activeCamera(); if (cam) { cam.settings.tangentWalkGraceSamples = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindCheckbox('tangentWalkAdaptive', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.tangentWalkAdaptive = v;
  markCaptureDirty(cam);
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
function updateContaminationAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.fieldView === 'gradient' || cam.settings.fieldView === 'effective' || cam.settings.fieldView === 'walked';
  toggleTrueContamBtn.disabled = !relevant;
  toggleReconContamBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showTrueContamination = false;
    cam.settings.showReconstructedContamination = false;
    toggleTrueContamBtn.classList.remove('active');
    toggleReconContamBtn.classList.remove('active');
    cam.trueContamData.fill(0); cam.trueContamTex.needsUpdate = true;
    cam.reconContamData.fill(0); cam.reconContamTex.needsUpdate = true;
  }
}
function updateGradientArrowAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.fieldView === 'gradient' || cam.settings.fieldView === 'effective' || cam.settings.fieldView === 'walked';
  toggleGradientArrowBtn.disabled = !relevant;
  toggleGradientArrowModeBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showGradientArrow = false;
    cam.settings.showGradientArrowPerpendicular = false;
    toggleGradientArrowBtn.classList.remove('active');
    toggleGradientArrowModeBtn.classList.remove('active');
    clearGradientArrowOverlay();
  }
}
function updateTangentWalkPathAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.fieldView === 'effective' || cam.settings.fieldView === 'walked';
  toggleTangentWalkPathBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showTangentWalkPath = false;
    toggleTangentWalkPathBtn.classList.remove('active');
    clearGradientArrowOverlay();
  }
}
bindRadioGroup('fieldView', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.fieldView = v as FieldView;
  markCaptureDirty(cam);
  updateContaminationAvailability();
  updateGradientArrowAvailability();
  updateTangentWalkPathAvailability();
});
updateContaminationAvailability();
updateGradientArrowAvailability();
updateTangentWalkPathAvailability();
bindSlider('gradientArrowScale', (v) => { const cam = activeCamera(); if (cam) cam.settings.gradientArrowScale = v; updateHoverOverlays(lastHoverClientX, lastHoverClientY); }, (v) => v.toFixed(1));
bindSlider('circleSamplePercentMin', (v) => { const cam = activeCamera(); if (cam) { cam.settings.circleSamplePercentMin = v; updateGradientCirclesDebug(cam); } }, (v) => `${v.toFixed(0)}%`);
bindSlider('circleSamplePercentMax', (v) => { const cam = activeCamera(); if (cam) { cam.settings.circleSamplePercentMax = v; updateGradientCirclesDebug(cam); } }, (v) => `${v.toFixed(0)}%`);
bindCheckbox('showRecoveredPoles', (v) => { const cam = activeCamera(); if (cam) cam.settings.showRecoveredPoles = v; });
bindCheckbox('showAxisVectors', (v) => { const cam = activeCamera(); if (cam) cam.settings.showAxisVectors = v; });
bindCheckbox('showTopCircles', (v) => { const cam = activeCamera(); if (cam) cam.settings.showTopCircles = v; });
bindSlider('weightSharpenPower', (v) => { const cam = activeCamera(); if (cam) { cam.settings.weightSharpenPower = v; updateGradientCirclesDebug(cam); } }, (v) => v.toFixed(1));
bindCheckbox('axesAutoCapture', (v) => { const cam = activeCamera(); if (cam) cam.settings.axesAutoCapture = v; });
bindSlider('axesCaptureInterval', (v) => { const cam = activeCamera(); if (cam) cam.settings.axesCaptureIntervalMs = v; }, (v) => `${v.toFixed(0)}`);

captureAxesBtn.addEventListener('click', () => { const cam = activeCamera(); if (cam) runAxesReconstruction(cam); });

// ── Per-frame update ─────────────────────────────────────────────────────

// Ground-truth pose + gizmo update -- simulated cameras only (a physical
// camera has no ground-truth pose to drive a gizmo from).
function updateGizmo(camera: SimulatedCamera): { hFovRad: number; vFovRad: number } {
  camera.camPos.set(camera.settings.camX, camera.settings.camY, camera.settings.camZ);
  euler.set(THREE.MathUtils.degToRad(camera.settings.camPitchDeg), THREE.MathUtils.degToRad(camera.settings.camYawDeg), 0);
  camera.camQuat.setFromEuler(euler);

  camera.gizmoCam.position.copy(camera.camPos);
  camera.gizmoCam.quaternion.copy(camera.camQuat);
  const hFovRad = 2 * Math.atan(SENSOR_WIDTH_MM / (2 * camera.settings.focalMM));
  const vFovRad = 2 * Math.atan(Math.tan(hFovRad / 2) / camera.aspect);
  camera.gizmoCam.fov = THREE.MathUtils.radToDeg(vFovRad);
  camera.gizmoCam.aspect = camera.aspect;
  camera.gizmoCam.updateProjectionMatrix();

  camera.gizmoBody.position.copy(camera.camPos);
  camera.gizmoBody.quaternion.copy(camera.camQuat);
  camera.camHelper.update();

  if (camera === activeCamera()) {
    readout.innerHTML =
      `h-fov: ${THREE.MathUtils.radToDeg(hFovRad).toFixed(1)}&deg; &nbsp; v-fov: ${camera.gizmoCam.fov.toFixed(1)}&deg;<br>` +
      `pole separation: ${THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(ROW_DIR.dot(COL_DIR), -1, 1))).toFixed(2)}&deg; (always 90&deg; — the orthogonal constraint)`;
  }

  return { hFovRad, vFovRad };
}

// Great-sphere overlays (poles/circles/frustum/patch/recovered markers) --
// repositioned (not rotated) to the camera's own origin each frame. A
// simulated camera anchors at its ground-truth camPos/camQuat, exactly as
// before. A physical camera has no ground truth, so it anchors at its own
// RECOVERED position/orientation once a decode exists (and shows nothing
// pose-dependent before that) -- a deliberate, plan-approved Stage A
// deviation from the pre-Stage-A app, which (having no per-camera-type
// concept at all) silently kept showing whatever the simulated sliders'
// last values happened to be even while real-capture mode was active. See
// this file's header comment / the Stage A report for the full rationale.
function updateSphereOverlays(camera: Camera, vFovRad: number) {
  const settings = camera.settings;
  camera.circlesGroup.visible = settings.showCircles;
  camera.sphereShell.visible = settings.showSphere;

  const recoveredPolesVisible = settings.showRecoveredPoles && camera.axesComputed;
  camera.recoveredRowPoleA.visible = recoveredPolesVisible;
  camera.recoveredRowPoleB.visible = recoveredPolesVisible;
  camera.recoveredColPoleA.visible = recoveredPolesVisible;
  camera.recoveredColPoleB.visible = recoveredPolesVisible;
  camera.axisVectorsLines.visible = settings.showAxisVectors;
  camera.gradientCirclesLines.visible = settings.showTopCircles;

  let anchorPos: THREE.Vector3;
  let anchorQuat: THREE.Quaternion | null;
  if (isSimulated(camera)) {
    anchorPos = camera.camPos;
    anchorQuat = camera.camQuat;
    camera.polesGroup.visible = settings.showPoles;
    if (settings.showPoles) {
      camera.rowPoleA.position.copy(ROW_DIR).multiplyScalar(SPHERE_RADIUS);
      camera.rowPoleB.position.copy(ROW_DIR).multiplyScalar(-SPHERE_RADIUS);
      camera.colPoleA.position.copy(COL_DIR).multiplyScalar(SPHERE_RADIUS);
      camera.colPoleB.position.copy(COL_DIR).multiplyScalar(-SPHERE_RADIUS);
    }
  } else {
    anchorPos = camera.lastPositionDecode?.camPos ?? new THREE.Vector3();
    anchorQuat = camera.lastPositionDecode?.recoveredCamQuat ?? null;
  }
  camera.sphereAnchor.position.copy(anchorPos);

  if (settings.showCircles) {
    const updateFamily = (ks: number[], pool: THREE.Line[], axis: 'row' | 'col', dir: THREE.Vector3) => {
      for (let i = 0; i < ks.length; i++) {
        const k = ks[i];
        const pointOnLine = axis === 'row' ? new THREE.Vector3(0, 0, k) : new THREE.Vector3(k, 0, 0);
        const n = greatCircleNormal(pointOnLine, dir, anchorPos);
        pool[i].visible = !!n;
        if (n) writeCirclePoints(pool[i], n, SPHERE_RADIUS);
      }
    };
    updateFamily(rowLineKs, camera.rowCirclePool, 'row', ROW_DIR);
    updateFamily(colLineKs, camera.colCirclePool, 'col', COL_DIR);
  }

  camera.frustumLine.visible = settings.showFrustum && !!anchorQuat;
  if (settings.showFrustum && anchorQuat) {
    const corners = [
      cornerDir(-1, -1, anchorQuat, vFovRad, camera.aspect),
      cornerDir(1, -1, anchorQuat, vFovRad, camera.aspect),
      cornerDir(1, 1, anchorQuat, vFovRad, camera.aspect),
      cornerDir(-1, 1, anchorQuat, vFovRad, camera.aspect),
    ];
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < 4; i++) {
      const a = corners[i], b = corners[(i + 1) % 4];
      for (let t = 0; t < 16; t++) pts.push(slerpUnit(a, b, t / 16).multiplyScalar(SPHERE_RADIUS));
    }
    camera.frustumLine.geometry.dispose();
    camera.frustumLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  }

  camera.patchMesh.visible = settings.showPatch && !!anchorQuat;
  if (settings.showPatch && anchorQuat) {
    const pos = camera.patchGeo.attributes.position as THREE.BufferAttribute;
    for (let j = 0; j <= PATCH_RES; j++) {
      const v = (j / PATCH_RES) * 2 - 1;
      for (let i = 0; i <= PATCH_RES; i++) {
        const u = (i / PATCH_RES) * 2 - 1;
        const d = cornerDir(u, v, anchorQuat, vFovRad, camera.aspect).multiplyScalar(SPHERE_RADIUS);
        const idx = j * (PATCH_RES + 1) + i;
        pos.setXYZ(idx, d.x, d.y, d.z);
      }
    }
    pos.needsUpdate = true;
    camera.patchGeo.computeVertexNormals();
  }
}

// ── Render ───────────────────────────────────────────────────────────────

function renderViewport(cam: THREE.Camera, x: number, y: number, w: number, h: number) {
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(scene, cam);
}

function layoutPip(camera: Camera) {
  const w = Math.min(320, innerWidth * 0.28);
  const h = w / camera.aspect;
  const margin = 20;
  camera.pipRect = { x: innerWidth - w - margin, y: innerHeight - h - margin, w, h };
  pipFrame.style.left = camera.pipRect.x + 'px';
  pipFrame.style.top = camera.pipRect.y + 'px';
  pipFrame.style.width = w + 'px';
  pipFrame.style.height = h + 'px';
  pipLabel.style.left = camera.pipRect.x + 'px';
  pipLabel.style.top = (camera.pipRect.y - 16) + 'px';
}

function resize() {
  renderer.setSize(innerWidth, innerHeight);
  viewerCam.aspect = innerWidth / innerHeight;
  viewerCam.updateProjectionMatrix();
  insideCam.aspect = innerWidth / innerHeight;
  insideCam.updateProjectionMatrix();
  const cam = activeCamera();
  if (cam) layoutPip(cam);
  gradientArrowCanvas.width = innerWidth;
  gradientArrowCanvas.height = innerHeight;
  gradientArrowCanvas.style.width = innerWidth + 'px';
  gradientArrowCanvas.style.height = innerHeight + 'px';
}
addEventListener('resize', resize);
resize();

function animate() {
  requestAnimationFrame(animate);

  // Cheap pass, every camera: keeps gizmo transforms/visibility and the
  // great-sphere overlays current for the always-on world view. Today
  // that's exactly one camera; structured as a loop now so Stage B (N
  // simulated cameras, all visible at once) is a small diff here, not
  // another rewrite of this function.
  for (const camera of cameras.values()) {
    if (isSimulated(camera)) updateGizmo(camera);
    // NOT updateGizmo()'s own returned vFovRad -- getAnalysisVFovRad is the
    // single source of truth every other analysis call site uses too (see
    // its own comment); for a simulated camera it reads back the same
    // focalMM-derived gizmoCam.fov updateGizmo just set, so this matches
    // exactly rather than duplicating the derivation.
    const vFovRad = getAnalysisVFovRad(camera);
    updateSphereOverlays(camera, vFovRad);

    if (isSimulated(camera)) {
      camera.gizmoBody.visible = globalState.mode === 'world' && camera.settings.showGizmoBody;
      camera.camHelper.visible = globalState.mode === 'world' && camera.settings.showFrustum;
    }
    updateRecoveredCamGizmo(camera);
    camera.recoveredFloorOverlay.visible = globalState.mode === 'world' && camera.settings.showRecoveredFloor && !!camera.lastPositionDecode;
  }
  floorMesh.visible = globalState.showFloor;

  // Expensive pass: only ever needed for the active camera (Through-Cam/
  // Projected-Cam/Inside-Sphere, and the PIP preview, only ever show one
  // camera at a time -- see this file's header comment).
  const active = activeCamera();
  const now = performance.now();
  if (active) {
    if (active.captureDirty && now - active.lastPreviewUpdate >= PREVIEW_UPDATE_INTERVAL_MS) {
      active.lastPreviewUpdate = now;
      active.captureDirty = false;
      if (isSimulated(active)) renderCamRT(active);
      updateDistortedPreview(active);
      if (globalState.mode === 'projected') buildProjectedTexture(active);
      if (globalState.mode === 'through') updateContaminationOverlays(active);
    }

    if (active.settings.axesAutoCapture && !active.axesCapturing && now - active.lastAxesCapture >= active.settings.axesCaptureIntervalMs) {
      active.lastAxesCapture = now;
      runAxesReconstruction(active);
    }
  }

  renderer.setViewport(0, 0, innerWidth, innerHeight);
  renderer.setScissorTest(false);
  renderer.setClearColor(0x0a0a0f, 1);
  renderer.clear();

  if (globalState.mode === 'world') {
    worldOrbit.update();
    renderViewport(viewerCam, 0, 0, innerWidth, innerHeight);
    if (active) renderPreviewViewport(active, active.pipRect.x, innerHeight - active.pipRect.y - active.pipRect.h, active.pipRect.w, active.pipRect.h);
  } else if (globalState.mode === 'through') {
    if (active) {
      const { x, y, w, h } = computeThroughRect(active);
      renderPreviewViewport(active, x, y, w, h);
      if (active.settings.showTrueContamination) renderTrueContamOverlay(active, x, y, w, h);
      if (active.settings.showReconstructedContamination) renderReconContamOverlay(active, x, y, w, h);
    }
  } else if (globalState.mode === 'projected') {
    if (active) {
      const availW = innerWidth - MARGINAL_THICKNESS;
      const availH = innerHeight - MARGINAL_THICKNESS;
      const winAspect = availW / availH;
      let w = availW, h = availH, x = 0, y = 0;
      if (winAspect > active.aspect) { w = availH * active.aspect; x = (availW - w) / 2; }
      else { h = availW / active.aspect; y = (availH - h) / 2; }
      renderProjectedViewport(active, x, innerHeight - y - h, w, h);
      drawMarginalLines(active, x, y, w, h);
      drawSampleLattice(active, x, y, w, h);
    }
  } else {
    // Inside-Sphere: only meaningful for a simulated camera's own ground-
    // truth pose (there is no equivalent for a physical camera -- see
    // updateSphereOverlays' header comment); falls back to the world
    // origin, looking however the free-look controls point, if the active
    // camera isn't simulated or doesn't exist yet.
    insideCam.position.copy(active && isSimulated(active) ? active.camPos : new THREE.Vector3());
    euler.set(insidePitch, insideYaw, 0);
    insideCam.quaternion.setFromEuler(euler);
    renderViewport(insideCam, 0, 0, innerWidth, innerHeight);
    if (active) renderPreviewViewport(active, active.pipRect.x, innerHeight - active.pipRect.y - active.pipRect.h, active.pipRect.w, active.pipRect.h);
  }
}

// Same persistence as every slider/checkbox -- only honor a saved value if
// it's still a real Mode.
const VALID_MODES: Mode[] = ['world', 'through', 'inside', 'projected'];
const savedMode = savedControls['mode'];
setMode(VALID_MODES.includes(savedMode as Mode) ? (savedMode as Mode) : 'world');
animate();

// ── Dev bridge ───────────────────────────────────────────────────────────
//
// Lets an external tool (scripts/dev-bridge/) send arbitrary JS to run
// directly in THIS module's scope — a literal `eval(code)` call written
// inline below, so it closes over every top-level const/let/function in
// this file (cameras, activeCamera, scene, ...) exactly as if typed into
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
      } else if (msg.type === 'realCapture' && msg.dataUrl) {
        // Broadcast from mobile-capture.html via the dev-bridge relay. Stage
        // A: routed to the active camera if it's already a PhysicalCamera;
        // otherwise there's nothing to ingest into (Stage C adds real
        // per-connection camera auto-creation) -- silently dropped, same as
        // today's app silently ignoring a realCapture message while
        // useRealCapture is off (ingestRealCapture was never called for it
        // either).
        const cam = activeCamera();
        if (cam && isPhysical(cam)) {
          ingestRealCapture(cam, msg.dataUrl).catch((e) => console.error('[realCapture] ingest failed:', e));
        }
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
