import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

// ── Math helpers ──────────────────────────────────────────────────────────────

function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++)
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

function computeHomography(src: number[][], dst: number[][]): number[][] {
  const A: number[][] = [], b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const [sx, sy] = src[i], [dx, dy] = dst[i];
    A.push([sx, sy, 1, 0, 0, 0, -dx*sx, -dx*sy]); b.push(dx);
    A.push([0, 0, 0, sx, sy, 1, -dy*sx, -dy*sy]); b.push(dy);
  }
  const h = solveLinear(A, b);
  return [[h[0],h[1],h[2]], [h[3],h[4],h[5]], [h[6],h[7],1]];
}

const norm3 = (v: number[]) => Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
const normalize3 = (v: number[]) => { const n = norm3(v); return [v[0]/n, v[1]/n, v[2]/n]; };
const cross3 = (a: number[], b: number[]) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];

interface Pose { r1: number[]; r2: number[]; r3: number[]; t: number[]; }

function extractPose(H: number[][], f: number, cx: number, cy: number): Pose {
  const Kinv = (j: number) => [
    (H[0][j] - cx * H[2][j]) / f,
    (H[1][j] - cy * H[2][j]) / f,
    H[2][j]
  ];
  const c0 = Kinv(0), c1 = Kinv(1), c2 = Kinv(2);
  const lambda = (norm3(c0) + norm3(c1)) / 2;
  const r1 = normalize3(c0);
  const r2 = normalize3(c1);
  const r3 = cross3(r1, r2);
  const t = c2.map(v => v / lambda);
  return { r1, r2, r3, t };
}

function project(X: number, Y: number, Z: number, { r1, r2, r3, t }: Pose, f: number, cx: number, cy: number) {
  const px = r1[0]*X + r2[0]*Y + r3[0]*Z + t[0];
  const py = r1[1]*X + r2[1]*Y + r3[1]*Z + t[1];
  const pz = r1[2]*X + r2[2]*Y + r3[2]*Z + t[2];
  return { x: f * px/pz + cx, y: f * py/pz + cy };
}

interface Corners { tl: { x: number; y: number }; tr: { x: number; y: number }; br: { x: number; y: number }; bl: { x: number; y: number }; }

function getPose({ tl, tr, br, bl }: Corners): Pose {
  const src = [[0,0],[1,0],[1,1],[0,1]];
  const dst = [[tl.x,tl.y],[tr.x,tr.y],[br.x,br.y],[bl.x,bl.y]];
  const H = computeHomography(src, dst);
  const f = (canvas.width + canvas.height) / 2;
  return extractPose(H, f, canvas.width/2, canvas.height/2);
}

// ── Three.js setup ────────────────────────────────────────────────────────────
//
// Camera space (our homography): X right, Y down, Z forward (into scene)
// Three.js world space:          X right, Y up,   Z toward viewer
//
// Conversion: camToThree(x,y,z) = (x, -y, -z)
// This is a proper rotation (det=1), so cross products are preserved.
//
// For a model sitting on the QR code, we map:
//   model local Y (up)  →  -r3 in camera space (toward camera)
//   model local X       →   r1 in camera space
//   model local Z       →   r2 in camera space

const threeRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, preserveDrawingBuffer: true });
threeRenderer.outputColorSpace = THREE.SRGBColorSpace;

const threeScene = new THREE.Scene();
const threeCamera = new THREE.PerspectiveCamera(60, 1, 0.001, 1000);
threeScene.add(new THREE.AmbientLight(0xffffff, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(1, 3, 2);
threeScene.add(sun);

const gltfLoader = new GLTFLoader();
type GlbEntry = 'loading' | 'error' | { model: THREE.Object3D; animations: THREE.AnimationClip[] };
const glbCache = new Map<string, GlbEntry>();      // url → 'loading' | { model, animations } | 'error'
const glbContainers = new Map<string, THREE.Object3D>(); // qr-id → THREE.Object3D in threeScene
const glbMixers = new Map<string, THREE.AnimationMixer>();     // qr-id → THREE.AnimationMixer
const clock = new THREE.Clock();

function loadGlb(url: string) {
  if (glbCache.has(url)) return;
  glbCache.set(url, 'loading');
  gltfLoader.load(url, gltf => {
    const model = gltf.scene;
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const scale = 1 / Math.max(...box.getSize(new THREE.Vector3()).toArray());
    model.scale.setScalar(scale);
    model.position.set(-center.x * scale, -box.min.y * scale, -center.z * scale);
    glbCache.set(url, { model, animations: gltf.animations });
  }, undefined, () => glbCache.set(url, 'error'));
}

// Build a Three.js Matrix4 placing the model at the QR pose.
// Container origin = QR center (0.5, 0.5, 0 in QR space).
// Columns: [camToThree(r1), camToThree(-r3), camToThree(r2), camToThree(t_center)]
function buildModelMatrix({ r1, r2, r3, t }: Pose) {
  const cx = r1[0]*0.5 + r2[0]*0.5 + t[0];
  const cy = r1[1]*0.5 + r2[1]*0.5 + t[1];
  const cz = r1[2]*0.5 + r2[2]*0.5 + t[2];
  const m = new THREE.Matrix4();
  // THREE.Matrix4.set() is row-major
  m.set(
     r1[0], -r3[0],  r2[0],  cx,
    -r1[1],  r3[1], -r2[1], -cy,
    -r1[2],  r3[2], -r2[2], -cz,
     0,      0,      0,      1
  );
  return m;
}

function isGlbUrl(text: string) {
  try { return new URL(text).pathname.toLowerCase().endsWith('.glb'); }
  catch { return false; }
}

// ── Camera ────────────────────────────────────────────────────────────────────

const canvas = document.getElementById('c') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const status = document.getElementById('status')!;
const video = document.createElement('video');
video.setAttribute('playsinline', '');

// ── Color from QR text ────────────────────────────────────────────────────────

const _cc = document.createElement('canvas');
_cc.width = _cc.height = 1;
const _cctx = _cc.getContext('2d')!;

function resolveColor(text: string): [number, number, number] | null {
  _cctx.fillStyle = 'rgb(1,2,3)';
  _cctx.fillStyle = text.trim().toLowerCase();
  _cctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = _cctx.getImageData(0, 0, 1, 1).data;
  if (r === 1 && g === 2 && b === 3) return null;
  return [r, g, b];
}

// ── Tracking ──────────────────────────────────────────────────────────────────

// ── 1€ filter ─────────────────────────────────────────────────────────────────
// Adapts cutoff frequency to signal speed: jitter (slow, high-freq) is filtered
// heavily; intentional motion (fast) passes through with minimal lag.
// minCutoff: base Hz at rest (lower = smoother but more lag when still)
// beta: how fast cutoff rises with speed (higher = less lag when moving)
class OneEuroFilter {
  minCutoff: number; beta: number; dCutoff: number;
  x: number | null; dx: number; t: number | null;
  constructor(minCutoff = 0.05, beta = 0.1, dCutoff = 1.0) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.x = null; this.dx = 0; this.t = null;
  }
  _alpha(cutoff: number, dt: number) { const r = 2 * Math.PI * cutoff * dt; return r / (r + 1); }
  filter(x: number, t: number) {
    const dt = this.t !== null ? (t - this.t) / 1000 : 1/30;
    this.t = t;
    if (this.x === null) { this.x = x; return x; }
    const dxRaw = (x - this.x) / dt;
    this.dx += this._alpha(this.dCutoff, dt) * (dxRaw - this.dx);
    this.x  += this._alpha(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (x - this.x);
    return this.x;
  }
}

function makeCornerFilters() {
  const f = () => ({ x: new OneEuroFilter(), y: new OneEuroFilter() });
  return { tl: f(), tr: f(), br: f(), bl: f() };
}

function applyFilters(filters: ReturnType<typeof makeCornerFilters>, { tl, tr, br, bl }: Corners): Corners {
  const t = performance.now();
  const pt = (f: { x: OneEuroFilter; y: OneEuroFilter }, p: { x: number; y: number }) => ({ x: f.x.filter(p.x, t), y: f.y.filter(p.y, t) });
  return { tl: pt(filters.tl, tl), tr: pt(filters.tr, tr),
           br: pt(filters.br, br), bl: pt(filters.bl, bl) };
}

interface TrackedState {
  smoothed: Corners;
  filters: ReturnType<typeof makeCornerFilters>;
  missed: number;
  rgb: [number, number, number];
  glbUrl: string | null;
}

const HOLD_FRAMES = 20;
const tracked = new Map<string, TrackedState>();

function updateTracked(barcodes: any[]) {
  const seen = new Set<string>();
  for (const code of barcodes) {
    const id = code.rawValue;
    seen.add(id);
    const [tl, tr, br, bl] = code.cornerPoints;
    const raw: Corners = { tl, tr, br, bl };
    const prev = tracked.get(id);
    if (isGlbUrl(id)) loadGlb(id);
    const filters = prev ? prev.filters : makeCornerFilters();
    tracked.set(id, {
      smoothed: applyFilters(filters, raw),
      filters,
      missed: 0,
      rgb: prev ? prev.rgb : (resolveColor(id) ?? [0, 220, 255]),
      glbUrl: isGlbUrl(id) ? id : null,
    });
  }
  for (const [id, state] of tracked) {
    if (!seen.has(id)) {
      state.missed++;
      if (state.missed > HOLD_FRAMES) {
        const obj = glbContainers.get(id);
        if (obj) { threeScene.remove(obj); glbContainers.delete(id); glbMixers.delete(id); }
        tracked.delete(id);
      }
    }
  }
}

// ── Detection loop ────────────────────────────────────────────────────────────

const _scanCanvas = document.createElement('canvas');
const _scanCtx = _scanCanvas.getContext('2d')!;

async function detectAll(detector: any) {
  _scanCanvas.width = video.videoWidth;
  _scanCanvas.height = video.videoHeight;
  _scanCtx.drawImage(video, 0, 0);
  const results = [];
  for (let i = 0; i < 8; i++) {
    const found = await detector.detect(_scanCanvas);
    if (!found.length) break;
    results.push(...found);
    _scanCtx.fillStyle = '#000';
    for (const code of found) {
      const { x, y, width, height } = code.boundingBox;
      _scanCtx.fillRect(x - 20, y - 20, width + 40, height + 40);
    }
  }
  return results;
}

async function detectionLoop() {
  if (!('BarcodeDetector' in window)) {
    status.textContent = 'BarcodeDetector not supported — try Chrome';
    return;
  }
  const detector = new (window as any).BarcodeDetector({ formats: ['qr_code'] });
  while (true) {
    if (video.readyState >= 2) {
      try { updateTracked(await detectAll(detector)); } catch (_) {}
    }
    await new Promise(r => setTimeout(r, 0));
  }
}

// ── Render loop ───────────────────────────────────────────────────────────────

function render() {
  requestAnimationFrame(render);
  if (video.readyState < 2) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // CSS cover: scale up until both dimensions fill the viewport, clip the overflow
  const scaleToFit = Math.max(window.innerWidth / canvas.width, window.innerHeight / canvas.height);
  canvas.style.width  = (canvas.width  * scaleToFit) + 'px';
  canvas.style.height = (canvas.height * scaleToFit) + 'px';

  ctx.drawImage(video, 0, 0);

  const f = (canvas.width + canvas.height) / 2;
  const cx = canvas.width / 2, cy = canvas.height / 2;
  let hasGlb = false;

  for (const [id, state] of tracked) {
    const pose = getPose(state.smoothed);

    if (state.glbUrl) {
      hasGlb = true;
      const cached = glbCache.get(state.glbUrl);
      if (cached && cached !== 'loading' && cached !== 'error') {
        let container = glbContainers.get(id);
        if (!container) {
          container = new THREE.Object3D();
          container.matrixAutoUpdate = false;
          const instance = SkeletonUtils.clone(cached.model);
          container.add(instance);
          threeScene.add(container);
          glbContainers.set(id, container);
          if (cached.animations.length > 0) {
            const mixer = new THREE.AnimationMixer(instance);
            cached.animations.forEach(clip => mixer.clipAction(clip).play());
            glbMixers.set(id, mixer);
          }
        }
        container.matrix.copy(buildModelMatrix(pose));
        container.matrixWorldNeedsUpdate = true;
      }
    } else {
      drawCube(pose, f, cx, cy, state.rgb);
    }
  }

  if (hasGlb) {
    const delta = clock.getDelta();
    for (const mixer of glbMixers.values()) mixer.update(delta);
    threeRenderer.setSize(canvas.width, canvas.height);
    threeCamera.fov = 2 * Math.atan(canvas.height / (2 * f)) * 180 / Math.PI;
    threeCamera.aspect = canvas.width / canvas.height;
    threeCamera.updateProjectionMatrix();
    threeRenderer.render(threeScene, threeCamera);
    ctx.drawImage(threeRenderer.domElement, 0, 0);
  }

  status.textContent = tracked.size > 0
    ? `${tracked.size} QR code(s) tracked`
    : 'Show QR codes to the camera';
}

// ── Camera selection (front/rear) ────────────────────────────────────────────
//
// Note: the mirror is a pure CSS display transform on <canvas id="c">. The QR
// detection and pose math above always operate on the raw, unmirrored video
// frame drawn into the canvas 2D context, so switching cameras / toggling the
// mirror class never needs any changes to the homography or projection math —
// video pixels and the AR overlay are drawn into the same raster before any
// mirroring happens, so they stay aligned regardless of facingMode.

const switchCamBtn = document.getElementById('switchCam') as HTMLButtonElement;
let currentStream: MediaStream | null = null;
let currentFacing = 'environment';

async function startCamera(desiredFacing: string) {
  const newStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: desiredFacing } }
  });

  if (currentStream) currentStream.getTracks().forEach(t => t.stop());
  currentStream = newStream;
  video.srcObject = currentStream;
  await video.play();

  // Devices without a rear camera (e.g. laptop webcams) often don't report
  // `facingMode` at all — treat that as front-facing, matching prior behavior.
  const settings = currentStream.getVideoTracks()[0].getSettings();
  currentFacing = settings.facingMode || 'user';
  canvas.classList.toggle('mirror', currentFacing === 'user');
}

switchCamBtn.addEventListener('click', async () => {
  const next = currentFacing === 'user' ? 'environment' : 'user';
  switchCamBtn.disabled = true;
  try { await startCamera(next); }
  catch (e: any) { status.textContent = 'Camera error: ' + e.message; }
  finally { switchCamBtn.disabled = false; }
});

startCamera('environment')
  .then(() => { render(); detectionLoop(); })
  .catch((e: any) => { status.textContent = 'Camera error: ' + e.message; });

// ── 2D cube drawing ───────────────────────────────────────────────────────────

function drawCube(pose: Pose, f: number, cx: number, cy: number, [r, g, b]: [number, number, number]) {
  const p = (x: number, y: number, z: number) => project(x, y, z, pose, f, cx, cy);
  const base = [p(0,0,0), p(1,0,0), p(1,1,0), p(0,1,0)];
  const top  = [p(0,0,-1), p(1,0,-1), p(1,1,-1), p(0,1,-1)];

  ctx.strokeStyle = `rgb(${r},${g},${b})`;
  ctx.lineWidth = 2;
  for (let i = 0; i < 4; i++)
    face([base[i], base[(i+1)%4], top[(i+1)%4], top[i]], `rgba(${r},${g},${b},0.12)`);
  face(top, `rgba(${r},${g},${b},0.3)`);
}

function face(pts: { x: number; y: number }[], fill: string) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.stroke();
}
