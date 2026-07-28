// Phone-side capture page: live camera viewfinder + hardware zoom (ported
// from src/main.ts's tracker page, same architecture) with a shutter button
// that sends the current frame to any open Sphere Lab tab over the dev
// bridge relay -- see scripts/dev-bridge/server.js's 'realCapture' handling
// and sphereLab.ts's ingestRealCapture. Doesn't run any of the actual
// analysis pipeline itself; this page's only job is getting a real photo
// off the phone and onto the laptop.
//
// Two capture modes: single (tap the shutter each time, as before) and
// video (streams frames automatically). Sphere Lab's own reconstruction
// pass is slow enough that it needs to say something about pacing -- see
// the "Capture mode + readiness" section below. Exactly what that signal
// MEANS depends on Sphere Lab's own useCapturePipelining toggle, relayed
// alongside it: off, "not ready" blocks sending outright (old strict
// one-frame-in-flight handshake); on, it's purely a status indicator for
// the shutter button's yellow "working" state -- Sphere Lab's mailbox can
// always take a fresher frame, busy or not, so sending is never blocked.

import * as THREE from 'three';
import { toGrayscale } from './decode.ts';
import { globalState } from './sphereLab/state.ts';
import { GRID_STEP } from './sphereLab/constants.ts';
import { C, R, rebuildFloorPatternData } from './sphereLab/floorPattern.ts';
import { getAnalysisVFovRad } from './sphereLab/math/geometry.ts';
import { flipRowsF64 } from './sphereLab/pipeline/distortion.ts';
import { computeGradient2x2Field } from './sphereLab/pipeline/gradientField.ts';
import { computeLsdRectanglesAuto, LsdRectangle } from './sphereLab/pipeline/lsdSegments.ts';
import { computePoseFromCapture, PoseComputeState } from './sphereLab/pipeline/poseCompute.ts';

const video = document.getElementById('v') as HTMLVideoElement;
const captureCanvas = document.getElementById('captureCanvas') as HTMLCanvasElement;
const captureCtx = captureCanvas.getContext('2d')!;
const camStatus = document.getElementById('camStatus')!;
const relayStatus = document.getElementById('relayStatus')!;
const zoomSlider = document.getElementById('zoom') as HTMLInputElement;
const resSlider = document.getElementById('res') as HTMLInputElement;
const resValue = document.getElementById('resValue')!;
const switchCamBtn = document.getElementById('switchCam') as HTMLButtonElement;
const shutterBtn = document.getElementById('shutter') as HTMLButtonElement;
const modeSingleBtn = document.getElementById('modeSingleBtn') as HTMLButtonElement;
const modeVideoBtn = document.getElementById('modeVideoBtn') as HTMLButtonElement;
const computeOnDeviceCheckbox = document.getElementById('computeOnDevice') as HTMLInputElement;
const sendDebugInfoCheckbox = document.getElementById('sendDebugInfo') as HTMLInputElement;
const sendCapturedImageCheckbox = document.getElementById('sendCapturedImage') as HTMLInputElement;
const poseReadoutEl = document.getElementById('poseReadout')!;
const arCanvas = document.getElementById('arCanvas') as HTMLCanvasElement;
const arOverlayCheckbox = document.getElementById('arOverlayEnabled') as HTMLInputElement;

// ── AR overlay: the known board sits FIXED in world space, only the camera
// moves ──────────────────────────────────────────────────────────────────
//
// The whole point of pose recovery is an ABSOLUTE camera position/
// orientation relative to the known, fixed De Bruijn board -- see this
// session's chat. The board itself never moves: it's the exact same static
// C*GRID_STEP x R*GRID_STEP rectangle at world ORIGIN that scene/floor.ts's
// floorMesh already is on the desktop (PlaneGeometry(C*GRID_STEP,
// R*GRID_STEP), rotation.x = -PI/2, no position offset). So this scene's
// plane/cube are built ONCE (and rebuilt only when boardSize itself changes,
// see applySettingsSync) and never repositioned; the only thing that updates
// per-capture is the AR CAMERA -- placed at the recovered camPos/
// recoveredCamQuat, with the SAME vertical FOV/aspect the pose pipeline
// itself used for ray-casting (getAnalysisVFovRad). That directly
// reproduces "standard AR": look straight through the phone into the
// reconstructed scene, from wherever the recovered pose says the phone
// actually is.
//
// Independent of computeOnDevice (its own checkbox, per an explicit ask --
// see this session's chat): a camera pose can arrive from EITHER of two
// sources -- this device's own computePoseFromCapture
// (captureComputeAndSendPose below), or a 'poseSync' message (see
// connectRelay's message handler) mirroring one BACK down from the desktop
// after IT finishes a desktop-compute capture (see devBridge/client.ts's
// pushPoseSync, sent exactly symmetric to this page's own poseResult send
// upward). Both normalize to the same ARCameraPose shape before reaching
// updateARCamera, which doesn't care which source produced it.
interface ARCameraPose {
  camPos: THREE.Vector3; recoveredCamQuat: THREE.Quaternion; aspect: number; fovDeg: number;
}

let arOverlayEnabled = false;
function setAROverlayEnabled(enabled: boolean) {
  arOverlayEnabled = enabled;
  arCanvas.classList.toggle('visible', enabled);
}
arOverlayCheckbox.addEventListener('change', () => setAROverlayEnabled(arOverlayCheckbox.checked));

const arScene = new THREE.Scene();
const arCamera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
arScene.add(arCamera);

// Same lighting rig as the desktop's own scene/renderer.ts -- only the cube
// actually needs it (a lit material, MeshStandardMaterial below, so its
// faces shade by angle instead of reading as a flat unlit silhouette); the
// plane stays MeshBasicMaterial (unlit) since it's a translucent floor
// decal, not a 3D object anyone needs to read shape from.
arScene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.2));
const arSun = new THREE.DirectionalLight(0xffffff, 0.8);
arSun.position.set(1, 2, 1);
arScene.add(arSun);

const arPlaneMat = new THREE.MeshBasicMaterial({ color: 0x33aaff, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false });
const arPlane = new THREE.Mesh(new THREE.PlaneGeometry(C * GRID_STEP, R * GRID_STEP), arPlaneMat);
arPlane.rotation.x = -Math.PI / 2; // flat in world XZ, same convention as scene/floor.ts's floorMesh
arPlane.visible = false; // no fix yet -- see updateARCamera
arScene.add(arPlane);

// 10 board cells per side -- a landmark big enough to spot from across the
// board. At the board's own center (world origin), resting on top of the
// plane.
const AR_CUBE_SIZE = GRID_STEP * 10;
const arCubeMat = new THREE.MeshStandardMaterial({ color: 0xff2222, roughness: 0.6 });
const arCube = new THREE.Mesh(new THREE.BoxGeometry(AR_CUBE_SIZE, AR_CUBE_SIZE, AR_CUBE_SIZE), arCubeMat);
arCube.position.set(0, AR_CUBE_SIZE / 2, 0);
arCube.visible = false; // no fix yet -- see updateARCamera
arScene.add(arCube);

// Rebuilds the board plane at a new size -- mirrors scene/floor.ts's own
// rebuildFloorPattern (position/rotation never change, only C/R do) --
// called from applySettingsSync whenever boardSize actually changes.
function rebuildARBoardGeometry() {
  arPlane.geometry.dispose();
  arPlane.geometry = new THREE.PlaneGeometry(C * GRID_STEP, R * GRID_STEP);
}

const arRenderer = new THREE.WebGLRenderer({ canvas: arCanvas, alpha: true, antialias: true });
arRenderer.setClearColor(0x000000, 0);
arRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

// Keeps the canvas's on-screen box pixel-matched to the video's own
// rendered box (video's `max-width/max-height: 100vw/100vh` can letterbox
// it below the full viewport) -- an AR overlay that's the wrong size or
// off-center would defeat the whole point of lining up with the live image.
function resizeARCanvas() {
  const w = video.clientWidth, h = video.clientHeight;
  if (w === 0 || h === 0) return;
  arRenderer.setSize(w, h);
}
new ResizeObserver(resizeARCanvas).observe(video);

// Only ever moves the CAMERA -- see this section's header comment on why
// the board itself (arPlane/arCube) is static. No fix (pose === null) hides
// the render output rather than leaving a stale camera pose on screen from
// a previous fix.
function updateARCamera(pose: ARCameraPose | null) {
  arPlane.visible = !!pose;
  arCube.visible = !!pose;
  if (!pose) return;
  arCamera.position.copy(pose.camPos);
  arCamera.quaternion.copy(pose.recoveredCamQuat);
  arCamera.fov = pose.fovDeg;
  arCamera.aspect = pose.aspect;
  arCamera.updateProjectionMatrix();
}

// Resolves a local computePoseFromCapture result (device-compute mode) into
// the same ARCameraPose shape a poseSync message already arrives in -- see
// updateARCamera's own comment on why it stays agnostic to the source.
function buildLocalARCameraPose(state: PoseComputeState): ARCameraPose | null {
  const decode = state.lastPositionDecode;
  if (!decode) return null;
  return {
    camPos: decode.camPos, recoveredCamQuat: decode.recoveredCamQuat,
    aspect: state.aspect, fovDeg: THREE.MathUtils.radToDeg(getAnalysisVFovRad(state)),
  };
}

// Deserializes a poseSync message's `fix` payload (see devBridge/client.ts's
// pushPoseSync) -- pure array->THREE-object reconstruction, no math.
function parseRemoteARCameraPose(fix: {
  camPos: number[]; recoveredCamQuat: number[]; aspect: number; vFovDeg: number;
} | null): ARCameraPose | null {
  if (!fix) return null;
  return {
    camPos: new THREE.Vector3().fromArray(fix.camPos),
    recoveredCamQuat: new THREE.Quaternion().fromArray(fix.recoveredCamQuat),
    aspect: fix.aspect, fovDeg: fix.vFovDeg,
  };
}

function arRenderLoop() {
  requestAnimationFrame(arRenderLoop);
  // Skips the render entirely while hidden -- this page is already CPU/GPU
  // constrained by continuous pose recovery in video mode, no reason to also
  // pay for a WebGL draw call nobody can see.
  if (arOverlayEnabled) arRenderer.render(arScene, arCamera);
}
arRenderLoop();

// ── Camera + zoom (ported from src/main.ts:33-99, same reasoning) ─────────

let currentStream: MediaStream | null = null;
let currentFacing = 'environment';
let zoomMin = 1, zoomMax = 1;

function sliderToZoom(t: number): number {
  return zoomMin * Math.pow(zoomMax / zoomMin, t);
}

function setupZoomControl() {
  const track = currentStream?.getVideoTracks()[0];
  let caps: any = null;
  try { caps = track && 'getCapabilities' in track ? (track as any).getCapabilities() : null; }
  catch { caps = null; }

  if (caps && caps.zoom && caps.zoom.min > 0 && caps.zoom.max > caps.zoom.min) {
    zoomSlider.disabled = false;
    zoomMin = caps.zoom.min;
    zoomMax = caps.zoom.max;
    zoomSlider.min = '0';
    zoomSlider.max = '1';
    zoomSlider.step = '0.001';
    zoomSlider.value = '0';
    track!.applyConstraints({ advanced: [{ zoom: zoomMin } as any] }).catch(() => {});
  } else {
    zoomSlider.disabled = true;
  }
}

zoomSlider.addEventListener('input', () => {
  const track = currentStream?.getVideoTracks()[0];
  const zoom = sliderToZoom(parseFloat(zoomSlider.value));
  track?.applyConstraints({ advanced: [{ zoom } as any] }).catch(() => {});
});

// ── Sent-image resolution ───────────────────────────────────────────────
//
// Controls the SENT frame's long-edge pixel size (captureAndSendFrame's own
// canvas downscale, replacing what used to be the fixed MAX_DIM=1600
// constant) -- not a camera hardware constraint, so this is a plain canvas
// resize with the short edge always scaled by the same factor to preserve
// the native aspect ratio, same as the old fixed-MAX_DIM math already did.
//
// The two ends of the slider aren't equally well-defined:
//  - HIGH end has a real technical ceiling: the stream's own native
//    resolution (video.videoWidth/Height once getUserMedia has negotiated
//    it, via the same "request an unreachably high `ideal`" trick
//    startCamera already uses) -- upscaling past that wouldn't add any
//    actual detail, so it's a genuine max, not a guess.
//  - LOW end has no such floor -- a canvas resize could go arbitrarily low
//    (down to 1x1). RES_MIN_DIM_FLOOR below is a judgment call, not a
//    hardware limit: 512px picks up the SAME number this app's own
//    simulated-camera default analysis resolution already uses
//    (camera/settings.ts's viewportW default) -- since a physical camera's
//    whatever arrives becomes ITS analysis resolution directly (see
//    pipeline/capture.ts's ingestRealCapture -- no further downsample the
//    way simulated captures get), that's the same "how few pixels does
//    this reconstruction pipeline actually need" question the app already
//    answered once, reused here rather than picking a fresh arbitrary
//    number.
const RES_MIN_DIM_FLOOR = 512;
const RES_DEFAULT_DIM = 1600; // matches the old fixed MAX_DIM, kept as the default so behavior doesn't silently change until the slider is touched
let resMinDim = RES_MIN_DIM_FLOOR, resMaxDim = RES_MIN_DIM_FLOOR;
let targetLongEdge = RES_DEFAULT_DIM;
let nativeShortOverLong = 1; // aspect ratio of the current stream, short edge / long edge

function sliderToDim(t: number): number {
  return resMinDim * Math.pow(resMaxDim / resMinDim, t);
}
function dimToSlider(dim: number): number {
  if (resMaxDim <= resMinDim) return 0;
  return Math.min(1, Math.max(0, Math.log(dim / resMinDim) / Math.log(resMaxDim / resMinDim)));
}
function updateResLabel() {
  const shortEdge = Math.round(targetLongEdge * nativeShortOverLong);
  resValue.textContent = `${targetLongEdge}×${shortEdge}`;
}

function setupResolutionControl() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) { resSlider.disabled = true; return; }
  const nativeLongEdge = Math.max(vw, vh);
  nativeShortOverLong = Math.min(vw, vh) / nativeLongEdge;
  resMaxDim = nativeLongEdge;
  // A device/webcam whose native resolution is already below the floor --
  // just pin the slider to that single value instead of presenting a
  // (min > max) range.
  resMinDim = Math.min(RES_MIN_DIM_FLOOR, resMaxDim);
  if (resMaxDim <= resMinDim) {
    resSlider.disabled = true;
    targetLongEdge = resMaxDim;
    updateResLabel();
    return;
  }
  resSlider.disabled = false;
  targetLongEdge = Math.min(Math.max(targetLongEdge, resMinDim), resMaxDim);
  resSlider.value = String(dimToSlider(targetLongEdge));
  updateResLabel();
}

resSlider.addEventListener('input', () => {
  targetLongEdge = Math.round(sliderToDim(parseFloat(resSlider.value)));
  updateResLabel();
});

// ── On-device compute: settings mirror ──────────────────────────────────
//
// globalState here is THIS page's own module instance (mobile-capture.html
// is a separate Vite entry point from sphere-lab.html -- a totally separate
// JS realm/module graph, not shared memory), so mutating it locally from a
// settingsSync message is safe -- see this session's on-device-pose-recovery
// plan. Source of truth stays the desktop's own sliders; this just mirrors
// whatever it last pushed. Defaults below match camera/settings.ts's own
// createDefaultCommonSettings so a device-compute cycle run before the
// first settingsSync arrives (e.g. right after toggling the checkbox on
// before the desktop's on-connect push lands) still produces a sane result.
let cameraSettings: PoseComputeState['settings'] = {
  horizFovDeg: 65, weightSharpenPower: 4, gridPeriodPhaseGapLowerBound: 0.005, minGrazingCos: 0.15,
  lsdToleranceDeg: 22.5, lsdRhoNoiseThreshold: 4, lsdMagnitudeBuckets: 1024, lsdNfaEpsilon: 1,
  lsdNfaTestExponent: 5, lsdMaxRetries: 2, lsdRetryToleranceFactor: 0.5, lsdRetryShrinkFraction: 0.2,
  lsdMergeMinSimilarity: 0.9, lsdJoinSteps: 0, lsdMinLengthPx: 3, lsdMaxTravelFactor: 1,
};
// Tracks the last boardSize a settingsSync actually applied, so
// rebuildFloorPatternData (which rebuilds the whole De Bruijn lookup table --
// not cheap) only runs when boardSize genuinely changed, not on every
// unrelated slider nudge riding along in the same message -- mirrors how
// only the desktop's own board-size slider handler calls the THREE-side
// rebuildFloorPattern today (see ui/cameraPanel.ts's 'boardSize' binding).
let knownBoardSize: number | null = null;

function applySettingsSync(msg: any) {
  if (msg.globalState) {
    globalState.useGPUFit = !!msg.globalState.useGPUFit;
    globalState.useGPUGradient = !!msg.globalState.useGPUGradient;
    globalState.useGPULsdFit = !!msg.globalState.useGPULsdFit;
    globalState.useGPUDecode = !!msg.globalState.useGPUDecode;
    const boardSize = msg.globalState.boardSize;
    if (typeof boardSize === 'number' && boardSize !== knownBoardSize) {
      knownBoardSize = boardSize;
      rebuildFloorPatternData(boardSize);
      rebuildARBoardGeometry();
    }
  }
  if (msg.cameraSettings) cameraSettings = { ...cameraSettings, ...msg.cameraSettings };
}

// ── Producer frame-rate diagnostics ─────────────────────────────────────
//
// Answers "is the camera hardware itself the bottleneck" independently of
// anything the capture/relay round trip does. requestVideoFrameCallback
// fires once per actually-decoded video frame (unlike requestAnimationFrame,
// which just ticks at the display's own rate regardless of whether a new
// camera frame has actually arrived), and its metadata.mediaTime is the
// frame's own timestamp on the media timeline -- so diffing consecutive
// mediaTimes measures real delivered-frame spacing, not callback jitter.
let nominalFrameRate: number | null = null;
let lastMediaTime: number | null = null;
let frameIntervalsMs: number[] = [];

function onVideoFrame(_now: number, metadata: { mediaTime: number }) {
  if (lastMediaTime !== null) {
    const dt = (metadata.mediaTime - lastMediaTime) * 1000;
    if (dt > 0) frameIntervalsMs.push(dt);
  }
  lastMediaTime = metadata.mediaTime;
  rvfc?.(onVideoFrame);
}
// Cast: requestVideoFrameCallback isn't in every TS DOM lib version yet;
// feature-detected at runtime regardless, so a missing type just means no
// diagnostics on a browser too old to have shipped it.
const rvfc = (video as any).requestVideoFrameCallback?.bind(video) as
  ((cb: (now: number, metadata: { mediaTime: number }) => void) => void) | undefined;
rvfc?.(onVideoFrame);

// Loop-tick accounting -- answers "is the video loop itself even running as
// often as expected on this phone" independently of network/encode cost
// entirely. loopTicks counts every requestAnimationFrame callback
// regardless of outcome; if loopTicks over a 2s window is well under
// ~120 (60Hz), requestAnimationFrame itself is being starved (backgrounded
// tab, thermal throttling, main-thread contention) -- a phone-side
// scheduling problem, not network. Of the ticks that DO run,
// backpressureBlockedTicks/readinessBlockedTicks say WHY a send was
// skipped: backpressure means the previous frame is still physically
// draining over the network (see canSend()'s own comment), readiness means
// Sphere Lab itself said not to send yet (shouldn't happen much when
// pipelined).
let loopTicks = 0;
let backpressureBlockedTicks = 0;
let readinessBlockedTicks = 0;
let sendsAttempted = 0;

// Flushes whatever's accumulated every couple seconds -- see server.js's
// frameStats broadcast and devBridge/client.ts's handler for where this
// ends up (readable via dev-bridge eval as camera.lastFrameStats).
setInterval(() => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const stats: any = { type: 'frameStats', nominalFrameRate, loopTicks, backpressureBlockedTicks, readinessBlockedTicks, sendsAttempted };
  if (frameIntervalsMs.length > 0) {
    stats.avgIntervalMs = frameIntervalsMs.reduce((a, b) => a + b, 0) / frameIntervalsMs.length;
    stats.maxIntervalMs = Math.max(...frameIntervalsMs);
    stats.sampleCount = frameIntervalsMs.length;
  }
  frameIntervalsMs = [];
  loopTicks = 0; backpressureBlockedTicks = 0; readinessBlockedTicks = 0; sendsAttempted = 0;
  ws.send(JSON.stringify(stats));
}, 2000);

async function startCamera(desiredFacing: string) {
  // width/height `ideal` far above any real sensor -- getUserMedia treats
  // `ideal` as "get as close as you can", so an unreachably high target
  // just resolves to whatever the device's actual max is, without needing
  // to query getCapabilities() (unavailable pre-stream anyway) or hardcode
  // a specific resolution that undershoots newer/higher-res phones.
  const newStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: desiredFacing }, width: { ideal: 7680 }, height: { ideal: 4320 } },
  });
  if (currentStream) currentStream.getTracks().forEach((t) => t.stop());
  currentStream = newStream;
  video.srcObject = currentStream;
  await video.play();
  const settings = currentStream.getVideoTracks()[0].getSettings();
  currentFacing = settings.facingMode || 'environment';
  video.classList.toggle('mirror', currentFacing === 'user');
  arCanvas.classList.toggle('mirror', currentFacing === 'user');
  setupZoomControl();
  setupResolutionControl();
  resizeARCanvas();
  camStatus.textContent = `${currentFacing} camera, ${settings.width}x${settings.height}`;
  // The camera hardware's OWN nominal capture rate -- distinct from how
  // often WE encode/send (see the frameStats reporting below), and a real
  // alternative explanation raised for video mode's idle gap: auto-exposure
  // in low light can drop a phone's actually-delivered frame rate well
  // below its nominal one, independent of anything the round-trip does.
  nominalFrameRate = settings.frameRate ?? null;
}

switchCamBtn.addEventListener('click', async () => {
  const next = currentFacing === 'user' ? 'environment' : 'user';
  switchCamBtn.disabled = true;
  try { await startCamera(next); }
  catch (e: any) { camStatus.textContent = 'camera error: ' + e.message; }
  finally { switchCamBtn.disabled = false; }
});

startCamera('environment').catch((e: any) => {
  camStatus.textContent = 'camera error: ' + e.message;
});

// ── Relay connection ────────────────────────────────────────────────────
//
// Rides the SAME https origin the page itself was loaded from, via vite's
// /dev-bridge websocket proxy (see vite.config.ts) -- a page loaded over
// https (required here for getUserMedia) can't open a plain insecure ws://
// connection to a non-localhost host, so this can't just point at the
// dev-bridge's own ws://<lan-ip>:8787 directly the way a laptop-local tab
// can. Works identically whether this page happens to be opened via the LAN
// IP or localhost.
let ws: WebSocket | null = null;
let reconnectTimer: number | undefined;

function setRelayStatus(text: string, down: boolean) {
  relayStatus.textContent = `relay: ${text}`;
  relayStatus.classList.toggle('down', down);
}

function scheduleReconnect() {
  ws = null;
  setRelayStatus('reconnecting…', true);
  clearTimeout(reconnectTimer);
  reconnectTimer = window.setTimeout(connectRelay, 2000);
}

// ── Capture mode + readiness ────────────────────────────────────────────
//
// Sphere Lab's reconstruction pass is slow enough that it needs to tell us
// when it's actually done with the last frame (see main.ts's animate loop,
// which watches axesCapturing and pushes captureReady over this same
// socket) -- both photo and video mode respect it, not just video, per an
// explicit ask: the shutter turns yellow and single-mode taps become a
// no-op whenever Sphere Lab isn't ready, exactly like video mode already
// has to gate its automatic sends.
let captureMode: 'single' | 'video' = 'single';
let sphereLabReady = true;
let readyTimeoutTimer: number | undefined;
// Mirrors Sphere Lab's globalState.useCapturePipelining, riding along on
// every captureReady message (see server.js's relay). Conservative default
// (false, i.e. old strict handshake) until the first real message arrives,
// since we don't know the desktop's setting yet at connect time. When true,
// sphereLabReady/the yellow "notReady" state is purely informational --
// Sphere Lab's mailbox absorbs a send at any time, so it no longer blocks
// captureAndSendFrame the way it still does when this is false.
let sphereLabPipelined = false;

function setCaptureMode(mode: 'single' | 'video') {
  captureMode = mode;
  modeSingleBtn.classList.toggle('active', mode === 'single');
  modeVideoBtn.classList.toggle('active', mode === 'video');
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'captureMode', mode }));
  // Entering video mode while device-compute is already on (re)starts the
  // self-paced compute loop -- see devicePoseLoop's own comment.
  if (mode === 'video' && computeOnDevice) devicePoseLoop();
}
modeSingleBtn.addEventListener('click', () => setCaptureMode('single'));
modeVideoBtn.addEventListener('click', () => setCaptureMode('video'));

// ── Compute pose on this device -- orthogonal to the single/video axis
// above (see this session's on-device-pose-recovery plan): when on, the
// phone runs the same pose-recovery math locally (pipeline/poseCompute.ts's
// computePoseFromCapture) and sends only the recovered pose, no image
// bytes. Still respects captureMode: single taps one compute+send cycle,
// video runs devicePoseLoop continuously.
let computeOnDevice = false;
function setComputeMode(onDevice: boolean) {
  computeOnDevice = onDevice;
  poseReadoutEl.classList.toggle('visible', onDevice);
  // AR overlay visibility is its OWN checkbox now (arOverlayEnabled) --
  // independent of computeOnDevice, since a poseSync message (see
  // connectRelay's message handler) can feed it from the desktop instead
  // when this is off. Switching modes doesn't clear the currently-shown
  // pose either -- whichever source (local or synced) last updated it stays
  // on screen until the OTHER source's next update, exactly like switching
  // captureMode doesn't blank the video feed.
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'computeMode', mode: onDevice ? 'device' : 'desktop' }));
  if (onDevice && captureMode === 'video') devicePoseLoop();
}
computeOnDeviceCheckbox.addEventListener('change', () => setComputeMode(computeOnDeviceCheckbox.checked));

// Debug-only extras riding on top of a device-compute poseResult message --
// both OFF by default and independently toggleable (see this session's chat:
// explicitly asked for as their OWN pair of switches, not bundled into
// computeOnDevice, specifically so they can stay off to avoid any perf
// impact on the actual compute-speed measurement computeOnDevice exists to
// show). sendCapturedImage pays for a JPEG encode (the same toDataURL cost
// the desktop-compute streaming path always pays); sendDebugInfo pays for
// extra serialization of pipeline intermediates -- neither happens at all
// unless explicitly turned on.
let sendDebugInfo = false;
let sendCapturedImage = false;
sendDebugInfoCheckbox.addEventListener('change', () => { sendDebugInfo = sendDebugInfoCheckbox.checked; });
sendCapturedImageCheckbox.addEventListener('change', () => { sendCapturedImage = sendCapturedImageCheckbox.checked; });

// If nothing ever answers (no Sphere Lab tab open, or one that closed mid-
// crunch) don't stay stuck yellow/stalled forever -- fall back to assuming
// ready after a while. A real captureReady message always overrides this.
const READY_TIMEOUT_MS = 8000;
function setReady(ready: boolean) {
  sphereLabReady = ready;
  shutterBtn.classList.toggle('notReady', !ready);
  clearTimeout(readyTimeoutTimer);
  if (!ready) readyTimeoutTimer = window.setTimeout(() => setReady(true), READY_TIMEOUT_MS);
}

function connectRelay() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  try { ws = new WebSocket(`${proto}//${location.host}/dev-bridge`); }
  catch { scheduleReconnect(); return; }

  ws.addEventListener('open', () => {
    ws!.send(JSON.stringify({ role: 'capture' }));
    // A reconnect gets a brand-new captureId server-side (see server.js),
    // which Sphere Lab will treat as a fresh phone -- re-announce whatever
    // mode was already selected, and drop any stale not-ready state from
    // before the drop, since it belonged to the OLD captureId.
    ws!.send(JSON.stringify({ type: 'captureMode', mode: captureMode }));
    ws!.send(JSON.stringify({ type: 'computeMode', mode: computeOnDevice ? 'device' : 'desktop' }));
    setReady(true);
    setRelayStatus('connected', false);
  });
  ws.addEventListener('close', scheduleReconnect);
  ws.addEventListener('error', () => {});
  ws.addEventListener('message', (ev) => {
    let msg: any;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'captureReady') {
      sphereLabPipelined = !!msg.pipelined;
      setReady(!!msg.ready);
    } else if (msg.type === 'settingsSync') {
      applySettingsSync(msg);
    } else if (msg.type === 'poseSync') {
      // The desktop's mirror-image send of this page's own poseResult (see
      // devBridge/client.ts's pushPoseSync) -- lets the AR overlay work with
      // computeOnDevice OFF, feeding updateARCamera from a desktop-compute
      // capture's pose instead of a local one. Only ever arrives for a
      // camera the desktop currently has in desktop-compute mode (see
      // main.ts's own guard), so this can't race/fight with a concurrent
      // local devicePoseLoop update.
      updateARCamera(parseRemoteARCameraPose(msg.fix ?? null));
    }
  });
}
connectRelay();

// ── Shutter / video streaming ────────────────────────────────────────────
//
// Downscaled to the resolution slider's own targetLongEdge before encoding
// -- NOT just a transfer-speed knob: Sphere Lab's ingestRealCapture resizes
// a physical camera's analysis buffers to match whatever resolution
// actually arrives (unlike a simulated capture, there's no further
// downsample step after this), so targetLongEdge IS the real analysis
// resolution, not merely a cap on top of one.

// Returns WHY, not just whether, so the video loop can count each reason
// separately (see loopTicks and friends above) -- the whole point being to
// tell "blocked because the network hasn't drained the last frame yet"
// (bandwidth-bound) apart from "blocked for some other reason" instead of
// collapsing both into one opaque boolean. Desktop-compute only -- see
// captureComputeAndSendPose's own comment on why device-compute doesn't
// need this gate at all.
function sendGateStatus(): 'ok' | 'backpressure' | 'not-ready' {
  // Transport-level backpressure, independent of whether Sphere Lab wants
  // more data: bufferedAmount > 0 means the PREVIOUS frame hasn't actually
  // left the device yet (queued by the browser/OS faster than the network
  // can drain it). Without this, an unthrottled video loop (pipelined mode
  // removed the old wait-for-ack gate, which used to cap the send rate as
  // a side effect) will happily encode+queue a new multi-hundred-KB JPEG on
  // every rAF tick regardless -- confirmed live: that produced a growing
  // multi-SECOND queueing delay per frame, dwarfing every other stage in
  // the pipeline. The mailbox on the desktop only helps once a message
  // arrives; it can't do anything about a backlog stuck in transit.
  if (ws && ws.bufferedAmount > 0) return 'backpressure';
  if (!(sphereLabPipelined || sphereLabReady)) return 'not-ready';
  return 'ok';
}
function canSend() {
  return sendGateStatus() === 'ok';
}

// Draws the current video frame onto captureCanvas at the resolution
// slider's own targetLongEdge, mirrored to match the on-screen preview when
// using the front camera -- shared by BOTH compute modes (desktop-compute's
// captureAndSendFrame below, and device-compute's captureComputeAndSendPose
// further down), which then diverge on what to DO with the drawn canvas
// (JPEG-encode and send the image vs. read pixels back and compute a pose
// locally).
function drawCurrentFrameToCanvas(): { cw: number; ch: number } {
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(1, targetLongEdge / Math.max(vw, vh));
  const cw = Math.round(vw * scale), ch = Math.round(vh * scale);
  captureCanvas.width = cw; captureCanvas.height = ch;
  if (currentFacing === 'user') {
    captureCtx.save();
    captureCtx.translate(cw, 0);
    captureCtx.scale(-1, 1);
    captureCtx.drawImage(video, 0, 0, cw, ch);
    captureCtx.restore();
  } else {
    captureCtx.drawImage(video, 0, 0, cw, ch);
  }
  return { cw, ch };
}

// Grabs the current video frame and sends it, if there's anywhere to send
// it to. Shared by the single-tap shutter and the video-mode loop below --
// callers are responsible for checking sendGateStatus()/canSend() first
// (video mode checks every tick; the shutter click handler checks once per
// tap).
function captureAndSendFrame() {
  if (!currentStream || video.videoWidth === 0) return;
  // Stamped here, before anything below -- the earliest point once the
  // send gate has already let this call through.
  const sentAt = Date.now();
  drawCurrentFrameToCanvas();
  // Stamped right after the canvas draw, before toDataURL -- splits "pull
  // the current video frame onto a canvas" (cheap compositing) from "JPEG
  // encode" (the actual toDataURL cost below), instead of bundling both
  // into one number the way an earlier version of this instrumentation did.
  const pulledAt = Date.now();
  const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.85);
  // Stamped between encode finishing and ws.send() starting, so the desktop
  // can split "JPEG encode, phone-side CPU" from "actual network transit"
  // instead of lumping both into one number and guessing which one a given
  // slow sample was.
  const encodedAt = Date.now();

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'realCapture', dataUrl, sentAt, pulledAt, encodedAt }));
    shutterBtn.classList.add('sent');
    setTimeout(() => shutterBtn.classList.remove('sent'), 300);
    // Optimistic -- Sphere Lab will confirm the real state via captureReady
    // once it's actually looked at this frame; this just stops us (or the
    // video loop) from firing off a second one in the meantime. Skipped
    // when pipelined: Sphere Lab's mailbox can always take another frame,
    // so forcing sphereLabReady false here would just make the yellow
    // state lie about whether it's actually still crunching (and did,
    // before this fix -- nothing ever cleared it back in pipelined mode
    // since the desktop only reports genuine busy/idle transitions).
    if (!sphereLabPipelined) setReady(false);
  } else {
    setRelayStatus('not connected -- capture NOT sent', true);
  }
}

// Diagnostic-only summary of state's intermediates, built when sendDebugInfo
// is on -- NOT a full serialization of every field (composite lines/decode
// grids carry THREE.Vector3-laden or large per-pixel data not worth wire
// cost for a debug toggle whose whole point is staying cheap when off, and
// affordable when on). Picked for "why is consistency/votes low" diagnosis:
// composite/vote counts (LSD/join-walk health), the decode grid's actual
// valid-sample ratio (directly explains few/no window agreement), and
// gridPeriodPhase's own period/phase/height (sanity-checks the distance
// estimate). All plain numbers -- JSON-safe with no reconstruction needed
// on the desktop side.
//
// `pipeline` (see this session's "Ship auxiliary pipeline intermediates"
// plan) is a SEPARATE, additional payload: the phone's own pipeline
// intermediates verbatim, assigned back onto the exact same camera.last*
// fields a desktop-compute capture already populates (pipeline/capture.ts's
// ingestRemotePose), rather than re-derived into a summary shape like the
// fields above. gridPeriodPhase/voteComposites are already sitting on
// `state` at zero marginal cost; lsdRectangles is the one genuinely NEW
// computation here (LSD rectangle data isn't retained anywhere in the
// production pose pipeline -- see computeGradient2x2Composites in
// pipeline/votes.ts, which discards them after reducing to composite
// lines) -- a second, explicit LSD pass, mirroring the same
// recompute-for-debug-display pattern overlays/lsdOverlay.ts's own
// updateLsdOverlay already uses desktop-side. rawMembers (the per-pixel
// flood-fill membership list) is deliberately excluded from the wire
// shape -- it scales with image resolution, not region count, and nothing
// on the desktop actually reads camera.lastLsdRectangles.rawMembers today.
function buildDebugPayload(state: PoseComputeState, lsdRects: LsdRectangle[]) {
  const grid = state.lastDecodeGrid;
  let validCount = 0, totalCount = 0;
  if (grid) {
    for (const row of grid.points) for (const pt of row) { totalCount++; if (pt.valid) validCount++; }
  }
  let correctCount = 0, wrongCount = 0;
  if (state.lastDecodeCorrectness) {
    for (const row of state.lastDecodeCorrectness) for (const cell of row) {
      if (!cell) continue;
      if (cell.correct) correctCount++; else wrongCount++;
    }
  }
  const gpp = state.lastGridPeriodPhase;
  return {
    compositeLineCount: state.lastVoteComposites?.length ?? 0,
    voteCount: state.lastVotes?.length ?? 0,
    gridPeriodPhase: gpp ? {
      period: gpp.period, phiRow: gpp.phiRow, phiCol: gpp.phiCol, height: gpp.height,
      seedPeriod: gpp.debug.seedPeriod, bracket: gpp.debug.bracket,
      rowLineCount: gpp.rowLines.length, colLineCount: gpp.colLines.length,
    } : null,
    decodeGrid: grid ? { rows: grid.rows, cols: grid.cols, validCount, totalCount } : null,
    decodeCorrectness: state.lastDecodeCorrectness ? { correctCount, wrongCount } : null,
    pipeline: {
      gridPeriodPhase: state.lastGridPeriodPhase,
      voteComposites: state.lastVoteComposites,
      lsdRectangles: lsdRects.map((r) => ({
        cx: r.cx, cy: r.cy, theta: r.theta, length: r.length, width: r.width,
        accepted: r.accepted, retries: r.retries, nfaLog10: r.nfaLog10,
      })),
    },
  };
}

// ── Device-compute: capture + run the pose pipeline locally + send just
// the result ──────────────────────────────────────────────────────────────
//
// The bufferedAmount backpressure gate above is irrelevant here -- a
// serialized pose is a few hundred bytes, never queues -- and there's no
// "Sphere Lab busy" readiness gate either, since the desktop does no work
// at all on this path (see ingestRemotePose, pipeline/capture.ts). The
// natural pacing is "compute (now the actual bottleneck), then send, then
// start the next capture" -- devicePoseLoop below just does that, no
// rAF-tick gating, so the local timing readout shows the phone's true
// compute speed with no artificial pacing.
let devicePoseComputing = false;
async function captureComputeAndSendPose() {
  if (!currentStream || video.videoWidth === 0) return;
  devicePoseComputing = true;
  try {
    const { cw, ch } = drawCurrentFrameToCanvas();
    // Same call ingestRealCapture makes on the desktop side (getImageData ->
    // toGrayscale), then the same top-down -> bottom-up row-flip orientation
    // fix ingestRealCapture applies before handing off to the reconstruction
    // pipeline (see pipeline/capture.ts).
    const topDown = captureCtx.getImageData(0, 0, cw, ch).data;
    const grayTopDown = toGrayscale(topDown, cw, ch);
    const gray = flipRowsF64(grayTopDown, cw, ch);

    const state: PoseComputeState = {
      aspect: cw / ch,
      settings: cameraSettings,
      lastVoteComposites: null, lastVotes: null, lastQuadricPair: null, lastGridPeriodPhase: null,
      lastRecoveredAxes: null, lastDecodeGrid: null, lastDecodeRotated: null, lastDecodeCorrectness: null,
      lastPositionDecode: null,
    };
    const t0 = performance.now();
    const timing = await computePoseFromCapture(state, gray, cw, ch);
    const totalMs = performance.now() - t0;
    updateARCamera(buildLocalARCameraPose(state));

    if (ws && ws.readyState === WebSocket.OPEN) {
      const msg: any = {
        type: 'poseResult', w: cw, h: ch,
        recoveredAxes: state.lastRecoveredAxes ? {
          Drow: state.lastRecoveredAxes.Drow.toArray(), Dcol: state.lastRecoveredAxes.Dcol.toArray(),
          Dnormal: state.lastRecoveredAxes.Dnormal.toArray(), distance: state.lastRecoveredAxes.distance,
        } : null,
        positionDecode: state.lastPositionDecode ? {
          row: state.lastPositionDecode.row, col: state.lastPositionDecode.col,
          consistency: state.lastPositionDecode.consistency, votes: state.lastPositionDecode.votes,
          totalWindows: state.lastPositionDecode.totalWindows,
          camPos: state.lastPositionDecode.camPos.toArray(),
          recoveredCamQuat: state.lastPositionDecode.recoveredCamQuat.toArray(),
          orientation: state.lastPositionDecode.orientation,
        } : null,
      };
      // Both off by default, independently toggled -- see their own
      // checkbox comments above. Neither is computed at all unless its
      // toggle is on, so leaving both off costs nothing beyond the
      // checkbox reads themselves. The second LSD pass below (see
      // buildDebugPayload's own comment on `pipeline.lsdRectangles`) is
      // the one genuinely NEW computation sendDebugInfo pays for -- never
      // runs unless this branch is taken.
      if (sendDebugInfo) {
        const field = computeGradient2x2Field(gray, cw, ch);
        const lsdRects = await computeLsdRectanglesAuto(field, {
          toleranceDeg: cameraSettings.lsdToleranceDeg,
          rhoNoiseThreshold: cameraSettings.lsdRhoNoiseThreshold,
          magnitudeBuckets: cameraSettings.lsdMagnitudeBuckets,
          nfaEpsilon: cameraSettings.lsdNfaEpsilon,
          nfaTestExponent: cameraSettings.lsdNfaTestExponent,
          maxRetries: cameraSettings.lsdMaxRetries,
          retryToleranceFactor: cameraSettings.lsdRetryToleranceFactor,
          retryShrinkFraction: cameraSettings.lsdRetryShrinkFraction,
        });
        msg.debug = buildDebugPayload(state, lsdRects);
      }
      if (sendCapturedImage) msg.dataUrl = captureCanvas.toDataURL('image/jpeg', 0.85);
      ws.send(JSON.stringify(msg));
    }

    // Local-only -- never sent over the network (the whole point of this
    // toggle is seeing the phone's TRUE compute speed, and timing is
    // deliberately excluded from the wire payload for now, see this
    // session's on-device-pose-recovery plan).
    const fps = totalMs > 0 ? 1000 / totalMs : 0;
    poseReadoutEl.textContent = `pose ${totalMs.toFixed(0)}ms (${fps.toFixed(1)}fps)  votes ${timing.votesMs.toFixed(0)} fit ${timing.fitMs.toFixed(0)} pose ${timing.poseMs.toFixed(0)} dist ${timing.distanceMs.toFixed(0)} decode ${timing.decodeMs.toFixed(0)}`
      + (state.lastPositionDecode ? '' : '  [no fix]');
  } finally {
    devicePoseComputing = false;
  }
}

// Continuous self-paced loop for device-compute + video mode -- starts the
// next capture+compute+send cycle as soon as the previous one resolves,
// rather than checking a rAF tick like the desktop-compute videoLoop below
// (compute time, not frame delivery or backpressure, is the bottleneck
// here). Re-entrant-safe: stops on its own once computeOnDevice or
// captureMode changes out from under it.
async function devicePoseLoop() {
  if (!computeOnDevice || captureMode !== 'video' || devicePoseComputing) return;
  await captureComputeAndSendPose();
  if (computeOnDevice && captureMode === 'video') devicePoseLoop();
}

shutterBtn.addEventListener('click', () => {
  // In video mode the button is a status indicator, not a trigger -- frames
  // already send themselves via the loop below.
  if (captureMode !== 'single') return;
  if (computeOnDevice) {
    if (devicePoseComputing) return;
    captureComputeAndSendPose();
    shutterBtn.classList.add('sent');
    setTimeout(() => shutterBtn.classList.remove('sent'), 300);
    return;
  }
  if (!canSend()) return;
  captureAndSendFrame();
});

// Ticks every frame; only actually sends in video mode. When Sphere Lab
// isn't pipelined, the gate is 'not-ready' whenever sphereLabReady is
// false, turning a slow reconstruction pass into a natural frame-rate cap
// instead of flooding the relay with frames nothing's looked at yet -- same
// as before. When pipelined, Sphere Lab's own mailbox does that job instead
// (always takes the freshest frame, drops stale ones), so the only gate
// left in practice is 'backpressure'. loopTicks/etc. (see above) count
// every outcome so a diagnostic script can tell which one actually explains
// the idle gap, instead of guessing from timing alone.
function videoLoop() {
  requestAnimationFrame(videoLoop);
  loopTicks++;
  // Device-compute video mode is driven by its own self-paced devicePoseLoop
  // (compute time, not rAF ticks/backpressure, is the pacing signal there) --
  // see this session's on-device-pose-recovery plan.
  if (captureMode !== 'video' || computeOnDevice) return;
  const status = sendGateStatus();
  if (status === 'backpressure') { backpressureBlockedTicks++; return; }
  if (status === 'not-ready') { readinessBlockedTicks++; return; }
  sendsAttempted++;
  captureAndSendFrame();
}
videoLoop();
