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
import { computeGradient2x2Field } from './sphereLab/pipeline/gradientField.ts';
import { computeLsdRectanglesAuto, LsdRectangle } from './sphereLab/pipeline/lsdSegments.ts';
import { computePoseFromCapture, PoseComputeState } from './sphereLab/pipeline/poseCompute.ts';
import { packPoseResultWithImage } from './sphereLab/devBridge/poseResultWire.ts';

const video = document.getElementById('v') as HTMLVideoElement;
const captureCanvas = document.getElementById('captureCanvas') as HTMLCanvasElement;
const captureCtx = captureCanvas.getContext('2d')!;
const camStatus = document.getElementById('camStatus')!;
const relayStatus = document.getElementById('relayStatus')!;
const zoomSlider = document.getElementById('zoom') as HTMLInputElement;
const resSelect = document.getElementById('resSelect') as HTMLSelectElement;
const detectResBtn = document.getElementById('detectResBtn') as HTMLButtonElement;
const probeLogToggleBtn = document.getElementById('probeLogToggleBtn') as HTMLButtonElement;
const probeLogCloseBtn = document.getElementById('probeLogCloseBtn') as HTMLButtonElement;
const probeLogPanel = document.getElementById('probeLogPanel') as HTMLDivElement;
const probeLogTbody = document.getElementById('probeLogTbody') as HTMLTableSectionElement;
const probeLogMatchHeader = document.getElementById('probeLogMatchHeader') as HTMLTableCellElement;
const fpsSlider = document.getElementById('fps') as HTMLInputElement;
const fpsValue = document.getElementById('fpsValue')!;
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

// Mirrors captureCanvas's own INTRINSIC (cw,ch) directly, not its rendered
// CSS box -- both canvases share the exact same CSS (max-width/max-height
// letterbox fit, see mobile-capture.html), so matching intrinsic dimensions
// is what keeps arCanvas scaled/positioned identically to captureCanvas
// (the browser's own replaced-element sizing does the rest, since both
// elements then have the same aspect ratio driving that letterbox). `false`
// (skip three.js's own inline-style sizing) since the stylesheet already
// owns display sizing for both canvases identically -- letting setSize
// ALSO write inline width/height here would just fight that. Only actually
// resizes the GL backing store when (cw,ch) genuinely changed, since this
// gets called every drawCurrentFrameToCanvas (i.e. every rAF tick).
let arSizedCw = 0, arSizedCh = 0;
function syncARRendererSize(cw: number, ch: number) {
  if (cw === arSizedCw && ch === arSizedCh) return;
  arSizedCw = cw; arSizedCh = ch;
  if (cw > 0 && ch > 0) arRenderer.setSize(cw, ch, false);
}

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
// There's no standard API to enumerate a camera's true discrete capture
// modes (MediaTrackCapabilities.width/height only report independent
// {min,max} RANGES, no paired combinations, no aspect-ratio pairing) -- see
// this session's chat. So instead of guessing at a continuous slider, this
// SWEEPS a heuristic candidate list of common real camera resolutions
// against the actual device, one at a time, as a basic (top-level, not
// `advanced`) EXACT constraint -- unlike a bare/advanced constraint (which
// spec'd to silently drop if unsatisfiable, see applyExactResolution's own
// comment), an `{exact: ...}` BASIC constraint genuinely rejects
// (OverconstrainedError) when the camera can't do it, giving a real
// per-candidate yes/no signal. Includes both (w,h) and (h,w) for every
// entry -- also genuinely ambiguous, device to device, whether
// width/height constraints operate in the camera's true sensor-native
// (always landscape) space or a device-orientation-corrected space, so
// probing both resolves that empirically instead of guessing.
//
// MANUALLY triggered (see #detectResBtn below), NOT run automatically on
// every camera start -- a real phone crashed during this sweep earlier this
// session. RES_PROBE_DELAY_MS pacing was originally added on the theory
// that back-to-back stream reconfigurations themselves were the problem;
// that turned out NOT to be the fix (confirmed: pacing alone didn't stop
// the crash) -- the actual cause was the continuous, unthrottled,
// full-native-resolution redraw of the now-VISIBLE captureCanvas on every
// rAF tick (see drawCurrentFrameToCanvas/previewLongEdgeCap and
// drawFullResFrameToSendCanvas), which this sweep's larger candidates made
// worse but didn't require to trigger. Kept manual-only regardless (still a
// heavier, more visible operation than users want running automatically),
// and the delay stays too (still reasonable pacing for a UI that's visibly
// flashing through candidates), but neither should be credited with fixing
// the crash itself.
const RES_PROBE_DELAY_MS = 300;
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const RES_CANDIDATES_BASE: { w: number; h: number }[] = [
  { w: 640, h: 480 }, { w: 1024, h: 768 }, { w: 1280, h: 960 }, { w: 1600, h: 1200 },
  { w: 2048, h: 1536 }, { w: 3264, h: 2448 }, { w: 4032, h: 3024 },
  { w: 640, h: 360 }, { w: 1280, h: 720 }, { w: 1920, h: 1080 }, { w: 2560, h: 1440 }, { w: 3840, h: 2160 },
];
// 'untested': never probed. 'supported': verified live -- a fresh stream
// requested at exactly (w,h) (see requestStreamAt) came back reporting
// EXACTLY (w,h). 'rejected': either the request threw, or it resolved but
// came back at some OTHER resolution than requested (the swapped-
// orientation case this whole verification step exists to catch).
type ResCandidateStatus = 'untested' | 'supported' | 'rejected';
interface ResCandidate { w: number; h: number; status: ResCandidateStatus }
let resCandidates: ResCandidate[] = [];

function buildResCandidateList(): ResCandidate[] {
  const seen = new Set<string>();
  const list: ResCandidate[] = [];
  for (const { w, h } of RES_CANDIDATES_BASE) {
    for (const [cw, ch] of [[w, h], [h, w]] as const) {
      const key = `${cw}x${ch}`;
      if (seen.has(key)) continue;
      seen.add(key);
      list.push({ w: cw, h: ch, status: 'untested' });
    }
  }
  list.sort((a, b) => a.w * a.h - b.w * b.h);
  return list;
}

// Rebuilds the <option> list from the current resCandidates -- called
// after every probe (not just once at the end) so the dropdown fills in
// live as the sweep progresses, rather than sitting inert/unexplained for
// however long the whole sweep takes. 'rejected' candidates are left OUT
// entirely now, not just disabled/grayed -- with everything shown genuinely
// confirmed-good, there's no reason to also show a pile of known-bad
// options (the full detail of WHY each one failed lives in the probe log
// instead, see probeResolution/renderProbeLog).
function renderResOptions() {
  const currentValue = resSelect.value;
  resSelect.innerHTML = '';
  const shown = resCandidates.filter((c) => c.status !== 'rejected');
  // Only reachable if setupDefaultResOption itself couldn't populate
  // anything (video dimensions not ready yet), or every candidate probed
  // so far has failed -- an empty <select> would just look broken rather
  // than explaining what to do.
  if (shown.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'tap "detect" →';
    resSelect.appendChild(opt);
    return;
  }
  for (const c of shown) {
    const opt = document.createElement('option');
    opt.value = `${c.w}x${c.h}`;
    opt.textContent = c.status === 'untested' ? `${c.w} × ${c.h} (untested)` : `${c.w} × ${c.h}`;
    resSelect.appendChild(opt);
  }
  if (Array.from(resSelect.options).some((o) => o.value === currentValue)) resSelect.value = currentValue;
}

// Requests a genuinely FRESH stream at exactly (w,h), pinned to the SAME
// physical camera via deviceId -- deliberately NOT
// track.applyConstraints() on an already-running track. Confirmed against
// a reference tool built for exactly this problem (addpipe/
// webcam-resolution-tester): applyConstraints() has uneven real-world
// support for actually renegotiating a running camera's resolution,
// especially across orientations, on mobile -- it can silently clamp to
// the nearest resolution along whatever orientation the camera already
// happens to be running, rather than a genuine renegotiation. A brand-new
// getUserMedia() call forces the SAME full negotiation a fresh page load
// would get.
//
// width/height are `ideal`, NOT `exact` -- confirmed live this session:
// `exact` made literally every candidate fail, not just the swapped-
// orientation ones. Real camera hardware (Android in particular) only
// exposes a small, FIXED list of actual stream configurations;
// MediaTrackCapabilities' width/height are independent min/max RANGES that
// make arbitrary in-between pairs look achievable when they mostly aren't
// -- `exact` rejects outright the instant a requested pair isn't one of
// the camera's own literal supported modes, even when something very
// close genuinely is available. `ideal` never fails that way: the browser
// always negotiates to its nearest actual supported config and hands back
// a real stream, which the caller then checks against what it actually got
// (video.videoWidth/videoHeight) -- same "verify, don't trust" principle
// as everywhere else in this section, just against a constraint type that
// doesn't spuriously reject in the first place. deviceId stays `exact`
// deliberately -- pinning the SAME physical camera across a teardown/
// rebuild is a real hard requirement, unlike the resolution itself.
// Returns null on any failure (camera busy, permission revoked, etc.) --
// never throws.
async function requestStreamAt(deviceId: string | undefined, w: number, h: number): Promise<MediaStream | null> {
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: w }, height: { ideal: h } },
    });
  } catch {
    return null;
  }
}

// Reconnects to a plain, unconstrained stream on the SAME physical camera
// (deviceId if known, else falls back to facingMode) -- used to bring the
// viewfinder back after a failed/rejected resolution request or after the
// sweep finishes, WITHOUT going through startCamera itself: that also
// calls setupDefaultResOption, which would blow away whatever candidates
// list is currently populated (the very thing the caller usually still
// wants showing in the dropdown).
async function reconnectPlainStream(deviceId: string | undefined) {
  try {
    const newStream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: currentFacing } },
    });
    await adoptStream(newStream);
    refreshCamStatusResolution();
  } catch (e: any) {
    camStatus.textContent = 'camera error: ' + e.message;
  }
}

// Applies a candidate already CONFIRMED supported (either by the sweep, or
// just whatever the camera is already streaming at by default) by
// swapping the live feed to a fresh stream at exactly (w,h) -- see
// requestStreamAt's own comment on why a fresh stream, not
// applyConstraints(). Returns whether the camera actually ended up at
// exactly (w,h), not just whether the request succeeded at all: a real
// device can resolve successfully while still landing on a DIFFERENT
// resolution than requested. Always refreshes camStatus with whatever the
// camera ACTUALLY ends up at either way (see refreshCamStatusResolution),
// so a mismatch or an outright rejection is visibly reported instead of
// silently assumed to have worked. On rejection, reconnects to SOME
// working stream rather than leaving the camera dead, since the old stream
// already had to be stopped before requesting the new one (real camera
// hardware generally can't have two opens on the same device at once).
//
// If resolutionAxesSwapped was detected by a prior sweep (see its own
// comment), the REQUEST sent here is swapped to compensate -- w,h below
// always mean the TRUE target (what video.videoWidth/videoHeight should
// end up as), never the raw request.
async function applyExactResolution(w: number, h: number): Promise<boolean> {
  if (video.videoWidth === w && video.videoHeight === h) { refreshCamStatusResolution(); return true; } // already exactly there
  const deviceId = currentStream?.getVideoTracks()[0]?.getSettings().deviceId;
  if (currentStream) currentStream.getTracks().forEach((t) => t.stop());
  const stream = resolutionAxesSwapped ? await requestStreamAt(deviceId, h, w) : await requestStreamAt(deviceId, w, h);
  if (!stream) {
    camStatus.textContent = `resolution ${w}x${h} REJECTED by camera`;
    await reconnectPlainStream(deviceId);
    return false;
  }
  await adoptStream(stream);
  const ok = video.videoWidth === w && video.videoHeight === h;
  if (!ok) camStatus.textContent = `resolution ${w}x${h} requested but camera gave ${video.videoWidth}x${video.videoHeight} instead`;
  refreshCamStatusResolution();
  return ok;
}

// Single source of truth for "what resolution is the camera actually at
// right now" -- reads video.videoWidth/videoHeight (what real captures
// actually use, see drawFullResFrameToSendCanvas) so camStatus and the
// resolution dropdown can never drift from reality the way camStatus used
// to (it was previously set ONCE in startCamera and never touched again,
// so it kept showing the pre-selection resolution forever after). Also
// re-syncs resSelect's displayed value, since a rejected/overridden
// request means the dropdown's own selection can no longer be trusted to
// reflect what's live either.
function refreshCamStatusResolution() {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  camStatus.textContent = `${currentFacing} camera, ${w}x${h}`;
  const value = `${w}x${h}`;
  if (Array.from(resSelect.options).some((o) => o.value === value)) resSelect.value = value;
}

resSelect.addEventListener('change', () => {
  const [w, h] = resSelect.value.split('x').map(Number);
  applyExactResolution(w, h);
});

// Populates the dropdown with just whatever the camera is ALREADY
// streaming at by default (from startCamera's own unconstrained
// negotiation) -- a single, always-safe, zero-extra-requests option, so
// the viewfinder/shutter are usable immediately without ever running the
// (much more invasive) sweep below. Called from startCamera, same as
// setupZoomControl/setupFramerateControl.
function setupDefaultResOption() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) { resSelect.disabled = true; return; }
  resCandidates = [{ w: vw, h: vh, status: 'supported' }];
  renderResOptions();
  resSelect.disabled = false;
}

// One row per probe attempt -- see the probeLogPanel toggle in
// mobile-capture.html. Answers, per candidate: what did we ask for, did
// getUserMedia even resolve, what did the camera actually settle on (via
// `ideal`, so this is meaningful even on failure -- see requestStreamAt's
// own comment), and did that match what was requested. Kept around for the
// whole session (not just the latest sweep) so a user can compare multiple
// sweep runs if they want; cleared only by an explicit "clear" action --
// there isn't one yet, so in practice this just grows across sweeps, which
// is fine for a debug tool nobody leaves running unattended.
interface ProbeLogEntry {
  requestedW: number; requestedH: number;
  resolved: boolean; error: string;
  actualW: number | null; actualH: number | null;
  matched: boolean;
}
let probeLog: ProbeLogEntry[] = [];

// e.matched (on the entry itself) is always the RAW, direct requested-vs-
// actual comparison -- never mutated once a probe finishes, so it stays a
// simple fact. Once resolutionAxesSwapped is known (see its own comment),
// the MEANINGFUL match check is the flipped one instead, so this recomputes
// it fresh at render time (never overwriting entry.matched) and relabels
// the column header to say so -- otherwise the table would keep showing
// "no" for rows the sweep has already reclassified as supported, which
// would look like a contradiction rather than the correct, reinterpreted
// answer.
function renderProbeLog() {
  probeLogMatchHeader.textContent = resolutionAxesSwapped ? 'match (flipped)?' : 'match?';
  probeLogTbody.innerHTML = '';
  for (const e of probeLog) {
    const tr = document.createElement('tr');
    const resolvedText = e.resolved ? 'yes' : `no${e.error ? ` — ${e.error}` : ''}`;
    const actualText = e.actualW != null ? `${e.actualW}×${e.actualH}` : '—';
    const effectiveMatch = resolutionAxesSwapped
      ? (e.actualW === e.requestedH && e.actualH === e.requestedW)
      : e.matched;
    tr.innerHTML = `<td>${e.requestedW}×${e.requestedH}</td><td>${resolvedText}</td><td>${actualText}</td><td>${effectiveMatch ? 'yes' : 'no'}</td>`;
    tr.className = effectiveMatch ? 'pass' : 'fail';
    probeLogTbody.appendChild(tr);
  }
}
probeLogToggleBtn.addEventListener('click', () => probeLogPanel.classList.toggle('visible'));
probeLogCloseBtn.addEventListener('click', () => probeLogPanel.classList.remove('visible'));

// Some platforms consistently report video.videoWidth/videoHeight SWAPPED
// relative to whatever width/height was actually requested -- confirmed
// live this session on Chrome for iOS, which (like every browser on iOS,
// by Apple's own App Store rules) is required to use WebKit as its actual
// engine regardless of branding. WebKit negotiates the camera in its own
// fixed sensor-native space, then rotates the delivered frame to match the
// phone's physical orientation WITHOUT also flipping which axis the
// original request numbers meant -- so on a phone held in portrait, asking
// for landscape dimensions comes back reporting portrait dimensions, and
// vice versa, every single time. Detected empirically per sweep (see
// sweepResolutionCandidates' own post-hoc majority-vote pass below), not
// via user-agent sniffing -- same "verify, don't assume" principle as
// everything else in this section. Once known, requestStreamAt/
// applyExactResolution swap the REQUEST to compensate, while still
// verifying against the TRUE target in video.videoWidth/videoHeight.
let resolutionAxesSwapped = false;

// One throwaway probe for the sweep below: opens a fresh, ISOLATED stream
// on a hidden <video> at exactly (w,h) via requestStreamAt, reads what
// actually came back, then tears it down -- never touches the main
// video/currentStream (sweepResolutionCandidates already stopped that for
// the sweep's duration). Mirrors the reference tool's own approach
// (addpipe/webcam-resolution-tester): a disposable probe per candidate,
// rather than repeatedly reconfiguring one live track. Always requests the
// RAW, un-swap-compensated (w,h) -- resolutionAxesSwapped isn't known yet
// during a sweep's own probing (that's what this data is FOR determining),
// see sweepResolutionCandidates' post-hoc pass for where compensation
// actually gets applied. Returns the full ProbeLogEntry (not just a
// pass/fail verdict) so the sweep loop can do that later reinterpretation
// without re-probing anything.
async function probeResolution(deviceId: string | undefined, w: number, h: number): Promise<ProbeLogEntry> {
  const entry: ProbeLogEntry = { requestedW: w, requestedH: h, resolved: false, error: '', actualW: null, actualH: null, matched: false };
  probeLog.push(entry);
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: w }, height: { ideal: h } },
    });
  } catch (e: any) {
    entry.error = e?.message || String(e);
    renderProbeLog();
    return entry;
  }
  entry.resolved = true;
  const probeVideo = document.createElement('video');
  probeVideo.muted = true;
  probeVideo.playsInline = true;
  probeVideo.srcObject = stream;
  try {
    await probeVideo.play();
    entry.actualW = probeVideo.videoWidth;
    entry.actualH = probeVideo.videoHeight;
    entry.matched = probeVideo.videoWidth === w && probeVideo.videoHeight === h;
  } catch (e: any) {
    entry.error = e?.message || String(e);
  } finally {
    stream.getTracks().forEach((t) => t.stop());
    renderProbeLog();
  }
  return entry;
}

// Manually triggered ONLY (see detectResBtn's click handler) -- each probe
// is a REAL, isolated stream open+close (see probeResolution), paced by
// RES_PROBE_DELAY_MS between every attempt. The main viewfinder goes dark
// for the sweep's duration (the live stream is stopped up front, below --
// most camera hardware can't have two opens on one device at once) and
// comes back once it's done; that's an accepted cost of getting a genuine
// empirical answer instead of a guess.
//
// Applies nothing on its own once done, INCLUDING not restoring whatever
// resolution was active before the sweep started -- the user picks what
// they actually want from the now-populated dropdown afterward, same as
// every other control in this file, rather than the sweep silently
// deciding for them. Just reconnects to a plain unconstrained stream
// (reconnectPlainStream, NOT startCamera -- that also calls
// setupDefaultResOption, which would wipe out the candidate list this
// sweep just spent time building) so the viewfinder isn't left dead.
async function sweepResolutionCandidates() {
  if (!currentStream) return;
  resSelect.disabled = true;
  detectResBtn.disabled = true;
  probeLog = [];
  renderProbeLog();
  const deviceId = currentStream.getVideoTracks()[0]?.getSettings().deviceId;
  const nativeW = video.videoWidth, nativeH = video.videoHeight;
  resCandidates = buildResCandidateList();
  if (nativeW && nativeH && !resCandidates.some((c) => c.w === nativeW && c.h === nativeH)) {
    resCandidates.push({ w: nativeW, h: nativeH, status: 'supported' }); // this IS the video element's current live size, no probe needed
    resCandidates.sort((a, b) => a.w * a.h - b.w * b.h);
  }
  renderResOptions();
  currentStream.getTracks().forEach((t) => t.stop());

  let i = 0;
  const attempts: { cand: ResCandidate; entry: ProbeLogEntry }[] = [];
  for (const cand of resCandidates) {
    if (cand.status !== 'untested') continue; // the injected native pair above is already known-good
    i++;
    detectResBtn.textContent = `detecting… ${i}/${resCandidates.length}`;
    const entry = await probeResolution(deviceId, cand.w, cand.h);
    cand.status = entry.matched ? 'supported' : 'rejected';
    attempts.push({ cand, entry });
    renderResOptions();
    await sleep(RES_PROBE_DELAY_MS);
  }

  // Post-hoc majority vote, reusing the raw data every probe above already
  // collected -- NOT a second sweep under the opposite assumption (see
  // resolutionAxesSwapped's own comment on why that's unnecessary). If most
  // resolved probes came back with actual dimensions swapped relative to
  // what was requested, this platform swaps axes -- so every candidate
  // that LOOKED rejected only because of that swap gets reclassified as
  // supported, using data already in hand.
  let straight = 0, swapped = 0;
  for (const { entry } of attempts) {
    if (entry.actualW == null) continue;
    if (entry.actualW === entry.requestedW && entry.actualH === entry.requestedH) straight++;
    else if (entry.actualW === entry.requestedH && entry.actualH === entry.requestedW) swapped++;
  }
  resolutionAxesSwapped = swapped > straight;
  // Refreshes the table's header/rows for the (possibly new) verdict even
  // when it comes back false -- a PRIOR sweep could have left the header
  // reading "match (flipped)?" and this run needs to be able to correct
  // that back, not just flip it on.
  renderProbeLog();
  if (resolutionAxesSwapped) {
    for (const { cand, entry } of attempts) {
      if (entry.actualW === cand.h && entry.actualH === cand.w) cand.status = 'supported';
    }
    renderResOptions();
  }

  detectResBtn.textContent = 'detect';
  detectResBtn.disabled = false;
  resSelect.disabled = false;
  await reconnectPlainStream(deviceId);
}
detectResBtn.addEventListener('click', () => sweepResolutionCandidates());

// ── Frame rate ───────────────────────────────────────────────────────────
//
// A lever against motion blur (see this session's chat): requesting a
// higher frame rate caps how long auto-exposure is allowed to leave the
// shutter open, since a frame's exposure can't exceed its own interval --
// works even where manual exposureTime control (a separate, far less
// widely supported constraint, notably absent on iOS Safari) isn't. Bounds
// are 1fps (a full second between frames, the slowest anyone would want) up
// to getCapabilities().frameRate.max -- "as fast as we can go" per an
// explicit ask -- with fallbacks (the currently-negotiated rate, then a
// flat 30) for a track that doesn't report frameRate capabilities at all
// (frameRate.max is part of the base Media Capture spec, unlike zoom/
// exposure, so this is expected to work broadly, but nothing guarantees it
// on every device).
const FPS_MIN = 1;
let fpsMax = FPS_MIN;

function setupFramerateControl() {
  const track = currentStream?.getVideoTracks()[0];
  let caps: any = null;
  try { caps = track && 'getCapabilities' in track ? (track as any).getCapabilities() : null; }
  catch { caps = null; }
  const current = track?.getSettings().frameRate;
  fpsMax = Math.max(FPS_MIN, Math.round(caps?.frameRate?.max ?? current ?? 30));

  if (fpsMax <= FPS_MIN) {
    fpsSlider.disabled = true;
    return;
  }
  fpsSlider.disabled = false;
  fpsSlider.min = String(FPS_MIN);
  fpsSlider.max = String(fpsMax);
  // No longer auto-applies fpsMax on load -- confirmed live this session
  // (see the resolution dropdown's own comment on sweepResolutionCandidates)
  // that an automatic, unprompted hardware reconfiguration on camera start
  // can crash a real phone; frameRate is exactly that kind of call
  // (applyConstraints -> real stream reconfiguration), so this now only
  // REFLECTS whatever the browser already negotiated on its own, matching
  // zoom/res's own "don't touch it automatically" posture -- the slider
  // still visually defaults toward the fast end, but nothing actually
  // fires until the user drags it themselves.
  const initial = Math.min(fpsMax, Math.max(FPS_MIN, Math.round(current ?? fpsMax)));
  fpsSlider.value = String(initial);
  fpsValue.textContent = `${initial}`;
}

fpsSlider.addEventListener('input', () => {
  const fps = parseInt(fpsSlider.value, 10);
  fpsValue.textContent = `${fps}`;
  const track = currentStream?.getVideoTracks()[0];
  track?.applyConstraints({ advanced: [{ frameRate: fps } as any] }).catch(() => {});
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
  useWorldVoteOrientation: false, worldVoteRefineSteps: 4,
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

// Makes `stream` the live camera feed -- swaps currentStream/video.srcObject
// to it and re-binds everything tied to the SPECIFIC MediaStreamTrack
// object (zoom, framerate), since a fresh getUserMedia() stream is a
// genuinely NEW track even when it's the same physical camera as before;
// controls left bound to the OLD (now-stopped) track would silently stop
// working otherwise. Shared by startCamera and applyExactResolution/
// sweepResolutionCandidates below, all of which now request a genuinely
// fresh stream per resolution rather than reconfiguring one track in place
// -- see this session's chat on why (a real-world reference tool aimed at
// the exact same problem, addpipe/webcam-resolution-tester, does the same
// -- MediaStreamTrack.applyConstraints() has uneven real-world support for
// actually renegotiating a running camera's resolution, especially across
// orientations, on mobile).
async function adoptStream(stream: MediaStream) {
  if (currentStream && currentStream !== stream) currentStream.getTracks().forEach((t) => t.stop());
  currentStream = stream;
  video.srcObject = stream;
  await video.play();
  const settings = stream.getVideoTracks()[0].getSettings();
  currentFacing = settings.facingMode || currentFacing;
  // No CSS mirror class anymore -- video itself is hidden (captureCanvas is
  // now the visible viewfinder, see this session's chat), and
  // drawCurrentFrameToCanvas already bakes the front-camera flip directly
  // into captureCanvas's own pixels; a CSS transform on top of that would
  // double-flip it.
  setupZoomControl();
  setupFramerateControl();
  // The camera hardware's OWN nominal capture rate -- distinct from how
  // often WE encode/send (see the frameStats reporting below), and a real
  // alternative explanation raised for video mode's idle gap: auto-exposure
  // in low light can drop a phone's actually-delivered frame rate well
  // below its nominal one, independent of anything the round-trip does.
  nominalFrameRate = settings.frameRate ?? null;
}

async function startCamera(desiredFacing: string) {
  // No width/height constraint at all here, ideal or exact -- whatever
  // resolution the browser's own default negotiation lands on is exactly
  // as arbitrary as any other guess, so there's no reason to bias it
  // toward "as high as possible" (the old `ideal: 7680x4320` trick) before
  // the user's actually picked anything. setupDefaultResOption just
  // reflects whatever this happens to be; sweepResolutionCandidates/the
  // dropdown are the real way to choose a resolution deliberately.
  const newStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: desiredFacing } },
  });
  await adoptStream(newStream);
  setupDefaultResOption(); // sweepResolutionCandidates is manual-only -- see its own comment on why
  // No explicit AR-canvas resize call needed -- videoLoop's continuous
  // drawCurrentFrameToCanvas (see its own comment) syncs captureCanvas AND
  // the AR renderer together every rAF tick, starting from the very next one.
  refreshCamStatusResolution();
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
// Sent at the stream's own actual negotiated resolution (see
// drawCurrentFrameToCanvas) -- NOT just a transfer-speed knob: Sphere Lab's
// ingestRealCapture resizes a physical camera's analysis buffers to match
// whatever resolution actually arrives (unlike a simulated capture, there's
// no further downsample step after this), so this IS the real analysis
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

// Draws the current video frame onto the VISIBLE captureCanvas -- but
// downscaled to fit the screen's own physical pixel dimensions (aspect
// preserved), not the camera's full negotiated resolution. This runs
// unconditionally every rAF tick (see videoLoop), and drawing/compositing a
// full-native-resolution (potentially 12+MP) on-screen canvas at 60fps is
// what was crashing Chrome on a real phone -- CSS max-width/max-height
// (mobile-capture.html) only scales the DISPLAYED box, it doesn't shrink
// the actual pixel buffer drawImage has to fill and the compositor has to
// upload, so a cap has to happen here instead. previewLongEdgeCap() is the
// screen's own long edge in physical pixels -- rendering any larger than
// that buys zero visible detail. The actual full-resolution frame sent to
// the server is drawn separately, on demand, by drawFullResFrameToSendCanvas
// below -- this function only ever feeds the live preview.
function previewLongEdgeCap(): number {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return Math.max(window.innerWidth, window.innerHeight) * dpr;
}

function drawCurrentFrameToCanvas(): { cw: number; ch: number } {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) { captureCanvas.width = 0; captureCanvas.height = 0; return { cw: 0, ch: 0 }; }
  const scale = Math.min(1, previewLongEdgeCap() / Math.max(vw, vh));
  const cw = Math.max(1, Math.round(vw * scale)), ch = Math.max(1, Math.round(vh * scale));
  // Guards the resize -- reassigning canvas.width/height clears the
  // backing bitmap even when set to the same value, so skipping it when
  // unchanged avoids a pointless reallocation on every one of these ticks.
  if (captureCanvas.width !== cw || captureCanvas.height !== ch) { captureCanvas.width = cw; captureCanvas.height = ch; }
  syncARRendererSize(cw, ch);
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

// Off-DOM canvas dedicated to full-resolution capture -- kept separate from
// the (now downscaled) preview captureCanvas above so the image actually
// sent to the server still carries the full resolution the user picked via
// the resolution dropdown. Never displayed, so no CSS/letterbox concerns;
// only ever drawn into on demand, by captureAndSendFrame/
// captureComputeAndSendPose below, which already only run when a capture is
// actually happening (shutter tap, a canSend()-gated video-mode tick, or a
// devicePoseLoop iteration) -- so no extra throttling logic is needed here,
// unlike the continuous preview draw above.
const sendCanvas = document.createElement('canvas');
const sendCtx = sendCanvas.getContext('2d')!;

function drawFullResFrameToSendCanvas(): { cw: number; ch: number } {
  const cw = video.videoWidth, ch = video.videoHeight;
  if (!cw || !ch) return { cw: 0, ch: 0 };
  sendCanvas.width = cw; sendCanvas.height = ch;
  if (currentFacing === 'user') {
    sendCtx.save();
    sendCtx.translate(cw, 0);
    sendCtx.scale(-1, 1);
    sendCtx.drawImage(video, 0, 0, cw, ch);
    sendCtx.restore();
  } else {
    sendCtx.drawImage(video, 0, 0, cw, ch);
  }
  return { cw, ch };
}

// Grabs the current video frame and sends it, if there's anywhere to send
// it to. Shared by the single-tap shutter and the video-mode loop below --
// callers are responsible for checking sendGateStatus()/canSend() first
// (video mode checks every tick; the shutter click handler checks once per
// tap).
// True from the moment a toBlob encode is kicked off until its callback
// fires -- toDataURL (the old encode call) was synchronous, so two
// overlapping captureAndSendFrame calls were never actually possible;
// toBlob's callback breaks that guarantee (a second video-mode tick could
// start a second encode before the first one's callback runs, and encode
// completion order isn't guaranteed to match start order). Without this
// guard, two out-of-order callbacks could send their JSON metadata
// messages in the OPPOSITE order from their binary frames, and
// devBridge/client.ts's captureId-keyed pairing (not a strict-adjacency
// assumption, see its own comment) would then match a frame's image bytes
// to the WRONG frame's timestamps. Cheap to just serialize instead: skip a
// tick outright if the previous encode hasn't finished yet, exactly the
// spirit of captureIngestBusy's own guard on the desktop decode side.
let sendEncodeInFlight = false;
function captureAndSendFrame() {
  if (!currentStream || video.videoWidth === 0 || video.videoHeight === 0 || sendEncodeInFlight) return;
  // Stamped here, before anything below -- the earliest point once the
  // send gate has already let this call through.
  const sentAt = Date.now();
  // Full-resolution draw happens HERE, on demand -- unlike the visible
  // captureCanvas (kept cheap/screen-capped by videoLoop's continuous
  // redraw), sendCanvas only gets drawn into when an actual capture is
  // happening, so paying the full-native-resolution drawImage cost here is
  // fine: it runs at most once per send, not every rAF tick.
  drawFullResFrameToSendCanvas();
  const pulledAt = Date.now();
  // toBlob, not toDataURL -- gives the JPEG's actual encoded bytes directly
  // (a Blob), sent as a real WebSocket BINARY frame below instead of
  // base64-text-encoded inside a JSON message. Base64 inflates payload size
  // by ~33% for zero benefit here -- see this session's chat -- and
  // WebSocket has supported binary frames natively the whole time.
  // Async (a callback, not a return value), so the metadata JSON message
  // and the binary frame that follows it are both sent from inside this
  // callback, back to back, keeping them adjacent on the wire.
  sendEncodeInFlight = true;
  sendCanvas.toBlob((blob) => {
    sendEncodeInFlight = false;
    if (!blob) return;
    // Stamped once the encode actually finishes, so the desktop can still
    // split "JPEG encode, phone-side CPU" from "actual network transit"
    // instead of lumping both into one number and guessing which one a
    // given slow sample was.
    const encodedAt = Date.now();
    if (!ws || ws.readyState !== WebSocket.OPEN) { setRelayStatus('not connected -- capture NOT sent', true); return; }
    // Metadata first, image bytes second -- devBridge/client.ts pairs a
    // realCapture JSON message with whichever binary frame arrives next
    // for that same captureId, so send order here matters.
    ws.send(JSON.stringify({ type: 'realCapture', sentAt, pulledAt, encodedAt }));
    ws.send(blob);
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
  }, 'image/jpeg', 0.85);
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
      chosenPeriod: gpp.debug.chosenPeriod, bracket: gpp.debug.bracket,
      rowLineCount: gpp.rowLines.length, colLineCount: gpp.colLines.length,
    } : null,
    decodeGrid: grid ? { rows: grid.rows, cols: grid.cols, validCount, totalCount } : null,
    decodeCorrectness: state.lastDecodeCorrectness ? { correctCount, wrongCount } : null,
    pipeline: {
      gridPeriodPhase: state.lastGridPeriodPhase,
      voteComposites: state.lastVoteComposites,
      lsdRectangles: lsdRects.map((r) => ({
        cx: r.cx, cy: r.cy, theta: r.theta, length: r.length, width: r.width,
        accepted: r.accepted, retries: r.retries, nfaLog10: r.nfaLog10, lineScore: r.lineScore,
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
//
// Draws its own full-resolution frame into sendCanvas (see
// drawFullResFrameToSendCanvas's own comment) rather than reading off the
// visible captureCanvas, which is now downscaled to screen resolution for
// the live preview -- pose recovery needs the actual full-resolution
// capture, same as the desktop-compute send path.
let devicePoseComputing = false;
async function captureComputeAndSendPose() {
  if (!currentStream || video.videoWidth === 0 || video.videoHeight === 0) return;
  devicePoseComputing = true;
  try {
    const { cw, ch } = drawFullResFrameToSendCanvas();
    // Same call ingestRealCapture makes on the desktop side (getImageData ->
    // toGrayscale) -- but NOT the extra flipRowsF64 that used to follow it
    // here. computePoseFromCapture's real, validated expected orientation is
    // top-down: ingestRealCapture stores camera.lastRealCaptureGray bottom-up
    // (one flip, purely for Through-Cam's own raw-preview convention), and
    // runAxesReconstruction then flips it AGAIN before calling
    // computePoseFromCapture -- flipRowsF64 is a pure involution (its own
    // exact inverse), so those two flips cancel out exactly, net top-down.
    // This function used to apply a single flip before calling
    // computePoseFromCapture directly, feeding it bottom-up instead --
    // confirmed live (this session's chat) to be the root cause of an
    // on-device Drow/Dcol axis swap and positionDecode failures relative to
    // the same image replayed through the desktop pipeline.
    const topDown = sendCtx.getImageData(0, 0, cw, ch).data;
    const grayTopDown = toGrayscale(topDown, cw, ch);

    const state: PoseComputeState = {
      aspect: cw / ch,
      settings: cameraSettings,
      lastVoteComposites: null, lastVotes: null, lastQuadricPair: null, lastGridPeriodPhase: null,
      lastRecoveredAxes: null, lastDecodeGrid: null, lastDecodeRotated: null, lastDecodeCorrectness: null,
      lastPositionDecode: null,
    };
    const t0 = performance.now();
    const timing = await computePoseFromCapture(state, grayTopDown, cw, ch);
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
        const field = computeGradient2x2Field(grayTopDown, cw, ch);
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
      // Raw bytes now (poseResultWire.ts), not a base64 dataUrl -- toBlob's
      // callback is awaited HERE, inside captureComputeAndSendPose's own
      // try block, so devicePoseComputing (which this function's own
      // finally clears, see its own comment) stays true for the encode's
      // full duration. Without that, devicePoseLoop's next iteration could
      // start redrawing sendCanvas before this frame's bytes are actually
      // read out -- the same hazard sendEncodeInFlight guards against for
      // captureAndSendFrame's own toBlob call above, just already covered
      // here by the guard this function already had.
      if (sendCapturedImage) {
        const blob = await new Promise<Blob | null>((resolve) => sendCanvas.toBlob(resolve, 'image/jpeg', 0.85));
        if (blob) ws.send(packPoseResultWithImage(msg, blob));
        else ws.send(JSON.stringify(msg));
      } else {
        ws.send(JSON.stringify(msg));
      }
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
  // Keeps captureCanvas (the VISIBLE viewfinder) live regardless of
  // capture/compute mode, at the same cadence videoLoop itself already
  // ticks -- but only draws the screen-capped preview resolution
  // (previewLongEdgeCap), not the camera's full negotiated resolution. The
  // capture functions below draw their OWN full-resolution frame into
  // sendCanvas on demand instead of reading off this one, so what actually
  // gets analyzed/sent is unaffected by this downscale.
  drawCurrentFrameToCanvas();
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
