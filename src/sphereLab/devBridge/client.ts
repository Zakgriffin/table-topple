import * as THREE from 'three';
import { PhysicalCamera } from '../camera/model.ts';
import { createPhysicalCamera } from '../camera/factory.ts';
import { findPhysicalCameraByConnection, removeCameraTab } from '../camera/lifecycle.ts';
import { activeCamera, cameras, isPhysical, nextCameraColor } from '../camera/store.ts';
import { getAnalysisVFovRad } from '../math/geometry.ts';
import type { RemotePoseMessage } from '../pipeline/capture.ts';
import { tryUnpackPoseResultWithImage } from './poseResultWire.ts';
import { renderer } from '../scene/renderer.ts';
import { globalState } from '../state.ts';
import { renderCameraTabs, refreshCameraPanel } from '../ui/cameraPanel.ts';
import { throughCamCanvas } from '../ui/dom.ts';

// Shared by both the realCapture and captureMode handlers below -- either
// one can be the first message ever received from a given phone (toggling
// to video before ever taking a photo is a real path now), so both need to
// be able to auto-create the tab. Deliberately NOT made active either way,
// same reasoning as before: a phone doing something in the background
// shouldn't yank focus from whatever camera the user is currently looking
// at.
function findOrCreatePhysicalCamera(connectionId: string | undefined): PhysicalCamera | undefined {
  let cam = connectionId ? findPhysicalCameraByConnection(connectionId) : undefined;
  if (!cam && connectionId) {
    cam = createPhysicalCamera(nextCameraColor(), connectionId);
    cameras.set(cam.id, cam);
    renderCameraTabs();
  }
  if (!cam) {
    const active = activeCamera();
    if (active && isPhysical(active)) cam = active;
  }
  return cam;
}

// Set by initDevBridge once it has a live socket; module-level (rather than
// staying a local var inside that IIFE, like before Stage C) specifically so
// renderCameraTabs' kick button can reach it without threading the socket
// through as a parameter everywhere.
export let devBridgeSocket: WebSocket | null = null;
export function sendToDevBridge(obj: unknown) {
  if (devBridgeSocket && devBridgeSocket.readyState === WebSocket.OPEN) devBridgeSocket.send(JSON.stringify(obj));
}

// Builds the 16-field PoseComputeState.settings payload for one physical
// camera's phone -- see pipeline/poseCompute.ts's PoseComputeState and this
// session's on-device-pose-recovery plan.
function buildCameraSettingsPayload(cam: PhysicalCamera) {
  const s = cam.settings;
  return {
    horizFovDeg: s.horizFovDeg, weightSharpenPower: s.weightSharpenPower,
    useWorldVoteOrientation: s.useWorldVoteOrientation, worldVoteRefineSteps: s.worldVoteRefineSteps,
    gridPeriodPhaseGapLowerBound: s.gridPeriodPhaseGapLowerBound, minGrazingCos: s.minGrazingCos,
    lsdToleranceDeg: s.lsdToleranceDeg, lsdRhoNoiseThreshold: s.lsdRhoNoiseThreshold,
    lsdRhoHighThreshold: s.lsdRhoHighThreshold, lsdCclSteps: s.lsdCclSteps,
    lsdNfaEpsilon: s.lsdNfaEpsilon,
    lsdNfaTestExponent: s.lsdNfaTestExponent, lsdMaxRetries: s.lsdMaxRetries,
    lsdRetryToleranceFactor: s.lsdRetryToleranceFactor, lsdRetryShrinkFraction: s.lsdRetryShrinkFraction,
    lsdMergeMinSimilarity: s.lsdMergeMinSimilarity, lsdJoinSteps: s.lsdJoinSteps,
    lsdMinLengthPx: s.lsdMinLengthPx, lsdMaxTravelFactor: s.lsdMaxTravelFactor,
  };
}

// Pushes the current pipeline-tunable settings to one physical camera's
// phone (source of truth stays the desktop's own sliders) -- see
// server.js's settingsSync routing (finds the one capture socket matching
// captureId, sends directly) and mobileCapture.ts's own receiving end.
// Called on: per-camera-settings slider changes (that one camera only, see
// ui/cameraPanel.ts), global useGPU*/boardSize changes (every connected
// physical camera), and the first time a physical camera is seen in
// main.ts's animate loop (camera.neverSyncedSettings) -- see this session's
// on-device-pose-recovery plan.
export function pushSettingsSync(cam: PhysicalCamera) {
  sendToDevBridge({
    type: 'settingsSync',
    captureId: cam.connectionId,
    globalState: {
      useGPUFit: globalState.useGPUFit, useGPUGradient: globalState.useGPUGradient,
      useGPULsdFit: globalState.useGPULsdFit, useGPUDecode: globalState.useGPUDecode,
      boardSize: globalState.boardSize,
    },
    cameraSettings: buildCameraSettingsPayload(cam),
  });
}

// Mirror image of a phone's own poseResult send (mobileCapture.ts) -- that's
// "phone computed a pose, ship it to the desktop"; this is "desktop computed
// a pose (from a desktop-compute capture), ship it back down to the phone",
// so a phone's own AR overlay (mobile-capture.html's arCanvas) can work even
// with "compute pose on this device" OFF, per an explicit ask (see this
// session's chat) that a phone should be able to get its pose FROM the
// server when it isn't computing one itself, symmetric with settingsSync's
// own desktop-source-of-truth-pushed-to-phone pattern. Only the CAMERA's
// pose travels over the wire -- the AR board itself is static, known, fixed
// geometry on both ends (mobileCapture.ts's arPlane/arCube mirror
// scene/floor.ts's own floorMesh, C*GRID_STEP x R*GRID_STEP at world
// origin), so there's nothing per-capture about it worth sending.
export function pushPoseSync(cam: PhysicalCamera) {
  const decode = cam.lastPositionDecode;
  if (!decode) {
    sendToDevBridge({ type: 'poseSync', captureId: cam.connectionId, fix: null });
    return;
  }
  sendToDevBridge({
    type: 'poseSync', captureId: cam.connectionId,
    fix: {
      camPos: decode.camPos.toArray(), recoveredCamQuat: decode.recoveredCamQuat.toArray(),
      aspect: cam.aspect, vFovDeg: THREE.MathUtils.radToDeg(getAnalysisVFovRad(cam)),
    },
  });
}

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
  let reconnectTimer: number | undefined;
  // randomUUID()'s canonical string form is always exactly this many ASCII
  // chars -- must match server.js's own CAPTURE_ID_BYTES exactly, since
  // that's what fixes the prefix width on a binary realCapture frame.
  const CAPTURE_ID_BYTES = 36;
  // realCapture's JSON metadata (sentAt/pulledAt/encodedAt) and its image
  // bytes now arrive as two separate messages (see this session's chat) --
  // held here, keyed by captureId, from whichever arrives first until the
  // OTHER one shows up and they can be combined into a PhysicalCamera's
  // pendingCapture. In practice the JSON always arrives first (see
  // mobileCapture.ts's captureAndSendFrame, which sends both from the same
  // callback in that order), but keying by captureId rather than relying on
  // strict adjacency also keeps this correct if more than one phone is
  // sending at once and their broadcasts interleave in this tab's own
  // message stream.
  const pendingRealCaptureMeta = new Map<string, { sentAt: number; pulledAt: number; encodedAt: number }>();

  function scheduleReconnect() {
    devBridgeSocket = null;
    clearTimeout(reconnectTimer);
    reconnectTimer = window.setTimeout(connect, 2000);
  }

  function connect() {
    let ws: WebSocket;
    try { ws = new WebSocket(`ws://localhost:${BRIDGE_PORT}`); }
    catch { scheduleReconnect(); return; }
    devBridgeSocket = ws;
    // Default binaryType is 'blob' -- realCapture's image bytes now arrive
    // as a genuine binary frame (see mobileCapture.ts/server.js), and
    // ArrayBuffer is the more convenient shape to slice the leading
    // captureId prefix off of below.
    ws.binaryType = 'arraybuffer';

    ws.addEventListener('open', () => ws.send(JSON.stringify({ role: 'browser' })));
    ws.addEventListener('close', scheduleReconnect);
    ws.addEventListener('error', () => {});
    ws.addEventListener('message', (ev) => {
      // server.js's binary realCapture frame -- a fixed CAPTURE_ID_BYTES-
      // byte ASCII captureId prefix (its own comment) followed by the raw
      // JPEG bytes. Checked BEFORE the JSON.parse below, not after it
      // fails: ev.data is only ever text (JSON) or this one binary shape,
      // never something that happens to parse as JSON by accident.
      if (ev.data instanceof ArrayBuffer) {
        const bytes = new Uint8Array(ev.data);
        const captureId = new TextDecoder().decode(bytes.subarray(0, CAPTURE_ID_BYTES));
        const rest = bytes.subarray(CAPTURE_ID_BYTES);
        // A device-compute phone's poseResult-with-image (see
        // poseResultWire.ts) shares this same "any binary WS message"
        // channel with realCapture's raw-JPEG frames below -- its own tag
        // byte (never 0xFF, JPEG's own SOI byte) disambiguates the two with
        // no server.js involvement, since server.js's binary relay is
        // already content-agnostic either way.
        const poseImage = tryUnpackPoseResultWithImage(rest);
        if (poseImage) {
          const cam = findOrCreatePhysicalCamera(captureId);
          if (cam) {
            const header = poseImage.header as RemotePoseMessage;
            // Same mailbox pattern as pendingCapture below -- always
            // overwrite with the freshest message, drained by main.ts's
            // animate loop once the camera is actually free, rather than
            // calling ingestRemotePose directly here (which used to silently
            // drop this message entirely if the camera happened to be busy).
            cam.pendingPoseResult = { ...header, imageBytes: poseImage.jpeg };
          }
          return;
        }
        const meta = pendingRealCaptureMeta.get(captureId);
        pendingRealCaptureMeta.delete(captureId);
        const cam = findOrCreatePhysicalCamera(captureId);
        if (cam && meta) {
          const now = Date.now();
          const blob = new Blob([rest], { type: 'image/jpeg' });
          // Mailbox, not a queue -- see the (former) realCapture JSON
          // handler's own comment on why: always overwrite with the
          // freshest frame, main.ts's animate loop pumps it once the
          // camera is actually free.
          cam.pendingCapture = { blob, sentAt: meta.sentAt, pulledAt: meta.pulledAt, encodedAt: meta.encodedAt, receivedAt: now, bytes: blob.size };
        }
        return;
      }
      let msg: any;
      try { msg = JSON.parse(ev.data); } catch { return; }

      if (msg.type === 'eval') {
        let ok = true, value: any, error: string | undefined;
        try { value = eval(msg.code); }
        catch (e: any) { ok = false; error = String(e?.stack ?? e); }
        try { value = value === undefined ? undefined : JSON.parse(JSON.stringify(value)); }
        catch { value = String(value); }
        ws.send(JSON.stringify({ type: 'evalResult', id: msg.id, ok, value, error }));
      } else if (msg.type === 'screenshot') {
        // Through-Cam's own content lives on a separate canvas now (see
        // scene/throughCam2D.ts) -- renderer.domElement shows nothing
        // useful while in that mode (its own clear color only), so a
        // screenshot has to follow whichever canvas is actually visible.
        // throughCamCanvas is also sized to the camera's TRUE captured
        // resolution rather than the window (unlike renderer.domElement),
        // so a Through-Cam screenshot now correctly returns that
        // resolution too, not the desktop window's.
        const source = globalState.mode === 'through' ? throughCamCanvas : renderer.domElement;
        const dataUrl = source.toDataURL('image/png');
        ws.send(JSON.stringify({ type: 'screenshotResult', id: msg.id, ok: true, dataUrl }));
      } else if (msg.type === 'realCapture') {
        // Broadcast from mobile-capture.html via the dev-bridge relay,
        // tagged with the sending phone's own connectionId (server.js
        // assigns one per 'capture' connection). See findOrCreatePhysicalCamera
        // above for the auto-create-but-don't-activate reasoning. No image
        // bytes in THIS message anymore (see the ArrayBuffer branch above)
        // -- just holds the timing fields until the binary frame carrying
        // this same captureId arrives right behind it, which is what
        // actually builds cam.pendingCapture. msg.sentAt/pulledAt/encodedAt
        // (Date.now() on the phone, see mobileCapture.ts) fall back to
        // "now" for an old/unpatched phone client so the derived durations
        // degrade to ~0 instead of NaN.
        if (msg.captureId) {
          findOrCreatePhysicalCamera(msg.captureId); // auto-create the tab even if the binary frame never arrives
          const now = Date.now();
          pendingRealCaptureMeta.set(msg.captureId, { sentAt: msg.sentAt ?? now, pulledAt: msg.pulledAt ?? now, encodedAt: msg.encodedAt ?? now });
        }
      } else if (msg.type === 'poseResult') {
        // Same broadcast/routing as realCapture (see server.js), sent
        // instead of it when the phone is in device-compute mode -- the
        // phone already ran the full pose-recovery pipeline itself and is
        // handing over just {w,h,recoveredAxes,positionDecode}, no image
        // bytes at all here (see the ArrayBuffer branch above for the
        // sendCapturedImage-on case, which arrives as its own binary frame
        // instead of this JSON message), UNLESS just sendDebugInfo is on, in
        // which case msg.debug rides along too -- see mobileCapture.ts's own
        // comments. Written into the SAME pendingPoseResult mailbox the
        // binary branch above uses (never calls ingestRemotePose directly)
        // so a message arriving while the camera is busy gets drained on the
        // next free tick instead of being silently dropped forever -- see
        // camera/model.ts's own comment on pendingPoseResult.
        const cam = findOrCreatePhysicalCamera(msg.captureId);
        if (cam) {
          cam.pendingPoseResult = {
            w: msg.w, h: msg.h, recoveredAxes: msg.recoveredAxes ?? null, positionDecode: msg.positionDecode ?? null,
            debug: msg.debug,
          };
        }
      } else if (msg.type === 'captureMode' && msg.mode) {
        // The phone's video/single toggle flipped -- purely a UI reflection
        // (see PhysicalCamera.captureMode's own comment), no pipeline effect
        // by itself.
        const cam = findOrCreatePhysicalCamera(msg.captureId);
        if (cam) {
          cam.captureMode = msg.mode === 'video' ? 'video' : 'single';
          if (cam === activeCamera()) refreshCameraPanel();
        }
      } else if (msg.type === 'computeMode' && msg.mode) {
        // The phone's desktop-compute/device-compute toggle flipped --
        // purely a UI reflection (see PhysicalCamera.computeMode's own
        // comment), same pattern as captureMode. Which of realCapture vs
        // poseResult actually arrives is decided by the phone itself (its
        // own message type), not by this field -- this is just what lets
        // Sphere Lab's UI show which mode a given phone is in.
        const cam = findOrCreatePhysicalCamera(msg.captureId);
        if (cam) {
          cam.computeMode = msg.mode === 'device' ? 'device' : 'desktop';
          if (cam === activeCamera()) refreshCameraPanel();
        }
      } else if (msg.type === 'frameStats') {
        // Diagnostic-only, doesn't create a tab on its own (unlike
        // realCapture/captureMode) -- purely informational, nothing to
        // auto-create a camera FOR if none exists yet.
        const cam = msg.captureId ? findPhysicalCameraByConnection(msg.captureId) : undefined;
        if (cam) {
          cam.lastFrameStats = {
            nominalFrameRate: msg.nominalFrameRate ?? null,
            avgIntervalMs: msg.avgIntervalMs ?? null, maxIntervalMs: msg.maxIntervalMs ?? null, sampleCount: msg.sampleCount ?? null,
            loopTicks: msg.loopTicks, backpressureBlockedTicks: msg.backpressureBlockedTicks,
            readinessBlockedTicks: msg.readinessBlockedTicks, sendsAttempted: msg.sendsAttempted,
          };
        }
      } else if (msg.type === 'captureDisconnected' && msg.captureId) {
        // The phone behind some physical camera(s) disconnected -- naturally
        // or via this tab's own kick button (see renderCameraTabs). Removes
        // the tab itself here, not optimistically when the kick was sent,
        // so it stays correct if the kick races with an unrelated
        // disconnect. Snapshotted to an array first since removeCameraTab
        // mutates the very `cameras` map being iterated.
        for (const cam of Array.from(cameras.values())) {
          if (cam.type === 'physical' && cam.connectionId === msg.captureId) removeCameraTab(cam.id);
        }
      }
    });
  }
  connect();

  // Low-rate unsolicited frame push so a reasonably fresh screenshot is
  // always on disk without an explicit request.
  setInterval(() => {
    if (devBridgeSocket && devBridgeSocket.readyState === WebSocket.OPEN) {
      devBridgeSocket.send(JSON.stringify({ type: 'frame', dataUrl: renderer.domElement.toDataURL('image/jpeg', 0.7) }));
    }
  }, 1000);
})();
