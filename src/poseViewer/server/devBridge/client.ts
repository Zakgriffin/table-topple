import { type FrameMeta, type PhysicalCamera } from '../../shared/camera/model.ts';
import { createPhysicalCamera } from '../camera/factory.ts';
import { findPhysicalCameraByConnection, removeCameraTab } from '../camera/lifecycle.ts';
import { activeCamera, cameras, isPhysical, nextCameraColor } from '../camera/store.ts';
import type { RemotePoseMessage } from '../../shared/camera/model.ts';
import { tryUnpackPoseResultWithImage } from '../../shared/devBridge/poseResultWire.ts';
import { renderer } from '../scene/renderer.ts';
import { globalState } from '../../shared/state.ts';
import { configStatus } from '../ui/dom.ts';
import { renderCameraTabs, refreshCameraPanel } from '../ui/cameraPanel.ts';
import { throughCamCanvas } from '../ui/dom.ts';
import { nowMs } from '../../../clock.ts';

// Desktop-side cap on the relayed IMU ring, matching mobileCapture.ts's own
// IMU_RING_CAPACITY -- ~40s at the ~60Hz browsers actually deliver. Both ends
// keep a buffer because they answer different questions: the phone's survives
// a websocket drop, this one is what a recording gets dumped from.
const IMU_RING_CAPACITY = 2400;

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

// Builds the 16-field PoseInput['settings'] payload for one physical
// camera's phone -- see pose/poseCompute.ts's PoseInput and this
// session's on-device-pose-recovery plan.
function buildCameraSettingsPayload(cam: PhysicalCamera) {
  const s = cam.settings;
  return {
    horizFovDeg: s.horizFovDeg,
    gridPeriodPhaseGapLowerBound: s.gridPeriodPhaseGapLowerBound, minGrazingCos: s.minGrazingCos,
    lsdToleranceDeg: s.lsdToleranceDeg, lsdRhoNoiseThreshold: s.lsdRhoNoiseThreshold,
    lsdRhoHighThreshold: s.lsdRhoHighThreshold, lsdCclSteps: s.lsdCclSteps,
    lsdMinRegionSize: s.lsdMinRegionSize,
    lsdNfaEpsilon: s.lsdNfaEpsilon,
    lsdNfaTestExponent: s.lsdNfaTestExponent,
    lsdMinLengthPx: s.lsdMinLengthPx,
  };
}

// Pushes the current pipeline-tunable settings to one physical camera's
// phone (source of truth stays the desktop's own sliders) -- see
// server.js's settingsSync routing (finds the one capture socket matching
// captureId, sends directly) and mobileCapture.ts's own receiving end.
// Called on: per-camera-settings slider changes (that one camera only, see
// ui/cameraPanel.ts), global forceCPU/boardSize changes (every connected
// physical camera), and the first time a physical camera is seen in
// main.ts's animate loop (camera.neverSyncedSettings) -- see this session's
// on-device-pose-recovery plan.
export function pushSettingsSync(cam: PhysicalCamera) {
  sendToDevBridge({
    type: 'settingsSync',
    captureId: cam.connectionId,
    globalState: {
      forceCPU: globalState.forceCPU,
      boardSize: globalState.boardSize,
    },
    cameraSettings: buildCameraSettingsPayload(cam),
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
  const pendingRealCaptureMeta = new Map<string, {
    sentAt: number; pulledAt: number; encodedAt: number;
    drawnAt?: number; frameMeta?: FrameMeta | null;
  }>();

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
          const now = nowMs();
          const blob = new Blob([rest], { type: 'image/jpeg' });
          // Mailbox, not a queue -- see the (former) realCapture JSON
          // handler's own comment on why: always overwrite with the
          // freshest frame, main.ts's animate loop pumps it once the
          // camera is actually free.
          cam.pendingCapture = {
            blob, sentAt: meta.sentAt, pulledAt: meta.pulledAt, encodedAt: meta.encodedAt,
            receivedAt: now, bytes: blob.size, drawnAt: meta.drawnAt, frameMeta: meta.frameMeta,
          };
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
        // (nowMs() on the phone, see poseViewer/clock.ts) fall back to
        // "now" for an old/unpatched phone client so the derived durations
        // degrade to ~0 instead of NaN.
        if (msg.captureId) {
          findOrCreatePhysicalCamera(msg.captureId); // auto-create the tab even if the binary frame never arrives
          const now = nowMs();
          // drawnAt/frameMeta stay UNDEFINED rather than falling back to
          // `now` the way the three above do. The fallbacks exist so derived
          // DURATIONS degrade to ~0 instead of NaN; but a capture timestamp
          // has no such harmless default -- substituting the desktop's own
          // clock would silently hand a filter a confidently wrong time on
          // the wrong machine's clock. Absent must stay distinguishable from
          // present-and-zero here.
          pendingRealCaptureMeta.set(msg.captureId, {
            sentAt: msg.sentAt ?? now, pulledAt: msg.pulledAt ?? now, encodedAt: msg.encodedAt ?? now,
            drawnAt: msg.drawnAt, frameMeta: msg.frameMeta ?? null,
          });
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
        // Pose Viewer's UI show which mode a given phone is in.
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
      } else if (msg.type === 'imu') {
        // Same posture as frameStats: doesn't auto-create a tab. A phone
        // streaming IMU without ever having sent a frame has nothing to
        // attach the samples to, and inventing a camera for it would put an
        // empty tab on screen for a phone that may never capture.
        const cam = msg.captureId ? findPhysicalCameraByConnection(msg.captureId) : undefined;
        if (cam && Array.isArray(msg.samples)) {
          for (const s of msg.samples) {
            // Positional tuples on the wire (see mobileCapture.ts's send) --
            // ~6 of these arrive 10x/second, and the field names cost more
            // bytes than the numbers do at that rate.
            cam.imuSamples.push({
              t: s[0], rrAlpha: s[1], rrBeta: s[2], rrGamma: s[3],
              ax: s[4], ay: s[5], az: s[6], agx: s[7], agy: s[8], agz: s[9], interval: s[10],
            });
          }
          if (cam.imuSamples.length > IMU_RING_CAPACITY) {
            cam.imuSamples.splice(0, cam.imuSamples.length - IMU_RING_CAPACITY);
          }
          cam.lastImuMeta = {
            screenAngle: msg.screenAngle ?? 0,
            rateHz: msg.rateHz ?? null,
            receivedAt: performance.timeOrigin + performance.now(),
            totalReceived: (cam.lastImuMeta?.totalReceived ?? 0) + msg.samples.length,
          };
        }
      } else if (msg.type === 'configSaved') {
        configStatus.textContent = msg.ok
          ? `saved to ${String(msg.path).split('/').pop()}`
          : `save failed: ${msg.error}`;
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

  // REMOVED: a 1Hz unsolicited frame push that existed only to keep
  // latest-frame.png warm without an explicit request. Do not reinstate it
  // without reading this, because it looked harmless and was not:
  //
  //  - It cost a full window-sized SYNCHRONOUS drawing-buffer readback plus a
  //    JPEG encode on the main thread, once a second, forever, purely so a file
  //    nothing reads programmatically would be at most a second stale.
  //  - It ran during every profiling session, since the bridge is connected
  //    exactly when someone is measuring. Any periodic gap on a flame chart is
  //    the first thing it would have been mistaken for.
  //  - It was WRONG in Through-Cam mode. The pull path (the 'screenshot'
  //    handler above) deliberately switches to throughCamCanvas there, because
  //    renderer.domElement shows only its clear color; this push always grabbed
  //    renderer.domElement, so it overwrote latest-frame.png with a blank frame
  //    every second -- clobbering the good screenshot a pull had just saved.
  //
  // The pull path is complete on its own: server.js's saveFrame writes the same
  // FRAME_PATH on every screenshotResult. Nothing was lost except staleness.
})();
