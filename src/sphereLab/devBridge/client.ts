import { PhysicalCamera } from '../camera/model.ts';
import { createPhysicalCamera } from '../camera/factory.ts';
import { findPhysicalCameraByConnection, removeCameraTab } from '../camera/lifecycle.ts';
import { activeCamera, cameras, isPhysical, nextCameraColor } from '../camera/store.ts';
import { ingestRemotePose } from '../pipeline/capture.ts';
import { renderer } from '../scene/renderer.ts';
import { globalState } from '../state.ts';
import { renderCameraTabs, refreshCameraPanel } from '../ui/cameraPanel.ts';

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
    gridPeriodPhaseGapLowerBound: s.gridPeriodPhaseGapLowerBound, minGrazingCos: s.minGrazingCos,
    lsdToleranceDeg: s.lsdToleranceDeg, lsdRhoNoiseThreshold: s.lsdRhoNoiseThreshold,
    lsdMagnitudeBuckets: s.lsdMagnitudeBuckets, lsdNfaEpsilon: s.lsdNfaEpsilon,
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

    ws.addEventListener('open', () => ws.send(JSON.stringify({ role: 'browser' })));
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
        ws.send(JSON.stringify({ type: 'evalResult', id: msg.id, ok, value, error }));
      } else if (msg.type === 'screenshot') {
        const dataUrl = renderer.domElement.toDataURL('image/png');
        ws.send(JSON.stringify({ type: 'screenshotResult', id: msg.id, ok: true, dataUrl }));
      } else if (msg.type === 'realCapture' && msg.dataUrl) {
        // Broadcast from mobile-capture.html via the dev-bridge relay,
        // tagged with the sending phone's own connectionId (server.js
        // assigns one per 'capture' connection). See findOrCreatePhysicalCamera
        // above for the auto-create-but-don't-activate reasoning.
        const cam = findOrCreatePhysicalCamera(msg.captureId);
        // Mailbox, not a queue -- always overwrite with the freshest frame.
        // main.ts's animate loop pumps this once the camera is actually
        // free. Unconditional (not gated on useCapturePipelining): a direct
        // ingestRealCapture call here could race a second one arriving
        // before the first's async image decode finishes, whereas the
        // mailbox+captureIngestBusy guard in main.ts can't overlap two
        // decodes of the same camera regardless of the toggle -- the
        // toggle only controls what the phone is told about whether it's
        // safe to send while busy, not how the desktop schedules ingest.
        // msg.sentAt/pulledAt/encodedAt (Date.now() on the phone, see
        // mobileCapture.ts) fall back to "now" for an old/unpatched phone
        // client so the derived durations degrade to ~0 instead of NaN.
        if (cam) {
          const now = Date.now();
          cam.pendingCapture = {
            dataUrl: msg.dataUrl, sentAt: msg.sentAt ?? now, pulledAt: msg.pulledAt ?? now, encodedAt: msg.encodedAt ?? now,
            receivedAt: now, bytes: msg.dataUrl.length,
          };
        }
      } else if (msg.type === 'poseResult') {
        // Same broadcast/routing as realCapture (see server.js), sent
        // instead of it when the phone is in device-compute mode -- the
        // phone already ran the full pose-recovery pipeline itself and is
        // handing over just {w,h,recoveredAxes,positionDecode}, no image
        // bytes at all, UNLESS the phone's own sendDebugInfo/
        // sendCapturedImage toggles are on (both default off), in which
        // case msg.debug/msg.dataUrl ride along too -- see
        // mobileCapture.ts's own comments. No mailbox needed for the
        // no-image case (ingestRemotePose is cheap, no async decode work)
        // -- but decoding an optional debug dataUrl IS async, so
        // ingestRemotePose itself is now async and called fire-and-forget
        // here, same pattern as ingestRealCapture's own call site.
        const cam = findOrCreatePhysicalCamera(msg.captureId);
        if (cam) {
          ingestRemotePose(cam, {
            w: msg.w, h: msg.h, recoveredAxes: msg.recoveredAxes ?? null, positionDecode: msg.positionDecode ?? null,
            debug: msg.debug, dataUrl: msg.dataUrl,
          }).catch((e) => console.error('[poseResult] ingest failed:', e));
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
