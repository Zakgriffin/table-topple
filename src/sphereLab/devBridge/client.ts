import { createPhysicalCamera } from '../camera/factory.ts';
import { findPhysicalCameraByConnection, removeCameraTab } from '../camera/lifecycle.ts';
import { activeCamera, cameras, isPhysical, nextCameraColor } from '../camera/store.ts';
import { ingestRealCapture } from '../pipeline/capture.ts';
import { renderer } from '../scene/renderer.ts';
import { renderCameraTabs } from '../ui/cameraPanel.ts';

// Set by initDevBridge once it has a live socket; module-level (rather than
// staying a local var inside that IIFE, like before Stage C) specifically so
// renderCameraTabs' kick button can reach it without threading the socket
// through as a parameter everywhere.
export let devBridgeSocket: WebSocket | null = null;
export function sendToDevBridge(obj: unknown) {
  if (devBridgeSocket && devBridgeSocket.readyState === WebSocket.OPEN) devBridgeSocket.send(JSON.stringify(obj));
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
        // assigns one per 'capture' connection). An unrecognized
        // connectionId auto-creates its own new physical camera/tab --
        // deliberately NOT made active, so a phone connecting in the
        // background doesn't yank focus away from whatever camera the user
        // is currently looking at (contrast addSimulatedCamera, where
        // becoming active IS wanted, since that's a direct user action).
        // A missing connectionId (a stale caller, or a dev-bridge server
        // predating this protocol) falls back to whatever the active camera
        // already is, if it happens to be physical -- purely defensive,
        // shouldn't be reachable against a current server.js.
        const connectionId: string | undefined = msg.captureId;
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
        if (cam) ingestRealCapture(cam, msg.dataUrl).catch((e) => console.error('[realCapture] ingest failed:', e));
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
