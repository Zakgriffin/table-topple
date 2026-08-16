// Local debug bridge: relays "run this JS in the page" and "grab a
// screenshot" requests from the CLI (cli.js, invoked by whoever's
// debugging) to whichever browser tab(s) have pose-viewer-server.html open, and
// relays the results back. Also relays real camera captures from
// mobile-capture.html (a phone, usually reached through vite's /dev-bridge
// websocket proxy -- see vite.config.ts) out to every connected Pose Viewer
// tab, broadcast-style, no request/response pairing needed for those.
//
// Multi-browser-client now (a Set, not a single slot) so a capture can reach
// more than one open Pose Viewer tab at once, per an explicit ask -- eval/
// screenshot requests still only ever go to whichever browser last
// connected, unchanged, since those were never meant to fan out.
//
// N-camera Stage C: each 'capture' connection (a phone) gets a stable
// randomUUID the moment it connects, included as captureId on every
// realCapture message forwarded to browser tabs -- that's what lets Sphere
// Lab tell two simultaneously-connected phones apart and auto-create a tab
// per phone rather than one shared "the real capture". A captureDisconnected
// broadcast fires whenever that connection closes (network drop, tab
// closed, or an explicit kick below), and kickCapture (sent BY a browser
// tab) closes the matching phone connection outright -- its own 'close'
// handler is what actually fires captureDisconnected, so kicking doesn't
// need its own separate broadcast path.

import { WebSocketServer } from 'ws';
import { writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { randomUUID } from 'crypto';

const PORT = 8787;
const FRAME_PATH = path.resolve(path.dirname(fileURLToPath(import.meta.url)), 'latest-frame.png');
// The one config file (see src/poseViewer/config.ts). Two levels up from
// scripts/dev-bridge/ -- the browser cannot write to the project directory
// itself, so promoting the live config onto disk has to come through here.
const CONFIG_PATH = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'pose-viewer.config.json');
// randomUUID()'s canonical string form is always exactly this many ASCII
// chars (8-4-4-4-12 hex digits + 4 hyphens) -- a fixed-width prefix means a
// receiving browser tab (devBridge/client.ts) can slice a binary
// realCapture frame apart with no length byte of its own needed.
const CAPTURE_ID_BYTES = 36;

const wss = new WebSocketServer({ port: PORT });
const browserSockets = new Set();
let latestBrowserSocket = null; // eval/screenshot still target just the most-recently-connected tab
const pending = new Map(); // request id -> controller ws
const captureSockets = new Map(); // capture ws -> its assigned captureId

function send(ws, obj) {
  try { ws.send(JSON.stringify(obj)); } catch { /* socket already closed */ }
}

function saveFrame(dataUrl) {
  const b64 = dataUrl.split(',')[1];
  if (!b64) return;
  writeFileSync(FRAME_PATH, Buffer.from(b64, 'base64'));
}

wss.on('connection', (ws) => {
  ws.on('message', (raw, isBinary) => {
    // A capture source's realCapture image bytes now arrive as a genuine
    // binary WebSocket frame (mobileCapture.ts), not base64-encoded inside
    // a JSON message -- see this session's chat. Only a KNOWN capture
    // connection is trusted to send one (browser/controller connections
    // never do); everything else is JSON, handled below as before. Relayed
    // to every browser tab prefixed with this connection's own captureId
    // (CAPTURE_ID_BYTES fixed-width ASCII, matching devBridge/client.ts's
    // own slice) so a receiving tab can tell which phone a given binary
    // frame came from even with more than one phone connected at once --
    // simple adjacency to the preceding realCapture JSON message isn't
    // enough once multiple capture connections' broadcasts can interleave
    // in a single browser tab's own message stream.
    if (isBinary) {
      const captureId = captureSockets.get(ws);
      if (captureId === undefined || captureId.length !== CAPTURE_ID_BYTES) return;
      const framed = Buffer.concat([Buffer.from(captureId, 'ascii'), raw]);
      for (const bs of browserSockets) {
        if (bs.readyState === bs.OPEN) bs.send(framed);
      }
      return;
    }
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }

    if (msg.role === 'browser') {
      browserSockets.add(ws);
      latestBrowserSocket = ws;
      console.log(`[bridge] browser connected (${browserSockets.size} total)`);
      return;
    }
    if (msg.role === 'controller') return;
    if (msg.role === 'capture') {
      const captureId = randomUUID();
      captureSockets.set(ws, captureId);
      console.log(`[bridge] capture source connected (id ${captureId})`);
      return;
    }

    // Controller -> browser (targets only the latest tab -- eval/screenshot
    // were never meant to fan out to multiple tabs at once)
    if ((msg.type === 'eval' || msg.type === 'screenshot') && msg.id) {
      pending.set(msg.id, ws);
      // msg.target routes an eval to a PHONE (mobile-capture.html) instead of
      // a Pose Viewer tab -- 'phone' for the only connected one, or an explicit
      // captureId when more than one is attached. Screenshot stays
      // browser-only: it reads the desktop's THREE canvas, which has no
      // counterpart on the phone.
      //
      // Worth the routing branch because a phone reload is expensive in a way
      // a desktop reload is not (iOS motion permission has to be re-granted
      // from a user gesture, and the camera stream re-negotiates) -- see
      // mobileCapture.ts's own eval handler.
      if (msg.type === 'eval' && msg.target) {
        const entries = [...captureSockets.entries()].filter(([s]) => s.readyState === s.OPEN);
        const match = msg.target === 'phone'
          ? (entries.length === 1 ? entries[0] : null)
          : entries.find(([, id]) => id === msg.target);
        if (!match) {
          const why = entries.length === 0
            ? 'no phone connected — is mobile-capture.html open?'
            : msg.target === 'phone'
              ? `${entries.length} phones connected — pass --phone=<captureId>: ${entries.map(([, id]) => id).join(', ')}`
              : `no phone with captureId ${msg.target}`;
          send(ws, { type: 'evalResult', id: msg.id, ok: false, error: why });
          pending.delete(msg.id);
          return;
        }
        send(match[0], msg);
        return;
      }
      if (!latestBrowserSocket || latestBrowserSocket.readyState !== latestBrowserSocket.OPEN) {
        send(ws, { type: msg.type + 'Result', id: msg.id, ok: false, error: 'no browser connected — is pose-viewer-server.html open?' });
        pending.delete(msg.id);
        return;
      }
      send(latestBrowserSocket, msg);
      return;
    }

    // Browser -> controller (response to a request)
    if (msg.type === 'evalResult' || msg.type === 'screenshotResult') {
      if (msg.type === 'screenshotResult' && msg.dataUrl) saveFrame(msg.dataUrl);
      const controllerWs = pending.get(msg.id);
      pending.delete(msg.id);
      if (controllerWs) {
        const { dataUrl, ...rest } = msg;
        send(controllerWs, msg.type === 'screenshotResult' ? { ...rest, path: FRAME_PATH } : msg);
      }
      return;
    }

    // Browser -> unsolicited auto frame. NOTHING SENDS THIS ANYMORE -- the 1Hz
    // pusher that did was removed (see devBridge/client.ts for why: a
    // synchronous full-canvas readback + JPEG encode every second, into a file
    // nothing reads, which it also corrupted in Through-Cam mode). Kept as a
    // no-cost inbox so an older tab left open against a newer server still
    // works rather than logging an unknown-message warning.
    if (msg.type === 'frame' && msg.dataUrl) {
      saveFrame(msg.dataUrl);
      return;
    }

    // Capture source (mobile-capture.html) -> broadcast to EVERY connected
    // Pose Viewer tab, not just the latest one -- this is the one message
    // type meant to fan out. captureId (this connection's own assigned id)
    // rides along so Pose Viewer can tell which phone a photo came from.
    // realCapture no longer carries the image itself (see the isBinary
    // branch above) -- just sentAt/pulledAt/encodedAt, which the receiving
    // tab holds onto until the binary frame carrying this same captureId
    // arrives right behind it. poseResult (on-device-compute mode -- see
    // this session's on-device-pose-recovery plan) rides the exact same
    // broadcast: a phone that computed its own pose sends
    // {w,h,recoveredAxes,positionDecode} instead, no image bytes at all
    // either way, but routing is identical, so this is a condition change,
    // not a new block.
    if (msg.type === 'realCapture' || msg.type === 'poseResult') {
      const captureId = captureSockets.get(ws);
      console.log(`[bridge] ${msg.type} received from ${captureId}, broadcasting to ${browserSockets.size} browser tab(s)`);
      for (const bs of browserSockets) {
        if (bs.readyState === bs.OPEN) send(bs, { ...msg, captureId });
      }
      return;
    }

    // Browser -> kick a specific phone connection outright (not just "stop
    // showing its tab locally", see this file's header comment). Closing it
    // here is the only work needed -- the 'close' handler below fires the
    // captureDisconnected broadcast every capture-socket close already goes
    // through, kick or not.
    // Browser tab -> disk: the "save config to disk" button. `json` is the
    // already-serialized config, so what lands on disk is byte-for-byte what
    // the page is holding rather than this process's idea of how to format it.
    // Replies to the SENDING tab only -- other tabs have their own live
    // localStorage overlay and would be wrong to claim they just saved.
    if (msg.type === 'saveConfig' && typeof msg.json === 'string') {
      try {
        writeFileSync(CONFIG_PATH, msg.json);
        console.log(`[bridge] wrote ${CONFIG_PATH} (${msg.json.length} bytes)`);
        send(ws, { type: 'configSaved', ok: true, path: CONFIG_PATH });
      } catch (err) {
        console.error('[bridge] config write failed:', err);
        send(ws, { type: 'configSaved', ok: false, error: String(err && err.message ? err.message : err) });
      }
      return;
    }

    if (msg.type === 'kickCapture' && msg.captureId) {
      for (const [capWs, id] of captureSockets) {
        if (id === msg.captureId) { capWs.close(); break; }
      }
      return;
    }

    // Capture source -> broadcast, same fan-out as realCapture -- announces
    // a video/single toggle flip on the phone so every open Pose Viewer tab
    // can reflect it (and auto-creates a tab the same way realCapture does,
    // so toggling to video before ever taking a photo still shows up).
    if (msg.type === 'captureMode' && msg.mode) {
      const captureId = captureSockets.get(ws);
      for (const bs of browserSockets) {
        if (bs.readyState === bs.OPEN) send(bs, { ...msg, captureId });
      }
      return;
    }

    // Capture source -> broadcast, same fan-out as realCapture/captureMode --
    // announces a desktop-compute/device-compute toggle flip on the phone
    // (see this session's on-device-pose-recovery plan), mirroring
    // captureMode's own single/video toggle exactly -- sent on toggle
    // change and on reconnect, auto-creates a tab the same way.
    if (msg.type === 'computeMode' && msg.mode) {
      const captureId = captureSockets.get(ws);
      for (const bs of browserSockets) {
        if (bs.readyState === bs.OPEN) send(bs, { ...msg, captureId });
      }
      return;
    }

    // Capture source -> broadcast, same fan-out as realCapture/captureMode --
    // periodic self-reported stats on how often the phone's camera hardware
    // is actually delivering new frames (see mobileCapture.ts's
    // requestVideoFrameCallback loop), for telling "the round trip is slow"
    // apart from "the phone's camera isn't producing frames any faster than
    // this in the first place."
    if (msg.type === 'frameStats') {
      const captureId = captureSockets.get(ws);
      for (const bs of browserSockets) {
        if (bs.readyState === bs.OPEN) send(bs, { ...msg, captureId });
      }
      return;
    }

    // Capture source -> broadcast, same fan-out as frameStats: batched
    // accelerometer/gyro samples from the phone (mobileCapture.ts's
    // devicemotion handler), ~10 messages/second each carrying ~6 samples.
    // Recording only for now -- nothing in the pose pipeline consumes these
    // yet; see the motion-blur/IMU plan's phase A.
    //
    // Deliberately NOT logged per message the way realCapture/poseResult are:
    // this arrives at 10Hz continuously for as long as the toggle is on, and
    // a console line per batch would bury every other bridge message.
    if (msg.type === 'imu') {
      const captureId = captureSockets.get(ws);
      for (const bs of browserSockets) {
        if (bs.readyState === bs.OPEN) send(bs, { ...msg, captureId });
      }
      return;
    }

    // Browser -> a specific phone: is Pose Viewer ready to receive/process
    // another frame from it (the
    // phone uses that to decide if "not ready" should still block sending
    // -- see mobileCapture.ts). Routed the same way kickCapture is (find
    // the one capture socket matching captureId), just sent instead of
    // closed.
    if (msg.type === 'captureReady' && msg.captureId) {
      for (const [capWs, id] of captureSockets) {
        if (id === msg.captureId) { send(capWs, { type: 'captureReady', ready: msg.ready }); break; }
      }
      return;
    }

    // Browser -> a specific phone: pushes the current pipeline-tunable
    // settings (globalState.useGPU*/boardSize + the 16 PoseInput
    // settings fields) so a phone computing its own pose stays in sync with
    // whatever the desktop's sliders (the source of truth) currently say --
    // see this session's on-device-pose-recovery plan. Routed the same way
    // captureReady is (find the one capture socket matching captureId, send
    // directly), not broadcast -- each physical camera's settings are its
    // own.
    if (msg.type === 'settingsSync' && msg.captureId) {
      for (const [capWs, id] of captureSockets) {
        if (id === msg.captureId) { send(capWs, msg); break; }
      }
      return;
    }

    // Browser -> a specific phone: the exact mirror image of realCapture/
    // poseResult (capture -> browser) -- the desktop ships its own
    // recovered pose back down to the phone that sent the image, so the
    // phone's AR overlay (mobile-capture.html) can work even when it isn't
    // computing a pose itself. Routed identically to settingsSync/
    // captureReady (find the one capture socket matching captureId, send
    // directly), not broadcast -- a pose only ever means something to the
    // one phone it was recovered from.
    if (msg.type === 'poseSync' && msg.captureId) {
      for (const [capWs, id] of captureSockets) {
        if (id === msg.captureId) { send(capWs, msg); break; }
      }
      return;
    }
  });

  ws.on('close', () => {
    if (browserSockets.delete(ws)) {
      if (latestBrowserSocket === ws) latestBrowserSocket = null;
      console.log(`[bridge] browser disconnected (${browserSockets.size} remaining)`);
      return;
    }
    const captureId = captureSockets.get(ws);
    if (captureId !== undefined) {
      captureSockets.delete(ws);
      console.log(`[bridge] capture source disconnected (id ${captureId})`);
      for (const bs of browserSockets) {
        if (bs.readyState === bs.OPEN) send(bs, { type: 'captureDisconnected', captureId });
      }
    }
  });
});

console.log(`[bridge] listening on ws://localhost:${PORT}`);
console.log(`[bridge] frames saved to ${FRAME_PATH}`);
