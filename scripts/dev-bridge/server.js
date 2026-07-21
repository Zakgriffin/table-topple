// Local debug bridge: relays "run this JS in the page" and "grab a
// screenshot" requests from the CLI (cli.js, invoked by whoever's
// debugging) to whichever browser tab(s) have sphere-lab.html open, and
// relays the results back. Also relays real camera captures from
// mobile-capture.html (a phone, usually reached through vite's /dev-bridge
// websocket proxy -- see vite.config.ts) out to every connected Sphere Lab
// tab, broadcast-style, no request/response pairing needed for those.
//
// Multi-browser-client now (a Set, not a single slot) so a capture can reach
// more than one open Sphere Lab tab at once, per an explicit ask -- eval/
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
  ws.on('message', (raw) => {
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
      if (!latestBrowserSocket || latestBrowserSocket.readyState !== latestBrowserSocket.OPEN) {
        send(ws, { type: msg.type + 'Result', id: msg.id, ok: false, error: 'no browser connected — is sphere-lab.html open?' });
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

    // Browser -> unsolicited low-rate auto frame (keeps latest-frame.png fresh)
    if (msg.type === 'frame' && msg.dataUrl) {
      saveFrame(msg.dataUrl);
      return;
    }

    // Capture source (mobile-capture.html) -> broadcast to EVERY connected
    // Sphere Lab tab, not just the latest one -- this is the one message
    // type meant to fan out. captureId (this connection's own assigned id)
    // rides along so Sphere Lab can tell which phone a photo came from.
    if (msg.type === 'realCapture' && msg.dataUrl) {
      const captureId = captureSockets.get(ws);
      console.log(`[bridge] real capture received from ${captureId}, broadcasting to ${browserSockets.size} browser tab(s)`);
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
    if (msg.type === 'kickCapture' && msg.captureId) {
      for (const [capWs, id] of captureSockets) {
        if (id === msg.captureId) { capWs.close(); break; }
      }
      return;
    }

    // Capture source -> broadcast, same fan-out as realCapture -- announces
    // a video/single toggle flip on the phone so every open Sphere Lab tab
    // can reflect it (and auto-creates a tab the same way realCapture does,
    // so toggling to video before ever taking a photo still shows up).
    if (msg.type === 'captureMode' && msg.mode) {
      const captureId = captureSockets.get(ws);
      for (const bs of browserSockets) {
        if (bs.readyState === bs.OPEN) send(bs, { ...msg, captureId });
      }
      return;
    }

    // Browser -> a specific phone: is Sphere Lab ready to receive/process
    // another frame from it. Routed the same way kickCapture is (find the
    // one capture socket matching captureId), just sent instead of closed.
    if (msg.type === 'captureReady' && msg.captureId) {
      for (const [capWs, id] of captureSockets) {
        if (id === msg.captureId) { send(capWs, { type: 'captureReady', ready: msg.ready }); break; }
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
