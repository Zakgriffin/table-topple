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

import { WebSocketServer } from 'ws';
import { writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const PORT = 8787;
const FRAME_PATH = path.resolve(path.dirname(fileURLToPath(import.meta.url)), 'latest-frame.png');

const wss = new WebSocketServer({ port: PORT });
const browserSockets = new Set();
let latestBrowserSocket = null; // eval/screenshot still target just the most-recently-connected tab
const pending = new Map(); // request id -> controller ws

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
    if (msg.role === 'controller' || msg.role === 'capture') {
      if (msg.role === 'capture') console.log('[bridge] capture source connected');
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
    // type meant to fan out.
    if (msg.type === 'realCapture' && msg.dataUrl) {
      console.log(`[bridge] real capture received, broadcasting to ${browserSockets.size} browser tab(s)`);
      for (const bs of browserSockets) {
        if (bs.readyState === bs.OPEN) send(bs, msg);
      }
      return;
    }
  });

  ws.on('close', () => {
    if (browserSockets.delete(ws)) {
      if (latestBrowserSocket === ws) latestBrowserSocket = null;
      console.log(`[bridge] browser disconnected (${browserSockets.size} remaining)`);
    }
  });
});

console.log(`[bridge] listening on ws://localhost:${PORT}`);
console.log(`[bridge] frames saved to ${FRAME_PATH}`);
