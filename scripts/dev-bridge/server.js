// Local debug bridge: relays "run this JS in the page" and "grab a
// screenshot" requests from the CLI (cli.js, invoked by whoever's
// debugging) to whichever browser tab has sphere-lab.html open, and relays
// the results back. Single-browser-client assumption — this is a personal
// dev tool, not a multi-user service. Never exposed beyond localhost.

import { WebSocketServer } from 'ws';
import { writeFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const PORT = 8787;
const FRAME_PATH = path.resolve(path.dirname(fileURLToPath(import.meta.url)), 'latest-frame.png');

const wss = new WebSocketServer({ port: PORT });
let browserSocket = null;
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
      browserSocket = ws;
      console.log('[bridge] browser connected');
      return;
    }
    if (msg.role === 'controller') return;

    // Controller -> browser
    if ((msg.type === 'eval' || msg.type === 'screenshot') && msg.id) {
      pending.set(msg.id, ws);
      if (!browserSocket || browserSocket.readyState !== browserSocket.OPEN) {
        send(ws, { type: msg.type + 'Result', id: msg.id, ok: false, error: 'no browser connected — is sphere-lab.html open?' });
        pending.delete(msg.id);
        return;
      }
      send(browserSocket, msg);
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
  });

  ws.on('close', () => {
    if (ws === browserSocket) { browserSocket = null; console.log('[bridge] browser disconnected'); }
  });
});

console.log(`[bridge] listening on ws://localhost:${PORT}`);
console.log(`[bridge] frames saved to ${FRAME_PATH}`);
