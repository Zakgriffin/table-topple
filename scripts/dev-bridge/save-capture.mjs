// Saves the currently-loaded real capture (activeCamera().lastRealCaptureGray/
// W/H from sphereLab.ts) to scripts/dev-bridge/saved-capture.json, so a page reload
// (needed to pick up hot-reload-missed changes, or just for a clean restart)
// doesn't force re-taking a photo on the phone every time -- see
// restore-capture.mjs for the other half. Sends the eval's code over the
// websocket message body rather than through cli.js's argv-based interface,
// since the grayscale buffer (~300-800KB base64'd) can exceed OS argv limits.
//
// Usage: node scripts/dev-bridge/save-capture.mjs

import { WebSocket } from 'ws';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const PORT = 8787;
const OUT_PATH = join(dirname(fileURLToPath(import.meta.url)), 'saved-capture.json');

// Stage A (N-camera refactor): lastRealCaptureGray/W/H moved from
// module-level globals onto the active camera object -- see
// activeCamera()/PhysicalCamera in sphereLab.ts.
const EVAL_CODE = `
(function() {
  const cam = activeCamera();
  if (!cam || cam.type !== 'physical') return JSON.stringify({ error: 'active camera is not a physical camera (toggle "use real capture" first)' });
  if (!cam.lastRealCaptureGray) return JSON.stringify({ error: 'no capture loaded' });
  const arr = cam.lastRealCaptureGray;
  const bytes = new Uint8Array(arr.length);
  for (let i = 0; i < arr.length; i++) bytes[i] = Math.max(0, Math.min(255, Math.round(arr[i])));
  let binary = '';
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return JSON.stringify({ w: cam.lastRealCaptureW, h: cam.lastRealCaptureH, b64: btoa(binary) });
})()
`;

const id = Math.random().toString(36).slice(2);
const ws = new WebSocket(`ws://localhost:${PORT}`);

const timeout = setTimeout(() => {
  console.error('timeout waiting for a response -- is server.js running, and is sphere-lab.html open in a browser?');
  process.exit(1);
}, 15000);

ws.on('open', () => {
  ws.send(JSON.stringify({ role: 'controller' }));
  ws.send(JSON.stringify({ type: 'eval', id, code: EVAL_CODE }));
});

ws.on('message', (raw) => {
  const msg = JSON.parse(raw.toString());
  if (msg.id !== id) return;
  clearTimeout(timeout);
  ws.close();
  if (!msg.ok) { console.error('eval failed:', msg.error); process.exit(1); }
  const data = JSON.parse(msg.value);
  if (data.error) { console.error(data.error); process.exit(1); }
  writeFileSync(OUT_PATH, JSON.stringify(data));
  console.log(`Saved ${data.w}x${data.h} capture (${(data.b64.length / 1024).toFixed(0)}KB base64) to ${OUT_PATH}`);
  process.exit(0);
});

ws.on('error', (e) => { console.error('bridge connection failed:', e.message); process.exit(1); });
