// Restores a capture previously saved by save-capture.mjs into the live
// sphere-lab.html page -- lets a page reload (e.g. to pick up a change HMR
// missed) be followed by a scripted restore instead of asking for a new
// phone photo every time. Mirrors ingestRealCapture's own post-decode steps
// (resizeCaptureBuffers if the size doesn't match, updateDistortedPreview,
// runAxesReconstruction) since this is deliberately re-entering that same
// pipeline partway through, not a separate path that could drift out of sync
// with it.
//
// Usage: node scripts/dev-bridge/restore-capture.mjs

import { WebSocket } from 'ws';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const PORT = 8787;
const IN_PATH = join(dirname(fileURLToPath(import.meta.url)), 'saved-capture.json');

const data = JSON.parse(readFileSync(IN_PATH, 'utf8'));

const EVAL_CODE = `
(function() {
  const w = ${data.w}, h = ${data.h};
  const binary = atob(${JSON.stringify(data.b64)});
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const gray = new Float64Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) gray[i] = bytes[i];
  if (w !== rtSize.w || h !== rtSize.h) resizeCaptureBuffers({ w, h });
  lastRealCaptureGray = gray;
  lastRealCaptureW = w; lastRealCaptureH = h;
  updateDistortedPreview();
  if (state.mode === 'projected') buildProjectedTexture();
  if (state.useRealCapture) runAxesReconstruction();
  return JSON.stringify({ restored: true, w, h });
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
  console.log(msg.value);
  process.exit(0);
});

ws.on('error', (e) => { console.error('bridge connection failed:', e.message); process.exit(1); });
