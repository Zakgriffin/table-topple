// Talks to server.js to either eval arbitrary JS inside the live
// sphere-lab.html page (direct eval in its module scope — sees state,
// scene, camPos, gizmoCam, everything declared top-level in sphereLab.ts)
// or pull a fresh screenshot of the canvas.
//
// Usage:
//   node scripts/dev-bridge/cli.js eval "state.camYawDeg"
//   node scripts/dev-bridge/cli.js screenshot

import { WebSocket } from 'ws';

const PORT = 8787;
const [, , cmd, ...rest] = process.argv;

if (!cmd || !['eval', 'screenshot'].includes(cmd)) {
  console.error('usage: node cli.js eval "<js code>" | node cli.js screenshot');
  process.exit(1);
}

const id = Math.random().toString(36).slice(2);
const ws = new WebSocket(`ws://localhost:${PORT}`);

const timeout = setTimeout(() => {
  console.error('timeout waiting for a response — is server.js running, and is sphere-lab.html open in a browser?');
  process.exit(1);
}, 8000);

ws.on('open', () => {
  ws.send(JSON.stringify({ role: 'controller' }));
  ws.send(cmd === 'eval'
    ? JSON.stringify({ type: 'eval', id, code: rest.join(' ') })
    : JSON.stringify({ type: 'screenshot', id }));
});

ws.on('message', (raw) => {
  const msg = JSON.parse(raw.toString());
  if (msg.id !== id) return;
  clearTimeout(timeout);
  console.log(JSON.stringify(msg, null, 2));
  ws.close();
  process.exit(msg.ok === false ? 1 : 0);
});

ws.on('error', (e) => { console.error('bridge connection failed:', e.message); process.exit(1); });
