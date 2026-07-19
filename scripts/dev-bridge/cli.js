// Talks to server.js to either eval arbitrary JS inside the live
// sphere-lab.html page (direct eval in its module scope — sees state,
// scene, camPos, gizmoCam, everything declared top-level in sphereLab.ts)
// or pull a fresh screenshot of the canvas.
//
// Usage:
//   node scripts/dev-bridge/cli.js eval "state.camYawDeg"
//   node scripts/dev-bridge/cli.js screenshot
//
// For real-capture testing: save-capture.mjs / restore-capture.mjs let a
// page reload (e.g. to pick up a change eval can't hot-apply, like a new
// module-scope function) be followed by a scripted restore of the last real
// photo instead of re-taking one on the phone every time. Round-trips
// through byte-rounded grayscale (not the original photo), so decode
// results after a restore may differ by a tiny amount from before -- fine
// for comparative testing, not pixel-exact.
//
// RACE CONDITION WARNING, learned the hard way: sphereLab.ts's eval handler
// runs `eval(msg.code)` synchronously and replies immediately with whatever
// that expression returns -- it does NOT await promises or wait for queued
// work. runAxesReconstruction() in particular queues its real work via
// requestAnimationFrame, then (if state.positionLM) runs a further async LM
// refinement pass after that -- so calling it and reading results (even a
// few hundred ms later via a fixed sleep) can silently read STALE state from
// whatever the LAST completed run left behind, not the one you just
// triggered. A whole investigation session was derailed by exactly this: a
// sweep over 8 flip-toggle combinations looked byte-identical across all 8
// (seemingly proving decode was fully invariant to axis labeling) purely
// because none of the later triggers had actually finished recomputing
// before their results were read back synchronously in the same script.
// The fix is to poll `axesCapturing` (a module-level flag, true while the
// rAF + LM pipeline is in flight, false once genuinely done) via SEPARATE
// eval calls after triggering, and only read results once it's false again
// -- see the sweep loops that use this pattern for a live example. A fixed
// sleep is not reliable even for a single combo: the LM refinement stage's
// own runtime varies, so a delay long enough for a fast fit can still catch
// a slower one mid-flight.
//
// KEEP THE BROWSER WINDOW/TAB IN FOCUS while triggering a capture from this
// CLI. runAxesReconstruction's requestAnimationFrame callback is subject to
// the same throttling any backgrounded tab gets -- a capture that normally
// takes a few seconds can instead sit at axesCapturing=true for 20-30+
// seconds (still eventually completes, not a hang) if the window lost focus
// right as it was triggered. Confirmed live: a capture that looked
// completely stuck resolved itself the moment the window was refocused.

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
