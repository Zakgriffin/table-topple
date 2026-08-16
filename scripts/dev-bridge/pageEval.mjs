// One websocket round-trip to the live page, as a function.
//
// cli.js, save-fixture.mjs, restore-fixture.mjs, profile-comparison.mjs and
// several one-off profiling scripts each grew their own copy of "connect, announce
// controller, send an eval, match the id, close" -- five copies of about
// twenty lines. This is that, once. The fixture scripts use it; the others
// still carry their own copies and can be moved over when they are next
// touched.
//
// The code is sent in the MESSAGE BODY rather than through cli.js's
// argv-based interface on purpose: a fixture's grayscale is megabytes of
// base64 in either direction, which exceeds OS argv limits.

import { WebSocket } from 'ws';

const PORT = 8787;

// Resolves with the page's returned value (whatever the page's eval handler
// stringified), rejects on a page-side throw or on timeout.
//
// The default timeout is generous compared with cli.js's 8s because a fixture
// round-trip is megabytes, and because the page may be mid-reconstruction:
// see cli.js's own header on rAF throttling in a backgrounded tab.
export function evalInPage(code, { timeoutMs = 20000 } = {}) {
  return new Promise((resolve, reject) => {
    const id = Math.random().toString(36).slice(2);
    const ws = new WebSocket(`ws://localhost:${PORT}`);
    const timer = setTimeout(() => {
      ws.close();
      reject(new Error('timed out waiting for the page -- is server.js running, and is pose-viewer-server.html open in a browser?'));
    }, timeoutMs);

    ws.on('open', () => {
      ws.send(JSON.stringify({ role: 'controller' }));
      ws.send(JSON.stringify({ type: 'eval', id, code }));
    });
    ws.on('message', (raw) => {
      const msg = JSON.parse(raw.toString());
      if (msg.id !== id) return;
      clearTimeout(timer);
      ws.close();
      if (!msg.ok) reject(new Error(`page eval failed: ${msg.error}`));
      else resolve(msg.value);
    });
    ws.on('error', (e) => { clearTimeout(timer); reject(new Error(`bridge connection failed: ${e.message}`)); });
  });
}

// The same, for evals that return a JSON string -- which is every one of them
// here, since the page's eval handler replies with whatever the expression
// evaluates to and an object would not survive the trip.
//
// A page-side `{ error }` becomes a rejection rather than a value: the two
// failure modes (the page threw, and the page declined) are the same failure
// to a caller, and forcing every caller to remember to check the second is how
// one of them ends up writing a fixture with `undefined` in it.
export async function evalJsonInPage(code, opts) {
  const parsed = JSON.parse(await evalInPage(code, opts));
  if (parsed && parsed.error) throw new Error(parsed.error);
  return parsed;
}
