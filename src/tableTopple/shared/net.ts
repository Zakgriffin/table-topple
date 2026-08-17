import { Value } from '@sinclair/typebox/value';
import { GameMessageSchema, type GameMessage } from './protocol.ts';

// ── The websocket: one connection, and everything that gates a send on it ──
//
// Same shape as Pose Viewer's client/relay.ts, for the same reason: this
// module takes a HOST rather than importing sim.ts/main.ts directly, so the
// arrow only ever points one way and this file can be imported from
// shared/combat.ts (desktop-only at runtime, see combat.ts's own header)
// without ever pulling game logic into the transport, or the transport into
// game logic. See relay.ts's header for the fuller argument -- it is the same
// hazard, a genuine import cycle risking a temporal-dead-zone fault, not
// stylistic preference.
//
// Rides the SAME origin the page was loaded from, through vite's own
// /tabletopple-ws proxy (see vite.config.ts's plugin) rather than a second
// port -- there is no separate relay process to point at; the plugin attaches
// straight to vite's dev server, so this comes up for free with `npm run dev`.
//
// ── WHAT NEVER BECOMES A GameMessage ──
//
// The `{ role }` sent right after `open` is transport, not a game message --
// protocol.ts's union never grows a 'hello' or 'ping' case, and the relay
// itself only ever looks at that first frame before switching to pure
// broadcast (see the plugin). Keeping that boundary here is what lets
// protocol.ts grow without this file ever changing.

export type Role = 'desktop' | 'phone';

export interface NetHost {
  /** A validated GameMessage arrived. Never called with anything that failed
   *  GameMessageSchema -- see the message listener below. */
  onMessage(msg: GameMessage): void;
}

// Null until a host attaches, and connect() is only ever called by a page's
// own boot -- same reasoning as relay.ts's `host`: the socket must not be
// able to deliver a message into nothing.
let host: NetHost | null = null;
export function attachNetHost(h: NetHost): void {
  host = h;
}

let ws: WebSocket | null = null;
let role: Role | null = null;
let reconnectTimer: number | undefined;

export function isConnected(): boolean {
  return !!ws && ws.readyState === WebSocket.OPEN;
}

/** Returns whether the message actually went out, same convention as
 *  relay.ts's sendJson -- a caller with nothing useful to do about a dropped
 *  send (the desktop mid-tick, with no connected phones yet) can ignore it. */
export function send(msg: GameMessage): boolean {
  if (!isConnected()) return false;
  if (!Value.Check(GameMessageSchema, msg)) {
    // A caller-side bug, not a network fault -- logged rather than thrown, so
    // one malformed message can't take the render loop down with it.
    console.error('net.ts: refusing to send a message that fails GameMessageSchema', msg);
    return false;
  }
  ws!.send(JSON.stringify(msg));
  return true;
}

function scheduleReconnect() {
  ws = null;
  clearTimeout(reconnectTimer);
  // Fixed delay, no backoff -- same as relay.ts. A LAN dev relay is either up
  // or it isn't; there is no remote service to be polite to.
  reconnectTimer = window.setTimeout(() => connect(role!), 2000);
}

export function connect(r: Role): void {
  role = r;
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  try { ws = new WebSocket(`${proto}//${location.host}/tabletopple-ws`); }
  catch { scheduleReconnect(); return; }

  ws.addEventListener('open', () => ws!.send(JSON.stringify({ role: r })));
  ws.addEventListener('close', scheduleReconnect);
  ws.addEventListener('error', () => {});
  ws.addEventListener('message', (ev) => {
    let parsed: unknown;
    try { parsed = JSON.parse(ev.data); } catch { return; }
    // Dropped rather than passed through loosely-typed: a message that fails
    // the schema is either a relay-side bug or a version skew between the two
    // hosts, and rendering half of a malformed denizenState is worse than
    // rendering none of it.
    if (!Value.Check(GameMessageSchema, parsed)) return;
    host?.onMessage(parsed as GameMessage);
  });
}
