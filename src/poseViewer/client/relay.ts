// ── The relay: one websocket, and everything that gates a send on it ─────
//
// Split out of the page's one file. Rides the SAME https origin the page
// itself was loaded from, via vite's /dev-bridge websocket proxy (see
// vite.config.ts) -- a page loaded over https (required here for
// getUserMedia) can't open a plain insecure ws:// connection to a
// non-localhost host, so this can't just point at the dev-bridge's own
// ws://<lan-ip>:8787 directly the way a laptop-local tab can. Works
// identically whether this page happens to be opened via the LAN IP or
// localhost.
//
// ── WHY THIS TAKES A HOST INSTEAD OF IMPORTING ONE ──
//
// Same reason imu.ts does, and the same hazard behind it. Four things the
// socket has to reach are the PAGE's, not the transport's: what to announce on
// (re)connect, what a settingsSync means, what a readiness change does to the
// shutter, and what scope an eval runs in. Importing them would make this
// module and main.ts mutually dependent, and a cycle here is the temporal-dead-
// zone fault this page already hit once (see main.ts). So the page REGISTERS
// them, and the arrow only ever points one way.
//
// Note in particular that `evalCode` is a host field rather than an `eval`
// call written here. A direct eval sees the scope of the module it is
// physically written in, so an eval living here would silently mean "whatever
// relay.ts happens to import" -- which is exactly how the phone's inspectable
// surface degraded during the last split. client/inspect.ts owns that scope
// now and declares it explicitly; main.ts wires the two together.

import { relayStatus } from './dom.ts';

export interface RelayHost {
  /** 'single' | 'video' -- re-announced on every (re)connect. */
  captureMode(): string;
  /** Whether pose is being computed locally -- re-announced on every (re)connect. */
  computingOnDevice(): boolean;
  /** The desktop pushed new settings down. */
  onSettingsSync(msg: any): void;
  /** The desktop reported busy/idle. Drives the shutter's yellow state only. */
  onReadyChange(ready: boolean): void;
  /** Runs one dev-bridge expression in the page's declared inspect scope. */
  evalCode(code: string): unknown;
}

// Null until main.ts attaches. connectRelay() is likewise called by main.ts
// rather than at module scope here, so the socket can never open -- and
// therefore never deliver a message into a null host -- before the page has
// registered one.
let host: RelayHost | null = null;

export function attachRelayHost(h: RelayHost): void {
  host = h;
}

export let ws: WebSocket | null = null;
let reconnectTimer: number | undefined;

export function setRelayStatus(text: string, down: boolean) {
  relayStatus.textContent = `relay: ${text}`;
  relayStatus.classList.toggle('down', down);
}

function scheduleReconnect() {
  ws = null;
  setRelayStatus('reconnecting…', true);
  clearTimeout(reconnectTimer);
  reconnectTimer = window.setTimeout(connectRelay, 2000);
}

// ── Sending ──────────────────────────────────────────────────────────────
//
// Both return whether the message actually went out, so a caller that has to
// say something about a failed send (captureAndSendFrame's "capture NOT sent")
// can, and every other caller can ignore it -- which is what the old inline
// `if (ws && ws.readyState === WebSocket.OPEN)` checks scattered through the
// page amounted to anyway.

export function isRelayOpen(): boolean {
  return !!ws && ws.readyState === WebSocket.OPEN;
}

export function sendJson(payload: unknown): boolean {
  if (!isRelayOpen()) return false;
  ws!.send(JSON.stringify(payload));
  return true;
}

export function sendBinary(payload: Blob | ArrayBuffer | ArrayBufferView): boolean {
  if (!isRelayOpen()) return false;
  ws!.send(payload as any);
  return true;
}

// Returns WHY, not just whether, so the video loop can count each reason
// separately (see main.ts's loopTicks and friends) -- the whole point being to
// tell "blocked because the network hasn't drained the last frame yet"
// (bandwidth-bound) apart from "blocked for some other reason" instead of
// collapsing both into one opaque boolean. Desktop-compute only -- see
// captureComputeAndSendPose's own comment on why device-compute doesn't
// need this gate at all.
export function sendGateStatus(): 'ok' | 'backpressure' | 'not-ready' {
  // Transport-level backpressure, independent of whether Pose Viewer wants
  // more data: bufferedAmount > 0 means the PREVIOUS frame hasn't actually
  // left the device yet (queued by the browser/OS faster than the network
  // can drain it). Without this, an unthrottled video loop (pipelined mode
  // removed the old wait-for-ack gate, which used to cap the send rate as
  // a side effect) will happily encode+queue a new multi-hundred-KB JPEG on
  // every rAF tick regardless -- confirmed live: that produced a growing
  // multi-SECOND queueing delay per frame, dwarfing every other stage in
  // the pipeline. The mailbox on the desktop only helps once a message
  // arrives; it can't do anything about a backlog stuck in transit.
  if (ws && ws.bufferedAmount > 0) return 'backpressure';
  return 'ok';
}
// `canSend()` -- a boolean wrapper reading `sendGateStatus() === 'ok'` -- was
// here and is CUT. Both of its callers had already been rewritten to read the
// reason rather than the boolean (the video loop counts each reason
// separately; the shutter needs to know it was backpressure), so it survived
// only as mentions in comments, which is precisely what an unreferenced export
// looks like: `noUnusedLocals` cannot see one, so it lives on as a closed loop.

// ── Connecting ───────────────────────────────────────────────────────────

export function connectRelay() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  try { ws = new WebSocket(`${proto}//${location.host}/dev-bridge`); }
  catch { scheduleReconnect(); return; }

  ws.addEventListener('open', () => {
    ws!.send(JSON.stringify({ role: 'capture' }));
    // A reconnect gets a brand-new captureId server-side (see server.js),
    // which Pose Viewer will treat as a fresh phone -- re-announce whatever
    // mode was already selected, and drop any stale not-ready state from
    // before the drop, since it belonged to the OLD captureId.
    ws!.send(JSON.stringify({ type: 'captureMode', mode: host!.captureMode() }));
    ws!.send(JSON.stringify({ type: 'computeMode', mode: host!.computingOnDevice() ? 'device' : 'desktop' }));
    host!.onReadyChange(true);
    setRelayStatus('connected', false);
  });
  ws.addEventListener('close', scheduleReconnect);
  ws.addEventListener('error', () => {});
  ws.addEventListener('message', (ev) => {
    let msg: any;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'captureReady') {
      host!.onReadyChange(!!msg.ready);
    } else if (msg.type === 'settingsSync') {
      host!.onSettingsSync(msg);
    } else if (msg.type === 'eval' && msg.id) {
      // ── Dev bridge, phone side ──────────────────────────────────────────
      //
      // Mirrors pose-viewer-server.html's own eval handler so this page can be
      // inspected live. It exists because a RELOAD COSTS FAR MORE HERE than
      // it does on the desktop: it drops the iOS motion-permission grant
      // (which can only be re-obtained from a user gesture), re-negotiates
      // the camera stream, and wipes any in-progress recording. Adding a
      // readout and reloading -- the only option before this -- therefore
      // costs a physical trip to the phone for every question asked of it,
      // and phase B is made of exactly those questions (does this device
      // report captureTime, what does getCapabilities say, what is the real
      // devicemotion rate).
      //
      // SAME SYNCHRONOUS SEMANTICS as the desktop handler, deliberately: it
      // does NOT await promises or queued work, so the race warning in
      // cli.js's header applies here unchanged. Poll a flag across separate
      // eval calls rather than trusting one round trip to have finished
      // anything asynchronous.
      //
      // The expression is evaluated by client/inspect.ts, NOT here -- see this
      // file's header on why that is a host field.
      let result: any, ok = true;
      try {
        result = host!.evalCode(msg.code);
        // Structured-clone can't cross the wire; JSON can't hold a
        // MediaStreamTrack or a cyclic THREE object. Round-tripping through
        // JSON here means a non-serializable answer reports WHY rather than
        // taking the socket down with a serialization throw at send time.
        result = JSON.parse(JSON.stringify(result ?? null));
      } catch (e: any) {
        ok = false;
        result = { error: String(e), stack: e?.stack };
      }
      ws!.send(JSON.stringify({ type: 'evalResult', id: msg.id, ok, value: result }));
    }
  });
}
