import { defineConfig, type Plugin } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';
import { WebSocketServer, type WebSocket } from 'ws';

// ── Table Topple's game-message relay ───────────────────────────────────────
//
// A dumb broadcaster, not a game server: it never parses a message's `type`,
// so shared/protocol.ts's union can grow without this plugin ever changing
// (see shared/net.ts's own header). Deliberately a vite plugin rather than a
// standalone script like scripts/dev-bridge/server.js -- there is exactly one
// authoritative host (table-topple-server.html) and any number of receivers,
// and that pairing wants to come up for free with `npm run dev`, not need a
// second terminal every time.
//
// Attaches straight to vite's own dev HTTP server via configureServer, the
// same server vite's HMR websocket already rides -- `noServer: true` plus a
// manual 'upgrade' listener gated on the URL path is what lets the two
// coexist: vite's own listener ignores an upgrade it doesn't recognize, and
// this one only ever acts on '/tabletopple-ws'.
//
// The first message from a socket is its role declaration (transport, not a
// game message -- see net.ts) and is consumed, never broadcast. Every message
// after that is forwarded verbatim to every OTHER open socket -- there is
// only one sender in practice (the desktop), so "every other socket" and
// "every phone" are the same set today; a second desktop or a phone that
// starts sending its own messages both fall out of this for free, with no
// relay change.
function tableToppleRelayPlugin(): Plugin {
  return {
    name: 'table-topple-relay',
    configureServer(server) {
      const wss = new WebSocketServer({ noServer: true });

      wss.on('connection', (socket: WebSocket) => {
        let registered = false;
        socket.on('message', (data, isBinary) => {
          if (isBinary) return; // this protocol is JSON-only, see protocol.ts
          if (!registered) { registered = true; return; } // the role handshake, consumed
          for (const other of wss.clients) {
            if (other !== socket && other.readyState === other.OPEN) other.send(data.toString());
          }
        });
      });

      server.httpServer?.on('upgrade', (req, socket, head) => {
        if (!req.url?.startsWith('/tabletopple-ws')) return;
        wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
      });
    },
  };
}

// HTTPS with a self-signed cert so getUserMedia() works when testing over
// LAN from a phone (mobile browsers only expose camera APIs in a secure
// context: https, or localhost). You'll need to accept the one-time
// certificate warning in the phone's browser.
export default defineConfig({
  plugins: [basicSsl(), tableToppleRelayPlugin()],
  server: {
    // CROSS-ORIGIN ISOLATION, and it is here for the CLOCK. Chrome coarsens
    // performance.now() to 100us by default and to 5us when the context is
    // cross-origin isolated -- a 20x sharper clock for every number the perf
    // harnesses report. It is not a micro-optimization of the measurement: at
    // 100us, the old pipeline's gpu/device.ts's allocation probe sums 78 sub-resolution
    // deltas and rounds every one of them to zero, which is how it reported
    // 0.60ms one session and 0.00ms the next for identical work.
    //
    // A header rather than a Chrome flag on purpose. Flags live in one
    // developer's browser, drift silently (chrome://flags' "Enable
    // benchmarking" auto-resets after 3 restarts while still appearing on),
    // and cannot be reviewed. This applies to every machine and every session
    // and is visible in the diff.
    //
    // require-corp blocks cross-origin subresources that do not opt in.
    // Nothing here loads any -- three.js and everything else is bundled by
    // vite, and the dev-bridge WebSocket rides the same-origin proxy below
    // (WebSockets are not subject to COEP regardless). If that ever changes
    // the failure is loud, and the fix is deleting these two lines.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    proxy: {
      // Proxies the dev-bridge relay's websocket through this same HTTPS
      // origin -- pose-viewer-client.html is loaded over https (required for
      // getUserMedia on a phone), and a page loaded over https can't open a
      // plain insecure ws:// connection to anything except localhost
      // (mixed-content blocking). Riding this proxy means the phone only
      // ever has to trust the ONE cert it already accepted for the page
      // itself, instead of a second one for the dev-bridge's own port.
      // Laptop-side Pose Viewer tabs keep connecting directly to
      // ws://localhost:8787, unchanged -- this proxy is only needed for the
      // non-localhost (phone) case.
      '/dev-bridge': {
        target: 'ws://localhost:8787',
        ws: true,
      },
    },
  },
});
