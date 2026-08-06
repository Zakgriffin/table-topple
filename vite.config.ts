import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// HTTPS with a self-signed cert so getUserMedia() works when testing over
// LAN from a phone (mobile browsers only expose camera APIs in a secure
// context: https, or localhost). You'll need to accept the one-time
// certificate warning in the phone's browser.
export default defineConfig({
  plugins: [basicSsl()],
  server: {
    // CROSS-ORIGIN ISOLATION, and it is here for the CLOCK. Chrome coarsens
    // performance.now() to 100us by default and to 5us when the context is
    // cross-origin isolated -- a 20x sharper clock for every number the perf
    // harnesses report. It is not a micro-optimization of the measurement: at
    // 100us, pipelineGPU/device.ts's allocation probe sums 78 sub-resolution
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
      // origin -- mobile-capture.html is loaded over https (required for
      // getUserMedia on a phone), and a page loaded over https can't open a
      // plain insecure ws:// connection to anything except localhost
      // (mixed-content blocking). Riding this proxy means the phone only
      // ever has to trust the ONE cert it already accepted for the page
      // itself, instead of a second one for the dev-bridge's own port.
      // Laptop-side Sphere Lab tabs keep connecting directly to
      // ws://localhost:8787, unchanged -- this proxy is only needed for the
      // non-localhost (phone) case.
      '/dev-bridge': {
        target: 'ws://localhost:8787',
        ws: true,
      },
    },
  },
});
