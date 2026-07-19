import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// HTTPS with a self-signed cert so getUserMedia() works when testing over
// LAN from a phone (mobile browsers only expose camera APIs in a secure
// context: https, or localhost). You'll need to accept the one-time
// certificate warning in the phone's browser.
export default defineConfig({
  plugins: [basicSsl()],
  server: {
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
