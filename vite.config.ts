import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// HTTPS with a self-signed cert so getUserMedia() works when testing over
// LAN from a phone (mobile browsers only expose camera APIs in a secure
// context: https, or localhost). You'll need to accept the one-time
// certificate warning in the phone's browser.
export default defineConfig({
  plugins: [basicSsl()],
  server: {
    https: true,
  },
});
