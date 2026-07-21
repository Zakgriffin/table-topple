// Phone-side capture page: live camera viewfinder + hardware zoom (ported
// from src/main.ts's tracker page, same architecture) with a shutter button
// that sends the current frame to any open Sphere Lab tab over the dev
// bridge relay -- see scripts/dev-bridge/server.js's 'realCapture' handling
// and sphereLab.ts's ingestRealCapture. Doesn't run any of the actual
// analysis pipeline itself; this page's only job is getting a real photo
// off the phone and onto the laptop.
//
// Two capture modes: single (tap the shutter each time, as before) and
// video (streams frames automatically). Either way, Sphere Lab's own
// reconstruction pass is slow enough that it needs to gate how fast frames
// arrive -- see the "Capture mode + readiness" section below for the
// signaling that makes that safe instead of flooding the relay with frames
// Sphere Lab hasn't finished the last one yet.

const video = document.getElementById('v') as HTMLVideoElement;
const captureCanvas = document.getElementById('captureCanvas') as HTMLCanvasElement;
const captureCtx = captureCanvas.getContext('2d')!;
const camStatus = document.getElementById('camStatus')!;
const relayStatus = document.getElementById('relayStatus')!;
const zoomSlider = document.getElementById('zoom') as HTMLInputElement;
const switchCamBtn = document.getElementById('switchCam') as HTMLButtonElement;
const shutterBtn = document.getElementById('shutter') as HTMLButtonElement;
const modeSingleBtn = document.getElementById('modeSingleBtn') as HTMLButtonElement;
const modeVideoBtn = document.getElementById('modeVideoBtn') as HTMLButtonElement;

// ── Camera + zoom (ported from src/main.ts:33-99, same reasoning) ─────────

let currentStream: MediaStream | null = null;
let currentFacing = 'environment';
let zoomMin = 1, zoomMax = 1;

function sliderToZoom(t: number): number {
  return zoomMin * Math.pow(zoomMax / zoomMin, t);
}

function setupZoomControl() {
  const track = currentStream?.getVideoTracks()[0];
  let caps: any = null;
  try { caps = track && 'getCapabilities' in track ? (track as any).getCapabilities() : null; }
  catch { caps = null; }

  if (caps && caps.zoom && caps.zoom.min > 0 && caps.zoom.max > caps.zoom.min) {
    zoomSlider.disabled = false;
    zoomMin = caps.zoom.min;
    zoomMax = caps.zoom.max;
    zoomSlider.min = '0';
    zoomSlider.max = '1';
    zoomSlider.step = '0.001';
    zoomSlider.value = '0';
    track!.applyConstraints({ advanced: [{ zoom: zoomMin } as any] }).catch(() => {});
  } else {
    zoomSlider.disabled = true;
  }
}

zoomSlider.addEventListener('input', () => {
  const track = currentStream?.getVideoTracks()[0];
  const zoom = sliderToZoom(parseFloat(zoomSlider.value));
  track?.applyConstraints({ advanced: [{ zoom } as any] }).catch(() => {});
});

async function startCamera(desiredFacing: string) {
  const newStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: desiredFacing } },
  });
  if (currentStream) currentStream.getTracks().forEach((t) => t.stop());
  currentStream = newStream;
  video.srcObject = currentStream;
  await video.play();
  const settings = currentStream.getVideoTracks()[0].getSettings();
  currentFacing = settings.facingMode || 'environment';
  video.classList.toggle('mirror', currentFacing === 'user');
  setupZoomControl();
  camStatus.textContent = `${currentFacing} camera, ${settings.width}x${settings.height}`;
}

switchCamBtn.addEventListener('click', async () => {
  const next = currentFacing === 'user' ? 'environment' : 'user';
  switchCamBtn.disabled = true;
  try { await startCamera(next); }
  catch (e: any) { camStatus.textContent = 'camera error: ' + e.message; }
  finally { switchCamBtn.disabled = false; }
});

startCamera('environment').catch((e: any) => {
  camStatus.textContent = 'camera error: ' + e.message;
});

// ── Relay connection ────────────────────────────────────────────────────
//
// Rides the SAME https origin the page itself was loaded from, via vite's
// /dev-bridge websocket proxy (see vite.config.ts) -- a page loaded over
// https (required here for getUserMedia) can't open a plain insecure ws://
// connection to a non-localhost host, so this can't just point at the
// dev-bridge's own ws://<lan-ip>:8787 directly the way a laptop-local tab
// can. Works identically whether this page happens to be opened via the LAN
// IP or localhost.
let ws: WebSocket | null = null;
let reconnectTimer: number | undefined;

function setRelayStatus(text: string, down: boolean) {
  relayStatus.textContent = `relay: ${text}`;
  relayStatus.classList.toggle('down', down);
}

function scheduleReconnect() {
  ws = null;
  setRelayStatus('reconnecting…', true);
  clearTimeout(reconnectTimer);
  reconnectTimer = window.setTimeout(connectRelay, 2000);
}

// ── Capture mode + readiness ────────────────────────────────────────────
//
// Sphere Lab's reconstruction pass is slow enough that it needs to tell us
// when it's actually done with the last frame (see main.ts's animate loop,
// which watches axesCapturing and pushes captureReady over this same
// socket) -- both photo and video mode respect it, not just video, per an
// explicit ask: the shutter turns yellow and single-mode taps become a
// no-op whenever Sphere Lab isn't ready, exactly like video mode already
// has to gate its automatic sends.
let captureMode: 'single' | 'video' = 'single';
let sphereLabReady = true;
let readyTimeoutTimer: number | undefined;

function setCaptureMode(mode: 'single' | 'video') {
  captureMode = mode;
  modeSingleBtn.classList.toggle('active', mode === 'single');
  modeVideoBtn.classList.toggle('active', mode === 'video');
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'captureMode', mode }));
}
modeSingleBtn.addEventListener('click', () => setCaptureMode('single'));
modeVideoBtn.addEventListener('click', () => setCaptureMode('video'));

// If nothing ever answers (no Sphere Lab tab open, or one that closed mid-
// crunch) don't stay stuck yellow/stalled forever -- fall back to assuming
// ready after a while. A real captureReady message always overrides this.
const READY_TIMEOUT_MS = 8000;
function setReady(ready: boolean) {
  sphereLabReady = ready;
  shutterBtn.classList.toggle('notReady', !ready);
  clearTimeout(readyTimeoutTimer);
  if (!ready) readyTimeoutTimer = window.setTimeout(() => setReady(true), READY_TIMEOUT_MS);
}

function connectRelay() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  try { ws = new WebSocket(`${proto}//${location.host}/dev-bridge`); }
  catch { scheduleReconnect(); return; }

  ws.addEventListener('open', () => {
    ws!.send(JSON.stringify({ role: 'capture' }));
    // A reconnect gets a brand-new captureId server-side (see server.js),
    // which Sphere Lab will treat as a fresh phone -- re-announce whatever
    // mode was already selected, and drop any stale not-ready state from
    // before the drop, since it belonged to the OLD captureId.
    ws!.send(JSON.stringify({ type: 'captureMode', mode: captureMode }));
    setReady(true);
    setRelayStatus('connected', false);
  });
  ws.addEventListener('close', scheduleReconnect);
  ws.addEventListener('error', () => {});
  ws.addEventListener('message', (ev) => {
    let msg: any;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'captureReady') setReady(!!msg.ready);
  });
}
connectRelay();

// ── Shutter / video streaming ────────────────────────────────────────────
//
// Capped to MAX_DIM on the long edge before encoding -- Sphere Lab
// resamples whatever arrives down to its own analysis resolution anyway
// (see ingestRealCapture), so sending a phone's full native resolution
// (often 3000-4000px) would just be a slower base64 transfer for no benefit.
const MAX_DIM = 1600;

// Grabs the current video frame and sends it, if there's anywhere to send
// it to. Shared by the single-tap shutter and the video-mode loop below --
// callers are responsible for checking sphereLabReady first (video mode
// checks every tick; the shutter click handler checks once per tap).
function captureAndSendFrame() {
  if (!currentStream || video.videoWidth === 0) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(1, MAX_DIM / Math.max(vw, vh));
  const cw = Math.round(vw * scale), ch = Math.round(vh * scale);
  captureCanvas.width = cw; captureCanvas.height = ch;
  // Mirror the draw too if the front camera's own preview is mirrored, so
  // the SENT image matches what was actually framed on screen, not a
  // left-right-flipped version of it.
  if (currentFacing === 'user') {
    captureCtx.save();
    captureCtx.translate(cw, 0);
    captureCtx.scale(-1, 1);
    captureCtx.drawImage(video, 0, 0, cw, ch);
    captureCtx.restore();
  } else {
    captureCtx.drawImage(video, 0, 0, cw, ch);
  }
  const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.85);

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'realCapture', dataUrl }));
    shutterBtn.classList.add('sent');
    setTimeout(() => shutterBtn.classList.remove('sent'), 300);
    // Optimistic -- Sphere Lab will confirm the real state via captureReady
    // once it's actually looked at this frame; this just stops us (or the
    // video loop) from firing off a second one in the meantime.
    setReady(false);
  } else {
    setRelayStatus('not connected -- capture NOT sent', true);
  }
}

shutterBtn.addEventListener('click', () => {
  // In video mode the button is a status indicator, not a trigger -- frames
  // already send themselves via the loop below.
  if (captureMode !== 'single' || !sphereLabReady) return;
  captureAndSendFrame();
});

// Ticks every frame; only actually sends in video mode, and only once
// Sphere Lab has said it's ready for another one -- that's what turns a
// slow reconstruction pass into a natural frame-rate cap instead of
// flooding the relay with frames nothing's looked at yet.
function videoLoop() {
  requestAnimationFrame(videoLoop);
  if (captureMode === 'video' && sphereLabReady) captureAndSendFrame();
}
videoLoop();
