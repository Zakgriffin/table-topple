// ── The camera: the stream, the controls that shape it, and the frames it
// delivers ───────────────────────────────────────────────────────────────
//
// Split out of the page's one file. Everything hard-won about NEGOTIATING a
// stream is src/capture/stream.ts -- that is platform knowledge and both
// clients need it. What is here is the PAGE's half: which control reflects
// what, what the status line says, how a resolution sweep is sequenced against
// the dropdown and the probe log, and the two canvases every capture path
// draws through.
//
// It imports only dom.ts and the neutral platform layers, so like dom.ts it
// can never be part of a cycle -- see main.ts's header on the temporal-dead-
// zone hazard this page already hit once as a single file.
//
// The one thing it does NOT own is the initial startCamera() call: that is the
// page's boot sequence and stays in main.ts, in the order it has always run.
// The CONTROLS register themselves here, the way imu.ts's checkboxes do.

import {
  type ProbeResult, type ResCandidate,
  RES_PROBE_DELAY_MS, applyFrameRate, applyZoom, buildResCandidateList, detectAxesSwapped,
  probeResolution, readFrameRate, readZoomRange, requestPlainStream, requestStreamAt, sleep, zoomAt,
} from '../../capture/stream.ts';
import { nowMs } from '../../clock.ts';
import type { FrameMeta } from '../shared/camera/model.ts';
import {
  camStatus, captureCanvas, captureCtx, detectResBtn, fpsSlider, fpsValue,
  probeLogCloseBtn, probeLogMatchHeader, probeLogPanel, probeLogTbody, probeLogToggleBtn,
  resSelect, switchCamBtn, video, zoomSlider,
} from './dom.ts';

// ── Camera + zoom ────────────────────────────────────────────────────────

export let currentStream: MediaStream | null = null;
export let currentFacing = 'environment';
let zoomRange: { min: number; max: number } | null = null;

function setupZoomControl() {
  zoomRange = readZoomRange(currentStream);
  if (!zoomRange) { zoomSlider.disabled = true; return; }
  zoomSlider.disabled = false;
  zoomSlider.min = '0';
  zoomSlider.max = '1';
  zoomSlider.step = '0.001';
  zoomSlider.value = '0';
  applyZoom(currentStream, zoomRange.min);
}

zoomSlider.addEventListener('input', () => {
  if (zoomRange) applyZoom(currentStream, zoomAt(zoomRange, parseFloat(zoomSlider.value)));
});

// ── Sent-image resolution, the page's half ──────────────────────────────
//
// WHY there is a sweep at all, WHY the constraints are shaped the way they are,
// and WHAT the axis swap is, all live in src/capture/stream.ts -- that is the
// platform knowledge, and both clients need it. What is left here is the
// dropdown, the probe-log table, and the sequencing of a sweep against them.
let resCandidates: ResCandidate[] = [];

// Rebuilds the <option> list from the current resCandidates -- called
// after every probe (not just once at the end) so the dropdown fills in
// live as the sweep progresses, rather than sitting inert/unexplained for
// however long the whole sweep takes. 'rejected' candidates are left OUT
// entirely now, not just disabled/grayed -- with everything shown genuinely
// confirmed-good, there's no reason to also show a pile of known-bad
// options (the full detail of WHY each one failed lives in the probe log
// instead, see probeResolution/renderProbeLog).
function renderResOptions() {
  const currentValue = resSelect.value;
  resSelect.innerHTML = '';
  const shown = resCandidates.filter((c) => c.status !== 'rejected');
  // Only reachable if setupDefaultResOption itself couldn't populate
  // anything (video dimensions not ready yet), or every candidate probed
  // so far has failed -- an empty <select> would just look broken rather
  // than explaining what to do.
  if (shown.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'tap "detect" →';
    resSelect.appendChild(opt);
    return;
  }
  for (const c of shown) {
    const opt = document.createElement('option');
    opt.value = `${c.w}x${c.h}`;
    opt.textContent = c.status === 'untested' ? `${c.w} × ${c.h} (untested)` : `${c.w} × ${c.h}`;
    resSelect.appendChild(opt);
  }
  if (Array.from(resSelect.options).some((o) => o.value === currentValue)) resSelect.value = currentValue;
}

// Reconnects to a plain, unconstrained stream on the SAME physical camera
// (deviceId if known, else falls back to facingMode) -- used to bring the
// viewfinder back after a failed/rejected resolution request or after the
// sweep finishes, WITHOUT going through startCamera itself: that also
// calls setupDefaultResOption, which would blow away whatever candidates
// list is currently populated (the very thing the caller usually still
// wants showing in the dropdown).
async function reconnectPlainStream(deviceId: string | undefined) {
  try {
    await adoptStream(await requestPlainStream(deviceId, currentFacing));
    refreshCamStatusResolution();
  } catch (e: any) {
    camStatus.textContent = 'camera error: ' + e.message;
  }
}

// Applies a candidate already CONFIRMED supported (either by the sweep, or
// just whatever the camera is already streaming at by default) by
// swapping the live feed to a fresh stream at exactly (w,h) -- see
// requestStreamAt's own comment on why a fresh stream, not
// applyConstraints(). Returns whether the camera actually ended up at
// exactly (w,h), not just whether the request succeeded at all: a real
// device can resolve successfully while still landing on a DIFFERENT
// resolution than requested. Always refreshes camStatus with whatever the
// camera ACTUALLY ends up at either way (see refreshCamStatusResolution),
// so a mismatch or an outright rejection is visibly reported instead of
// silently assumed to have worked. On rejection, reconnects to SOME
// working stream rather than leaving the camera dead, since the old stream
// already had to be stopped before requesting the new one (real camera
// hardware generally can't have two opens on the same device at once).
//
// If resolutionAxesSwapped was detected by a prior sweep (see its own
// comment), the REQUEST sent here is swapped to compensate -- w,h below
// always mean the TRUE target (what video.videoWidth/videoHeight should
// end up as), never the raw request.
async function applyExactResolution(w: number, h: number): Promise<boolean> {
  if (video.videoWidth === w && video.videoHeight === h) { refreshCamStatusResolution(); return true; } // already exactly there
  const deviceId = currentStream?.getVideoTracks()[0]?.getSettings().deviceId;
  if (currentStream) currentStream.getTracks().forEach((t) => t.stop());
  const stream = resolutionAxesSwapped ? await requestStreamAt(deviceId, h, w) : await requestStreamAt(deviceId, w, h);
  if (!stream) {
    camStatus.textContent = `resolution ${w}x${h} REJECTED by camera`;
    await reconnectPlainStream(deviceId);
    return false;
  }
  await adoptStream(stream);
  const ok = video.videoWidth === w && video.videoHeight === h;
  if (!ok) camStatus.textContent = `resolution ${w}x${h} requested but camera gave ${video.videoWidth}x${video.videoHeight} instead`;
  refreshCamStatusResolution();
  return ok;
}

// Single source of truth for "what resolution is the camera actually at
// right now" -- reads video.videoWidth/videoHeight (what real captures
// actually use, see drawFullResFrameToSendCanvas) so camStatus and the
// resolution dropdown can never drift from reality the way camStatus used
// to (it was previously set ONCE in startCamera and never touched again,
// so it kept showing the pre-selection resolution forever after). Also
// re-syncs resSelect's displayed value, since a rejected/overridden
// request means the dropdown's own selection can no longer be trusted to
// reflect what's live either.
function refreshCamStatusResolution() {
  const w = video.videoWidth, h = video.videoHeight;
  if (!w || !h) return;
  camStatus.textContent = `${currentFacing} camera, ${w}x${h}`;
  const value = `${w}x${h}`;
  if (Array.from(resSelect.options).some((o) => o.value === value)) resSelect.value = value;
}

resSelect.addEventListener('change', () => {
  const [w, h] = resSelect.value.split('x').map(Number);
  applyExactResolution(w, h);
});

// Populates the dropdown with just whatever the camera is ALREADY
// streaming at by default (from startCamera's own unconstrained
// negotiation) -- a single, always-safe, zero-extra-requests option, so
// the viewfinder/shutter are usable immediately without ever running the
// (much more invasive) sweep below. Called from startCamera, same as
// setupZoomControl/setupFramerateControl.
function setupDefaultResOption() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) { resSelect.disabled = true; return; }
  resCandidates = [{ w: vw, h: vh, status: 'supported' }];
  renderResOptions();
  resSelect.disabled = false;
}

// The probe-log table's backing rows. `ProbeResult` is capture/stream.ts's --
// what was asked for, whether getUserMedia resolved, what the camera settled
// on. Kept for the whole session (not just the latest sweep) so runs can be
// compared; there is no clear action yet, so in practice it grows across
// sweeps, which is fine for a debug tool nobody leaves running.
export let probeLog: ProbeResult[] = [];

// e.matched (on the entry itself) is always the RAW, direct requested-vs-
// actual comparison -- never mutated once a probe finishes, so it stays a
// simple fact. Once resolutionAxesSwapped is known (see its own comment),
// the MEANINGFUL match check is the flipped one instead, so this recomputes
// it fresh at render time (never overwriting entry.matched) and relabels
// the column header to say so -- otherwise the table would keep showing
// "no" for rows the sweep has already reclassified as supported, which
// would look like a contradiction rather than the correct, reinterpreted
// answer.
function renderProbeLog() {
  probeLogMatchHeader.textContent = resolutionAxesSwapped ? 'match (flipped)?' : 'match?';
  probeLogTbody.innerHTML = '';
  for (const e of probeLog) {
    const tr = document.createElement('tr');
    const resolvedText = e.resolved ? 'yes' : `no${e.error ? ` — ${e.error}` : ''}`;
    const actualText = e.actualW != null ? `${e.actualW}×${e.actualH}` : '—';
    const effectiveMatch = resolutionAxesSwapped
      ? (e.actualW === e.requestedH && e.actualH === e.requestedW)
      : e.matched;
    tr.innerHTML = `<td>${e.requestedW}×${e.requestedH}</td><td>${resolvedText}</td><td>${actualText}</td><td>${effectiveMatch ? 'yes' : 'no'}</td>`;
    tr.className = effectiveMatch ? 'pass' : 'fail';
    probeLogTbody.appendChild(tr);
  }
}
probeLogToggleBtn.addEventListener('click', () => probeLogPanel.classList.toggle('visible'));
probeLogCloseBtn.addEventListener('click', () => probeLogPanel.classList.remove('visible'));

// Some platforms consistently report video.videoWidth/videoHeight SWAPPED
// relative to whatever width/height was actually requested -- confirmed
// live this session on Chrome for iOS, which (like every browser on iOS,
// by Apple's own App Store rules) is required to use WebKit as its actual
// engine regardless of branding. WebKit negotiates the camera in its own
// fixed sensor-native space, then rotates the delivered frame to match the
// phone's physical orientation WITHOUT also flipping which axis the
// original request numbers meant -- so on a phone held in portrait, asking
// for landscape dimensions comes back reporting portrait dimensions, and
// vice versa, every single time. Detected empirically per sweep (see
// sweepResolutionCandidates' own post-hoc majority-vote pass below), not
// via user-agent sniffing -- same "verify, don't assume" principle as
// everything else in this section. Once known, requestStreamAt/
// applyExactResolution swap the REQUEST to compensate, while still
// verifying against the TRUE target in video.videoWidth/videoHeight.
export let resolutionAxesSwapped = false;

// Manually triggered ONLY (see detectResBtn's click handler) -- each probe
// is a REAL, isolated stream open+close (see probeResolution), paced by
// RES_PROBE_DELAY_MS between every attempt. The main viewfinder goes dark
// for the sweep's duration (the live stream is stopped up front, below --
// most camera hardware can't have two opens on one device at once) and
// comes back once it's done; that's an accepted cost of getting a genuine
// empirical answer instead of a guess.
//
// Applies nothing on its own once done, INCLUDING not restoring whatever
// resolution was active before the sweep started -- the user picks what
// they actually want from the now-populated dropdown afterward, same as
// every other control in this file, rather than the sweep silently
// deciding for them. Just reconnects to a plain unconstrained stream
// (reconnectPlainStream, NOT startCamera -- that also calls
// setupDefaultResOption, which would wipe out the candidate list this
// sweep just spent time building) so the viewfinder isn't left dead.
async function sweepResolutionCandidates() {
  if (!currentStream) return;
  resSelect.disabled = true;
  detectResBtn.disabled = true;
  probeLog = [];
  renderProbeLog();
  const deviceId = currentStream.getVideoTracks()[0]?.getSettings().deviceId;
  const nativeW = video.videoWidth, nativeH = video.videoHeight;
  resCandidates = buildResCandidateList();
  if (nativeW && nativeH && !resCandidates.some((c) => c.w === nativeW && c.h === nativeH)) {
    resCandidates.push({ w: nativeW, h: nativeH, status: 'supported' }); // this IS the video element's current live size, no probe needed
    resCandidates.sort((a, b) => a.w * a.h - b.w * b.h);
  }
  renderResOptions();
  currentStream.getTracks().forEach((t) => t.stop());

  let i = 0;
  const attempts: { cand: ResCandidate; entry: ProbeResult }[] = [];
  for (const cand of resCandidates) {
    if (cand.status !== 'untested') continue; // the injected native pair above is already known-good
    i++;
    detectResBtn.textContent = `detecting… ${i}/${resCandidates.length}`;
    const entry = await probeResolution(deviceId, cand.w, cand.h);
    cand.status = entry.matched ? 'supported' : 'rejected';
    attempts.push({ cand, entry });
    renderResOptions();
    await sleep(RES_PROBE_DELAY_MS);
  }

  // Post-hoc majority vote, reusing the raw data every probe above already
  // collected -- NOT a second sweep under the opposite assumption (see
  // resolutionAxesSwapped's own comment on why that's unnecessary). If most
  // resolved probes came back with actual dimensions swapped relative to
  // what was requested, this platform swaps axes -- so every candidate
  // that LOOKED rejected only because of that swap gets reclassified as
  // supported, using data already in hand.
  resolutionAxesSwapped = detectAxesSwapped(attempts.map((a) => a.entry));
  // Refreshes the table's header/rows for the (possibly new) verdict even
  // when it comes back false -- a PRIOR sweep could have left the header
  // reading "match (flipped)?" and this run needs to be able to correct
  // that back, not just flip it on.
  renderProbeLog();
  if (resolutionAxesSwapped) {
    for (const { cand, entry } of attempts) {
      if (entry.actualW === cand.h && entry.actualH === cand.w) cand.status = 'supported';
    }
    renderResOptions();
  }

  detectResBtn.textContent = 'detect';
  detectResBtn.disabled = false;
  resSelect.disabled = false;
  await reconnectPlainStream(deviceId);
}
detectResBtn.addEventListener('click', () => sweepResolutionCandidates());

// ── Frame rate, the page's half ─────────────────────────────────────────
//
// Why a frame-rate control exists at all -- it is a lever against MOTION BLUR,
// not smoothness -- is in capture/stream.ts's readFrameRate. Bounds are 1fps (a
// full second between frames, the slowest anyone would want) up to whatever the
// camera admits to.
const FPS_MIN = 1;
let fpsMax = FPS_MIN;

function setupFramerateControl() {
  const { max, negotiated } = readFrameRate(currentStream);
  const current = negotiated ?? undefined;
  fpsMax = Math.max(FPS_MIN, max);

  if (fpsMax <= FPS_MIN) {
    fpsSlider.disabled = true;
    return;
  }
  fpsSlider.disabled = false;
  fpsSlider.min = String(FPS_MIN);
  fpsSlider.max = String(fpsMax);
  // No longer auto-applies fpsMax on load -- confirmed live this session
  // (see the resolution dropdown's own comment on sweepResolutionCandidates)
  // that an automatic, unprompted hardware reconfiguration on camera start
  // can crash a real phone; frameRate is exactly that kind of call
  // (applyConstraints -> real stream reconfiguration), so this now only
  // REFLECTS whatever the browser already negotiated on its own, matching
  // zoom/res's own "don't touch it automatically" posture -- the slider
  // still visually defaults toward the fast end, but nothing actually
  // fires until the user drags it themselves.
  const initial = Math.min(fpsMax, Math.max(FPS_MIN, Math.round(current ?? fpsMax)));
  fpsSlider.value = String(initial);
  fpsValue.textContent = `${initial}`;
}

fpsSlider.addEventListener('input', () => {
  const fps = parseInt(fpsSlider.value, 10);
  fpsValue.textContent = `${fps}`;
  applyFrameRate(currentStream, fps);
});

// ── Producer frame-rate diagnostics ─────────────────────────────────────
//
// Answers "is the camera hardware itself the bottleneck" independently of
// anything the capture/relay round trip does. requestVideoFrameCallback
// fires once per actually-decoded video frame (unlike requestAnimationFrame,
// which just ticks at the display's own rate regardless of whether a new
// camera frame has actually arrived), and its metadata.mediaTime is the
// frame's own timestamp on the media timeline -- so diffing consecutive
// mediaTimes measures real delivered-frame spacing, not callback jitter.
//
// These are the CAMERA's half of the page's frame diagnostics. The other half
// -- loopTicks and the blocked-send counters -- describes the VIDEO LOOP and
// stays with it in main.ts; main.ts is what joins the two into one frameStats
// message.
export let nominalFrameRate: number | null = null;
let lastMediaTime: number | null = null;
let frameIntervalsMs: number[] = [];

// The most recent decoded video frame's own timing, republished by the rVFC
// callback below and attached to whatever realCapture goes out next.
//
// WHY THIS EXISTS: captureAndSendFrame stamps `sentAt` at DRAW
// time, which is not when the photons landed -- between them sit the sensor's
// exposure, the camera stack's processing, and however long the frame sat in
// the video element before the send gate let a capture through. Feeding a
// filter a frame timestamped at draw time gives a constant unmodelled lag,
// and a constant lag against a moving camera is a pose error PROPORTIONAL TO
// VELOCITY -- i.e. exactly the symptom the IMU work is meant to remove, which
// makes it the one error that can masquerade as success. See the motion-blur/
// IMU plan's phase B.
//
// `captureTime` is the field actually wanted and it is NOT universally
// implemented for getUserMedia sources (it is specified as optional, and is
// most reliably present on WebRTC-sourced video). `presentationTime` is
// always there. Both are recorded rather than one being picked here, because
// which of them is trustworthy is a per-device question that phase B's
// cross-correlation is meant to ANSWER, not one to guess at now.
export let latestFrameMeta: FrameMeta | null = null;

function onVideoFrame(_now: number, metadata: any) {
  if (lastMediaTime !== null) {
    const dt = (metadata.mediaTime - lastMediaTime) * 1000;
    if (dt > 0) frameIntervalsMs.push(dt);
  }
  lastMediaTime = metadata.mediaTime;
  latestFrameMeta = {
    mediaTime: metadata.mediaTime,
    presentationTime: metadata.presentationTime,
    expectedDisplayTime: metadata.expectedDisplayTime,
    captureTime: metadata.captureTime ?? null,
    processingDuration: metadata.processingDuration ?? null,
    presentedFrames: metadata.presentedFrames ?? null,
    observedAt: nowMs(),
  };
  rvfc?.(onVideoFrame);
}
// Cast: requestVideoFrameCallback isn't in every TS DOM lib version yet;
// feature-detected at runtime regardless, so a missing type just means no
// diagnostics on a browser too old to have shipped it.
const rvfc = (video as any).requestVideoFrameCallback?.bind(video) as
  ((cb: (now: number, metadata: any) => void) => void) | undefined;
rvfc?.(onVideoFrame);

// Hands the accumulated inter-frame spacings to whoever reports them and
// clears the accumulator, so the window is "since the last report" rather
// than "since the page loaded". Returns null when no frame pair landed in
// the window at all -- distinct from a window whose frames were all evenly
// spaced, and the caller omits the fields entirely in that case rather than
// reporting a zero it did not measure.
export function takeFrameIntervalStats(): { avgIntervalMs: number; maxIntervalMs: number; sampleCount: number } | null {
  if (frameIntervalsMs.length === 0) return null;
  const stats = {
    avgIntervalMs: frameIntervalsMs.reduce((a, b) => a + b, 0) / frameIntervalsMs.length,
    maxIntervalMs: Math.max(...frameIntervalsMs),
    sampleCount: frameIntervalsMs.length,
  };
  frameIntervalsMs = [];
  return stats;
}

// ── Adopting a stream, and starting one ──────────────────────────────────

// Makes `stream` the live camera feed -- swaps currentStream/video.srcObject
// to it and re-binds everything tied to the SPECIFIC MediaStreamTrack
// object (zoom, framerate), since a fresh getUserMedia() stream is a
// genuinely NEW track even when it's the same physical camera as before;
// controls left bound to the OLD (now-stopped) track would silently stop
// working otherwise. Shared by startCamera and applyExactResolution/
// sweepResolutionCandidates above, all of which now request a genuinely
// fresh stream per resolution rather than reconfiguring one track in place
// -- see this session's chat on why (a real-world reference tool aimed at
// the exact same problem, addpipe/webcam-resolution-tester, does the same
// -- MediaStreamTrack.applyConstraints() has uneven real-world support for
// actually renegotiating a running camera's resolution, especially across
// orientations, on mobile).
async function adoptStream(stream: MediaStream) {
  if (currentStream && currentStream !== stream) currentStream.getTracks().forEach((t) => t.stop());
  currentStream = stream;
  video.srcObject = stream;
  await video.play();
  const settings = stream.getVideoTracks()[0].getSettings();
  currentFacing = settings.facingMode || currentFacing;
  // No CSS mirror class anymore -- video itself is hidden (captureCanvas is
  // now the visible viewfinder, see this session's chat), and
  // drawCurrentFrameToCanvas already bakes the front-camera flip directly
  // into captureCanvas's own pixels; a CSS transform on top of that would
  // double-flip it.
  setupZoomControl();
  setupFramerateControl();
  // The camera hardware's OWN nominal capture rate -- distinct from how
  // often WE encode/send (see main.ts's frameStats reporting), and a real
  // alternative explanation raised for video mode's idle gap: auto-exposure
  // in low light can drop a phone's actually-delivered frame rate well
  // below its nominal one, independent of anything the round-trip does.
  nominalFrameRate = settings.frameRate ?? null;
}

export async function startCamera(desiredFacing: string) {
  // No width/height constraint at all here, ideal or exact -- whatever
  // resolution the browser's own default negotiation lands on is exactly
  // as arbitrary as any other guess, so there's no reason to bias it
  // toward "as high as possible" (the old `ideal: 7680x4320` trick) before
  // the user's actually picked anything. setupDefaultResOption just
  // reflects whatever this happens to be; sweepResolutionCandidates/the
  // dropdown are the real way to choose a resolution deliberately.
  const newStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: desiredFacing } },
  });
  await adoptStream(newStream);
  setupDefaultResOption(); // sweepResolutionCandidates is manual-only -- see its own comment on why
  refreshCamStatusResolution();
}

switchCamBtn.addEventListener('click', async () => {
  const next = currentFacing === 'user' ? 'environment' : 'user';
  switchCamBtn.disabled = true;
  try { await startCamera(next); }
  catch (e: any) { camStatus.textContent = 'camera error: ' + e.message; }
  finally { switchCamBtn.disabled = false; }
});

// ── The two canvases ─────────────────────────────────────────────────────

// Draws the current video frame onto the VISIBLE captureCanvas -- but
// downscaled to fit the screen's own physical pixel dimensions (aspect
// preserved), not the camera's full negotiated resolution. This runs
// unconditionally every rAF tick (see main.ts's videoLoop), and drawing/
// compositing a full-native-resolution (potentially 12+MP) on-screen canvas at
// 60fps is what was crashing Chrome on a real phone -- CSS max-width/max-height
// (pose-viewer-client.html) only scales the DISPLAYED box, it doesn't shrink
// the actual pixel buffer drawImage has to fill and the compositor has to
// upload, so a cap has to happen here instead. previewLongEdgeCap() is the
// screen's own long edge in physical pixels -- rendering any larger than
// that buys zero visible detail. The actual full-resolution frame sent to
// the server is drawn separately, on demand, by drawFullResFrameToSendCanvas
// below -- this function only ever feeds the live preview.
function previewLongEdgeCap(): number {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  return Math.max(window.innerWidth, window.innerHeight) * dpr;
}

export function drawCurrentFrameToCanvas(): { cw: number; ch: number } {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) { captureCanvas.width = 0; captureCanvas.height = 0; return { cw: 0, ch: 0 }; }
  const scale = Math.min(1, previewLongEdgeCap() / Math.max(vw, vh));
  const cw = Math.max(1, Math.round(vw * scale)), ch = Math.max(1, Math.round(vh * scale));
  // Guards the resize -- reassigning canvas.width/height clears the
  // backing bitmap even when set to the same value, so skipping it when
  // unchanged avoids a pointless reallocation on every one of these ticks.
  if (captureCanvas.width !== cw || captureCanvas.height !== ch) { captureCanvas.width = cw; captureCanvas.height = ch; }
  if (currentFacing === 'user') {
    captureCtx.save();
    captureCtx.translate(cw, 0);
    captureCtx.scale(-1, 1);
    captureCtx.drawImage(video, 0, 0, cw, ch);
    captureCtx.restore();
  } else {
    captureCtx.drawImage(video, 0, 0, cw, ch);
  }
  return { cw, ch };
}

// Off-DOM canvas dedicated to full-resolution capture -- kept separate from
// the (now downscaled) preview captureCanvas above so the image actually
// sent to the server still carries the full resolution the user picked via
// the resolution dropdown. Never displayed, so no CSS/letterbox concerns;
// only ever drawn into on demand, by main.ts's captureAndSendFrame/
// captureComputeAndSendPose, which already only run when a capture is
// actually happening (shutter tap, a sendGateStatus()-gated video-mode tick,
// or a devicePoseLoop iteration) -- so no extra throttling logic is needed here,
// unlike the continuous preview draw above.
export const sendCanvas = document.createElement('canvas');
export const sendCtx = sendCanvas.getContext('2d')!;

export function drawFullResFrameToSendCanvas(): { cw: number; ch: number } {
  const cw = video.videoWidth, ch = video.videoHeight;
  if (!cw || !ch) return { cw: 0, ch: 0 };
  sendCanvas.width = cw; sendCanvas.height = ch;
  if (currentFacing === 'user') {
    sendCtx.save();
    sendCtx.translate(cw, 0);
    sendCtx.scale(-1, 1);
    sendCtx.drawImage(video, 0, 0, cw, ch);
    sendCtx.restore();
  } else {
    sendCtx.drawImage(video, 0, 0, cw, ch);
  }
  return { cw, ch };
}

// The one question every capture path asks before drawing: is there a live
// stream with real dimensions behind it. Exported as a predicate rather than
// leaving each caller to spell out the same three-way check against
// currentStream and video's dimensions -- they had drifted apart once already
// (captureAndSendFrame checked all three, captureComputeAndSendPose checked
// the same three, and nothing tied them together).
export function hasLiveFrame(): boolean {
  return !!currentStream && video.videoWidth > 0 && video.videoHeight > 0;
}

// What the camera is actually delivering right now, for the recording header.
export function frameDims(): { w: number; h: number } {
  return { w: video.videoWidth, h: video.videoHeight };
}
