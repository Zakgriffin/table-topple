// Phone-side capture page: live camera viewfinder + hardware zoom (ported
// from src/main.ts's tracker page, same architecture) with a shutter button
// that sends the current frame to any open Pose Viewer tab over the dev
// bridge relay -- see scripts/dev-bridge/server.js's 'realCapture' handling
// and poseViewer.ts's ingestRealCapture. Doesn't run any of the actual
// analysis pipeline itself; this page's only job is getting a real photo
// off the phone and onto the laptop.
//
// Two capture modes: single (tap the shutter each time, as before) and
// video (streams frames automatically). Pose Viewer's own reconstruction
// pass is slow enough that it needs to say something about pacing -- see
// the "Capture mode + readiness" section below. Exactly what that signal
// "Not ready" is purely a status indicator for the shutter button's yellow
// "working" state -- Pose Viewer's capture mailbox can always take a fresher
// frame, busy or not, so sending is never blocked by it.
//
// ── WHAT IS LEFT IN THIS FILE ────────────────────────────────────────────
//
// The page's own subjects, and the boot sequence that starts them. Four
// modules carry the rest, each importable without this one:
//
//   dom.ts          every element, resolved once. the LEAF -- imports nothing
//   camera.ts       the stream, its controls, and the two capture canvases
//   imu.ts          recording, judging and correcting with the IMU
//   relay.ts        one websocket, and everything that gates a send on it
//   poseRecords.ts  the pose ring, and whether a recording is even valid
//   inspect.ts      the declared scope a dev-bridge eval runs in
//
// What stays here is what genuinely belongs to the PAGE: the config mirror,
// the capture/compute mode switches, the two send paths, the video loop and
// its own tick accounting, and the wiring that hands each module the few
// things only this file knows.
//
// ── WHY THE MODULES TAKE HOSTS INSTEAD OF IMPORTING THIS FILE ────────────
//
// This page hit a TEMPORAL DEAD ZONE fault as a single file -- a render loop
// had to be *scheduled* rather than called, because it read bindings declared
// below it. The fix that stuck is structural: nothing in client/ imports
// main.ts, so no cycle can form and no module can be mid-initialization when
// another reads it. Where a module needs something of this page's, this page
// REGISTERS it, at the END of this file, where every binding those closures
// read is already initialized. Keep that direction.

import { toGrayscale } from '../../pose/grayscale.ts';
import { config, fetchConfigFile } from '../shared/config.ts';
import { globalState } from '../shared/state.ts';
// Only the data-side rebuild is needed here: this page draws the viewfinder and
// nothing at board scale. THE AR OVERLAY LEFT -- it is Table Topple's, and it
// now lives in src/tableTopple/client/. See this file's header.
import { rebuildFloorPatternData } from '../shared/floorPattern.ts';
import type { PositionDecodeResult, RecoveredAxes } from '../shared/camera/model.ts';
import type { PipelineSettings } from '../shared/harness/input.ts';

// ── THE ON-DEVICE POSE PATH IS DARK ──────────────────────────────────────
//
// This page ran the WHOLE pipeline locally (device-compute mode): gradient ->
// LSD -> votes -> fit -> period/phase -> decode, via computePoseFromCapture, and
// sent only the recovered pose over the wire. the old pipeline is deleted and this page
// is not yet on src/pose, so every capture now reports a FAILED DECODE.
//
// That is a real behaviour change and it is deliberate: a failed decode is a
// state this page already handles everywhere (`ok: false`, no IMU anchor, the
// overlay hides, intrinsics still publish), so the page runs end to end instead
// of being half-deleted. What it does NOT do is recover a pose.
//
// Wiring pose in here is its own job: the phone needs a WebGPU device and a
// per-resolution PoseContext, which is a different lifecycle from the desktop's.
type LocalPose = { recoveredAxes: RecoveredAxes | null; positionDecode: PositionDecodeResult | null };

import {
  currentFacing, drawCurrentFrameToCanvas, drawFullResFrameToSendCanvas, hasLiveFrame,
  latestFrameMeta, nominalFrameRate, sendCanvas, sendCtx, startCamera, takeFrameIntervalStats,
} from './camera.ts';
import {
  FRAME_PAIR_CAP, WORLD_UP, attachImuHost, framePairs, imuStats, imuTracker, lastAcceptedQuat,
  refreshImuFrameContext, rotationVectorBetween, setUpCamera, useImuCorrection,
} from './imu.ts';
import { evalInScope } from './inspect.ts';
import { noteSettingsSync, recordPose, settingsSyncedAt, syncedBoardSize } from './poseRecords.ts';
import {
  attachRelayHost, connectRelay, isRelayOpen, sendBinary, sendGateStatus, sendJson, setRelayStatus,
} from './relay.ts';
import { packPoseResultWithImage } from '../shared/devBridge/poseResultWire.ts';
import { nowMs } from '../../clock.ts';
import {
  camStatus, chromeToggleBtn, computeOnDeviceCheckbox, imuCheckbox, imuCorrectionCheckbox,
  modeSingleBtn, modeVideoBtn, poseReadoutEl, reloadConfigBtn, reloadConfigStatus,
  sendCapturedImageCheckbox, sendDebugInfoCheckbox, shutterBtn,
} from './dom.ts';

// mobile-capture.html carries no `checked` attributes -- config.phone owns
// these five defaults, the same way config.camera owns every desktop control
// (see poseViewer/config.ts).
//
// Applying is a DISPATCH, not a plain assignment, so each box's own change
// listener runs: setComputeMode tells the desktop, the IMU one asks for the
// motion permission, and so on. Assigning `.checked` alone fires nothing, and
// a box that looks on while nothing is recording is a lie. Boot passes
// `silent` because those listeners are registered further down this file and
// the seeding has to happen before anything reads the values -- boot's own
// wiring covers the same ground.
function applyPhoneConfig(phone: typeof config.phone, silent: boolean): void {
  for (const [box, value] of [
    [computeOnDeviceCheckbox, phone.computeOnDevice],
    [sendDebugInfoCheckbox, phone.sendDebugInfo],
    [sendCapturedImageCheckbox, phone.sendCapturedImage],
    [imuCheckbox, phone.imuEnabled],
    [imuCorrectionCheckbox, phone.imuCorrection],
  ] as [HTMLInputElement, boolean][]) {
    const changed = box.checked !== value;
    box.checked = value;
    if (changed && !silent) box.dispatchEvent(new Event('change'));
  }
}
applyPhoneConfig(config.phone, true);

// Re-reads pose-viewer.config.json and applies config.phone in place. Unlike
// the desktop's load button this does NOT reload: a reload here drops the
// camera stream and the websocket, and the desktop sees the reconnect as an
// entirely new phone (a fresh captureId means a fresh camera tab). The surface
// is five booleans that all already have working change handlers, so applying
// in place is both possible and honest here in a way it is not over there.
//
// Nothing to discard first, either: this page never writes the localStorage
// overlay -- only pose-viewer-server.html does -- so the file is already the only
// thing it was reading.
reloadConfigBtn.addEventListener('click', async () => {
  reloadConfigStatus.textContent = 'loading…';
  try {
    const fresh = await fetchConfigFile();
    applyPhoneConfig(fresh.phone, false);
    config.phone = fresh.phone;
    reloadConfigStatus.textContent = 'config reloaded';
  } catch (err) {
    reloadConfigStatus.textContent = `load failed: ${err instanceof Error ? err.message : String(err)}`;
  }
});

// ── Hiding the page's own UI ─────────────────────────────────────────────
//
// One switch that takes every control off the screen, leaving the viewfinder
// and nothing else -- see mobile-capture.html's
// `body.chromeHidden` rule, which is where the list of what counts as chrome
// lives. Kept there rather than as a set of element handles here so the list
// sits next to the elements it names and cannot drift out of step with them.
//
// Purely presentational: nothing downstream reads this state, so hiding the
// controls does not pause capture, pose recovery, or the relay. Note that it
// takes the SHUTTER with it -- which is the intent (in video mode capture is
// continuous, so a clean view is exactly what you want), but it does mean a
// single-photo capture needs the controls brought back first.
chromeToggleBtn.addEventListener('click', () => {
  const hidden = document.body.classList.toggle('chromeHidden');
  chromeToggleBtn.classList.toggle('active', hidden);
  chromeToggleBtn.setAttribute('aria-pressed', String(hidden));
  // A focused button stays outlined over an otherwise-clean picture, and also
  // treats the next Enter/Space as another click.
  chromeToggleBtn.blur();
});

// ── On-device compute: settings mirror ──────────────────────────────────
//
// globalState here is THIS page's own module instance (mobile-capture.html
// is a separate Vite entry point from pose-viewer-server.html -- a totally separate
// JS realm/module graph, not shared memory), so mutating it locally from a
// settingsSync message is safe -- see this session's on-device-pose-recovery
// plan. Source of truth stays the desktop's own sliders; this just mirrors
// whatever it last pushed. Defaults below match camera/settings.ts's own
// createDefaultCommonSettings so a device-compute cycle run before the
// first settingsSync arrives (e.g. right after toggling the checkbox on
// before the desktop's on-connect push lands) still produces a sane result.
//
// THESE ARE A HAND-MAINTAINED COPY and they have already drifted once (the
// 2026-08-05 default rebaseline moved seven of them and this list was missed
// until the compiler caught an unrelated change here). If a default moves in
// createDefaultCommonSettings, move it here too -- the drift is invisible in
// normal use, because the desktop's on-connect settingsSync overwrites all of
// it within a second of the phone connecting, and only shows up in the narrow
// window before that push lands.
let cameraSettings: PipelineSettings = { ...config.camera.common };

// WHEN a sync landed and WHAT board size it carried are poseRecords.ts's --
// they say whether a recording is valid at all, so they live with the ring
// they qualify. See its un-synced-defaults trap.
function applySettingsSync(msg: any) {
  let boardSize: number | null = null;
  if (msg.globalState) {
    // One flag where there were eight. An older desktop build sending the
    // per-stage set leaves this undefined -> false -> the GPU path, which is
    // what every one of those toggles defaulted to anyway, so a version skew
    // lands on the shipping configuration rather than on all-CPU.
    globalState.forceCPU = !!msg.globalState.forceCPU;
    const incoming = msg.globalState.boardSize;
    if (typeof incoming === 'number') {
      boardSize = incoming;
      // Only rebuild when it genuinely CHANGED: rebuildFloorPatternData
      // rebuilds the whole De Bruijn lookup table, which is not cheap, and a
      // settingsSync carries every slider whether or not any of them moved.
      if (incoming !== syncedBoardSize()) {
        // globalState.boardSize was MISSING here, and the omission is why a
        // settings diff read as a board mismatch: rebuildFloorPatternData
        // updates the decode's actual inputs (R/C/torus/debruijnLookup in
        // floorPattern.ts) but this page's own globalState.boardSize stayed at
        // its hardcoded default forever. Harmless today -- nothing here reads
        // it -- but it makes the phone misreport its own configuration, which
        // is exactly how the far worse bug in poseRecords.ts stayed hidden.
        globalState.boardSize = incoming;
        rebuildFloorPatternData(incoming);
      }
    }
  }
  if (msg.cameraSettings) cameraSettings = { ...cameraSettings, ...msg.cameraSettings };
  noteSettingsSync(boardSize);
}

// ── Loop-tick accounting ─────────────────────────────────────────────────
//
// Answers "is the video loop itself even running as often as expected on this
// phone" independently of network/encode cost entirely. loopTicks counts every
// requestAnimationFrame callback regardless of outcome; if loopTicks over a 2s
// window is well under ~120 (60Hz), requestAnimationFrame itself is being
// starved (backgrounded tab, thermal throttling, main-thread contention) -- a
// phone-side scheduling problem, not network. Of the ticks that DO run,
// backpressureBlockedTicks/readinessBlockedTicks say WHY a send was skipped:
// backpressure means the previous frame is still physically draining over the
// network (see relay.ts's sendGateStatus), readiness means Pose Viewer itself
// said not to send yet (shouldn't happen much when pipelined).
//
// These describe the LOOP, so they live with it here. The camera's own half of
// the diagnostics -- delivered-frame spacing and the hardware's nominal rate --
// is camera.ts's; this is the one place the two are joined into one message.
let loopTicks = 0;
let backpressureBlockedTicks = 0;
let readinessBlockedTicks = 0;
let sendsAttempted = 0;

// Flushes whatever's accumulated every couple seconds -- see server.js's
// frameStats broadcast and devBridge/client.ts's handler for where this
// ends up. Read it on the DESKTOP, whose PhysicalCamera parks it as
// `lastFrameStats` -- not to be confused with this client's own `camera`
// eval namespace, which is client/camera.ts (see client/inspect.ts).
setInterval(() => {
  // Returns BEFORE draining anything: a window that could not be reported is
  // not a window that happened, and clearing here would silently discard it.
  if (!isRelayOpen()) return;
  const stats: any = { type: 'frameStats', nominalFrameRate, loopTicks, backpressureBlockedTicks, readinessBlockedTicks, sendsAttempted };
  const intervals = takeFrameIntervalStats();
  if (intervals) Object.assign(stats, intervals);
  loopTicks = 0; backpressureBlockedTicks = 0; readinessBlockedTicks = 0; sendsAttempted = 0;
  sendJson(stats);
}, 2000);

// ── Capture mode + readiness ────────────────────────────────────────────
//
// Pose Viewer's reconstruction pass is slow enough that it needs to tell us
// when it's actually done with the last frame (see main.ts's animate loop,
// which watches axesCapturing and pushes captureReady over this same
// socket) -- both photo and video mode respect it, not just video, per an
// explicit ask: the shutter turns yellow and single-mode taps become a
// no-op whenever Pose Viewer isn't ready, exactly like video mode already
// has to gate its automatic sends.
let captureMode: 'single' | 'video' = 'single';
let readyTimeoutTimer: number | undefined;
function setCaptureMode(mode: 'single' | 'video') {
  captureMode = mode;
  modeSingleBtn.classList.toggle('active', mode === 'single');
  modeVideoBtn.classList.toggle('active', mode === 'video');
  sendJson({ type: 'captureMode', mode });
  // Entering video mode while device-compute is already on (re)starts the
  // self-paced compute loop -- see devicePoseLoop's own comment.
  if (mode === 'video' && computeOnDevice) devicePoseLoop();
}
modeSingleBtn.addEventListener('click', () => setCaptureMode('single'));
modeVideoBtn.addEventListener('click', () => setCaptureMode('video'));

// ── Compute pose on this device -- orthogonal to the single/video axis
// above (see this session's on-device-pose-recovery plan): when on, the
// phone runs the same pose-recovery math locally (pose/poseCompute.ts's
// computePoseFromCapture) and sends only the recovered pose, no image
// bytes. Still respects captureMode: single taps one compute+send cycle,
// video runs devicePoseLoop continuously.
let computeOnDevice = config.phone.computeOnDevice;
function setComputeMode(onDevice: boolean) {
  computeOnDevice = onDevice;
  poseReadoutEl.classList.toggle('visible', onDevice);
  // Switching modes doesn't clear the currently-shown pose -- it stays on
  // screen until the next update, exactly like switching captureMode doesn't
  // blank the video feed.
  sendJson({ type: 'computeMode', mode: onDevice ? 'device' : 'desktop' });
  if (onDevice && captureMode === 'video') devicePoseLoop();
}
computeOnDeviceCheckbox.addEventListener('change', () => setComputeMode(computeOnDeviceCheckbox.checked));

// Debug-only extras riding on top of a device-compute poseResult message --
// both OFF by default and independently toggleable (see this session's chat:
// explicitly asked for as their OWN pair of switches, not bundled into
// computeOnDevice, specifically so they can stay off to avoid any perf
// impact on the actual compute-speed measurement computeOnDevice exists to
// show). sendCapturedImage pays for a JPEG encode (the same toDataURL cost
// the desktop-compute streaming path always pays); sendDebugInfo pays for
// extra serialization of pipeline intermediates -- neither happens at all
// unless explicitly turned on.
let sendDebugInfo = config.phone.sendDebugInfo;
let sendCapturedImage = config.phone.sendCapturedImage;
sendDebugInfoCheckbox.addEventListener('change', () => { sendDebugInfo = sendDebugInfoCheckbox.checked; });
sendCapturedImageCheckbox.addEventListener('change', () => { sendCapturedImage = sendCapturedImageCheckbox.checked; });

// If nothing ever answers (no Pose Viewer tab open, or one that closed mid-
// crunch) don't stay stuck yellow/stalled forever -- fall back to assuming
// ready after a while. A real captureReady message always overrides this.
const READY_TIMEOUT_MS = 8000;
// Display only. Nothing gates on this any more: Pose Viewer's capture mailbox
// always accepts the freshest frame, so the yellow shutter reports that the
// desktop is busy without ever blocking a send.
function setReady(ready: boolean) {
  shutterBtn.classList.toggle('notReady', !ready);
  clearTimeout(readyTimeoutTimer);
  if (!ready) readyTimeoutTimer = window.setTimeout(() => setReady(true), READY_TIMEOUT_MS);
}

// ── Shutter / video streaming ────────────────────────────────────────────
//
// Sent at the stream's own actual negotiated resolution (see
// camera.ts's drawFullResFrameToSendCanvas) -- NOT just a transfer-speed knob:
// Pose Viewer's ingestRealCapture resizes a physical camera's analysis buffers
// to match whatever resolution actually arrives (unlike a simulated capture,
// there's no further downsample step after this), so this IS the real analysis
// resolution, not merely a cap on top of one.

// Grabs the current video frame and sends it, if there's anywhere to send
// it to. Shared by the single-tap shutter and the video-mode loop below --
// callers are responsible for checking sendGateStatus() first (video mode
// checks every tick; the shutter click handler checks once per tap).
// True from the moment a toBlob encode is kicked off until its callback
// fires -- toDataURL (the old encode call) was synchronous, so two
// overlapping captureAndSendFrame calls were never actually possible;
// toBlob's callback breaks that guarantee (a second video-mode tick could
// start a second encode before the first one's callback runs, and encode
// completion order isn't guaranteed to match start order). Without this
// guard, two out-of-order callbacks could send their JSON metadata
// messages in the OPPOSITE order from their binary frames, and
// devBridge/client.ts's captureId-keyed pairing (not a strict-adjacency
// assumption, see its own comment) would then match a frame's image bytes
// to the WRONG frame's timestamps. Cheap to just serialize instead: skip a
// tick outright if the previous encode hasn't finished yet, exactly the
// spirit of captureIngestBusy's own guard on the desktop decode side.
let sendEncodeInFlight = false;
function captureAndSendFrame() {
  if (!hasLiveFrame() || sendEncodeInFlight) return;
  // Stamped here, before anything below -- the earliest point once the
  // send gate has already let this call through.
  const sentAt = nowMs();
  // Full-resolution draw happens HERE, on demand -- unlike the visible
  // captureCanvas (kept cheap/screen-capped by videoLoop's continuous
  // redraw), sendCanvas only gets drawn into when an actual capture is
  // happening, so paying the full-native-resolution drawImage cost here is
  // fine: it runs at most once per send, not every rAF tick.
  drawFullResFrameToSendCanvas();
  const pulledAt = nowMs();
  // High-resolution monotonic twin of pulledAt (see nowMs) -- pulledAt stays
  // as-is so the existing latency readouts keep the exact number they've
  // always had.
  const drawnAt = nowMs();
  // toBlob, not toDataURL -- gives the JPEG's actual encoded bytes directly
  // (a Blob), sent as a real WebSocket BINARY frame below instead of
  // base64-text-encoded inside a JSON message. Base64 inflates payload size
  // by ~33% for zero benefit here -- see this session's chat -- and
  // WebSocket has supported binary frames natively the whole time.
  // Async (a callback, not a return value), so the metadata JSON message
  // and the binary frame that follows it are both sent from inside this
  // callback, back to back, keeping them adjacent on the wire.
  sendEncodeInFlight = true;
  sendCanvas.toBlob((blob) => {
    sendEncodeInFlight = false;
    if (!blob) return;
    // Stamped once the encode actually finishes, so the desktop can still
    // split "JPEG encode, phone-side CPU" from "actual network transit"
    // instead of lumping both into one number and guessing which one a
    // given slow sample was.
    const encodedAt = nowMs();
    if (!isRelayOpen()) { setRelayStatus('not connected -- capture NOT sent', true); return; }
    // Metadata first, image bytes second -- devBridge/client.ts pairs a
    // realCapture JSON message with whichever binary frame arrives next
    // for that same captureId, so send order here matters.
    // frameMeta rides along beside the three existing sentAt/pulledAt/encodedAt stamps
    // rather than replacing them -- those feed the latency readouts already
    // in place, and the project's own measurement notes are emphatic about
    // not invalidating a working instrument to add a new one. This is the
    // field a filter should timestamp the measurement with; sentAt is draw
    // time. See camera.ts's latestFrameMeta for why the difference matters.
    //
    // `drawnAt` is the honest upper bound on capture time and the fallback
    // when rVFC is unavailable: drawFullResFrameToSendCanvas has already run
    // by here, so the frame in sendCanvas is whatever the video element held
    // at that moment. If latestFrameMeta.observedAt is far behind drawnAt,
    // the drawn frame is older than the last callback saw and the pairing is
    // suspect -- recorded so a replay can detect that rather than assume.
    sendJson({ type: 'realCapture', sentAt, pulledAt, encodedAt, drawnAt, frameMeta: latestFrameMeta });
    sendBinary(blob);
    shutterBtn.classList.add('sent');
    setTimeout(() => shutterBtn.classList.remove('sent'), 300);
    // Deliberately NOT setting poseViewerReady false here. Pose Viewer's mailbox
    // can always take another frame, so forcing the yellow state on would make
    // it lie about whether the desktop is actually still crunching -- and
    // nothing would clear it, since the desktop only reports genuine busy/idle
    // transitions.
  }, 'image/jpeg', 0.85);
}

// buildDebugPayload lived here: it summarized the decode grid, the period/phase
// result, the vote composites and a SECOND explicit LSD pass into the debug blob
// this page sends when `sendDebugInfo` is on. Every one of its inputs came from
// the deleted pipeline, so it went with them, and the debug branch below no
// longer attaches anything.

// ── Device-compute: capture + run the pose pipeline locally + send just
// the result ──────────────────────────────────────────────────────────────
//
// The bufferedAmount backpressure gate is irrelevant here -- a
// serialized pose is a few hundred bytes, never queues -- and there's no
// "Pose Viewer busy" readiness gate either, since the desktop does no work
// at all on this path (see ingestRemotePose, pipeline/capture.ts). The
// natural pacing is "compute (now the actual bottleneck), then send, then
// start the next capture" -- devicePoseLoop below just does that, no
// rAF-tick gating, so the local timing readout shows the phone's true
// compute speed with no artificial pacing.
//
// Draws its own full-resolution frame into sendCanvas (see
// camera.ts's drawFullResFrameToSendCanvas) rather than reading off the
// visible captureCanvas, which is now downscaled to screen resolution for
// the live preview -- pose recovery needs the actual full-resolution
// capture, same as the desktop-compute send path.
let devicePoseComputing = false;
async function captureComputeAndSendPose() {
  if (!hasLiveFrame()) return;
  devicePoseComputing = true;
  try {
    const { cw, ch } = drawFullResFrameToSendCanvas();
    // Snapshotted BEFORE the pipeline runs, not after: latestFrameMeta is
    // republished by the rVFC callback on every decoded frame, so reading it
    // once computePoseFromCapture has finished would attach a LATER frame's
    // timing to this frame's pose. The whole reconstruction is tens of
    // milliseconds, which at 60Hz is several frames of drift.
    const tDrawn = nowMs();
    const frameMetaAtDraw = latestFrameMeta;
    // Re-derived every reconstruction rather than only on the toggle: both
    // inputs (screen orientation, which camera is selected) can change without
    // anything notifying this code, and a stale transform is the silent-wrong-
    // answer failure this whole file is trying to avoid. Two quaternion
    // constructions once per ~136ms is not worth an event listener that could
    // be missed.
    refreshImuFrameContext();
    // Same call ingestRealCapture makes on the desktop side (getImageData ->
    // toGrayscale) -- but NOT the extra flipRowsF64 that used to follow it
    // here. computePoseFromCapture's real, validated expected orientation is
    // top-down: ingestRealCapture stores camera.lastRealCaptureGray bottom-up
    // (one flip, purely for Through-Cam's own raw-preview convention), and
    // runAxesReconstruction then flips it AGAIN before calling
    // computePoseFromCapture -- flipRowsF64 is a pure involution (its own
    // exact inverse), so those two flips cancel out exactly, net top-down.
    // This function used to apply a single flip before calling
    // computePoseFromCapture directly, feeding it bottom-up instead --
    // confirmed live (this session's chat) to be the root cause of an
    // on-device Drow/Dcol axis swap and positionDecode failures relative to
    // the same image replayed through the desktop pipeline.
    const topDown = sendCtx.getImageData(0, 0, cw, ch).data;
    const grayTopDown = toGrayscale(topDown, cw, ch);

    // The twelve-null-field state literal that used to be built here is gone:
    // the pipeline reads exactly these two fields and returns everything else.
    const poseInput = { aspect: cw / ch, settings: cameraSettings };
    // ── TODO(phone-on-pose): THE SPAN GOES HERE ─────────────────────────
    //
    // This is where a profiler span belongs. There is not one yet, and that is
    // deliberate: there is NOTHING HERE TO TIME. The pose
    // computation went with the old pipeline, so what stood between the two
    // `performance.now()` calls that used to be here was `void grayTopDown;`
    // and an object literal -- the stopwatch reported ~0ms every frame, and a
    // span would have reported the same 0ms as a row on a flamechart, which is
    // a more confident way of saying something false.
    //
    // So the stopwatch is deleted rather than converted, and `computeMs` is
    // gone from PoseRecord with it. **When this page is wired onto `src/pose`,
    // open a span around `runPose` here** -- and note that pose also hands
    // back `frame.gpu`, so the phone gets the same per-pass GPU breakdown the
    // desktop has, through `profiling/clocks.ts`'s `ingestGpuFrame`.
    // Both inputs the wiring needs are assembled and then discarded, the same
    // way `grayTopDown` has been since the pose path went dark. `poseInput` lost
    // its last reader when the AR overlay left for Table Topple -- it fed the
    // overlay's intrinsics -- but it is the shape `runPose` takes, so it is kept
    // built rather than deleted and re-derived by whoever wires this page.
    void grayTopDown; void poseInput;
    const pose: LocalPose = { recoveredAxes: null, positionDecode: null };
    const pd = pose.positionDecode;
    // `synced` and `boardSize` are filled in by recordPose itself -- see
    // poseRecords.ts on why a record cannot be allowed to disagree with the
    // module that tracks them.
    recordPose({
      tDrawn, frameMeta: frameMetaAtDraw,
      ok: !!pd,
      camPos: pd ? pd.camPos.toArray() : null,
      camQuat: pd ? pd.recoveredCamQuat.toArray() : null,
      distance: pose.recoveredAxes ? pose.recoveredAxes.distance : null,
      dnormal: pose.recoveredAxes ? pose.recoveredAxes.Dnormal.toArray() : null,
      consistency: pd ? pd.consistency : null,
    });
    // ── The one place IMU correction actually changes behaviour ──────────
    //
    // Anchored at tDrawn, NOT now: the reconstruction just took ~136ms, so
    // treating the result as a measurement of the present would hand the
    // tracker a pose that is a whole fix-interval stale and let it "correct"
    // toward the past. This is the delayed-measurement problem in its
    // smallest form -- there is no state buffer to replay here, but at least
    // the timestamp is honest.
    if (pd) {
      const res = imuTracker.anchor(tDrawn, pd.camPos, pd.recoveredCamQuat);
      imuStats.fixes++;
      imuStats.lastVisionDeltaDeg = res.visionDeltaDeg;
      imuStats.lastGyroPathDeg = res.gyroPathDeg;
      if (!res.accepted) imuStats.rejected++;
      // Collected before the accept/reject branch matters: a rejected pair is
      // exactly a pair whose vision rotation is untrustworthy, so it must NOT
      // feed the frame-convention solve.
      if (res.accepted && res.reason === 'ok') {
        const dev = imuTracker.netDeviceRotation();
        const cam = rotationVectorBetween(lastAcceptedQuat, pd.recoveredCamQuat);
        if (dev.length() > 1e-6) {
          framePairs.push({ device: dev, camera: cam });
          if (framePairs.length > FRAME_PAIR_CAP) framePairs.shift();
        }
      }
      if (res.accepted) {
        lastAcceptedQuat.copy(pd.recoveredCamQuat);
        // World up expressed in the camera's own frame -- the vision-side half
        // of the direct calibration. Only updated on an ACCEPTED pose, so a
        // flipped one can never be snapped.
        setUpCamera(WORLD_UP.clone().applyQuaternion(pd.recoveredCamQuat.clone().invert()).normalize());
      }
    }

    // The IMU A/B's counter. It used to sit opposite a write of the overlay
    // camera -- with correction OFF the reconstruction was the only writer and a
    // failed decode hid the overlay, with it ON the render loop coasted through.
    // That contrast WAS the A/B, and it left with the overlay; what remains here
    // is the prediction count, which still says how often the tracker carried a
    // frame the decode could not.
    if (useImuCorrection && imuTracker.hasFix()) imuStats.predicted++;

    if (isRelayOpen()) {
      const msg: any = {
        type: 'poseResult', w: cw, h: ch,
        recoveredAxes: pose.recoveredAxes ? {
          Drow: pose.recoveredAxes.Drow.toArray(), Dcol: pose.recoveredAxes.Dcol.toArray(),
          Dnormal: pose.recoveredAxes.Dnormal.toArray(), distance: pose.recoveredAxes.distance,
        } : null,
        positionDecode: pd ? {
          row: pd.row, col: pd.col,
          consistency: pd.consistency,
          camPos: pd.camPos.toArray(),
          recoveredCamQuat: pd.recoveredCamQuat.toArray(),
          orientation: pd.orientation,
        } : null,
      };
      // `sendDebugInfo` used to attach a debug blob here: the decode grid's
      // coverage, the period/phase result, the vote composites, and a SECOND
      // explicit LSD pass run purely for the desktop's rectangle overlay. All of
      // it came from the deleted pipeline. The toggle and its checkbox stay --
      // the image half below is independent of it -- but there is nothing left
      // to attach.
      void sendDebugInfo;
      // Raw bytes now (poseResultWire.ts), not a base64 dataUrl -- toBlob's
      // callback is awaited HERE, inside captureComputeAndSendPose's own
      // try block, so devicePoseComputing (which this function's own
      // finally clears, see its own comment) stays true for the encode's
      // full duration. Without that, devicePoseLoop's next iteration could
      // start redrawing sendCanvas before this frame's bytes are actually
      // read out -- the same hazard sendEncodeInFlight guards against for
      // captureAndSendFrame's own toBlob call above, just already covered
      // here by the guard this function already had.
      if (sendCapturedImage) {
        const blob = await new Promise<Blob | null>((resolve) => sendCanvas.toBlob(resolve, 'image/jpeg', 0.85));
        if (blob) sendBinary(packPoseResultWithImage(msg, blob));
        else sendJson(msg);
      } else {
        sendJson(msg);
      }
    }

    // Local-only -- never sent over the network (the whole point of this
    // toggle is seeing the phone's TRUE compute speed, and timing is
    // deliberately excluded from the wire payload for now, see this
    // session's on-device-pose-recovery plan).
    // NO MILLISECONDS HERE, and that is the honest reading rather than a
    // regression. This used to print `pose <totalMs>ms (<fps>fps)` off the
    // stopwatch deleted above -- which, once the pose computation went with
    // the old pipeline, was timing an empty statement and reporting ~0ms at an
    // implausible frame rate. A readout that says nothing beats one that says
    // the phone reconstructs instantly.
    //
    // The number comes back with the pipeline: see the TODO at the capture site.
    poseReadoutEl.textContent = pd ? 'pose recovered' : 'no pose pipeline on device yet  [no fix]';
  } finally {
    devicePoseComputing = false;
  }
}

// Continuous self-paced loop for device-compute + video mode -- starts the
// next capture+compute+send cycle as soon as the previous one resolves,
// rather than checking a rAF tick like the desktop-compute videoLoop below
// (compute time, not frame delivery or backpressure, is the bottleneck
// here). Re-entrant-safe: stops on its own once computeOnDevice or
// captureMode changes out from under it.
async function devicePoseLoop() {
  if (!computeOnDevice || captureMode !== 'video' || devicePoseComputing) return;
  await captureComputeAndSendPose();
  if (computeOnDevice && captureMode === 'video') devicePoseLoop();
}

shutterBtn.addEventListener('click', () => {
  // In video mode the button is a status indicator, not a trigger -- frames
  // already send themselves via the loop below.
  if (captureMode !== 'single') return;
  if (computeOnDevice) {
    if (devicePoseComputing) return;
    captureComputeAndSendPose();
    shutterBtn.classList.add('sent');
    setTimeout(() => shutterBtn.classList.remove('sent'), 300);
    return;
  }
  if (sendGateStatus() !== 'ok') return;
  captureAndSendFrame();
});

// Ticks every frame; only actually sends in video mode. Pose Viewer's own
// mailbox is what paces this -- it always takes the freshest frame and drops
// stale ones -- so the only gate left is 'backpressure'. loopTicks/etc. count
// every outcome so a diagnostic script can tell which one actually explains
// the idle gap, instead of guessing from timing alone.
function videoLoop() {
  requestAnimationFrame(videoLoop);
  // Keeps captureCanvas (the VISIBLE viewfinder) live regardless of
  // capture/compute mode, at the same cadence videoLoop itself already
  // ticks -- but only draws the screen-capped preview resolution
  // (previewLongEdgeCap), not the camera's full negotiated resolution. The
  // capture functions above draw their OWN full-resolution frame into
  // sendCanvas on demand instead of reading off this one, so what actually
  // gets analyzed/sent is unaffected by this downscale.
  drawCurrentFrameToCanvas();
  loopTicks++;
  // Device-compute video mode is driven by its own self-paced devicePoseLoop
  // (compute time, not rAF ticks/backpressure, is the pacing signal there) --
  // see this session's on-device-pose-recovery plan.
  if (captureMode !== 'video' || computeOnDevice) return;
  const status = sendGateStatus();
  if (status === 'backpressure') { backpressureBlockedTicks++; return; }
  if (status === 'not-ready') { readinessBlockedTicks++; return; }
  sendsAttempted++;
  captureAndSendFrame();
}

// ── Boot, and handing each module what it cannot know ────────────────────
//
// Everything below runs at the END of this file, deliberately: these closures
// read `computeOnDevice`, `captureMode`, `cameraSettings` and friends, all
// declared above, and attaching earlier would be fine only by accident of when
// the first call happens. This is the same hazard the AR render loop hit as a
// single file -- reading a binding that had not been initialized yet -- made
// structural instead of avoided by scheduling.
//
// The order within this block matters in exactly one place: connectRelay()
// must follow attachRelayHost(), because the socket's very first 'open' event
// asks the host what modes to announce.

attachImuHost({
  computingOnDevice: () => computeOnDevice,
  settingsSyncedAt,
  facing: () => (currentFacing === 'user' ? 'user' : 'environment'),
  sendBatch: (payload) => sendJson(payload),
});

attachRelayHost({
  captureMode: () => captureMode,
  computingOnDevice: () => computeOnDevice,
  onSettingsSync: applySettingsSync,
  onReadyChange: setReady,
  // The dev-bridge eval's scope is DECLARED, in inspect.ts, rather than being
  // whatever this file or relay.ts happens to import. See that file's header
  // for why -- the surface has already degraded once without a compiler word.
  evalCode: evalInScope,
});
connectRelay();

startCamera('environment').catch((e: any) => {
  camStatus.textContent = 'camera error: ' + e.message;
});

videoLoop();
