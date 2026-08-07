import * as THREE from 'three';
import { type CompositeLine } from '../types.ts';
import type { RemotePoseMessage } from '../pipeline/capture.ts';
import type { Intermediates, PendingIntermediates } from '../../pose/intermediates.ts';
import { type GridPeriodPhaseResult } from '../../pose/stages/period/gridPeriodPhase.ts';
import type { PoseComputeTiming } from '../../pose/poseCompute.ts';
import { type TransferSummary } from '../../pose/gpu/fieldResidency.ts';
import { type DecodeCellDebug, type DecodeSampleGrid, type PositionDecodeResult, type ProjectedBins, type RecoveredAxes, type Vote } from '../types.ts';
import { type ProfileSpan } from '../profiling/profiler.ts';
import { type PhysicalCameraSettings, type SimulatedCameraSettings } from './settings.ts';

// requestVideoFrameCallback's metadata for one decoded video frame, captured
// on the phone (mobileCapture.ts's onVideoFrame) and relayed with the frame
// it describes. All times are on the PHONE's clock, in epoch milliseconds,
// except mediaTime/presentationTime/expectedDisplayTime which are on the
// browser's own media/presentation timelines.
//
// `captureTime` is the field genuinely wanted for IMU fusion and it is
// OPTIONAL IN THE SPEC -- widely present for WebRTC sources, not guaranteed
// for getUserMedia ones. Everything else is recorded alongside it precisely
// because which field is trustworthy is per-device, and picking one here
// would bake in a guess that the calibration step is supposed to test.
export interface FrameMeta {
  mediaTime: number; presentationTime: number; expectedDisplayTime: number;
  captureTime: number | null; processingDuration: number | null;
  presentedFrames: number | null; observedAt: number;
}

// ── Camera model ─────────────────────────────────────────────────────────
//
// Any number of these can exist at once (zero included -- see this file's
// header note); every per-camera THREE object/buffer lives on this object
// rather than as a module-level singleton.

export interface CameraBase {
  id: string;
  name: string;
  color: THREE.Color;

  // -- recovered/decoded state -- already fully source-agnostic (MATH_QUAT-
  // frame recovery + solveRecoveredCamQuat), unchanged by this stage.
  lastRecoveredAxes: RecoveredAxes | null;
  lastPositionDecode: PositionDecodeResult | null;
  lastDecodeGrid: DecodeSampleGrid | null;
  lastDecodeRotated: DecodeSampleGrid | null;
  lastDecodeCorrectness: (DecodeCellDebug | null)[][] | null;
  // The requested intermediates' deferred readback, parked here by
  // computePoseFromCapture and drained by pipeline/axesReconstruction.ts's
  // runVisualTail. Non-null only between a reconstruction finishing and its
  // visual tail running, and only when something was actually asked for.
  pendingIntermediates: PendingIntermediates | null;
  // What that drain handed back -- the pose run's own fx/fy/regionId/regions/
  // rects, for the overlays that used to recompute them. See
  // pose/intermediates.ts.
  intermediates: Intermediates | null;
  lastProjectedBins: ProjectedBins | null;
  // Bus traffic the LSD chain incurred on this camera's last frame -- see
  // pose/poseCompute.ts's PoseComputeState for why it is recorded rather
  // than derived from the toggles. Drives the readout under the GPU toggles.
  lastChainTransfers: TransferSummary | null;
  lastVotes: Vote[];
  // Root-tagged composite lines the votes above were cast from (pipeline/
  // votes.ts's computeGradient2x2Composites) -- the SAME lines pipeline/
  // gridPeriodPhase.ts classifies into rowLines/colLines, and what the
  // "color composite lines by row/col family" debug overlay draws, so a
  // line's root is guaranteed to mean the same thing in all three places.
  lastVoteComposites: { root: number; line: CompositeLine }[] | null;
  // Raw fit result (Drow/Dcol/Dnormal only, no distance) BEFORE period-search
  // gating -- the pole markers render off this even when gridPeriodPhase
  // fails and lastRecoveredAxes ends up null (see
  // pose/poseCompute.ts/applyPoseVisualizations). Its own field, rather
  // than reusing lastRecoveredAxes, specifically so that degenerate-period
  // behavior survives the Step 2 refactor instead of silently regressing.
  lastQuadricPair: { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null;
  axesComputed: boolean;
  axesCapturing: boolean;
  lastAxesCapture: number;
  // Cached output of runAxesReconstruction's own capture stage (a fresh GPU
  // render+readback for simulated, or whatever's currently in
  // lastRealCaptureGray for physical), already flipped to analysis row
  // order -- recomputeFromLastCapture (pipeline/axesReconstruction.ts)
  // reads this instead of taking a fresh capture, so settings that only
  // affect stages downstream of capture (LSD tuning,
  // gridPeriodPhaseGapLowerBound, minGrazingCos, forceCPU) can recompute without re-rendering/re-photographing.
  // Deliberately a SEPARATE buffer from lastNoisedPreviewGray -- that one is
  // also written by the passive preview loop on its own throttled cycle
  // (see pipeline/preview.ts), which would otherwise risk this drifting
  // from what reconstruction actually last used. Invalidated (set to null)
  // by resizeCaptureBuffers so a stale-sized buffer can never be reused
  // after a viewport/supersample resize.
  lastAxesCaptureGray: { gray: Float64Array; w: number; h: number } | null;
  // -- deferred-visualization mailbox --
  //
  // One slot, freshest-wins, exactly like pendingCapture below, but holding no
  // payload at all: the display tail is an idempotent function of this
  // camera's already-settled state, so "there is newer state to paint" is the
  // whole message. Set at the end of recomputeStages, drained by animate()
  // once this camera is no longer capturing (pipeline/axesReconstruction.ts's
  // markVisualsDirty/drainVisuals).
  visualsDirty: boolean;
  // Guards the drain against re-entering itself: the tail awaits real GPU work
  // (the projection, the texture paint), so it spans several animate() ticks
  // and would otherwise be started again on each of them.
  visualsDraining: boolean;
  // The per-stage timings computePoseFromCapture returned for the frame the
  // mailbox is holding, kept here rather than passed as an argument because
  // the drain runs long after its producer returned. Read only to build the
  // readout's timing line, so a null (nothing captured yet, or a pose that
  // arrived from a device-compute phone) just means that line is omitted.
  lastPoseTiming: PoseComputeTiming | null;

  // -- capture/analysis buffers, shared shape for both camera types --
  rtSize: { w: number; h: number };
  aspect: number; // rtSize.w / rtSize.h -- replaces the old module-level RT_ASPECT
  pipRect: { x: number; y: number; w: number; h: number };
  captureDirty: boolean;
  lastPreviewUpdate: number;
  lastNoisedPreviewGray: Float64Array | null;
  // The Through-Cam field-pixel index (row*w+col) currently under the
  // cursor, or null if the cursor isn't over the field this frame --
  // tracked unconditionally by overlays/hoverDebugOverlays.ts's
  // updateHoverOverlays (not just when its own gradient-arrow hover is on),
  // so OTHER hover-driven repaints (overlays/lsdOverlay.ts's raw-regions
  // highlight) can read it directly off the camera without a circular
  // import back into hoverDebugOverlays.ts (which already imports FROM
  // lsdOverlay.ts).
  lastHoverFieldIndex: number | null;
  lastGridPeriodPhase: GridPeriodPhaseResult | null;
  // Interactive pan/zoom state for the period/phase debug plot (overlays/
  // gridPeriodPhaseOverlays.ts) -- null means "no interaction yet, use the
  // default bracket-relative view". Deliberately NOT in settings (not
  // persisted) -- this is scroll position, not a configuration choice, and
  // a saved zoom level from a totally different capture wouldn't mean
  // anything as a default for the next session.
  gridPeriodPhaseViewMin: number | null;
  gridPeriodPhaseViewMax: number | null;

  distortedPreviewData: Uint8Array; distortedPreviewTex: THREE.DataTexture;
  projectedPreviewData: Uint8Array; projectedPreviewTex: THREE.DataTexture;
  trueContamData: Uint8Array; trueContamTex: THREE.DataTexture;
  reconContamData: Uint8Array; reconContamTex: THREE.DataTexture;
  topGradientData: Uint8Array; topGradientTex: THREE.DataTexture;
  // Per-pixel raster overlays for pose/stages/lsd/lsdSegments.ts's own debug views
  // (overlays/lsdOverlay.ts) -- same "flat Uint8Array + DataTexture + quad"
  // shape as trueContamData/reconContamData above, not the shared SVG
  // overlay LSD rectangles/composite lines use, since these paint a colored
  // pixel PER MEMBER PIXEL of a region rather than a small number of
  // vector shapes. lsdRawRegionsData: accepted rectangles' own rawMembers,
  // colored per-blob. lsdRejectedData: rejected rectangles' rawMembers, flat red.
  lsdRawRegionsData: Uint8Array; lsdRawRegionsTex: THREE.DataTexture;
  lsdRejectedData: Uint8Array; lsdRejectedTex: THREE.DataTexture;

  // -- THREE objects: recovered side (both camera types have these) --
  recoveredCamGizmo: THREE.Mesh; recoveredCamAxes: THREE.AxesHelper;
  recoveredRowPoleA: THREE.Mesh; recoveredRowPoleB: THREE.Mesh;
  recoveredColPoleA: THREE.Mesh; recoveredColPoleB: THREE.Mesh;
  recoveredFloorOverlayMat: THREE.MeshBasicMaterial; recoveredFloorOverlay: THREE.Mesh;
  // The World-view recovered-floor quad's OUTLINE -- a 4-point closed
  // THREE.LineLoop, drawn unconditionally from pose+FOV alone whenever
  // lastRecoveredAxes/lastPositionDecode exist, in EITHER compute mode (see
  // overlays/recoveredOverlays.ts's updateRecoveredFloorOutline and this
  // session's on-device-pose-recovery plan). Deliberately separate from
  // recoveredFloorOverlay (the actual projected-image FILL, a textured
  // Mesh) -- that one still requires lastProjectedBins (real pixel data),
  // which device-compute mode never populates.
  recoveredFloorOutline: THREE.LineLoop;

  // -- Great-sphere group: repositioned (not rotated) to the camera's own
  // origin each frame, since every direction it draws is expressed in WORLD
  // axes. --
  sphereAnchor: THREE.Object3D;
  sphereShell: THREE.Mesh;
  circlesGroup: THREE.Group;
  rowCirclePool: THREE.Line[]; colCirclePool: THREE.Line[];
  frustumLine: THREE.LineLoop;
  patchGeo: THREE.BufferGeometry; patchMat: THREE.MeshBasicMaterial; patchMesh: THREE.Mesh;
  // Rendered as a thin flat TRIANGLE RIBBON (2 triangles per circle segment,
  // extruded in-plane to +-halfWidth around the true SPHERE_RADIUS), not a
  // native GL line -- a real "fat line" addon (three/addons/lines,
  // LineSegments2/LineMaterial) was tried first for an adjustable stroke
  // weight, but rendered nothing at all in this environment (confirmed via
  // live dev-bridge inspection: draw calls happened, geometry data was
  // correct, no GL errors, yet zero pixels) for reasons not worth chasing
  // further -- ordinary triangles + MeshBasicMaterial has no such failure
  // mode. See updateGradientCirclesDebug for the ribbon construction.
  gradientCirclesGeo: THREE.BufferGeometry; gradientCirclesMat: THREE.MeshBasicMaterial; gradientCirclesLines: THREE.Mesh;
  axisVectorsGeo: THREE.BufferGeometry; axisVectorsMat: THREE.LineBasicMaterial; axisVectorsLines: THREE.LineSegments;
}
export interface SimulatedCamera extends CameraBase {
  type: 'simulated';
  settings: SimulatedCameraSettings;
  // Ground-truth pose, driven by settings.camX/Y/Z/camYawDeg/camPitchDeg --
  // see updateGizmo.
  camPos: THREE.Vector3; camQuat: THREE.Quaternion;
  gizmoCam: THREE.PerspectiveCamera; gizmoBody: THREE.Mesh; gizmoAxes: THREE.AxesHelper;
  camHelper: THREE.CameraHelper;
  camRT: THREE.WebGLRenderTarget;
  captureRTSize: { w: number; h: number };
  // Ground-truth pole markers (ROW_DIR/COL_DIR comparison) -- no equivalent
  // for a physical camera, since there's no ground truth to compare against.
  polesGroup: THREE.Group;
  rowPoleA: THREE.Mesh; rowPoleB: THREE.Mesh; colPoleA: THREE.Mesh; colPoleB: THREE.Mesh;
}
export interface PhysicalCamera extends CameraBase {
  type: 'physical';
  settings: PhysicalCameraSettings;
  lastRealCaptureGray: Float64Array | null;
  lastRealCaptureW: number; lastRealCaptureH: number;
  // The dev-bridge server's own id for the phone connection this camera was
  // auto-created for (see initDevBridge's realCapture handler) -- every
  // PhysicalCamera has one now, and always will: there's no manual/UI path
  // to create one anymore (see this file's header), only a real phone
  // connecting. That's what lets the tab bar's close button unconditionally
  // KICK any physical camera (see renderCameraTabs) instead of needing to
  // ask whether there's really a connection behind it to kick.
  connectionId: string;
  // Purely a reflection of whatever mode.html's mode toggle last reported
  // (see devBridge/client.ts's captureMode handler) -- Sphere Lab never
  // sets this itself, only displays it (see ui/cameraPanel.ts).
  captureMode: 'single' | 'video';
  // Same idea, for the phone's "compute pose on this device" toggle (see
  // this session's on-device-pose-recovery plan) -- purely a reflection of
  // devBridge/client.ts's computeMode handler, never set by Sphere Lab
  // itself. 'device' means the phone sends {type:'poseResult'} (already-
  // computed pose, no image); 'desktop' (the default -- see
  // camera/factory.ts) means it keeps sending {type:'realCapture'} like
  // today.
  computeMode: 'desktop' | 'device';
  // Mirrors lastReportedPipelined's "force a mismatch on the first tick"
  // trick (see its own comment) for pushing an initial settingsSync the
  // moment a physical camera is seen in main.ts's animate loop, rather than
  // waiting for a settings slider to actually change first -- see this
  // session's on-device-pose-recovery plan. Cleared (set false) right after
  // that first send.
  neverSyncedSettings: boolean;
  // Diagnostic-only summary of the phone's own pipeline intermediates for
  // its most recent device-compute pose (composite/vote counts, decode
  // grid valid-sample ratio, gridPeriodPhase's period/phase/height) -- only
  // populated when the phone's sendDebugInfo toggle is on (default off),
  // see pipeline/capture.ts's ingestRemotePose. null otherwise, INCLUDING
  // right after that toggle gets switched off (never left showing a stale
  // frame's debug data).
  lastRemoteDebug: {
    compositeLineCount: number; voteCount: number;
    gridPeriodPhase: {
      period: number; phiRow: number; phiCol: number; height: number | null;
      seedPeriod: number; bracket: [number, number]; rowLineCount: number; colLineCount: number;
    } | null;
    decodeGrid: { rows: number; cols: number; validCount: number; totalCount: number } | null;
    decodeCorrectness: { correctCount: number; wrongCount: number } | null;
  } | null;
  // Mirrors axesCapturing, but tracks what was last actually SENT to the
  // phone as a captureReady signal (see main.ts's animate loop), so that
  // signal only goes out on a genuine true/false transition instead of
  // every frame.
  lastReportedReady: boolean;
  // Mailbox slot for an arriving capture (see devBridge/client.ts's
  // realCapture handler and main.ts's animate loop pump) --
  // always overwritten with the newest arrived frame, never queued, so a
  // desktop that falls behind naturally drops stale frames instead of
  // working through a backlog. sentAt/pulledAt/encodedAt/receivedAt are all
  // nowMs() (sphereLab/clock.ts) -- epoch ms, so they stay comparable across
  // the phone and the desktop the way bare performance.now() could not, but
  // also sub-millisecond and monotonic, which Date.now() was not. These four
  // were the last Date.now() holdouts and are what that module unified. So
  // ingestRealCapture can split "pull the video frame onto a canvas" from
  // "JPEG encode" from "actual network transit" on pop, instead of lumping
  // them into one number and guessing which one dominated a given slow
  // sample -- see lastPullMs/lastEncodeMs/lastTransitMs.
  // bytes is blob.size -- the real JPEG byte count now that the image
  // travels as a genuine binary WebSocket frame (devBridge/client.ts)
  // rather than base64 text inside JSON, so this is exact rather than the
  // ~1.33x-inflated string-length approximation it used to be.
  pendingCapture: {
    blob: Blob; sentAt: number; pulledAt: number; encodedAt: number; receivedAt: number; bytes: number;
    // ── Added for IMU fusion; both OPTIONAL because an older phone client
    // against a newer desktop simply won't send them ──────────────────────
    //
    // drawnAt is the phone's high-resolution monotonic twin of pulledAt.
    // frameMeta is requestVideoFrameCallback's own metadata for the frame
    // that was drawn, and is the only thing here that says anything about
    // when the photons actually landed -- sentAt/pulledAt/encodedAt all
    // describe the SEND, which happens an unknown amount of time after
    // capture. A filter must timestamp its measurement from these, not from
    // sentAt; see mobileCapture.ts's latestFrameMeta comment for why a
    // constant timestamp error is specifically dangerous here (it produces
    // pose error proportional to velocity, which is indistinguishable from
    // the problem being solved).
    drawnAt?: number;
    frameMeta?: FrameMeta | null;
  } | null;
  // Mailbox slot for a device-compute phone's poseResult, exactly mirroring
  // pendingCapture's own "always overwrite with the newest arrived message,
  // never queued" semantics -- see devBridge/client.ts's poseResult
  // handling (both the plain-JSON no-image branch and the binary
  // poseResultWire.ts branch write here, never call ingestRemotePose
  // directly) and main.ts's animate loop, which drains it under the same
  // captureIngestBusy guard as pendingCapture. A given phone only ever
  // populates one of these two mailboxes (computeMode is stable per
  // connection), so sharing captureIngestBusy between them is correct, not
  // a hack.
  pendingPoseResult: RemotePoseMessage | null;
  // True from the moment the pump pulls a frame out of EITHER mailbox above
  // until ingestRealCapture's/ingestRemotePose's own async work has finished
  // (both then hand off into runAxesReconstruction/applyPoseVisualizations,
  // which own axesCapturing itself from that point). Needed as its own flag
  // because that async work happens BEFORE axesCapturing flips true --
  // without this, a frame landing mid-decode could get popped a second time
  // and race the first one.
  captureIngestBusy: boolean;
  // Diagnostic-only, for tracking down video mode's idle round-trip gap
  // (this session's chat). Opened the instant runAxesReconstruction's
  // finally block flips axesCapturing back to false, closed the instant
  // ingestRealCapture starts on the next mailbox frame -- so its duration
  // IS the round trip (phone capture+encode, network there, mailbox pump
  // delay, network back) with zero attribution to any single stage on its
  // own. Never null in practice now that spans record unconditionally; the
  // type stays nullable because nothing has opened one yet before the first
  // reconstruction completes.
  idleSpan: ProfileSpan | null;
  // Approximate phone-side "pull the current video frame onto a canvas"
  // duration (canvas resize + drawImage, NOT the JPEG encode itself) for
  // the most recently ingested frame -- pendingCapture.pulledAt -
  // pendingCapture.sentAt.
  lastPullMs: number | null;
  // Approximate phone-side JPEG encode duration (toDataURL only, now that
  // pullMs is split out separately) for the most recently ingested frame --
  // pendingCapture.encodedAt - pendingCapture.pulledAt. Cross-device
  // wall-clock diff (see pendingCapture's own comment on why that's
  // "approximate" rather than nanosecond-precise).
  lastEncodeMs: number | null;
  // Approximate actual network transit duration (ws.send on the phone to
  // this message handler on the desktop, including the relay hop) for the
  // most recently ingested frame -- pendingCapture.receivedAt -
  // pendingCapture.encodedAt.
  lastTransitMs: number | null;
  // Rolling histories (capped, oldest dropped) of lastPullMs/lastEncodeMs/
  // lastTransitMs -- a single sample is noisy, this is what lets a
  // diagnostic script report a real distribution and tell pull-bound/
  // encode-bound/transit-bound samples apart instead of guessing.
  pullMsHistory: number[];
  encodeMsHistory: number[];
  transitMsHistory: number[];
  // pendingCapture.bytes for the same samples, same index alignment as
  // transitMsHistory -- lets a diagnostic script compute actual throughput
  // (bytes / transit ms) instead of guessing whether a given duration is
  // bandwidth-bound from timing alone.
  payloadBytesHistory: number[];
  // Self-reported by mobile-capture.html every ~2s (its flush interval).
  // Two distinct diagnostic purposes bundled in one message:
  //   - nominalFrameRate/avgIntervalMs/maxIntervalMs/sampleCount (from
  //     requestVideoFrameCallback) -- lets us tell "the round trip is slow"
  //     apart from "the phone's camera hardware itself isn't producing new
  //     frames any faster than this" (auto-exposure in low light can drop a
  //     phone camera's actual delivered frame rate well below nominal).
  //     avgIntervalMs/maxIntervalMs/sampleCount are null if
  //     requestVideoFrameCallback never fired during that window (missing
  //     browser support, or camera hardware not delivering frames at all).
  //   - loopTicks/backpressureBlockedTicks/readinessBlockedTicks/
  //     sendsAttempted (from the video loop itself) -- says WHY the loop
  //     skipped a tick: backpressure means bufferedAmount > 0 (a previous
  //     frame is still physically draining over the network, i.e.
  //     bandwidth-bound), not-ready means Sphere Lab itself said to wait
  //     (shouldn't happen much when pipelined), and if loopTicks over the
  //     ~2s window is well under what 60Hz would predict, the phone's own
  //     requestAnimationFrame is being starved (backgrounded tab, thermal
  //     throttling) independently of network entirely.
  // null until the phone's sent at least one frameStats message.
  lastFrameStats: {
    nominalFrameRate: number | null;
    avgIntervalMs: number | null; maxIntervalMs: number | null; sampleCount: number | null;
    loopTicks: number; backpressureBlockedTicks: number; readinessBlockedTicks: number; sendsAttempted: number;
  } | null;

  // ── IMU (phase A of the motion-blur/IMU plan: RECORDING ONLY) ──────────
  //
  // Raw devicemotion samples relayed from this phone, newest last, capped at
  // IMU_RING_CAPACITY. Nothing in the pose pipeline reads these -- they exist
  // so a session can be dumped and replayed offline, which is the only way a
  // filter can be tuned against a fixed dataset rather than against whatever
  // the phone happened to be doing at the time.
  //
  // `t` is the phone's own high-resolution monotonic epoch clock
  // (mobileCapture.ts's nowMs), stamped when the event HANDLER RAN -- which
  // is not when the sensor sampled. That offset is a phase B calibration,
  // not a defect, and correcting it here by guesswork would destroy the
  // information needed to estimate it.
  //
  // Angular rates are DEGREES/SECOND and accelerations m/s^2, in the DEVICE
  // frame, not the camera's -- see mobileCapture.ts's block comment on axes
  // before using any of this for anything.
  imuSamples: {
    t: number;
    rrAlpha: number; rrBeta: number; rrGamma: number;
    ax: number; ay: number; az: number;
    agx: number; agy: number; agz: number;
    interval: number;
  }[];
  // Per-batch context + liveness, from the most recent 'imu' message.
  // screenAngle is what a replay needs to get from the DeviceMotion frame to
  // anything camera-relative; receivedAt lets a stale stream be told apart
  // from a phone that simply isn't moving (both leave the samples static).
  lastImuMeta: { screenAngle: number; rateHz: number | null; receivedAt: number; totalReceived: number } | null;

  // The most recently INGESTED frame's raw timing, kept because pendingCapture
  // is a mailbox that gets popped and discarded -- see ingestRealCapture's own
  // comment. Everything except receivedAtDesktop is on the PHONE's clock, the
  // same clock imuSamples[].t uses, and the two are only meaningful together.
  // receivedAtDesktop is carried alongside precisely so the cross-device skew
  // stays visible rather than being silently baked into a difference.
  lastCaptureTiming: {
    sentAt: number; pulledAt: number; encodedAt: number;
    drawnAt: number | null; frameMeta: FrameMeta | null;
    receivedAtDesktop: number;
  } | null;
}
export type Camera = SimulatedCamera | PhysicalCamera;
