import * as THREE from 'three';
import { CompositeLine } from '../pipeline/bucketFillJoin.ts';
import type { RemotePoseMessage } from '../pipeline/capture.ts';
import { GridPeriodPhaseResult } from '../pipeline/gridPeriodPhase.ts';
import { GrownRegion, LsdRectangle } from '../pipeline/lsdSegments.ts';
import { DecodeCellDebug, DecodeSampleGrid, PositionDecodeResult, ProjectedBins, RecoveredAxes, Vote } from '../types.ts';
import { ProfileSpan } from '../profiling/profiler.ts';
import { PhysicalCameraSettings, SimulatedCameraSettings } from './settings.ts';

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
  lastProjectedBins: ProjectedBins | null;
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
  // pipeline/poseCompute.ts/applyPoseVisualizations). Its own field, rather
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
  // affect stages downstream of capture (LSD/join-walk tuning,
  // weightSharpenPower, gridPeriodPhaseGapLowerBound, minGrazingCos, the
  // useGPU* toggles) can recompute without re-rendering/re-photographing.
  // Deliberately a SEPARATE buffer from lastNoisedPreviewGray -- that one is
  // also written by the passive preview loop on its own throttled cycle
  // (see pipeline/preview.ts), which would otherwise risk this drifting
  // from what reconstruction actually last used. Invalidated (set to null)
  // by resizeCaptureBuffers so a stale-sized buffer can never be reused
  // after a viewport/supersample resize.
  lastAxesCaptureGray: { gray: Float64Array; w: number; h: number } | null;

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
  // The from-scratch traditional LSD pipeline's own debug output (pipeline/
  // lsdSegments.ts) -- accepted rectangles AND rejected/retried candidates
  // (see LsdRectangle's own `accepted`/`retries` fields), so the debug
  // overlay can show both, not just the survivors.
  lastLsdRectangles: LsdRectangle[] | null;
  // growRegionsCCL's own raw output (pre-fit, pre-NFA) for the SAME call
  // updateLsdOverlay's showLsdRawRegions branch already makes -- cached here
  // purely so hover highlighting (overlays/lsdOverlay.ts's
  // repaintLsdRawRegionsHighlight) can look up "which region owns this pixel" on
  // every pointermove without re-running growRegionsCCL itself.
  // mag/theta are the full fields, kept so the edge-connectivity hover view
  // (overlays/lsdOverlay.ts's drawEdgeConnectivityPreview) can re-run
  // growRegionsCCL's OWN edge predicate for the single hovered pixel on every
  // pointermove without recomputing the gradient field. They replaced a
  // regionSumCos/regionSumSin/strideDivider trio that existed only to feed the
  // JFA grower's growth-candidate arrows -- that view needed per-region
  // aggregate direction and per-pixel stride state to reconstruct where a
  // pixel would jump next; the predicate is a plain pairwise test between
  // 8-neighbors now, so the raw fields are all it takes.
  // roundsRun/converged report what the round loop actually did, so the
  // readout can distinguish "reached the fixpoint in 9 rounds" from "still
  // mid-growth because the CCL-steps scrubber is capping it".
  lastLsdGrownRegions: {
    regionId: Int32Array; regions: GrownRegion[];
    mag: Float64Array; theta: Float64Array;
    roundsRun: number; converged: boolean;
  } | null;
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
  tangentWalkPathData: Uint8Array; tangentWalkPathTex: THREE.DataTexture;
  // Per-pixel raster overlays for pipeline/lsdSegments.ts's own debug views
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
  // Same "only send on a real change" throttle as lastReportedReady, but
  // for globalState.useCapturePipelining riding along on the same message
  // (see main.ts's animate loop) -- deliberately initialized to the
  // OPPOSITE of that setting's actual default (see factory.ts) so a
  // freshly-connected phone gets synced immediately instead of waiting for
  // the first real busy/idle transition, which might not happen for a
  // while.
  lastReportedPipelined: boolean;
  // Mailbox slot for globalState.useCapturePipelining (see devBridge/
  // client.ts's realCapture handler and main.ts's animate loop pump) --
  // always overwritten with the newest arrived frame, never queued, so a
  // desktop that falls behind naturally drops stale frames instead of
  // working through a backlog. sentAt/pulledAt/encodedAt/receivedAt are all
  // Date.now() (wall-clock, cross-device -- NOT performance.now(), which
  // has an unrelated per-process epoch on the phone vs the desktop) so
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
  // own. Null whenever profilerEnabled() is false, same as any other span.
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
}
export type Camera = SimulatedCamera | PhysicalCamera;
