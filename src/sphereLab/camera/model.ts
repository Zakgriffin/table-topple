import * as THREE from 'three';
import type { RemotePoseMessage } from '../pipeline/capture.ts';
import { type ProjectedBins } from '../types.ts';
import { type Span } from '../profiling/profiler.ts';
import { type PhysicalCameraSettings, type SimulatedCameraSettings } from './settings.ts';

// ── The decode sample lattice, as the Projected-Cam overlay draws it ──────
//
// The lattice `decode.build` sampled: one point per cell, in FLOOR (u, v)
// coordinates, which is the same space pipeline/projectedBins.ts projects the
// image into -- so a dot goes on the Projected-Cam rect with no further
// geometry. It is separable by construction (`u` depends only on the column,
// `v` only on the row), which is why this is two short arrays and not one of
// length rows*cols.
//
// `packed` is the DEVICE's own per-cell verdict, copied verbatim rather than
// recomputed: bit 0 is "resolvable" and carries decode.build's three guards --
// the grazing cutoff, behind-the-camera, and on-screen -- and bit 1 is the
// sampled bit. Re-deriving that host-side would put a second copy of the
// projection in an overlay, which is the mistake this whole display path exists
// to avoid; asking for the buffer costs 83 KiB.
export interface DecodeLattice {
  rows: number; cols: number;
  /** Floor u of column j, and floor v of row i. */
  uAt: Float64Array; vAt: Float64Array;
  /** Per cell `i * cols + j`: bit 0 resolvable, bit 1 the sampled bit. */
  packed: Uint32Array;
  /**
   * Per cell: 1 agrees with the printed board, 0 disagrees, -1 not comparable
   * (unresolvable, or outside the winning orientation's extent).
   *
   * NULL when the frame did not decode, and ALSO when the host re-derivation
   * disagreed with the counters the device reported -- see
   * pipeline/decodeLattice.ts. A lattice with no rings is the honest rendering
   * of "we cannot say"; rings drawn from a re-derivation that failed its own
   * check would be worse than none.
   */
  correct: Int8Array | null;
}

// The region CSR, exactly as `src/pose2` holds it: three buffers that are one
// fact. Grouped rather than spread across three optional fields because two of
// them are meaningless alone -- an offset into an absent `members` is not a
// partial answer, it is a crash waiting for a caller who forgot to check the
// third field. `fx`/`fy` stay separate above because each IS a field on its own.
//
// TOP-DOWN pixel indices (`y * w + x`), ascending within a region --
// collect.finalize sorts each slice, since the atomic cursor that fills it hands
// out slots in arrival order. Regions are a disjoint partition of the pixels
// that survived the edge floor, so `members` is bounded by `w * h` exactly and
// most of it is unused on a typical frame.
export interface RegionCsr {
  /** Pixel index per member, region slices laid end to end. */
  members: Uint32Array;
  /** Where region r's slice starts in `members`. */
  offsets: Uint32Array;
  /** How long it is. */
  sizes: Uint32Array;
}

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

// ── The pose, as the app DISPLAYS it ─────────────────────────────────────
//
// ONE object, where there used to be thirteen `last*` fields on the camera. An
// atomically swappable object and an exploded copy of it spread across a camera
// are not the same thing: the copy can be observed halfway through being
// replaced, and pipeline/axesReconstruction.ts's visual tail reads camera fields
// on both sides of its awaits.
//
// ── FLAT, AND EVERY FIELD OPTIONAL. THAT IS THE WHOLE RULE ──
//
// This is what the app KNOWS about the pose on screen, and each field is present
// exactly when it is known. Two different things make a field absent, and the
// type deliberately does not distinguish them, because no reader should care:
//
//   - the RUN did not produce it (a degenerate fit has no triad; an undecodable
//     frame has no anchor; a pose relayed from a phone has neither a detector
//     report nor a status);
//   - nobody ASKED for it. `src/pose2` keeps its intermediates on the device and
//     reads back 128 bytes; anything larger arrives only when a display toggle
//     put it in that frame's request. See INSPECT (what may ever be read) and
//     `inspectFor` (what this frame asked for) in pipeline/axesReconstruction.ts.
//
// It was NOT flat before, and the split it had was historical rather than
// meaningful: `votes` and `voteComposites` sat at the top level while `fx`/`fy`
// sat inside a nested `intermediates` bag, though all four are opt-in on exactly
// the same terms. Three more fields (`gridPeriodPhase`, `chainTransfers`,
// `timing`) were typed `null` -- placeholders for shapes the deleted pipeline
// had, written on every path and read by nothing.
//
// ── ABSENCE IS NOT A FAILURE REPORT. `status` IS ──
//
// The readout used to infer WHY a frame produced nothing from which field came
// back null, which is why `quadricPair` existed at all: the same three vectors
// as `recoveredAxes`, gated one stage earlier, so that "degenerate fit" could be
// told from "no lattice". The pipeline reports that directly -- §15's bits, in
// `status` -- so the inference is gone and so is the duplicate triad.
//
// ORIENTATION, the trap that outlives every refactor: `fx`/`fy`, `rects`,
// `regions` and `lattice` are the PIPELINE's own buffers, so they are TOP-DOWN,
// the dominant convention everywhere except the preview textures. The flip
// happens at the PAINT, and it is not merely a row reversal -- a vertical flip
// also negates the vertical derivative and shifts the 2x2 stencil by a row, so
// flipRows(computeField(g)) and computeField(flipRows(g)) are different arrays.
export interface CameraPose {
  /** `POSE2_STATUS` bits, verbatim (see src/pose2/pose.ts). Absent when no local
   *  run produced them, i.e. a pose relayed from a phone. This is the ONLY
   *  honest source for why a frame came back empty -- `ordinary` outcomes, `cap`
   *  outcomes and `budget` outcomes all look identical from a null field. */
  status?: number;
  /** How many lines the detector found, off the pose block rather than off
   *  `votes.length` -- that array is opt-in, and a readout keyed to its length
   *  reported "0 lines" on a good frame whenever the circles overlay was off. */
  lineCount?: number;
  /** How many line-support regions collect kept. Also off the pose block, so it
   *  needs no readback of its own; CLAMPED to maxRegions, unlike the block's own
   *  value, which is raw so `finish` can report `regionOverflow`. */
  regionCount?: number;

  /** The recovered axis frame and its scale. Absent when the fit was degenerate
   *  or the lattice never resolved -- `status` says which. */
  recoveredAxes?: RecoveredAxes;
  /** Where the camera is, in board coordinates and in the world. Absent when no
   *  anchor won. */
  positionDecode?: PositionDecodeResult;

  // ── Opt-in display buffers, present only when this frame requested them ──

  /** The gradient field. f32 from the device, not the f64 the app's own gradient
   *  functions produce -- see types.ts's FloatField for why nothing converts. */
  fx?: Float32Array;
  fy?: Float32Array;
  /** One vote normal per detected line, in MATH_QUAT's fixed math frame.
   *  Zero-weight votes are dropped rather than passed on. */
  votes?: { n: THREE.Vector3; weight: number }[];
  /**
   * The detected segments. `region` is the index of the region each was fitted
   * to -- the key every other view of that region colours by. `index` is the
   * segment's own slot in the pipeline's line arrays, which is what joins it to
   * `rectified`/`lineFamily` below.
   */
  composites?: { region: number; index: number; line: CompositeLine }[];
  /**
   * Per LINE, 4 f32: value, weight, crossMin, crossMax -- the segment's
   * coordinates once the recovered axes flatten the floor. `value` is the
   * coordinate the line holds CONSTANT (xCol for a row line, xRow for a column
   * line) and cross[Min,Max] span the one it RUNS ALONG. Gnomonic units:
   * multiply by `recoveredAxes.distance` for floor coordinates.
   *
   * Flat and unbounded by its own length, like `rects` -- see `lineCount`.
   */
  rectified?: Float32Array;
  /** Per LINE: 1 row family, 0 column family, -1 not classified (a degenerate
   *  line, or one whose endpoints had no gnomonic image). */
  lineFamily?: Int8Array;
  /**
   * One fitted rectangle per REGION, flat: 10 f32 each --
   * cx, cy, theta, length, width, nfaLog10, accepted, pad, n, k. Bounded by
   * `regionCount`, not by the array's length.
   *
   * FLAT, and deliberately not unpacked into objects: `src/pose2` hands back
   * bytes and the one consumer walks them once per capture. A parallel array of
   * ten-field objects would be the deleted pipeline's LsdRectangle rebuilt on
   * this side, which is what this type exists to avoid.
   */
  rects?: Float32Array;
  /** Which pixels each region is made of. Only the two per-pixel rasters need it. */
  regions?: RegionCsr;
  /** Where decode sampled the floor, and what it read. */
  lattice?: DecodeLattice;
}

// ── The display half of the deleted pipeline's vocabulary ────────────────
//
// Three shapes the UI genuinely draws, re-declared here as the APP's own rather
// than imported from a library that no longer exists. They are geometry, not
// pipeline structure -- an axis triad, a decoded position, a line segment --
// which is why these three came across and the rest did not.
export interface RecoveredAxes { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3; distance: number }
export interface CompositeLine { x1: number; y1: number; x2: number; y2: number }
// `votes` and `totalWindows` were here and are gone: the winning anchor's vote
// count and the number of windows that voted at all were carried on both paths
// and displayed by neither. So were `correct`/`wrong`, added one commit earlier
// on the belief that the sample lattice's self-check read them here -- it reads
// them off the Pose2Result directly, at unpack time, and never needed a copy on
// the display type.
export interface PositionDecodeResult {
  row: number; col: number; consistency: number;
  camPos: THREE.Vector3;
  // The camera's TRUE world orientation, solved entirely from the pattern.
  // Anything placed into the actual 3D scene needs this to convert
  // recoveredAxes' Drow/Dcol/Dnormal (expressed in MATH_QUAT's fixed math
  // frame) into true world space first.
  recoveredCamQuat: THREE.Quaternion;
  // Which of the 4 cardinal rotations the decode matched at -- display-only.
  orientation: number;
}

// The deferred-visualization mailbox's payload -- the pose the tail is to
// paint, plus the live pipeline result it has to read to finish painting it.
//
// The pose here is the SAME OBJECT that was published to camera.pose, not a
// copy, so the tail can ask "is what I was handed still what is on screen?" by
// identity before it publishes an enriched version over the top.
//
// `result` is the raw PoseResult, carried ONLY here and only until the tail
// runs. It is what holds the arena slices, and the reason those do not travel
// on CameraPose: this field has a stated expiry (the next reconstruction) and
// the pose on screen does not.
// `result` used to be the raw PoseResult, carried only until the tail ran,
// holding the arena slices the tail read its intermediates from. There is no
// PoseResult and no arena, so the payload is the pose alone.
export interface PendingVisuals {
  pose: CameraPose;
}

export interface CameraBase {
  id: string;
  name: string;
  color: THREE.Color;

  // -- recovered/decoded state -- already fully source-agnostic (MATH_QUAT-
  // frame recovery + solveRecoveredCamQuat).
  //
  // The pose currently on screen, or null until this camera has recovered one.
  // Published in one assignment, twice per reconstruction: once when the pose
  // itself is final, and again by the visual tail with `intermediates` filled
  // in (see pipeline/axesReconstruction.ts's applyPoseResult and
  // runVisualTailBody). Both are whole-object swaps -- there is no moment at
  // which some of these fields are this run's and the rest are the last one's.
  //
  // `axesComputed` was a fourteenth field and is gone too: it only ever meant
  // `!!quadricPair` (orientation-fit success), and two sites set it by hand.
  pose: CameraPose | null;
  lastProjectedBins: ProjectedBins | null;
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
  // One slot, freshest-wins, exactly like pendingCapture below -- and it
  // carries a PAYLOAD, which pendingCapture and pendingPoseResult always did
  // and this one did not.
  //
  // It was a bare boolean, on the reasoning that the tail is an idempotent
  // function of this camera's already-settled state, so "there is newer state
  // to paint" was the whole message. That was true of the PAINTING and false
  // of the READING: the tail took a handle off the camera, awaited it, and
  // wrote four fields back, so a reconstruction landing mid-tail put frame N's
  // decode grid next to frame N+1's pose. The boolean had no payload because
  // the payload was smeared across thirteen fields; with one pose object there
  // is something to put in the slot, and the tail reads nothing it was not
  // handed. Posted at the end of recomputeStages, drained by animate() once
  // this camera is no longer capturing (pipeline/axesReconstruction.ts).
  pendingVisuals: PendingVisuals | null;
  // Guards the drain against re-entering itself: the tail awaits real GPU work
  // (the projection, the texture paint), so it spans several animate() ticks
  // and would otherwise be started again on each of them.
  visualsDraining: boolean;

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
  // Per-pixel raster overlays for pose/stages/lsd/the LSD stage's own debug views
  // (overlays/lsdOverlay.ts) -- same "flat Uint8Array + DataTexture + quad"
  // shape as trueContamData/reconContamData above, not the shared SVG
  // overlay LSD rectangles/composite lines use, since these paint a colored
  // pixel PER MEMBER PIXEL of a region rather than a small number of
  // vector shapes. lsdRawRegionsData: the grown regions' member pixels, colored
  // per-blob. lsdRejectedData: the member pixels of the regions whose rectangle
  // was rejected, flat red -- joined `rects[i]` to `regions[i]`, see
  // pose/stages/lsd/types.ts.
  lsdRawRegionsData: Uint8Array; lsdRawRegionsTex: THREE.DataTexture;
  lsdRejectedData: Uint8Array; lsdRejectedTex: THREE.DataTexture;

  // -- THREE objects: recovered side (both camera types have these) --
  recoveredCamGizmo: THREE.Mesh; recoveredCamAxes: THREE.AxesHelper;
  recoveredRowPoleA: THREE.Mesh; recoveredRowPoleB: THREE.Mesh;
  recoveredColPoleA: THREE.Mesh; recoveredColPoleB: THREE.Mesh;
  recoveredFloorOverlayMat: THREE.MeshBasicMaterial; recoveredFloorOverlay: THREE.Mesh;
  // The World-view recovered-floor quad's OUTLINE -- a 4-point closed
  // THREE.LineLoop, drawn unconditionally from pose+FOV alone whenever
  // pose.recoveredAxes/pose.positionDecode exist, in EITHER compute mode (see
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
  // sample -- see the `link.pull`/`link.encode` peer spans in profiling/clocks.ts.
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
  idleSpan: Span | null;
  // ── The four latency histories and the three lastXMs fields are GONE ──
  //
  // They were a hand-rolled ring buffer per quantity, written every frame and
  // read by exactly one diagnostic script -- a second way to measure a duration
  // alongside the profiler, which is what the profiling rewrite exists to end.
  // The phone-side durations are `peer` spans in the one record store now (see
  // profiling/clocks.ts), the newest record IS the latest sample, and the byte
  // count rides on `ingest.run` as a span attribute.
  //
  // The raw phone-clock STAMPS survive on `lastCaptureTiming` below, because the
  // IMU work needs absolute capture times rather than durations.
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
