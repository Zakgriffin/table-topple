import * as THREE from 'three';
import { type Camera, type CameraPose, type CompositeLine, type PendingVisuals } from '../camera/model.ts';
import { activeCamera, isPhysical, isSimulated } from '../camera/store.ts';
import { COL_DIR, GRID_STEP, ROW_DIR, SPHERE_RADIUS } from '../constants.ts';
import { board } from '../floorPattern.ts';
import { getGPUDevice } from '../../gpu/device.ts';
import { boardDims } from '../../pose/board.ts';
import { POSE_STATUS, decodeLayout } from '../../pose/pose.ts';
import {
  type PoseContext, type PoseFrame, type PoseSettings,
  createPoseContext, destroyPoseContext, runPose,
} from '../../pose/run.ts';
import { angleBetweenDegV } from '../math/geometry.ts';
import { updatePositionReadoutText } from '../overlays/projectedCamOverlays.ts';
import { applyRecoveredFloorOverlay, updateRecoveredCamGizmo, updateRecoveredFloorOutline } from '../overlays/recoveredOverlays.ts';
import { updateGradientCirclesDebug } from '../overlays/sphereOverlays.ts';
import { globalState } from '../state.ts';
import { axesReadout, captureAxesBtn, lsdChainTransfers } from '../ui/dom.ts';
import { backendFromForceCPU } from '../backend.ts';
import { captureDistortedGrayscale, getAnalysisVFovRad } from './capture.ts';
import { computeProjectedBinsAuto, paintProjectedTexture, type ProjectedSampleResult } from './projectedBins.ts';
import { buildDecodeLattice } from './decodeLattice.ts';
import { flipRowsF64 } from './distortion.ts';
import { refreshModeVisualizations } from './modeRefresh.ts';
import { type Span, spanEnd } from '../profiling/profiler.ts';
import { ingestGpuFrame } from '../profiling/clocks.ts';
import { appSpan } from '../profiling/stages.ts';

// Shared pole-marker/gizmo/floor-overlay/readout tail -- called after EITHER
// a real local reconstruction (recomputeStages below) or an already-computed
// pose arriving from a device-compute phone (pipeline/capture.ts's
// ingestRemotePose), so both paths render identically off the same
// camera.pose object instead of duplicating this logic -- see this session's
// on-device-pose-recovery plan. `extraReadoutLine`, if given, is appended as
// the readout's last line (recomputeStages passes its per-stage timing
// breakdown; a device-compute pose has no local timing to report, so
// ingestRemotePose passes nothing).
// ── WHY THIS FRAME PRODUCED NOTHING, from the pipeline rather than by guess ──
//
// The readout used to say "degenerate fit" whenever the triad field was null,
// which was the only failure it could name -- everything downstream of the fit
// arrived as an equally null `recoveredAxes` or `positionDecode` and read as the
// same thing. §15's bits distinguish them at the source, and they distinguish
// three CATEGORIES the display was flattening together: an ORDINARY outcome (no
// board in this frame), a CAP (raise a constant), and a BUDGET (the round count
// was too small). Collapsing those into "failed" is exactly what §15 warns
// against, one level up.
//
// Ordered upstream-first and returns the FIRST hit, because the later bits are
// consequences: a degenerate fit guarantees no lattice, which guarantees no
// anchor, and reporting all three would bury the cause under its own effects.
// `gridOverflow` is deliberately absent -- it is diagnostic and the frame decoded
// correctly anyway.
function failureLine(pose: CameraPose | null | undefined): string | null {
  if (!pose) return 'no capture yet';
  // A remote pose carries no status, and its own absence of axes is not a
  // failure to report -- the phone sends what it recovered, not what it did not.
  if (pose.status === undefined) return null;
  const s = pose.status;
  const S = POSE_STATUS;
  if (s & S.noRegions) return 'no regions above the edge floor -- nothing to fit';
  if (s & S.noVotes) return 'no line segments survived the NFA test';
  if (s & S.fitDegenerate) return 'degenerate fit -- the votes span no plane pair';
  if (s & S.gppNoSamples) return 'no lines could be rectified onto the recovered axes';
  if (s & S.gppNoCandidates) return 'no grid period fit the rectified lines';
  if (s & S.layoutInvalid) return 'no lattice -- the period or the hull came out empty';
  if (s & S.decodeNoAnchor) return 'no board anchor matched -- the pattern was not recognised';
  // Capacity and budget, reported even on a frame that otherwise worked: these
  // mean "raise a constant" and "raise the round count", not "no board".
  if (s & S.regionOverflow) return `more than ${MAX_REGIONS} regions -- raise MAX_REGIONS`;
  if (s & S.lineOverflow) return `more than ${MAX_LINES} lines -- raise MAX_LINES`;
  if (s & S.growNotConverged) return 'region growing hit its round budget before converging';
  return null;
}

export function applyPoseVisualizations(camera: Camera, isActive: boolean, extraReadoutLine?: string) {
  let orientationErrorLine: string | null = null;
  const pose = camera.pose;
  // Straight off `recoveredAxes` now. There was a second field, `quadricPair`,
  // holding the SAME THREE VECTORS gated one stage earlier -- so that the readout
  // could tell "the fit was degenerate" from "the lattice never resolved" by
  // which of the two came back null. `status` reports that outright (see
  // failureLine), so the duplicate triad is gone; and both of its drawing
  // consumers ANDed it with `positionDecode` anyway, which cannot be present
  // unless the lattice resolved.
  const rowDirRecovered = pose?.recoveredAxes?.Drow ?? null;
  const colDirRecovered = pose?.recoveredAxes?.Dcol ?? null;
  if (pose?.positionDecode && rowDirRecovered && colDirRecovered) {
    const { recoveredCamQuat } = pose.positionDecode;
    const rowDirWorld = rowDirRecovered.clone().applyQuaternion(recoveredCamQuat);
    const colDirWorld = colDirRecovered.clone().applyQuaternion(recoveredCamQuat);
    // Decode's own 4-way disambiguation (tallyPositionVotes, see
    // decodeGrid.ts) can legitimately swap which of Drow/Dcol maps to
    // the world ROW vs COL axis (and negate either) -- fitPairOfPlanes
    // only ever recovers the row/col PLANE PAIR up to that ambiguity,
    // by construction (it's a property of the quadric fit, not a bug).
    // axisErr is UNDIRECTED (angle to the nearer of +axis/-axis) since
    // both ends of an axis already get their own pole marker -- a
    // clean 180-degree flip isn't actually wrong, just a labeling
    // choice for which end is which. Picking whichever of the two
    // (row->ROW,col->COL) / (row->COL,col->ROW) pairings has the lower
    // TOTAL undirected error is legitimate here (unlike the old
    // pre-decode version of this check used to be) because
    // rowDirWorld/colDirWorld are genuinely in world space now --
    // decode has already resolved which pairing is physically correct,
    // this just detects which one it was.
    const axisErr = (v: THREE.Vector3, axis: THREE.Vector3) => Math.min(angleBetweenDegV(v, axis), angleBetweenDegV(v, axis.clone().negate()));
    const errUnswapped = axisErr(rowDirWorld, ROW_DIR) + axisErr(colDirWorld, COL_DIR);
    const errSwapped = axisErr(rowDirWorld, COL_DIR) + axisErr(colDirWorld, ROW_DIR);
    const swapped = errSwapped < errUnswapped;
    // Red pole markers always track whichever recovered vector ended
    // up closest to the world ROW axis, blue always tracks whichever
    // is closest to COL -- a fixed rowDirWorld->red assignment would
    // sometimes put red poles next to the blue ground-truth poles
    // whenever swapped is true.
    const redDirWorld = swapped ? colDirWorld : rowDirWorld;
    const blueDirWorld = swapped ? rowDirWorld : colDirWorld;
    camera.recoveredRowPoleA.position.copy(redDirWorld).multiplyScalar(SPHERE_RADIUS);
    camera.recoveredRowPoleB.position.copy(redDirWorld).multiplyScalar(-SPHERE_RADIUS);
    camera.recoveredColPoleA.position.copy(blueDirWorld).multiplyScalar(SPHERE_RADIUS);
    camera.recoveredColPoleB.position.copy(blueDirWorld).multiplyScalar(-SPHERE_RADIUS);

    if (isSimulated(camera)) {
      const rowErr = axisErr(redDirWorld, ROW_DIR);
      const colErr = axisErr(blueDirWorld, COL_DIR);
      orientationErrorLine = `row err ${rowErr.toFixed(2)}°  col err ${colErr.toFixed(2)}°  [post-decode${swapped ? ', swapped' : ''}]`;
    }
  }
  updateRecoveredCamGizmo(camera);
  applyRecoveredFloorOverlay(camera);
  // Drawn unconditionally alongside the fill above -- guards on
  // recoveredAxes/positionDecode only (both present in EITHER compute mode),
  // unlike applyRecoveredFloorOverlay which still requires real pixel data
  // (lastProjectedBins).
  updateRecoveredFloorOutline(camera);

  if (isActive) {
    const haveGroundTruth = isSimulated(camera);
    // The DETECTOR's count, not the vote array's length -- that array only
    // arrives when a toggle asked for it (see inspectFor), and keying this line
    // to it reported "0 votes" on a perfectly good frame whenever the circles
    // overlay was off. ABSENT, not -1: a pose relayed from a phone is a pose,
    // not a detector report, and "we were not told" is exactly what an absent
    // field means. Says "no local detector run" rather than naming the remote
    // path, because a frame that never got to run one reads the same here.
    const lines = [pose?.lineCount === undefined
      ? 'no local detector run'
      : `${pose.lineCount} lines voted`];
    const failed = failureLine(pose);
    if (failed) lines.push(failed);
    else if (orientationErrorLine) lines.push(orientationErrorLine);
    if (pose?.recoveredAxes) {
      const trueDist = isSimulated(camera) ? camera.camPos.y : NaN;
      const dist = pose.recoveredAxes.distance;
      if (haveGroundTruth) {
        const err = (Math.abs(dist - trueDist) / trueDist) * 100;
        lines.push(`distance ${dist.toFixed(2)} (${err.toFixed(1)}% err)  true ${trueDist.toFixed(2)}`);
      } else {
        lines.push(`distance ${dist.toFixed(2)}`);
      }
    }
    const decode = pose?.positionDecode;
    if (decode) {
      lines.push(`decoded torus (row,col): (${decode.row}, ${decode.col})  consistency ${(decode.consistency * 100).toFixed(1)}%  camPos (${decode.camPos.x.toFixed(2)}, ${decode.camPos.y.toFixed(2)}, ${decode.camPos.z.toFixed(2)})`);
    }
    if (extraReadoutLine) lines.push(extraReadoutLine);
    axesReadout.textContent = lines.join('\n');
    updatePositionReadoutText(camera);
  }
}

// ── The LSD chain's bus-traffic readout ──────────────────────────────────
//
// What the chain actually cost the bus on the last frame. It used to answer a
// question the nine per-stage toggles raised -- savings along the chain are
// superadditive, so flipping one checkbox changed the traffic of stages you did
// not touch -- and with one switch left it answers a simpler one: is the chain
// still at ONE crossing (gray up, nothing down), or has something started
// pulling data back down? Sourced from FieldResidency's own ledger, which
// records a transfer at the moment it decides to make one, so the number cannot
// drift from the behaviour the way a hand-maintained estimate would.
//
// Worth reading alongside the measurement that motivated it, and against it:
// crossings stopped predicting wall clock once the grower moved to GPU (the top
// four configurations sit within 10.1-10.8ms regardless of whether they move
// 0.92MB or 2.42MB). So this readout shows what the configuration COSTS THE
// BUS, which is a real and stable quantity -- not what it costs in time.
export function updateChainTransfersReadout(camera: Camera | undefined) {
  // Two different empty states, deliberately not collapsed into one: on the
  // Global tab there may well be a captured camera sitting right there, and
  // saying "no capture yet" would be flatly untrue.
  if (!camera) {
    lsdChainTransfers.textContent = 'LSD chain: select a camera to see its bus traffic.';
    return;
  }
  // `chainTransfers` was the bus-crossing ledger the deleted pipeline kept, and
  // it is typed `null` on CameraPose now: src/pose crosses the bus once per
  // frame by construction, so there is no per-run traffic to summarize and the
  // ledger that measured it is gone too.
  lsdChainTransfers.textContent = 'LSD chain: no bus ledger (src/pose reads back once per frame).';
}

// ── Axes/position reconstruction (the big orchestrator) ──────────────────
//
// Split into two entry points so every settings slider -- not just "capture
// now" -- can recompute everything downstream of it (see this session's
// chat): runAxesReconstruction does the expensive capture (stage 1: a real
// GPU render+readback for simulated, or picking up whatever the phone last
// sent for physical) and stores it into camera.lastAxesCaptureGray;
// recomputeFromLastCapture does everything downstream of that (gradient
// field through pole markers/overlays), reading the cached capture instead
// of taking a fresh one. Both share recomputeStages for that downstream
// work so the guard/busy-UI/span bookkeeping isn't duplicated in a way that
// could drift out of sync.

// ── The display tail, and the mailbox that defers it ─────────────────────
//
// Everything below the pose itself: the projected bins + texture paint, the
// pole markers/floor overlay, and the mode-specific overlays. Split out of
// recomputeStages so it runs deferred, through a one-slot mailbox drained by
// animate(), rather than inline inside the pose window -- that ~20ms of
// display GPU work used to serialize against the same device queue the pose
// stages were using. It costs one animation frame (~16ms) of overlay lag.
//
// THE MAILBOX CARRIES A PAYLOAD, and it did not always. The old argument for a
// bare boolean was that this function reads nothing but already-settled camera
// state, so "there is newer state to paint" is the whole message. That held for
// the painting and not for the reading: the tail took the intermediates handle
// off the camera, awaited it, and wrote four fields back, so a reconstruction
// landing mid-tail wrote frame N's decode grid over frame N+1's state. Now the
// pose it is to paint arrives as an argument (`posted`), and the only thing it
// writes back is one pose object -- guarded by identity, see below.
//
// Coalescing is still free: freshest-wins on a one-slot mailbox means running
// once after three captures paints the third, not the first.
//
// `isActive` is re-evaluated HERE rather than inherited from whenever the
// reconstruction started, since deferral makes "was this the active camera
// 160ms ago" the wrong question -- what the readouts and mode overlays want
// is which camera is on screen at PAINT time.
async function runVisualTail(camera: Camera, posted: PendingVisuals): Promise<void> {
  // The tail's own root. It did not need one while structure came from the call
  // stack -- the tail simply had no parent open, so its spans were roots by
  // default. With structure declared, `app.project` and `app.overlays` have to
  // name something, and the drain's `pose.drain` belongs under this rather than
  // under the pose that handed the handle over a frame ago.
  const tailSpan = appSpan('app.tail');
  try {
    await runVisualTailBody(camera, posted);
  } finally {
    spanEnd(tailSpan);
  }
}

async function runVisualTailBody(camera: Camera, posted: PendingVisuals): Promise<void> {
  const isActive = camera === activeCamera();

  // ── THE DRAIN IS GONE, AND SO IS THE WRITE IT EXISTED TO GUARD ──────────
  //
  // This began by awaiting the display intermediates and REPUBLISHING the pose
  // with them filled in -- the second half of applyPoseResult's seam. That
  // existed because the old pipeline finalized the pose early and host-side, so
  // the display's own reads were a SECOND round trip, worth deferring so the
  // pose did not wait 0.45MB for data only an overlay wanted.
  //
  // pose has ONE fence. The pose is not known until the map resolves either,
  // and the inspected buffers ride in the same staging buffer behind it (see
  // src/pose/run.ts), so there is nothing left to defer: the pose published by
  // recomputeStages is already complete.
  //
  // What went with it is the reason this function had a non-idempotent step at
  // all. It now reads settled state and writes none, so the identity check that
  // stopped an older pose overwriting a newer one has nothing to guard -- and
  // the payload narrows to its remaining job, naming WHICH pose to paint.
  //
  // The tail itself stays, for the job that is still real: the projection and
  // the texture paint are display GPU work on the same queue as the pose, and
  // keeping them off the capture path is what deferral buys.

  // Painting projectedPreviewTex is a real GPU texture upload -- worth
  // skipping unless this camera's Projected-Cam view is what's actually
  // on screen right now. The numeric half (bins) stays unconditional --
  // camera.lastProjectedBins feeds the World-view floor overlay's decal
  // map for every camera, not just the displayed one. The RGBA half
  // also has to run whenever the World-view floor overlay is on, though
  // -- that overlay (see overlays/recoveredOverlays.ts) reuses
  // projectedPreviewTex as its decal map, so skipping the paint here
  // left it sitting at its all-zero (alpha 0, invisible) initial
  // contents for any camera that never happened to be viewed in
  // Projected-Cam mode first.
  const showProjected = (isActive && globalState.mode === 'projected')
    || (globalState.mode === 'world' && camera.settings.showRecoveredFloor);

  // The readout describes the toggle configuration that produced the frame
  // being painted right now, which is why it sits in the tail rather than
  // next to the chain that recorded it -- deferred, those are different
  // moments, and this is the one the user is looking at.
  if (isActive) updateChainTransfersReadout(camera);
  updateGradientCirclesDebug(camera);

  const projectSpan = appSpan('app.project');
  // Captured outside the `if` (stays null when there's no recovered axes to
  // project) so it can be handed to refreshModeVisualizations below instead
  // of that call recomputing the exact same (possibly GPU) result a second
  // time -- see modeRefresh.ts's own comment on precomputedProjection.
  let projResult: ProjectedSampleResult = null;
  if (camera.pose?.recoveredAxes) {
    projResult = await computeProjectedBinsAuto(camera, backendFromForceCPU(globalState.forceCPU));
    if (showProjected) paintProjectedTexture(camera, projResult);
  }
  spanEnd(projectSpan);

  const overlaySpan = appSpan('app.overlays');
  // applyPoseVisualizations MUST stay downstream of the projection above, not
  // move back up next to the pose: applyRecoveredFloorOverlay builds the floor
  // quad's geometry out of lastProjectedBins' own u/v extent (see
  // overlays/recoveredOverlays.ts), so hoisting it would size this frame's
  // floor from the PREVIOUS frame's bins. That coupling is the reason the
  // whole tail defers as one unit instead of only its expensive half.
  // The payload's own timings, not a camera field: this line describes the run
  // being painted, and the drain runs long after that run returned.
  // The per-stage breakdown (votes/fit/pose/distance/decode) came off the
  // deleted pipeline's PoseComputeTiming. pose does not report host-side stage
  // spans -- it is one submit -- so only the projection's own time is left.
  //
  // Read off the span rather than a performance.now() pair of its own, which is
  // what this was: `app.project` brackets exactly the same region, so the pair
  // was a second stopwatch measuring an interval already measured, and the two
  // could only ever agree or be a bug.
  const timingLine = `project ${(projectSpan.end - projectSpan.start).toFixed(0)}ms`;
  applyPoseVisualizations(camera, isActive, timingLine);
  spanEnd(overlaySpan);

  if (isActive) {
    // Everything mode-specific (Through-Cam's contamination/top-gradient/
    // bucket-fill/join/hover overlays and its grid-period/phase plot,
    // Projected-Cam's texture, World's recovered-floor decal) -- see
    // pipeline/modeRefresh.ts. Only meaningful for whichever camera is
    // actually on screen. projResult (computed above, possibly null) is
    // handed over so this doesn't pay for a second (possibly GPU)
    // re-projection of data it already has in hand. NOT part of
    // applyPoseVisualizations -- device-compute mode (ingestRemotePose)
    // deliberately never calls this, since it would re-derive
    // lastProjectedBins from zero real pixel data and undo the "no image ->
    // no fill" guard applyRecoveredFloorOverlay just applied.
    await refreshModeVisualizations(camera, globalState.mode, { value: projResult });
  }
}


// Drains the mailbox for one camera. Called from animate() every tick, for
// every camera -- cheap to the point of free when nothing is dirty, which is
// most ticks.
//
// Two things have to be true to START: something is pending, and this camera
// is not mid-reconstruction.
//
// NEITHER OF THIS GUARD'S ORIGINAL TWO REASONS SURVIVES, AND IT IS STILL HERE.
// Both halves of that are deliberate, so the history is worth keeping straight:
//
//   1. THE PROFILER's shared span stack, which a drain overlapping a capture
//      used to corrupt. Gone -- profiler.ts records flat intervals and joins
//      them against a declared table, so nothing can be reparented into
//      anything.
//   2. STALENESS. The tail used to take the intermediates handle off the
//      camera, await resolve(), and write four fields back, then read
//      lastRecoveredAxes across another await and lastPoseTiming across a
//      third. A reconstruction landing mid-tail wrote frame N's decode grid
//      over frame N+1's state. Gone too: the tail is handed its pose as a
//      payload and reads no camera pose field across an await, and its one
//      write is guarded by identity (see runVisualTailBody).
//
// So what remains is not a correctness argument at all -- it is the two
// supporting reasons below, which never justified it alone and are the reason
// dropping it was never expected to be a win. Dropping it is a ONE-LINE change
// here and in the two entry points, and it should be made against a MEASUREMENT
// on a real device, not on the strength of the hazard having gone away:
//
// That exclusion is MUTUAL: runAxesReconstruction and recomputeFromLastCapture
// both decline to start while visualsDraining is set.
//
//   A. It costs nothing anyone can feel. The window it blocks is the same
//      window that was already blocked before deferral existed -- the tail
//      used to run INSIDE axesCapturing, so "reconstruction + tail" was one
//      uninterruptible stretch either way. All this does is split the flag
//      that covers it in two.
//   B. Overlapping wouldn't buy much. The tail's cost is display GPU work on
//      the SAME device queue the pose stages use, so running it alongside the
//      next reconstruction trades a serial wait for queue contention. The win
//      here was never parallelism -- it is that the tail stops being AWAITED on
//      the pose path, and it gets that either way by running in the idle gap
//      between captures (~340ms of one, at the default 500ms interval against a
//      ~159ms reconstruction).
//
// Still per-camera, and that is now simply correct rather than a limitation:
// two cameras reconstructing at once collide in neither the profiler (flat
// records) nor the pose (each publishes its own camera's own field), and
// axesCapturing is per-camera too. There is exactly one camera today anyway
// (see main.ts's animate loop).
//
// Placement in animate() matters just as much as the guard: this must run
// BEFORE the auto-capture trigger. runAxesReconstruction sets axesCapturing
// synchronously, so on the one tick where a reconstruction has just finished
// and the next is about to start, whichever runs first wins the tick -- and if
// the capture wins every time (interval shorter than a reconstruction) the
// drain never runs at all.
export function drainVisuals(camera: Camera): void {
  const posted = camera.pendingVisuals;
  if (!posted || camera.visualsDraining || camera.axesCapturing) return;
  camera.pendingVisuals = null;
  camera.visualsDraining = true;
  runVisualTail(camera, posted)
    .catch((e) => console.error('[visuals] deferred refresh failed:', e))
    .finally(() => {
      camera.visualsDraining = false;
      // UNREACHABLE while the mutual exclusion above holds -- nothing can set
      // axesCapturing between this drain starting and finishing. Kept as the
      // invariant's own tripwire rather than deleted as dead code: if either
      // capture-side guard is ever relaxed, this is what stops a half-repainted
      // frame from being the LAST thing painted, by re-arming for a clean pass.
      // `??=`, not `=`: a newer reconstruction that posted while this drain ran
      // owns the slot, and re-arming with the payload we already spent would
      // put the OLDER pose back in front of it. Re-running on the spent payload
      // is free and paints the same thing -- resolve() is idempotent.
      if (camera.axesCapturing) camera.pendingVisuals ??= posted;
    });
}

// Assumes camera.lastAxesCaptureGray is already populated and
// camera.axesCapturing is already true -- callers (runAxesReconstruction,
// recomputeFromLastCapture) own the guard/busy-UI/RAF wrapper and the
// capture step itself.
// ── The one place a pose result lands on a Camera ─────────────────────────
//
// This is the seam, and it is one assignment. `runPose` returns a block and
// some bytes; `toCameraPose` turns those into the flat display type; this
// publishes it as camera.pose, the pose that is on screen -- a statement about
// the app's own bookkeeping the library should have no opinion on.
//
// It used to unpack the result into thirteen separate mutations, which was the
// staleness hazard's actual home: an atomically swappable object and an
// exploded copy of it spread across a camera are not the same thing. There is
// no halfway state to observe now because there is nothing to be halfway
// through -- and nothing arrives LATER either, since pose's one fence means the
// display buffers are already in the object being published.
function applyPoseResult(camera: Camera, pose: CameraPose): void {
  camera.pose = pose;
  // Posts to the mailbox. The pose is final; the tail is what is deferred. The
  // raw result used to ride along here because it held the arena slices the
  // drain read from -- there is no result type and no arena, so the payload is
  // the pose alone.
  camera.pendingVisuals = { pose };
}

// A pose that knows NOTHING -- which is now literally `{}`, and that is the flat
// type paying for itself. It is published when the reconstruction could not run
// at all (no device, or `runPose` threw), as opposed to running and finding no
// board: that case returns a real pose whose `status` says which §15 outcome it
// was. Both callers write their own readout text, so there is nothing to carry.
const NO_POSE: CameraPose = {};

// ── THE POSE CONTEXT: one per camera, per capture size ──────────────────
//
// It owns the pipeline's device buffers and the board's hash table, so it has to
// outlive a frame -- and every buffer is sized from (w, h) at plan time, so it
// must NOT outlive a resize.
//
// A WeakMap rather than a field on Camera: this is a CACHE KEYED ON a camera,
// not a property of one, and camera/model.ts would otherwise have to name a
// pipeline type to hold it. Same shape, and the same reasoning, as
// projectedBins.ts's srcGradCache.
const poseContexts = new WeakMap<Camera, { ctx: PoseContext; w: number; h: number }>();

// Caps on the two counts the pipeline cannot bound from the image alone. Shared
// with the sweep's own choice so a Pose Viewer capture and a swept pose overflow
// at the same point -- `lineOverflow`/`regionOverflow` mean "raise a constant"
// (§15), and they should mean it at the same constant in both places.
const MAX_REGIONS = 16384;
const MAX_LINES = 16384;

// ── WHAT THE DISPLAY READS BACK, declared once ───────────────────────────
//
// The catalogue, not the per-frame selection: it sizes the one staging buffer,
// so a display toggle becomes a per-frame decision instead of a context rebuild.
// It GROWS as each overlay is re-pointed at pose -- adding a name here is safe
// and costs a rebuild on the next capture.
//
// `triad` and `layout` are here because the POSE ITSELF needs them, not merely
// an overlay: the 128-byte pose block carries a position and a quaternion, and
// the app's `recoveredAxes` is the axis triad those are expressed against, which
// does not fit in the block's seven spare slots.
//
// Cheap by construction: this camera plans with `alias` off (the default), where
// every buffer already holds its own slot for the whole frame, so declaring one
// costs staging bytes and nothing else. See src/pose/buffers.ts's PlanOptions.
//
// `members` is the one genuinely large name here (1.17 MiB at 480x640, one u32
// per pixel) and it is still only staging bytes while `alias` is off -- but it
// is the reason the two rasters gate on their own toggles below rather than
// riding along with the rectangles.
const INSPECT = [
  'triad', 'layout', 'fx', 'fy', 'votes', 'lines', 'lineScan', 'rects',
  'members', 'regionOffsets', 'regionSizes', 'packed', 'result', 'samples', 'family',
] as const;

// ── WHAT THIS FRAME ACTUALLY ASKS FOR ────────────────────────────────────
//
// The catalogue above is what MAY be read; this is what crosses the bus, and
// keeping them apart is the point of the mechanism. `fx`/`fy` are 1.17 MiB each
// at 480x640 -- at the default capture interval, pulling them back for nobody is
// several MB/s spent on an overlay that is switched off.
//
// THIS FUNCTION IS ONE HALF OF A CONTRACT. overlays/pipelineField.ts returns null
// when the fields were not requested, and its consumers then draw NOTHING rather
// than recomputing a gradient of their own -- a silent fallback would reinstate
// the duplicated stage exactly where it is hardest to notice. So a toggle listed
// in one place and not the other fails as a blank overlay, which is visible,
// instead of as a second implementation, which is not.
//
// It also mirrors preview.ts's `overlaysNeedGray`, one stage earlier and for the
// same class of mistake. A new toggle whose overlay calls `pipelineField()`
// belongs in BOTH.
//
// Every toggle here must also INVALIDATE: turning one on cannot paint from a run
// that was never asked for its input, so the handlers in
// overlays/hoverDebugOverlays.ts call recomputeFromLastCapture rather than the
// overlay's own updater. That is a real cost -- flipping one re-runs the pose --
// and it is the trade: one pipeline, asked, instead of a second one, duplicated.
function inspectFor(camera: Camera): readonly string[] {
  const s = camera.settings;
  // Not optional: `recoveredAxes` is the axis triad the pose block's position and
  // quaternion are expressed against, and it does not fit in that block. These
  // two are the pose, not an overlay.
  const want = ['triad', 'layout'];
  const fields = s.showTopGradient || s.showTrueContamination || s.showReconstructedContamination
    || s.showGradientArrow || s.showLevelLineArrow;
  if (fields) want.push('fx', 'fy');
  // The same gate updateGradientCirclesDebug applies to itself (it returns
  // immediately when neither is on), stated one stage earlier so the bytes are
  // not fetched either. Its own comment is the reason it matters: a real capture
  // is hundreds of thousands of votes, which is why showTopCircles defaults off.
  if (s.showTopCircles || s.showAxisVectors) want.push('votes');
  // The segments themselves, plus what is needed to say WHICH REGION produced
  // each one -- see unpackComposites. Both are kilobytes.
  if (s.showLsdComposite) want.push('lines', 'lineScan');
  // The fitted rectangles, for the segment outlines and the accept/reject
  // readout. EITHER toggle: the rejected view is the same buffer filtered the
  // other way, so gating on showLsdSegments alone would leave it dark whenever
  // it was the only one on.
  if (s.showLsdSegments || s.showLsdRejected) want.push('rects');
  // The two per-pixel rasters. They paint a pixel per MEMBER PIXEL, which is why
  // they need the CSR and the rectangle outlines do not -- and why they are the
  // only display request that costs more than kilobytes. The rejected raster
  // additionally needs `rects` to know which regions to paint, and gets it from
  // the line above, which is gated on the same toggle.
  if (s.showLsdRawRegions || s.showLsdRejected) {
    want.push('members', 'regionOffsets', 'regionSizes');
  }
  // The decode sample lattice. `layout` is already in `want` -- it is the pose --
  // and it carries the cell POSITIONS, so this asks only for what the device
  // decided about each cell. 83 KiB at the board's dimensions, and it is what
  // stops the overlay from reprojecting the lattice itself.
  if (s.showSampleLattice) want.push('packed', 'result');
  // The rectified lines, and the family colouring of the composite lines -- TWO
  // VIEWS OF ONE CLASSIFICATION, which is why one pair of buffers serves both.
  // `samples`/`family` are per LINE (uncompacted), so they join to `composites`
  // by index; `rowSamples`/`colSamples` hold the same numbers compacted per
  // family, which would have cost a second join to get back to a line.
  if (s.showRectifiedLines || (s.showLsdComposite && s.showCompositeLineFamilies)) {
    want.push('samples', 'family');
    // The Through-Cam view needs the segments themselves to colour; the
    // Projected-Cam one does not, but asking twice is free -- runPose dedupes.
    if (s.showLsdComposite) want.push('lines', 'lineScan');
  }
  return want;
}

async function poseContextFor(camera: Camera, w: number, h: number): Promise<PoseContext | null> {
  const held = poseContexts.get(camera);
  if (held && held.w === w && held.h === h) return held.ctx;
  const device = await getGPUDevice();
  if (!device) return null;
  // Destroyed here rather than left to GC: a supersample drag walks through a
  // dozen sizes, and each context is ~16 MiB of device buffers.
  if (held) destroyPoseContext(held.ctx);
  // Read live rather than captured: `board` is swapped wholesale by the
  // board-size control, and a context outlives at most one such swap -- its
  // buffer sizes are baked from these dims, so it is rebuilt, not reused.
  const ctx = createPoseContext(device, {
    w, h, maxRegions: MAX_REGIONS, maxLines: MAX_LINES, ...boardDims(board),
  }, board, { inspect: INSPECT });
  poseContexts.set(camera, { ctx, w, h });
  return ctx;
}

// Straight off this camera's own sliders. The pipeline takes no defaults -- see
// config.ts: every one of these has exactly one source, and it is the JSON.
function poseSettingsFor(camera: Camera): PoseSettings {
  const s = camera.settings;
  const vFovRad = getAnalysisVFovRad(camera);
  return {
    grow: { rhoLow: s.lsdRhoNoiseThreshold, toleranceDeg: s.lsdToleranceDeg },
    collect: { rhoHigh: s.lsdRhoHighThreshold, minRegionSize: s.lsdMinRegionSize },
    lsdFit: {
      rho: s.lsdRhoNoiseThreshold, toleranceDeg: s.lsdToleranceDeg,
      nfaTestExponent: s.lsdNfaTestExponent, nfaEpsilon: s.lsdNfaEpsilon,
    },
    lines: { minLengthPx: s.lsdMinLengthPx },
    votes: { vFovRad },
    gpp: { vFovRad, cellPitch: GRID_STEP, minGrazingCos: s.minGrazingCos },
    layout: { vFovRad, cellPitch: GRID_STEP, minGrazingCos: s.minGrazingCos },
  };
}

// ── TWO SUCCESS GATES, WHERE THERE WERE THREE ────────────────────────────
//
//   recoveredAxes    a triad, and a lattice to scale it   !fitDegenerate && layout.valid
//   positionDecode   ...and an anchor in the board        pose.ok
//
// The third was `quadricPair`: the same triad gated on `!fitDegenerate` alone,
// so that a frame whose fit succeeded but whose lattice did not could still show
// pole markers and say "the axes are right, the position is not". Nothing ever
// used it that way -- both of its consumers ANDed it with `positionDecode`,
// which cannot exist without the lattice -- and the ONE thing it did buy, a
// readout able to distinguish a degenerate fit from a failed lattice, is now
// `status` saying so directly. See failureLine.
//
// They are still not derived from one another: a collapsed "did it work" flag is
// what §15 warns about, one level up.

// One vote per detected line: (nx, ny, nz, weight), a vec4 per slot, in
// MATH_QUAT's fixed math frame -- which is the frame updateGradientCirclesDebug
// rotates OUT of, and the frame the stage tests score these against.
//
// ZERO-WEIGHT VOTES ARE DROPPED, not passed on. A line whose two endpoints
// back-project to a degenerate plane votes with weight 0 rather than not voting
// (the buffer is written for every line, so the slot exists either way). Keeping
// them would put a zero into the min/max the circle colouring normalizes
// against, which is a display bug with no upstream cause.
function unpackVotes(bytes: ArrayBuffer, lineCount: number): { n: THREE.Vector3; weight: number }[] {
  const v = new Float32Array(bytes);
  // Clamped: `lineCount` is what the detector FOUND, and lines.emit truncates to
  // maxLines when it overflows (§15's lineOverflow). The buffer holds the kept
  // ones, so the buffer's own length is the authority.
  const n = Math.min(lineCount, v.length / 4);
  const out: { n: THREE.Vector3; weight: number }[] = [];
  for (let i = 0; i < n; i++) {
    const weight = v[i * 4 + 3]!;
    if (weight <= 0) continue;
    out.push({ n: new THREE.Vector3(v[i * 4]!, v[i * 4 + 1]!, v[i * 4 + 2]!), weight });
  }
  return out;
}

// The detected segments, each tagged with the region it came from.
//
// ── THE TAG IS THE REGION INDEX, AND THAT IS A DELIBERATE DEPARTURE ──
//
// The old app hashed a colour from `root`: the region's LOWEST-INDEX MEMBER
// PIXEL. Reproducing that exactly is possible -- it is `members[regionOffsets[r]]`
// -- but it costs the full region CSR (1.17 MiB of `members`) on a view that
// otherwise needs kilobytes, and `root` was never meaningful: lsdOverlay.ts's own
// comment calls it "just some deterministic, stable member pixel to hash off of".
//
// What DOES matter is that this view and the raw-regions raster agree, so a
// segment and the blob it was fitted to come out the same colour. Hashing the
// region index satisfies that in both, and the raster can keep using the CSR it
// needs for its pixels anyway. The visible change is that the palette is
// shuffled relative to the old app; the property the palette is FOR is kept.
//
// ── HOW A LINE FINDS ITS REGION, without re-deriving the predicate ──
//
// `lineScan` is the exclusive prefix of the accept flags, so region r emitted a
// line exactly when its running index INCREMENTS -- and that line is at
// lineScan[r].x. Reading it this way rather than re-testing `rects`'s accepted
// flag and length matters: the predicate lives in LINES_FLAG_WGSL, and a second
// copy here would drift silently the first time a threshold moves.
//
// `regionCount` comes from the POSE BLOCK, not from a `counts` readback. `finish`
// writes `pose[17] = counts.x`, so the number was always crossing the bus with
// the pose and inspecting `counts` for it was a second copy of a value we
// already had.
function unpackComposites(
  lines: ArrayBuffer, lineScan: ArrayBuffer, regionCount: number, lineCount: number,
): { region: number; index: number; line: CompositeLine }[] {
  const seg = new Float32Array(lines);
  const scan = new Uint32Array(lineScan);
  const emitted = Math.min(lineCount, seg.length / 4);
  const out: { region: number; index: number; line: CompositeLine }[] = [];
  for (let r = 0; r < regionCount; r++) {
    // vec2 per entry, and only .x carries the flag -- see BUFFERS.lineFlag.
    const at = scan[r * 2]!;
    // The tail past the last region compares against the total, which is what
    // `lineScan[r + 1]` would have held had the scan run one element further.
    const next = r + 1 < regionCount ? scan[(r + 1) * 2]! : emitted;
    if (next <= at || at >= emitted) continue;
    out.push({
      // `at` IS the line's own slot, and carrying it is what lets the family
      // colouring join this segment to its rectified coordinates without a
      // second scan of the same flags.
      region: r, index: at,
      line: { x1: seg[at * 4]!, y1: seg[at * 4 + 1]!, x2: seg[at * 4 + 2]!, y2: seg[at * 4 + 3]! },
    });
  }
  return out;
}

// (isRow, isCol) per line, collapsed to one signed byte. Both zero is a real
// third state -- gpp.classify writes it for a degenerate line and for the dead
// tail past `lineCount` -- and it is NOT the same fact as "column family", which
// is the encoding trap GPP_CLASSIFY_WGSL's own header records one level down.
function unpackFamilies(bytes: ArrayBuffer): Int8Array {
  const f = new Uint32Array(bytes);
  const out = new Int8Array(f.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = f[i * 2] === 1 ? 1 : (f[i * 2 + 1] === 1 ? 0 : -1);
  }
  return out;
}

// ── THE ONE PLACE THE TWO REAL SHAPES BECOME THE DISPLAY'S ONE ───────────
//
// `runPose` returns exactly two things: a 128-byte block and raw bytes per
// requested buffer. This turns them into the flat `CameraPose` the overlays read
// -- see camera/model.ts, whose rule is that every field is present exactly when
// it is known, whether that is because the RUN produced it or because a toggle
// ASKED for it.
//
// Spread-conditionally throughout, so an absent field is genuinely ABSENT rather
// than an empty array or a null standing in for one. That distinction is
// load-bearing on the display side: `votes: []` cannot be told from "no lines
// detected", and every reader that wanted the second meaning was reaching for
// `lineCount` anyway.
//
// TOP-DOWN, which is the trap overlays/pipelineField.ts exists to hold: `fx`/
// `fy`, `rects`, `regions` and `lattice` are the pipeline's own buffers and every
// display surface is bottom-up. The flip belongs at the PAINT, and it is not a
// row reversal -- it also negates the vertical derivative.
function toCameraPose(frame: PoseFrame): CameraPose {
  const { pose, inspected } = frame;
  // Three vec3<f32> at stride 16 -- see BUFFERS.triad. The stride is why this is
  // `i * 4` and not `i * 3`, and getting it wrong reads Drow.w as Dcol.x.
  const t = new Float32Array(inspected['triad']!);
  const axis = (i: number) => new THREE.Vector3(t[i * 4]!, t[i * 4 + 1]!, t[i * 4 + 2]!);
  const layout = decodeLayout(inspected['layout']!);
  const fitOk = (pose.status & POSE_STATUS.fitDegenerate) === 0;

  // CLAMPED, and the block's own value is not: collect.regionMeta writes the raw
  // label count so `finish` can compare it against maxRegions and report §15's
  // regionOverflow. Every array indexed by region is sized for maxRegions, so
  // publishing the raw count would walk a display loop off the end of `rects`.
  // The overflow is still reported, in `status`.
  const regionCount = Math.min(pose.regionCount, MAX_REGIONS);

  return {
    status: pose.status,
    lineCount: pose.lineCount,
    regionCount,

    // `distance` IS the camera's height above the floor -- the same quantity the
    // pose block reports as `height`, read here off the block that also carries
    // the axes it belongs with, so the two cannot be paired across a frame.
    ...(fitOk && layout.valid === 1
      ? { recoveredAxes: { Drow: axis(0), Dcol: axis(1), Dnormal: axis(2), distance: layout.distance } }
      : {}),
    ...(pose.ok
      ? {
        positionDecode: {
          row: pose.boardRow, col: pose.boardCol, consistency: pose.consistency,
          camPos: new THREE.Vector3(pose.position.x, pose.position.y, pose.position.z),
          recoveredCamQuat: new THREE.Quaternion(
            pose.quaternion.x, pose.quaternion.y, pose.quaternion.z, pose.quaternion.w),
          orientation: pose.orientation,
        },
      }
      : {}),

    // ── Opt-in from here down: present iff inspectFor asked ──
    ...(inspected['fx'] ? { fx: new Float32Array(inspected['fx']) } : {}),
    ...(inspected['fy'] ? { fy: new Float32Array(inspected['fy']) } : {}),
    ...(inspected['votes'] ? { votes: unpackVotes(inspected['votes'], pose.lineCount) } : {}),
    ...(inspected['lines'] && inspected['lineScan']
      ? { composites: unpackComposites(inspected['lines'], inspected['lineScan'], regionCount, pose.lineCount) }
      : {}),
    ...(inspected['rects'] ? { rects: new Float32Array(inspected['rects']) } : {}),
    ...(inspected['samples'] ? { rectified: new Float32Array(inspected['samples']) } : {}),
    ...(inspected['family'] ? { lineFamily: unpackFamilies(inspected['family']) } : {}),
    // All three or none -- see RegionCsr. They are requested together and this is
    // the one place that could hand back a half of one.
    ...(inspected['members'] && inspected['regionOffsets'] && inspected['regionSizes']
      ? {
        regions: {
          members: new Uint32Array(inspected['members']),
          offsets: new Uint32Array(inspected['regionOffsets']),
          sizes: new Uint32Array(inspected['regionSizes']),
        },
      }
      : {}),
    // Reassembled from the block that describes the lattice and the device's own
    // per-cell verdict. Returns null on a frame with no lattice, and `undefined`
    // drops the key -- so the overlay's "absent means blank" rule needs no second
    // condition.
    ...(inspected['packed'] && inspected['result']
      ? { lattice: buildDecodeLattice(layout, inspected['packed'], inspected['result'], pose) ?? undefined }
      : {}),
  };
}

// ── THE RECONSTRUCTION ITSELF ────────────────────────────────────────────
//
// One upload, one submit, one readback, one pose. Everything between the capture
// and `camera.pose` is inside `runPose`; this function is the app boundary and
// nothing more -- a context, this camera's settings, and the mapping onto the
// app's own display type.
//
// A FAILED FRAME PUBLISHES AN EMPTY POSE RATHER THAN KEEPING THE LAST ONE. An
// ordinary frame with no decodable board (§15's `ordinary` bits) is not an
// error, and leaving the previous capture's pose on screen would show a stale
// answer as if it were this frame's.
async function recomputeStages(camera: Camera) {
  const cap = camera.lastAxesCaptureGray;
  if (!cap) return;
  // Dropped, not released: a payload this camera never drained used to own the
  // chain's device buffers. There are none to own now.
  camera.pendingVisuals = null;

  const ctx = await poseContextFor(camera, cap.w, cap.h);
  if (!ctx) {
    if (camera === activeCamera()) axesReadout.textContent = 'no WebGPU device -- pose recovery is unavailable';
    applyPoseResult(camera, NO_POSE);
    return;
  }

  const poseSpan = appSpan('app.pose');
  let frame: PoseFrame;
  try {
    // Narrowed at the boundary: the capture path produces f64 and the pipeline
    // takes f32 (§4). The one place that conversion is paid, and it is paid here
    // rather than inside the library so `runPose` keeps a single input shape.
    frame = await runPose(ctx, Float32Array.from(cap.gray), poseSettingsFor(camera), inspectFor(camera));
  } catch (e) {
    // pose throws rather than falling back -- by design, and this is the app
    // boundary that has to say so out loud. An ORDINARY undecodable frame does
    // NOT come through here: it returns a pose with `ok` false and §15's bits
    // set. Anything that throws is the device or a caller mistake, so it gets a
    // readout of its own rather than being flattened into "no board found".
    console.error('[pose] reconstruction failed:', e);
    if (camera === activeCamera()) axesReadout.textContent = `pose failed: ${e instanceof Error ? e.message : String(e)}`;
    applyPoseResult(camera, NO_POSE);
    return;
  } finally {
    spanEnd(poseSpan);
  }
  // Translated and filed here, after the host span is closed, because the GPU
  // spans declare it as their parent and the join attaches a child to the
  // occurrence whose interval CONTAINS it -- which `app.pose` only does once it
  // has an end. Absent on a device without `timestamp-query`, which is a normal
  // state and not a degraded one.
  if (frame.gpu) ingestGpuFrame(frame.gpu, 'app.pose');
  applyPoseResult(camera, toCameraPose(frame));
}

export function runAxesReconstruction(camera: Camera) {
  if (camera.axesCapturing) return; // don't stack overlapping captures
  if (camera.visualsDraining) return; // ...or start one underneath a repaint, see drainVisuals
  camera.axesCapturing = true;
  const isActive = camera === activeCamera();
  const prevLabel = captureAxesBtn.textContent;
  if (isActive) {
    captureAxesBtn.disabled = true;
    captureAxesBtn.textContent = '⏳ computing...';
    axesReadout.textContent = 'computing...';
  }
  requestAnimationFrame(async () => {
    let rootSpan: Span | null = null;
    try {
      rootSpan = appSpan('app.reconstruct', { kind: 'capture' });
      if (isPhysical(camera) && !camera.lastRealCaptureGray) {
        if (isActive) axesReadout.textContent = 'waiting for a real capture -- take a photo on the phone page';
        return;
      }
      const captureSpan = appSpan('app.capture');
      // rawGray is top-down now, from either source (pipeline/capture.ts's
      // captureDistortedGrayscale/ingestRealCapture both flip once at their
      // own source, see their own comments) -- top-down is this whole
      // pipeline's one dominant convention, so it feeds lastAxesCaptureGray
      // (-> computePoseFromCapture) directly, no flip needed. The ONE
      // remaining flip in this whole pipeline lives right here instead:
      // lastNoisedPreviewGray (and everything painted from it --
      // distortedPreviewData and the other flipY=false preview/overlay
      // textures in camera/factory.ts) is the one consumer that genuinely
      // needs bottom-up, so it gets flipped on the way OUT of the dominant
      // top-down convention, not the math on the way in.
      const { gray: rawGray, w, h } = isPhysical(camera)
        ? { gray: camera.lastRealCaptureGray!, w: camera.lastRealCaptureW, h: camera.lastRealCaptureH }
        : captureDistortedGrayscale(camera);
      camera.lastNoisedPreviewGray = flipRowsF64(rawGray, w, h);
      spanEnd(captureSpan);
      camera.lastAxesCaptureGray = { gray: rawGray, w, h };

      await recomputeStages(camera);
    } finally {
      spanEnd(rootSpan);
      if (isActive) {
        captureAxesBtn.disabled = false;
        captureAxesBtn.textContent = prevLabel;
      }
      camera.axesCapturing = false;
      // See PhysicalCamera.idleSpan's own comment -- this brackets exactly
      // the round trip ingestRealCapture is on the other end of closing.
      if (isPhysical(camera)) camera.idleSpan = appSpan('app.idle');
    }
  });
}

// Re-runs everything downstream of the capture stage (gradient field
// through pole markers/overlays), reusing whatever runAxesReconstruction
// last stored into camera.lastAxesCaptureGray instead of taking a fresh
// capture -- the recompute half of "every setting recomputes everything
// downstream of it" for settings that don't invalidate the capture itself
// (LSD tuning, gridPeriodPhaseGapLowerBound, minGrazingCos, forceCPU). Guarded by the same axesCapturing
// flag as runAxesReconstruction (reused, not a separate flag) so a rapid
// slider drag naturally self-throttles: a call that arrives while one is
// already in flight just no-ops, same as an overlapping "capture now"
// click today.
export function recomputeFromLastCapture(camera: Camera) {
  if (camera.axesCapturing) return;
  if (camera.visualsDraining) return; // see drainVisuals -- same self-throttle, one stage later
  if (!camera.lastAxesCaptureGray) return; // nothing captured yet
  camera.axesCapturing = true;
  const isActive = camera === activeCamera();
  const prevLabel = captureAxesBtn.textContent;
  if (isActive) {
    captureAxesBtn.disabled = true;
    captureAxesBtn.textContent = '⏳ computing...';
    axesReadout.textContent = 'computing...';
  }
  requestAnimationFrame(async () => {
    let rootSpan: Span | null = null;
    try {
      rootSpan = appSpan('app.reconstruct', { kind: 'recompute' });
      await recomputeStages(camera);
    } finally {
      spanEnd(rootSpan);
      if (isActive) {
        captureAxesBtn.disabled = false;
        captureAxesBtn.textContent = prevLabel;
      }
      camera.axesCapturing = false;
    }
  });
}
