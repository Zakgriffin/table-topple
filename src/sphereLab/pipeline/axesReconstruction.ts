import * as THREE from 'three';
import { type Camera, type CameraPose, type PendingVisuals } from '../camera/model.ts';
import { activeCamera, isPhysical, isSimulated } from '../camera/store.ts';
import { COL_DIR, ROW_DIR, SPHERE_RADIUS } from '../constants.ts';
import { angleBetweenDegV } from '../math/geometry.ts';
import { updatePositionReadoutText } from '../overlays/projectedCamOverlays.ts';
import { applyRecoveredFloorOverlay, updateRecoveredCamGizmo, updateRecoveredFloorOutline } from '../overlays/recoveredOverlays.ts';
import { updateGradientCirclesDebug } from '../overlays/sphereOverlays.ts';
import { globalState } from '../state.ts';
import { axesReadout, captureAxesBtn, lsdChainTransfers } from '../ui/dom.ts';
import { backendFromForceCPU } from '../../pose/backend.ts';
import { captureDistortedGrayscale } from './capture.ts';
import { computeProjectedBinsAuto, paintProjectedTexture, type ProjectedSampleResult } from './projectedBins.ts';
import { flipRowsF64 } from './distortion.ts';
import { refreshModeVisualizations } from './modeRefresh.ts';
import { readSliceF32, readSliceU32 } from '../../pose/gpu/device.ts';
import { readRegionMembers } from '../../pose/stages/lsd/lsdFit.gpu.ts';
import type { GrownRegion } from '../../pose/stages/lsd/types.ts';
import type { Intermediates } from '../camera/model.ts';
import { computePoseFromCapture, type PoseResult } from '../../pose/poseCompute.ts';
import { type StageRecord, spanEnd } from '../profiling/profiler.ts';
import { poseSpan } from '../../pose/timing/stages.ts';
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
export function applyPoseVisualizations(camera: Camera, isActive: boolean, extraReadoutLine?: string) {
  let orientationErrorLine: string | null = null;
  const pose = camera.pose;
  // Pole markers read the pose's quadricPair (its own field specifically so
  // this survives even when gridPeriodPhase fails and recoveredAxes ends up
  // null -- see pose/poseCompute.ts's PoseResult) instead of locally
  // recomputing rowDirRecovered/colDirRecovered.
  const rowDirRecovered = pose?.quadricPair?.Drow ?? null;
  const colDirRecovered = pose?.quadricPair?.Dcol ?? null;
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
    // Zero for a device-compute pose, and honestly so: the phone sends its
    // recovered axes, not the vote vectors they were fit from. This used to
    // read a camera field the remote path never wrote, so it showed whatever
    // the last LOCAL reconstruction on this camera had counted.
    const voteCount = pose?.votes.length ?? 0;
    const lines = [`${voteCount} votes  (${voteCount} fed to fit)`];
    if (rowDirRecovered && colDirRecovered) {
      if (orientationErrorLine) lines.push(orientationErrorLine);
    } else {
      lines.push(`degenerate fit`);
    }
    const gpp = pose?.gridPeriodPhase;
    if (pose?.recoveredAxes && gpp) {
      const trueDist = isSimulated(camera) ? camera.camPos.y : NaN;
      const dist = pose.recoveredAxes.distance;
      if (haveGroundTruth) {
        const err = (Math.abs(dist - trueDist) / trueDist) * 100;
        lines.push(`distance ${dist.toFixed(2)} (${err.toFixed(1)}% err)  true ${trueDist.toFixed(2)}  period ${gpp.period.toFixed(4)}  [gridPeriodPhase]`);
      } else {
        lines.push(`distance ${dist.toFixed(2)}  period ${gpp.period.toFixed(4)}  [gridPeriodPhase]`);
      }
    } else if (pose?.quadricPair) {
      lines.push(`distance: no period found (gridPeriodPhase)`);
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
  const s = camera.pose?.chainTransfers;
  if (!s) {
    lsdChainTransfers.textContent = 'LSD chain: no capture yet.';
    return;
  }
  const mb = s.bytes / 1048576;
  const size = mb >= 0.01 ? `${mb.toFixed(2)} MB` : `${s.bytes} B`;
  const head = `LSD chain: ${s.crossings} crossing${s.crossings === 1 ? '' : 's'}, ${size}`;
  if (s.entries.length === 0) {
    // Genuinely zero, not missing: an all-CPU chain never touches the bus.
    lsdChainTransfers.textContent = `${head} -- nothing crosses (all-CPU chain).`;
    return;
  }
  const detail = s.entries
    .map((e) => `${e.what}${e.direction === 'up' ? '↑' : '↓'}`)
    .join(' ');
  lsdChainTransfers.textContent = `${head}  [${detail}]`;
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

  // FIRST, before anything reads the pose's intermediates. The pose itself was
  // published the moment it was final; its per-pixel fields and its decode grid
  // were left on the device (see pose/intermediates.ts) so the pose did not have
  // to wait 0.45MB for display data. This is the moment they land, and the
  // second half of applyPoseResult's seam: the same pose object, republished
  // with `intermediates` filled in. updateGradientCirclesDebug,
  // applyPoseVisualizations and every mode overlay below are downstream of it.
  //
  // This is the ONE thing in this function that is not idempotent-by-reading-
  // settled-state: it consumes a handle. resolve() is itself idempotent -- a
  // second call re-hands the same object rather than draining twice -- so a
  // second drain over the same payload paints the same thing.
  //
  // THE IDENTITY CHECK IS THE POINT OF THE PAYLOAD. Publishing is the only
  // thing here that can write over a newer pose, so it happens only while the
  // pose this drain was handed is still the one on screen. Unreachable today
  // (drainVisuals and the two reconstruction entry points exclude each other),
  // and that exclusion is exactly what this makes droppable: without it, a
  // reconstruction that landed while the readback was in flight would otherwise
  // get its axes overwritten by the previous frame's, decode grid and all.
  {
    const got = await readDisplayIntermediates(camera, posted.result);
    if (camera.pose === posted.pose) camera.pose = { ...posted.pose, intermediates: got };
  }

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
  const projectStart = performance.now();
  // Captured outside the `if` (stays null when there's no recovered axes to
  // project) so it can be handed to refreshModeVisualizations below instead
  // of that call recomputing the exact same (possibly GPU) result a second
  // time -- see modeRefresh.ts's own comment on precomputedProjection.
  let projResult: ProjectedSampleResult = null;
  if (camera.pose?.recoveredAxes) {
    projResult = await computeProjectedBinsAuto(camera, backendFromForceCPU(globalState.forceCPU));
    if (showProjected) paintProjectedTexture(camera, projResult);
  }
  const projectMs = performance.now() - projectStart;
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
  const t = posted.pose.timing;
  let timingLine: string | undefined;
  if (t) {
    timingLine = `votes ${t.votesMs.toFixed(0)}ms  fit ${t.fitMs.toFixed(0)}ms  pose ${t.poseMs.toFixed(0)}ms  distance ${t.distanceMs.toFixed(0)}ms  project ${projectMs.toFixed(0)}ms  decode ${t.decodeMs.toFixed(0)}ms`;
  }
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
// What the DISPLAY half needs handed back from this run, declared here because
// this is the only caller that has a drain to resolve it in and the only one
// that knows which overlays are on.
//
// 'decodeGrid' is unconditional: the decodeGrid/decodeRotated/decodeCorrectness
// intermediates feed the Projected-Cam view and the pose readout, and the drain
// always runs, so there is nothing to gate on.
//
// fx/fy are asked for only when something draws them. THIS FUNCTION IS THE
// OTHER HALF OF A CONTRACT: overlays/pipelineField.ts returns null when they
// were not requested and its consumers then draw nothing, so a toggle listed
// in one place and not the other fails as a blank overlay rather than as a
// silent recomputation. The list mirrors preview.ts's overlaysNeedGray, which
// exists for the same class of mistake one stage earlier -- if you add a
// toggle whose overlay reads pipelineField(), it belongs in BOTH.
//
// Every toggle here also has to invalidate: turning an overlay on cannot paint
// from a run that was never asked to produce its input, so the handlers call
// recomputeFromLastCapture rather than the overlay's own updater. That is a
// real cost change -- flipping one of these re-runs the pose stages instead of
// recomputing one field on the CPU -- and it is the trade the whole step is
// making: one pipeline, asked, instead of a second one, duplicated.
async function readDisplayIntermediates(camera: Camera, result: PoseResult): Promise<Intermediates> {
  const s = camera.settings;
  const out: Intermediates = {};
  // `pose.drain` is opened HERE now, by the app, and that is the honest place
  // for it: the drain was never a pipeline stage, it is the display half
  // deciding to pay for a readback. The library still names it as the OWNER of
  // those transfers, so the crossings join inside this span exactly as before.
  const drainSpan = poseSpan('pose.drain');
  try {

    // fx/fy: the contamination rasters, the top-gradient raster, the hover
    // arrows, and the LSD edge-connectivity preview.
    const needField = s.showTrueContamination || s.showReconstructedContamination
      || s.showTopGradient
      || s.showGradientArrow || s.showLevelLineArrow
      || s.showLsdRawRegions;
    // The three LSD debug views. ALL THREE need the region member lists now,
    // not just the raw-region one: a rectangle no longer carries its own
    // members, so the rejected raster and the segment hues join `rects[i]` to
    // `regions[i]` (see LsdRectangle). That is why these two conditions are the
    // same condition, and updateLsdOverlay requires both arrays together.
    //
    // `regionId` stays gated on the raw-region view alone -- it is a separate
    // n*4 readback and only the hover highlight looks up a pixel's region.
    const needRects = s.showLsdSegments || s.showLsdRejected || s.showLsdRawRegions;
    const needRegions = needRects;
    const needRegionId = s.showLsdRawRegions;

    // ── the reads, written out ──
    //
    // This used to be `displayIntermediates(camera)` returning a SET OF NAMES
    // that went into the pipeline as `want`, and a `pending.resolve()` here that
    // drained whatever had been named. Two halves that had to agree, several call
    // sites away from each other, plus a handle carrying ownership of device
    // buffers between them.
    //
    // Now the settings decide what to read and the reads happen here. The
    // conditions are the same conditions; what is gone is the mechanism that used
    // to carry them across the boundary -- and with it the question of who frees
    // what, since the pipeline's arena frees all of it at the next run.
    const { chain, cpuChain } = result;
    if (cpuChain) {
      // Already host arrays: no readback, no widening, no transfer. Handing them
      // over is a pointer copy, which is why the CPU path pays literally nothing
      // for display data.
      if (needField) { out.fx = cpuChain.fx; out.fy = cpuChain.fy; }
      if (needRegionId) out.regionId = cpuChain.regionId;
      if (needRegions) out.regions = cpuChain.regions as GrownRegion[];
    } else if (chain) {
      const { arena, n } = chain;
      if (needField) {
        out.fx = new Float64Array(await readSliceF32(arena, chain.fx, n * 4, 'fx', 'pose.drain'));
        out.fy = new Float64Array(await readSliceF32(arena, chain.fy, n * 4, 'fy', 'pose.drain'));
      }
      if (needRegionId) {
        const raw = await readSliceU32(arena, chain.regionId, n * 4, 'regionId', 'pose.drain');
        // BIT REINTERPRETATION, not a numeric convert -- that is what makes the
        // -1 sentinel survive the trip (collect's scatter writes a real -1).
        out.regionId = new Int32Array(raw.buffer.slice(0));
      }
      if (needRegions) {
        out.regions = (await readRegionMembers(
          arena, chain.regions, chain.regionCount, chain.memberCount, 'pose.drain',
        )) as GrownRegion[];
      }
    }
    // Host-side on both backends, so this is never a read.
    if (needRects) out.rects = result.rects;

    // The decode grid, 0.45MB. Projected-Cam and the world-floor decal are the
    // only things that draw it, but it is read unconditionally for now because
    // several readouts consume `decodeRotated` outside those views; narrowing it
    // is a separate change with its own display consequences.
    if (result.readGrid) {
      const g = await result.readGrid();
      out.decodeGrid = g.grid;
      out.decodeRotated = g.rotated;
      out.decodeCorrectness = g.correctness;
    }
    return out;
  } finally {
    spanEnd(drainSpan);
  }
}

// ── The one place a pose result lands on a Camera ─────────────────────────
//
// This is the seam, and it is now one assignment. The pipeline returns a
// PoseResult; the app publishes it as camera.pose, the pose that is on screen
// -- a statement about the app's own bookkeeping the library should have no
// opinion on. What crosses here is the DIFFERENCE between those two ideas (see
// camera/model.ts's CameraPose): the undrained handle stays behind in the
// mailbox, `intermediates` starts empty, and `timing` widens to nullable
// because the remote path has none.
//
// It used to unpack the result into thirteen separate mutations, which was the
// staleness hazard's actual home: an atomically swappable object and an
// exploded copy of it spread across a camera are not the same thing. There is
// no halfway state to observe now because there is nothing to be halfway
// through.
//
// The intermediates are NOT filled here: they may still be on the device, and
// they arrive with the drain, which republishes this same object with them in
// it. Starting empty is the point -- the previous run's decode grid can no
// longer be read next to this run's axes, because it is not in this object.
function applyPoseResult(camera: Camera, result: PoseResult): void {
  const { chain, cpuChain, readGrid, ...rest } = result;
  void chain; void cpuChain; void readGrid;
  const pose: CameraPose = { ...rest, intermediates: {} };
  camera.pose = pose;
  // Posts to the mailbox. The pose is final; the tail is what is deferred. The
  // RESULT rides along rather than the pose, because it is what holds the arena
  // slices -- and those expire at the next reconstruction, which is exactly why
  // they are not on the pose that stays on screen.
  camera.pendingVisuals = { pose, result };
}

async function recomputeStages(camera: Camera) {
  const { gray, w, h } = camera.lastAxesCaptureGray!;

  // Every stage from the 2x2-gradient composite lines through
  // runPositionDecode lives in pose/poseCompute.ts's computePoseFromCapture --
  // a pure function of a capture plus the two camera parameters a real Camera
  // structurally satisfies (PoseInput), so it can also run standalone on the
  // phone and headless in node. computeProjectedBinsAuto/paintProjectedTexture
  // (in runVisualTail) are deliberately NOT part of that shared prefix --
  // confirmed not on the critical path to a pose (distance is already
  // finalized by gridPeriodPhase before that stage would run); they exist
  // only to feed Projected-Cam/World-floor-decal DISPLAY.
  // Dropped, not released. A payload this camera never drained (a newer
  // reconstruction superseded it before runVisualTail ran) used to own the
  // chain's device buffers and would hold them until device loss, so it had to
  // be released here, before the next run, so two sets never coexisted. The
  // arena frees the previous run's slices itself at the top of the next one, so
  // all that is left is to stop pointing at them.
  camera.pendingVisuals = null;

  // Nothing is asked for up front any more -- see computePoseFromCapture. The
  // last thing that was (the region CSR, for each rectangle's member list) is
  // read in the drain below like everything else, ONCE, now that a rectangle
  // joins to its region by index instead of carrying a copy.
  const result = await computePoseFromCapture(
    camera, gray, w, h, backendFromForceCPU(globalState.forceCPU),
  );
  // Publishes the pose AND posts the tail's payload -- the pose is final here,
  // and everything past this point is display. Deferred, that display work
  // stops being awaited inside the reconstruction's own window, where it
  // serialized ~20ms of GPU work against the same device queue the pose stages
  // were using.
  applyPoseResult(camera, result);
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
    let rootSpan: StageRecord | null = null;
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
    let rootSpan: StageRecord | null = null;
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
