import * as THREE from 'three';
import { type Camera, type CameraPose, type FrameMeta, type PhysicalCamera, type SimulatedCamera } from '../../shared/camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { toGrayscale } from '../../../pose/grayscale.ts';
import { renderer, scene } from '../scene/renderer.ts';
import { spanEnd } from '../../../profiling/profiler.ts';
import { appSpan } from '../../../profiling/stages.ts';
import { ingestLinkSpans } from '../../../profiling/clocks.ts';
import { globalState } from '../../shared/state.ts';
import { layoutPip } from '../ui/layout.ts';
import { applyPoseVisualizations, runAxesReconstruction } from './axesReconstruction.ts';
import { backendFromForceCPU } from '../../shared/backend.ts';
import { type RemotePoseMessage } from '../../shared/camera/model.ts';
import { buildProjectedTexture, computeProjectedBinsAuto, paintProjectedTexture } from './projectedBins.ts';
import { addGaussianNoise, applyAntialiasFilter, downsampleBoxAverage, flipRowsF64, separableBoxBlur } from './distortion.ts';
import { updateDistortedPreview } from './preview.ts';

// ── Per-camera capture/analysis pipeline ─────────────────────────────────

// Relocated to math/geometry.ts (see this session's on-device-pose-recovery
// plan) so decodeGrid.ts/pipeline/projectSamples.gpu.ts -- both on the
// critical path to a pose, needed to stay importable without a #gl canvas --
// don't have to import THIS file (which pulls in the scene/renderer.ts
// singleton) just for this one function. Re-exported here unchanged so
// every existing desktop import (main.ts, axesReconstruction.ts,
// overlays/recoveredOverlays.ts, overlays/contaminationOverlays.ts) keeps
// working without modification.
export { getAnalysisVFovRad } from '../../shared/math/geometry.ts';

export function markCaptureDirty(camera: Camera) {
  camera.captureDirty = true;
}

// Called once at camera creation and again whenever the viewportW/H/
// captureSupersample sliders change -- or, with an explicit override,
// whenever a real capture arrives at a different resolution than whatever's
// currently allocated (see ingestRealCapture).
export function resizeCaptureBuffers(camera: Camera, explicitSize?: { w: number; h: number }) {
  camera.captureDirty = true;
  camera.rtSize = explicitSize ?? { w: Math.round(camera.settings.viewportW), h: Math.round(camera.settings.viewportH) };
  camera.aspect = camera.rtSize.w / camera.rtSize.h;
  const { w, h } = camera.rtSize;
  // Same "stale buffer, wrong offsets" bug class as the overlay buffers
  // below -- a cached capture at the OLD size must never be fed to
  // recomputeFromLastCapture after a resize.
  camera.lastAxesCaptureGray = null;

  if (isSimulated(camera)) {
    camera.captureRTSize = { w: w * camera.settings.captureSupersample, h: h * camera.settings.captureSupersample };
    camera.camRT.setSize(camera.captureRTSize.w, camera.captureRTSize.h);
    camera.gizmoCam.aspect = camera.aspect;
    camera.gizmoCam.updateProjectionMatrix();
  }

  camera.distortedPreviewData = new Uint8Array(w * h * 4);
  camera.distortedPreviewTex.image = { data: camera.distortedPreviewData, width: w, height: h };
  // WebGL2 typically allocates a texture's GPU storage immutably on first
  // upload -- dispose() forces three.js to drop the old GL texture object so
  // the next upload allocates fresh storage at the new size.
  camera.distortedPreviewTex.dispose();
  camera.distortedPreviewTex.needsUpdate = true;

  // Initial size only -- see camera/factory.ts's own comment on
  // projectedPreviewData/Tex; paintProjectedTexture reallocates this per
  // capture to match its own square-cell bucket grid.
  camera.projectedPreviewData = new Uint8Array(w * h * 4);
  camera.projectedPreviewTex.image = { data: camera.projectedPreviewData, width: w, height: h };
  camera.projectedPreviewTex.dispose();
  camera.projectedPreviewTex.needsUpdate = true;

  camera.trueContamData = new Uint8Array(w * h * 4);
  camera.trueContamTex.image = { data: camera.trueContamData, width: w, height: h };
  camera.trueContamTex.dispose();
  camera.trueContamTex.needsUpdate = true;

  camera.reconContamData = new Uint8Array(w * h * 4);
  camera.reconContamTex.image = { data: camera.reconContamData, width: w, height: h };
  camera.reconContamTex.dispose();
  camera.reconContamTex.needsUpdate = true;


  // These overlay buffers (top-gradient, tangent-walk-path, LSD raw-regions/
  // rejected) were missing from this function entirely -- their Uint8Array
  // stayed allocated at whatever size they were created at while
  // camera.rtSize moved on, so painting into them read/wrote the OLD-sized
  // buffer at the wrong offsets -- the diagonal "streaking" artifact.
  // magFilter is reapplied since a fresh DataTexture doesn't carry it over
  // from the disposed one (see camera/factory.ts's own comment on why these
  // specifically need NearestFilter).
  camera.topGradientData = new Uint8Array(w * h * 4);
  camera.topGradientTex.image = { data: camera.topGradientData, width: w, height: h };
  camera.topGradientTex.dispose();
  camera.topGradientTex.needsUpdate = true;

  camera.lsdRawRegionsData = new Uint8Array(w * h * 4);
  camera.lsdRawRegionsTex.image = { data: camera.lsdRawRegionsData, width: w, height: h };
  camera.lsdRawRegionsTex.magFilter = THREE.NearestFilter;
  camera.lsdRawRegionsTex.dispose();
  camera.lsdRawRegionsTex.needsUpdate = true;

  camera.lsdRejectedData = new Uint8Array(w * h * 4);
  camera.lsdRejectedTex.image = { data: camera.lsdRejectedData, width: w, height: h };
  camera.lsdRejectedTex.magFilter = THREE.NearestFilter;
  camera.lsdRejectedTex.dispose();
  camera.lsdRejectedTex.needsUpdate = true;

  if (camera === activeCamera()) layoutPip(camera);
}

// Renders gizmoCam's view into camRT -- pulled out into its own function so
// the real analysis path (captureDistortedGrayscale) can always force a
// truly fresh capture regardless of the passive preview's dirty/throttle
// gating in animate().
export function renderCamRT(camera: SimulatedCamera) {
  const dpr = renderer.getPixelRatio();
  const prevRT = renderer.getRenderTarget();
  renderer.setRenderTarget(camera.camRT);
  renderer.setViewport(0, 0, camera.captureRTSize.w / dpr, camera.captureRTSize.h / dpr);
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.render(scene, camera.gizmoCam);
  renderer.setRenderTarget(prevRT);
}

// Render+blur happen at captureSupersample x rtSize, THEN get box-downsampled
// to rtSize -- see pre-Stage-A history for why (physical lens blur acts on a
// near-continuous image; only the sensor's final discretization should
// introduce the pixel grid). GL's readback is natively bottom-up, but this
// function's own contract is top-down -- the one dominant row convention the
// reconstruction math (pose/stages/votes/votes.ts's toNDC) and every capture source
// (this, ingestRealCapture, ingestRemotePose, client/main.ts's on-device
// pipeline) now share, so nothing downstream of a capture needs to know or
// care which GPU/decode path produced it. flipRowsF64 (its own exact
// inverse) is the one place that native bottom-up-ness gets converted, right
// here at the source -- see runAxesReconstruction's own comment for where
// the ONE remaining flip in this whole pipeline lives (restoring bottom-up
// only for the flipY=false preview/overlay textures that actually need it).
export function captureDistortedGrayscale(camera: SimulatedCamera): { gray: Float64Array; w: number; h: number } {
  renderCamRT(camera);
  const { w: cw, h: ch } = camera.captureRTSize;
  const raw = new Uint8Array(cw * ch * 4);
  renderer.readRenderTargetPixels(camera.camRT, 0, 0, cw, ch, raw);
  const hiResGray = toGrayscale(raw, cw, ch);
  const antialiased = applyAntialiasFilter(hiResGray, cw, ch, camera.settings.captureSupersample);
  const hiResBlurred = separableBoxBlur(antialiased, cw, ch, Math.round(camera.settings.simBlur * camera.settings.captureSupersample));
  const downsampled = downsampleBoxAverage(hiResBlurred, cw, ch, camera.settings.captureSupersample, camera.rtSize.w, camera.rtSize.h);
  addGaussianNoise(downsampled, camera.settings.simNoise);
  const gray = flipRowsF64(downsampled, camera.rtSize.w, camera.rtSize.h);
  return { gray, w: camera.rtSize.w, h: camera.rtSize.h };
}

// Decodes an incoming JPEG blob at whatever resolution it actually arrived
// at -- no resampling of the image itself; resizeCaptureBuffers instead
// reallocates the analysis buffers to MATCH it when it differs from what's
// currently allocated, so the phone's own resolution dropdown (see
// client/camera.ts) is the real analysis resolution end-to-end, not a
// cap on top of a further downsample. Converts to grayscale and stores it
// top-down, matching canvas 2D getImageData's own native order -- no flip
// needed here, since top-down is this whole pipeline's one dominant
// convention now (see captureDistortedGrayscale's own comment). `pending` is
// whatever main.ts's mailbox pump just popped -- see PhysicalCamera's own
// comment on idleSpan, and profiling/clocks.ts for the `peer` spans the
// phone-side stamps become -- which tell "pull the video frame" apart from
// "JPEG encode" instead of lumping them together, and which deliberately do
// NOT include a transit span.
//
// pending.blob arrives as a genuine binary WebSocket frame now (devBridge/
// client.ts), not a base64 data: URL -- decoded via a blob: object URL
// instead, but through the exact same img.src -> drawImage -> getImageData
// -> toGrayscale pipeline as before. Both URL schemes decode through the
// browser's own native JPEG decoder identically; only the WIRE encoding
// changed, not the pixel data or its orientation.
export async function ingestRealCapture(
  camera: PhysicalCamera,
  pending: {
    blob: Blob; sentAt: number; pulledAt: number; encodedAt: number; receivedAt: number; bytes: number;
    drawnAt?: number; frameMeta?: FrameMeta | null;
  },
): Promise<void> {
  if (camera.idleSpan) { spanEnd(camera.idleSpan); camera.idleSpan = null; }
  // The phone-side half of this frame's journey, as `peer` spans in the one
  // record store. It used to be three subtractions into three `lastXMs` fields
  // plus four rolling arrays -- a second way to measure a duration, with one
  // reader, which is now gone. Only two spans come out of three subtractions:
  // clocks.ts explains why "transit" was never a duration.
  ingestLinkSpans(pending);
  // pendingCapture is a MAILBOX -- main.ts pops it and it is gone. Everything
  // above survives only as a derived duration, which is exactly what the IMU
  // work cannot use: a filter needs the ABSOLUTE time this frame was
  // captured, on the phone's own clock, to line the measurement up against
  // IMU samples stamped by the same clock. So the raw phone-side stamps are
  // retained here rather than recomputed later, because later they no longer
  // exist.
  //
  // ON THE CLOCK: `receivedAt (DESKTOP clock) - encodedAt (PHONE clock)`
  // measures ~-38ms on this pair of machines -- a NEGATIVE transit time, i.e.
  // cross-device clock SKEW rather than network time. 38ms is a third of the
  // whole prediction horizon this project is trying to cover. That is why no
  // `link.transit` span exists; see profiling/clocks.ts.
  // Everything stored below is phone-clock-only and therefore mutually
  // consistent with imuSamples; nothing here may be compared against a
  // desktop timestamp without solving for that offset first.
  camera.lastCaptureTiming = {
    sentAt: pending.sentAt, pulledAt: pending.pulledAt, encodedAt: pending.encodedAt,
    drawnAt: pending.drawnAt ?? null, frameMeta: pending.frameMeta ?? null,
    receivedAtDesktop: pending.receivedAt,
  };
  const rootSpan = appSpan('ingest.run', { bytes: pending.bytes });

  const decodeSpan = appSpan('ingest.decode');
  const img = new Image();
  const objectUrl = URL.createObjectURL(pending.blob);
  try {
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('failed to decode incoming capture image'));
      img.src = objectUrl;
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
  spanEnd(decodeSpan);
  const w = img.naturalWidth, h = img.naturalHeight;
  if (w !== camera.rtSize.w || h !== camera.rtSize.h) resizeCaptureBuffers(camera, { w, h });

  const readbackSpan = appSpan('ingest.readback');
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = w; tmpCanvas.height = h;
  const tctx = tmpCanvas.getContext('2d')!;
  tctx.drawImage(img, 0, 0);
  const topDown = tctx.getImageData(0, 0, w, h).data;
  camera.lastRealCaptureGray = toGrayscale(topDown, w, h);
  camera.lastRealCaptureW = w; camera.lastRealCaptureH = h;
  spanEnd(readbackSpan);
  spanEnd(rootSpan);

  updateDistortedPreview(camera);
  if (globalState.mode === 'projected' && camera === activeCamera()) buildProjectedTexture(camera, backendFromForceCPU(globalState.forceCPU));
  runAxesReconstruction(camera);
}

// Ingests an already-computed pose from a phone in device-compute mode (see
// this session's on-device-pose-recovery plan and client/main.ts's
// computePoseFromCapture call) -- deserializes the plain-array-serialized
// Vector3/Quaternion fields back into THREE objects and assigns directly
// into ONE camera.pose, published in a single assignment, no pipeline call at all
// (the phone already did that work). `debug`/`imageBytes` are both optional,
// riding along only when the phone's own sendDebugInfo/sendCapturedImage
// toggles are on (both default off) -- see client/main.ts's own comments;
// this is the "richer optional payload" extensibility point the original
// plan called out (debug intermediates/the actual capture for projection),
// added on request for live diagnosis of a suspect on-device decode rather
// than left for a hypothetical future need. Called exclusively from
// main.ts's animate-loop mailbox pump now (camera.pendingPoseResult), the
// same way ingestRealCapture trusts its own pendingCapture pump -- no
// internal busy-guard needed here, the pump already checked
// axesCapturing/captureIngestBusy before calling in.

export async function ingestRemotePose(
  camera: PhysicalCamera,
  msg: RemotePoseMessage,
): Promise<void> {
  if (msg.w !== camera.rtSize.w || msg.h !== camera.rtSize.h) resizeCaptureBuffers(camera, { w: msg.w, h: msg.h });

  // A pose from the phone REPLACES whatever a local reconstruction was in the
  // middle of showing, so intermediates still parked from that local capture
  // are stale by definition -- resolving them later would paint this frame's
  // remote pose against the previous local frame's decode. Dropped rather than
  // read. Dropping the whole mailbox slot is what makes that true for the POSE
  // as well: a tail that ran afterwards would repaint the local pose over this
  // one. There is no release beside it any more -- the arena frees the local
  // run's slices at the next reconstruction, so dropping the pointer is all of
  // it.
  camera.pendingVisuals = null;

  const recoveredAxes = msg.recoveredAxes ? {
    Drow: new THREE.Vector3().fromArray(msg.recoveredAxes.Drow),
    Dcol: new THREE.Vector3().fromArray(msg.recoveredAxes.Dcol),
    Dnormal: new THREE.Vector3().fromArray(msg.recoveredAxes.Dnormal),
    distance: msg.recoveredAxes.distance,
  } : null;
  // Tier 1 of this session's "Ship auxiliary pipeline intermediates" plan: the
  // phone's own pipeline intermediates, landing on the exact same pose fields a
  // desktop-compute capture already populates -- overlays/
  // gridPeriodPhaseOverlays.ts and the composite-line-family hover-debug
  // overlay read gridPeriodPhase/voteComposites and need no changes to work
  // from device-compute data.
  //
  // The phone's lsdRectangles are NOT unpacked. They used to be assigned to a
  // lastLsdRectangles field whose own comment admitted nothing rendered from it
  // -- the overlay always recomputed its own chain instead. Now that the
  // overlay draws from the pose's own rects (the LOCAL run's, with real
  // members), backfilling a members-less remote copy would put a second,
  // emptier source of the same thing back on the camera for nothing to read.
  //
  // Null whenever msg.debug.pipeline is absent (sendDebugInfo off, or an older
  // phone build still connected) -- and that is now automatic rather than five
  // careful clears, which is the point of building ONE object: there is no
  // field left over from the previous frame to forget to overwrite.
  const pipelineDebug = msg.debug?.pipeline;
  // THE REMOTE POSE IS A WHOLE POSE OR IT IS NOTHING. Assembled here and
  // published in one assignment, exactly like a local run's (see
  // axesReconstruction.ts's applyPoseResult).
  //
  // A REMOTE POSE IS MOSTLY ABSENCES, and the flat CameraPose now says so by
  // omission rather than by sentinel. What the phone sends is what it
  // RECOVERED -- an axis frame and a decoded position. It sends no detector
  // report (so no `lineCount`, no `regionCount`, no vote normals), no §15
  // `status` (there was no local run to have one), and none of the opt-in
  // display buffers, which are device memory on a machine that is not this one.
  //
  // Every one of those used to be a written field: `-1` for the counts, `[]` for
  // the arrays, `null` for the rest. They existed because the type demanded a
  // value, and each one had to argue in a comment for why its particular
  // placeholder did not read as an answer. Leaving the key out says it once.
  const pose: CameraPose = {
    ...(recoveredAxes ? { recoveredAxes } : {}),
    ...(msg.positionDecode
      ? {
        positionDecode: {
          row: msg.positionDecode.row, col: msg.positionDecode.col,
          consistency: msg.positionDecode.consistency,
          camPos: new THREE.Vector3().fromArray(msg.positionDecode.camPos),
          recoveredCamQuat: new THREE.Quaternion().fromArray(msg.positionDecode.recoveredCamQuat),
          orientation: msg.positionDecode.orientation,
        },
      }
      : {}),
    // The phone's composite lines are the one display array that DOES cross the
    // wire, inside the optional debug blob.
    // `index` is the array position, which is the best available answer: the
    // wire carries a list, not the pipeline's own line slots. It is only ever
    // used to look up `rectified`/`lineFamily`, and a remote pose has neither,
    // so nothing can join against a wrong one. `c.root` is dropped with the
    // `region` field -- see CameraPose.composites.
    ...(pipelineDebug?.voteComposites
      ? { composites: pipelineDebug.voteComposites.map((c, i) => ({ index: i, line: c.line })) }
      : {}),
  };
  camera.pose = pose;

  // Diagnostic-only debug summary -- null whenever sendDebugInfo is off,
  // and null in PRACTICE either way right now: the phone's buildDebugPayload
  // went with the deleted pipeline (see client/main.ts's note where it lived),
  // so nothing attaches one. Kept because the field is still the extensibility
  // point. Also clears a prior
  // frame's leftover value if the toggle just got switched off, same
  // "don't show stale data" principle as everything else in this function.
  camera.lastRemoteDebug = msg.debug ?? null;

  // No real image reaches the desktop in device-compute mode UNLESS the
  // phone's sendCapturedImage toggle is on (see client/main.ts) -- without
  // it, clear every real-pixel-derived buffer this cycle's pose data can't
  // refresh, rather than leaving stale content from a PRIOR capture on this
  // same camera visible (World-view fill, Through-Cam raw preview, a
  // recomputeFromLastCapture slider drag -- see overlays/recoveredOverlays.ts's
  // applyRecoveredFloorOverlay, pipeline/preview.ts's updateDistortedPreview,
  // and axesReconstruction.ts's own comment on this being an accepted
  // consequence of the architecture otherwise). With it: decode exactly like
  // ingestRealCapture does, so Through-Cam's raw preview becomes inspectable
  // -- purely for visual debugging, the phone already computed the pose
  // itself, this does NOT re-run the reconstruction pipeline.
  if (msg.imageBytes) {
    const img = new Image();
    // Raw bytes now (poseResultWire.ts), not a base64 data: URL -- decoded
    // via a blob: object URL instead, same pattern ingestRealCapture already
    // uses for its own JPEG blob, so it needs the same revoke-when-done care
    // (a data: URL never needed this).
    const objectUrl = URL.createObjectURL(new Blob([msg.imageBytes], { type: 'image/jpeg' }));
    try {
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('failed to decode incoming debug image'));
        img.src = objectUrl;
      });
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
    const w = img.naturalWidth, h = img.naturalHeight;
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = w; tmpCanvas.height = h;
    const tctx = tmpCanvas.getContext('2d')!;
    tctx.drawImage(img, 0, 0);
    const topDown = tctx.getImageData(0, 0, w, h).data;
    camera.lastRealCaptureGray = toGrayscale(topDown, w, h);
    camera.lastRealCaptureW = w; camera.lastRealCaptureH = h;

    // Unlocks Projected-Cam view and the recovered-floor image fill for a
    // device-compute camera -- same two calls recomputeStages
    // (axesReconstruction.ts) makes for a desktop-compute capture, run here
    // against the phone's own sent image + its own already-recovered axes.
    // Strictly inside this `if (msg.imageBytes)` branch (see this session's
    // "Ship auxiliary pipeline intermediates" plan): with sendCapturedImage
    // off, msg.imageBytes is undefined and none of this runs at all.
    if (pose.recoveredAxes) {
      const projResult = await computeProjectedBinsAuto(camera, backendFromForceCPU(globalState.forceCPU));
      // Painting the texture is a real GPU upload -- same showProjected
      // gating recomputeStages uses, so it's skipped for a camera that
      // isn't actually being viewed in Projected-Cam/World-with-floor mode.
      const showProjected = (camera === activeCamera() && globalState.mode === 'projected')
        || (globalState.mode === 'world' && camera.settings.showRecoveredFloor);
      if (showProjected) paintProjectedTexture(camera, projResult);
    } else {
      camera.lastProjectedBins = null;
    }
  } else {
    camera.lastRealCaptureGray = null;
    camera.lastProjectedBins = null;
  }
  camera.lastAxesCaptureGray = null;
  updateDistortedPreview(camera);

  applyPoseVisualizations(camera, camera === activeCamera());
}
