// Great Sphere Lab — a visual testbed, not part of the tracking pipeline.
//
// Point of the exercise: a straight floor line, together with a camera's
// optical center, spans a plane through that center. Intersect that plane
// with the unit sphere centered on the camera and you get a great circle.
// A family of parallel floor lines all share one direction, and that shared
// direction is exactly what a vanishing point *is* — so every great circle
// from one family passes through the same antipodal point pair on the
// sphere. That crossing point pair is the vanishing point, made literally
// visible instead of algebraic. The floor grid here is two orthogonal
// families (world X and world Z), so their two pole pairs sit exactly 90°
// apart on the sphere, always — that's the "orthogonal constraint" the real
// pipeline (src/orthogonalVp.ts) searches for, seen from the inside.
//
// ── N-camera architecture note ────────────────────────────────────────────
// This app models N cameras -- any number of simulated ones (created/
// destroyed by hand, "+" in the tab bar) plus any number of physical ones
// (each auto-appearing the moment a phone connects through the dev bridge,
// see devBridge/client.ts) -- instead of one hardcoded camera. Every mutable
// THREE object/buffer that used to be a single module-level binding lives on
// a `Camera` object (camera/model.ts); truly shared things (the scene,
// renderer, floor, De Bruijn pattern, the world-view orbit controls,
// ROW_DIR/COL_DIR, MATH_QUAT) stay module-level. There is deliberately NO
// default camera: `cameras` starts empty and the tab bar starts on its
// always-present "Global" tab (activeCameraId === '' means exactly that --
// no camera selected, only global settings shown, see refreshCameraPanel).
// `activeCamera()` is whichever camera's detail panel (sliders/readouts/
// Through-Cam/Projected-Cam/Inside-Sphere) is currently shown; the cheap
// per-frame gizmo/overlay update loop below runs for every camera in
// `cameras` regardless of which is active (that's what makes the World view
// show all of them at once), while the expensive preview-render/auto-capture
// work only ever runs for the active one, since Through-Cam/Projected-Cam/
// Inside-Sphere/the PIP preview only ever show one camera at a time.
//
// A physical camera's type is fixed for its entire life: it's created only
// by a real phone connecting (see devBridge/client.ts's realCapture handler)
// and destroyed only by that connection closing (kicked from the tab bar, or
// a natural disconnect -- both funnel through the same server-side close
// handler, see scripts/dev-bridge/server.js). There is no UI path to
// manually conjure a physical camera or flip an existing camera's type --
// camera "type" answers a factual question (is this backed by a real phone
// or not), not a togglable setting.
//
// ── Directory layout ───────────────────────────────────────────────────────
//
// THE POSE LIBRARY LIVES OUTSIDE THIS APP, at src/pose/. Sphere Lab is a
// CONSUMER of it, not its owner -- src/mobileCapture.ts (the phone) is the
// other one, and the headless tests are a third. Its entry point is
// pose/poseCompute.ts, it is organized by pipeline STAGE (stages/gradient,
// stages/lsd, stages/votes, stages/period, stages/decode, plus gpu/ for the
// device plumbing), and it imports nothing from sphereLab/ except the shared
// leaves below. That constraint is checked, not hoped for: see
// tests/libraryBoundary.test.ts.
//
// types.ts/state.ts/constants.ts   shared types + tiny bits of module state
// math/geometry.ts                 pure sphere/ray-casting math
// profiling/profiler.ts            the one host clock
//    ^ the four above are the SHARED LEAVES: both this app and src/pose/
//      depend on them, and the board game depends on constants.ts too, which
//      is why they did not move into the library.
// camera/                          the Camera data model: settings, types,
//                                   the live store, factories, add/remove/kick
// scene/                           THREE.js scene setup shared by every
//                                   camera: renderer, floor+pattern, quad
//                                   blitters, world/inside-sphere controls
// pipeline/                        what is left after the library moved out:
//                                   capture, preview, the reconstruction
//                                   DRIVER (axesReconstruction), mode refresh,
//                                   and the display-only projection stage
//                                   (projectedBins + projectSamples.gpu)
// harness/                         the verifies + timing, detached in step 3
// overlays/                        paints pipeline results into the 3D scene
//                                   or 2D debug canvases
// ui/                              DOM wiring: controls, camera panel, mode
//                                   switching, viewport layout
// devBridge/                       websocket relay to scripts/dev-bridge/
//
// This file just ties the above together: the per-frame animate() loop and
// the one-time boot sequence.

import * as THREE from 'three';
import { activeCamera, cameras, isSimulated, isPhysical } from './camera/store.ts';
import { config } from './config.ts';
import { globalState } from './state.ts';
import { euler } from './constants.ts';
import { canvas, readout } from './ui/dom.ts';
import { setMode, setPanelCollapsed } from './ui/mode.ts';
import { renderCameraTabs, refreshCameraPanel } from './ui/cameraPanel.ts';
import { renderViewport, layoutPip, resize, computeThroughRect } from './ui/layout.ts';
import './ui/cameraPanel.ts'; // side effect: wires every slider/checkbox to the active camera
import { renderer } from './scene/renderer.ts';
import { floorMesh } from './scene/floor.ts';
import { viewerCam, worldOrbit, insideCam, insideYaw, insidePitch } from './scene/viewerControls.ts';
import { renderPreviewViewport, renderProjectedViewport } from './scene/quadRenderers.ts';
import {
  resizeThroughCamCanvas, drawThroughCamPreview, drawThroughCamTrueContam, drawThroughCamReconContam,
  drawThroughCamTopGradient, drawThroughCamLsdRawRegions, drawThroughCamLsdRejected,
} from './scene/throughCam2D.ts';
import { getAnalysisVFovRad, ingestRealCapture, ingestRemotePose, markCaptureDirty, resizeCaptureBuffers, renderCamRT } from './pipeline/capture.ts';
import { updateDistortedPreview, PREVIEW_UPDATE_INTERVAL_MS } from './pipeline/preview.ts';
import { buildProjectedTexture } from './pipeline/projectedBins.ts';
import { drainVisuals, runAxesReconstruction } from './pipeline/axesReconstruction.ts';
import { refreshModeVisualizations } from './pipeline/modeRefresh.ts';
import { updateContaminationOverlays } from './overlays/contaminationOverlays.ts';
import { updateGizmo, updateSphereOverlays } from './overlays/sphereOverlays.ts';
import { updateRecoveredCamGizmo } from './overlays/recoveredOverlays.ts';
import { drawSampleLattice } from './overlays/projectedCamOverlays.ts';
import { drawGridPeriodPhaseProjected } from './overlays/gridPeriodPhaseOverlays.ts';
import { pushPoseSync, pushSettingsSync, sendToDevBridge } from './devBridge/client.ts'; // also opens the dev-bridge websocket as a side effect

// ── Dev-console surface ──────────────────────────────────────────────────
//
// Every module's exports, flattened onto globalThis so the dev bridge can call
// them by bare name. Back when this app was one file, an eval'd snippet like
// `activeCamera().settings` just worked, because everything shared one top-level
// scope. Split across modules, devBridge/client.ts's direct eval sees only its
// OWN imports -- so the rest has to sit somewhere a bare identifier naturally
// falls back to, and that is globalThis.
//
// import.meta.glob rather than 66 hand-numbered `import * as NS0..NS65` lines,
// and the reason is not just brevity:
//
//   - THE OLD FORM NAMED NOTHING USEFUL BUT BROKE EVERY AUDIT. Because each
//     module was imported, every export it had appeared "used", so a
//     find-the-dead-exports pass returned nothing. Two dead GPU modules and a
//     whole retired distance path hid behind exactly that. A glob names no
//     export at all, so "grep for this symbol outside its own file" becomes a
//     valid deadness test again.
//   - It had already drifted: the numbering had gaps where modules were
//     deleted, and adding a module meant picking the next free integer.
//   - Nothing has to be edited to expose a new module. That matters more than
//     it sounds: editing any source file reloads the vite page, which destroys
//     the capture and the warm-up state a measurement was taken against.
//
// The two exclusions are load-bearing, not tidiness: this file would otherwise
// glob ITSELF into a circular import, and a .d.ts has no runtime module to
// import at all.
//
// ../pose/** IS LOAD-BEARING TOO, and it is the one line of this move that no
// typechecker could have caught. These patterns are resolved by vite at build
// time from strings, so when the pose library moved out of sphereLab/, every
// one of its exports would have silently stopped appearing on globalThis --
// and the dev bridge reaches them BY BARE NAME (runLsdChain, growRegionsCCL,
// computeLsdRectangles, and every harness entry point that calls them). The
// failure mode is a ReferenceError in a console eval, at the far end of a
// websocket, in a session where the page has to be re-warmed to try again.
const DEV_MODULES = import.meta.glob(
  ['./**/*.ts', '../pose/**/*.ts', '!./main.ts', '!./**/*.d.ts', '!../pose/**/*.d.ts'],
  { eager: true },
) as Record<string, Record<string, unknown>>;
// Collisions are REPORTED rather than silently resolved. Two modules exporting
// the same name means a bare console call could reach either one, and which it
// reaches depends on glob order -- the kind of thing that wastes an hour when a
// number comes back wrong. The old form had the same hazard and no warning.
{
  const seen = new Map<string, string>();
  const clashes: string[] = [];
  for (const [path, mod] of Object.entries(DEV_MODULES)) {
    for (const name of Object.keys(mod)) {
      const prev = seen.get(name);
      if (prev) clashes.push(`${name}: ${prev} vs ${path}`);
      else seen.set(name, path);
    }
    Object.assign(globalThis, mod);
  }
  if (clashes.length) console.warn('[devConsole] name collisions on globalThis:\n  ' + clashes.join('\n  '));
}
// The locals -- module-scope bindings of THIS file, which no glob can reach.
Object.assign(
  globalThis,
  { THREE, activeCamera, cameras, isSimulated, isPhysical, globalState, euler, canvas, readout, config,
    setMode, setPanelCollapsed, renderCameraTabs, refreshCameraPanel, renderViewport, layoutPip, resize,
    renderer, floorMesh, viewerCam, worldOrbit, insideCam, renderPreviewViewport, renderProjectedViewport,
    getAnalysisVFovRad, markCaptureDirty, resizeCaptureBuffers,
    renderCamRT, updateDistortedPreview, PREVIEW_UPDATE_INTERVAL_MS, buildProjectedTexture, runAxesReconstruction,
    updateContaminationOverlays, updateGizmo, updateSphereOverlays, updateRecoveredCamGizmo,
    drawSampleLattice, computeThroughRect, drawGridPeriodPhaseProjected },
);

type Mode = 'world' | 'through' | 'inside' | 'projected';

function animate() {
  requestAnimationFrame(animate);

  // Cheap pass, every camera: keeps gizmo transforms/visibility and the
  // great-sphere overlays current for the always-on world view. Today
  // that's exactly one camera; structured as a loop now so Stage B (N
  // simulated cameras, all visible at once) is a small diff here, not
  // another rewrite of this function.
  for (const camera of cameras.values()) {
    if (isSimulated(camera)) updateGizmo(camera);
    // NOT updateGizmo()'s own returned vFovRad -- getAnalysisVFovRad is the
    // single source of truth every other analysis call site uses too (see
    // its own comment); it recomputes from the exact same settings.horizFovDeg
    // + aspect updateGizmo just used, so this matches exactly rather than
    // duplicating the derivation or trusting two separate code paths to
    // agree.
    const vFovRad = getAnalysisVFovRad(camera);
    updateSphereOverlays(camera, vFovRad);

    if (isSimulated(camera)) {
      camera.gizmoBody.visible = globalState.mode === 'world' && camera.settings.showGizmoBody;
      camera.camHelper.visible = globalState.mode === 'world' && camera.settings.showFrustum;
    }
    updateRecoveredCamGizmo(camera);
    // recoveredFloorOverlay (the projected-image FILL) still requires real
    // pixel data (lastProjectedBins); recoveredFloorOutline (pose+FOV only,
    // see this session's on-device-pose-recovery plan) shows whenever pose
    // data exists at all, in EITHER compute mode.
    camera.recoveredFloorOverlay.visible = globalState.mode === 'world' && camera.settings.showRecoveredFloor && !!camera.lastPositionDecode && !!camera.lastProjectedBins;
    camera.recoveredFloorOutline.visible = globalState.mode === 'world' && camera.settings.showRecoveredFloor && !!camera.lastPositionDecode;

    if (isPhysical(camera)) {
      // On-connect settingsSync push -- mirrors lastReportedPipelined's own
      // "force a mismatch on the first tick" trick (see its own comment):
      // neverSyncedSettings starts true, so a freshly-connected phone gets
      // the desktop's current settings immediately instead of waiting for a
      // slider to actually change first (see this session's
      // on-device-pose-recovery plan).
      if (camera.neverSyncedSettings) {
        camera.neverSyncedSettings = false;
        pushSettingsSync(camera);
      }
      // Drain the mailbox every tick a frame's sitting in it and the camera
      // is actually free (see devBridge/client.ts's realCapture handler,
      // which always writes here now rather than calling ingestRealCapture
      // directly -- the mailbox can never overlap two decodes of the same
      // camera's capture buffers, which a direct call could).
      if (!camera.axesCapturing && !camera.captureIngestBusy && camera.pendingCapture) {
        const pending = camera.pendingCapture;
        camera.pendingCapture = null;
        camera.captureIngestBusy = true;
        ingestRealCapture(camera, pending)
          .catch((e) => console.error('[realCapture] ingest failed:', e))
          .finally(() => { camera.captureIngestBusy = false; });
      }
      // Symmetric drain for a device-compute phone's poseResult -- see
      // camera/model.ts's own comment on pendingPoseResult. A camera never
      // has both mailboxes populated (computeMode is stable per phone), so
      // sharing captureIngestBusy to gate both drains is correct.
      if (!camera.axesCapturing && !camera.captureIngestBusy && camera.pendingPoseResult) {
        const pending = camera.pendingPoseResult;
        camera.pendingPoseResult = null;
        camera.captureIngestBusy = true;
        ingestRemotePose(camera, pending)
          .catch((e) => console.error('[poseResult] ingest failed:', e))
          .finally(() => { camera.captureIngestBusy = false; });
      }
      // Tell the phone behind a physical camera whether Sphere Lab is still
      // crunching the last frame -- axesCapturing, exactly what drives the
      // shutter button's yellow "working" state. Purely informational on the
      // phone: the freshest-wins mailbox absorbs a send at any time, so it can
      // capture while yellow. Only sent on an actual change, not every frame.
      const ready = !camera.axesCapturing;
      if (ready !== camera.lastReportedReady) {
        const justFinished = ready && !camera.lastReportedReady;
        camera.lastReportedReady = ready;
        sendToDevBridge({ type: 'captureReady', captureId: camera.connectionId, ready });
        // Ships the freshly recovered pose back down to the phone (see
        // devBridge/client.ts's pushPoseSync) right as a desktop-compute
        // capture finishes -- same busy->ready edge captureReady itself
        // watches. Skipped in device-compute mode: the phone already has
        // this pose locally, and axesCapturing never toggles for that path
        // anyway (ingestRemotePose assigns pose data directly).
        if (justFinished && camera.computeMode === 'desktop') pushPoseSync(camera);
      }
    }

    // Drain the deferred-visualization mailbox (see
    // pipeline/axesReconstruction.ts's drainVisuals). Per-camera rather than
    // active-only because most of that tail is not active-only: pole markers,
    // the recovered-cam gizmo and the World-view floor decal are drawn for
    // EVERY camera, and only the mode-specific overlays are restricted to
    // whichever one is on screen. No-op on the overwhelming majority of ticks.
    //
    // Deliberately here, in the cheap per-camera pass, rather than down in the
    // `active` block below -- that block's auto-capture trigger flips
    // axesCapturing synchronously, and this drain declines to start while that
    // flag is set, so running afterwards would starve it on exactly the ticks
    // it could have run (every one, once the capture interval drops below a
    // reconstruction's duration).
    drainVisuals(camera);
  }
  floorMesh.visible = globalState.showFloor;
  // Expensive pass: only ever needed for the active camera (Through-Cam/
  // Projected-Cam/Inside-Sphere, and the PIP preview, only ever show one
  // camera at a time -- see this file's header comment).
  const active = activeCamera();
  const now = performance.now();
  if (active) {
    if (active.captureDirty && now - active.lastPreviewUpdate >= PREVIEW_UPDATE_INTERVAL_MS) {
      active.lastPreviewUpdate = now;
      active.captureDirty = false;
      if (isSimulated(active)) renderCamRT(active); // settings changed -- unlike a pure mode switch, this trigger genuinely needs a fresh render first
      refreshModeVisualizations(active, globalState.mode);
    }

    // visualsDraining joins axesCapturing here, and has to be tested HERE
    // rather than left to runAxesReconstruction's own guard: lastAxesCapture is
    // stamped before the call, so an attempt that bails inside would still burn
    // the interval slot and push the next capture out by a whole period
    // instead of by the drain's ~20ms. Both flags mean the same thing to this
    // trigger -- this camera is busy, don't restart the clock.
    if (active.settings.axesAutoCapture && !active.axesCapturing && !active.visualsDraining
      && now - active.lastAxesCapture >= active.settings.axesCaptureIntervalMs) {
      active.lastAxesCapture = now;
      runAxesReconstruction(active);
    }
  }

  renderer.setViewport(0, 0, innerWidth, innerHeight);
  renderer.setScissorTest(false);
  renderer.setClearColor(0x0a0a0f, 1);
  renderer.clear();

  if (globalState.mode === 'world') {
    worldOrbit.update();
    renderViewport(viewerCam, 0, 0, innerWidth, innerHeight);
    if (active) renderPreviewViewport(active, active.pipRect.x, innerHeight - active.pipRect.y - active.pipRect.h, active.pipRect.w, active.pipRect.h);
  } else if (globalState.mode === 'through') {
    if (active) {
      resizeThroughCamCanvas(active);
      drawThroughCamPreview(active);
      if (active.settings.showTrueContamination) drawThroughCamTrueContam(active);
      if (active.settings.showReconstructedContamination) drawThroughCamReconContam(active);
      if (active.settings.showTopGradient) drawThroughCamTopGradient(active);
      if (active.settings.showLsdRawRegions) drawThroughCamLsdRawRegions(active);
      if (active.settings.showLsdRejected) drawThroughCamLsdRejected(active);
    }
  } else if (globalState.mode === 'projected') {
    if (active) {
      // "use true cardinal orientation" (settings.ts) -- purely a display
      // rotation by however many 90-degree steps decode found the pattern
      // actually sitting at (camera.lastPositionDecode.orientation), so
      // rotating 1 or 3 steps swaps which of the camera's own aspect/
      // 1/aspect the ROTATED content's true shape needs, same as rotating a
      // photo 90 degrees swaps its width/height. +2 (180 degrees) corrects
      // an empirically-confirmed offset between decode's orientation index
      // and the actual on-screen rotation direction (see this session's chat).
      const rotationSteps = active.settings.useTrueCardinalOrientation && active.lastPositionDecode
        ? (active.lastPositionDecode.orientation + 2) % 4 : 0;
      // The bucket grid's OWN aspect (bins.w/bins.h), not active.aspect (the
      // viewport's) -- pose/stages/decode/decodeGrid.ts's squareCellBucketDims sizes
      // bucketW/bucketH from the recovered floor extent's aspect ratio
      // specifically so each bucket is a square in world units; stretching
      // that onto a rect shaped for the viewport's own (generally
      // different) aspect would undo it on screen. Falls back to
      // active.aspect before the first capture (no bins yet).
      const bins = active.lastProjectedBins;
      const binsAspect = bins ? bins.w / bins.h : active.aspect;
      const contentAspect = (rotationSteps === 1 || rotationSteps === 3) ? 1 / binsAspect : binsAspect;
      const availW = innerWidth;
      const availH = innerHeight;
      const winAspect = availW / availH;
      let w = availW, h = availH, x = 0, y = 0;
      if (winAspect > contentAspect) { w = availH * contentAspect; x = (availW - w) / 2; }
      else { h = availW / contentAspect; y = (availH - h) / 2; }
      renderProjectedViewport(active, x, innerHeight - y - h, w, h, rotationSteps);
      drawGridPeriodPhaseProjected(active, x, y, w, h, rotationSteps);
    }
  } else {
    // Inside-Sphere: only meaningful for a simulated camera's own ground-
    // truth pose (there is no equivalent for a physical camera -- see
    // updateSphereOverlays' header comment); falls back to the world
    // origin, looking however the free-look controls point, if the active
    // camera isn't simulated or doesn't exist yet.
    insideCam.position.copy(active && isSimulated(active) ? active.camPos : new THREE.Vector3());
    euler.set(insidePitch, insideYaw, 0);
    insideCam.quaternion.setFromEuler(euler);
    renderViewport(insideCam, 0, 0, innerWidth, innerHeight);
    if (active) renderPreviewViewport(active, active.pipRect.x, innerHeight - active.pipRect.y - active.pipRect.h, active.pipRect.w, active.pipRect.h);
  }
}

// config.global.mode is typed Mode, but it arrives from a JSON file a human
// edits -- so it is checked against the real list rather than trusted, the
// same way the field-view radio group falls back to a real option.
const VALID_MODES: Mode[] = ['world', 'through', 'inside', 'projected'];
setMode(VALID_MODES.includes(globalState.mode) ? globalState.mode : 'through');

// No default camera (see this file's header) -- activeCameraId is already
// '' at this point, so this just paints the tab bar (Global tab only, "+")
// and the Global-only panel state for the very first frame.
renderCameraTabs();
refreshCameraPanel();
animate();
