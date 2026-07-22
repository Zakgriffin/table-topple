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
// types.ts/state.ts/constants.ts   shared types + tiny bits of module state
// math/geometry.ts                 pure sphere/ray-casting math
// camera/                          the Camera data model: settings, types,
//                                   the live store, factories, add/remove/kick
// scene/                           THREE.js scene setup shared by every
//                                   camera: renderer, floor+pattern, quad
//                                   blitters, world/inside-sphere controls
// pipeline/                        the CV/reconstruction math -- no THREE
//                                   scene graph, no DOM; plain data in,
//                                   plain data out
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
import type { Camera, SimulatedCamera } from './camera/model.ts';
import { globalState } from './state.ts';
import { euler } from './constants.ts';
import { canvas, readout, savedControls } from './ui/dom.ts';
import { setMode, setPanelCollapsed } from './ui/mode.ts';
import { renderCameraTabs, refreshCameraPanel } from './ui/cameraPanel.ts';
import { renderViewport, layoutPip, resize } from './ui/layout.ts';
import './ui/cameraPanel.ts'; // side effect: wires every slider/checkbox to the active camera
import { renderer } from './scene/renderer.ts';
import { floorMesh } from './scene/floor.ts';
import { viewerCam, worldOrbit, insideCam, insideYaw, insidePitch } from './scene/viewerControls.ts';
import {
  renderPreviewViewport, renderProjectedViewport, renderTrueContamOverlay, renderReconContamOverlay, renderTopGradientOverlay, renderTangentWalkPathOverlay, renderBucketFillOverlay, renderBucketFillJoinOverlay,
} from './scene/quadRenderers.ts';
import { getAnalysisVFovRad, markCaptureDirty, resizeCaptureBuffers, renderCamRT } from './pipeline/capture.ts';
import { updateDistortedPreview, PREVIEW_UPDATE_INTERVAL_MS } from './pipeline/preview.ts';
import { buildProjectedTexture } from './pipeline/decodeGrid.ts';
import { runAxesReconstruction } from './pipeline/axesReconstruction.ts';
import { updateContaminationOverlays } from './overlays/contaminationOverlays.ts';
import { updateTopGradientOverlay } from './overlays/gradientHighlightOverlays.ts';
import { updateBucketFillOverlay } from './overlays/bucketFillOverlay.ts';
import { updateBucketFillJoinOverlay } from './overlays/bucketFillJoinOverlay.ts';
import { updateGizmo, updateSphereOverlays } from './overlays/sphereOverlays.ts';
import { updateRecoveredCamGizmo } from './overlays/recoveredOverlays.ts';
import { drawMarginalLines, drawSampleLattice, MARGINAL_THICKNESS } from './overlays/projectedCamOverlays.ts';
import { computeThroughRect, lastHoverClientX, lastHoverClientY, updateHoverOverlays } from './overlays/hoverDebugOverlays.ts';
import { sendToDevBridge } from './devBridge/client.ts'; // also opens the dev-bridge websocket as a side effect

// Every module's exports, purely so devBridge/client.ts's `eval(msg.code)`
// can still see the whole app as one flat scope -- back when this was a
// single file, an eval'd snippet like `activeCamera().settings` just worked
// because everything was already in the same top-level scope; split across
// modules, direct eval only sees devBridge/client.ts's own imports unless
// the rest is put somewhere it naturally falls back to. Attaching everything
// to globalThis here restores that: a bare identifier direct eval can't
// resolve lexically still falls through to the global scope, same as before.
import * as NS0 from './camera/factory.ts';
import * as NS1 from './camera/lifecycle.ts';
import * as NS2 from './camera/model.ts';
import * as NS3 from './camera/settings.ts';
import * as NS4 from './camera/store.ts';
import * as NS5 from './constants.ts';
import * as NS6 from './devBridge/client.ts';
import * as NS7 from './math/geometry.ts';
import * as NS8 from './overlays/contaminationOverlays.ts';
import * as NS9 from './overlays/hoverDebugOverlays.ts';
import * as NS10 from './overlays/projectedCamOverlays.ts';
import * as NS11 from './overlays/recoveredOverlays.ts';
import * as NS12 from './overlays/sphereOverlays.ts';
import * as NS13 from './pipeline/axesReconstruction.ts';
import * as NS14 from './pipeline/capture.ts';
import * as NS15 from './pipeline/contamination.ts';
import * as NS16 from './pipeline/decodeGrid.ts';
import * as NS17 from './pipeline/distortion.ts';
import * as NS18 from './pipeline/gradientField.ts';
import * as NS19 from './pipeline/orientationLM.ts';
import * as NS20 from './pipeline/positionLM.ts';
import * as NS21 from './pipeline/preview.ts';
import * as NS22 from './pipeline/tangentWalk.ts';
import * as NS23 from './pipeline/votes.ts';
import * as NS24 from './scene/floor.ts';
import * as NS25 from './scene/quadRenderers.ts';
import * as NS26 from './scene/renderer.ts';
import * as NS27 from './scene/viewerControls.ts';
import * as NS28 from './state.ts';
import * as NS30 from './ui/cameraPanel.ts';
import * as NS31 from './ui/dom.ts';
import * as NS32 from './ui/layout.ts';
import * as NS33 from './ui/mode.ts';
import * as NS34 from './pipelineGPU/device.ts';
import * as NS35 from './pipelineGPU/voteGeneration.ts';
import * as NS36 from './pipelineGPU/positionLM.ts';
import * as NS37 from './profiling/profiler.ts';
import * as NS38 from './pipelineGPU/fitPlanes.ts';
import * as NS39 from './pipelineGPU/decodeTally.ts';
import * as NS40 from './pipelineGPU/voteBandSelect.ts';
import * as NS41 from './pipelineGPU/projectSamples.ts';
import * as NS42 from './overlays/gradientHighlightOverlays.ts';
import * as NS43 from './pipeline/gradientHighlight.ts';
import * as NS44 from './overlays/bucketFillOverlay.ts';
import * as NS45 from './pipeline/bucketFillSegments.ts';
import * as NS46 from './overlays/bucketFillJoinOverlay.ts';
import * as NS47 from './pipeline/bucketFillJoin.ts';
Object.assign(
  globalThis,
  NS0, NS1, NS2, NS3, NS4, NS5, NS6, NS7, NS8, NS9, NS10, NS11, NS12, NS13, NS14, NS15, NS16, NS17,
  NS18, NS19, NS20, NS21, NS22, NS23, NS24, NS25, NS26, NS27, NS28, NS30, NS31, NS32, NS33, NS34, NS35, NS36, NS37, NS38, NS39, NS40, NS41, NS42, NS43, NS44, NS45, NS46, NS47,
  { THREE, activeCamera, cameras, isSimulated, isPhysical, globalState, euler, canvas, readout, savedControls,
    setMode, setPanelCollapsed, renderCameraTabs, refreshCameraPanel, renderViewport, layoutPip, resize,
    renderer, floorMesh, viewerCam, worldOrbit, insideCam, renderPreviewViewport, renderProjectedViewport,
    renderTrueContamOverlay, renderReconContamOverlay, getAnalysisVFovRad, markCaptureDirty, resizeCaptureBuffers,
    renderCamRT, updateDistortedPreview, PREVIEW_UPDATE_INTERVAL_MS, buildProjectedTexture, runAxesReconstruction,
    updateContaminationOverlays, updateGizmo, updateSphereOverlays, updateRecoveredCamGizmo, drawMarginalLines,
    drawSampleLattice, MARGINAL_THICKNESS, computeThroughRect },
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
    camera.recoveredFloorOverlay.visible = globalState.mode === 'world' && camera.settings.showRecoveredFloor && !!camera.lastPositionDecode;

    // Tell the phone behind a physical camera whether it's safe to send
    // another frame -- axesCapturing is exactly "still crunching the last
    // one" (see runAxesReconstruction). Only fires on an actual true/false
    // transition, not every frame, via the lastReportedReady comparison.
    if (isPhysical(camera)) {
      const ready = !camera.axesCapturing;
      if (ready !== camera.lastReportedReady) {
        camera.lastReportedReady = ready;
        sendToDevBridge({ type: 'captureReady', captureId: camera.connectionId, ready });
      }
    }
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
      if (isSimulated(active)) renderCamRT(active);
      updateDistortedPreview(active);
      if (globalState.mode === 'projected') buildProjectedTexture(active);
      if (globalState.mode === 'through') {
        updateContaminationOverlays(active); updateTopGradientOverlay(active); updateBucketFillOverlay(active); updateBucketFillJoinOverlay(active);
        updateHoverOverlays(lastHoverClientX, lastHoverClientY); // refreshes the persistent bucket-fill segment markers to match, even without a pointermove
      }
    }

    if (active.settings.axesAutoCapture && !active.axesCapturing && now - active.lastAxesCapture >= active.settings.axesCaptureIntervalMs) {
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
      const { x, y, w, h } = computeThroughRect(active);
      renderPreviewViewport(active, x, y, w, h);
      if (active.settings.showTrueContamination) renderTrueContamOverlay(active, x, y, w, h);
      if (active.settings.showReconstructedContamination) renderReconContamOverlay(active, x, y, w, h);
      if (active.settings.showTopGradient) renderTopGradientOverlay(active, x, y, w, h);
      if (active.settings.showTangentWalkPath) renderTangentWalkPathOverlay(active, x, y, w, h);
      if (active.settings.showBucketFillSegments) renderBucketFillOverlay(active, x, y, w, h);
      if (active.settings.showBucketFillJoin) renderBucketFillJoinOverlay(active, x, y, w, h);
    }
  } else if (globalState.mode === 'projected') {
    if (active) {
      const availW = innerWidth - MARGINAL_THICKNESS;
      const availH = innerHeight - MARGINAL_THICKNESS;
      const winAspect = availW / availH;
      let w = availW, h = availH, x = 0, y = 0;
      if (winAspect > active.aspect) { w = availH * active.aspect; x = (availW - w) / 2; }
      else { h = availW / active.aspect; y = (availH - h) / 2; }
      renderProjectedViewport(active, x, innerHeight - y - h, w, h);
      drawMarginalLines(active, x, y, w, h);
      drawSampleLattice(active, x, y, w, h);
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

// Same persistence as every slider/checkbox -- only honor a saved value if
// it's still a real Mode.
const VALID_MODES: Mode[] = ['world', 'through', 'inside', 'projected'];
const savedMode = savedControls['mode'];
setMode(VALID_MODES.includes(savedMode as Mode) ? (savedMode as Mode) : 'world');

// No default camera (see this file's header) -- activeCameraId is already
// '' at this point, so this just paints the tab bar (Global tab only, "+")
// and the Global-only panel state for the very first frame.
renderCameraTabs();
refreshCameraPanel();
animate();
