import { addSimulatedCamera, removeCameraTab, selectGlobalTab } from '../camera/lifecycle.ts';
import { type PhysicalCamera } from '../../shared/camera/model.ts';
import { activeCamera, activeCameraId, cameras, isPhysical, isSimulated, setActiveCameraId } from '../camera/store.ts';
import { pushSettingsSync, sendToDevBridge } from '../devBridge/client.ts';
import { rebuildGridLineKs } from '../../shared/math/geometry.ts';
import { config, configAsJson, discardSavedOverlay, fetchConfigFile, registerActiveCameraSettingsSource } from '../../shared/config.ts';
import { updateContaminationAvailability } from '../overlays/contaminationOverlays.ts';
import { updateTopGradientAvailability } from '../overlays/gradientHighlightOverlays.ts';
import { lastHoverClientX, lastHoverClientY, updateGradientArrowAvailability, updateHoverOverlays } from '../overlays/hoverDebugOverlays.ts';
import { updateLsdAvailability, updateLsdOverlay } from '../overlays/lsdOverlay.ts';
import { updateGradientCirclesDebug } from '../overlays/sphereOverlays.ts';
import { drawGridPeriodPhasePlot } from '../overlays/gridPeriodPhaseOverlays.ts';
import { recomputeFromLastCapture, runAxesReconstruction, updateChainTransfersReadout } from '../pipeline/axesReconstruction.ts';
import { markCaptureDirty, resizeCaptureBuffers } from '../pipeline/capture.ts';
import { backendFromForceCPU } from '../../shared/backend.ts';
import { buildProjectedTexture } from '../pipeline/projectedBins.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { isWebGPUSupported } from '../../../gpu/device.ts';
import { profilerReset, profilerSetDevToolsMirror } from '../../../profiling/profiler.ts';
import { rebuildFloorPattern, rebuildFloorTexture } from '../scene/floor.ts';
import { globalState } from '../../shared/state.ts';
import { type FieldView } from '../../shared/types.ts';
import { bindCheckbox, bindRadioGroup, bindSlider, loadConfigBtn, saveConfigBtn, configStatus, cameraSettingsSectionsEl, cameraTabsEl, captureAxesBtn, fieldViewRawLabel, globalSettingsSectionEl, gpuVotesStatus, physCameraDetailFields, physCaptureModeReadout, setSectionHidden, simCameraDetailFields, simDistortionSection, simOnlyFieldViews, toggleCompositeLineFamiliesBtn, toggleDistinctnessCurveBtn, toggleGapHistogramBtn, toggleGradientArrowBtn, toggleProductCurveBtn, toggleHideFieldBtn, toggleLevelLineArrowBtn, toggleLineJoinBtn, toggleLsdCompositeBtn, toggleLsdRawRegionsBtn, toggleLsdRejectedBtn, toggleLsdSegmentsBtn, toggleReconContamBtn, toggleTopGradientBtn, toggleRectifiedLinesBtn, toggleSampleLatticeBtn, toggleTrueCardinalOrientationBtn, toggleTrueContamBtn, toggleValueHistogramBtn } from './dom.ts';
import { layoutPip } from './layout.ts';

// Tells config.ts which camera's settings persistConfig should capture before
// it writes -- see registerActiveCameraSettingsSource for why it is pushed down
// rather than imported. Registered before the bindings below, since each of
// those applies once at bind time and persists as it goes.
registerActiveCameraSettingsSource(() => activeCamera()?.settings ?? null);

// Promotes the live config (disk defaults + this browser's localStorage
// overlay) back onto pose-viewer.config.json, through the dev bridge -- a
// browser page cannot write to the project directory itself. Deliberately
// explicit rather than a write-through on every slider: the file is the
// reviewable default, and a config that churned on every drag would be
// impossible to read a diff of.
saveConfigBtn.addEventListener('click', () => {
  configStatus.textContent = 'saving…';
  sendToDevBridge({ type: 'saveConfig', json: configAsJson() });
});

// The other direction: throw away this browser's overlay and take the file.
//
// It RELOADS rather than re-applying in place, and that is the point. Settings
// are spread across N camera objects, ~45 DOM controls and globalState by the
// time the page is running, and re-driving all of that would be a second code
// path that could disagree with boot about what the file means. A reload IS
// the boot path, so "load" and "open the page fresh" cannot diverge.
//
// The file is fetched and validated BEFORE the overlay is discarded, so a
// broken config leaves you exactly where you were, with the reason on screen,
// rather than dropping your edits and then failing to come back up.
loadConfigBtn.addEventListener('click', async () => {
  if (!confirm('Discard this browser\'s config edits and reload from pose-viewer.config.json?\n\nThe current capture is lost.')) return;
  configStatus.textContent = 'loading…';
  try {
    await fetchConfigFile();
  } catch (err) {
    configStatus.textContent = `load failed, nothing discarded: ${err instanceof Error ? err.message : String(err)}`;
    return;
  }
  discardSavedOverlay();
  location.reload();
});

// Rebuilds the tab bar from `cameras` (Map iteration = creation order) --
// called after anything that adds/removes/renames a camera or changes which
// one is active. Cheap enough (a handful of plain DOM nodes) to just rebuild
// wholesale rather than diff.
export function renderCameraTabs() {
  cameraTabsEl.innerHTML = '';

  // Always present, never closable -- the panel's own home when no camera
  // is selected (including on a fresh load: there's no default camera
  // anymore, see this file's header). No --tab-color override, so it picks
  // up .cameraTab's own neutral default instead of a camera's own color.
  const globalTab = document.createElement('button');
  globalTab.className = 'cameraTab globalTab' + (activeCameraId === '' ? ' active' : '');
  globalTab.textContent = 'Global';
  globalTab.title = 'global settings (shared by every camera)';
  globalTab.addEventListener('click', () => selectGlobalTab());
  cameraTabsEl.appendChild(globalTab);

  for (const camera of cameras.values()) {
    const tab = document.createElement('button');
    tab.className = 'cameraTab' + (camera.id === activeCameraId ? ' active' : '');
    tab.style.setProperty('--tab-color', `#${camera.color.getHexString()}`);
    tab.title = camera.type === 'simulated' ? 'simulated camera' : 'physical camera';
    const label = document.createElement('span');
    label.textContent = camera.name;
    tab.appendChild(label);
    // A physical camera is ALWAYS a real phone connection now (see this
    // file's header) -- its close button KICKS that connection server-side
    // instead of removing the tab locally; tab removal itself waits for the
    // resulting captureDisconnected broadcast rather than happening
    // optimistically, so it stays correct if the kick races with some other
    // disconnect reason. A simulated camera just gets removed locally and
    // immediately, same as Stage B. Every camera can be closed now,
    // including the last one -- zero cameras is a normal, supported state
    // (see removeCameraTab).
    {
      const isPhysicalCam = camera.type === 'physical';
      const close = document.createElement('span');
      close.className = 'cameraTabClose';
      close.textContent = isPhysicalCam ? '⏻' : '×';
      close.title = isPhysicalCam ? 'kick (disconnect the phone)' : 'remove this camera';
      close.addEventListener('click', (e) => {
        e.stopPropagation();
        if (isPhysicalCam) sendToDevBridge({ type: 'kickCapture', captureId: (camera as PhysicalCamera).connectionId });
        else removeCameraTab(camera.id);
      });
      tab.appendChild(close);
    }
    tab.addEventListener('click', () => {
      if (camera.id === activeCameraId) return;
      setActiveCameraId(camera.id);
      renderCameraTabs();
      refreshCameraPanel();
    });
    cameraTabsEl.appendChild(tab);
  }
  const addBtn = document.createElement('button');
  addBtn.className = 'cameraTabAdd';
  addBtn.textContent = '+';
  addBtn.title = 'add a simulated camera';
  addBtn.addEventListener('click', () => addSimulatedCamera());
  cameraTabsEl.appendChild(addBtn);
}

// Drives a slider from code the same way a drag does: set the value, then
// dispatch the 'input' event bindSlider is already listening for, so the
// readout, the setting write, persistence and the onChange reaction all happen
// through the one path. Module-level rather than local to refreshCameraPanel
// because the join toggle below is a second caller -- a button that moves a
// slider is exactly this, and doing it by assignment would leave the thumb
// where it was.
function setPanelNum(id: string, v: number) {
  const el = document.getElementById(id) as HTMLInputElement | null;
  if (!el) return;
  el.value = String(v);
  el.dispatchEvent(new Event('input'));
}

// Re-syncs the WHOLE side panel to match whatever's currently selected --
// either a specific camera (per-camera controls, further split into
// simulated-only/physical-only sub-fields) or the Global tab (just
// globalSettingsSection). Per-camera writes already redirect correctly on
// every tick regardless of what's currently displayed (bindSlider/
// bindCheckbox's onChange callbacks all look up activeCamera() fresh each
// time they fire) -- this function is purely the other direction, state ->
// DOM. Dispatches the SAME 'input'/'change' events bindSlider/bindCheckbox
// already listen for, reusing their existing fmt/persist/onChange logic
// wholesale instead of duplicating it; the onChange round-trip this causes
// (writing the same value straight back to the SAME camera it was just read
// from) is a harmless no-op.
export function refreshCameraPanel() {
  const cam = activeCamera();
  setSectionHidden(globalSettingsSectionEl, !!cam);
  setSectionHidden(cameraSettingsSectionsEl, !cam);
  // Before the early return, and passing whatever `cam` is: the traffic
  // readout is per-camera, so switching tabs has to repoint it (or clear it on
  // the Global tab) rather than leave another camera's numbers sitting under
  // the toggles looking current.
  updateChainTransfersReadout(cam);
  if (!cam) return;

  // Every slider/checkbox inside cameraSettingsSectionsEl reads its
  // accent-color from --cam-accent (see pose-viewer-server.html's CSS -- falls back
  // to the fixed green everywhere else, e.g. the Global tab's own controls,
  // which aren't tied to any one camera) -- setting it once here, on the
  // shared container, is enough for every control inside to pick it up via
  // ordinary CSS inheritance, no per-control wiring needed.
  cameraSettingsSectionsEl.style.setProperty('--cam-accent', `#${cam.color.getHexString()}`);

  setSectionHidden(simCameraDetailFields, !isSimulated(cam));
  setSectionHidden(physCameraDetailFields, isSimulated(cam));
  setSectionHidden(simDistortionSection, !isSimulated(cam));
  setSectionHidden(simOnlyFieldViews, !isSimulated(cam));
  fieldViewRawLabel.textContent = isSimulated(cam) ? 'raw' : 'capture';

  const setNum = setPanelNum;
  const setBool = (id: string, v: boolean) => {
    const el = document.getElementById(id) as HTMLInputElement | null;
    if (!el) return;
    el.checked = v;
    el.dispatchEvent(new Event('change'));
  };

  if (isSimulated(cam)) {
    setNum('camX', cam.settings.camX); setNum('camY', cam.settings.camY); setNum('camZ', cam.settings.camZ);
    setNum('camYaw', cam.settings.camYawDeg); setNum('camTilt', cam.settings.tiltDeg); setNum('camFov', cam.settings.horizFovDeg);
    setNum('simNoise', cam.settings.simNoise); setNum('simBlur', cam.settings.simBlur); setNum('captureSupersample', cam.settings.captureSupersample);
    setNum('viewportW', cam.settings.viewportW); setNum('viewportH', cam.settings.viewportH);
    setBool('aspectLocked', cam.settings.aspectLocked);
  } else {
    setNum('realCaptureFovDeg', cam.settings.horizFovDeg);
    physCaptureModeReadout.textContent = cam.captureMode;
  }

  setBool('showSphere', cam.settings.showSphere); setBool('showCircles', cam.settings.showCircles);
  setBool('showPoles', cam.settings.showPoles); setBool('showFrustum', cam.settings.showFrustum);
  setBool('showPatch', cam.settings.showPatch); setBool('showGizmoBody', cam.settings.showGizmoBody);
  setBool('showRecoveredFloor', cam.settings.showRecoveredFloor); setNum('recoveredFloorOpacity', cam.settings.recoveredFloorOpacity);
  setBool('latticeBoundsProjection', cam.settings.latticeBoundsProjection);
  setNum('gridPeriodPhaseBinCount', cam.settings.gridPeriodPhaseBinCount);
  setNum('gridPeriodPhaseGapLowerBound', cam.settings.gridPeriodPhaseGapLowerBound);


  const fieldViewId = 'fieldView' + cam.settings.fieldView[0].toUpperCase() + cam.settings.fieldView.slice(1);
  const fieldViewInput = document.getElementById(fieldViewId) as HTMLInputElement | null;
  if (fieldViewInput) { fieldViewInput.checked = true; fieldViewInput.dispatchEvent(new Event('change')); }

  setNum('gradientArrowScale', cam.settings.gradientArrowScale);
  setNum('lsdToleranceDeg', cam.settings.lsdToleranceDeg);
  setNum('lsdRhoNoiseThreshold', cam.settings.lsdRhoNoiseThreshold);
  setNum('lsdRhoHighThreshold', cam.settings.lsdRhoHighThreshold);
  setNum('lsdCclSteps', cam.settings.lsdCclSteps);
  setNum('lsdMinRegionSize', cam.settings.lsdMinRegionSize);
  setNum('lsdNfaEpsilon', cam.settings.lsdNfaEpsilon);
  setNum('lsdNfaTestExponent', cam.settings.lsdNfaTestExponent);
  setNum('lsdMinLengthPx', cam.settings.lsdMinLengthPx);
  setNum('joinKSigma', cam.settings.joinKSigma);
  setNum('joinEndpointNoisePx', cam.settings.joinEndpointNoisePx);
  setNum('joinReachFrac', cam.settings.joinReachFrac);
  setNum('joinRounds', cam.settings.joinRounds);
  setNum('joinMaxResidualPx', cam.settings.joinMaxResidualPx);
  setBool('joinPolarityAbs', cam.settings.joinPolarityAbs);
  setBool('showRecoveredPoles', cam.settings.showRecoveredPoles); setBool('showAxisVectors', cam.settings.showAxisVectors);
  setBool('showTopCircles', cam.settings.showTopCircles);
  setNum('topCirclesLineWidth', cam.settings.topCirclesLineWidth);
  setNum('minGrazingCos', cam.settings.minGrazingCos);
  setBool('axesAutoCapture', cam.settings.axesAutoCapture);
  setNum('axesCaptureInterval', cam.settings.axesCaptureIntervalMs);

  toggleHideFieldBtn.classList.toggle('active', cam.settings.hideField);
  toggleTrueContamBtn.classList.toggle('active', cam.settings.showTrueContamination);
  toggleReconContamBtn.classList.toggle('active', cam.settings.showReconstructedContamination);
  toggleTrueCardinalOrientationBtn.classList.toggle('active', cam.settings.useTrueCardinalOrientation);
  toggleSampleLatticeBtn.classList.toggle('active', cam.settings.showSampleLattice);
  toggleRectifiedLinesBtn.classList.toggle('active', cam.settings.showRectifiedLines);
  toggleGradientArrowBtn.classList.toggle('active', cam.settings.showGradientArrow);
  toggleLevelLineArrowBtn.classList.toggle('active', cam.settings.showLevelLineArrow);
  toggleTopGradientBtn.classList.toggle('active', cam.settings.showTopGradient);
  toggleLsdSegmentsBtn.classList.toggle('active', cam.settings.showLsdSegments);
  toggleLsdRejectedBtn.classList.toggle('active', cam.settings.showLsdRejected);
  toggleLsdRawRegionsBtn.classList.toggle('active', cam.settings.showLsdRawRegions);
  toggleLineJoinBtn.classList.toggle('active', cam.settings.joinKSigma > 0);
  toggleLsdCompositeBtn.classList.toggle('active', cam.settings.showLsdComposite);
  toggleCompositeLineFamiliesBtn.classList.toggle('active', cam.settings.showCompositeLineFamilies);
  toggleGapHistogramBtn.classList.toggle('active', cam.settings.showGapHistogram);
  toggleValueHistogramBtn.classList.toggle('active', cam.settings.showValueHistogram);
  toggleDistinctnessCurveBtn.classList.toggle('active', cam.settings.showDistinctnessCurve);
  toggleProductCurveBtn.classList.toggle('active', cam.settings.showProductCurve);
  updateContaminationAvailability();
  updateGradientArrowAvailability();
  updateTopGradientAvailability();
  updateLsdAvailability();

  updateDistortedPreview(cam);
  if (globalState.mode === 'projected') buildProjectedTexture(cam, backendFromForceCPU(globalState.forceCPU));
  markCaptureDirty(cam);
  layoutPip(cam);
  drawGridPeriodPhasePlot(cam);
}


function rerunOnRealCaptureSettingChange() {
  const cam = activeCamera();
  // computeMode === 'desktop' guard: this trigger's whole purpose is "redo
  // the DESKTOP's own reconstruction" -- never correct to fire automatically
  // in device-compute mode, regardless of why lastRealCaptureGray happens to
  // be populated there (e.g. the phone's sendCapturedImage debug toggle).
  // Confirmed live this session: without this, a tab switch while that
  // toggle was on silently clobbered the phone's own on-device pose with a
  // desktop recompute, which of course looked "correct" since desktop-
  // compute is the known-good path -- there was no bug in the on-device
  // result being displayed, just never a clean look at it.
  if (cam && isPhysical(cam) && cam.computeMode === 'desktop' && cam.lastRealCaptureGray) runAxesReconstruction(cam);
}
// Per-camera-settings sliders (the 16 fields making up PoseInput['settings'],
// see pose/poseCompute.ts) push a fresh settingsSync to THAT camera's own
// phone only -- source of truth stays the desktop's own sliders (see this
// session's on-device-pose-recovery plan). No-op for a simulated camera
// (nothing to push to) or a camera not currently active.
function pushSettingsIfPhysical() {
  const cam = activeCamera();
  if (cam && isPhysical(cam)) pushSettingsSync(cam);
}
// Recomputes immediately, exactly like every other per-camera slider here.
// This used to debounce its rerun behind a 200ms timer (plus a
// value-actually-changed guard to keep refreshCameraPanel's own re-sync from
// re-arming it), which bought nothing: runAxesReconstruction already
// self-throttles on camera.axesCapturing, so a fast drag drops overlapping
// calls rather than queueing them -- the same reason no other slider needs a
// timer.
bindSlider('realCaptureFovDeg', config.camera.common.horizFovDeg, (v) => {
  const cam = activeCamera();
  if (!cam || !isPhysical(cam)) return;
  cam.settings.horizFovDeg = v;
  markCaptureDirty(cam);
  pushSettingsSync(cam);
  rerunOnRealCaptureSettingChange();
}, (v) => `${v.toFixed(0)}°`);
// Tier 1 (invalidates the capture itself -- see this session's chat on
// "every setting recomputes everything downstream of it"): markCaptureDirty
// still drives the passive preview loop, runAxesReconstruction is the real
// recompute -- it self-throttles via camera.axesCapturing, so a fast drag
// just drops overlapping calls rather than queueing them, same as an
// overlapping "capture now" click.
bindSlider('camX', config.camera.simulated.camX, (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camX = v; markCaptureDirty(cam); runAxesReconstruction(cam); } });
bindSlider('camY', config.camera.simulated.camY, (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camY = v; markCaptureDirty(cam); runAxesReconstruction(cam); } });
bindSlider('camZ', config.camera.simulated.camZ, (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camZ = v; markCaptureDirty(cam); runAxesReconstruction(cam); } });
bindSlider('camYaw', config.camera.simulated.camYawDeg, (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camYawDeg = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('camTilt', config.camera.simulated.tiltDeg, (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.tiltDeg = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('camFov', config.camera.common.horizFovDeg, (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.horizFovDeg = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => `${v.toFixed(0)}°`);

export let syncingViewportAspect = false;
function clampViewport(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(v)));
}
bindSlider('viewportW', config.camera.common.viewportW, (v) => {
  const cam = activeCamera(); if (!cam) return;
  const oldAspect = cam.settings.viewportW / cam.settings.viewportH;
  cam.settings.viewportW = v;
  if (cam.settings.aspectLocked && !syncingViewportAspect) {
    syncingViewportAspect = true;
    const hInput = document.getElementById('viewportH') as HTMLInputElement;
    hInput.value = String(clampViewport(v / oldAspect, 96, 2000));
    hInput.dispatchEvent(new Event('input'));
    syncingViewportAspect = false;
  }
  resizeCaptureBuffers(cam);
  runAxesReconstruction(cam);
}, (v) => v.toFixed(0));
bindSlider('viewportH', config.camera.common.viewportH, (v) => {
  const cam = activeCamera(); if (!cam) return;
  const oldAspect = cam.settings.viewportW / cam.settings.viewportH;
  cam.settings.viewportH = v;
  if (cam.settings.aspectLocked && !syncingViewportAspect) {
    syncingViewportAspect = true;
    const wInput = document.getElementById('viewportW') as HTMLInputElement;
    wInput.value = String(clampViewport(v * oldAspect, 128, 2000));
    wInput.dispatchEvent(new Event('input'));
    syncingViewportAspect = false;
  }
  resizeCaptureBuffers(cam);
  runAxesReconstruction(cam);
}, (v) => v.toFixed(0));
bindCheckbox('aspectLocked', config.camera.common.aspectLocked, (v) => { const cam = activeCamera(); if (cam) cam.settings.aspectLocked = v; });

bindCheckbox('showSphere', config.camera.common.showSphere, (v) => { const cam = activeCamera(); if (cam) cam.settings.showSphere = v; });
bindCheckbox('showCircles', config.camera.common.showCircles, (v) => { const cam = activeCamera(); if (cam) cam.settings.showCircles = v; });
bindCheckbox('showPoles', config.camera.common.showPoles, (v) => { const cam = activeCamera(); if (cam) cam.settings.showPoles = v; });
bindCheckbox('showFrustum', config.camera.common.showFrustum, (v) => { const cam = activeCamera(); if (cam) cam.settings.showFrustum = v; });
bindCheckbox('showPatch', config.camera.common.showPatch, (v) => { const cam = activeCamera(); if (cam) cam.settings.showPatch = v; });
bindCheckbox('showFloor', config.global.showFloor, (v) => { globalState.showFloor = v; });
// (`overlayPaneThisViewOnly` is bound in ui/mode.ts, with the panel chrome it
// belongs to -- it selects which of the right pane's view groups are on screen
// and touches no camera at all.)
bindSlider('floorCellOutlineSubdiv', config.global.floorCellOutlineSubdiv, (v) => {
  globalState.floorCellOutlineSubdiv = v;
  rebuildFloorTexture();
  for (const cam of cameras.values()) markCaptureDirty(cam); // this IS the real rendered floor, so every camera's capture path needs to re-render too
}, (v) => v.toFixed(0));
// Global fields (this slider, and forceCPU below) push to EVERY connected
// physical camera's phone, not just the active one -- conceptually shared even
// though their recompute-trigger only targets activeCamera().
//
// forceCPU is pushed WHOLESALE, where the nine toggles it replaced were not:
// The display-projection toggle was deliberately excluded from the sync because it only affects
// display-only projection, which is not part of the phone-portable pose
// pipeline. Collapsing them makes that exclusion unexpressible -- one flag
// cannot be half-synced -- so the phone now also switches its (unused) display
// projection. Harmless, and worth one sentence rather than a second flag.
function pushSettingsSyncToAllPhysical() {
  for (const cam of cameras.values()) if (isPhysical(cam)) pushSettingsSync(cam);
}
bindSlider('boardSize', config.global.boardSize, (v) => {
  globalState.boardSize = v;
  rebuildFloorPattern(v); // re-crops the torus, rebuilds the decode lookup table, resizes the floor mesh/texture/reference lines
  rebuildGridLineKs(); // reads HALF_R/HALF_C, which rebuildFloorPattern just updated -- must run after it
  // The GPU decode-tally's hash table used to be cached here and invalidated on
  // a board-size change. That cache died with the old pipeline. src/pose rebuilds its
  // board buffers per context (board.ts), so there is nothing to invalidate --
  // but a context built against the OLD board is now stale, and wiring that up
  // is part of swapping the app onto pose.
  for (const cam of cameras.values()) markCaptureDirty(cam); // this IS the real rendered floor, so every camera's capture path needs to re-render/re-decode against the new board
  pushSettingsSyncToAllPhysical();
}, (v) => v.toFixed(0));
// THE global GPU/CPU switch, replacing nine per-stage toggles -- see state.ts's
// forceCPU for why they went. Flipping it changes what the NEXT recompute
// produces, so re-run against whichever camera is on screen immediately rather
// than waiting for the next unrelated capture or slider.
bindCheckbox('forceCPU', config.global.forceCPU, (v) => {
  globalState.forceCPU = v;
  const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam);
  pushSettingsSyncToAllPhysical();
});
// Doesn't affect any already-computed camera state, just how the NEXT
// physical-camera frame gets scheduled -- no recomputeFromLastCapture call
// needed (unlike forceCPU above).
// Same story -- this only picks WHEN the display tail runs, never what it
// computes, so there is nothing to recompute on a flip. Turning it off leaves
// any already-posted mailbox slot to be drained normally (posting the payload
// and drainVisuals are both unconditional; only the caller in recomputeStages
// consults the flag), so a camera can't be stranded mid-repaint by a toggle.

// ── The profiler toggle ──────────────────────────────────────────────────
//
// WHAT THIS TOGGLE DOES, because it is much narrower than it used to be: it
// gates the DevTools MIRROR and nothing else. Spans themselves record
// unconditionally, into the one profiler store, and the per-module WebGPU
// timestamp queries this switch once armed are DELETED (they lived in the old
// pipeline's gpu/gpuTimeline.ts; the pose library now times every pass
// itself, unconditionally, and hands the raw nanoseconds back on the frame).
// The old comment here still described those queries and their
// 9.3ms-around-a-0.07ms-kernel failure, which had not been reachable from this
// checkbox for some time.
//
// Bound by hand rather than through bindCheckbox, and that is the whole point
// of it being here instead of one line above. bindCheckbox PERSISTS to
// localStorage and restores at bind time, which is correct for a setting and
// wrong for this: a performance.measure per span is real main-thread work
// inside whatever is being measured, so a checkbox that quietly survived a
// reload would tax every measurement taken afterwards, and every conclusion
// drawn from one. So it starts OFF on every load, always.
//
// Explicitly assigning `checked = false` is load-bearing, not belt-and-braces:
// browsers restore form control state across a plain reload on their own, so
// an absent `checked` attribute in the HTML is NOT enough to guarantee off.
const profilerCheckbox = document.getElementById('profilerEnabled') as HTMLInputElement;

function applyProfilerToggle() {
  const on = profilerCheckbox.checked;
  // Reset on the way ON, per session -- deliberately NOT per capture.
  // profilerReset also calls performance.clearMeasures(), so resetting each
  // reconstruction would delete earlier captures out of the DevTools recording
  // currently in progress, which is the exact opposite of "switch it on and
  // every following capture shows up". Turning it OFF leaves the records intact
  // so formatFlamechart() still has something to print.
  if (on) profilerReset();
  profilerSetDevToolsMirror(on);
}
profilerCheckbox.checked = false;
profilerCheckbox.addEventListener('change', applyProfilerToggle);
applyProfilerToggle();

// Sits at the TOP of the GPU toggle block now (it applies to all of them), so
// it can no longer say "the checkbox above".
gpuVotesStatus.textContent = isWebGPUSupported()
  ? 'WebGPU is available in this browser.'
  : 'WebGPU is not available in this browser -- every toggle below will silently fall back to the CPU pipeline.';
bindCheckbox('showGizmoBody', config.camera.common.showGizmoBody, (v) => { const cam = activeCamera(); if (cam) cam.settings.showGizmoBody = v; });
bindCheckbox('showRecoveredFloor', config.camera.common.showRecoveredFloor, (v) => { const cam = activeCamera(); if (cam) cam.settings.showRecoveredFloor = v; });
bindSlider('recoveredFloorOpacity', config.camera.common.recoveredFloorOpacity, (v) => { const cam = activeCamera(); if (cam) { cam.settings.recoveredFloorOpacity = v; cam.recoveredFloorOverlayMat.opacity = v; } }, (v) => v.toFixed(2));
// A full reconstruction, not just a reprojection, even though the pose does not
// depend on this. The window changes `lastProjectedBins`, and the World-view
// floor quad's GEOMETRY is built from that extent by applyRecoveredFloorOverlay
// -- which only runs on a fresh decode. Repainting the texture alone would leave
// the decal sized to the old window with the new crop stretched across it.
bindCheckbox('latticeBoundsProjection', config.camera.common.latticeBoundsProjection, (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.latticeBoundsProjection = v;
  recomputeFromLastCapture(cam);
});
bindSlider('gridPeriodPhaseBinCount', config.camera.common.gridPeriodPhaseBinCount, (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.gridPeriodPhaseBinCount = v;
  drawGridPeriodPhasePlot(cam);
}, (v) => v.toFixed(0));

// PURELY VISUAL, and currently not even that -- see below. The
// "feeds computeGridPeriodPhase (stage 7) directly" recompute that used to be
// here was a full pipeline re-run on every drag of a slider that reaches no
// stage at all: poseSettingsFor does not pass it, and GppSettings has no field
// for it. It went with the old pipeline, whose host-side period search read it.
//
// Its own doc comment still describes a filter on gridPeriodPhaseOverlays.ts's
// per-family median gaps, and that file does not read it either -- the plot is
// an empty state awaiting the intermediate readback. So this is a knob with no
// consumer, kept with the rest of that plot's shell rather than deleted
// piecemeal. Whether the shell stays is one decision, not seven.
bindSlider('gridPeriodPhaseGapLowerBound', config.camera.common.gridPeriodPhaseGapLowerBound, (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.gridPeriodPhaseGapLowerBound = v;
  drawGridPeriodPhasePlot(cam);
  pushSettingsIfPhysical();
}, (v) => v.toFixed(4));
bindSlider('simNoise', config.camera.simulated.simNoise, (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.simNoise = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => v.toFixed(0));
bindSlider('simBlur', config.camera.simulated.simBlur, (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.simBlur = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => v.toFixed(0));
bindSlider('captureSupersample', config.camera.simulated.captureSupersample, (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.captureSupersample = v; resizeCaptureBuffers(cam); runAxesReconstruction(cam); } }, (v) => `${v.toFixed(0)}x`);
bindRadioGroup('fieldView', config.camera.common.fieldView, (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.fieldView = v as FieldView;
  markCaptureDirty(cam);
  updateContaminationAvailability();
  updateGradientArrowAvailability();
  updateTopGradientAvailability();
  updateLsdAvailability();
});
updateContaminationAvailability();
updateGradientArrowAvailability();
updateTopGradientAvailability();
updateLsdAvailability();
bindSlider('gradientArrowScale', config.camera.common.gradientArrowScale, (v) => { const cam = activeCamera(); if (cam) cam.settings.gradientArrowScale = v; updateHoverOverlays(lastHoverClientX, lastHoverClientY); }, (v) => v.toFixed(1));
// The from-scratch traditional LSD pipeline (pose/stages/lsd/) --
// this DOES have a live debug overlay (the accepted/rejected/raw-region
// views), so every stage-2..5 change here recomputes and redraws it
// immediately.
function refreshLsd() {
  const cam = activeCamera(); if (!cam) return;
  updateLsdOverlay(cam); // handles its own full redraw (SVG rectangles + raster raw-regions/rejected) internally
}
// refreshLsd() updates the live-preview LSD debug overlay only (reads
// camera.lastNoisedPreviewGray) -- a separate concern from the PRODUCTION
// segments these same settings feed, so recomputeFromLastCapture is also needed
// here or the pose's own `composites` and `rects` go stale.
bindSlider('lsdToleranceDeg', config.camera.common.lsdToleranceDeg, (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdToleranceDeg = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => `${v.toFixed(1)}°`);
bindSlider('lsdRhoNoiseThreshold', config.camera.common.lsdRhoNoiseThreshold, (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdRhoNoiseThreshold = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(3));
bindSlider('lsdRhoHighThreshold', config.camera.common.lsdRhoHighThreshold, (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdRhoHighThreshold = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(3));
// 0 is the REAL algorithm (run grow to its fixpoint), not "off" -- labelled
// "auto" rather than "0" so the slider's left end does not read as a
// disabled/no-growth state.
//
// IT REACHED NOTHING UNTIL 2026-08-20: poseSettingsFor never passed it on, so
// encodeGrow took its own GROW_ROUNDS default and this slider moved a value
// that was persisted, pushed to the phone and stored in fixtures without ever
// changing a pixel. It is wired now, which makes it genuinely pose-moving --
// below convergence the labelling is half-grown, and the run says so through
// the growNotConverged status bit.
bindSlider('lsdCclSteps', config.camera.common.lsdCclSteps, (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdCclSteps = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => (v === 0 ? 'auto' : v.toFixed(0)));
bindSlider('lsdMinRegionSize', config.camera.common.lsdMinRegionSize, (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdMinRegionSize = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(0));
bindSlider('lsdNfaEpsilon', config.camera.common.lsdNfaEpsilon, (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdNfaEpsilon = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(2));
bindSlider('lsdNfaTestExponent', config.camera.common.lsdNfaTestExponent, (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdNfaTestExponent = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(0));
// Feeds pose/stages/votes/votes.ts's computeGradient2x2Composites (production)
// directly, not a live debug
// overlay, so recomputeFromLastCapture (not a repaint call) is what picks
// up the new value -- reusing the last capture rather than waiting for the
// next unrelated "capture now"/axesAutoCapture tick.
bindSlider('lsdMinLengthPx', config.camera.common.lsdMinLengthPx, (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdMinLengthPx = v; recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(0));

// ── S5c, the join ────────────────────────────────────────────────────────
//
// Six knobs and a button, all driving the pose. Every one of them takes the
// same reaction -- recompute from the last capture -- because the join runs on
// the device and its composites cannot be recoloured into existence.
//
// `joinRefresh` rather than `refreshLsd`: the LSD debug overlay draws the
// PRE-join rectangles, which none of these change.
function joinRefresh() {
  const cam = activeCamera(); if (!cam) return;
  toggleLineJoinBtn.classList.toggle('active', cam.settings.joinKSigma > 0);
  recomputeFromLastCapture(cam);
  pushSettingsIfPhysical();
}
bindSlider('joinKSigma', config.camera.common.joinKSigma, (v) => { const cam = activeCamera(); if (cam) cam.settings.joinKSigma = v; joinRefresh(); }, (v) => (v === 0 ? 'off' : v.toFixed(1)));
bindSlider('joinEndpointNoisePx', config.camera.common.joinEndpointNoisePx, (v) => { const cam = activeCamera(); if (cam) cam.settings.joinEndpointNoisePx = v; joinRefresh(); }, (v) => v.toFixed(2));
bindSlider('joinReachFrac', config.camera.common.joinReachFrac, (v) => { const cam = activeCamera(); if (cam) cam.settings.joinReachFrac = v; joinRefresh(); }, (v) => v.toFixed(2));
bindSlider('joinRounds', config.camera.common.joinRounds, (v) => { const cam = activeCamera(); if (cam) cam.settings.joinRounds = v; joinRefresh(); }, (v) => v.toFixed(0));
bindSlider('joinMaxResidualPx', config.camera.common.joinMaxResidualPx, (v) => { const cam = activeCamera(); if (cam) cam.settings.joinMaxResidualPx = v; joinRefresh(); }, (v) => v.toFixed(1));
bindCheckbox('joinPolarityAbs', config.camera.common.joinPolarityAbs, (v) => { const cam = activeCamera(); if (cam) cam.settings.joinPolarityAbs = v; joinRefresh(); });

// The one-click A/B, and it drives the SAME setting the kSigma slider does --
// which is why it lives here rather than with the overlay toggles in
// hoverDebugOverlays.ts. Switching off remembers the kSigma that was in force
// so switching back on restores it, instead of snapping to a hardcoded 3 and
// quietly discarding whatever the slider was set to. The remembered value is
// module state on purpose: it is the undo for a button press, not a
// configuration, and it should not survive a reload as one.
let lastJoinKSigma = config.camera.common.joinKSigma || 3;
toggleLineJoinBtn.addEventListener('click', () => {
  const cam = activeCamera(); if (!cam) return;
  if (cam.settings.joinKSigma > 0) lastJoinKSigma = cam.settings.joinKSigma;
  // Through setNum, not by assignment: the slider is the control of record, and
  // dispatching its own 'input' event is what keeps the thumb, the readout, the
  // setting, persistence and the recompute in one path instead of five.
  setPanelNum('joinKSigma', cam.settings.joinKSigma > 0 ? 0 : lastJoinKSigma);
});

bindCheckbox('showRecoveredPoles', config.camera.common.showRecoveredPoles, (v) => { const cam = activeCamera(); if (cam) cam.settings.showRecoveredPoles = v; });
// Turning either on refreshes immediately -- updateGradientCirclesDebug now
// skips its work while both are off (see its own comment), so the geometry
// sitting there when you flip one on could otherwise be stale until the
// next capture.
bindCheckbox('showAxisVectors', config.camera.common.showAxisVectors, (v) => { const cam = activeCamera(); if (cam) { cam.settings.showAxisVectors = v; if (v) updateGradientCirclesDebug(cam); } });
bindCheckbox('showTopCircles', config.camera.common.showTopCircles, (v) => { const cam = activeCamera(); if (cam) { cam.settings.showTopCircles = v; if (v) updateGradientCirclesDebug(cam); } });
bindSlider('topCirclesLineWidth', config.camera.common.topCirclesLineWidth, (v) => { const cam = activeCamera(); if (cam) { cam.settings.topCirclesLineWidth = v; updateGradientCirclesDebug(cam); } }, (v) => v.toFixed(1));
// Feeds projectSamplesCPU/buildDecodeSampleGrid (stages 9+10) -- recompute
// from the last capture rather than waiting for the next unrelated one.
bindSlider('minGrazingCos', config.camera.common.minGrazingCos, (v) => { const cam = activeCamera(); if (cam) { cam.settings.minGrazingCos = v; recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(2));
bindCheckbox('axesAutoCapture', config.camera.common.axesAutoCapture, (v) => { const cam = activeCamera(); if (cam) cam.settings.axesAutoCapture = v; });
bindSlider('axesCaptureInterval', config.camera.common.axesCaptureIntervalMs, (v) => { const cam = activeCamera(); if (cam) cam.settings.axesCaptureIntervalMs = v; }, (v) => `${v.toFixed(0)}`);

captureAxesBtn.addEventListener('click', () => { const cam = activeCamera(); if (cam) runAxesReconstruction(cam); });

