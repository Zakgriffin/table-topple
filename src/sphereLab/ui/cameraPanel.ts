import { addSimulatedCamera, removeCameraTab, selectGlobalTab } from '../camera/lifecycle.ts';
import { PhysicalCamera } from '../camera/model.ts';
import { activeCamera, activeCameraId, cameras, isPhysical, isSimulated, setActiveCameraId } from '../camera/store.ts';
import { pushSettingsSync, sendToDevBridge } from '../devBridge/client.ts';
import { rebuildGridLineKs } from '../math/geometry.ts';
import { updateContaminationAvailability } from '../overlays/contaminationOverlays.ts';
import { updateTopGradientAvailability, updateTopGradientOverlay } from '../overlays/gradientHighlightOverlays.ts';
import { lastHoverClientX, lastHoverClientY, updateGradientArrowAvailability, updateHoverOverlays } from '../overlays/hoverDebugOverlays.ts';
import { updateLsdAvailability, updateLsdOverlay } from '../overlays/lsdOverlay.ts';
import { updateGradientCirclesDebug } from '../overlays/sphereOverlays.ts';
import { drawGridPeriodPhasePlot } from '../overlays/gridPeriodPhaseOverlays.ts';
import { recomputeFromLastCapture, runAxesReconstruction, updateChainTransfersReadout } from '../pipeline/axesReconstruction.ts';
import { markCaptureDirty, resizeCaptureBuffers } from '../pipeline/capture.ts';
import { buildProjectedTexture } from '../pipeline/decodeGrid.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { isWebGPUSupported } from '../pipelineGPU/device.ts';
import { getRoots, profilerEnabled, profilerReset, profilerSetEnabled } from '../profiling/profiler.ts';
import { invalidateHashTableCache } from '../pipelineGPU/decodeTally.ts';
import { invalidateTorusBufferCache } from '../pipelineGPU/positionLM.ts';
import { rebuildFloorPattern, rebuildFloorTexture } from '../scene/floor.ts';
import { globalState } from '../state.ts';
import { FieldView } from '../types.ts';
import { bindCheckbox, bindRadioGroup, bindSlider, savedControls, cameraSettingsSectionsEl, cameraTabsEl, captureAxesBtn, fieldViewRawLabel, globalSettingsSectionEl, gpuVotesStatus, useGPUDecodeRow, physCameraDetailFields, physCaptureModeReadout, profilerStatus, setSectionHidden, simCameraDetailFields, simDistortionSection, simOnlyFieldViews, toggleCompositeLineFamiliesBtn, toggleDistinctnessCurveBtn, toggleGapHistogramBtn, toggleGradientArrowBtn, toggleProductCurveBtn, toggleHideFieldBtn, toggleLevelLineArrowBtn, toggleLsdCompositeBtn, toggleLsdRawRegionsBtn, toggleLsdRejectedBtn, toggleLsdSegmentsBtn, toggleReconContamBtn, toggleTopGradientBtn, toggleTrueCardinalOrientationBtn, toggleTrueContamBtn, toggleValueHistogramBtn } from './dom.ts';
import { layoutPip } from './layout.ts';

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
  // accent-color from --cam-accent (see sphere-lab.html's CSS -- falls back
  // to the fixed green everywhere else, e.g. the Global tab's own controls,
  // which aren't tied to any one camera) -- setting it once here, on the
  // shared container, is enough for every control inside to pick it up via
  // ordinary CSS inheritance, no per-control wiring needed.
  cameraSettingsSectionsEl.style.setProperty('--cam-accent', `#${cam.color.getHexString()}`);

  setSectionHidden(simCameraDetailFields, !isSimulated(cam));
  setSectionHidden(physCameraDetailFields, isSimulated(cam));
  setSectionHidden(simDistortionSection, !isSimulated(cam));
  setSectionHidden(simOnlyFieldViews, !isSimulated(cam));
  fieldViewRawLabel.textContent = isSimulated(cam) ? 'raw (no blur, no noise)' : 'capture';

  const setNum = (id: string, v: number) => {
    const el = document.getElementById(id) as HTMLInputElement | null;
    if (!el) return;
    el.value = String(v);
    el.dispatchEvent(new Event('input'));
  };
  const setBool = (id: string, v: boolean) => {
    const el = document.getElementById(id) as HTMLInputElement | null;
    if (!el) return;
    el.checked = v;
    el.dispatchEvent(new Event('change'));
  };

  if (isSimulated(cam)) {
    setNum('camX', cam.settings.camX); setNum('camY', cam.settings.camY); setNum('camZ', cam.settings.camZ);
    setNum('camYaw', cam.settings.camYawDeg); setNum('camPitch', cam.settings.camPitchDeg); setNum('camFov', cam.settings.horizFovDeg);
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
  setBool('showSampleLattice', cam.settings.showSampleLattice);
  setNum('gridPeriodPhaseBinCount', cam.settings.gridPeriodPhaseBinCount);
  setNum('gridPeriodPhaseGapLowerBound', cam.settings.gridPeriodPhaseGapLowerBound);

  setNum('coherenceRadius', cam.settings.coherenceRadius);

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
  setNum('lsdMaxRetries', cam.settings.lsdMaxRetries);
  setNum('lsdRetryToleranceFactor', cam.settings.lsdRetryToleranceFactor);
  setNum('lsdRetryShrinkFraction', cam.settings.lsdRetryShrinkFraction);
  setNum('lsdJoinSteps', cam.settings.lsdJoinSteps);
  setNum('lsdMergeMinSimilarity', cam.settings.lsdMergeMinSimilarity);
  setNum('lsdMaxTravelFactor', cam.settings.lsdMaxTravelFactor);
  setNum('lsdMinLengthPx', cam.settings.lsdMinLengthPx);
  setBool('showRecoveredPoles', cam.settings.showRecoveredPoles); setBool('showAxisVectors', cam.settings.showAxisVectors);
  setBool('showTopCircles', cam.settings.showTopCircles);
  setNum('topCirclesLineWidth', cam.settings.topCirclesLineWidth);
  setNum('weightSharpenPower', cam.settings.weightSharpenPower);
  setBool('useWorldVoteOrientation', cam.settings.useWorldVoteOrientation);
  setNum('worldVoteRefineSteps', cam.settings.worldVoteRefineSteps);
  setNum('minGrazingCos', cam.settings.minGrazingCos);
  setBool('axesAutoCapture', cam.settings.axesAutoCapture);
  setNum('axesCaptureInterval', cam.settings.axesCaptureIntervalMs);

  toggleHideFieldBtn.classList.toggle('active', cam.settings.hideField);
  toggleTrueContamBtn.classList.toggle('active', cam.settings.showTrueContamination);
  toggleReconContamBtn.classList.toggle('active', cam.settings.showReconstructedContamination);
  toggleTrueCardinalOrientationBtn.classList.toggle('active', cam.settings.useTrueCardinalOrientation);
  toggleGradientArrowBtn.classList.toggle('active', cam.settings.showGradientArrow);
  toggleLevelLineArrowBtn.classList.toggle('active', cam.settings.showLevelLineArrow);
  toggleTopGradientBtn.classList.toggle('active', cam.settings.showTopGradient);
  toggleLsdSegmentsBtn.classList.toggle('active', cam.settings.showLsdSegments);
  toggleLsdRejectedBtn.classList.toggle('active', cam.settings.showLsdRejected);
  toggleLsdRawRegionsBtn.classList.toggle('active', cam.settings.showLsdRawRegions);
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
  if (globalState.mode === 'projected') buildProjectedTexture(cam);
  markCaptureDirty(cam);
  layoutPip(cam);
  drawGridPeriodPhasePlot(cam);
}


export function rerunOnRealCaptureSettingChange() {
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
// Per-camera-settings sliders (the 16 fields making up PoseComputeState.settings,
// see pipeline/poseCompute.ts) push a fresh settingsSync to THAT camera's own
// phone only -- source of truth stays the desktop's own sliders (see this
// session's on-device-pose-recovery plan). No-op for a simulated camera
// (nothing to push to) or a camera not currently active.
function pushSettingsIfPhysical() {
  const cam = activeCamera();
  if (cam && isPhysical(cam)) pushSettingsSync(cam);
}
export let realCaptureFovRerunTimer: number | undefined;
bindSlider('realCaptureFovDeg', (v) => {
  const cam = activeCamera();
  if (!cam || !isPhysical(cam)) return;
  // refreshCameraPanel's own setNum re-syncs this exact slider to whatever
  // the camera's CURRENT horizFovDeg already is every time the panel
  // redraws (e.g. a plain tab switch) -- that dispatches this same 'input'
  // event even though nothing changed, which used to unconditionally
  // re-schedule a recompute below. Only worth doing on a genuine change.
  const changed = v !== cam.settings.horizFovDeg;
  cam.settings.horizFovDeg = v;
  markCaptureDirty(cam);
  pushSettingsSync(cam);
  if (!changed) return;
  clearTimeout(realCaptureFovRerunTimer);
  realCaptureFovRerunTimer = window.setTimeout(rerunOnRealCaptureSettingChange, 200);
}, (v) => `${v.toFixed(0)}°`);
// Tier 1 (invalidates the capture itself -- see this session's chat on
// "every setting recomputes everything downstream of it"): markCaptureDirty
// still drives the passive preview loop, runAxesReconstruction is the real
// recompute -- it self-throttles via camera.axesCapturing, so a fast drag
// just drops overlapping calls rather than queueing them, same as an
// overlapping "capture now" click.
bindSlider('camX', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camX = v; markCaptureDirty(cam); runAxesReconstruction(cam); } });
bindSlider('camY', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camY = v; markCaptureDirty(cam); runAxesReconstruction(cam); } });
bindSlider('camZ', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camZ = v; markCaptureDirty(cam); runAxesReconstruction(cam); } });
bindSlider('camYaw', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camYawDeg = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('camPitch', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camPitchDeg = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('camFov', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.horizFovDeg = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => `${v.toFixed(0)}°`);

export let syncingViewportAspect = false;
export function clampViewport(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.round(v)));
}
bindSlider('viewportW', (v) => {
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
bindSlider('viewportH', (v) => {
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
bindCheckbox('aspectLocked', (v) => { const cam = activeCamera(); if (cam) cam.settings.aspectLocked = v; });

bindCheckbox('showSphere', (v) => { const cam = activeCamera(); if (cam) cam.settings.showSphere = v; });
bindCheckbox('showCircles', (v) => { const cam = activeCamera(); if (cam) cam.settings.showCircles = v; });
bindCheckbox('showPoles', (v) => { const cam = activeCamera(); if (cam) cam.settings.showPoles = v; });
bindCheckbox('showFrustum', (v) => { const cam = activeCamera(); if (cam) cam.settings.showFrustum = v; });
bindCheckbox('showPatch', (v) => { const cam = activeCamera(); if (cam) cam.settings.showPatch = v; });
bindCheckbox('showFloor', (v) => { globalState.showFloor = v; });
bindSlider('floorCellOutlineSubdiv', (v) => {
  globalState.floorCellOutlineSubdiv = v;
  rebuildFloorTexture();
  for (const cam of cameras.values()) markCaptureDirty(cam); // this IS the real rendered floor, so every camera's capture path needs to re-render too
}, (v) => v.toFixed(0));
// Global fields (this slider, and the four useGPU* checkboxes below except
// useGPUProject -- see this session's on-device-pose-recovery plan) push to
// EVERY connected physical camera's phone, not just the active one --
// conceptually shared even though their recompute-trigger only targets
// activeCamera(). useGPUProject is deliberately excluded: it only affects
// computeProjectedBinsAndMarginalsAuto (display-only projection), which is
// NOT part of the phone-portable pose pipeline (pipeline/poseCompute.ts).
function pushSettingsSyncToAllPhysical() {
  for (const cam of cameras.values()) if (isPhysical(cam)) pushSettingsSync(cam);
}
// One-time migration off the old 256 default, and it has to run BEFORE the
// binding below. bindSlider restores whatever was last persisted (dom.ts's
// savedControls, localStorage), which on any browser that has opened this page
// before would silently override the new 144 default the moment it loads --
// and a stale 256 doesn't just mis-size the desktop floor, pushSettingsSync
// below sends it straight down to the phone, where it re-crops the decode
// table and pulls the AR board off the printed one.
//
// Only the EXACT old default is dropped, so a board size somebody actually
// chose is left alone. Safe to delete once no browser is still carrying a 256.
if (savedControls.boardSize === '256') delete savedControls.boardSize;

bindSlider('boardSize', (v) => {
  globalState.boardSize = v;
  rebuildFloorPattern(v); // re-crops the torus, rebuilds the decode lookup table, resizes the floor mesh/texture/reference lines
  rebuildGridLineKs(); // reads HALF_R/HALF_C, which rebuildFloorPattern just updated -- must run after it
  invalidateHashTableCache(); // GPU decode-tally's hash table was built from the OLD debruijnLookup
  invalidateTorusBufferCache(); // GPU Phase 3's torus-brightness buffer was built from the OLD torus
  for (const cam of cameras.values()) markCaptureDirty(cam); // this IS the real rendered floor, so every camera's capture path needs to re-render/re-decode against the new board
  pushSettingsSyncToAllPhysical();
}, (v) => v.toFixed(0));
// Global (not per-camera) toggles selecting the code path for the
// gradient/fit/project/decode stages -- flipping one changes what the NEXT
// recompute produces, so re-run it against whichever camera's on screen
// immediately rather than waiting for the next unrelated capture/slider.
bindCheckbox('useGPUFit', (v) => { globalState.useGPUFit = v; const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam); pushSettingsSyncToAllPhysical(); });
bindCheckbox('useGPUDecode', (v) => { globalState.useGPUDecode = v; const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam); pushSettingsSyncToAllPhysical(); });
bindCheckbox('useGPUProject', (v) => { globalState.useGPUProject = v; const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam); });
bindCheckbox('useGPUGradient', (v) => { globalState.useGPUGradient = v; const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam); pushSettingsSyncToAllPhysical(); });
// Wired like every other useGPU* toggle now. It used to be deliberately
// unwired -- forced unchecked, no change handler -- because the GPU path was
// pinned off for a CORRECTNESS reason (a mod-pi/directed NFA parity mismatch)
// and bindCheckbox would have restored a stale `checked` from localStorage and
// silently re-enabled a wrong path. That mismatch is resolved and the kernel is
// verified identical to CPU (pipelineGPU/lsdFitVerify.ts), so this is now an
// ordinary A/B toggle. It defaulted OFF for a while after that, on performance
// grounds -- the GPU path was upload-bound and slower in isolation -- and now
// defaults ON, because it stopped being upload-bound once its neighbours could
// hand it their buffers. Its own compute never changed.
bindCheckbox('useGPULsdFit', (v) => {
  globalState.useGPULsdFit = v;
  const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam);
  pushSettingsSyncToAllPhysical();
});
// Dispatches independently of useGPULsdFit -- stage 2+3 vs stage 4+5, either
// can be on GPU without the other. Defaults OFF for a correctness reason, not
// a perf one (see state.ts): this is the one stage that cannot be bit-identical
// to CPU. The debug round scrubber in overlays/lsdOverlay.ts stays on the CPU
// grower regardless, since it needs the intermediate rounds this path skips.
// Supersedes useGPUDecode's grid upload when on -- the packed grid is built on
// device and the tally reads it in place, so it never crosses the bus. Output is
// observably identical to the CPU route (verified: 0 bit diffs, matching winner
// and consistency), so this is a pure perf toggle, not a behavioural one.
bindCheckbox('useGPUDecodeFused', (v) => {
  globalState.useGPUDecodeFused = v;
  updateDecodeTallyActive();
  const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam);
  pushSettingsSyncToAllPhysical();
});
// Only meaningful with useGPUGrowRegions on -- it consumes that path's own
// device buffers. It DOES win now, having been off for a long time on the
// grounds that it merely traded a label readback for a CSR readback. What
// changed is stage 1: with the gradient device-resident, the CPU collect needs
// fx/fy pulled back down, so leaving this off costs three readbacks rather than
// one. See state.ts for the numbers.
bindCheckbox('useGPUCollectRegions', (v) => {
  globalState.useGPUCollectRegions = v;
  const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam);
  pushSettingsSyncToAllPhysical();
});
// An ISLAND -- gridPeriodPhase.ts's coarse sweep shares no buffers with the LSD
// chain, so this composes with nothing and can be flipped in isolation.
bindCheckbox('useGPUPeriodSweep', (v) => {
  globalState.useGPUPeriodSweep = v;
  const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam);
  pushSettingsSyncToAllPhysical();
});
bindCheckbox('useGPUGrowRegions', (v) => {
  globalState.useGPUGrowRegions = v;
  const cam = activeCamera(); if (cam) recomputeFromLastCapture(cam);
  pushSettingsSyncToAllPhysical();
});
// Doesn't affect any already-computed camera state, just how the NEXT
// physical-camera frame gets scheduled -- no recomputeFromLastCapture call
// needed (unlike the useGPU* toggles above).
bindCheckbox('useCapturePipelining', (v) => { globalState.useCapturePipelining = v; });
// Same story -- this only picks WHEN the display tail runs, never what it
// computes, so there is nothing to recompute on a flip. Turning it off leaves
// any already-posted mailbox slot to be drained normally (markVisualsDirty and
// drainVisuals are both unconditional; only the caller in recomputeStages
// consults the flag), so a camera can't be stranded mid-repaint by a toggle.
bindCheckbox('useDeferredVisuals', (v) => { globalState.useDeferredVisuals = v; });

// ── The profiler toggle ──────────────────────────────────────────────────
//
// Bound by hand rather than through bindCheckbox, and that is the whole point
// of it being here instead of one line above. bindCheckbox PERSISTS to
// localStorage and restores at bind time, which is correct for a setting and
// wrong for this: profiling is the one switch on this panel that changes what
// the pipeline COSTS rather than what it computes. It turns on WebGPU
// timestamp queries in fitPlanes/decodeTally whose resolve pays its own
// mapAsync -- the recorded case is a `GPU dispatch` span reading 9.3ms around
// a kernel that actually ran for 0.07ms. A checkbox that quietly survived a
// reload would tax every measurement taken afterwards, and every conclusion
// drawn from one. So it starts OFF on every load, always.
//
// Explicitly assigning `checked = false` is load-bearing, not belt-and-braces:
// browsers restore form control state across a plain reload on their own, so
// an absent `checked` attribute in the HTML is NOT enough to guarantee off.
const profilerCheckbox = document.getElementById('profilerEnabled') as HTMLInputElement;

// Called every frame from animate() so the capture count stays live, which is
// why it memoizes on the rendered string rather than writing unconditionally:
// a textContent write per frame WHILE PROFILING would add main-thread work to
// the very recording being taken. Reduces to one call and one string compare
// on frames where nothing changed, which is nearly all of them.
let lastProfilerStatusText = '';
export function updateProfilerStatus() {
  let text: string;
  if (!profilerEnabled()) {
    text = 'Profiler off. On: spans mirror into DevTools’ Timings track, and GPU dispatch spans inflate.';
  } else {
    // ROOT SPANS, not captures -- with deferred visuals on, one capture leaves
    // three (axesReconstruction, then the drain's projectBins and
    // poleMarkers+overlays, which are roots precisely BECAUSE the tail no
    // longer runs inside the reconstruction). Calling them captures would
    // misreport the thing this readout exists to show.
    const n = getRoots().length;
    text = n === 0
      ? 'Recording. Start the DevTools capture FIRST, then take one here.'
      : `Recording -- ${n} root span${n === 1 ? '' : 's'}. formatFlamechart() for the text form.`;
  }
  if (text === lastProfilerStatusText) return;
  lastProfilerStatusText = text;
  profilerStatus.textContent = text;
}

function applyProfilerToggle() {
  const on = profilerCheckbox.checked;
  // Reset on the way ON, per session -- deliberately NOT per capture.
  // profilerReset also calls performance.clearMeasures(), so resetting each
  // reconstruction would delete earlier captures out of the DevTools recording
  // currently in progress, which is the exact opposite of "switch it on and
  // every following capture shows up". Turning it OFF leaves the tree intact so
  // formatFlamechart() still has something to print.
  if (on) profilerReset();
  profilerSetEnabled(on);
  updateProfilerStatus();
}
profilerCheckbox.checked = false;
profilerCheckbox.addEventListener('change', applyProfilerToggle);
applyProfilerToggle();
// The fused decode path is tested FIRST in runPositionDecode and returns on
// success, so while it is on, useGPUDecode selects between two tally
// implementations that never run. Dimming says that, where the nesting alone
// would suggest the opposite (a child usually needs its parent). Deliberately
// only a visual state: the checkbox stays bound and clickable, because the way
// you make it relevant again is to turn the OTHER one off, and a dead row would
// hide that. Falls back gracefully -- if fused bails at runtime the tally
// toggle does decide the fallback, which is a good reason not to lock it.
function updateDecodeTallyActive() {
  useGPUDecodeRow.classList.toggle('inactive', globalState.useGPUDecodeFused);
}
updateDecodeTallyActive();

// Sits at the TOP of the GPU toggle block now (it applies to all of them), so
// it can no longer say "the checkbox above".
gpuVotesStatus.textContent = isWebGPUSupported()
  ? 'WebGPU is available in this browser.'
  : 'WebGPU is not available in this browser -- every toggle below will silently fall back to the CPU pipeline.';
bindCheckbox('showGizmoBody', (v) => { const cam = activeCamera(); if (cam) cam.settings.showGizmoBody = v; });
bindCheckbox('showRecoveredFloor', (v) => { const cam = activeCamera(); if (cam) cam.settings.showRecoveredFloor = v; });
bindSlider('recoveredFloorOpacity', (v) => { const cam = activeCamera(); if (cam) { cam.settings.recoveredFloorOpacity = v; cam.recoveredFloorOverlayMat.opacity = v; } }, (v) => v.toFixed(2));
bindCheckbox('showSampleLattice', (v) => { const cam = activeCamera(); if (cam) cam.settings.showSampleLattice = v; });
bindSlider('gridPeriodPhaseBinCount', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.gridPeriodPhaseBinCount = v;
  drawGridPeriodPhasePlot(cam);
}, (v) => v.toFixed(0));

bindSlider('gridPeriodPhaseGapLowerBound', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.gridPeriodPhaseGapLowerBound = v;
  drawGridPeriodPhasePlot(cam);
  recomputeFromLastCapture(cam); // feeds computeGridPeriodPhase (stage 7) directly
  pushSettingsIfPhysical();
}, (v) => v.toFixed(4));
bindSlider('simNoise', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.simNoise = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => v.toFixed(0));
bindSlider('simBlur', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.simBlur = v; markCaptureDirty(cam); runAxesReconstruction(cam); } }, (v) => v.toFixed(0));
bindSlider('captureSupersample', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.captureSupersample = v; resizeCaptureBuffers(cam); runAxesReconstruction(cam); } }, (v) => `${v.toFixed(0)}x`);
bindSlider('coherenceRadius', (v) => { const cam = activeCamera(); if (cam) { cam.settings.coherenceRadius = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindRadioGroup('fieldView', (v) => {
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
bindSlider('gradientArrowScale', (v) => { const cam = activeCamera(); if (cam) cam.settings.gradientArrowScale = v; updateHoverOverlays(lastHoverClientX, lastHoverClientY); }, (v) => v.toFixed(1));
// The from-scratch traditional LSD pipeline (pipeline/lsdSegments.ts) --
// this DOES have a live debug overlay (the accepted/rejected/raw-region
// views), so every stage-2..5 change here recomputes and redraws it
// immediately.
function refreshLsd() {
  const cam = activeCamera(); if (!cam) return;
  updateLsdOverlay(cam); // handles its own full redraw (SVG rectangles + raster raw-regions/rejected) internally
}
// refreshLsd() updates the live-preview LSD debug overlay only (reads
// camera.lastNoisedPreviewGray) -- a separate, already-correct concern from
// the PRODUCTION composite lines these same settings feed (stage 3, see
// pipeline/votes.ts's computeGradient2x2Composites), so recomputeFromLastCapture
// is also needed here or camera.lastVoteComposites/lastGridPeriodPhase go stale.
bindSlider('lsdToleranceDeg', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdToleranceDeg = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => `${v.toFixed(1)}°`);
bindSlider('lsdRhoNoiseThreshold', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdRhoNoiseThreshold = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(3));
bindSlider('lsdRhoHighThreshold', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdRhoHighThreshold = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(3));
// 0 is the REAL algorithm (run growRegionsCCL to its fixpoint), not "off" --
// labelled "auto" rather than "0" so the slider's own left end doesn't read as
// a disabled/no-growth state the way the grow-steps slider it replaces did.
bindSlider('lsdCclSteps', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdCclSteps = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => (v === 0 ? 'auto' : v.toFixed(0)));
bindSlider('lsdMinRegionSize', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdMinRegionSize = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(0));
bindSlider('lsdNfaEpsilon', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdNfaEpsilon = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(2));
bindSlider('lsdNfaTestExponent', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdNfaTestExponent = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(0));
// Still BOUND, though their inputs carry `disabled` (see sphere-lab.html): the
// stage-5 retry loop is retired and nothing reads these three, but keeping the
// bindings means the readouts still render the persisted values instead of
// showing blank, and un-disabling the inputs is all it takes to bring the
// retired fitRegionWithRetries back for a comparison.
bindSlider('lsdMaxRetries', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdMaxRetries = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(0));
bindSlider('lsdRetryToleranceFactor', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdRetryToleranceFactor = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(2));
bindSlider('lsdRetryShrinkFraction', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdRetryShrinkFraction = v; refreshLsd(); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(2));
// The join walk's own 4 params -- feed pipeline/votes.ts's
// computeGradient2x2Composites (production) directly, not a live debug
// overlay, so recomputeFromLastCapture (not a repaint call) is what picks
// up the new value -- reusing the last capture rather than waiting for the
// next unrelated "capture now"/axesAutoCapture tick.
bindSlider('lsdJoinSteps', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdJoinSteps = v; recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(0));
bindSlider('lsdMergeMinSimilarity', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdMergeMinSimilarity = v; recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(2));
bindSlider('lsdMaxTravelFactor', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdMaxTravelFactor = v; recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(1));
bindSlider('lsdMinLengthPx', (v) => { const cam = activeCamera(); if (cam) { cam.settings.lsdMinLengthPx = v; recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(0));
bindCheckbox('showRecoveredPoles', (v) => { const cam = activeCamera(); if (cam) cam.settings.showRecoveredPoles = v; });
// Turning either on refreshes immediately -- updateGradientCirclesDebug now
// skips its work while both are off (see its own comment), so the geometry
// sitting there when you flip one on could otherwise be stale until the
// next capture.
bindCheckbox('showAxisVectors', (v) => { const cam = activeCamera(); if (cam) { cam.settings.showAxisVectors = v; if (v) updateGradientCirclesDebug(cam); } });
bindCheckbox('showTopCircles', (v) => { const cam = activeCamera(); if (cam) { cam.settings.showTopCircles = v; if (v) updateGradientCirclesDebug(cam); } });
bindSlider('topCirclesLineWidth', (v) => { const cam = activeCamera(); if (cam) { cam.settings.topCirclesLineWidth = v; updateGradientCirclesDebug(cam); } }, (v) => v.toFixed(1));
bindSlider('weightSharpenPower', (v) => { const cam = activeCamera(); if (cam) { cam.settings.weightSharpenPower = v; updateGradientCirclesDebug(cam); recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(1));
bindCheckbox('useWorldVoteOrientation', (v) => { const cam = activeCamera(); if (cam) { cam.settings.useWorldVoteOrientation = v; recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); });
bindSlider('worldVoteRefineSteps', (v) => { const cam = activeCamera(); if (cam) { cam.settings.worldVoteRefineSteps = v; recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(0));
// Feeds projectSamplesCPU/buildDecodeSampleGrid (stages 9+10) -- recompute
// from the last capture rather than waiting for the next unrelated one.
bindSlider('minGrazingCos', (v) => { const cam = activeCamera(); if (cam) { cam.settings.minGrazingCos = v; recomputeFromLastCapture(cam); } pushSettingsIfPhysical(); }, (v) => v.toFixed(2));
bindCheckbox('axesAutoCapture', (v) => { const cam = activeCamera(); if (cam) cam.settings.axesAutoCapture = v; });
bindSlider('axesCaptureInterval', (v) => { const cam = activeCamera(); if (cam) cam.settings.axesCaptureIntervalMs = v; }, (v) => `${v.toFixed(0)}`);

captureAxesBtn.addEventListener('click', () => { const cam = activeCamera(); if (cam) runAxesReconstruction(cam); });

