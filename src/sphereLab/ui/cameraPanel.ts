import { addSimulatedCamera, removeCameraTab, selectGlobalTab } from '../camera/lifecycle.ts';
import { PhysicalCamera } from '../camera/model.ts';
import { activeCamera, activeCameraId, cameras, isPhysical, isSimulated, setActiveCameraId } from '../camera/store.ts';
import { sendToDevBridge } from '../devBridge/client.ts';
import { rebuildGridLineKs } from '../math/geometry.ts';
import { updateBucketFillAvailability, updateBucketFillOverlay } from '../overlays/bucketFillOverlay.ts';
import { updateBucketFillCompositeAvailability, updateBucketFillJoinAvailability, updateBucketFillJoinOverlay, updateBucketFillMergeMarkersAvailability } from '../overlays/bucketFillJoinOverlay.ts';
import { updateContaminationAvailability } from '../overlays/contaminationOverlays.ts';
import { updateTopGradientAvailability, updateTopGradientOverlay } from '../overlays/gradientHighlightOverlays.ts';
import { lastHoverClientX, lastHoverClientY, updateGradientArrowAvailability, updateHoverOverlays } from '../overlays/hoverDebugOverlays.ts';
import { updateGradientCirclesDebug } from '../overlays/sphereOverlays.ts';
import { drawGridPeriodPhasePlot } from '../overlays/gridPeriodPhaseOverlays.ts';
import { runAxesReconstruction } from '../pipeline/axesReconstruction.ts';
import { markCaptureDirty, resizeCaptureBuffers } from '../pipeline/capture.ts';
import { buildProjectedTexture } from '../pipeline/decodeGrid.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { isWebGPUSupported } from '../pipelineGPU/device.ts';
import { invalidateHashTableCache } from '../pipelineGPU/decodeTally.ts';
import { invalidateTorusBufferCache } from '../pipelineGPU/positionLM.ts';
import { rebuildFloorPattern, rebuildFloorTexture } from '../scene/floor.ts';
import { globalState } from '../state.ts';
import { FieldView } from '../types.ts';
import { bindCheckbox, bindRadioGroup, bindSlider, cameraSettingsSectionsEl, cameraTabsEl, captureAxesBtn, fieldViewRawLabel, globalSettingsSectionEl, gpuVotesStatus, physCameraDetailFields, physCaptureModeReadout, setSectionHidden, simCameraDetailFields, simDistortionSection, simOnlyFieldViews, toggleBucketFillBtn, toggleBucketFillCompositeBtn, toggleBucketFillJoinBtn, toggleBucketFillMarkersBtn, toggleBucketFillMergeMarkersBtn, toggleGradientArrowBtn, toggleGradientArrowModeBtn, toggleHideFieldBtn, toggleReconContamBtn, toggleTopGradientBtn, toggleTrueContamBtn } from './dom.ts';
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
  setBool('showRecoveredFloor', cam.settings.showRecoveredFloor); setBool('showSampleLattice', cam.settings.showSampleLattice);
  setBool('showNewSampleLattice', cam.settings.showNewSampleLattice);
  setBool('showGridPeriodPhaseDebug', cam.settings.showGridPeriodPhaseDebug);
  setNum('gridPeriodPhaseBinCount', cam.settings.gridPeriodPhaseBinCount);
  setBool('showCompositeLineFamilies', cam.settings.showCompositeLineFamilies);

  setNum('simGradRadius', cam.settings.simGradRadius); setNum('coherenceRadius', cam.settings.coherenceRadius);
  setNum('tangentWalkMaxSteps', cam.settings.tangentWalkMaxSteps); setNum('tangentWalkDeviationDeg', cam.settings.tangentWalkDeviationDeg);
  setNum('tangentWalkMagFraction', cam.settings.tangentWalkMagFraction); setNum('tangentWalkGraceSamples', cam.settings.tangentWalkGraceSamples);
  setBool('tangentWalkAdaptive', cam.settings.tangentWalkAdaptive);

  const fieldViewId = 'fieldView' + cam.settings.fieldView[0].toUpperCase() + cam.settings.fieldView.slice(1);
  const fieldViewInput = document.getElementById(fieldViewId) as HTMLInputElement | null;
  if (fieldViewInput) { fieldViewInput.checked = true; fieldViewInput.dispatchEvent(new Event('change')); }

  setNum('gradientArrowScale', cam.settings.gradientArrowScale);
  setNum('bucketFillToleranceDeg', cam.settings.bucketFillToleranceDeg);
  setNum('bucketFillMagnitudeThreshold', cam.settings.bucketFillMagnitudeThreshold);
  setNum('bucketFillMinLengthPx', cam.settings.bucketFillMinLengthPx);
  setNum('bucketFillJoinSteps', cam.settings.bucketFillJoinSteps);
  setNum('bucketFillMergeMinSimilarity', cam.settings.bucketFillMergeMinSimilarity);
  setNum('bucketFillMaxTravelFactor', cam.settings.bucketFillMaxTravelFactor);
  setBool('showRecoveredPoles', cam.settings.showRecoveredPoles); setBool('showAxisVectors', cam.settings.showAxisVectors);
  setBool('showTopCircles', cam.settings.showTopCircles);
  setNum('topCirclesLineWidth', cam.settings.topCirclesLineWidth);
  setNum('weightSharpenPower', cam.settings.weightSharpenPower);
  setNum('minGrazingCos', cam.settings.minGrazingCos);
  setBool('axesAutoCapture', cam.settings.axesAutoCapture);
  setNum('axesCaptureInterval', cam.settings.axesCaptureIntervalMs);

  toggleHideFieldBtn.classList.toggle('active', cam.settings.hideField);
  toggleTrueContamBtn.classList.toggle('active', cam.settings.showTrueContamination);
  toggleReconContamBtn.classList.toggle('active', cam.settings.showReconstructedContamination);
  toggleGradientArrowBtn.classList.toggle('active', cam.settings.showGradientArrow);
  toggleGradientArrowModeBtn.classList.toggle('active', cam.settings.showGradientArrowPerpendicular);
  toggleTopGradientBtn.classList.toggle('active', cam.settings.showTopGradient);
  toggleBucketFillBtn.classList.toggle('active', cam.settings.showBucketFillSegments);
  toggleBucketFillMarkersBtn.classList.toggle('active', cam.settings.showBucketFillMarkers);
  toggleBucketFillJoinBtn.classList.toggle('active', cam.settings.showBucketFillJoin);
  toggleBucketFillCompositeBtn.classList.toggle('active', cam.settings.showBucketFillComposite);
  toggleBucketFillMergeMarkersBtn.classList.toggle('active', cam.settings.showBucketFillMergeMarkers);
  updateContaminationAvailability();
  updateGradientArrowAvailability();
  updateTopGradientAvailability();
  updateBucketFillAvailability();
  updateBucketFillJoinAvailability();
  updateBucketFillCompositeAvailability();
  updateBucketFillMergeMarkersAvailability();

  updateDistortedPreview(cam);
  if (globalState.mode === 'projected') buildProjectedTexture(cam);
  markCaptureDirty(cam);
  layoutPip(cam);
  drawGridPeriodPhasePlot(cam);
}


export function rerunOnRealCaptureSettingChange() {
  const cam = activeCamera();
  if (cam && isPhysical(cam) && cam.lastRealCaptureGray) runAxesReconstruction(cam);
}
export let realCaptureFovRerunTimer: number | undefined;
bindSlider('realCaptureFovDeg', (v) => {
  const cam = activeCamera();
  if (!cam || !isPhysical(cam)) return;
  cam.settings.horizFovDeg = v;
  markCaptureDirty(cam);
  clearTimeout(realCaptureFovRerunTimer);
  realCaptureFovRerunTimer = window.setTimeout(rerunOnRealCaptureSettingChange, 200);
}, (v) => `${v.toFixed(0)}°`);
bindSlider('camX', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camX = v; markCaptureDirty(cam); } });
bindSlider('camY', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camY = v; markCaptureDirty(cam); } });
bindSlider('camZ', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camZ = v; markCaptureDirty(cam); } });
bindSlider('camYaw', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camYawDeg = v; markCaptureDirty(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('camPitch', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.camPitchDeg = v; markCaptureDirty(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('camFov', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.horizFovDeg = v; markCaptureDirty(cam); } }, (v) => `${v.toFixed(0)}°`);

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
bindSlider('boardSize', (v) => {
  globalState.boardSize = v;
  rebuildFloorPattern(v); // re-crops the torus, rebuilds the decode lookup table, resizes the floor mesh/texture/reference lines
  rebuildGridLineKs(); // reads HALF_R/HALF_C, which rebuildFloorPattern just updated -- must run after it
  invalidateHashTableCache(); // GPU decode-tally's hash table was built from the OLD debruijnLookup
  invalidateTorusBufferCache(); // GPU Phase 3's torus-brightness buffer was built from the OLD torus
  for (const cam of cameras.values()) markCaptureDirty(cam); // this IS the real rendered floor, so every camera's capture path needs to re-render/re-decode against the new board
}, (v) => v.toFixed(0));
bindCheckbox('useGPUFit', (v) => { globalState.useGPUFit = v; });
bindCheckbox('useGPUDecode', (v) => { globalState.useGPUDecode = v; });
bindCheckbox('useGPUProject', (v) => { globalState.useGPUProject = v; });
gpuVotesStatus.textContent = isWebGPUSupported()
  ? 'WebGPU is available in this browser.'
  : 'WebGPU is not available in this browser -- the checkbox above will silently fall back to the CPU pipeline.';
bindCheckbox('showGizmoBody', (v) => { const cam = activeCamera(); if (cam) cam.settings.showGizmoBody = v; });
bindCheckbox('showRecoveredFloor', (v) => { const cam = activeCamera(); if (cam) cam.settings.showRecoveredFloor = v; });
bindCheckbox('showSampleLattice', (v) => { const cam = activeCamera(); if (cam) cam.settings.showSampleLattice = v; });
bindCheckbox('showNewSampleLattice', (v) => { const cam = activeCamera(); if (cam) cam.settings.showNewSampleLattice = v; });
bindCheckbox('showGridPeriodPhaseDebug', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showGridPeriodPhaseDebug = v;
  drawGridPeriodPhasePlot(cam);
});
bindSlider('gridPeriodPhaseBinCount', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.gridPeriodPhaseBinCount = v;
  drawGridPeriodPhasePlot(cam);
}, (v) => v.toFixed(0));
bindCheckbox('showCompositeLineFamilies', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.showCompositeLineFamilies = v;
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
bindSlider('simNoise', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.simNoise = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('simBlur', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.simBlur = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('simGradRadius', (v) => { const cam = activeCamera(); if (cam) { cam.settings.simGradRadius = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('captureSupersample', (v) => { const cam = activeCamera(); if (cam && isSimulated(cam)) { cam.settings.captureSupersample = v; resizeCaptureBuffers(cam); } }, (v) => `${v.toFixed(0)}x`);
bindSlider('coherenceRadius', (v) => { const cam = activeCamera(); if (cam) { cam.settings.coherenceRadius = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('tangentWalkMaxSteps', (v) => { const cam = activeCamera(); if (cam) { cam.settings.tangentWalkMaxSteps = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindSlider('tangentWalkDeviationDeg', (v) => { const cam = activeCamera(); if (cam) { cam.settings.tangentWalkDeviationDeg = v; markCaptureDirty(cam); } }, (v) => `${v.toFixed(0)}°`);
bindSlider('tangentWalkMagFraction', (v) => { const cam = activeCamera(); if (cam) { cam.settings.tangentWalkMagFraction = v; markCaptureDirty(cam); } }, (v) => v.toFixed(2));
bindSlider('tangentWalkGraceSamples', (v) => { const cam = activeCamera(); if (cam) { cam.settings.tangentWalkGraceSamples = v; markCaptureDirty(cam); } }, (v) => v.toFixed(0));
bindCheckbox('tangentWalkAdaptive', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.tangentWalkAdaptive = v;
  markCaptureDirty(cam);
  updateHoverOverlays(lastHoverClientX, lastHoverClientY);
});
bindRadioGroup('fieldView', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.fieldView = v as FieldView;
  markCaptureDirty(cam);
  updateContaminationAvailability();
  updateGradientArrowAvailability();
  updateTopGradientAvailability();
  updateBucketFillAvailability();
  updateBucketFillJoinAvailability();
  updateBucketFillCompositeAvailability();
  updateBucketFillMergeMarkersAvailability();
});
updateContaminationAvailability();
updateGradientArrowAvailability();
updateTopGradientAvailability();
updateBucketFillAvailability();
updateBucketFillJoinAvailability();
updateBucketFillCompositeAvailability();
updateBucketFillMergeMarkersAvailability();
bindSlider('gradientArrowScale', (v) => { const cam = activeCamera(); if (cam) cam.settings.gradientArrowScale = v; updateHoverOverlays(lastHoverClientX, lastHoverClientY); }, (v) => v.toFixed(1));
bindSlider('bucketFillToleranceDeg', (v) => { const cam = activeCamera(); if (cam) { cam.settings.bucketFillToleranceDeg = v; updateBucketFillOverlay(cam); updateHoverOverlays(lastHoverClientX, lastHoverClientY); } }, (v) => `${v.toFixed(1)}°`);
bindSlider('bucketFillMagnitudeThreshold', (v) => { const cam = activeCamera(); if (cam) { cam.settings.bucketFillMagnitudeThreshold = v; updateBucketFillOverlay(cam); updateHoverOverlays(lastHoverClientX, lastHoverClientY); } }, (v) => v.toFixed(1));
bindSlider('bucketFillMinLengthPx', (v) => {
  const cam = activeCamera(); if (!cam) return;
  cam.settings.bucketFillMinLengthPx = v;
  updateBucketFillOverlay(cam); // base raster: filtered segments disappear from it too
  updateBucketFillJoinOverlay(cam); // join walk: filtered segments never get fronts
  updateHoverOverlays(lastHoverClientX, lastHoverClientY); // endpoint markers
}, (v) => v.toFixed(0));
bindSlider('bucketFillJoinSteps', (v) => { const cam = activeCamera(); if (cam) { cam.settings.bucketFillJoinSteps = v; updateBucketFillJoinOverlay(cam); updateHoverOverlays(lastHoverClientX, lastHoverClientY); } }, (v) => v.toFixed(0));
bindSlider('bucketFillMergeMinSimilarity', (v) => { const cam = activeCamera(); if (cam) { cam.settings.bucketFillMergeMinSimilarity = v; updateBucketFillJoinOverlay(cam); updateHoverOverlays(lastHoverClientX, lastHoverClientY); } }, (v) => v.toFixed(2));
bindSlider('bucketFillMaxTravelFactor', (v) => { const cam = activeCamera(); if (cam) { cam.settings.bucketFillMaxTravelFactor = v; updateBucketFillJoinOverlay(cam); updateHoverOverlays(lastHoverClientX, lastHoverClientY); } }, (v) => v.toFixed(1));
bindCheckbox('showRecoveredPoles', (v) => { const cam = activeCamera(); if (cam) cam.settings.showRecoveredPoles = v; });
// Turning either on refreshes immediately -- updateGradientCirclesDebug now
// skips its work while both are off (see its own comment), so the geometry
// sitting there when you flip one on could otherwise be stale until the
// next capture.
bindCheckbox('showAxisVectors', (v) => { const cam = activeCamera(); if (cam) { cam.settings.showAxisVectors = v; if (v) updateGradientCirclesDebug(cam); } });
bindCheckbox('showTopCircles', (v) => { const cam = activeCamera(); if (cam) { cam.settings.showTopCircles = v; if (v) updateGradientCirclesDebug(cam); } });
bindSlider('topCirclesLineWidth', (v) => { const cam = activeCamera(); if (cam) { cam.settings.topCirclesLineWidth = v; updateGradientCirclesDebug(cam); } }, (v) => v.toFixed(1));
bindSlider('weightSharpenPower', (v) => { const cam = activeCamera(); if (cam) { cam.settings.weightSharpenPower = v; updateGradientCirclesDebug(cam); } }, (v) => v.toFixed(1));
// Only takes effect on the next capture (feeds projectSamplesCPU/
// buildDecodeSampleGrid inside runAxesReconstruction, not any live preview).
bindSlider('minGrazingCos', (v) => { const cam = activeCamera(); if (cam) cam.settings.minGrazingCos = v; }, (v) => v.toFixed(2));
bindCheckbox('axesAutoCapture', (v) => { const cam = activeCamera(); if (cam) cam.settings.axesAutoCapture = v; });
bindSlider('axesCaptureInterval', (v) => { const cam = activeCamera(); if (cam) cam.settings.axesCaptureIntervalMs = v; }, (v) => `${v.toFixed(0)}`);

captureAxesBtn.addEventListener('click', () => { const cam = activeCamera(); if (cam) runAxesReconstruction(cam); });

