import { activeCamera } from '../camera/store.ts';
import { clearArrowOverlays } from '../overlays/hoverDebugOverlays.ts';
import { hideGridPeriodPhaseProjected } from '../overlays/gridPeriodPhaseOverlays.ts';
import { refreshModeVisualizations } from '../pipeline/modeRefresh.ts';
import { worldOrbit } from '../scene/viewerControls.ts';
import { persistConfig } from '../../shared/config.ts';
import { globalState } from '../../shared/state.ts';
import { type Mode } from '../../shared/types.ts';
import { bindCheckbox, clearLsdSvgOverlay, insideHint, modeBtns, overlayPaneTitle, overlayPanel, overlayPanelToggle, panel, panelToggle, pipFrame, pipLabel, setSectionHidden, throughCamCanvas } from './dom.ts';

// ── THE RIGHT PANE: one chronological list, filtered to a view ───────────
//
// The pane's markup is ONE list of sections in pipeline order, and each section
// declares which views it is worth looking at in, via `data-views`. A view's
// "shown" list is therefore a QUERY over that markup rather than a second list
// living here -- the two-lists-drifting failure this codebase keeps paying for.
// `data-views="world inside"` is how the two 3D views share one list: they
// render the same scene from two cameras.
//
// Two behaviours, one function, so a mode switch and the filter switch cannot
// disagree about what the pane should look like.
//
// FILTERED (the default): only this view's sections, so the pane stays short
// enough to read as one list.
//
// UNFILTERED: the whole pipeline in order, and every off-view section gets a TAG
// on its heading naming the view it belongs to. It is NOT dimmed, and that is a
// correction rather than a preference: these controls are fully live -- setting
// another view up before switching to it is the whole point -- and opacity is
// already spoken for in this panel, where `.section.disabled`, `button:disabled`
// and `.chk.inactive` all use it for "you cannot use this". Dimmed live controls
// read as broken ones.
export function refreshOverlayPaneVisibility() {
  const m = globalState.mode;
  const only = globalState.overlayPaneThisViewOnly;
  // Straight off the mode BUTTON's own text, so the title cannot drift from the
  // control that selects it -- there is no second table of display names.
  overlayPaneTitle.textContent = modeBtns[m].textContent;
  for (const el of overlayPanel.querySelectorAll<HTMLElement>('[data-views]')) {
    const views = el.dataset.views!.split(' ');
    const isCurrent = views.includes(m);
    setSectionHidden(el, only && !isCurrent);
    // The tag is built here rather than written into the markup 13 times: it is
    // derived from `data-views`, so it cannot disagree with the attribute that
    // actually decides visibility.
    const h2 = el.querySelector('h2');
    if (!h2) continue;
    let tag = h2.querySelector<HTMLSpanElement>('.viewTag');
    if (!tag) {
      tag = document.createElement('span');
      tag.className = 'viewTag';
      h2.prepend(tag); // prepended because it floats right -- a float must precede its line
    }
    tag.textContent = only || isCurrent ? '' : views.join(' / ');
  }
}

// ── Mode switching ───────────────────────────────────────────────────────

export function setMode(m: Mode) {
  globalState.mode = m;
  persistConfig();
  for (const k of Object.keys(modeBtns) as Mode[]) modeBtns[k].classList.toggle('active', k === m);
  worldOrbit.enabled = m === 'world';
  insideHint.style.display = m === 'inside' ? 'block' : 'none';
  // Through-Cam's own dedicated 2D canvas (see scene/throughCam2D.ts) --
  // only actually drawn into inside animate()'s 'through' branch, but
  // still needs hiding on every OTHER mode switch, same as the WebGL quads
  // it replaced used to stop drawing automatically just by that branch not
  // running.
  throughCamCanvas.style.display = m === 'through' ? 'block' : 'none';
  pipFrame.style.display = m === 'through' || m === 'projected' ? 'none' : 'block';
  pipLabel.style.display = m === 'through' || m === 'projected' ? 'none' : 'block';
  const cam = activeCamera();
  if (m !== 'projected') hideGridPeriodPhaseProjected();
  // Every mode has content now, so the panel is never hidden outright -- it was,
  // for world/inside, back when the only overlays that existed were Through-Cam
  // ones.
  refreshOverlayPaneVisibility();
  if (m !== 'through') { clearArrowOverlays(); clearLsdSvgOverlay(); }
  // Whatever this mode actually renders may be stale from whenever it was
  // last computed -- see pipeline/modeRefresh.ts.
  if (cam) refreshModeVisualizations(cam, m);
}
modeBtns.world.addEventListener('click', () => setMode('world'));
modeBtns.through.addEventListener('click', () => setMode('through'));
modeBtns.inside.addEventListener('click', () => setMode('inside'));
modeBtns.projected.addEventListener('click', () => setMode('projected'));

// Panel chrome, bound here rather than in ui/cameraPanel.ts with the sliders:
// it selects which of the right pane's view groups are on screen and touches no
// camera, no setting a camera owns and no pipeline. Binding it there would also
// mean cameraPanel importing this module, and the arrows currently run the
// other way.
bindCheckbox('overlayPaneThisViewOnly', globalState.overlayPaneThisViewOnly, (v) => {
  globalState.overlayPaneThisViewOnly = v;
  refreshOverlayPaneVisibility();
});

export function setPanelCollapsed(collapsed: boolean) {
  panel.classList.toggle('collapsed', collapsed);
  panelToggle.classList.toggle('collapsed', collapsed);
  panelToggle.textContent = collapsed ? '›' : '‹';
  globalState.panelCollapsed = collapsed;
  persistConfig();
}
panelToggle.addEventListener('click', () => setPanelCollapsed(!panel.classList.contains('collapsed')));
setPanelCollapsed(globalState.panelCollapsed);

function setOverlayPanelCollapsed(collapsed: boolean) {
  overlayPanel.classList.toggle('collapsed', collapsed);
  overlayPanelToggle.classList.toggle('collapsed', collapsed);
  overlayPanelToggle.textContent = collapsed ? '‹' : '›';
  globalState.overlayPanelCollapsed = collapsed;
  persistConfig();
}
overlayPanelToggle.addEventListener('click', () => setOverlayPanelCollapsed(!overlayPanel.classList.contains('collapsed')));
setOverlayPanelCollapsed(globalState.overlayPanelCollapsed);
