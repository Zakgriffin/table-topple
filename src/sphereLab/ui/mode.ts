import { activeCamera } from '../camera/store.ts';
import { clearArrowOverlays } from '../overlays/hoverDebugOverlays.ts';
import { hideGridPeriodPhaseProjected } from '../overlays/gridPeriodPhaseOverlays.ts';
import { refreshModeVisualizations } from '../pipeline/modeRefresh.ts';
import { worldOrbit } from '../scene/viewerControls.ts';
import { globalState } from '../state.ts';
import { Mode } from '../types.ts';
import { arrowToggles, clearLsdSvgOverlay, contamToggles, insideHint, modeBtns, overlayPanel, overlayPanelToggle, panel, panelToggle, persistControl, pipFrame, pipLabel, projectedToggles, savedControls, setSectionHidden, throughCamCanvas } from './dom.ts';

// ── Mode switching ───────────────────────────────────────────────────────

export function setMode(m: Mode) {
  globalState.mode = m;
  persistControl('mode', m);
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
  // The overlay panel itself only has content for through/projected -- kept
  // entirely off-screen for world/inside rather than showing an empty card.
  setSectionHidden(overlayPanel, m !== 'through' && m !== 'projected');
  overlayPanelToggle.style.display = m === 'through' || m === 'projected' ? '' : 'none';
  setSectionHidden(contamToggles, m !== 'through');
  setSectionHidden(arrowToggles, m !== 'through');
  setSectionHidden(projectedToggles, m !== 'projected');
  if (m !== 'through') { clearArrowOverlays(); clearLsdSvgOverlay(); }
  // Whatever this mode actually renders may be stale from whenever it was
  // last computed -- see pipeline/modeRefresh.ts.
  if (cam) refreshModeVisualizations(cam, m);
}
modeBtns.world.addEventListener('click', () => setMode('world'));
modeBtns.through.addEventListener('click', () => setMode('through'));
modeBtns.inside.addEventListener('click', () => setMode('inside'));
modeBtns.projected.addEventListener('click', () => setMode('projected'));

export function setPanelCollapsed(collapsed: boolean) {
  panel.classList.toggle('collapsed', collapsed);
  panelToggle.classList.toggle('collapsed', collapsed);
  panelToggle.textContent = collapsed ? '›' : '‹';
  persistControl('panelCollapsed', collapsed ? '1' : '0');
}
panelToggle.addEventListener('click', () => setPanelCollapsed(!panel.classList.contains('collapsed')));
setPanelCollapsed(savedControls['panelCollapsed'] === '1');

export function setOverlayPanelCollapsed(collapsed: boolean) {
  overlayPanel.classList.toggle('collapsed', collapsed);
  overlayPanelToggle.classList.toggle('collapsed', collapsed);
  overlayPanelToggle.textContent = collapsed ? '‹' : '›';
  persistControl('overlayPanelCollapsed', collapsed ? '1' : '0');
}
overlayPanelToggle.addEventListener('click', () => setOverlayPanelCollapsed(!overlayPanel.classList.contains('collapsed')));
setOverlayPanelCollapsed(savedControls['overlayPanelCollapsed'] === '1');
