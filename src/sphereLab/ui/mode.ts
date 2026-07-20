import { activeCamera } from '../camera/store.ts';
import { updateContaminationOverlays } from '../overlays/contaminationOverlays.ts';
import { clearGradientArrowOverlay } from '../overlays/hoverDebugOverlays.ts';
import { hideMarginalLines, hideSampleLattice } from '../overlays/projectedCamOverlays.ts';
import { buildProjectedTexture } from '../pipeline/decodeGrid.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { worldOrbit } from '../scene/viewerControls.ts';
import { globalState } from '../state.ts';
import { Mode } from '../types.ts';
import { arrowToggles, contamToggles, insideHint, modeBtns, panel, panelToggle, persistControl, pipFrame, pipLabel, savedControls } from './dom.ts';

// ── Mode switching ───────────────────────────────────────────────────────

export function setMode(m: Mode) {
  globalState.mode = m;
  persistControl('mode', m);
  for (const k of Object.keys(modeBtns) as Mode[]) modeBtns[k].classList.toggle('active', k === m);
  worldOrbit.enabled = m === 'world';
  insideHint.style.display = m === 'inside' ? 'block' : 'none';
  pipFrame.style.display = m === 'through' || m === 'projected' ? 'none' : 'block';
  pipLabel.style.display = m === 'through' || m === 'projected' ? 'none' : 'block';
  const cam = activeCamera();
  if (m === 'projected') { if (cam) buildProjectedTexture(cam); }
  else { hideMarginalLines(); hideSampleLattice(); }
  contamToggles.style.display = m === 'through' ? 'flex' : 'none';
  arrowToggles.style.display = m === 'through' ? 'flex' : 'none';
  if (m !== 'through') clearGradientArrowOverlay();
  if (m === 'through' && cam) { updateDistortedPreview(cam); updateContaminationOverlays(cam); }
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
