import { persistConfig } from '../config.ts';
import { type Mode } from '../types.ts';

export const canvas = document.getElementById('gl') as HTMLCanvasElement;
export const saveConfigBtn = document.getElementById('saveConfigBtn') as HTMLButtonElement;
export const loadConfigBtn = document.getElementById('loadConfigBtn') as HTMLButtonElement;
export const configStatus = document.getElementById('configStatus') as HTMLDivElement;
export const throughCamCanvas = document.getElementById('throughCamCanvas') as HTMLCanvasElement;
export const throughCamCtx = throughCamCanvas.getContext('2d')!;
export const panel = document.getElementById('panel') as HTMLDivElement;
export const panelToggle = document.getElementById('panelToggle') as HTMLButtonElement;
export const pipFrame = document.getElementById('pipFrame') as HTMLDivElement;
export const pipLabel = document.getElementById('pipLabel') as HTMLDivElement;
export const insideHint = document.getElementById('insideHint') as HTMLDivElement;
export const readout = document.getElementById('readout') as HTMLDivElement;
export const axesReadout = document.getElementById('axesReadout') as HTMLDivElement;
export const captureAxesBtn = document.getElementById('captureAxesBtn') as HTMLButtonElement;
export const positionReadout = document.getElementById('positionReadout') as HTMLDivElement;
export const gridPeriodPhasePlotSvg = document.getElementById('gridPeriodPhasePlot') as unknown as SVGSVGElement;
export const toggleGapHistogramBtn = document.getElementById('toggleGapHistogram') as HTMLButtonElement;
export const toggleValueHistogramBtn = document.getElementById('toggleValueHistogram') as HTMLButtonElement;
export const toggleDistinctnessCurveBtn = document.getElementById('toggleDistinctnessCurve') as HTMLButtonElement;
export const toggleProductCurveBtn = document.getElementById('toggleProductCurve') as HTMLButtonElement;
// The Projected-Cam overlay surface. Same shape as lsdSvgOverlay below -- a
// full-viewport SVG whose children are positioned in screen coordinates -- so
// the two views' overlays are written the same way. It replaced a canvas that
// was moved and resized on every animation frame.
export const projectedSvgOverlay = document.getElementById('projectedSvgOverlay') as unknown as SVGSVGElement;
export const sampleLatticeGroup = document.getElementById('sampleLatticeGroup') as unknown as SVGGElement;
export const overlayPanel = document.getElementById('overlayPanel') as HTMLDivElement;
export const overlayPanelToggle = document.getElementById('overlayPanelToggle') as HTMLButtonElement;
export const contamToggles = document.getElementById('contamToggles') as HTMLDivElement;
export const toggleHideFieldBtn = document.getElementById('toggleHideField') as HTMLButtonElement;
export const toggleTrueContamBtn = document.getElementById('toggleTrueContam') as HTMLButtonElement;
export const toggleReconContamBtn = document.getElementById('toggleReconContam') as HTMLButtonElement;
export const toggleTopGradientBtn = document.getElementById('toggleTopGradient') as HTMLButtonElement;
export const toggleLsdSegmentsBtn = document.getElementById('toggleLsdSegments') as HTMLButtonElement;
export const toggleLsdRejectedBtn = document.getElementById('toggleLsdRejected') as HTMLButtonElement;
export const toggleLsdRawRegionsBtn = document.getElementById('toggleLsdRawRegions') as HTMLButtonElement;
export const toggleLsdCompositeBtn = document.getElementById('toggleLsdComposite') as HTMLButtonElement;
export const toggleCompositeLineFamiliesBtn = document.getElementById('toggleCompositeLineFamilies') as HTMLButtonElement;
export const lsdReadout = document.getElementById('lsdReadout') as HTMLDivElement;
export const lsdSvgOverlay = document.getElementById('lsdSvgOverlay') as unknown as SVGSVGElement;
export const lsdRectanglesGroup = document.getElementById('lsdRectanglesGroup') as unknown as SVGGElement;
export const lsdCompositeGroup = document.getElementById('lsdCompositeGroup') as unknown as SVGGElement;
export const gradientArrowGroup = document.getElementById('gradientArrowGroup') as unknown as SVGGElement;
export const levelLineArrowGroup = document.getElementById('levelLineArrowGroup') as unknown as SVGGElement;
export const growthCandidateGroup = document.getElementById('growthCandidateGroup') as unknown as SVGGElement;
export const toggleGradientArrowBtn = document.getElementById('toggleGradientArrow') as HTMLButtonElement;
export const toggleLevelLineArrowBtn = document.getElementById('toggleLevelLineArrow') as HTMLButtonElement;
export const arrowToggles = document.getElementById('arrowToggles') as HTMLDivElement;
export const projectedToggles = document.getElementById('projectedToggles') as HTMLDivElement;
export const toggleTrueCardinalOrientationBtn = document.getElementById('toggleTrueCardinalOrientation') as HTMLButtonElement;
export const toggleSampleLatticeBtn = document.getElementById('toggleSampleLattice') as HTMLButtonElement;
export const simDistortionSection = document.getElementById('simDistortionSection') as HTMLDivElement;

export const modeBtns: Record<Mode, HTMLButtonElement> = {
  world: document.getElementById('modeWorld') as HTMLButtonElement,
  through: document.getElementById('modeThrough') as HTMLButtonElement,
  inside: document.getElementById('modeInside') as HTMLButtonElement,
  projected: document.getElementById('modeProjected') as HTMLButtonElement,
};

// ── Control binding ──────────────────────────────────────────────────────
//
// Every binder takes its INITIAL VALUE as an argument rather than reading one
// from the DOM. The controls in sphere-lab.html carry no `value=`/`checked`
// attribute anymore -- config.ts owns every default -- so `input.value` on a
// range with no attribute is not "the default", it is the HTML spec's
// midpoint of min and max, which is nobody's intended setting.
//
// Passing it at the call site rather than looking it up by element id is also
// what let the id -> setting mapping stop being a guess. It never held: the
// LSD overlay booleans persisted under their BUTTON's id (`toggleLsdSegments`)
// while the setting was `showLsdSegments`, `camFov`/`realCaptureFovDeg` are two
// controls for one `horizFovDeg`, and `camYaw` is `camYawDeg`. Now the binding
// names the field it drives, right next to the callback that writes it.
//
// Persistence is no longer per-control either. Each change writes the WHOLE
// config object (see config.ts's persistConfig), because a control's value is
// only ever a view onto a field in it.

export function bindSlider(id: string, initial: number, onChange: (v: number) => void, fmt: (v: number) => string = (v) => v.toFixed(1)) {
  const input = document.getElementById(id) as HTMLInputElement;
  const val = document.getElementById(id + 'Val') as HTMLSpanElement;
  input.value = String(initial);
  const apply = () => { const v = parseFloat(input.value); val.textContent = fmt(v); onChange(v); persistConfig(); };
  input.addEventListener('input', apply);
  apply();
}

export function bindCheckbox(id: string, initial: boolean, onChange: (v: boolean) => void) {
  const input = document.getElementById(id) as HTMLInputElement;
  input.checked = initial;
  const apply = () => { onChange(input.checked); persistConfig(); };
  input.addEventListener('change', apply);
  apply();
}

export function bindRadioGroup(name: string, initial: string, onChange: (v: string) => void) {
  const inputs = Array.from(document.getElementsByName(name)) as HTMLInputElement[];
  // A stored value that no longer matches any option (a renamed or deleted
  // field view, say) would leave every radio unchecked -- and there is no
  // `checked` attribute in the HTML to fall back to anymore -- so fall back to
  // the first option, which is at least a real one.
  const chosen = inputs.some((inp) => inp.value === initial) ? initial : inputs[0]?.value;
  for (const inp of inputs) inp.checked = inp.value === chosen;
  const apply = () => {
    const checked = inputs.find((inp) => inp.checked);
    if (!checked) return;
    onChange(checked.value);
    persistConfig();
  };
  for (const inp of inputs) inp.addEventListener('change', apply);
  apply();
}

export function setSectionHidden(el: HTMLElement, hidden: boolean) {
  el.classList.toggle('hidden', hidden);
}

// LSD rectangles/composite lines are real DOM elements (an <svg>'s
// children), not a Three.js quad gated by the per-frame mode branch in
// main.ts's animate() -- unlike the raster overlays (rendered only inside
// the `mode === 'through'` branch, so they naturally stop drawing the
// instant mode changes), nothing else clears this SVG content on a mode
// switch, so it would otherwise sit on screen indefinitely on top of
// World/Projected/Inside-Sphere too. Call from setMode alongside
// overlays/hoverDebugOverlays.ts's own clearArrowOverlays (the gradient/
// level-line arrow groups live in this same <svg> now, but are cleared by
// that sibling function instead of this one -- see its own comment for why).
export function clearLsdSvgOverlay() {
  while (lsdRectanglesGroup.firstChild) lsdRectanglesGroup.removeChild(lsdRectanglesGroup.firstChild);
  while (lsdCompositeGroup.firstChild) lsdCompositeGroup.removeChild(lsdCompositeGroup.firstChild);
  while (growthCandidateGroup.firstChild) growthCandidateGroup.removeChild(growthCandidateGroup.firstChild);
}

export const globalSettingsSectionEl = document.getElementById('globalSettingsSection') as HTMLDivElement;
export const gpuVotesStatus = document.getElementById('gpuVotesStatus') as HTMLDivElement;
export const lsdChainTransfers = document.getElementById('lsdChainTransfers') as HTMLDivElement;
export const cameraSettingsSectionsEl = document.getElementById('cameraSettingsSections') as HTMLDivElement;
export const simCameraDetailFields = document.getElementById('simCameraDetailFields') as HTMLDivElement;
export const physCameraDetailFields = document.getElementById('physCameraDetailFields') as HTMLDivElement;
export const simOnlyFieldViews = document.getElementById('simOnlyFieldViews') as HTMLDivElement;
export const fieldViewRawLabel = document.getElementById('fieldViewRawLabel') as HTMLSpanElement;
export const physCaptureModeReadout = document.getElementById('physCaptureModeReadout') as HTMLSpanElement;


export const cameraTabsEl = document.getElementById('cameraTabs') as HTMLDivElement;
