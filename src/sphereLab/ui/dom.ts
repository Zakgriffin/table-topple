import { Mode } from '../types.ts';

export const canvas = document.getElementById('gl') as HTMLCanvasElement;
export const panel = document.getElementById('panel') as HTMLDivElement;
export const panelToggle = document.getElementById('panelToggle') as HTMLButtonElement;
export const pipFrame = document.getElementById('pipFrame') as HTMLDivElement;
export const pipLabel = document.getElementById('pipLabel') as HTMLDivElement;
export const insideHint = document.getElementById('insideHint') as HTMLDivElement;
export const readout = document.getElementById('readout') as HTMLDivElement;
export const axesReadout = document.getElementById('axesReadout') as HTMLDivElement;
export const captureAxesBtn = document.getElementById('captureAxesBtn') as HTMLButtonElement;
export const positionReadout = document.getElementById('positionReadout') as HTMLDivElement;
export const marginalRightCanvas = document.getElementById('marginalRight') as HTMLCanvasElement;
export const marginalBottomCanvas = document.getElementById('marginalBottom') as HTMLCanvasElement;
export const marginalRightCtx = marginalRightCanvas.getContext('2d')!;
export const marginalBottomCtx = marginalBottomCanvas.getContext('2d')!;
export const sampleLatticeCanvas = document.getElementById('sampleLattice') as HTMLCanvasElement;
export const sampleLatticeCtx = sampleLatticeCanvas.getContext('2d')!;
export const contamToggles = document.getElementById('contamToggles') as HTMLDivElement;
export const toggleHideFieldBtn = document.getElementById('toggleHideField') as HTMLButtonElement;
export const toggleTrueContamBtn = document.getElementById('toggleTrueContam') as HTMLButtonElement;
export const toggleReconContamBtn = document.getElementById('toggleReconContam') as HTMLButtonElement;
export const gradientArrowCanvas = document.getElementById('gradientArrowOverlay') as HTMLCanvasElement;
export const gradientArrowCtx = gradientArrowCanvas.getContext('2d')!;
export const toggleGradientArrowBtn = document.getElementById('toggleGradientArrow') as HTMLButtonElement;
export const toggleGradientArrowModeBtn = document.getElementById('toggleGradientArrowMode') as HTMLButtonElement;
export const toggleTangentWalkPathBtn = document.getElementById('toggleTangentWalkPath') as HTMLButtonElement;
export const arrowToggles = document.getElementById('arrowToggles') as HTMLDivElement;
export const simDistortionSection = document.getElementById('simDistortionSection') as HTMLDivElement;

export const modeBtns: Record<Mode, HTMLButtonElement> = {
  world: document.getElementById('modeWorld') as HTMLButtonElement,
  through: document.getElementById('modeThrough') as HTMLButtonElement,
  inside: document.getElementById('modeInside') as HTMLButtonElement,
  projected: document.getElementById('modeProjected') as HTMLButtonElement,
};

// Persist every slider/checkbox under one localStorage key so a dev-server
// restart or a revisit doesn't reset the scene back to defaults.
export const STORAGE_KEY = 'sphereLab.controls';
export let savedControls: Record<string, string> = {};
try { savedControls = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}'); } catch { savedControls = {}; }
export function persistControl(id: string, value: string) {
  savedControls[id] = value;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(savedControls));
}

export function bindSlider(id: string, onChange: (v: number) => void, fmt: (v: number) => string = (v) => v.toFixed(1)) {
  const input = document.getElementById(id) as HTMLInputElement;
  const val = document.getElementById(id + 'Val') as HTMLSpanElement;
  if (savedControls[id] !== undefined) input.value = savedControls[id];
  const apply = () => { const v = parseFloat(input.value); val.textContent = fmt(v); onChange(v); persistControl(id, input.value); };
  input.addEventListener('input', apply);
  apply();
}

export function bindCheckbox(id: string, onChange: (v: boolean) => void) {
  const input = document.getElementById(id) as HTMLInputElement;
  if (savedControls[id] !== undefined) input.checked = savedControls[id] === '1';
  const apply = () => { onChange(input.checked); persistControl(id, input.checked ? '1' : '0'); };
  input.addEventListener('change', apply);
  apply();
}

export function bindRadioGroup(name: string, onChange: (v: string) => void) {
  const inputs = Array.from(document.getElementsByName(name)) as HTMLInputElement[];
  // Only honor a saved value if it still matches one of the CURRENT options --
  // otherwise a renamed/removed option value (e.g. an old 'normal' after this
  // group's options changed) would leave every input unchecked instead of
  // falling back to the HTML's own default `checked` attribute.
  if (savedControls[name] !== undefined && inputs.some((inp) => inp.value === savedControls[name])) {
    for (const inp of inputs) inp.checked = inp.value === savedControls[name];
  }
  const apply = () => {
    const checked = inputs.find((inp) => inp.checked);
    if (!checked) return;
    onChange(checked.value);
    persistControl(name, checked.value);
  };
  for (const inp of inputs) inp.addEventListener('change', apply);
  apply();
}

export function setSectionHidden(el: HTMLElement, hidden: boolean) {
  el.classList.toggle('hidden', hidden);
}

export const globalSettingsSectionEl = document.getElementById('globalSettingsSection') as HTMLDivElement;
export const cameraSettingsSectionsEl = document.getElementById('cameraSettingsSections') as HTMLDivElement;
export const simCameraDetailFields = document.getElementById('simCameraDetailFields') as HTMLDivElement;
export const physCameraDetailFields = document.getElementById('physCameraDetailFields') as HTMLDivElement;
export const simOnlyFieldViews = document.getElementById('simOnlyFieldViews') as HTMLDivElement;
export const fieldViewRawLabel = document.getElementById('fieldViewRawLabel') as HTMLSpanElement;
export const physCaptureModeReadout = document.getElementById('physCaptureModeReadout') as HTMLSpanElement;


export const cameraTabsEl = document.getElementById('cameraTabs') as HTMLDivElement;
