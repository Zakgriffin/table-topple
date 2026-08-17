// ── Every element this page touches, resolved once ───────────────────────
//
// The LEAF of this client's module graph: it imports nothing from its siblings,
// so it can never be part of a cycle and can never be the module that is still
// initializing when someone reads it. That matters more here than the tidiness
// does -- see main.ts's header on the temporal-dead-zone hazard this page
// already hit once as a single file.
//
// Mirrors the desktop's ui/dom.ts in shape, deliberately, so the two clients'
// pages read the same way.

export const video = document.getElementById('v') as HTMLVideoElement;
export const captureCanvas = document.getElementById('captureCanvas') as HTMLCanvasElement;
export const captureCtx = captureCanvas.getContext('2d')!;
// The AR overlay's own transparent canvas, stacked directly over captureCanvas
// and carrying the identical letterbox fit -- see client/overlay.ts, which owns
// the THREE renderer that draws into it.
export const arCanvas = document.getElementById('arCanvas') as HTMLCanvasElement;
export const camStatus = document.getElementById('camStatus')!;
export const relayStatus = document.getElementById('relayStatus')!;

export const zoomSlider = document.getElementById('zoom') as HTMLInputElement;
export const resSelect = document.getElementById('resSelect') as HTMLSelectElement;
export const detectResBtn = document.getElementById('detectResBtn') as HTMLButtonElement;
export const probeLogToggleBtn = document.getElementById('probeLogToggleBtn') as HTMLButtonElement;
export const probeLogCloseBtn = document.getElementById('probeLogCloseBtn') as HTMLButtonElement;
export const probeLogPanel = document.getElementById('probeLogPanel') as HTMLDivElement;
export const probeLogTbody = document.getElementById('probeLogTbody') as HTMLTableSectionElement;
export const probeLogMatchHeader = document.getElementById('probeLogMatchHeader') as HTMLTableCellElement;
export const fpsSlider = document.getElementById('fps') as HTMLInputElement;
export const fpsValue = document.getElementById('fpsValue')!;
export const switchCamBtn = document.getElementById('switchCam') as HTMLButtonElement;
export const shutterBtn = document.getElementById('shutter') as HTMLButtonElement;
export const modeSingleBtn = document.getElementById('modeSingleBtn') as HTMLButtonElement;
export const modeVideoBtn = document.getElementById('modeVideoBtn') as HTMLButtonElement;

export const computeOnDeviceCheckbox = document.getElementById('computeOnDevice') as HTMLInputElement;
export const arOverlayCheckbox = document.getElementById('arOverlayEnabled') as HTMLInputElement;
export const sendDebugInfoCheckbox = document.getElementById('sendDebugInfo') as HTMLInputElement;
export const sendCapturedImageCheckbox = document.getElementById('sendCapturedImage') as HTMLInputElement;
export const poseReadoutEl = document.getElementById('poseReadout')!;
export const imuCheckbox = document.getElementById('imuEnabled') as HTMLInputElement;
export const imuCorrectionCheckbox = document.getElementById('imuCorrection') as HTMLInputElement;
export const imuReadoutEl = document.getElementById('imuReadout')!;

// Deliberately NOT the same ids pose-viewer-server.html uses for its own pair:
// the desktop's ui/dom.ts runs getElementById at module scope and is loaded on
// this page too (via scene/renderer.ts), so sharing an id would hand the
// desktop's binding this page's button.
export const reloadConfigBtn = document.getElementById('reloadConfigBtn') as HTMLButtonElement;
export const reloadConfigStatus = document.getElementById('reloadConfigStatus') as HTMLDivElement;

// The one collapsible panel and the toggle that parks beside it -- the same
// mechanism and the same class name (`collapsed`) pose-viewer-server.html's own
// #panel uses, so the two pages' UI reads as one project. Replaces the
// `chromeToggle` button and the `body.chromeHidden` rule that used to hide seven
// separately-positioned pills by listing every id by hand.
export const panel = document.getElementById('panel') as HTMLDivElement;
export const panelToggleBtn = document.getElementById('panelToggle') as HTMLButtonElement;
