// ── Every element this page touches, resolved once ───────────────────────
//
// The LEAF of this client's module graph: it imports nothing from its siblings,
// so it can never be part of a cycle and can never be the module that is still
// initializing when someone reads it.
//
// Mirrors poseViewer/client/dom.ts in shape, deliberately, so the two clients'
// pages read the same way -- the isolation rule forbids sharing the file, not
// the pattern.

export const video = document.getElementById('v') as HTMLVideoElement;
export const viewfinder = document.getElementById('viewfinder') as HTMLCanvasElement;
export const viewfinderCtx = viewfinder.getContext('2d')!;
/** The transparent canvas the board game is drawn into, over the feed. */
export const arCanvas = document.getElementById('arCanvas') as HTMLCanvasElement;
export const statusEl = document.getElementById('status')!;
export const chromeToggleBtn = document.getElementById('chromeToggle') as HTMLButtonElement;
