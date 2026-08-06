import { CameraSettingsCommon, COMMON_KEYS, PhysicalCameraSettings, SIM_ONLY_KEYS, SimulatedCameraSettings, SimulatedOnlySettings } from './camera/settings.ts';
import { Mode } from './types.ts';

// ── One config object, one file on disk ──────────────────────────────────
//
// Every configurable value in Sphere Lab lives in sphere-lab.config.json and
// nowhere else. There are deliberately NO literal defaults left in TypeScript
// and no `value=`/`checked` attributes left on the controls in
// sphere-lab.html -- both used to be defaults in their own right, and which of
// the two won depended on whether a control was global (the HTML attribute
// won, because bindSlider applies it at load) or per-camera (the TypeScript
// literal won, because refreshCameraPanel overwrites the DOM from the camera
// the moment one exists). That was not a rule anyone could hold in their head.
//
// localStorage now holds this SAME object, in this same shape, rather than a
// flat map of DOM-element-id -> string. That map keyed some booleans by their
// BUTTON's id (`toggleLsdSegments`) and the matching setting by its field name
// (`showLsdSegments`), read only a hand-picked subset back in at camera
// creation, and stored everything as strings.
//
// The disk file is the DEFAULTS and is only rewritten when you ask for it (the
// "save config to disk" button -> the dev bridge). localStorage is the live
// overlay on top. So: edit freely, and promote what is worth keeping.

export interface GlobalSettings {
  // The 3D canvas has exactly one current view regardless of camera
  // count/selection, so this is global rather than per-camera.
  mode: Mode;
  // UI chrome, persisted for the same reason every slider is: reopening the
  // page should not undo how you left the workspace.
  panelCollapsed: boolean;
  overlayPanelCollapsed: boolean;
  // The floor is one shared object, not owned by any camera.
  showFloor: boolean;
  floorCellOutlineSubdiv: number;
  // The De Bruijn floor pattern's board size (cells per side) -- overrides
  // debruijn.ts's ORDER5_CANDIDATE.cropSize at runtime via scene/floor.ts's
  // rebuildFloorPattern. 144 so that one pattern cell is one board-game cell
  // (BOARD_CELLS in src/game/constants.ts) and the AR overlay's board lands on
  // the physical one 1:1 rather than at some scale factor.
  boardSize: number;
  // ── The one GPU/CPU switch, replacing nine per-stage ones (2026-08-05) ──
  //
  // There used to be a `useGPU*` toggle per stage: gradient, growRegions,
  // collectRegions, lsdFit, fit, periodSweep, decode, decodeFused, project. All
  // nine defaulted true. They were the INSTRUMENT that produced every measured
  // number on record, and the answer they produced was "everything is GPU" --
  // at which point a runtime switch for a choice nobody makes at runtime is
  // pure carrying cost. Nine independent booleans is 2^9 of nominal correctness
  // surface, and every new stage doubled it.
  //
  // They also structurally blocked the endgame: you cannot record a fixed
  // command list, or fuse the pipeline into one submit, if any stage might run
  // on the CPU this frame. And they made FieldResidency a transfer-DECISION
  // engine (which side is each stage on, and what must therefore cross?) rather
  // than a plain named-slot arena.
  //
  // What is NOT deleted is the CPU implementations. They earned their keep
  // twice over -- they caught the `layout:'auto'` binding prune that silently
  // no-op'd every submit while returning plausible garbage, and the
  // BOUNDARY_EPS bug that was real on BOTH sides. They are now REFERENCE
  // implementations rather than production branches, reachable through this
  // single switch, and pipelineGPU/lsdChainVerify.ts's two-configuration
  // differential is what keeps them from rotting unnoticed now that production
  // never runs them.
  //
  // Note this is NOT the no-WebGPU fallback and does not need to be: every GPU
  // stage already falls back on its own when there is no device or a dispatch
  // fails validation, and that path is independent of this flag (the flag gates
  // ENTRY, a `return null/false` gates FALLBACK). This exists for harnesses and
  // for answering "is the GPU lying to me" by hand.
  forceCPU: boolean;
}

// The phone page's own debug toggles (mobile-capture.html). Not part of
// PoseComputeState.settings -- those are pushed down per-camera by
// devBridge/client.ts's settingsSync and are already covered by
// camera.common -- these are the switches that only mean anything on the
// device: whether it computes its own pose, what it ships back, and the IMU.
export interface PhoneSettings {
  computeOnDevice: boolean;
  sendDebugInfo: boolean;
  sendCapturedImage: boolean;
  imuEnabled: boolean;
  imuCorrection: boolean;
}

export interface SphereLabConfig {
  global: GlobalSettings;
  camera: {
    common: CameraSettingsCommon;
    simulated: SimulatedOnlySettings;
    // A physical camera is a real phone: it has no position/lens sliders of
    // its own, only a handful of common settings that mean something
    // different for a photo than for a render. Typed as an explicit Pick
    // rather than Partial<CameraSettingsCommon> so that "which settings does
    // a physical camera override" stays a listed fact.
    physical: Pick<CameraSettingsCommon, 'fieldView'>;
  };
  phone: PhoneSettings;
}

const GLOBAL_KEYS = {
  mode: true, panelCollapsed: true, overlayPanelCollapsed: true,
  showFloor: true, floorCellOutlineSubdiv: true, boardSize: true, forceCPU: true,
} satisfies Record<keyof GlobalSettings, true>;

const PHONE_KEYS = {
  computeOnDevice: true, sendDebugInfo: true, sendCapturedImage: true,
  imuEnabled: true, imuCorrection: true,
} satisfies Record<keyof PhoneSettings, true>;

const PHYSICAL_KEYS = { fieldView: true } satisfies Record<keyof SphereLabConfig['camera']['physical'], true>;

// Fetched, not imported. A static `import config from '../sphere-lab.config.json'`
// would put the file in vite's module graph, so saving it would trigger an HMR
// full reload -- which wipes whatever capture/measurement is on screen, i.e.
// exactly the state you were trying to preserve by saving. Fetching keeps the
// file outside the graph, so a save is silent.
//
// The cost of staying out of the module graph is that a production `vite build`
// would not emit the file either (it is not under publicDir). This project only
// ever runs `npm run dev`, where vite serves the project root at `/`, so the
// fetch resolves. If a real build is ever added, the file has to be copied into
// the output -- the failure is loud (this throws at boot), not silent.
const CONFIG_URL = '/sphere-lab.config.json';
const STORAGE_KEY = 'sphereLab.config';

// Missing keys are a hard failure, by design. Every literal default is gone,
// so there is nothing sensible to substitute -- a silent `undefined` would
// travel a long way (into a camera's settings, down the websocket to a phone,
// into a GPU uniform) before surfacing as something that looks like a
// reconstruction bug rather than a config one.
function requireKeys(section: unknown, keys: Record<string, true>, path: string): void {
  if (typeof section !== 'object' || section === null) throw new Error(`config: ${path} is missing or not an object`);
  const missing = Object.keys(keys).filter((k) => !(k in (section as Record<string, unknown>)));
  if (missing.length > 0) throw new Error(`config: ${path} is missing ${missing.length} key(s): ${missing.join(', ')}`);
}

function validate(raw: unknown): SphereLabConfig {
  if (typeof raw !== 'object' || raw === null) throw new Error('config: root is not an object');
  const c = raw as SphereLabConfig;
  requireKeys(c.global, GLOBAL_KEYS, 'global');
  requireKeys(c.camera?.common, COMMON_KEYS, 'camera.common');
  requireKeys(c.camera?.simulated, SIM_ONLY_KEYS, 'camera.simulated');
  requireKeys(c.camera?.physical, PHYSICAL_KEYS, 'camera.physical');
  requireKeys(c.phone, PHONE_KEYS, 'phone');
  return c;
}

// The localStorage overlay is merged one section at a time, per key. A whole-
// section replace would mean a config.json that GAINS a setting never reaches
// any browser that has saved that section before -- the new key would be
// absent from the stored copy and would clobber the file's value with
// undefined, which validate() has already run and would not catch.
function overlay<T extends object>(base: T, saved: unknown): T {
  if (typeof saved !== 'object' || saved === null) return base;
  for (const k of Object.keys(base) as (keyof T)[]) {
    const v = (saved as Record<string, unknown>)[k as string];
    if (v !== undefined) base[k] = v as T[keyof T];
  }
  return base;
}

async function load(): Promise<SphereLabConfig> {
  const res = await fetch(CONFIG_URL, { cache: 'no-store' });
  if (!res.ok) throw new Error(`config: GET ${CONFIG_URL} failed (${res.status})`);
  const disk = validate(await res.json());

  let saved: Record<string, unknown> = {};
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}'); } catch { saved = {}; }
  overlay(disk.global, saved.global);
  const savedCamera = saved.camera as Record<string, unknown> | undefined;
  overlay(disk.camera.common, savedCamera?.common);
  overlay(disk.camera.simulated, savedCamera?.simulated);
  overlay(disk.camera.physical, savedCamera?.physical);
  overlay(disk.phone, saved.phone);
  return disk;
}

// Top-level await: every module that imports this one -- which, through
// state.ts's globalState, is most of them -- is guaranteed to see a fully
// loaded config at its own first line. That is what lets the rest of the app
// keep reading settings synchronously, with no "config not ready yet" state to
// thread through 40 call sites.
export const config: SphereLabConfig = await load();

// Where persistConfig finds the camera whose settings it should capture.
//
// A getter pushed down from above rather than an `activeCamera()` import,
// because config.ts sits at the BOTTOM of the import graph -- state.ts's
// globalState is literally config.global, and most of the pipeline reads
// globalState -- so importing camera/store.ts here would close a cycle back
// through it. Through a module with a top-level await that is not a style
// complaint: an importer can end up holding a binding that has not been
// initialized yet. ui/cameraPanel.ts registers it once at boot.
let activeCameraSettings: () => CameraSettingsCommon | SimulatedCameraSettings | null = () => null;
export function registerActiveCameraSettingsSource(fn: () => CameraSettingsCommon | SimulatedCameraSettings | null): void {
  activeCameraSettings = fn;
}

// The one persistence entry point. Captures the active camera first, so that
// callers never have to remember whether the thing they just changed was a
// global or a per-camera setting -- both end up in the same object either way.
export function persistConfig(): void {
  const settings = activeCameraSettings();
  if (settings) mirrorCameraIntoConfig(settings);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

// Copies a camera's live settings back into the config object, so the values
// you are actually looking at become both what persists and what the NEXT
// camera is created from. Deliberately last-touched-wins across cameras: the
// config holds ONE camera template, not one per camera.
//
// The two casts are the price of copying by a runtime key list; the manifests
// in camera/settings.ts are what make that list provably complete.
function mirrorCameraIntoConfig(settings: CameraSettingsCommon | SimulatedCameraSettings): void {
  const isSim = 'camX' in settings;
  for (const k of Object.keys(COMMON_KEYS) as (keyof CameraSettingsCommon)[]) {
    // fieldView is the one common setting a physical camera keeps separately
    // (see SphereLabConfig.camera.physical) -- letting a phone's 'raw' land in
    // the common block would make it the next SIMULATED camera's field view.
    if (k === 'fieldView' && !isSim) continue;
    (config.camera.common as unknown as Record<string, unknown>)[k] = settings[k];
  }
  if (isSim) {
    for (const k of Object.keys(SIM_ONLY_KEYS) as (keyof SimulatedOnlySettings)[]) {
      (config.camera.simulated as unknown as Record<string, unknown>)[k] = settings[k];
    }
  } else {
    config.camera.physical.fieldView = settings.fieldView;
  }
}

export function createDefaultSimulatedSettings(): SimulatedCameraSettings {
  return { ...structuredClone(config.camera.common), ...structuredClone(config.camera.simulated) };
}

export function createDefaultPhysicalSettings(): PhysicalCameraSettings {
  // camera.physical overrides the common block: 'raw' rather than 'noised',
  // because the simulated-distortion field views (noised/antialiased/
  // downsampled) do not exist for a real photo and are hidden from the
  // field-view list entirely for a physical camera (see refreshCameraPanel).
  return { ...structuredClone(config.camera.common), ...structuredClone(config.camera.physical) };
}

// Hands the current config to the dev bridge, which writes it over
// sphere-lab.config.json. Sent as an already-serialized string so what lands
// on disk is exactly what the browser is holding.
export function configAsJson(): string {
  return JSON.stringify(config, null, 2) + '\n';
}
