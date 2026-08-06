import type { Static, TObject, TSchema } from '@sinclair/typebox';
import { Value } from '@sinclair/typebox/value';
import { CameraSettingsCommonSchema, SimulatedOnlySettingsSchema } from './camera/settings.ts';
import type { CameraSettingsCommon, PhysicalCameraSettings, SimulatedCameraSettings } from './camera/settings.ts';
import { GlobalSettingsSchema, PhoneSettingsSchema, PhysicalOverridesSchema, SphereLabConfigSchema } from './configSchema.ts';
import type { SphereLabConfig } from './configSchema.ts';

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
//
// This module is only the LOADING half. The shape lives in configSchema.ts and
// camera/settings.ts, which are pure so `npm run check:config` can validate the
// file from the command line rather than at boot.

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

// ── Two different trust levels, two different reactions ─────────────────
//
// THE FILE is validated whole, and a failure throws. It is a human-edited
// artifact under version control, so "this is wrong" is actionable and the
// only safe response is to refuse to boot: every literal default is gone, so
// there is nothing sensible to substitute, and a bad value would otherwise
// travel a long way (into a camera's settings, down the websocket to a phone,
// into a GPU uniform) before surfacing as something that looks like a
// reconstruction bug rather than a config one.
//
// THE localStorage OVERLAY is validated per key, and a failure is dropped with
// a warning. It is machine-written, possibly by an OLDER build of this page
// whose schema differed, and it is not something anyone can open and fix --
// throwing would brick the page with no way back in short of clearing site
// data by hand. Falling back to the file's value is always available and
// always valid, because the file was just validated.
function describe(schema: TSchema, value: unknown, path: string): string {
  const first = [...Value.Errors(schema, value)][0];
  const at = first ? `${path}${first.path}` : path;
  return first ? `${at}: ${first.message} (got ${JSON.stringify(first.value)})` : `${at}: invalid`;
}

function validate(raw: unknown): SphereLabConfig {
  if (Value.Check(SphereLabConfigSchema, raw)) return raw;
  // Every error, not just the first: a stale config file after a rename tends
  // to have one problem per renamed setting, and fixing them one boot at a
  // time is miserable.
  const errors = [...Value.Errors(SphereLabConfigSchema, raw)]
    .map((e) => `  ${e.path || '/'}: ${e.message} (got ${JSON.stringify(e.value)})`);
  throw new Error(`config: ${CONFIG_URL} is invalid\n${errors.join('\n')}`);
}

// Merged one section at a time, per key. A whole-section replace would mean a
// config.json that GAINS a setting never reaches any browser that has saved
// that section before -- the new key would be absent from the stored copy and
// would clobber the file's freshly validated value with undefined.
//
// The per-key schema lookup is also what closes the hole this used to have:
// validation ran on the file and the overlay was applied AFTER it, so the
// least trustworthy input in the system was the one input never checked.
// (It could never add or remove a key -- this loop walks the BASE's keys --
// so the only thing it could ever have smuggled in was a wrong type, which is
// exactly what a presence check could not see.)
function overlay<T extends TObject>(base: Static<T>, saved: unknown, schema: T, path: string): void {
  if (typeof saved !== 'object' || saved === null) return;
  const target = base as Record<string, unknown>;
  for (const k of Object.keys(target)) {
    const v = (saved as Record<string, unknown>)[k];
    if (v === undefined) continue;
    const keySchema = schema.properties[k];
    if (Value.Check(keySchema, v)) target[k] = v;
    else console.warn(`config: ignoring saved ${describe(keySchema, v, `${path}.${k}`)} — using ${JSON.stringify(target[k])} from ${CONFIG_URL}`);
  }
}

// The file alone, validated, with no localStorage overlay applied. Shared by
// boot and by the "load config from disk" buttons, so that what a load reads
// is byte-for-byte what a fresh boot would read.
export async function fetchConfigFile(): Promise<SphereLabConfig> {
  const res = await fetch(CONFIG_URL, { cache: 'no-store' });
  if (!res.ok) throw new Error(`config: GET ${CONFIG_URL} failed (${res.status})`);
  return validate(await res.json());
}

// Drops this browser's live overlay, so the next boot is the file exactly.
// Separate from fetchConfigFile on purpose: a load validates FIRST and only
// discards once it knows the file it is falling back to is good.
export function discardSavedOverlay(): void {
  localStorage.removeItem(STORAGE_KEY);
}

async function load(): Promise<SphereLabConfig> {
  const disk = await fetchConfigFile();

  let saved: Record<string, unknown> = {};
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}'); } catch { saved = {}; }
  const savedCamera = saved.camera as Record<string, unknown> | undefined;
  overlay(disk.global, saved.global, GlobalSettingsSchema, 'global');
  overlay(disk.camera.common, savedCamera?.common, CameraSettingsCommonSchema, 'camera.common');
  overlay(disk.camera.simulated, savedCamera?.simulated, SimulatedOnlySettingsSchema, 'camera.simulated');
  overlay(disk.camera.physical, savedCamera?.physical, PhysicalOverridesSchema, 'camera.physical');
  overlay(disk.phone, saved.phone, PhoneSettingsSchema, 'phone');
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
  const src = settings as Record<string, unknown>;
  // Object.keys of the SCHEMA, not of the settings object: a camera carries
  // exactly these fields today, but reading the schema is what keeps this
  // honest if one ever picks up a stray property, and it is the same list the
  // validator uses.
  for (const k of Object.keys(CameraSettingsCommonSchema.properties)) {
    // fieldView is the one common setting a physical camera keeps separately
    // (see camera.physical) -- letting a phone's 'raw' land in the common
    // block would make it the next SIMULATED camera's field view.
    if (k === 'fieldView' && !isSim) continue;
    (config.camera.common as unknown as Record<string, unknown>)[k] = src[k];
  }
  if (isSim) {
    for (const k of Object.keys(SimulatedOnlySettingsSchema.properties)) {
      (config.camera.simulated as unknown as Record<string, unknown>)[k] = src[k];
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
