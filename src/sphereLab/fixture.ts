import { Type } from '@sinclair/typebox';
import type { Static } from '@sinclair/typebox';
import { Value } from '@sinclair/typebox/value';
import { physicalSettingsFrom } from './camera/settings.ts';
import type { PhysicalCameraSettings } from './camera/settings.ts';
import { SphereLabConfigSchema } from './configSchema.ts';
import type { SphereLabConfig } from './configSchema.ts';
import { backendFromForceCPU } from './pipeline/backend.ts';
import type { Backend } from './pipeline/backend.ts';

// ── A fixture: a capture that carries the configuration it is to be run under ──
//
// The problem this exists to fix, in the user's words: the old timing runs
// "never specified exactly what the configuration was ... so our timing tests
// especially mean little to nothing carrying forward." Every number on record
// from before the config file was a measurement of an unknown machine. A
// fixture is the smallest thing that makes a run RE-DERIVABLE: the pixels, and
// the config those pixels are to be processed with, in one file.
//
// So a fixture is not a recording of provenance. For a physical capture the
// config had nothing to do with producing the pixels -- a photo is a photo. It
// is a DECLARATION: run this input under this configuration. That is what makes
// attaching a config to a capture taken before the config file existed honest
// rather than invented (see fixtures/default.json's note), and it is what makes
// a fixture usable as a test input rather than only as a souvenir.
//
// ── This module is PURE, and that is load-bearing ─────────────────────────
//
// No fetch, no localStorage, no DOM, no top-level await, no THREE. Same
// constraint configSchema.ts and camera/settings.ts carry, for the same reason:
// scripts/ imports it directly (node strips the TypeScript), and step 3's
// headless test runner will too. If this file ever grows a browser dependency,
// `npm run check:fixtures` is what breaks first, which is the intended alarm.
//
// ── The whole config, not a pipeline-relevant subset ──────────────────────
//
// Snapshotting only "the settings the pipeline reads" would require drawing the
// library boundary first, which is the open question this whole restructure is
// working toward -- and getting it wrong means a fixture silently fails to pin
// the one setting that mattered. The whole object is 83 lines of JSON. Take all
// of it and validate it with the schema that already exists.

const strict = { additionalProperties: false } as const;

// Bumped whenever the shape changes in a way that makes an older file wrong
// rather than merely older. Type.Literal, so an old fixture fails validation
// with "expected 2, got 1" instead of being half-read.
export const FIXTURE_VERSION = 1;

export const FixtureSchema = Type.Object({
  fixtureVersion: Type.Literal(FIXTURE_VERSION),
  // Free-form, and duplicated in the filename. The filename is what you type;
  // this is what survives being copied somewhere else.
  name: Type.String(),
  savedAt: Type.String(),
  // What the capture is OF, and anything a later reader needs to know about how
  // it was taken. Always present, possibly empty -- an optional field is one
  // more shape for every consumer to branch on.
  note: Type.String(),
  config: SphereLabConfigSchema,
  capture: Type.Object({
    w: Type.Integer({ minimum: 1 }),
    h: Type.Integer({ minimum: 1 }),
    // Base64 of the raw little-endian bytes of a Float64Array of length w*h.
    //
    // ── Why float64 and not the bytes this used to store ──
    //
    // The pre-fixture saved-capture.json rounded each sample to a byte, and
    // cli.js's header admitted the consequence: "decode results after a restore
    // may differ by a tiny amount from before -- fine for comparative testing,
    // not pixel-exact." That is exactly the caveat a fixture cannot have. gray
    // is 0.299r + 0.587g + 0.114b (see decode.ts's toGrayscale), which is not
    // integral, so rounding it is a real perturbation of the pipeline's input,
    // and a number measured against the live capture would not be a number
    // measured against the fixture.
    //
    // The cost is ~8 bytes per pixel, ~3.3MB of base64 for a 480x640 frame.
    // Paid deliberately: a fixture that is not bit-identical to what the page
    // held is not a fixture, and the fix for a large unreadable tail is
    // `jq 'del(.capture.grayF64B64)' fixtures/x.json`, not a lossy encoding.
    // It is written LAST in the file so `head` shows the config.
    grayF64B64: Type.String(),
  }, strict),
}, strict);

export type Fixture = Static<typeof FixtureSchema>;

// A 3.3MB base64 blob is one of the fields being validated, and TypeBox's error
// objects carry the offending value. Printing that raw turns "your fixture has
// a typo" into three megabytes of terminal.
function render(value: unknown): string {
  const s = JSON.stringify(value);
  if (s === undefined) return String(value);
  return s.length > 120 ? `${s.slice(0, 117)}...` : s;
}

// Every error rather than just the first, matching config.ts's validate() and
// for the same reason: a stale fixture after a config rename has one problem
// per renamed setting, and fixing them one run at a time is miserable.
export function validateFixture(raw: unknown, source: string): Fixture {
  if (Value.Check(FixtureSchema, raw)) return raw;
  const errors = [...Value.Errors(FixtureSchema, raw)]
    .map((e) => `  ${e.path || '/'}: ${e.message} (got ${render(e.value)})`);
  throw new Error(`fixture: ${source} is invalid\n${errors.join('\n')}`);
}

// btoa/atob rather than node's Buffer or the browser's TextDecoder, because
// both halves of this module have to run in both places. The chunking is not
// optional: String.fromCharCode.apply on a 2.4MB array overflows the argument
// stack.
const B64_CHUNK = 8192;

// Raw bytes of a typed array's buffer are host-endian. Every machine this runs
// on is little-endian, so rather than carry a byteOrder field that nothing
// would ever branch on, assert the assumption and fail loudly if it is ever
// false. A fixture written big-endian and read little-endian would otherwise
// decode to garbage that still has the right length.
function assertLittleEndian(): void {
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
    throw new Error('fixture: big-endian host -- the gray encoding assumes little-endian');
  }
}

export function encodeGray(gray: Float64Array): string {
  assertLittleEndian();
  const bytes = new Uint8Array(gray.buffer, gray.byteOffset, gray.length * 8);
  let binary = '';
  for (let i = 0; i < bytes.length; i += B64_CHUNK) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + B64_CHUNK)));
  }
  return btoa(binary);
}

// Split out from decodeGray so that restore-fixture.mjs can call it inside the
// PAGE, where there is no Fixture object -- only the three fields it inlines
// into the eval it sends. Same decoder either way; the alternative was a second
// copy of it living in a template string.
//
// The length check is the point: a truncated or mis-sized blob would otherwise
// produce a Float64Array of the wrong length, and every stage downstream would
// read a plausible image with a shifted row stride.
export function decodeGrayB64(grayF64B64: string, w: number, h: number, source: string): Float64Array {
  assertLittleEndian();
  const binary = atob(grayF64B64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const expected = w * h * 8;
  if (bytes.byteLength !== expected) {
    throw new Error(`fixture: ${source} gray is ${bytes.byteLength} bytes, expected ${expected} for ${w}x${h} float64`);
  }
  return new Float64Array(bytes.buffer, bytes.byteOffset, w * h);
}

export function decodeGray(fixture: Fixture): Float64Array {
  const { w, h, grayF64B64 } = fixture.capture;
  return decodeGrayB64(grayF64B64, w, h, fixture.name);
}

// The settings the pipeline runs with, rebuilt from the snapshot by the same
// rule camera/factory.ts uses for a live physical camera. A fixture is always a
// physical capture (save-fixture.mjs refuses anything else), so there is no
// simulated form of this.
export function fixtureSettings(fixture: Fixture): PhysicalCameraSettings {
  return physicalSettingsFrom(fixture.config.camera);
}

// What the APP would choose for this config -- not what a harness must use. A
// differential run sweeps both backends over one fixture, which is the whole
// argument for backend being a parameter rather than a settings field (see
// pipeline/backend.ts). This is the default, nothing more.
export function fixtureBackend(fixture: Fixture): Backend {
  return backendFromForceCPU(fixture.config.global.forceCPU);
}

// Every leaf where two configs disagree, as `path: a -> b` lines. Used by
// restore-fixture.mjs to refuse to push a fixture into a page configured
// differently, which is the live-page half of pinning: headless callers get the
// fixture's config by construction, but a browser has its own localStorage
// overlay and no way to notice it drifted.
//
// Walks the UNION of both key sets so a key present on one side only is
// reported rather than skipped -- that is the shape a version skew takes.
export function configDiff(a: SphereLabConfig, b: SphereLabConfig): string[] {
  const out: string[] = [];
  const walk = (x: unknown, y: unknown, path: string): void => {
    const objish = (v: unknown) => typeof v === 'object' && v !== null;
    if (objish(x) && objish(y)) {
      const keys = new Set([...Object.keys(x as object), ...Object.keys(y as object)]);
      for (const k of [...keys].sort()) {
        walk((x as Record<string, unknown>)[k], (y as Record<string, unknown>)[k], path ? `${path}.${k}` : k);
      }
      return;
    }
    if (x !== y) out.push(`${path}: ${render(x)} -> ${render(y)}`);
  };
  walk(a, b, '');
  return out;
}

// One line for a script to print. Names the handful of settings that most
// often explain a differing result, so a run's log says what it ran under
// without dumping the whole config.
export function fixtureSummary(fixture: Fixture): string {
  const s = fixtureSettings(fixture);
  return `${fixture.name}: ${fixture.capture.w}x${fixture.capture.h}, saved ${fixture.savedAt}, `
    + `backend ${fixtureBackend(fixture)}, tol ${s.lsdToleranceDeg}deg, rhoNoise ${s.lsdRhoNoiseThreshold}, `
    + `fov ${s.horizFovDeg}deg, board ${fixture.config.global.boardSize}`;
}
