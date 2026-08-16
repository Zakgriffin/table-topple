// Saves the capture currently loaded in the live page, PLUS the configuration
// the page is running it under, to fixtures/<name>.json -- see
// src/poseViewer/shared/fixture.ts for what a fixture is and why the config half is
// the point.
//
// Two jobs, and it is worth being clear that they are different:
//
//   - The old one (this was save-capture.mjs): a page reload -- needed to pick
//     up a change HMR missed, or just for a clean restart -- does not force
//     re-taking a photo on the phone. restore-fixture.mjs is that other half.
//   - The new one: a fixture is a re-derivable INPUT. A timing or a test that
//     names one says exactly what it ran on and under what settings, which is
//     the thing every historical measurement in this project failed to record.
//
// Usage:
//   npm run fixture:save                      -> fixtures/default.json
//   npm run fixture:save -- lowlight --note "phone at 30cm, room lights off"
//   npm run fixture:save -- default --force   (overwrite an existing fixture)

import { mkdirSync, existsSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FIXTURE_VERSION, fixtureSummary, validateFixture } from '../../src/poseViewer/shared/fixture.ts';
import { evalJsonInPage } from './pageEval.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const FIXTURES_DIR = path.join(ROOT, 'fixtures');

const argv = process.argv.slice(2);
const force = argv.includes('--force');
const noteAt = argv.indexOf('--note');
const note = noteAt >= 0 ? (argv[noteAt + 1] ?? '') : '';
// -1 when there is no --note, so that index 0 (usually the name) is not
// excluded as if it were the note's value.
const noteValueAt = noteAt >= 0 ? noteAt + 1 : -1;
const positional = argv.filter((a, i) => !a.startsWith('--') && i !== noteValueAt);
const name = positional[0] ?? 'default';

if (!/^[a-z0-9][a-z0-9-]*$/.test(name)) {
  console.error(`✗ fixture name must be lowercase kebab-case (got ${JSON.stringify(name)})`);
  process.exit(1);
}
const outPath = path.join(FIXTURES_DIR, `${name}.json`);
if (existsSync(outPath) && !force) {
  console.error(`✗ ${path.relative(ROOT, outPath)} already exists -- pass --force to overwrite.`);
  console.error('  A fixture is what a recorded number refers to, so replacing one silently would');
  console.error('  make those numbers un-re-derivable in exactly the way fixtures exist to prevent.');
  process.exit(1);
}

// persistConfig() first, so the snapshot is of the ACTIVE camera's live
// settings rather than of whatever the config object last happened to hold.
// Every UI control already calls it on change (see ui/dom.ts's bindSlider), so
// this is normally a no-op -- but a setting pushed in by settingsSync or set
// from the console is not, and "the config I saved was stale" is not a failure
// anyone would notice until a number disagreed months later.
//
// encodeGray is fixture.ts's own encoder, reached by bare name off globalThis
// (main.ts globs every module onto it). Not reimplemented here: the encoding
// is half of the format, and a second copy of it in a template string is a
// copy nobody would think to update.
const EVAL_CODE = `
(function() {
  const cam = activeCamera();
  if (!cam || cam.type !== 'physical') return JSON.stringify({ error: 'active camera is not a physical camera (toggle "use real capture" first)' });
  if (!cam.lastRealCaptureGray) return JSON.stringify({ error: 'no capture loaded' });
  const w = cam.lastRealCaptureW, h = cam.lastRealCaptureH;
  if (cam.lastRealCaptureGray.length !== w * h) {
    return JSON.stringify({ error: 'capture buffer is ' + cam.lastRealCaptureGray.length + ' samples but the page says it is ' + w + 'x' + h });
  }
  persistConfig();
  return JSON.stringify({ w, h, grayF64B64: encodeGray(cam.lastRealCaptureGray), config });
})()
`;

let page;
try {
  page = await evalJsonInPage(EVAL_CODE);
} catch (err) {
  console.error(`✗ ${err.message}`);
  process.exit(1);
}

const fixture = {
  fixtureVersion: FIXTURE_VERSION,
  name,
  savedAt: new Date().toISOString(),
  note,
  config: page.config,
  // Last, so `head fixtures/x.json` shows the config rather than the blob.
  capture: { w: page.w, h: page.h, grayF64B64: page.grayF64B64 },
};

// Validated BEFORE it is written, not after. An invalid fixture on disk is a
// trap for whoever reads it next; an invalid one that never lands is a message
// on this terminal, right next to the thing that caused it.
try {
  validateFixture(fixture, `<page snapshot for ${name}>`);
} catch (err) {
  console.error(`✗ refusing to write an invalid fixture:\n${err.message}`);
  process.exit(1);
}

mkdirSync(FIXTURES_DIR, { recursive: true });
writeFileSync(outPath, JSON.stringify(fixture, null, 2) + '\n');
const mb = (page.grayF64B64.length / 1024 / 1024).toFixed(1);
console.log(`✓ wrote ${path.relative(ROOT, outPath)} (${mb}MB base64)`);
console.log(`  ${fixtureSummary(fixture)}`);
