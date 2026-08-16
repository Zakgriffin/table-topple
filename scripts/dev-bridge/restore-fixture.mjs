// Pushes a fixture (see src/poseViewer/shared/fixture.ts) back into the live page:
// its pixels, and a check that the page is configured the way the fixture says
// it should be run.
//
// The pixel half is what save-capture.mjs/restore-capture.mjs always did -- a
// page reload can be followed by a scripted restore instead of asking for a
// new phone photo. It re-enters ingestRealCapture's own post-decode steps
// (resizeCaptureBuffers if the size differs, updateDistortedPreview,
// buildProjectedTexture, runAxesReconstruction) rather than doing something
// equivalent-looking of its own, so it cannot drift out of step with the real
// ingest path.
//
// ── The config half, and why this REFUSES rather than applies ─────────────
//
// A fixture pins a configuration. A headless caller gets that for free -- the
// fixture's config is the only config there is. A browser does not: it has a
// localStorage overlay of its own that no fixture wrote, so restoring pixels
// into it can silently produce a run under settings the fixture never named,
// which is the precise failure that voided every historical timing number.
//
// So the diff is checked and a mismatch stops the restore. It does NOT apply
// the fixture's config, and that is deliberate: cameraPanel.ts's load button
// spells out why re-driving settings in place is a second code path that can
// disagree with boot about what a config means. The two honest ways to make
// the page match are both already there -- edit pose-viewer.config.json to the
// fixture's values and hit "load config from disk", or accept the difference
// knowingly with --force, which prints the diff and restores anyway.
//
// Usage:
//   npm run fixture:restore                  -> fixtures/default.json
//   npm run fixture:restore -- lowlight
//   npm run fixture:restore -- lowlight --force   (restore despite a config diff)

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { configDiff, fixtureSummary, validateFixture } from '../../src/poseViewer/shared/fixture.ts';
import { evalJsonInPage } from './pageEval.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

const argv = process.argv.slice(2);
const force = argv.includes('--force');
const name = argv.find((a) => !a.startsWith('--')) ?? 'default';
const inPath = path.join(ROOT, 'fixtures', `${name}.json`);

let fixture;
try {
  fixture = validateFixture(JSON.parse(readFileSync(inPath, 'utf8')), path.relative(ROOT, inPath));
} catch (err) {
  console.error(`✗ ${err.message}`);
  process.exit(1);
}
console.log(fixtureSummary(fixture));

// Read the page's config before touching anything, so a mismatch costs nothing
// -- no half-restored state to undo.
let live;
try {
  live = await evalJsonInPage(`
(function() {
  const cam = activeCamera();
  if (!cam || cam.type !== 'physical') return JSON.stringify({ error: 'active camera is not a physical camera -- toggle "use real capture" first' });
  persistConfig();
  return JSON.stringify({ config });
})()
`);
} catch (err) {
  console.error(`✗ ${err.message}`);
  process.exit(1);
}

const diff = configDiff(fixture.config, live.config);
if (diff.length > 0) {
  const verb = force ? 'restoring anyway (--force)' : 'refusing to restore';
  console.error(`\n${diff.length} setting(s) differ (fixture -> page), ${verb}:`);
  for (const line of diff) console.error(`    ${line}`);
  if (!force) {
    console.error('\n  Make the page match: edit pose-viewer.config.json to these values and use the');
    console.error('  "load config from disk" button (it reloads, which is the boot path, so it cannot');
    console.error('  disagree with a fresh page). Or pass --force to restore under the page\'s own config.');
    process.exit(1);
  }
  console.error('');
}

// decodeGrayB64 is fixture.ts's own decoder, reached by bare name off
// globalThis (main.ts globs every module onto it) -- the same function this
// script would otherwise have to reimplement inside a template string, where
// nobody would find it to keep it in step with the encoder.
//
// buildProjectedTexture takes an explicit backend now (see pipeline/backend.ts);
// the previous restore-capture.mjs still called it with one argument, so it was
// passing `undefined` where a Backend belongs and silently taking the GPU
// branch regardless of the forceCPU setting.
const RESTORE_CODE = `
(function() {
  const w = ${fixture.capture.w}, h = ${fixture.capture.h};
  const gray = decodeGrayB64(${JSON.stringify(fixture.capture.grayF64B64)}, w, h, ${JSON.stringify(fixture.name)});
  const cam = activeCamera();
  if (!cam || cam.type !== 'physical') return JSON.stringify({ error: 'active camera is not a physical camera -- toggle "use real capture" first' });
  if (w !== cam.rtSize.w || h !== cam.rtSize.h) resizeCaptureBuffers(cam, { w, h });
  cam.lastRealCaptureGray = gray;
  cam.lastRealCaptureW = w; cam.lastRealCaptureH = h;
  updateDistortedPreview(cam);
  const backend = backendFromForceCPU(globalState.forceCPU);
  if (globalState.mode === 'projected') buildProjectedTexture(cam, backend);
  runAxesReconstruction(cam);
  return JSON.stringify({ restored: true, w, h, backend });
})()
`;

try {
  const res = await evalJsonInPage(RESTORE_CODE);
  console.log(`✓ restored ${res.w}x${res.h} into the page, reconstruction queued on ${res.backend}`);
  // runAxesReconstruction queues its real work in a requestAnimationFrame and
  // returns immediately -- see cli.js's RACE CONDITION WARNING. Anything that
  // reads a result has to poll activeCamera().axesCapturing until it is false;
  // this script deliberately does not wait, so say so rather than let the tick
  // above read as "the reconstruction is done".
  console.log('  (queued, not finished -- poll activeCamera().axesCapturing before reading any result)');
} catch (err) {
  console.error(`✗ ${err.message}`);
  process.exit(1);
}
