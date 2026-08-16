// Validates every fixture in fixtures/ against the same schema the page and
// the (future) test runner use: `npm run check:fixtures`.
//
// The sibling of check-config.mjs, and it exists for the same two reasons.
//
// WHEN it fails is the first. A fixture is read by scripts, by harnesses and
// eventually by tests, and "the fixture was stale" is a diagnosis nobody
// reaches quickly -- a config file that gained or renamed a setting leaves
// every fixture saved before it holding the old shape, and a fixture is the
// thing a recorded number refers to. This turns that into one line of output.
//
// The second is the purity alarm. This imports src/poseViewer/fixture.ts
// directly (node strips the TypeScript), which only works because that module
// -- and configSchema.ts and camera/settings.ts below it -- have no fetch, no
// localStorage and no top-level await. If any of them ever grows a browser
// dependency, this script is what breaks first, before a headless test runner
// is built on the same assumption.

import { readdirSync, readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodeGray, fixtureSummary, validateFixture } from '../src/poseViewer/fixture.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FIXTURES_DIR = path.join(ROOT, 'fixtures');

if (!existsSync(FIXTURES_DIR)) {
  console.log('no fixtures/ directory -- nothing to check');
  process.exit(0);
}

const files = readdirSync(FIXTURES_DIR).filter((f) => f.endsWith('.json')).sort();
if (files.length === 0) {
  console.log('fixtures/ is empty -- nothing to check');
  process.exit(0);
}

let failed = 0;
for (const file of files) {
  const rel = path.join('fixtures', file);
  try {
    const fixture = validateFixture(JSON.parse(readFileSync(path.join(FIXTURES_DIR, file), 'utf8')), rel);
    // The schema can only say the blob is a string. Decoding is what checks
    // that it is the right NUMBER of bytes for the dimensions beside it -- a
    // truncated fixture would otherwise pass validation and then hand every
    // stage downstream a plausible image with a shifted row stride.
    const gray = decodeGray(fixture);
    // Values outside [0, 255] cannot come from toGrayscale, so they mean the
    // decode is misaligned or the file was written by something else. Assert
    // on the impossible rather than report it as a statistic.
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < gray.length; i++) { if (gray[i] < lo) lo = gray[i]; if (gray[i] > hi) hi = gray[i]; }
    if (!(lo >= 0 && hi <= 255)) throw new Error(`fixture: ${rel} decodes to samples in [${lo}, ${hi}], outside grayscale range`);
    if (fixture.name !== path.basename(file, '.json')) {
      throw new Error(`fixture: ${rel} carries name ${JSON.stringify(fixture.name)}, which does not match its filename`);
    }
    console.log(`✓ ${rel} — ${fixtureSummary(fixture)}`);
  } catch (err) {
    failed++;
    console.error(`✗ ${err.message}`);
  }
}

console.log(`${files.length - failed}/${files.length} fixture(s) valid`);
process.exit(failed > 0 ? 1 : 0);
