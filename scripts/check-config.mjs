// Validates pose-viewer.config.json against the same schema the page uses, from
// the command line: `npm run check:config`.
//
// The point is WHEN it fails. config.ts refuses to boot on an invalid file --
// which is right, since every literal default is gone and there is nothing
// safe to substitute -- but "the page is white and the console has an error"
// is a poor way to learn you fat-fingered a number. This is the same check,
// before the browser is involved.
//
// It imports the schema module directly. Node strips the TypeScript itself
// (v23+), and src/poseViewer/shared/configSchema.ts is deliberately pure -- no fetch,
// no localStorage, no top-level await -- precisely so this is possible. If
// that module ever grows a browser dependency, this script is what breaks
// first, which is the intended alarm.

import { readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Value } from '@sinclair/typebox/value';
import { PoseViewerConfigSchema } from '../src/poseViewer/shared/configSchema.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CONFIG_PATH = path.join(ROOT, 'pose-viewer.config.json');

let raw;
try {
  raw = JSON.parse(readFileSync(CONFIG_PATH, 'utf8'));
} catch (err) {
  console.error(`✗ ${path.relative(ROOT, CONFIG_PATH)}: ${err.message}`);
  process.exit(1);
}

// Every error rather than just the first: renaming a setting tends to produce
// one problem per rename (the new key missing, the old key now disallowed by
// additionalProperties), and fixing those one run at a time is miserable.
const errors = [...Value.Errors(PoseViewerConfigSchema, raw)];
if (errors.length > 0) {
  console.error(`✗ ${path.relative(ROOT, CONFIG_PATH)} — ${errors.length} problem(s):`);
  for (const e of errors) console.error(`    ${e.path || '/'}: ${e.message} (got ${JSON.stringify(e.value)})`);
  process.exit(1);
}

const count = (o) => Object.values(o).reduce((n, v) => n + (v && typeof v === 'object' ? count(v) : 1), 0);
console.log(`✓ ${path.relative(ROOT, CONFIG_PATH)} — ${count(raw)} settings, all valid`);
