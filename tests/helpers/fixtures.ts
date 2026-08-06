import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateFixture } from '../../src/sphereLab/fixture.ts';
import type { Fixture } from '../../src/sphereLab/fixture.ts';
import { inputFromFixture } from '../../src/sphereLab/harness/input.ts';
import type { HarnessInput } from '../../src/sphereLab/harness/input.ts';

// The node-side fixture loader. The browser's is cameraInput.ts's
// fixtureInput(), which fetches the same file over the dev server; both go
// through fixture.ts's validateFixture, so a test and a page cannot disagree
// about what a valid fixture is.

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');

export function loadFixture(name = 'default'): Fixture {
  const file = path.join(ROOT, 'fixtures', `${name}.json`);
  return validateFixture(JSON.parse(readFileSync(file, 'utf8')), path.relative(ROOT, file));
}

export function loadInput(name = 'default'): HarnessInput {
  return inputFromFixture(loadFixture(name));
}

// Relative comparison, because every golden number in this suite is a
// floating-point result of the same f64 arithmetic on every platform EXCEPT
// where libm differs (Math.sin/cos/atan2). 1e-9 is far tighter than any real
// change to the pipeline would produce and far looser than that difference.
export function closeTo(actual: number, expected: number, rel = 1e-9): boolean {
  if (actual === expected) return true;
  return Math.abs(actual - expected) <= rel * Math.max(1, Math.abs(expected));
}
