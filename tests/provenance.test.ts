import test from 'node:test';
import assert from 'node:assert/strict';
import { provenance } from '../src/sphereLab/harness/input.ts';
import type { HarnessInput } from '../src/sphereLab/harness/input.ts';
import { loadInput } from './helpers/fixtures.ts';

// ── What a report has to carry to be comparable to another one ────────────
//
// Two hashes rather than one, and the tests that matter are the ones proving
// they are INDEPENDENT. A single "hash of everything" would pass any test that
// only checks "the fingerprint changed when something changed" -- and would be
// useless for the question actually asked of it, which is WHICH half moved.
//
// See harness/input.ts. The backend is deliberately in neither: it is per-call,
// several harnesses sweep it, and the reports that care record it separately.

function withSettings(base: HarnessInput, patch: Record<string, unknown>): HarnessInput {
  return { ...base, settings: { ...base.settings, ...patch } as HarnessInput['settings'] };
}

test('provenance is stable: the same input twice hashes the same', () => {
  const a = provenance(loadInput());
  const b = provenance(loadInput());
  assert.deepEqual(a, b);
  assert.match(a.configHash, /^[0-9a-f]{8}$/);
  assert.match(a.captureHash, /^[0-9a-f]{8}$/);
});

// ── THE INDEPENDENCE PAIR ──
//
// Each of these asserts BOTH halves: the one that moved and the one that must
// not have. Dropping the second assertion in either would let a single combined
// hash pass both.

test('a settings change moves configHash and leaves captureHash alone', () => {
  const base = loadInput();
  const a = provenance(base);
  const b = provenance(withSettings(base, { lsdToleranceDeg: base.settings.lsdToleranceDeg + 0.5 }));
  assert.notEqual(a.configHash, b.configHash, 'a pipeline setting must reach the config hash');
  assert.equal(a.captureHash, b.captureHash, 'and must NOT reach the capture hash');
});

test('a pixel change moves captureHash and leaves configHash alone', () => {
  const base = loadInput();
  const gray = Float64Array.from(base.gray);
  gray[gray.length >> 1] += 1 / 255;
  const a = provenance(base);
  const b = provenance({ ...base, gray });
  assert.notEqual(a.captureHash, b.captureHash, 'ONE changed sample must be visible');
  assert.equal(a.configHash, b.configHash, 'and must not disturb the config hash');
});

test('aspect is part of the configuration, because the pipeline reads it', () => {
  // PoseInput is `{ aspect, settings }` -- the whole surface besides the pixels
  // and the backend. Leaving aspect out would make two genuinely different runs
  // claim the same fingerprint.
  const base = loadInput();
  assert.notEqual(provenance(base).configHash, provenance({ ...base, aspect: base.aspect * 1.1 }).configHash);
});

test('key order in the settings object does not change the hash', () => {
  // Two objects holding the same configuration are the same configuration. A
  // report calling them different would be worse than no report -- and
  // JSON.stringify alone WOULD call them different, which is why the hash
  // canonicalizes first.
  const base = loadInput();
  const reversed = Object.fromEntries(
    Object.entries(base.settings).reverse(),
  ) as HarnessInput['settings'];
  assert.equal(provenance(base).configHash, provenance({ ...base, settings: reversed }).configHash);
  assert.notEqual(
    JSON.stringify(base.settings), JSON.stringify(reversed),
    'the control: these must actually stringify differently, or this test proves nothing',
  );
});

test('provenance names the input and its size', () => {
  const p = provenance(loadInput());
  assert.equal(p.input, 'fixture:default');
  assert.equal(p.w, 480);
  assert.equal(p.h, 640);
});
