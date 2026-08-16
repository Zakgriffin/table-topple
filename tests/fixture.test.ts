import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { configDiff, decodeGray, encodeGray, fixtureBackend, fixtureSettings, validateFixture } from '../src/poseViewer/fixture.ts';
import { loadFixture, loadInput } from './helpers/fixtures.ts';

// The format itself. Cheap, and it is what every other test in this suite
// stands on -- a fixture that decodes to the wrong length would make every
// pipeline assertion below a measurement of garbage that still had a plausible
// shape.

test('the fixture validates and decodes to its stated dimensions', () => {
  const f = loadFixture();
  const gray = decodeGray(f);
  assert.equal(gray.length, f.capture.w * f.capture.h);
});

test('every sample is inside the grayscale range', () => {
  // toGrayscale is a convex combination of three bytes, so nothing outside
  // [0, 255] can come from a real capture. Out-of-range values would mean the
  // float64 decode is misaligned -- which produces enormous or denormal
  // numbers, not slightly-wrong ones, so this is a sharp check and not a
  // statistical one.
  const gray = decodeGray(loadFixture());
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < gray.length; i++) { if (gray[i] < lo) lo = gray[i]; if (gray[i] > hi) hi = gray[i]; }
  assert.ok(lo >= 0 && hi <= 255, `samples span [${lo}, ${hi}]`);
});

test('gray survives an encode/decode round trip exactly', () => {
  // The property the float64 encoding exists for. A FRACTIONAL sample, because
  // that is precisely what the old byte-per-pixel format destroyed and what an
  // integral-only test would not notice.
  const src = new Float64Array([0.299 * 17 + 0.587 * 200 + 0.114 * 3, 254.99999999, 0, 255, 1 / 3]);
  const f = { ...loadFixture(), capture: { w: 5, h: 1, grayF64B64: encodeGray(src) } };
  assert.deepEqual(Array.from(decodeGray(validateFixture(f, 'round-trip'))), Array.from(src));
});

test('a fixture rebuilds the settings and backend it was saved under', () => {
  const f = loadFixture();
  const s = fixtureSettings(f);
  // The rule in camera/settings.ts: common, then the physical overrides on top.
  assert.equal(s.fieldView, f.config.camera.physical.fieldView);
  assert.equal(s.lsdToleranceDeg, f.config.camera.common.lsdToleranceDeg);
  assert.equal(fixtureBackend(f), f.config.global.forceCPU ? 'cpu' : 'gpu');
});

test('configDiff reports a changed leaf and a missing key, and nothing on equal configs', () => {
  // The check restore-fixture.mjs refuses on. A diff that silently skipped a
  // key present on one side only would pass a version skew straight through,
  // which is the shape this failure actually takes.
  const f = loadFixture();
  const b = structuredClone(f.config);
  b.camera.common.lsdToleranceDeg = 12;
  delete (b.phone as Partial<typeof b.phone>).imuEnabled;
  const diff = configDiff(f.config, b);
  assert.deepEqual(diff, [
    'camera.common.lsdToleranceDeg: 9.5 -> 12',
    'phone.imuEnabled: false -> undefined',
  ]);
  assert.deepEqual(configDiff(f.config, structuredClone(f.config)), []);
});

test('a harness input carries the capture aspect, not a configured viewport', () => {
  const f = loadFixture();
  const input = loadInput();
  assert.equal(input.aspect, f.capture.w / f.capture.h);
  assert.equal(input.label, 'fixture:default');
});
