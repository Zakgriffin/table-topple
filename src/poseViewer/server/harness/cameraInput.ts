import { activeCamera } from '../camera/store.ts';
import type { Camera } from '../../shared/camera/model.ts';
import { validateFixture } from '../../shared/fixture.ts';
import { inputFromFixture } from '../../shared/harness/input.ts';
import type { HarnessInput } from '../../shared/harness/input.ts';

// ── The app-side adapters, kept in ONE module on purpose ──────────────────
//
// Everything ambient that the verifies used to do individually -- reach for
// activeCamera(), pull a gray off it, know which of its two gray buffers is the
// right one -- happens here and nowhere else. That is what makes the harnesses
// themselves importable outside a browser: their coupling to the running app is
// a function you call to build their argument, not something they do on entry.
//
// So the console form gained one call:
//
//     await verifyLsdChain(cameraInput())
//     await verifyLsdChain(await fixtureInput('default'))
//
// which is the trade. A default argument of `cameraInput()` would restore the
// old one-word form and would also restore the property that made every one of
// these results un-re-derivable: the input would stop being written down.

// Throws rather than returning a string. The verifies used to return
// 'no active camera' / 'no capture yet' as an in-band VALUE, so every one of
// them had a `Report | string` return type that every caller had to narrow --
// including timeReconstruction, which forwarded the union outward. A missing
// capture is a caller error, not a result.
export function cameraInput(camera?: Camera | null): HarnessInput {
  const cam = camera ?? activeCamera() ?? null;
  if (!cam) throw new Error('cameraInput: no active camera');
  const cap = cam.lastAxesCaptureGray;
  if (!cap) throw new Error('cameraInput: no capture yet -- run a capture first');
  return {
    label: `camera:${cam.id}`,
    // lastAxesCaptureGray, NOT lastNoisedPreviewGray -- this is the top-down
    // buffer computePoseFromCapture is handed. See input.ts on why four of
    // these harnesses used to read the flipped display copy instead.
    gray: cap.gray,
    w: cap.w,
    h: cap.h,
    aspect: cam.aspect,
    settings: cam.settings,
  };
}

// Fetched from the dev server, which serves the project root at `/` -- the same
// route and the same reasoning as config.ts's CONFIG_URL (fetched rather than
// imported, so writing a fixture does not put it in vite's module graph and
// trigger a full reload that wipes the capture you were measuring).
//
// The node-side loader is a readFileSync in the test helper. Two loaders, one
// validator: the shape check is fixture.ts's, so a browser and a test cannot
// disagree about what a valid fixture is.
export async function fixtureInput(name = 'default'): Promise<HarnessInput> {
  const url = `/fixtures/${name}.json`;
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`fixtureInput: GET ${url} failed (${res.status})`);
  return inputFromFixture(validateFixture(await res.json(), url));
}
