// ── WebGPU device bootstrap + the few buffer helpers poseViewer still uses ──
//
// Lazily requested on first use, not at module load -- WebGPU may not be
// available (older browser, older iOS), so nothing should touch navigator.gpu
// until the GPU path is actually asked for.
//
// ── WHAT THIS FILE USED TO BE ──
//
// It lived at the old pipeline's `gpu/device.ts` and was 516 lines. It moved here when
// the old pipeline was deleted, because it never was pose math: it is generic device
// plumbing, and poseViewer's OWN projection overlay (pipeline/projectSamples.gpu.ts)
// depends on it and always did.
//
// Roughly three quarters of it was instrumentation for the old pipeline's
// performance investigation, and all of it went with that pipeline:
//
//   - the TRANSFER LEDGER and its byte-vs-fence probe, which existed to answer
//     "what does a bus crossing actually cost -- bytes, or the fence". Its only
//     consumer was harness/reconstructionTiming.ts;
//   - the ALLOCATION PROBE, which sized the case for a persistent buffer arena.
//     The arena it argued for was built, then deleted with the rest;
//   - the SLICE-AWARE transfers (`writeSlice`, `readSliceF32/U32`), which were
//     the arena's read path;
//   - the TIMESTAMP QUERY primitives, whose only consumer was gpuTimeline.ts.
//
// None of that is a loss to mourn here: `src/pose` does its own timing, has no
// arena, and crosses the bus exactly once per frame by construction, which is
// the property that made the whole ledger unnecessary rather than merely unused.
//
// What remains is what one caller needs, plus `isWebGPUSupported` for the UI.

let devicePromise: Promise<GPUDevice | null> | null = null;

export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

// ── The one device-creation decision that cannot be revisited per frame ──
//
// `timestamp-query` must be asked for HERE, at requestDevice, or the device
// simply cannot write a timestamp for the rest of its life. Everything else
// about GPU timing is a per-submit choice; this is not one, which is why it is
// settled at the earliest possible point and by a single function.
//
// REQUESTED WHEN OFFERED, NEVER REQUIRED. A feature named in `requiredFeatures`
// that the adapter does not expose makes requestDevice REJECT -- so requiring it
// unconditionally would turn "this machine cannot be profiled" into "this
// machine cannot run the pipeline", trading the whole app for an instrument.
// The check-then-request below is what keeps the absence merely uninstrumented.
//
// Three callers, and they must agree: the app (getGPUDevice), the sweep, and the
// test helper. They agree by calling this rather than by three copies of the
// same two lines -- the sweep and the tests are exactly where GPU timing gets
// read, so a site that quietly skipped the feature would present as "timing is
// unavailable on this machine" rather than as the omission it is.
export async function requestDeviceWithOptionalTimestamps(adapter: GPUAdapter): Promise<GPUDevice> {
  const wanted: GPUFeatureName[] = adapter.features.has('timestamp-query')
    ? ['timestamp-query']
    : [];
  return await adapter.requestDevice({ requiredFeatures: wanted });
}

// ── Dawn's timestamps are QUANTIZED unless you say otherwise ─────────────
//
// **The single most important fact about GPU timing in this project, and it
// cost a long misdiagnosis to find.** Dawn ships a `timestamp_quantization`
// toggle, ON by default, which is a side-channel mitigation: it rounds every
// timestamp to a coarse grid. Measured here, that grid is **65536 ns**, against
// passes that take single-digit microseconds -- so a pass's begin and end land
// in the SAME tick and its duration reads exactly **zero**. A typical frame came
// back with 114 of 136 passes at zero and five distinct values in total.
//
// It does not fail loudly. It returns plausible-looking numbers that are almost
// all zero, which reads as "the GPU is idle" or "the counter degraded" rather
// than as a setting.
//
// Passed at INSTANCE creation, not device creation, so it belongs to whoever
// calls `create()` -- both node entry points (the test helper and the sweep)
// share this constant so they cannot drift apart. It is node-only: the browser
// has no `create()`, and see below for what the app needs instead.
//
// ── THE BROWSER NEEDS THE SAME THING, BY A DIFFERENT ROUTE ──
//
// Chrome applies the same mitigation. An app-side timing capture must run Chrome
// with `--disable-dawn-features=timestamp_quantization` (or
// `--enable-webgpu-developer-features`), or every number it reports will be on
// the 65536 ns grid. **This is UNVERIFIED in Chrome as of 2026-08-15** -- it is
// the reason to check before trusting an app-side GPU breakdown.
export const DAWN_NODE_FLAGS = ['disable-dawn-features=timestamp_quantization'];

// Whether a device can time a pass. Read this, not the adapter: a device only
// has the features it was CREATED with, so an adapter that offers the feature
// says nothing about a device that did not ask for it. Anything encoding
// timestamp writes must gate on this, since doing so on a device without the
// feature is a validation error rather than a no-op.
export function canTimestamp(device: GPUDevice): boolean {
  return device.features.has('timestamp-query');
}

export async function getGPUDevice(): Promise<GPUDevice | null> {
  if (!devicePromise) {
    devicePromise = (async () => {
      if (!isWebGPUSupported()) return null;
      try {
        const adapter = await navigator.gpu!.requestAdapter();
        if (!adapter) return null;
        const device = await requestDeviceWithOptionalTimestamps(adapter);
        device.lost.then((info) => {
          console.error('[gpu] WebGPU device lost:', info.message);
          devicePromise = null; // let a later call re-request a fresh device
        });
        return device;
      } catch (e) {
        console.error('[gpu] failed to acquire a WebGPU device:', e);
        return null;
      }
    })();
  }
  return devicePromise;
}

export function uploadFloat32(device: GPUDevice, data: Float32Array, extraUsage = 0): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

export function createStorageBuffer(device: GPUDevice, byteLength: number, extraUsage = 0): GPUBuffer {
  return device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage,
  });
}

// WGSL uniform buffers use std140-like alignment (vec4/f32 pack cleanly, vec3
// does not) -- every uniform struct on this path is written out as plain
// vec4-or-scalar fields specifically to avoid that trap, so a bare byte copy
// here is always safe.
export function uploadUniform(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
  const buffer = device.createBuffer({
    size: Math.ceil(data.byteLength / 16) * 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
  buffer.unmap();
  return buffer;
}

// The point where the GPU's result crosses back into JS-visible memory.
// mapAsync is the real cost, not the copyBufferToBuffer -- that is device-local
// and effectively free next to a round trip through the driver.
async function readAt<T extends Float32Array | Uint32Array>(
  device: GPUDevice, buffer: GPUBuffer, offset: number, byteLength: number,
  View: { new (b: ArrayBuffer): T },
): Promise<T> {
  const staging = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, offset, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new View(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}

export async function readFloat32(device: GPUDevice, buffer: GPUBuffer, byteLength: number): Promise<Float32Array> {
  return await readAt(device, buffer, 0, byteLength, Float32Array);
}

export async function readUint32(device: GPUDevice, buffer: GPUBuffer, byteLength: number): Promise<Uint32Array> {
  return await readAt(device, buffer, 0, byteLength, Uint32Array);
}

const WORKGROUP_SIZE = 8; // 8x8 = 64 threads/workgroup, a safe default across desktop and mobile GPUs
export function dispatchCount(dim: number): number {
  return Math.ceil(dim / WORKGROUP_SIZE);
}
