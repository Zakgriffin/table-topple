import { create, globals } from 'webgpu';

// ── A real GPUDevice under `node --test` ──────────────────────────────────
//
// Dawn -- the same engine Chrome uses -- as a Node addon, so a stage test runs
// the ACTUAL shader against the ACTUAL validator without a browser.
//
// ── Which package, and why not the one the plan named ──
//
// `@webgpu/dawn-node` is what full_system_breakdown.md §19 names, and it is not
// installable: it lives in the Dawn source tree and has never been published to
// npm, so using it means a depot_tools + CMake build. `webgpu` (the package,
// github.com/dawn-gpu/node-webgpu) is that binding, packaged -- maintained by
// the Dawn and WebGPU-spec authors, with the binary bundled rather than
// downloaded, and its only install script strips the macOS quarantine
// attribute.
//
// `@kmamal/gpu` was tried first and rejected: it works, but a process holding
// one of its instances never exits, which hangs `node --test` with no output.
//
// ── Why every body gets an error scope ──
//
// A WebGPU validation failure is reported ASYNCHRONOUSLY. It does not throw --
// it makes every command using the offending resource a silent no-op, so the
// symptom is plausible-looking zeros, not an exception. Confirmed here on the
// first probe: a `layout: 'auto'` mismatch across two entry points returned
// zeros with nothing raised anywhere, and only an error scope revealed it.
//
// That is why withDevice wraps each body in one. A stage test that merely
// asserted on numbers would PASS against a pipeline that never ran -- and for
// most of this pipeline, zero is a plausible answer.

// The flag objects a browser puts on `window`. Installed once, globally,
// because src/pose2 is written against the browser spelling and should not know
// it is under test.
Object.assign(globalThis, globals);

// ── THE GPU INSTANCE MUST BE RETAINED, AND THIS IS NOT OPTIONAL ──────────
//
// `create()` returns the GPU instance that owns the adapter and device. Writing
// `create([]).requestAdapter()` drops the only reference to it, and once V8
// collects it the native side is freed underneath the still-live device: the
// next mapAsync segfaults the process.
//
// It presents as flaky infrastructure rather than as a bug. The crash lands
// whenever GC happens to run, so a single test passes, two tests pass
// sometimes, and a longer run dies -- which reads exactly like "Dawn cannot do
// more than one pipeline run per process". It is not; it is this line.
//
// Module scope is the retention. Do not inline it.
const gpuInstance = create([]);

let cached: GPUDevice | null = null;
let unavailable: string | null = null;

/**
 * One device for the whole test process. Sharing rather than creating one per
 * file is not only faster: a test that corrupts device state stays visible to
 * the next test instead of being hidden by a fresh device, which is the same
 * reason the pipeline itself gets a run-twice check.
 */
export async function getTestDevice(): Promise<GPUDevice | null> {
  if (cached) return cached;
  if (unavailable) return null;
  try {
    const adapter = await gpuInstance.requestAdapter();
    if (!adapter) { unavailable = 'no adapter'; return null; }
    cached = await adapter.requestDevice();
    return cached;
  } catch (e) {
    unavailable = String(e);
    return null;
  }
}

/**
 * Runs `body` against a device with a validation error scope around it, and
 * fails if anything inside failed validation -- even when the body's own
 * assertions passed. See this file's header for why that is the load-bearing
 * part rather than a nicety.
 */
export async function withDevice(body: (device: GPUDevice) => Promise<void>): Promise<void> {
  const device = await getTestDevice();
  if (!device) throw new Error(`no WebGPU device available for tests (${unavailable})`);
  device.pushErrorScope('validation');
  try {
    await body(device);
  } finally {
    const err = await device.popErrorScope();
    if (err) throw new Error(`WebGPU validation error during the test body: ${err.message}`);
  }
}

/** Copies `bytes` out of `src` and returns them. One fence; tests only. */
export async function readBack(device: GPUDevice, src: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, bytes);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  // Copied out of the mapped range rather than viewed onto it: the range stops
  // being valid the moment the buffer is unmapped.
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}

export async function readF32(device: GPUDevice, src: GPUBuffer, count: number): Promise<Float32Array> {
  return new Float32Array(await readBack(device, src, count * 4));
}

export async function readU32(device: GPUDevice, src: GPUBuffer, count: number): Promise<Uint32Array> {
  return new Uint32Array(await readBack(device, src, count * 4));
}
