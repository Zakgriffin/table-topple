// ── WebGPU device bootstrap ────────────────────────────────────────────────
//
// Lazily requested on first use, not at module load -- WebGPU may not be
// available (older browser, older iOS -- see this session's chat for why
// that's no longer a blanket "iPhone can't do this" the way it used to be),
// and the CPU/GPU pipeline choice is a manual toggle (see state.ts), so
// nothing should touch navigator.gpu until the GPU path is actually asked
// for.

import { spanEnd, spanStart } from '../profiling/profiler.ts';

let devicePromise: Promise<GPUDevice | null> | null = null;

export function isWebGPUSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export async function getGPUDevice(): Promise<GPUDevice | null> {
  if (!devicePromise) {
    devicePromise = (async () => {
      if (!isWebGPUSupported()) return null;
      try {
        const adapter = await navigator.gpu!.requestAdapter();
        if (!adapter) return null;
        // 'timestamp-query' is optional -- request it opportunistically so
        // the profiler (see profiling/profiler.ts) can get real GPU kernel
        // timings when the browser/GPU supports it, but nothing depends on
        // it being present.
        const requiredFeatures: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
        const device = await adapter.requestDevice({ requiredFeatures });
        device.lost.then((info) => {
          console.error('[pipelineGPU] WebGPU device lost:', info.message);
          devicePromise = null; // let a later call re-request a fresh device
        });
        return device;
      } catch (e) {
        console.error('[pipelineGPU] failed to acquire a WebGPU device:', e);
        return null;
      }
    })();
  }
  return devicePromise;
}

// ── Buffer helpers ──────────────────────────────────────────────────────
//
// Every GPU-resident intermediate in this pipeline stays a plain storage
// buffer (STORAGE | COPY_SRC | COPY_DST as needed) -- nothing is read back
// to CPU except the final vote array, see voteGeneration.ts.

export function uploadFloat32(device: GPUDevice, data: Float32Array, extraUsage = 0): GPUBuffer {
  const s = spanStart(`CPU→GPU upload (${data.byteLength}B)`);
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  spanEnd(s);
  return buffer;
}

export function createStorageBuffer(device: GPUDevice, byteLength: number, extraUsage = 0): GPUBuffer {
  return device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage,
  });
}

// WGSL uniform buffers use std140-like alignment (vec4/f32 pack cleanly,
// vec3 doesn't) -- every uniform struct in this pipeline is written out as
// plain vec4-or-scalar fields specifically to avoid that trap, so a bare
// byte copy here is always safe.
export function uploadUniform(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
  const s = spanStart(`CPU→GPU upload uniform (${data.byteLength}B)`);
  const buffer = device.createBuffer({
    size: Math.ceil(data.byteLength / 16) * 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
  buffer.unmap();
  spanEnd(s);
  return buffer;
}

// The one point in this whole pipeline where the GPU's result has to cross
// back into JS-visible memory -- mapAsync is the real cost here, not the
// copyBufferToBuffer (device-local, effectively free next to a PCIe/unified-
// memory round trip through the driver). See profiling/profiler.ts's
// attachGPUKernelBreakdown for how this compares against actual kernel time.
export async function readFloat32(device: GPUDevice, buffer: GPUBuffer, byteLength: number): Promise<Float32Array> {
  const s = spanStart(`GPU→CPU readback (${byteLength}B)`);
  const staging = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  spanEnd(s);
  return result;
}

export const WORKGROUP_SIZE = 8; // 8x8 = 64 threads/workgroup, a safe default across desktop and mobile GPUs
export function dispatchCount(dim: number): number {
  return Math.ceil(dim / WORKGROUP_SIZE);
}

// ── GPU-kernel timestamp queries ────────────────────────────────────────
//
// Only meaningful if the device was granted the 'timestamp-query' feature
// above. Used by the profiler (profiling/profiler.ts) to get true GPU
// kernel execution time, as opposed to CPU-side wall-clock time around
// dispatch+submit+readback (which also includes driver/queue overhead).

export function supportsTimestampQuery(device: GPUDevice): boolean {
  return device.features.has('timestamp-query');
}

// pairCount timestamp pairs (begin/end) -- one pair per GPU pass being timed.
export function createTimestampQuerySet(device: GPUDevice, pairCount: number): GPUQuerySet {
  return device.createQuerySet({ type: 'timestamp', count: pairCount * 2 });
}

// Resolves a timestamp query set into per-pair durations, in milliseconds.
export async function resolveTimestamps(device: GPUDevice, querySet: GPUQuerySet, pairCount: number): Promise<number[]> {
  const count = pairCount * 2;
  const resolveBuf = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.resolveQuerySet(querySet, 0, count, resolveBuf, 0);
  encoder.copyBufferToBuffer(resolveBuf, 0, staging, 0, count * 8);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const raw = new BigInt64Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  resolveBuf.destroy();
  const durations: number[] = [];
  for (let i = 0; i < pairCount; i++) durations.push(Number(raw[i * 2 + 1] - raw[i * 2]) / 1e6); // ns -> ms
  return durations;
}
