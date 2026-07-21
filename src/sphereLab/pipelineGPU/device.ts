// ── WebGPU device bootstrap ────────────────────────────────────────────────
//
// Lazily requested on first use, not at module load -- WebGPU may not be
// available (older browser, older iOS -- see this session's chat for why
// that's no longer a blanket "iPhone can't do this" the way it used to be),
// and the CPU/GPU pipeline choice is a manual toggle (see state.ts), so
// nothing should touch navigator.gpu until the GPU path is actually asked
// for.

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
        const device = await adapter.requestDevice();
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

// WGSL uniform buffers use std140-like alignment (vec4/f32 pack cleanly,
// vec3 doesn't) -- every uniform struct in this pipeline is written out as
// plain vec4-or-scalar fields specifically to avoid that trap, so a bare
// byte copy here is always safe.
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

export async function readFloat32(device: GPUDevice, buffer: GPUBuffer, byteLength: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return result;
}

export const WORKGROUP_SIZE = 8; // 8x8 = 64 threads/workgroup, a safe default across desktop and mobile GPUs
export function dispatchCount(dim: number): number {
  return Math.ceil(dim / WORKGROUP_SIZE);
}
