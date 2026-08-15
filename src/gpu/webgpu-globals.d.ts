// TypeScript 6.0.3's bundled DOM lib has the WebGPU *interface* types
// (GPUDevice, GPUBuffer, etc.) but not yet the runtime const-flag globals a
// real browser provides (window.GPUBufferUsage and friends) -- @webgpu/types
// conflicts with the interfaces TS already has rather than filling the gap,
// so this just declares the handful of flag objects we actually use,
// matching the stable, spec-defined bit values every WebGPU implementation
// uses. Safe to delete once a TypeScript version ships these natively.

declare const GPUBufferUsage: {
  readonly MAP_READ: number; readonly MAP_WRITE: number;
  readonly COPY_SRC: number; readonly COPY_DST: number;
  readonly INDEX: number; readonly VERTEX: number;
  readonly UNIFORM: number; readonly STORAGE: number;
  readonly INDIRECT: number; readonly QUERY_RESOLVE: number;
};
declare const GPUShaderStage: {
  readonly VERTEX: number; readonly FRAGMENT: number; readonly COMPUTE: number;
};
declare const GPUMapMode: {
  readonly READ: number; readonly WRITE: number;
};
declare const GPUColorWrite: {
  readonly RED: number; readonly GREEN: number; readonly BLUE: number; readonly ALPHA: number; readonly ALL: number;
};
declare const GPUTextureUsage: {
  readonly COPY_SRC: number; readonly COPY_DST: number;
  readonly TEXTURE_BINDING: number; readonly STORAGE_BINDING: number; readonly RENDER_ATTACHMENT: number;
};
