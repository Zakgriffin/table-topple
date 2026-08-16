import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { getTestDevice, readF32, readU32, withDevice } from './helpers/gpu.ts';
import { createBuffers, planPool } from '../src/pose/buffers.ts';
import type { Dims } from '../src/pose/pipeline.ts';

// ── Does the headless GPU harness actually work? ──────────────────────────
//
// These test the TEST INFRASTRUCTURE, not the pipeline. They exist because
// every stage test to come is worthless if the harness cannot tell a shader
// that ran from one that silently did not -- and under WebGPU those look
// identical unless something checks an error scope.

test('a device is available', async () => {
  const device = await getTestDevice();
  assert.ok(device, 'no WebGPU device -- @kmamal/gpu install script may not have run (npm approve-scripts @kmamal/gpu)');
});

test('a compute pass runs and its result reads back', async () => {
  await withDevice(async (device) => {
    const code = `
      @group(0) @binding(0) var<storage, read_write> data: array<f32>;
      @compute @workgroup_size(8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
        data[g.x] = data[g.x] * data[g.x];
      }`;
    const buf = device.createBuffer({
      size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    const pipeline = device.createComputePipeline({
      layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(pipeline);
    p.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] }));
    p.dispatchWorkgroups(1);
    p.end();
    device.queue.submit([enc.finish()]);
    assert.deepEqual([...await readF32(device, buf, 8)], [1, 4, 9, 16, 25, 36, 49, 64]);
    buf.destroy();
  });
});

test('indirect dispatch off a device-written args buffer', async () => {
  // The pattern the entire pipeline rests on: a pass writes workgroup counts,
  // a later pass launches over them, and the host never learns the number.
  // If this did not work, nothing in the design would.
  await withDevice(async (device) => {
    const code = `
      @group(0) @binding(0) var<storage, read_write> data: array<u32>;
      @group(1) @binding(0) var<storage, read_write> args: array<u32>;
      @compute @workgroup_size(1) fn setArgs() { args[0] = 3u; args[1] = 1u; args[2] = 1u; }
      @compute @workgroup_size(1) fn mark(@builtin(global_invocation_id) g: vec3<u32>) { data[g.x] = 7u; }`;
    const st = () => ({ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } });
    const l0 = device.createBindGroupLayout({ entries: [st()] });
    const l1 = device.createBindGroupLayout({ entries: [st()] });
    const mod = device.createShaderModule({ code });
    const pSet = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [l0, l1] }), compute: { module: mod, entryPoint: 'setArgs' } });
    const pMark = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [l0] }), compute: { module: mod, entryPoint: 'mark' } });

    const data = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const args = device.createBuffer({ size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    const bg0 = device.createBindGroup({ layout: l0, entries: [{ binding: 0, resource: { buffer: data } }] });
    const bg1 = device.createBindGroup({ layout: l1, entries: [{ binding: 0, resource: { buffer: args } }] });

    const enc = device.createCommandEncoder();
    enc.clearBuffer(data);
    let p = enc.beginComputePass();
    p.setPipeline(pSet); p.setBindGroup(0, bg0); p.setBindGroup(1, bg1); p.dispatchWorkgroups(1); p.end();
    p = enc.beginComputePass();
    p.setPipeline(pMark); p.setBindGroup(0, bg0); p.dispatchWorkgroupsIndirect(args, 0); p.end();
    device.queue.submit([enc.finish()]);

    // Exactly 3 workgroups ran, because that is what the first pass wrote.
    assert.deepEqual([...await readU32(device, data, 8)], [7, 7, 7, 0, 0, 0, 0, 0]);
    data.destroy(); args.destroy();
  });
});

test('withDevice fails a body whose commands silently no-opped', async () => {
  // THE LOAD-BEARING TEST FOR THE HARNESS. A validation failure does not throw
  // -- it makes every command using the resource a no-op, so a stage that never
  // ran returns zeros, and zeros are a plausible answer almost everywhere in
  // this pipeline. Proven here by building a bind group that fails validation
  // and asserting the harness surfaces it even though nothing else complains.
  await assert.rejects(
    () => withDevice(async (device) => {
      const code = `
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(1) fn main() { data[0] = 1.0; }`;
      const pipeline = device.createComputePipeline({
        layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
      // STORAGE is missing from the usage flags, so the bind group is invalid.
      const bad = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      const enc = device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(pipeline);
      p.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bad } }] }));
      p.dispatchWorkgroups(1);
      p.end();
      device.queue.submit([enc.finish()]);
      // Deliberately asserts nothing. A body like this would pass under a naive
      // harness; the error scope is the only thing that catches it.
    }),
    /WebGPU validation error/,
  );
});

test('the planned buffer set allocates on a real device', async () => {
  await withDevice(async (device) => {
    // Small dims -- the plan is resolution-independent and a 64x64 image keeps
    // this fast. Both modes, because alias:true is the one that shares
    // allocations and could in principle produce an invalid size.
    const dims: Dims = {
      w: 64, h: 64, maxRegions: 1024, maxLines: 1024, maxCells: 144 * 144,
      torusR: 144, torusC: 144, hashSlots: 65536,
    };
    for (const alias of [false, true]) {
      const plan = planPool(dims, { alias });
      const bufs = createBuffers(device, plan);
      for (const name of Object.keys(bufs)) assert.ok(bufs[name], `'${name}' did not allocate`);
      // Aliased names must resolve to the SAME GPUBuffer -- that is the whole
      // mechanism, and it is invisible at every use site by design.
      const multi = plan.slots.filter((s) => s.occupants.length > 1);
      if (alias) {
        assert.ok(multi.length > 0, 'alias:true produced no shared slots');
        for (const slot of multi) {
          const first = bufs[slot.occupants[0]!];
          for (const other of slot.occupants) assert.equal(bufs[other], first, `slot ${slot.index} did not share`);
        }
      } else {
        assert.equal(multi.length, 0, 'alias:false must not share any slot');
      }
      const seen = new Set(Object.values(bufs));
      for (const b of seen) b.destroy();
    }
  });
});
