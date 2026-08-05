import { createStorageBuffer, getGPUDevice, readUint32, uploadUint32, uploadUniform } from './device.ts';
import { PREFIX_SUM_ADD_WGSL, PREFIX_SUM_SCAN_WGSL } from './prefixSum.wgsl.ts';

export const SCAN_BLOCK = 256;

interface ScanPipelines {
  scanLayout: GPUBindGroupLayout; scan: GPUComputePipeline;
  addLayout: GPUBindGroupLayout; add: GPUComputePipeline;
}

const cache = new WeakMap<GPUDevice, ScanPipelines>();
function getPipelines(device: GPUDevice): ScanPipelines {
  let p = cache.get(device);
  if (!p) {
    const entry = (i: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
      binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type },
    });
    const scanLayout = device.createBindGroupLayout({
      label: 'prefixSum.scan',
      entries: [entry(0, 'uniform'), entry(1, 'read-only-storage'), entry(2, 'storage'), entry(3, 'storage')],
    });
    const addLayout = device.createBindGroupLayout({
      label: 'prefixSum.add',
      entries: [
        entry(0, 'uniform'), entry(1, 'read-only-storage'), entry(2, 'storage'),
        entry(3, 'read-only-storage'), entry(4, 'storage'),
      ],
    });
    p = {
      scanLayout,
      scan: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [scanLayout] }),
        compute: { module: device.createShaderModule({ code: PREFIX_SUM_SCAN_WGSL, label: 'prefixSum.scan' }), entryPoint: 'scanBlock' },
        label: 'prefixSum.scanBlock',
      }),
      addLayout,
      add: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [addLayout] }),
        compute: { module: device.createShaderModule({ code: PREFIX_SUM_ADD_WGSL, label: 'prefixSum.add' }), entryPoint: 'addOffsets' },
        label: 'prefixSum.addOffsets',
      }),
    };
    cache.set(device, p);
  }
  return p;
}

function uniformFor(device: GPUDevice, n: number): GPUBuffer {
  const buf = new ArrayBuffer(16);
  new DataView(buf).setUint32(0, n, true);
  return uploadUniform(device, buf);
}

// Encodes an EXCLUSIVE prefix sum of `inBuf[0..n)` into `outBuf`, plus the grand
// total into `totalBuf[0]`, onto a command encoder the CALLER owns and submits.
//
// Encoding rather than submitting is the point: the first real consumer
// (collectRegionsFromLabels' CSR build) wants the histogram, this scan, and the
// scatter that follows in ONE submission, and a scan that submitted internally
// would force a needless GPU/CPU boundary between them.
//
// `inBuf` and `outBuf` must be DISTINCT buffers -- addOffsets reads inBuf[n-1]
// after outBuf has been overwritten, so aliasing them would corrupt the total.
// `inBuf` needs read-only-storage usage; `outBuf` and `totalBuf` need storage.
//
// Returns the temporaries it allocated. The caller must destroy them AFTER the
// submission completes -- they are live GPU-side until then, so destroying them
// before submit (or before an await on the result) kills the scan.
export function encodeExclusiveScan(
  device: GPUDevice, encoder: GPUCommandEncoder,
  inBuf: GPUBuffer, outBuf: GPUBuffer, totalBuf: GPUBuffer, n: number,
): GPUBuffer[] {
  if (n <= 0) return [];
  const p = getPipelines(device);
  const temps: GPUBuffer[] = [];
  const numBlocks = Math.ceil(n / SCAN_BLOCK);

  const blockSums = createStorageBuffer(device, numBlocks * 4);
  // Zero-initialized per the WebGPU spec, which is what makes the single-block
  // base case correct with no special-casing: blockOffsets stays all-zero and
  // addOffsets adds nothing.
  const blockOffsets = createStorageBuffer(device, numBlocks * 4);
  const uni = uniformFor(device, n);
  temps.push(blockSums, blockOffsets, uni);

  {
    const bg = device.createBindGroup({
      layout: p.scanLayout,
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: blockSums } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(p.scan);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(numBlocks);
    pass.end();
  }

  // RECURSION, in the host rather than the shader: blockSums is just another
  // array to scan, so the same routine handles it, and the depth adapts to n
  // instead of assuming "two levels is always enough". 196k elements is 768
  // blocks is 3 blocks is 1. Each level is its own compute pass, which is also
  // what supplies the storage-buffer barrier between a level and its parent.
  if (numBlocks > 1) {
    // A throwaway total for the inner level -- only the outermost one is the
    // caller's, and the inner sums' grand total is already blockSums' own.
    const innerTotal = createStorageBuffer(device, 4);
    temps.push(innerTotal, ...encodeExclusiveScan(device, encoder, blockSums, blockOffsets, innerTotal, numBlocks));
  }

  {
    const bg = device.createBindGroup({
      layout: p.addLayout,
      entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: outBuf } },
        { binding: 3, resource: { buffer: blockOffsets } },
        { binding: 4, resource: { buffer: totalBuf } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(p.add);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(numBlocks);
    pass.end();
  }
  return temps;
}

// Standalone convenience wrapper: scan a plain array, get the scan and the total
// back. Not for pipeline use (it submits and reads back on its own); this is
// what the verification harness and any ad-hoc check calls.
export async function exclusiveScanU32(values: Uint32Array): Promise<{ scan: Uint32Array; total: number } | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const n = values.length;
  if (n === 0) return { scan: new Uint32Array(0), total: 0 };
  device.pushErrorScope('validation');

  const inBuf = uploadUint32(device, values, 0, 'prefixSum:in');
  const outBuf = createStorageBuffer(device, n * 4);
  const totalBuf = createStorageBuffer(device, 4);
  const encoder = device.createCommandEncoder();
  const temps = encodeExclusiveScan(device, encoder, inBuf, outBuf, totalBuf, n);
  device.queue.submit([encoder.finish()]);

  const [scan, total] = await Promise.all([
    readUint32(device, outBuf, n * 4, 'prefixSum:out'),
    readUint32(device, totalBuf, 4, 'prefixSum:total'),
  ]);
  for (const b of [inBuf, outBuf, totalBuf, ...temps]) b.destroy();

  const err = await device.popErrorScope();
  if (err) {
    console.error('exclusiveScanU32: WebGPU validation error --', err.message);
    return null;
  }
  return { scan: new Uint32Array(scan), total: total[0] };
}
