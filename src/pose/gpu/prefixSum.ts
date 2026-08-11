import { type Alloc, type Arena, type Slice, allocZeroed, bind, createArena } from './arena.ts';
import { getGPUDevice, readSliceU32, writeSlice } from './device.ts';
import { PREFIX_SUM_ADD_WGSL, PREFIX_SUM_SCAN_WGSL } from './prefixSum.wgsl.ts';
import { gpuTimelineSlot } from './gpuTimeline.ts';

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

function uniformFor(arena: Arena, alloc: Alloc, n: number): Slice {
  const buf = new ArrayBuffer(16);
  new DataView(buf).setUint32(0, n, true);
  const s = alloc(16, 'scan.n');
  arena.device.queue.writeBuffer(s.buffer, s.offset, buf);
  return s;
}

// Encodes an EXCLUSIVE prefix sum of `inBuf[0..n)` into `outBuf`, plus the grand
// total into `totalBuf[0]`, onto a command encoder the CALLER owns and submits.
//
// Encoding rather than submitting is the point: the first real consumer
// (collectRegionsFromLabels' CSR build) wants the histogram, this scan, and the
// scatter that follows in ONE submission, and a scan that submitted internally
// would force a needless GPU/CPU boundary between them.
//
// `inSlice` and `outSlice` must be DISTINCT -- addOffsets reads in[n-1] after
// out has been overwritten, so aliasing them would corrupt the total. With one
// arena behind every slice that is now an ALIASING question rather than a
// buffer-identity one, and it is the caller's to get right: two separate
// alloc() calls cannot overlap, so passing two distinct slices is sufficient.
//
// Returns nothing. It used to return the temporaries it allocated, for the
// caller to destroy AFTER the submission completed -- a rule that could not be
// enforced and had to be restated at every call site. Arena temporaries are
// freed by the frame's reset(), which by construction happens after the submit.
export function encodeExclusiveScan(
  arena: Arena, alloc: Alloc, encoder: GPUCommandEncoder,
  inSlice: Slice, outSlice: Slice, totalSlice: Slice, n: number,
): void {
  if (n <= 0) return;
  const device = arena.device;
  const p = getPipelines(device);
  const numBlocks = Math.ceil(n / SCAN_BLOCK);

  const blockSums = alloc(numBlocks * 4, 'scan.blockSums');
  // ZEROED, and it is load-bearing rather than redundant now. The single-block
  // base case relies on nothing writing blockOffsets: addOffsets reads it and
  // adds zero. `createBuffer` used to guarantee that for free; an arena slice is
  // last frame's bytes. See allocZeroed's header for the whole audit.
  const blockOffsets = allocZeroed(arena, alloc, encoder, numBlocks * 4, 'scan.blockOffsets');
  const uni = uniformFor(arena, alloc, n);

  {
    const bg = device.createBindGroup({
      layout: p.scanLayout,
      entries: [
        { binding: 0, resource: bind(uni, arena) },
        { binding: 1, resource: bind(inSlice, arena) },
        { binding: 2, resource: bind(outSlice, arena) },
        { binding: 3, resource: bind(blockSums, arena) },
      ],
    });
    const pass = encoder.beginComputePass(gpuTimelineSlot('scan:block'));
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
    const innerTotal = alloc(4, 'scan.innerTotal');
    encodeExclusiveScan(arena, alloc, encoder, blockSums, blockOffsets, innerTotal, numBlocks);
  }

  {
    const bg = device.createBindGroup({
      layout: p.addLayout,
      entries: [
        { binding: 0, resource: bind(uni, arena) },
        { binding: 1, resource: bind(inSlice, arena) },
        { binding: 2, resource: bind(outSlice, arena) },
        { binding: 3, resource: bind(blockOffsets, arena) },
        { binding: 4, resource: bind(totalSlice, arena) },
      ],
    });
    const pass = encoder.beginComputePass(gpuTimelineSlot('scan:addOffsets'));
    pass.setPipeline(p.add);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(numBlocks);
    pass.end();
  }
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

  // Its OWN arena, destroyed in the finally. A standalone helper that borrowed
  // the pipeline's arena would be resetting someone else's frame.
  const arena = createArena(device, { initialBytes: n * 4 * 4 + 4096 });
  try {
    const inSlice = arena.alloc(n * 4, 'prefixSum:in');
    writeSlice(arena, inSlice, values, 'prefixSum:in');
    const outSlice = arena.alloc(n * 4, 'prefixSum:out');
    const totalSlice = arena.alloc(4, 'prefixSum:total');
    const encoder = device.createCommandEncoder();
    encodeExclusiveScan(arena, arena.alloc, encoder, inSlice, outSlice, totalSlice, n);
    device.queue.submit([encoder.finish()]);

    const [scan, total] = await Promise.all([
      readSliceU32(arena, outSlice, n * 4, 'prefixSum:out'),
      readSliceU32(arena, totalSlice, 4, 'prefixSum:total'),
    ]);

    const err = await device.popErrorScope();
    if (err) {
      console.error('exclusiveScanU32: WebGPU validation error --', err.message);
      return null;
    }
    return { scan: new Uint32Array(scan), total: total[0] };
  } finally {
    arena.destroy();
  }
}
