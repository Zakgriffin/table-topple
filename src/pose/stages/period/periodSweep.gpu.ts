import {
  createStorageBuffer, getGPUDevice, readFloat32, uploadFloat32, uploadUniform,
} from '../../gpu/device.ts';
import { PERIOD_SWEEP_WGSL } from './periodSweep.wgsl.ts';

interface SweepPipeline { bindGroupLayout: GPUBindGroupLayout; pipeline: GPUComputePipeline }

const cache = new WeakMap<GPUDevice, SweepPipeline>();
function getPipeline(device: GPUDevice): SweepPipeline {
  let p = cache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: PERIOD_SWEEP_WGSL, label: 'periodSweep' });
    // Explicit layout rather than 'auto' -- see growRegions.ts's getPipelines
    // for what 'auto' cost us there. Only one entry point here, so 'auto' would
    // in fact be safe; the explicit layout is for consistency and because it
    // makes the read-only/read-write split of each binding visible at a glance.
    const bindGroupLayout = device.createBindGroupLayout({
      label: 'periodSweep',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    p = {
      bindGroupLayout,
      pipeline: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module, entryPoint: 'sweep' }, label: 'periodSweep.sweep',
      }),
    };
    cache.set(device, p);
  }
  return p;
}

// GPU-resident counterpart to the coarse period sweep inside
// pose/stages/period/gridPeriodPhase.ts's computeGridPeriodPhase -- the loop that scores
// every integer cell-count candidate n in [nMin, nMax] by resultantAt(spread/n).
//
// Scope is deliberately JUST the sweep. Peak finding, the distinctness test and
// the golden-section refinement all stay on CPU: the first two are trivial next
// to the sweep, and the refinement is inherently SERIAL (each golden-section
// probe depends on the previous comparison), so it would be a per-iteration
// round trip for two circularFits' worth of work.
//
// A good island in the sense that matters for transfer cost: input is a few KB
// of per-line values regardless of image size, output is one f32 per candidate.
// Nothing here touches the LSD chain's buffers, so it never needs to be part of
// the residency work -- see this file's sibling growRegions.ts.
//
// `scaled` must already be (value - minValue)/spread with row and col families
// concatenated in that order -- see the shader's header for why the host does
// that division in f64 rather than the shader doing it in f32.
//
// Returns scores in ASCENDING PERIOD order (index 0 == n=nMax), matching the
// order computeGridPeriodPhase builds coarseSamples in. Null if WebGPU is
// unavailable or the dispatch failed validation.
export async function sweepResultantsGPU(
  scaled: Float32Array, weights: Float32Array, rowCount: number, colCount: number,
  nMin: number, nMax: number,
): Promise<Float32Array | null> {
  const candidateCount = nMax - nMin + 1;
  if (candidateCount <= 0) return null;
  const device = await getGPUDevice();
  if (!device) return null;
  device.pushErrorScope('validation');
  const { pipeline, bindGroupLayout } = getPipeline(device);

  const scaledBuf = uploadFloat32(device, scaled, 0, 'sweep:scaled');
  const weightsBuf = uploadFloat32(device, weights, 0, 'sweep:weights');
  const scoresBuf = createStorageBuffer(device, candidateCount * 4);

  const uni = new ArrayBuffer(16);
  const dv = new DataView(uni);
  dv.setUint32(0, rowCount, true); dv.setUint32(4, colCount, true);
  dv.setUint32(8, nMin, true); dv.setUint32(12, nMax, true);
  const uniBuf = uploadUniform(device, uni);

  const bg = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniBuf } },
      { binding: 1, resource: { buffer: scaledBuf } },
      { binding: 2, resource: { buffer: weightsBuf } },
      { binding: 3, resource: { buffer: scoresBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  // One workgroup per candidate -- NOT dispatchCount(candidateCount). The
  // workgroup IS the unit of work here (its 64 threads cooperate on one
  // candidate's reduction), so the dispatch size is the candidate count itself.
  pass.dispatchWorkgroups(candidateCount);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const scores = await readFloat32(device, scoresBuf, candidateCount * 4, 'sweep:scores');
  for (const b of [scaledBuf, weightsBuf, scoresBuf, uniBuf]) b.destroy();

  const err = await device.popErrorScope();
  if (err) {
    console.error('sweepResultantsGPU: WebGPU validation error, falling back to CPU --', err.message);
    return null;
  }
  return scores;
}
