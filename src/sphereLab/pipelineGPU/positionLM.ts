import * as THREE from 'three';
import { solveLinearSystem } from '../pipeline/orientationLM.ts';
import { attachGPUKernelBreakdown, profilerEnabled, ProfileSpan, spanEnd, spanStart } from '../profiling/profiler.ts';
import { OrientationFit, PhotometricSample, PositionFit } from '../types.ts';
import { createStorageBuffer, createTimestampQuerySet, dispatchCount, getGPUDevice, readFloat32, resolveTimestamps, supportsTimestampQuery, uploadFloat32, uploadUniform } from './device.ts';
import { PHOTOMETRIC_RESIDUALS_WGSL } from './positionLM.wgsl.ts';

const pipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();
function getPipeline(device: GPUDevice): GPUComputePipeline {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: PHOTOMETRIC_RESIDUALS_WGSL, label: 'photometricResiduals' });
    p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: 'photometricResiduals' });
    pipelineCache.set(device, p);
  }
  return p;
}

// The torus brightness pattern never changes during the app's lifetime --
// upload once per device and reuse across every call, same reasoning as the
// pipeline cache above.
const torusBufCache = new WeakMap<GPUDevice, GPUBuffer>();
function getTorusBuffer(device: GPUDevice, torus: Uint8Array[], R: number, C: number): GPUBuffer {
  let buf = torusBufCache.get(device);
  if (!buf) {
    const data = new Float32Array(R * C);
    for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) data[r * C + c] = torus[r][c] ? 20 : 235;
    buf = uploadFloat32(device, data);
    torusBufCache.set(device, buf);
  }
  return buf;
}

function buildP3Uniforms(
  w: number, h: number, sampleCount: number, torusR: number, torusC: number,
  distance: number, vFovRad: number, aspect: number, minGrazingCos: number, epsRot: number, epsPos: number,
  wx0: number, wz0: number, q: THREE.Quaternion, camQuat: THREE.Quaternion,
  drow0: THREE.Vector3, dcol0: THREE.Vector3, dnormal0: THREE.Vector3,
): ArrayBuffer {
  const buf = new ArrayBuffer(144);
  const dv = new DataView(buf);
  dv.setFloat32(0, w, true); dv.setFloat32(4, h, true); dv.setUint32(8, sampleCount, true); dv.setInt32(12, torusR, true);
  dv.setInt32(16, torusC, true); dv.setFloat32(20, distance, true); dv.setFloat32(24, vFovRad, true); dv.setFloat32(28, aspect, true);
  dv.setFloat32(32, minGrazingCos, true); dv.setFloat32(36, epsRot, true); dv.setFloat32(40, epsPos, true); // 44 pad
  dv.setFloat32(48, wx0, true); dv.setFloat32(52, wz0, true); // 56/60 pad
  dv.setFloat32(64, q.x, true); dv.setFloat32(68, q.y, true); dv.setFloat32(72, q.z, true); dv.setFloat32(76, q.w, true);
  dv.setFloat32(80, camQuat.x, true); dv.setFloat32(84, camQuat.y, true); dv.setFloat32(88, camQuat.z, true); dv.setFloat32(92, camQuat.w, true);
  dv.setFloat32(96, drow0.x, true); dv.setFloat32(100, drow0.y, true); dv.setFloat32(104, drow0.z, true);
  dv.setFloat32(112, dcol0.x, true); dv.setFloat32(116, dcol0.y, true); dv.setFloat32(120, dcol0.z, true);
  dv.setFloat32(128, dnormal0.x, true); dv.setFloat32(132, dnormal0.y, true); dv.setFloat32(136, dnormal0.z, true);
  return buf;
}

// GPU-resident counterpart to pipeline/positionLM.ts's
// refineOrientationAndPositionLM -- see positionLM.wgsl.ts's header comment
// for exactly what stays on GPU (the per-sample residual + all 5
// finite-difference Jacobian-column evaluations, one dispatch per
// iteration) vs what stays on CPU (the JtJ/Jtr reduction, linear solve, and
// the LM accept/reject/lambda control flow -- all tiny, all inherently
// sequential across iterations). Returns null if WebGPU isn't available;
// caller falls back to the CPU version, which stays the source of truth.
export async function refineOrientationAndPositionLMGPU(
  samples: PhotometricSample[], w: number, h: number,
  initial: OrientationFit, distance: number, initialWorldX0: number, initialWorldZ0: number,
  camQuat: THREE.Quaternion, vFovRad: number, aspect: number,
  torus: Uint8Array[], torusR: number, torusC: number,
  maxIterations = 20,
): Promise<(PositionFit & { iterations: number; initialCost: number; finalCost: number }) | null> {
  const device = await getGPUDevice();
  if (!device) return null;
  const pipeline = getPipeline(device);
  const torusBuf = getTorusBuffer(device, torus, torusR, torusC);

  const n = samples.length;
  const pxBuf = uploadFloat32(device, Float32Array.from(samples, (s) => s.px));
  const pyBuf = uploadFloat32(device, Float32Array.from(samples, (s) => s.py));
  const obsBuf = uploadFloat32(device, Float32Array.from(samples, (s) => s.observed));
  const outBuf = createStorageBuffer(device, n * 6 * 8); // vec2<f32> x 6 per sample

  const MIN_GRAZING_COS = 0.15;
  const EPS_ROT = 1e-5, EPS_POS = 1e-3;
  const Drow0 = initial.Drow.clone(), Dcol0 = initial.Dcol.clone(), Dnormal0 = initial.Dnormal.clone();

  // Runs one dispatch: residual + 5 Jacobian-column evaluations for every
  // sample, at the given candidate pose. Returns the raw vec2(residual,valid)
  // sextuple per sample, flattened. `label` is only used for profiler output.
  async function evalResiduals(q: THREE.Quaternion, wx0: number, wz0: number, label: string): Promise<Float32Array> {
    const dispatchSpan = spanStart(`${label} (GPU round-trip)`);
    const uniformsBuf = uploadUniform(device!, buildP3Uniforms(
      w, h, n, torusR, torusC, distance, vFovRad, aspect, MIN_GRAZING_COS, EPS_ROT, EPS_POS,
      wx0, wz0, q, camQuat, Drow0, Dcol0, Dnormal0,
    ));
    const bindGroup = device!.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [pxBuf, pyBuf, obsBuf, torusBuf, outBuf].map((buffer, i) => ({ binding: i + 1, resource: { buffer } }))
        .concat([{ binding: 0, resource: { buffer: uniformsBuf } }]),
    });
    const wantTimestamps = profilerEnabled() && supportsTimestampQuery(device!);
    const querySet = wantTimestamps ? createTimestampQuerySet(device!, 1) : null;
    const encoder = device!.createCommandEncoder();
    const pass = encoder.beginComputePass(querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchCount(n));
    pass.end();
    device!.queue.submit([encoder.finish()]);
    if (querySet) {
      const [durationMs] = await resolveTimestamps(device!, querySet, 1);
      attachGPUKernelBreakdown([{ name: `${label} kernel`, durationMs }]);
      querySet.destroy();
    }
    const raw = await readFloat32(device!, outBuf, n * 6 * 8);
    uniformsBuf.destroy();
    spanEnd(dispatchSpan);
    return raw;
  }

  // raw is n*6 vec2s (residual,valid) flattened -- column 0 is baseline,
  // 1-3 are rotation-axis perturbations, 4-5 are position perturbations.
  function cost(raw: Float32Array, col: number): number {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const o = (i * 6 + col) * 2;
      if (raw[o + 1] === 0) continue;
      const r = raw[o];
      s += r * r;
    }
    return s;
  }

  const q = new THREE.Quaternion();
  let worldX0 = initialWorldX0, worldZ0 = initialWorldZ0;
  const candidateNormal = (qq: THREE.Quaternion) => {
    const nrm = Dnormal0.clone().applyQuaternion(qq);
    const checkDir = new THREE.Vector3(0, 0, -1).applyQuaternion(camQuat); // cornerDir(0,0,camQuat,...) simplifies to this
    if (checkDir.dot(nrm) > 0) nrm.negate();
    return nrm;
  };

  const initialRaw = await evalResiduals(q, worldX0, worldZ0, 'initial');
  const initialCost = cost(initialRaw, 0);
  let curCost = initialCost;
  let lambda = 1e-3;
  const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const P = 5;

  let iterations = 0;
  for (; iterations < maxIterations; iterations++) {
    const iterSpan: ProfileSpan | null = spanStart(`LM iter ${iterations}`);
    try {
      const raw = await evalResiduals(q, worldX0, worldZ0, `iter ${iterations} baseline`);
      // Count of valid baseline samples -- CPU's early "n===0 -> break".
      let validCount = 0;
      for (let i = 0; i < n; i++) if (raw[(i * 6) * 2 + 1] !== 0) validCount++;
      if (validCount === 0) break;

      // Build the 5 Jacobian columns, index-aligned (see positionLM.wgsl.ts's
      // header comment for why this differs slightly from the CPU's
      // post-compaction positional alignment, and why it shouldn't matter here).
      const solveSpan = spanStart('CPU (JtJ/solve)');
      const J: Float64Array[] = [];
      const epsList = [EPS_ROT, EPS_ROT, EPS_ROT, EPS_POS, EPS_POS];
      for (let colIdx = 1; colIdx <= 5; colIdx++) {
        const col = new Float64Array(n);
        const eps = epsList[colIdx - 1];
        for (let i = 0; i < n; i++) {
          const o0 = (i * 6 + 0) * 2, ok = (i * 6 + colIdx) * 2;
          if (raw[o0 + 1] === 0 || raw[ok + 1] === 0) continue;
          col[i] = (raw[ok] - raw[o0]) / eps;
        }
        J.push(col);
      }
      const r0 = new Float64Array(n);
      for (let i = 0; i < n; i++) { const o0 = (i * 6) * 2; r0[i] = raw[o0 + 1] === 0 ? 0 : raw[o0]; }

      const JtJ: number[][] = Array.from({ length: P }, () => new Array(P).fill(0));
      const Jtr: number[] = new Array(P).fill(0);
      for (let a = 0; a < P; a++) {
        for (let b = 0; b < P; b++) {
          let s = 0; for (let i = 0; i < n; i++) s += J[a][i] * J[b][i];
          JtJ[a][b] = s;
        }
        let s = 0; for (let i = 0; i < n; i++) s += J[a][i] * r0[i];
        Jtr[a] = s;
      }
      const A = JtJ.map((row, a) => row.map((v, b) => v + (a === b ? lambda * (JtJ[a][a] || 1) : 0)));
      const rhs = Jtr.map((v) => -v);
      const delta = solveLinearSystem(A, rhs);
      spanEnd(solveSpan);
      if (!delta) break;

      const deltaRotVec = new THREE.Vector3(delta[0], delta[1], delta[2]);
      const deltaRotAngle = deltaRotVec.length();
      const deltaWX = delta[3], deltaWZ = delta[4];
      if (deltaRotAngle < 1e-10 && Math.abs(deltaWX) < 1e-10 && Math.abs(deltaWZ) < 1e-10) break;

      const qTry = deltaRotAngle > 1e-12
        ? new THREE.Quaternion().setFromAxisAngle(deltaRotVec.normalize(), deltaRotAngle).multiply(q).normalize()
        : q.clone();
      const wx0Try = worldX0 + deltaWX, wz0Try = worldZ0 + deltaWZ;

      const tryRaw = await evalResiduals(qTry, wx0Try, wz0Try, `iter ${iterations} tryPoint`);
      const tryCost = cost(tryRaw, 0);
      if (tryCost < curCost) {
        q.copy(qTry); worldX0 = wx0Try; worldZ0 = wz0Try;
        curCost = tryCost;
        lambda = Math.max(lambda * 0.5, 1e-8);
      } else {
        lambda = Math.min(lambda * 3, 1e8);
      }
    } finally {
      spanEnd(iterSpan);
    }
  }

  for (const b of [pxBuf, pyBuf, obsBuf, outBuf]) b.destroy();

  return {
    Drow: Drow0.clone().applyQuaternion(q), Dcol: Dcol0.clone().applyQuaternion(q), Dnormal: candidateNormal(q),
    worldX0, worldZ0, distance,
    iterations, initialCost, finalCost: curCost,
  };
}
