// ── WebGPU device bootstrap ────────────────────────────────────────────────
//
// Lazily requested on first use, not at module load -- WebGPU may not be
// available (older browser, older iOS -- see this session's chat for why
// that's no longer a blanket "iPhone can't do this" the way it used to be),
// and the CPU/GPU pipeline choice is a manual toggle (see state.ts), so
// nothing should touch navigator.gpu until the GPU path is actually asked
// for.

import { spanEnd, spanStart } from '../../sphereLab/profiling/profiler.ts';

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
          console.error('[pose/gpu] WebGPU device lost:', info.message);
          devicePromise = null; // let a later call re-request a fresh device
        });
        return device;
      } catch (e) {
        console.error('[pose/gpu] failed to acquire a WebGPU device:', e);
        return null;
      }
    })();
  }
  return devicePromise;
}

// ── Transfer ledger + the byte-vs-fence probe ─────────────────────────────
//
// Every bus crossing in the pipeline funnels through the four helpers below,
// which makes this the one place that can answer the question the 2026-08-05
// reframe turns on: WHAT does a crossing actually cost -- bytes, or the fence?
//
// The two known regimes, so the numbers below are read correctly:
//   - a 4-byte readback pays fence latency plus however much work was already
//     queued ahead of it, and NO byte-proportional cost whatsoever;
//   - a 786KB readback pays that SAME fence, plus real byte-proportional CPU
//     work: `getMappedRange().slice(0)` here, and then (for f32 fields) a
//     widening loop in FieldResidency.cpu() on top.
// A flamegraph cannot separate those, because both land inside one span.
//
// PROBE MODE separates them, at the cost of making every readback ~3x slower
// (so its wall-clock column is meaningless -- read only the attribution):
//   1. a 4-byte read from the same buffer, which DRAINS whatever was queued;
//   2. a second 4-byte read, now against an empty queue -- a BARE FENCE;
//   3. the real read.
// Then `ms - bareFenceMs` is the byte-proportional cost of this specific
// transfer, and `queueDrainMs - bareFenceMs` is what was sitting in the queue
// ahead of it. Two probes rather than one because a single probe would leave
// the real read paying a fence the probe already absorbed, over-attributing a
// whole round trip to bytes.
//
// Uploads are recorded too, but they never fence (mappedAtCreation is
// synchronous), so their `ms` is pure CPU: allocation plus memcpy. The f64->f32
// NARROWING that precedes them happens in the caller, and FieldResidency
// reports that separately via recordTransfer -- see its own comment.

export interface TransferSample {
  what: string;
  // 'readback' is the ONLY kind that fences -- uploads are synchronous
  // (mappedAtCreation) and 'convert' is pure CPU. Counting fences means
  // counting readbacks, so this distinction has to be in the data rather
  // than inferred from the label.
  kind: 'readback' | 'upload' | 'convert';
  dir: 'up' | 'down';
  bytes: number;
  ms: number;
  // performance.now() when this crossing STARTED. Recorded so a consumer can
  // tell CONCURRENT crossings from sequential ones, which is the difference
  // between counting readbacks and counting STALLS -- a distinction the perf
  // notes had to keep repairing by hand ("readbacks are not fences: collect's
  // two, tally's two and the CSR's four each share ONE Promise.all after one
  // submit, so 12 readbacks are ~8 distinct stalls"). Two samples whose
  // [startMs, startMs+ms] windows overlap were waiting on the same submit, so
  // their costs OVERLAP rather than sum, and a consumer that adds them
  // double-counts. With this field that grouping is computed instead of
  // remembered.
  startMs: number;
  // Probe mode only; null otherwise. See the header for how to read them.
  bareFenceMs: number | null;
  queueDrainMs: number | null;
}

let ledger: TransferSample[] = [];
let probeEnabled = false;

export function transferLedgerReset(): void { ledger = []; }
export function transferLedger(): readonly TransferSample[] { return ledger; }
export function setTransferProbe(on: boolean): void { probeEnabled = on; }

// `label` defaults to UNLABELLED rather than to a type name so an unlabelled
// call site is visible AS a defect in the readout instead of quietly merging
// into a 'u32' bucket with ten unrelated transfers. That is not hypothetical:
// the first run of this instrument reported "11 u32 readbacks, 4.30ms of fence"
// and could not say which eleven, which is the difference between knowing there
// are 16 fences and knowing which 4 to kill.

// For byte-proportional work that happens OUTSIDE these helpers but is part of
// the same crossing -- currently FieldResidency's f64<->f32 conversions, which
// are the single largest per-crossing cost and are invisible from in here.
export function recordTransfer(s: TransferSample): void { ledger.push(s); }

// A 4-byte read from `buffer`, used only by probe mode. Every buffer reaching
// the read helpers is at least 4 bytes and already carries COPY_SRC (it is
// about to be copied from anyway), so this needs no cooperation from callers.
async function bareRead(device: GPUDevice, buffer: GPUBuffer): Promise<number> {
  const t0 = performance.now();
  const staging = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  staging.unmap();
  staging.destroy();
  return performance.now() - t0;
}

// ── Buffer helpers ──────────────────────────────────────────────────────
//
// Every GPU-resident intermediate in this pipeline stays a plain storage
// buffer (STORAGE | COPY_SRC | COPY_DST as needed). What comes back to the
// host is enumerated by the crossings table in
// sphereLab/harness/reconstructionTiming.ts -- and every one of those is
// load-bearing, i.e. the
// host needs the value to size a dispatch, terminate a loop, or finish a
// computation that has not been ported.

export function uploadFloat32(device: GPUDevice, data: Float32Array, extraUsage = 0, label = 'UNLABELLED f32'): GPUBuffer {
  const s = spanStart(`CPU→GPU upload (${data.byteLength}B)`);
  const t0 = performance.now();
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    mappedAtCreation: true,
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  ledger.push({ what: label, kind: 'upload', dir: 'up', bytes: data.byteLength, ms: performance.now() - t0, startMs: t0, bareFenceMs: null, queueDrainMs: null });
  spanEnd(s);
  return buffer;
}

export function uploadUint32(device: GPUDevice, data: Uint32Array, extraUsage = 0, label = 'UNLABELLED u32'): GPUBuffer {
  const s = spanStart(`CPU→GPU upload (${data.byteLength}B)`);
  const t0 = performance.now();
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    mappedAtCreation: true,
  });
  new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  ledger.push({ what: label, kind: 'upload', dir: 'up', bytes: data.byteLength, ms: performance.now() - t0, startMs: t0, bareFenceMs: null, queueDrainMs: null });
  spanEnd(s);
  return buffer;
}

// ── The allocation probe ─────────────────────────────────────────────────
//
// Answers the question perf-TODO item 5 (the persistent buffer arena) rests on
// and that NOTHING else can currently answer: what does creating and destroying
// ~60 buffers per reconstruction actually cost?
//
// It is genuinely invisible to every other instrument. The transfer ledger above
// counts BYTES MOVED, so an allocation that is never uploaded or read back --
// which is most of them, since intermediates stay device-side by design -- does
// not appear in it at all. The probe's three buckets (GPU compute / fence /
// bytes) have no allocation column either, so this cost is sitting inside the
// unattributed remainder of the median. It may well be ZERO: drivers commonly
// pool same-size allocations, and every one of these buffers is recreated at an
// identical size every frame, which is the easiest possible case to pool. That
// is the outcome worth finding out cheaply, because it retires the arena.
//
// PATCHES device.createBuffer AND GPUBuffer.prototype.destroy rather than
// wrapping createStorageBuffer and friends, and that is the point rather than a
// shortcut: the allocation sites are spread across nine modules and several of
// them (prefixSum's recursion, device.ts's own staging buffers) do not go
// through a shared helper. A wrapper can MISS a site and quietly under-report;
// a patch on the constructor cannot. It also catches whatever a future stage
// adds without that stage having to know this exists.
//
// Cost of the instrument itself is one performance.now() pair per call, so
// unlike the transfer probe it does not distort what it measures -- but it still
// runs in its own rep, because mixing it into the timed reps would put its
// overhead into the headline median for no reason.
export interface AllocationSample {
  creates: number;
  createMs: number;
  bytesAllocated: number;
  destroys: number;
  destroyMs: number;
  // Same-size-and-usage repeats, i.e. how much of the churn a driver-side pool
  // or an arena could plausibly serve. A high number here is the ARGUMENT for
  // item 5; a low one means the sizes genuinely vary and an arena would have to
  // take worst-case bounds to help at all.
  repeatedShapes: number;
  // The biggest single allocations, descending -- what an arena would target
  // first. Keyed by size+usage.
  top: { bytes: number; usage: number; count: number }[];
}

let alloc: AllocationSample | null = null;
let shapeCounts: Map<string, { bytes: number; usage: number; count: number }> | null = null;
let restoreAlloc: (() => void) | null = null;

export function allocationProbeResult(): AllocationSample | null { return alloc; }

// Installs (on=true) or removes the patches. Safe to call unbalanced -- an
// install over an existing one restores first, so a throw between the two never
// leaves createBuffer permanently wrapped.
export function setAllocationProbe(device: GPUDevice, on: boolean): void {
  restoreAlloc?.();
  restoreAlloc = null;
  if (!on) return;

  alloc = { creates: 0, createMs: 0, bytesAllocated: 0, destroys: 0, destroyMs: 0, repeatedShapes: 0, top: [] };
  shapeCounts = new Map();

  const realCreate = device.createBuffer.bind(device);
  const proto = (globalThis as { GPUBuffer?: { prototype: GPUBuffer } }).GPUBuffer?.prototype;
  const realDestroy = proto?.destroy;

  device.createBuffer = (desc: GPUBufferDescriptor): GPUBuffer => {
    const t = performance.now();
    const b = realCreate(desc);
    alloc!.createMs += performance.now() - t;
    alloc!.creates++;
    alloc!.bytesAllocated += desc.size;
    const key = `${desc.size}:${desc.usage}`;
    const prev = shapeCounts!.get(key);
    if (prev) { prev.count++; alloc!.repeatedShapes++; }
    else shapeCounts!.set(key, { bytes: desc.size, usage: desc.usage, count: 1 });
    return b;
  };

  // destroy() is patched on the PROTOTYPE, not per buffer, because buffers are
  // created by more than one path (mappedAtCreation uploads, staging buffers,
  // and anything three.js or a future stage makes). Guarded: if the environment
  // does not expose GPUBuffer globally, allocation still gets counted and only
  // the destroy half goes unmeasured, which is better than refusing to run.
  if (proto && realDestroy) {
    proto.destroy = function patchedDestroy(this: GPUBuffer) {
      const t = performance.now();
      realDestroy.call(this);
      alloc!.destroyMs += performance.now() - t;
      alloc!.destroys++;
    };
  }

  restoreAlloc = () => {
    delete (device as { createBuffer?: unknown }).createBuffer;
    if (proto && realDestroy) proto.destroy = realDestroy;
    if (alloc && shapeCounts) {
      alloc.top = [...shapeCounts.values()].sort((a, b) => b.bytes * b.count - a.bytes * a.count).slice(0, 6);
    }
  };
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
// memory round trip through the driver). To compare this against actual kernel
// time, read it beside gpuTimeline.ts's table -- they measure different clocks
// and are expected to disagree, since a readback blocks until everything queued
// ahead of it has executed.
export async function readFloat32(device: GPUDevice, buffer: GPUBuffer, byteLength: number, label = 'UNLABELLED f32'): Promise<Float32Array> {
  const s = spanStart(`GPU→CPU readback (${byteLength}B)`);
  // Both probes BEFORE the timed read, so the real read is measured against a
  // drained queue and its excess over bareFenceMs is byte cost alone.
  const queueDrainMs = probeEnabled ? await bareRead(device, buffer) : null;
  const bareFenceMs = probeEnabled ? await bareRead(device, buffer) : null;
  const t0 = performance.now();
  const staging = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  ledger.push({ what: label, kind: 'readback', dir: 'down', bytes: byteLength, ms: performance.now() - t0, startMs: t0, bareFenceMs, queueDrainMs });
  spanEnd(s);
  return result;
}

export async function readUint32(device: GPUDevice, buffer: GPUBuffer, byteLength: number, label = 'UNLABELLED u32'): Promise<Uint32Array> {
  const s = spanStart(`GPU→CPU readback (${byteLength}B)`);
  const queueDrainMs = probeEnabled ? await bareRead(device, buffer) : null;
  const bareFenceMs = probeEnabled ? await bareRead(device, buffer) : null;
  const t0 = performance.now();
  const staging = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  ledger.push({ what: label, kind: 'readback', dir: 'down', bytes: byteLength, ms: performance.now() - t0, startMs: t0, bareFenceMs, queueDrainMs });
  spanEnd(s);
  return result;
}

const WORKGROUP_SIZE = 8; // 8x8 = 64 threads/workgroup, a safe default across desktop and mobile GPUs
export function dispatchCount(dim: number): number {
  return Math.ceil(dim / WORKGROUP_SIZE);
}

// ── GPU-kernel timestamp queries ────────────────────────────────────────
//
// The PRIMITIVES only. All measurement built on them lives in gpuTimeline.ts,
// which is the one instrument for device execution time -- see its header.
//
// There used to be a second implementation here (`resolveTimestamps`), which
// each caller drove for its own module: its own query set, its own submit, its
// own mapAsync. That made measuring cost a fence PER MODULE, landing in the
// middle of the work being measured, and the numbers were unusable -- fitPlanes
// reported 9.3ms around a 0.07ms kernel. It is deleted rather than fixed
// because gpuTimeline already does the same job correctly with one fence for
// the whole pipeline, and two implementations of one measurement is how a
// ranking gets quoted from the wrong one.

export function supportsTimestampQuery(device: GPUDevice): boolean {
  return device.features.has('timestamp-query');
}

// pairCount timestamp pairs (begin/end) -- one pair per GPU pass being timed.
export function createTimestampQuerySet(device: GPUDevice, pairCount: number): GPUQuerySet {
  return device.createQuerySet({ type: 'timestamp', count: pairCount * 2 });
}
