// ── FieldResidency: where each intermediate currently lives ────────────────
//
// Stages 1-4 of the LSD chain (gradient2x2 -> growRegionsCCL ->
// collectRegionsFromLabels -> lsdFit) each dispatch to CPU or GPU on their
// own toggle. Before this file, every GPU module hardcoded its own transfers:
// it uploaded whatever it needed on entry and read its result back on exit,
// unconditionally. That makes each stage self-contained, and it is why the
// chain paid a bus crossing at nearly every stage boundary -- the gradient
// field was uploaded TWICE per frame, by growRegions.ts and lsdFit.ts, each allocating
// and destroying its own copy of the same numbers.
//
// The fix is to stop letting stages decide about transfers at all. A stage
// PUBLISHES its output on the side it produced it, and ASKS for its input on
// the side it wants to consume it. This object is the only thing that knows
// both, so it is the only thing that can decide a transfer is needed -- and it
// decides "needed" as "producer and consumer are on opposite sides of the bus",
// which is the actual criterion. Two adjacent GPU stages then share one buffer
// and no crossing happens between them, without either stage knowing the other
// exists.
//
// The consequence worth stating plainly, because it is the whole point: the
// per-stage toggles stay fully independent and every combination stays correct,
// but their COSTS stop being independent. A GPU stage with CPU neighbours on
// both sides still pays 2 crossings; two adjacent GPU stages pay 2 between them
// TOTAL, not 4. So savings are superadditive along a contiguous run, and a
// stage that loses on its own (the LSD fit at 3.5ms CPU vs 8ms GPU, all of it
// upload) can flip to a win purely because its neighbour got switched on, with
// its own compute unchanged.
//
// The floor for the chain is 2 crossings -- gray up, rectangles down -- because
// the join walk downstream is CPU and always wants real rectangles. Everything
// between those two points is optional traffic, and with stage 1 (gradient2x2)
// now publishing here too, that floor is actually reachable: gray is the only
// thing that has to go up, since it is the only input the chain does not
// produce itself.
//
// ── The single-assignment invariant ──
//
// A field is written ONCE per frame and never mutated in place afterwards.
// Every stage in this pipeline already allocates fresh output arrays rather
// than writing through its inputs, so this costs nothing to adopt -- but it is
// load-bearing here, because it is what makes "a copy exists on side X" mean
// the same thing as "the copy on side X is current". Without it every accessor
// would need a dirty bit and every stage would need to declare its writes.
// `provide*` throws on a second write to the same slot rather than trusting the
// caller to have meant it.

import { GrownRegion } from '../pipeline/lsdSegments.ts';
import {
  createStorageBuffer, getGPUDevice, readFloat32, readUint32, recordTransfer, uploadFloat32, uploadUint32,
} from './device.ts';

// The per-pixel scalar fields of the chain. Each is exactly `n` elements.
//
// `gray` is produced on CPU by capture and is the chain's entry point; `label`
// is produced by stage 3 and consumed by stage 3b; `regionId` is stage 3b's
// per-pixel output and is read ONLY by the raw-region debug overlay. The rest
// are stage 1/2 intermediates. Region data proper is NOT here -- its CPU and
// GPU forms are structurally different rather than the same numbers at two
// precisions, so it gets its own slot below.
export type FieldName = 'gray' | 'fx' | 'fy' | 'label' | 'regionId';

// How a field is stored on each side. 'f32' fields are Float64Array on CPU and
// f32 on GPU -- the widening is lossy in the same 1e-7 class lsdFit and the
// grower were already verified against, not a new source of divergence. 'i32'
// fields are Int32Array on CPU and i32/u32 on GPU, transferred by BIT
// REINTERPRETATION in both directions. That is what makes regionId's -1
// sentinel survive the trip unchanged (it is written as a real -1 by
// collectRegions.wgsl.ts's scatter), and it matches what growRegions.ts already
// did by hand on its label readback.
type FieldKind = 'f32' | 'i32';

const FIELD_KINDS: Record<FieldName, FieldKind> = {
  gray: 'f32', fx: 'f32', fy: 'f32', label: 'i32', regionId: 'i32',
};

interface Slot {
  cpu: Float64Array | Int32Array | null;
  gpu: GPUBuffer | null;
}

// Stage 3b's output in its GPU form: a CSR over region members, exactly the
// buffers collectRegionsGPU already builds on device. `offsets`/`sizes` are
// per-region and `members` is the concatenation; because regions are laid out
// contiguously, offsets[i] + sizes[i] == offsets[i+1], so this carries the same
// information as lsdFit's regionCount+1 offsets array in a different shape.
export interface RegionSetGPU {
  offsets: GPUBuffer;
  sizes: GPUBuffer;
  members: GPUBuffer;
  meanDirs: GPUBuffer; // vec2<f32> per region -- a direction, never an angle
  regionCount: number;
  memberCount: number;
}

export interface TransferEntry {
  what: string;
  direction: 'up' | 'down';
  bytes: number;
}

// A transfer that DID happen. There is deliberately no record of transfers
// avoided: the useful readout is what the current toggle configuration actually
// costs, and "avoided" is only definable against some other configuration.
export interface TransferSummary {
  crossings: number;
  bytes: number;
  entries: readonly TransferEntry[];
}

export class FieldResidency {
  // Null when WebGPU is unavailable OR when no GPU stage in the chain is
  // enabled -- the factory below does not request a device it has no use for,
  // so a fully-CPU frame never touches navigator.gpu. Every GPU stage should
  // check this and fall back exactly as it already checks getGPUDevice().
  readonly device: GPUDevice | null;
  readonly n: number;

  private slots = new Map<FieldName, Slot>();
  private regionsCPUValue: GrownRegion[] | null = null;
  private regionsGPUValue: RegionSetGPU | null = null;
  private owned: GPUBuffer[] = [];
  private transfers: TransferEntry[] = [];

  private constructor(device: GPUDevice | null, n: number) {
    this.device = device;
    this.n = n;
  }

  // `wantGPU` is the OR of the chain's own toggles, resolved by the caller.
  // Passing false keeps this a pure bookkeeping object with no device attached,
  // which is what makes an all-CPU frame cost exactly what it did before.
  static async create(n: number, wantGPU: boolean): Promise<FieldResidency> {
    return new FieldResidency(wantGPU ? await getGPUDevice() : null, n);
  }

  private slot(name: FieldName): Slot {
    let s = this.slots.get(name);
    if (!s) { s = { cpu: null, gpu: null }; this.slots.set(name, s); }
    return s;
  }

  private assertUnwritten(name: string, s: { cpu: unknown; gpu: unknown }): void {
    if (s.cpu || s.gpu) {
      throw new Error(`FieldResidency: '${name}' written twice in one frame -- see the single-assignment invariant`);
    }
  }

  // ── Publishing ──

  provideCPU(name: FieldName, data: Float64Array | Int32Array): void {
    const s = this.slot(name);
    this.assertUnwritten(name, s);
    s.cpu = data;
  }

  // Takes OWNERSHIP of the buffer: it is destroyed by destroy() along with
  // everything else this object allocated. Stages keep owning their own scratch
  // buffers and should destroy those themselves -- only published outputs are
  // handed over.
  provideGPU(name: FieldName, buffer: GPUBuffer): void {
    const s = this.slot(name);
    this.assertUnwritten(name, s);
    s.gpu = buffer;
    this.owned.push(buffer);
  }

  hasCPU(name: FieldName): boolean { return this.slots.get(name)?.cpu != null; }
  hasGPU(name: FieldName): boolean { return this.slots.get(name)?.gpu != null; }

  // ── Consuming ──
  //
  // Note the asymmetry in these two signatures, which is real and not an
  // oversight: an upload is synchronous (mappedAtCreation, no queue round
  // trip) while a readback must await mapAsync. So a CPU stage reaching for GPU
  // data pays a suspension and a GPU stage reaching for CPU data does not.

  gpu(name: FieldName): GPUBuffer {
    const device = this.device;
    if (!device) throw new Error(`FieldResidency: gpu('${name}') with no device -- check .device first`);
    const s = this.slot(name);
    if (s.gpu) return s.gpu;
    if (!s.cpu) throw new Error(`FieldResidency: '${name}' read before it was provided`);

    // Note these upload helpers do NOT set COPY_SRC, so the buffer they return
    // cannot be read back -- and does not need to be. Reaching this line means
    // a CPU copy exists, and since fields are single-assignment that copy stays
    // current forever, so cpu() below will always short-circuit to it and never
    // try to download this buffer.
    // The f64 -> f32 NARROWING is timed separately from the upload itself
    // because it is invisible from inside device.ts and is, on the 2026-08-05
    // reading of the flamegraph, plausibly the larger half: it walks n elements
    // and allocates a fresh n*4 buffer, every field, every frame. `i32` fields
    // pay none of it -- the Uint32Array is a REINTERPRETING VIEW over the same
    // memory, no copy and no loop -- which makes the two branches below a
    // free controlled comparison of exactly this cost.
    const kind = FIELD_KINDS[name];
    const tConv = performance.now();
    const narrowed = kind === 'f32'
      ? new Float32Array(s.cpu as Float64Array)
      : new Uint32Array(s.cpu.buffer, s.cpu.byteOffset, s.cpu.length);
    recordTransfer({
      what: `${name} (f64→f32 narrow)`, kind: 'convert', dir: 'up', bytes: this.n * 4,
      ms: performance.now() - tConv, bareFenceMs: null, queueDrainMs: null,
    });
    const buf = kind === 'f32'
      ? uploadFloat32(device, narrowed as Float32Array, 0, name)
      : uploadUint32(device, narrowed as Uint32Array, 0, name);
    this.transfers.push({ what: name, direction: 'up', bytes: this.n * 4 });
    // Cached, not transient. This is the line that removes the duplicate
    // mag/theta upload: the second consumer on this side finds the buffer
    // already here and pays nothing.
    s.gpu = buf;
    this.owned.push(buf);
    return buf;
  }

  async cpu(name: FieldName): Promise<Float64Array | Int32Array> {
    const s = this.slot(name);
    if (s.cpu) return s.cpu;
    if (!s.gpu) throw new Error(`FieldResidency: '${name}' read before it was provided`);
    const device = this.device;
    if (!device) throw new Error(`FieldResidency: '${name}' is GPU-resident but there is no device`);

    const bytes = this.n * 4;
    const kind = FIELD_KINDS[name];
    // Mirror of the upload path: the readback helper's own cost is recorded
    // inside device.ts, and the WIDENING that follows is recorded here. On an
    // f32 field this is the third full pass over the data for one crossing
    // (device.ts's slice(0) is the second), and it allocates n*8 rather than
    // n*4 -- the single most expensive line in a readback, and entirely
    // deletable by storing Float32Array on the CPU side (perf TODO item 2).
    const raw = kind === 'f32'
      ? await readFloat32(device, s.gpu, bytes, name)
      : await readUint32(device, s.gpu, bytes, name);
    const tConv = performance.now();
    const out = kind === 'f32'
      ? new Float64Array(raw as Float32Array)
      : new Int32Array((raw as Uint32Array).buffer.slice(0));
    recordTransfer({
      what: `${name} (f32→f64 widen)`, kind: 'convert', dir: 'down', bytes,
      ms: performance.now() - tConv, bareFenceMs: null, queueDrainMs: null,
    });
    this.transfers.push({ what: name, direction: 'down', bytes });
    s.cpu = out;
    return out;
  }

  // Typed conveniences, so consumers don't cast at every call site. The
  // narrowing is by field name, which is what the caller actually knows.
  async cpuF64(name: Exclude<FieldName, 'label' | 'regionId'>): Promise<Float64Array> {
    return await this.cpu(name) as Float64Array;
  }
  async cpuI32(name: 'label' | 'regionId'): Promise<Int32Array> {
    return await this.cpu(name) as Int32Array;
  }

  // ── The region set (stage 3b -> stage 4) ──
  //
  // Kept separate from the scalar fields because its two forms are not the same
  // numbers at two precisions: on CPU it is an array of objects each owning its
  // own members array, on GPU it is four buffers plus two counts. Converting is
  // a restructure, not a widening, so it cannot share the generic path above.
  // This is also the single most valuable seam in the chain -- it is the CSR
  // build + readback that made the GPU fit lose in isolation.

  provideRegionsCPU(regions: GrownRegion[]): void {
    this.assertUnwritten('regions', { cpu: this.regionsCPUValue, gpu: this.regionsGPUValue });
    this.regionsCPUValue = regions;
  }

  provideRegionsGPU(rs: RegionSetGPU): void {
    this.assertUnwritten('regions', { cpu: this.regionsCPUValue, gpu: this.regionsGPUValue });
    this.regionsGPUValue = rs;
    this.owned.push(rs.offsets, rs.sizes, rs.members, rs.meanDirs);
  }

  hasRegionsCPU(): boolean { return this.regionsCPUValue != null; }
  hasRegionsGPU(): boolean { return this.regionsGPUValue != null; }

  regionsGPU(): RegionSetGPU {
    const device = this.device;
    if (!device) throw new Error('FieldResidency: regionsGPU() with no device -- check .device first');
    if (this.regionsGPUValue) return this.regionsGPUValue;
    const regions = this.regionsCPUValue;
    if (!regions) throw new Error('FieldResidency: regions read before they were provided');

    // The CSR build lsdFit.ts used to do inline on every call, now done once
    // here and only when the two sides genuinely differ.
    const regionCount = regions.length;
    const offsets = new Uint32Array(regionCount);
    const sizes = new Uint32Array(regionCount);
    let total = 0;
    for (let i = 0; i < regionCount; i++) {
      offsets[i] = total;
      sizes[i] = regions[i].members.length;
      total += sizes[i];
    }
    const members = new Uint32Array(total);
    const meanDirs = new Float32Array(regionCount * 2);
    for (let i = 0; i < regionCount; i++) {
      members.set(regions[i].members, offsets[i]);
      meanDirs[i * 2] = regions[i].meanUx;
      meanDirs[i * 2 + 1] = regions[i].meanUy;
    }

    // Padded to at least one element each, because a zero-byte GPUBuffer is not
    // a legal binding and an empty capture is a perfectly ordinary input (dark
    // frame, everything below rhoHigh). collectRegionsGPU pads its meanAngles
    // buffer for the same reason -- the alternative is making every consumer
    // branch on emptiness before it binds.
    const pad = <T extends Uint32Array | Float32Array>(a: T): T => (a.length > 0 ? a : new (a.constructor as new (n: number) => T)(1));
    const rs: RegionSetGPU = {
      offsets: uploadUint32(device, pad(offsets), 0, 'regions:offsets'),
      sizes: uploadUint32(device, pad(sizes), 0, 'regions:sizes'),
      members: uploadUint32(device, pad(members), 0, 'regions:members'),
      meanDirs: uploadFloat32(device, pad(meanDirs), 0, 'regions:meanDirs'),
      regionCount,
      memberCount: total,
    };
    this.transfers.push({
      what: 'regions', direction: 'up',
      bytes: (regionCount * 4 + total) * 4,
    });
    this.regionsGPUValue = rs;
    this.owned.push(rs.offsets, rs.sizes, rs.members, rs.meanDirs);
    return rs;
  }

  async regionsCPU(): Promise<GrownRegion[]> {
    if (this.regionsCPUValue) return this.regionsCPUValue;
    const rs = this.regionsGPUValue;
    if (!rs) throw new Error('FieldResidency: regions read before they were provided');
    const device = this.device;
    if (!device) throw new Error('FieldResidency: regions are GPU-resident but there is no device');

    const { regionCount, memberCount } = rs;
    const regions: GrownRegion[] = [];
    if (regionCount > 0) {
      const [offRaw, sizeRaw, memRaw, meanRaw] = await Promise.all([
        readUint32(device, rs.offsets, regionCount * 4, 'regions:offsets'),
        readUint32(device, rs.sizes, regionCount * 4, 'regions:sizes'),
        memberCount > 0 ? readUint32(device, rs.members, memberCount * 4, 'regions:members') : Promise.resolve(new Uint32Array(0)),
        readFloat32(device, rs.meanDirs, regionCount * 8, 'regions:meanDirs'),
      ]);
      for (let r = 0; r < regionCount; r++) {
        regions.push({
          // subarray + from, not a view onto memRaw.buffer: a readback's buffer
          // can carry a byteOffset, and a raw Int32Array view would silently
          // read from the wrong place. The copy is required anyway -- memRaw is
          // transient and GrownRegion.members outlives it.
          members: Int32Array.from(memRaw.subarray(offRaw[r], offRaw[r] + sizeRaw[r])),
          meanUx: meanRaw[r * 2], meanUy: meanRaw[r * 2 + 1],
        });
      }
      this.transfers.push({
        what: 'regions', direction: 'down',
        bytes: (regionCount * 4 + memberCount) * 4,
      });
    }
    this.regionsCPUValue = regions;
    return regions;
  }

  // ── Accounting ──
  //
  // Every transfer this object decided to make, which is exactly what the
  // planned UI readout ("chain: 4 crossings, 3.1 MB") wants. It is free to
  // collect because the decision point and the accounting point are the same
  // line of code -- that is the other reason for centralizing transfers here.
  summary(): TransferSummary {
    let bytes = 0;
    for (const t of this.transfers) bytes += t.bytes;
    return { crossings: this.transfers.length, bytes, entries: this.transfers };
  }

  destroy(): void {
    for (const b of this.owned) b.destroy();
    this.owned = [];
    this.slots.clear();
    this.regionsCPUValue = null;
    this.regionsGPUValue = null;
  }
}
