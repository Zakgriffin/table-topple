// ── An instrumented GPUDevice, for counting what a frame actually did ─────
//
// The pipeline's headline property is "one submit, one fence, one map", and the
// only honest way to check it is from OUTSIDE: asking `runPose2` how many
// passes it encoded and believing the answer tests nothing, because the number
// it would report is the same variable the feature under test maintains.
//
// So this counts the WebGPU calls themselves. Every count here comes from the
// API surface, not from any bookkeeping inside src/pose2.
//
// ── IT MUST NOT BE A PROXY, AND THAT COST A CRASH TO LEARN ──
//
// The obvious implementation wraps the device and its buffers in Proxies. It
// kills the process outright -- not an exception, a native abort.
//
// The reason is that these objects are handed BACK to native methods:
// `copyBufferToBuffer(src, 0, staging, ...)` and `mapAsync` reach into the Dawn
// addon's own wrapper for the buffer, and a Proxy is not that object no matter
// how faithfully its traps forward. A JS-side wrapper is invisible to a JS-side
// caller and fatal to a native one.
//
// So nothing is wrapped. The methods are SHADOWED on the real instances with
// own properties, which leaves object identity untouched -- native calls receive
// exactly the objects they created. The cost is that this mutates a device other
// tests share, which is what `restore()` is for.

export interface DeviceCounts {
  /** beginComputePass calls -- the encoded pass count. */
  passes: number;
  /** queue.submit calls. Each submitted command buffer counts once. */
  submits: number;
  /** mapAsync calls on any MAP_READ buffer -- the fences the host waits on. */
  maps: number;
  /** Labels of the compute passes, in encode order. */
  passLabels: string[];
}

export interface Counting {
  device: GPUDevice;
  counts: DeviceCounts;
  /**
   * Puts every shadowed method back. MUST be called -- the device is shared for
   * the whole test process, so a counter left installed would keep counting
   * into a stale object for every later test.
   */
  restore(): void;
}

/** Shadows `obj[name]` with `impl`, and returns the undo. */
function shadow<T extends object>(obj: T, name: string, impl: unknown): () => void {
  const had = Object.prototype.hasOwnProperty.call(obj, name);
  const prev = (obj as Record<string, unknown>)[name];
  (obj as Record<string, unknown>)[name] = impl;
  return () => {
    if (had) (obj as Record<string, unknown>)[name] = prev;
    else delete (obj as Record<string, unknown>)[name];
  };
}

export function countingDevice(device: GPUDevice): Counting {
  const counts: DeviceCounts = { passes: 0, submits: 0, maps: 0, passLabels: [] };
  const undo: (() => void)[] = [];

  // Bound to the real receiver before shadowing, so the replacement can call
  // through to the original without recursing into itself.
  const realCreateEncoder = device.createCommandEncoder.bind(device);
  const realCreateBuffer = device.createBuffer.bind(device);
  const realSubmit = device.queue.submit.bind(device.queue);

  undo.push(shadow(device, 'createCommandEncoder', (d?: GPUCommandEncoderDescriptor) => {
    const enc = realCreateEncoder(d);
    const realBegin = enc.beginComputePass.bind(enc);
    // Shadowed on the encoder INSTANCE, which is created fresh per frame and
    // discarded, so this one needs no undo.
    shadow(enc, 'beginComputePass', (desc?: GPUComputePassDescriptor) => {
      counts.passes++;
      counts.passLabels.push(desc?.label ?? '');
      return realBegin(desc);
    });
    return enc;
  }));

  undo.push(shadow(device, 'createBuffer', (d: GPUBufferDescriptor) => {
    const buf = realCreateBuffer(d);
    if ((d.usage & GPUBufferUsage.MAP_READ) !== 0) {
      const realMap = buf.mapAsync.bind(buf);
      shadow(buf, 'mapAsync', (mode: number, offset?: number, size?: number) => {
        counts.maps++;
        return realMap(mode, offset, size);
      });
    }
    return buf;
  }));

  undo.push(shadow(device.queue, 'submit', (bufs: GPUCommandBuffer[]) => {
    counts.submits += bufs.length;
    return realSubmit(bufs);
  }));

  return {
    device,
    counts,
    restore() { for (const u of undo.reverse()) u(); undo.length = 0; },
  };
}
