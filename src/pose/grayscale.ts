// ── Grayscale, the pipeline's one input format ────────────────────────────
//
// MOVED HERE from src/decode.ts, whose only other exports (`Patch`,
// `SampledCell`) had zero importers and went with it. Every caller of the pose
// library has an RGBA buffer and needs this before it can call `runPose`, so
// the conversion belongs next to the thing that consumes it rather than in an
// app module each app would have to reach into.

export function toGrayscale(rgba: Uint8ClampedArray | Uint8Array, w: number, h: number): Float64Array {
  const gray = new Float64Array(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return gray;
}

