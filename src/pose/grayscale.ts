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

/**
 * The same conversion, straight to the type `runPose` actually takes, into a
 * buffer the caller owns.
 *
 * `toGrayscale` above returns Float64Array, which every caller then has to
 * narrow -- `Float32Array.from(gray)` -- because `runPose` takes f32
 * deliberately (§4: that narrowing loop was measured as part of the
 * byte-proportional cost per reconstruction, which is why the entry point does
 * not take f64 any more). So the f64 version's callers pay one extra full-image
 * pass and one extra full-image allocation to produce a value the pipeline
 * cannot use as-is.
 *
 * That is survivable for a page that reconstructs on a shutter tap. It is not
 * for one doing it continuously: at 480x640 the pair costs 307k needless writes
 * and a 1.2 MB allocation EVERY frame, on a phone, competing with the very
 * pipeline being timed.
 *
 * `out` is optional and reused when given -- a client with a fixed capture
 * resolution allocates once for the life of the page. Passing a wrong-sized
 * buffer throws rather than silently converting part of an image, which is the
 * failure that would otherwise read as a decode problem.
 *
 * Deliberately an ADDITION here rather than a copy of the loop in an app: this
 * module is already the one home for "RGBA in, what the pipeline eats out", and
 * two apps needing it is the argument for extending it, not for each writing
 * its own.
 */
export function toGrayscaleF32(
  rgba: Uint8ClampedArray | Uint8Array, w: number, h: number, out?: Float32Array,
): Float32Array {
  const n = w * h;
  if (out && out.length !== n) {
    throw new Error(`toGrayscaleF32: out is ${out.length} samples, expected ${n} for ${w}x${h}`);
  }
  const gray = out ?? new Float32Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    gray[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }
  return gray;
}

