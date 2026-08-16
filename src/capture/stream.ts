// ── getUserMedia, and what a camera will actually give you ───────────────
//
// The device-input half of the viewfinder: opening streams, discovering which
// resolutions a camera really supports, and reading its zoom range. What a page
// DOES with any of it -- which <video> it paints, what its dropdown says, how
// it reports status -- belongs to that page.
//
// A neutral platform layer, a peer of `src/gpu/`, because BOTH clients need it
// independently and the isolation rule forbids them sharing it through each
// other. Everything here was established by probing real devices, and none of
// it is recoverable by reading a spec.

/**
 * Requests a genuinely FRESH stream at (w, h), pinned to the SAME physical
 * camera via deviceId -- deliberately NOT `track.applyConstraints()` on an
 * already-running track.
 *
 * Confirmed against a reference tool built for exactly this problem
 * (addpipe/webcam-resolution-tester): `applyConstraints()` has uneven
 * real-world support for renegotiating a running camera's resolution,
 * especially across orientations, on mobile -- it can silently clamp to the
 * nearest resolution along whatever orientation the camera already happens to
 * be running, rather than genuinely renegotiating. A brand-new `getUserMedia()`
 * forces the SAME full negotiation a fresh page load would get.
 *
 * **width/height are `ideal`, NOT `exact`** -- confirmed live: `exact` made
 * literally every candidate fail, not just the swapped-orientation ones. Real
 * camera hardware (Android in particular) exposes a small FIXED list of stream
 * configurations, while `MediaTrackCapabilities`' width/height are independent
 * min/max RANGES that make arbitrary in-between pairs look achievable when they
 * mostly are not. `exact` rejects the instant a requested pair is not one of the
 * camera's literal modes, even when something very close is available; `ideal`
 * never fails that way -- the browser negotiates to its nearest real config and
 * hands back a stream, which the caller then CHECKS against what it actually got.
 * Same "verify, don't trust" principle as the rest of this module, against a
 * constraint type that does not spuriously reject in the first place.
 *
 * `deviceId` stays `exact` deliberately: pinning the same physical camera
 * across a teardown/rebuild is a hard requirement, unlike the resolution.
 *
 * Returns null on any failure (camera busy, permission revoked, ...) -- never
 * throws.
 */
export async function requestStreamAt(
  deviceId: string | undefined, w: number, h: number,
): Promise<MediaStream | null> {
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: w }, height: { ideal: h } },
    });
  } catch {
    return null;
  }
}

/** An unconstrained stream on a specific camera, or on a facing direction. */
export async function requestPlainStream(
  deviceId: string | undefined, facing: string,
): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: facing } },
  });
}

// ── Resolution candidates ────────────────────────────────────────────────
//
// There is no standard API to enumerate a camera's true discrete capture modes
// (`MediaTrackCapabilities.width/height` only report independent {min,max}
// RANGES -- no paired combinations, no aspect-ratio pairing). So instead of
// guessing at a continuous slider, a caller SWEEPS this heuristic list of common
// real resolutions against the actual device, one at a time.
//
// Both (w,h) and (h,w) are included for every entry, because it is genuinely
// ambiguous device to device whether width/height constraints operate in the
// camera's sensor-native (always landscape) space or a device-orientation-
// corrected one. Probing both resolves that empirically instead of guessing.

const RES_CANDIDATES_BASE: readonly { w: number; h: number }[] = [
  { w: 640, h: 480 }, { w: 1024, h: 768 }, { w: 1280, h: 960 }, { w: 1600, h: 1200 },
  { w: 2048, h: 1536 }, { w: 3264, h: 2448 }, { w: 4032, h: 3024 },
  { w: 640, h: 360 }, { w: 1280, h: 720 }, { w: 1920, h: 1080 }, { w: 2560, h: 1440 }, { w: 3840, h: 2160 },
];

/**
 * `untested`: never probed. `supported`: verified live -- a fresh stream
 * requested at exactly (w,h) came back reporting EXACTLY (w,h). `rejected`:
 * either the request threw, or it resolved and came back at some OTHER
 * resolution -- the swapped-orientation case this verification exists to catch.
 */
export type ResCandidateStatus = 'untested' | 'supported' | 'rejected';
export interface ResCandidate { w: number; h: number; status: ResCandidateStatus }

export function buildResCandidateList(): ResCandidate[] {
  const seen = new Set<string>();
  const list: ResCandidate[] = [];
  for (const { w, h } of RES_CANDIDATES_BASE) {
    for (const [cw, ch] of [[w, h], [h, w]] as const) {
      const key = `${cw}x${ch}`;
      if (seen.has(key)) continue;
      seen.add(key);
      list.push({ w: cw, h: ch, status: 'untested' });
    }
  }
  list.sort((a, b) => a.w * a.h - b.w * b.h);
  return list;
}

/**
 * Pacing between probes.
 *
 * Originally added on the theory that back-to-back stream reconfigurations were
 * crashing a real phone mid-sweep. **That turned out NOT to be the fix** --
 * pacing alone did not stop it. The actual cause was an unthrottled
 * full-native-resolution canvas redraw on every rAF tick, which the sweep's
 * larger candidates made worse but did not require. The delay stays because it
 * is reasonable pacing for a UI visibly flashing through candidates; it should
 * not be credited with fixing the crash.
 */
export const RES_PROBE_DELAY_MS = 300;

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * One probe attempt: what was asked for, whether getUserMedia even resolved,
 * what the camera actually settled on, and whether that matched.
 *
 * `matched` is always the RAW, direct requested-vs-actual comparison and is
 * never mutated once a probe finishes, so it stays a simple fact. A caller that
 * has since discovered swapped axes reinterprets it rather than overwriting it.
 */
export interface ProbeResult {
  requestedW: number; requestedH: number;
  resolved: boolean; error: string;
  actualW: number | null; actualH: number | null;
  matched: boolean;
}

/**
 * Opens a fresh, ISOLATED stream on a throwaway <video> at (w, h), reads what
 * actually came back, then tears it down. Never touches the caller's own
 * stream.
 *
 * Mirrors the reference tool's approach: a disposable probe per candidate,
 * rather than repeatedly reconfiguring one live track. Always requests the RAW,
 * un-swap-compensated (w,h) -- during a sweep the swap is not known yet, and
 * this data is what determines it.
 */
export async function probeResolution(
  deviceId: string | undefined, w: number, h: number,
): Promise<ProbeResult> {
  const entry: ProbeResult = {
    requestedW: w, requestedH: h, resolved: false, error: '',
    actualW: null, actualH: null, matched: false,
  };
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: w }, height: { ideal: h } },
    });
  } catch (e: any) {
    entry.error = e?.message || String(e);
    return entry;
  }
  entry.resolved = true;
  const probeVideo = document.createElement('video');
  probeVideo.muted = true;
  probeVideo.playsInline = true;
  probeVideo.srcObject = stream;
  try {
    await probeVideo.play();
    entry.actualW = probeVideo.videoWidth;
    entry.actualH = probeVideo.videoHeight;
    entry.matched = probeVideo.videoWidth === w && probeVideo.videoHeight === h;
  } catch (e: any) {
    entry.error = e?.message || String(e);
  } finally {
    stream.getTracks().forEach((t) => t.stop());
  }
  return entry;
}

/**
 * Whether this platform reports `videoWidth`/`videoHeight` SWAPPED relative to
 * what was requested, decided by majority vote over a completed sweep.
 *
 * Confirmed live on Chrome for iOS -- which, like every browser on iOS by
 * Apple's own App Store rules, must use WebKit as its engine regardless of
 * branding. WebKit negotiates the camera in its own fixed sensor-native space,
 * then rotates the delivered frame to match the phone's physical orientation
 * WITHOUT flipping which axis the original request numbers meant. So on a phone
 * held in portrait, asking for landscape dimensions comes back reporting
 * portrait dimensions, every single time.
 *
 * Detected empirically rather than by user-agent sniffing -- the same "verify,
 * don't assume" principle as everything else here. A probe counts as swapped
 * evidence only when it resolved and came back at the transpose of what it
 * asked for; square requests are excluded, since they cannot distinguish.
 */
export function detectAxesSwapped(results: readonly ProbeResult[]): boolean {
  let direct = 0, swapped = 0;
  for (const e of results) {
    if (!e.resolved || e.actualW === null || e.actualH === null) continue;
    if (e.requestedW === e.requestedH) continue;
    if (e.actualW === e.requestedW && e.actualH === e.requestedH) direct++;
    else if (e.actualW === e.requestedH && e.actualH === e.requestedW) swapped++;
  }
  return swapped > direct;
}

// ── Zoom ─────────────────────────────────────────────────────────────────

/** The track's zoom range, or null when the camera does not expose one. */
export function readZoomRange(stream: MediaStream | null): { min: number; max: number } | null {
  const track = stream?.getVideoTracks()[0];
  let caps: any = null;
  try { caps = track && 'getCapabilities' in track ? (track as any).getCapabilities() : null; }
  catch { caps = null; }
  if (caps && caps.zoom && caps.zoom.min > 0 && caps.zoom.max > caps.zoom.min) {
    return { min: caps.zoom.min, max: caps.zoom.max };
  }
  return null;
}

/** Geometric interpolation across the range, so a slider feels linear. */
export function zoomAt(range: { min: number; max: number }, t: number): number {
  return range.min * Math.pow(range.max / range.min, t);
}

export function applyZoom(stream: MediaStream | null, zoom: number): void {
  const track = stream?.getVideoTracks()[0];
  track?.applyConstraints({ advanced: [{ zoom } as any] }).catch(() => {});
}

// ── Frame rate ───────────────────────────────────────────────────────────

/**
 * The highest frame rate this camera admits to, and whatever it negotiated on
 * its own.
 *
 * `frameRate.max` is part of the base Media Capture spec (unlike zoom or
 * exposure), so it is expected to work broadly -- but nothing guarantees it on
 * every device, hence the fallback chain: reported max, then the rate actually
 * negotiated, then a flat 30.
 *
 * Frame rate is a lever against MOTION BLUR, not just smoothness: requesting a
 * higher rate caps how long auto-exposure may leave the shutter open, since a
 * frame's exposure cannot exceed its own interval. That works even where manual
 * `exposureTime` control does not -- a separate and far less widely supported
 * constraint, notably absent on iOS Safari.
 */
export function readFrameRate(stream: MediaStream | null): { max: number; negotiated: number | null } {
  const track = stream?.getVideoTracks()[0];
  let caps: any = null;
  try { caps = track && 'getCapabilities' in track ? (track as any).getCapabilities() : null; }
  catch { caps = null; }
  const negotiated = track?.getSettings().frameRate ?? null;
  return { max: Math.max(1, Math.round(caps?.frameRate?.max ?? negotiated ?? 30)), negotiated };
}

/**
 * `advanced`, deliberately: a basic constraint that the camera cannot satisfy
 * rejects the whole call, while an advanced one is best-effort. A frame-rate
 * request that quietly does nothing is the right failure here -- the viewfinder
 * keeps running at whatever it had.
 */
export function applyFrameRate(stream: MediaStream | null, fps: number): void {
  const track = stream?.getVideoTracks()[0];
  track?.applyConstraints({ advanced: [{ frameRate: fps } as any] }).catch(() => {});
}
