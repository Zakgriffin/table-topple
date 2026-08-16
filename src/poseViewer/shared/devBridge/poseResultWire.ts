// Shared by client/relay.ts (pack) and server/devBridge/client.ts (unpack) so the
// two ends of this binary framing can't drift apart -- see this session's
// "no more base64-in-JSON for poseResult's optional image" plan.
//
// Wire shape, as it arrives at devBridge/client.ts AFTER server.js's own
// CAPTURE_ID_BYTES prefix has already been sliced off:
//   byte 0:            POSE_RESULT_IMAGE_TAG
//   bytes [1..5):      uint32 LE = header byte length (N)
//   bytes [5..5+N):    UTF-8 JSON header (today's poseResult fields, minus dataUrl)
//   bytes [5+N..end):  raw JPEG bytes (canvas.toBlob output, never base64)

// Chosen specifically to never equal 0xFF -- a legacy realCapture binary
// frame is just raw JPEG bytes with no tag of its own, and JPEG's first byte
// is always the 0xFF SOI marker, so this value can never collide with one.
// That's what lets devBridge/client.ts tell the two binary-frame shapes
// apart with zero server.js involvement (server.js's relay is already
// content-agnostic either way).
const POSE_RESULT_IMAGE_TAG = 1;

const HEADER_LEN_BYTES = 4;
const PREFIX_BYTES = 1 + HEADER_LEN_BYTES;

export function packPoseResultWithImage(header: unknown, jpeg: Blob): Blob {
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const prefix = new Uint8Array(PREFIX_BYTES);
  prefix[0] = POSE_RESULT_IMAGE_TAG;
  new DataView(prefix.buffer).setUint32(1, headerBytes.byteLength, true);
  // One Blob, three parts -- the browser's WebSocket implementation reads
  // Blob parts into wire bytes internally (same as captureAndSendFrame's own
  // ws.send(blob)), so no manual concatenation into a single Uint8Array is
  // needed here.
  return new Blob([prefix, headerBytes, jpeg]);
}

export function tryUnpackPoseResultWithImage(payload: Uint8Array<ArrayBuffer>): { header: unknown; jpeg: Uint8Array<ArrayBuffer> } | null {
  if (payload.length < PREFIX_BYTES || payload[0] !== POSE_RESULT_IMAGE_TAG) return null;
  // payload is itself a subarray view (devBridge/client.ts slices the
  // captureId prefix off a larger ArrayBuffer first) -- DataView must be
  // built with its own byteOffset/byteLength, not payload.buffer alone,
  // or the length read below would be off by however many bytes were
  // sliced off the front.
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const headerLen = view.getUint32(1, true);
  const headerStart = PREFIX_BYTES;
  const headerEnd = headerStart + headerLen;
  const headerJson = new TextDecoder().decode(payload.subarray(headerStart, headerEnd));
  const jpeg = payload.subarray(headerEnd);
  return { header: JSON.parse(headerJson), jpeg };
}
