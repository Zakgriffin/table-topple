// ── The one clock ────────────────────────────────────────────────────────
//
// Epoch milliseconds, sub-millisecond, and MONOTONIC within a process. Every
// timestamp in this project should come from here.
//
// It exists because there used to be two clocks with opposing properties, and
// which one a given field used was an accident of when it was written:
//
//   Date.now()        epoch ms, comparable ACROSS devices, but whole-millisecond
//                     and it follows OS wall-clock corrections -- so it can jump
//                     backwards mid-session.
//   performance.now()  sub-microsecond and monotonic, but measured from a
//                     per-process origin, so a phone's value and a desktop's are
//                     not comparable at all.
//
// The phone->desktop latency fields (sentAt/pulledAt/encodedAt/receivedAt) took
// Date.now() because they cross devices and needed a shared epoch. Everything
// timing a local operation took performance.now(). Both were right about their
// own case and neither was right in general.
//
// `performance.timeOrigin + performance.now()` is both: timeOrigin is the epoch
// ms at which this process's clock started, so adding it back yields an epoch
// timestamp that is still monotonic and still sub-millisecond. It is directly
// comparable to a Date.now() value and to another device's nowMs().
//
// THE ONE TRADEOFF, so it is a decision rather than an oversight: if an OS wall
// clock is corrected mid-session (NTP), Date.now() follows it and this does not,
// so two devices that were in sync would drift apart permanently. Accepted,
// because the failure it prevents is worse and is not hypothetical -- IMU fusion
// integrates acceleration across frame timestamps, and a backwards clock step
// injects a fictitious velocity that never decays. A slow constant offset
// between two devices is a calibration problem; a step is a corrupted integral.
//
// Cross-device clock SKEW is unaffected either way: two devices' epochs were
// already only as aligned as their NTP sync made them, and nothing here changes
// that. See camera/model.ts's pendingCapture for what the cross-device
// timestamps are actually used to derive.
export function nowMs(): number {
  return performance.timeOrigin + performance.now();
}
