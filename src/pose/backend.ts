// ── Where a stage runs, as an argument rather than an ambient read ────────
//
// Every stage that has both a CPU and a GPU implementation used to decide
// between them by reading `globalState.forceCPU` from inside itself. That made
// the choice AMBIENT: a result -- a pose, a timing, a verify delta -- did not
// carry the configuration that produced it, so it could not be re-derived
// later. That is the same root cause behind the historical timing numbers being
// void (the config was never pinned alongside them) and behind stages not being
// unit-testable (a test could not state what it was testing without mutating a
// global first).
//
// ── Why this is NOT a field on LsdSettings/LsdCompositeSettings ──
//
// Those objects are ALGORITHM PARAMETERS -- what to compute. This is EXECUTION
// POLICY -- where to compute it. Keeping them apart is what makes the
// differential test expressible as its natural sentence: "the same settings, on
// two backends". Fold the backend into the settings object and that harness has
// to MUTATE settings between the two runs, which is exactly what
// harness/lsdChainVerify.ts USED to do to the global (it saved
// globalState.forceCPU, wrote it twice, and restored it in a finally, so a
// throw between the two runs left the running app reconfigured). It now takes
// the backend as an argument, which is what this type is for. One object that
// both the caller and the harness want to hold still, and one knob the harness
// wants to sweep, are different things and should not share a type.
//
// ── 'gpu' means PREFER, not REQUIRE ──
//
// Every GPU stage already falls back to its CPU implementation when the device
// is missing or the dispatch bails, and that behaviour is unchanged: 'gpu' asks
// for the device path and accepts CPU when it cannot be had. So this type says
// what was REQUESTED, never what actually ran -- which is why the profiler spans
// are still named from the branch actually taken (`gradient2x2 (GPU)` vs
// `(CPU)`) rather than from this value. A run that requested 'gpu' on a machine
// with no WebGPU is a legitimate all-CPU run, and the spans are what say so.
export type Backend = 'cpu' | 'gpu';

// The single conversion point from the app's boolean to this type. The app-side
// control is still a `forceCPU` checkbox in sphere-lab.config.json, and the
// library should not know that -- callers on the app side go through here, and
// nothing below the boundary ever sees the boolean again.
export function backendFromForceCPU(forceCPU: boolean): Backend {
  return forceCPU ? 'cpu' : 'gpu';
}
