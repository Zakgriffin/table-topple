// ── Where a stage runs, as an argument rather than an ambient read ────────
//
// Every path that has both a CPU and a GPU implementation used to decide
// between them by reading `globalState.forceCPU` from inside itself. That made
// the choice AMBIENT: a result did not carry the configuration that produced
// it, so it could not be re-derived later. Passing it as an argument is what
// makes a comparison expressible as its natural sentence -- "the same settings,
// on two backends" -- instead of something that has to MUTATE a global between
// two runs and restore it in a `finally`.
//
// ── WHAT THIS STILL SELECTS, now that src/pose is gone ──
//
// It used to choose the backend for every stage of the pose pipeline. It does
// not any more: `src/pose2` is all-GPU with no CPU path and no backend flag, by
// design -- a CPU implementation inside the pipeline is exactly what that
// rewrite deleted.
//
// What is left is the app's OWN work: `pipeline/projectedBins.ts` genuinely has
// two implementations of sample projection (`projectSamplesCPU` and
// `projectSamplesGPU`) and this is what picks between them. That is why the type
// moved here from `src/pose/backend.ts` instead of dying with it -- the concept
// belongs to the app now, not to a pose library that no longer exists.
//
// The `forceCPU` checkbox in sphere-lab.config.json is still the app-side
// control, and `backendFromForceCPU` is still the single conversion point.
export type Backend = 'cpu' | 'gpu';

export function backendFromForceCPU(forceCPU: boolean): Backend {
  return forceCPU ? 'cpu' : 'gpu';
}
