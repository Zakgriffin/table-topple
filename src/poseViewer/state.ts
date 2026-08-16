import { config } from './config.ts';

// ── Global settings ──────────────────────────────────────────────────────
//
// Everything that applies regardless of which camera is active/exists -- per
// the N-camera plan's explicit global/per-camera split.
//
// This IS config.global, not a copy of it: writing globalState.forceCPU writes
// straight into the config object, so persisting is just persistConfig() with
// nothing to mirror first. (Per-camera settings cannot work that way -- there
// are N cameras and one config template, so those are copied back by
// config.ts's syncCameraIntoConfig.) See config.ts for what each field means
// and pose-viewer.config.json for its value.
export const globalState = config.global;
