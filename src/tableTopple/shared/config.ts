import { Value } from '@sinclair/typebox/value';
import { TableToppleConfigSchema } from './configSchema.ts';
import type { TableToppleConfig } from './configSchema.ts';

// ── One config object, one file on disk ──────────────────────────────────
//
// The LOADING half only; the shape is configSchema.ts, which stays pure so the
// command-line check can use it. See that file for why Table Topple has a
// config of its own and what it deliberately does not have.
//
// Fetched, not imported, for the same reason Pose Viewer's is: a static
// `import cfg from '../../../table-topple.config.json'` would put the file in
// vite's module graph, so editing it would trigger an HMR full reload -- which
// on this page drops the camera stream and the WebGPU device along with
// whatever was being looked at. Fetching keeps it outside the graph.
//
// The cost of staying out of the module graph is that a production
// `vite build` would not emit the file (it is not under publicDir). This
// project only ever runs `npm run dev`, where vite serves the project root at
// `/`, so the fetch resolves. If a real build is ever added, the file has to be
// copied into the output -- the failure is loud (this throws at boot), not
// silent.
const CONFIG_URL = '/table-topple.config.json';

// Validated whole, and a failure THROWS. Same reasoning as Pose Viewer's file
// half: it is a human-edited artifact under version control, every literal
// default is gone, and there is nothing safe to substitute. A bad threshold
// would otherwise travel a long way -- into a GPU uniform, out as a pose, into
// a camera placed somewhere wrong -- before surfacing as something that looks
// like a reconstruction bug rather than a config one.
//
// There is no second trust level here because there is no second source: this
// client has no localStorage overlay to be lenient about.
function validate(raw: unknown): TableToppleConfig {
  if (Value.Check(TableToppleConfigSchema, raw)) return raw;
  // Every error, not just the first: a stale file after a rename tends to have
  // one problem per renamed setting, and fixing them one boot at a time is
  // miserable.
  const errors = [...Value.Errors(TableToppleConfigSchema, raw)]
    .map((e) => `  ${e.path || '/'}: ${e.message} (got ${JSON.stringify(e.value)})`);
  throw new Error(`config: ${CONFIG_URL} is invalid\n${errors.join('\n')}`);
}

export async function fetchConfigFile(): Promise<TableToppleConfig> {
  const res = await fetch(CONFIG_URL, { cache: 'no-store' });
  if (!res.ok) throw new Error(`config: GET ${CONFIG_URL} failed (${res.status})`);
  return validate(await res.json());
}

// Top-level await, so every module that imports this one sees a fully loaded
// config at its own first line and the rest of the page can read settings
// synchronously, with no "config not ready yet" state threaded through.
//
// NOTE the ordering hazard that comes with it, the same one Pose Viewer's
// config.ts carries: a module with a top-level await can hand an importer an
// uninitialized binding if a cycle closes back through it. Nothing in
// tableTopple/shared imports the client, which is what keeps that impossible.
export const config: TableToppleConfig = await fetchConfigFile();
