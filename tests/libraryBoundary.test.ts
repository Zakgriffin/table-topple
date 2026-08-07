import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// ── The library boundary, asserted rather than hoped for ──────────────────
//
// src/pose/ is a library and src/sphereLab/ is one of its consumers. Nothing
// in the type system enforces that direction: a stray import from the library
// back into the app typechecks perfectly and keeps working, right up until
// someone tries to run the pipeline somewhere there is no DOM.
//
// This is the test that catches it, and it has to hook the module loader
// rather than read the source, for a reason worth stating: what matters is the
// RUNTIME closure, and that is not what grepping imports tells you.
//
//   - `import type { X } from './app.ts'` is fully erased. It costs nothing at
//     runtime and is fine.
//   - `import { type X } from './app.ts'` is NOT. Under verbatimModuleSyntax
//     it emits a real module load, so it drags the whole transitive closure of
//     that module in at runtime while looking, to a reader, exactly like the
//     erased form one character away.
//
// That distinction is not academic -- it is the entire finding this boundary
// work started from. decodeGrid.ts had one `import { type Camera }`, and it
// was pulling camera/model.ts and camera/settings.ts into the pose pipeline.
// Before the split, node importing poseCompute.ts loaded 41 modules; a
// tsc-visible import graph made it look like 66, and neither number was the
// one that mattered.
//
// What this test is BLIND to, stated so a green run is not over-read:
//   - Type-only edges, deliberately. They constrain a future tsc project
//     split, not the ability to run headless.
//   - Anything reached by a dynamic `import()` inside a function body that
//     this import never calls. There are none on this path today (checked),
//     but a new one would not show up here.
//   - Whether the modules it DOES load are any good. This is a dependency
//     test, not a correctness one.

// Both this app and src/pose/ depend on these, and the board game depends on
// constants.ts as well -- which is precisely why they did not move into the
// library. They are leaves: none of them reaches back into sphereLab/.
//
// This list is deliberately EXHAUSTIVE rather than a prefix rule. A new entry
// appearing here should be a decision someone makes on purpose, because every
// one of them is a module the pose library cannot be extracted without.
const SHARED_LEAVES = new Set([
  'src/debruijn.ts',
  'src/linalg.ts',
  'src/sphereLab/constants.ts',
  'src/sphereLab/floorPattern.ts',
  'src/sphereLab/math/geometry.ts',
  'src/sphereLab/profiling/profiler.ts',
  'src/sphereLab/types.ts',
]);

const loaded: string[] = [];
registerHooks({
  load(url, ctx, next) {
    if (url.includes('/src/')) loaded.push('src/' + url.split('/src/')[1]);
    return next(url, ctx);
  },
});

// The library's entry point, imported for its side effect of loading its own
// closure. Importing the deepest public entry rather than a leaf is the point:
// a leaf would understate what running a real reconstruction actually pulls in.
await import('../src/pose/poseCompute.ts');

test('the pose library loads nothing from the app it is supposed to be independent of', () => {
  const strays = loaded
    .filter((f) => !f.startsWith('src/pose/') && !SHARED_LEAVES.has(f))
    .sort();
  assert.deepEqual(
    strays, [],
    `src/pose/ pulled ${strays.length} module(s) outside the library and its shared leaves at RUNTIME.\n` +
    `The usual cause is \`import { type X }\` where \`import type { X }\` was meant -- the first is a real\n` +
    `module load under verbatimModuleSyntax, the second is erased. Offenders:\n  ${strays.join('\n  ')}`,
  );
});

test('every shared leaf is actually still used, so the allowlist cannot rot', () => {
  const unused = [...SHARED_LEAVES].filter((f) => !loaded.includes(f)).sort();
  assert.deepEqual(
    unused, [],
    `SHARED_LEAVES lists ${unused.length} module(s) the library no longer loads. Deleting an entry that ` +
    `stopped being needed is how this allowlist stays a statement about the boundary rather than a list ` +
    `of things nobody has checked in a while:\n  ${unused.join('\n  ')}`,
  );
});

test('the library entry point really did load its pipeline, not just a type stub', () => {
  // Guards the two tests above from passing vacuously. If poseCompute.ts were
  // ever reduced to re-exporting types, `loaded` would be nearly empty and a
  // clean boundary would mean nothing at all.
  const own = loaded.filter((f) => f.startsWith('src/pose/'));
  assert.ok(
    own.length >= 20,
    `expected the library's own closure to be substantial; got ${own.length} modules, which suggests ` +
    `this test is no longer exercising what it claims to.`,
  );
  for (const stage of ['gradient', 'lsd', 'votes', 'period', 'decode']) {
    assert.ok(
      own.some((f) => f.startsWith(`src/pose/stages/${stage}/`)),
      `no module from stage '${stage}' was loaded -- the pipeline is not fully wired through this entry point`,
    );
  }
});
