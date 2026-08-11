import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { ARENA_ALIGN, BumpPlanner, alignUp } from '../src/pose/gpu/arena.ts';

// ── The arena's offset arithmetic ─────────────────────────────────────────
//
// `BumpPlanner` is the pure half of src/pose/gpu/arena.ts, split out for
// exactly this reason: every property worth asserting about the partitioning is
// a property of the arithmetic, and none of it needs a GPUBuffer -- so it can be
// checked here rather than only in a browser over the dev bridge.
//
// What is NOT covered here, and has to be checked on a device: that every
// binding actually goes through bind()/sliceRange() (a call site can reach
// `slice.buffer` directly and bypass the generation check -- tsc cannot see
// that), and that 256-byte alignment satisfies the real adapter's
// minStorageBufferOffsetAlignment / minUniformBufferOffsetAlignment.

test('alignUp rounds up to the alignment and leaves exact multiples alone', () => {
  assert.equal(alignUp(1, 256), 256);
  assert.equal(alignUp(255, 256), 256);
  assert.equal(alignUp(256, 256), 256);
  assert.equal(alignUp(257, 256), 512);
  assert.equal(alignUp(0, 256), 0);
});

test('slices never overlap and every offset is aligned', () => {
  const p = new BumpPlanner(1 << 20);
  // Deliberately ragged sizes -- the whole point of the alignment is that a
  // caller asking for 4 bytes cannot leave the next slice misaligned.
  const sizes = [4, 1, 300, 4096, 12, 255, 257, 1_000];
  const placed: { offset: number; end: number }[] = [];
  for (const s of sizes) {
    const r = p.take(s);
    assert.equal(r.spilled, false);
    assert.equal(r.offset % ARENA_ALIGN, 0, `offset ${r.offset} is not ${ARENA_ALIGN}-aligned`);
    assert.ok(r.bytes >= s, 'a slice came back smaller than requested');
    placed.push({ offset: r.offset, end: r.offset + r.bytes });
  }
  for (let i = 0; i < placed.length; i++) {
    for (let j = i + 1; j < placed.length; j++) {
      const a = placed[i], b = placed[j];
      assert.ok(a.end <= b.offset || b.end <= a.offset, `slices ${i} and ${j} overlap`);
    }
  }
});

test('a zero-byte request still gets a real, bindable slice', () => {
  // An empty capture is an ordinary input, and a zero-sized binding is not
  // legal in WebGPU -- so the floor is one aligned block rather than a branch at
  // every call site.
  const p = new BumpPlanner(1 << 16);
  const r = p.take(0);
  assert.equal(r.spilled, false);
  assert.equal(r.bytes, ARENA_ALIGN);
});

test('reset reuses the same offsets; the high-water mark does not reset', () => {
  const p = new BumpPlanner(1 << 20);
  const first = [p.take(1000).offset, p.take(2000).offset, p.take(3000).offset];
  const usedAfterFrame = p.used;
  p.reset();
  assert.equal(p.used, 0, 'reset did not rewind the cursor');
  // ASSERTED IMMEDIATELY AFTER RESET, before anything is re-taken, and that
  // ordering is the whole test. Checking it after a second identical frame
  // measures a quantity INVARIANT under the bug -- the mark simply climbs back
  // to the same value -- so the assertion passes whether or not reset() clears
  // it. Verified by negative control: clearing `high` in reset() fails this
  // line and nothing else in the file.
  assert.equal(p.highWater, usedAfterFrame, 'reset() cleared the high-water mark');
  const second = [p.take(1000).offset, p.take(2000).offset, p.take(3000).offset];
  assert.deepEqual(second, first, 'the same allocation sequence produced different offsets after reset');
});

test('an allocation that does not fit spills instead of overlapping', () => {
  // Capacity for one block only. The second request cannot be served without
  // aliasing the first, so it must come back spilled -- never with an offset.
  const p = new BumpPlanner(ARENA_ALIGN);
  const a = p.take(ARENA_ALIGN);
  assert.equal(a.spilled, false);
  const b = p.take(ARENA_ALIGN);
  assert.equal(b.spilled, true, 'an over-capacity request was served from the arena');
});

test('the high-water mark counts what was ASKED for, not what was served', () => {
  // The property that makes the arena self-tune: a frame that spilled has to
  // report the capacity that WOULD have fitted, or reset() grows to the old
  // capacity and spills again forever.
  const p = new BumpPlanner(ARENA_ALIGN);
  p.take(ARENA_ALIGN); // served, fills it
  p.take(ARENA_ALIGN * 3); // spills
  assert.ok(
    p.highWater >= ARENA_ALIGN * 4,
    `high water ${p.highWater} does not cover the spilled request`,
  );
});

test('a spill does not consume arena space, so later small allocations still fit', () => {
  // A spilled request must not advance the cursor: if it did, one oversized
  // allocation would push every subsequent slice out of the arena too, turning a
  // single bad bound into a fully unallocated frame.
  const p = new BumpPlanner(ARENA_ALIGN * 2);
  const a = p.take(ARENA_ALIGN);
  const big = p.take(ARENA_ALIGN * 10);
  const c = p.take(ARENA_ALIGN);
  assert.equal(a.spilled, false);
  assert.equal(big.spilled, true);
  assert.equal(c.spilled, false, 'a spill consumed arena space it never used');
  assert.equal(c.offset, ARENA_ALIGN, 'the cursor moved for a spilled allocation');
});
