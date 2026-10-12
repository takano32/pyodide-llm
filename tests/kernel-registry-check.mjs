// node tests/kernel-registry-check.mjs
// T375.4: the two tables of src/runtime/jobs.ts by themselves, with stand-ins for the kernels (they write down what
// they are called with and compute nothing): no wasm, no engine, no model, no thread. Under a second.
//
//   WORK    the kinds of jobs. Each kind's one call, its tile and its blocks against a second copy of what the kind is
//           (KINDS below: which module, which kernel, which of a job's arguments in which place; the blocks as
//           blockRows() cut them before the table); a kind of the relaxed module on a thread without one; that every
//           kernel a line calls is one of ADDRESSES, and takes its addresses as BigInt on a 64-bit memory; that warmUp()
//           calls every kernel of every line, on the scratch and nowhere else.
//   PACKED  how a matrix's weights are packed. Each packing's bytes of a row and every weight as the int8 it is against
//           Python's registry of the dtypes (engine/dtypes.py: what it packs, read back here); the kinds of its jobs;
//           its corrections' kernel; what makes the activations ready, with relaxed SIMD and without; and that its
//           names are the quantized kinds Python's plan says (contracts/plan.ts's Stored).
//
// What it does not see: what the kernels compute (smoke, forward-check, kernels-in-browser), and that the engine reads
// the tables as it should (plan-facts-check, memory-check and unchanged.mjs's calls do).
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { runtimeUrl, python } from "./tree.mjs";
import { contract } from "./contracts.mjs";
import { leave } from "./leave.mjs";

const J = await import(runtimeUrl("jobs.ts"));
const { WORK, KIND, PACKED, ADDRESSES, SCRATCH, CONTROL_BYTES, packingOf, runner, warmUp, addressed } = J;

// kernels that write down their calls: [module, name, arguments]
const standIn = (module, said) => new Proxy({}, {
  get: (_, name) => {
    if (typeof name !== "string") return undefined;
    assert.ok(Object.hasOwn(ADDRESSES, name), `${name} is called, and ADDRESSES has no kernel of that name`);
    return (...args) => { said.push([module, name, args]); };
  },
});
const both = () => {
  const said = [];
  return { said, k: standIn("plain", said), r: standIn("relaxed", said) };
};

// ---- WORK
// the second copy: every kind in the order of its number, the module it calls, and of a job
// [kind, 1, 2, 3, 4, 5, 6, 7, 8, rows 9] and rows 20 to 30 what its kernel is handed. tile: of count 4 tokens, strides
// 100 and 200. rowBytes: the bytes of a row n wide as the blocks count it (blockRows() before the table: float32 4 n,
// ternary n / 4, any other n). T: the ternary kernels' mask.
const T = 3;
const KINDS = [
  ["matmul_q8r", "relaxed", [1, 2, 3, 4, 5, 6, 7, 20, 30], { tile: ["matmul_q8r_tile", [1, 2, 3, 4, 5, 6, 7, 20, 30, 4, 100, 200]], rowBytes: (n) => n }],
  ["matmul_q8", "plain", [1, 2, 3, 4, 5, 7, 20, 30], { rowBytes: (n) => n }],
  ["matmul_f32", "plain", [1, 2, 4, 7, 20, 30], { rowBytes: (n) => 4 * n }],
  ["attention", "plain", [1, 2, 3, 4, 5, 6, 9, 7, 8, 20, 30], {}],
  ["attention_f16", "plain", [1, 2, 3, 4, 5, 6, 9, 7, 8, 20, 30], {}],
  ["matmul_q6r", "relaxed", [1, 2, 3, 4, 5, 6, 7, 20, 30], { rowBytes: (n) => n }],
  ["matmul_q6", "plain", [1, 2, 3, 4, 5, 7, 20, 30], { rowBytes: (n) => n }],
  ["delta_rule", "plain", [1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 30], {}],
  ["matmul_t2r", "relaxed", [1, 2, 3, 4, 5, 7, 20, 30, T], { tile: ["matmul_t2r_tile", [1, 2, 3, 4, 5, 7, 20, 30, 4, 100, 200, T]], rowBytes: (n) => n / 4 }],
  ["matmul_t2", "plain", [1, 2, 3, 4, 5, 7, 20, 30, T], { rowBytes: (n) => n / 4 }],
];
const jobOf = (kind, more = {}) => [KIND[kind], 1, 2, 3, 4, 5, 6, more.n ?? 7, 8, 9, more.count ?? 1, 100, 200, 300];

assert.deepEqual(Object.keys(WORK), KINDS.map(([name]) => name), "the kinds of jobs, in the order of their numbers");
assert.deepEqual(Object.entries(KIND), Object.keys(WORK).map((name, number) => [name, number]), "a kind's number is its place in WORK");
assert.equal(new Set(Object.values(KIND)).size, KINDS.length);
for (const [name, from, , { tile, rowBytes }] of KINDS) {
  assert.equal(WORK[name].from, from, `${name}: whose kernels`);
  assert.equal(Boolean(WORK[name].tile), Boolean(tile), `${name}: whether it has a tile`);
  assert.equal(Boolean(WORK[name].rowBytes), Boolean(rowBytes), `${name}: whether its rows are a matrix's`);
}
{
  // one token: one call, of the kind's own kernel in its own module
  for (const [name, from, args] of KINDS) {
    const { said, k, r } = both();
    runner(k, r)(jobOf(name), 20, 30);
    assert.deepEqual(said, [[from, name, args]], `${name}: a token's rows`);
    // (and from the control area's table: a job is read there as float64)
    const view = both();
    runner(view.k, view.r)(Float64Array.from(jobOf(name)), 20, 30);
    assert.deepEqual(view.said, [[from, name, args]], `${name}: a token's rows, the job read off a table of float64`);
  }
  console.log(`ok: the ${KINDS.length} kinds of jobs: one token's rows are one call, of the kind's kernel in the kind's module, with the job's arguments in their places`);
}
{
  // count tokens: a tile where the kind has one; else blocks of the rows, every token in turn, each call one token's
  let tiles = 0, blocks = 0;
  for (const [name, from, args, { tile, rowBytes }] of KINDS) {
    if (!rowBytes) continue;
    for (const n of [64, 768, 4096, 20000, 80000]) {
      const { said, k, r } = both();
      runner(k, r)(jobOf(name, { n, count: 4 }), 20, 30 + 100);
      if (tile) {
        const [kernel, tiled] = tile;
        assert.deepEqual(said, [[from, kernel, tiled.map((a) => (a === 7 ? n : a === 30 ? 130 : a))]], `${name}: four tokens are one tile`);
        tiles += 1;
        continue;
      }
      const step = Math.max(1, Math.floor(16384 / rowBytes(n))), expected = [];
      for (let row = 20; row < 130; row += step) {
        for (let t = 0; t < 4; t++) {
          // out, a and b are a stride on for every token (b is no argument of matmul_f32's)
          const moved = args.map((a, i) => (a === 7 ? n : a === 20 ? row : a === 30 ? Math.min(row + step, 130) : i === 0 ? 1 + t * 100 : i === 1 ? 2 + t * 200 : i === 2 && a === 3 ? 3 + t * 300 : a));
          expected.push([from, name, moved]);
        }
      }
      assert.deepEqual(said, expected, `${name}, ${n} wide: four tokens go in blocks of ${step} rows, a token after another`);
      blocks += 1;
    }
  }
  console.log(`ok: count tokens: a tile's one call (${tiles} cases), or blocks of 16 KB of rows with every token in turn (${blocks} cases, the blocks as they were before the table)`);
}
{
  // a thread without the relaxed module: the plain kinds run, a relaxed one says which kind it is
  const said = [], k = standIn("plain", said), runRows = runner(k, null);
  for (const [name, from, args] of KINDS) {
    if (from === "plain") {
      said.length = 0;
      runRows(jobOf(name), 20, 30);
      assert.deepEqual(said, [["plain", name, args]]);
    } else {
      for (const count of [1, 4]) {
        assert.throws(() => runRows(jobOf(name, { count }), 20, 30), new RegExp(`^Error: A job of the kind ${name} takes the relaxed kernels, and this thread has none\\.$`), `${name}, ${count}`);
      }
    }
  }
  assert.throws(() => runRows([KINDS.length, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 0, 0], 0, 1), TypeError, "a number that is no kind's");
  console.log("ok: without the relaxed module the plain kinds run, and a kind of the relaxed module is refused by its name");
}
{
  // a 64-bit memory: the addresses of every kernel a kind calls are BigInt, the rest Numbers
  const said = [], record = (module) => Object.fromEntries(Object.keys(ADDRESSES).map((name) => [name, (...args) => { said.push([module, name, args]); }]));
  const runRows = runner(addressed(record("plain"), true), addressed(record("relaxed"), true));
  for (const [name, , , { tile }] of KINDS) {
    for (const count of tile ? [1, 4] : [1]) {
      said.length = 0;
      runRows(jobOf(name, { count }), 20, 30);
      const [[, kernel, args]] = said;
      args.forEach((a, i) => assert.equal(typeof a, ADDRESSES[kernel].includes(i) ? "bigint" : "number", `${kernel}: argument ${i}`));
    }
  }
  console.log("ok: on a 64-bit memory every kind's kernel takes its addresses, and nothing else, as BigInt");
}
{
  // the warm-up: every kernel of every line, 4000 times each, on the scratch's addresses and nowhere else
  const kernelsOf = (from) => KINDS.filter(([, module]) => module === from).flatMap(([name, , , { tile }]) => [name, ...(tile ? [tile[0]] : [])]).sort();
  for (const relaxed of [true, false]) {
    const { said, k, r } = both();
    warmUp(k, relaxed ? r : null);
    const counted = {};
    for (const [module, name, args] of said) {
      counted[`${module} ${name}`] = (counted[`${module} ${name}`] ?? 0) + 1;
      for (const i of ADDRESSES[name]) assert.ok(args[i] >= SCRATCH && args[i] + 64 <= CONTROL_BYTES, `${name}: argument ${i} is in the scratch`);
    }
    const expected = Object.fromEntries([...kernelsOf("plain").map((name) => `plain ${name}`), ...(relaxed ? kernelsOf("relaxed").map((name) => `relaxed ${name}`) : [])].map((name) => [name, 4000]));
    assert.deepEqual(Object.fromEntries(Object.entries(counted).sort()), Object.fromEntries(Object.entries(expected).sort()), `the warm-up${relaxed ? "" : " without the relaxed module"}`);
  }
  console.log("ok: the warm-up calls every kernel of every kind 4000 times (the relaxed ones where the thread has them), with its addresses in the scratch");
}

// ---- PACKED
{
  // Python's registry: the quantized dtypes by the name a plan's tensor has, the bytes of a row of each, and a matrix
  // packed by each with its weights as int8
  const said = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c", `
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from engine.dtypes import DTYPES, QUANTIZED
rng = np.random.default_rng(4)
out = {}
for name in QUANTIZED:
    dtype = DTYPES[name]
    n = 256
    rows = rng.standard_normal((3, n)).astype(np.float32)
    if not dtype.rounds:
        rows = (rng.integers(-1, 2, (3, n)) * 0.5).astype(np.float32)
        rows[:, ::dtype.group(n)] = 0.5
    packed, scales = dtype.pack(rows)
    packed = np.ascontiguousarray(packed).reshape(-1)
    values = dtype.unpack(packed)
    assert len(values) == 3 * n and len(set(values.tolist())) > 2
    out[dtype.short] = {"group": dtype.group(n), "row": [dtype.stored_bytes(width, dtype.group(width))[0] for width in (128, 256, 1024)], "packed": packed.tolist(), "values": values.tolist()}
print(json.dumps(out))`, python()], { cwd: fileURLToPath(new URL("..", import.meta.url)) }).toString());
  assert.deepEqual(Object.keys(PACKED).sort(), Object.keys(said).sort(), "the packings are the quantized dtypes of Python's registry (engine/dtypes.py), by the names a plan's tensors have");
  const stored = contract("plan.ts").literalsOf("Stored");
  assert.deepEqual([...stored].filter((name) => !Object.hasOwn(PACKED, name)).sort(), ["f16", "f32"], "contracts/plan.ts's Stored: the packings, and float32 and float16 which are none");
  for (const [name, { row, packed, values }] of Object.entries(said)) {
    const packing = PACKED[name];
    assert.deepEqual([128, 256, 1024].map((n) => packing.rowBytes(n)), row, `${name}: the bytes of a row`);
    // (the values from some byte on, as a tensor is somewhere in a checkpoint)
    for (const at of [0, 40]) {
      const U = new Uint8Array(at + packed.length);
      U.set(packed, at);
      assert.deepEqual(values.map((_, i) => packing.value(U, at, i)), values, `${name}: every weight as the int8 it is, the values at byte ${at}`);
    }
  }
  console.log(`ok: the ${Object.keys(PACKED).length} packings (${Object.keys(PACKED).join(", ")}) are Python's quantized dtypes: the bytes of a row and every weight of a packed matrix as Python reads them`);
}
{
  // the facts of each packing, a second copy: the kinds of its jobs, its corrections' kernel, the bytes of a group of
  // activations in xs, and what makes the activations ready (relaxed, plain)
  const FACTS = {
    int8: { relaxed: "matmul_q8r", plain: "matmul_q8", sums: "int8_sums", xs: 4, ready: [[["quantize_x", [1, 2, 3, 96, 64]]], [["quantize_x", [1, 2, 3, 96, 0]]]] },
    int6: { relaxed: "matmul_q6r", plain: "matmul_q6", sums: "six_sums", xs: 4, ready: [[["quantize_x", [1, 2, 3, 96, 64]]], [["quantize_x", [1, 2, 3, 96, 0]]]] },
    ternary: { relaxed: "matmul_t2r", plain: "matmul_t2", sums: null, xs: 8,
      ready: [[["quantize_x", [1, 2, 3, 96, 0]], ["interleave", [1, 2, 96]]], [["quantize_x", [1, 2, 3, 96, 0]], ["interleave", [1, 2, 96]]]] },
  };
  assert.deepEqual(Object.keys(PACKED), Object.keys(FACTS));
  for (const [name, fact] of Object.entries(FACTS)) {
    const packing = PACKED[name];
    assert.deepEqual([packing.jobs.relaxed, packing.jobs.plain], [KIND[fact.relaxed], KIND[fact.plain]], `${name}: the kinds of its jobs`);
    assert.deepEqual([WORK[fact.relaxed].from, WORK[fact.plain].from], ["relaxed", "plain"], `${name}: its relaxed job is the relaxed module's, the other the plain one's`);
    assert.ok(WORK[fact.relaxed].rowBytes && WORK[fact.plain].rowBytes, `${name}: its jobs are a matrix's`);
    assert.equal(packing.sums, fact.sums, `${name}: the kernel of its corrections`);
    assert.ok(packing.sums === null || Object.hasOwn(ADDRESSES, packing.sums));
    assert.equal(packing.xs, fact.xs, `${name}: the bytes of a group of activations in xs`);
    [true, false].forEach((relaxed, i) => {
      const said = [], ready = packing.ready(standIn("plain", said), relaxed);
      assert.deepEqual(said, [], `${name}: choosing calls nothing`);
      ready(1, 2, 3, 96);
      assert.deepEqual(said.map(([, kernel, args]) => [kernel, args]), fact.ready[i], `${name}: the activations made ready${relaxed ? ", relaxed" : ""}`);
    });
  }
  // (the blocks of a packing's job count a row's bytes as the packing does, but for int6's, counted as int8's as they always were)
  for (const [name, packing] of Object.entries(PACKED)) {
    const kind = Object.keys(KIND)[packing.jobs.plain];
    assert.equal(WORK[kind].rowBytes(256), name === "int6" ? 256 : packing.rowBytes(256), `${name}: a row in a block`);
  }
  assert.deepEqual(["int8", "int6", "ternary", "f32", "f16", "", undefined, "toString", "constructor"].map((name) => packingOf(name)),
    [PACKED.int8, PACKED.int6, PACKED.ternary, null, null, null, null, null, null], "packingOf(): a packing's line by its name, null for anything else");
  console.log("ok: each packing's jobs (relaxed and plain), corrections, activations made ready and bytes in xs; packingOf() knows the packings and nothing else");
}
await leave(0);
