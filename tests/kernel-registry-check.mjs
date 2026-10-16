// node tests/kernel-registry-check.mjs
// T375.4: the two tables of src/runtime/jobs.ts by themselves, with stand-ins for the kernels (they write down what
// they are called with and compute nothing): no wasm, no model, no thread. About a second.
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
// And the engine on them (createForward() with kernels that write down their calls, Python's plans of a small model
// in each packing): what it asks is what the model's packing says.
//
// What it does not see: what the kernels compute (smoke, forward-check, kernels-in-browser).
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { runtimeUrl, python } from "./tree.mjs";
import { contract } from "./contracts.mjs";
import { leave } from "./leave.mjs";

const J = await import(runtimeUrl("jobs.ts"));
const { WORK, KIND, PACKED, ADDRESSES, SCRATCH, CONTROL_BYTES, BLOCK_TYPES, BLOCK_TABLES, SPREAD_THREADS, spreadBytes, packingOf, weightBytes, runner, warmUp, addressed } = J;
const { gpuWeightBytes, aloneTakes, cpuReadBytes } = await import(runtimeUrl("forward/alone.js"));

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
const T = 3, B = 16;
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
  // T363.5: llama.cpp's blocks. The job's sixth argument is the type's number (B: IQ2_XXS's, 66 bytes a block of 256),
  // which the kernels take last; kernel: the kernel of a token's rows where it has another name than the kind; cold:
  // not warmed (the tables its kernels read are an engine's). The signed tile takes, last, the calling thread's own
  // bytes of the scratch at the job's last argument (the coordinator's: the scratch itself)
  ["matmul_sbr", "relaxed", [1, 2, 3, 4, 5, 7, 20, 30, B], { tile: ["matmul_sbr_tile", [1, 2, 3, 4, 5, 7, 20, 30, 4, 100, 200, B]], rowBytes: (n) => (n / 256) * 66, a6: B, cold: true }],
  ["matmul_sb", "plain", [1, 2, 3, 4, 5, 7, 20, 30, B], { rowBytes: (n) => (n / 256) * 66, a6: B, cold: true }],
  ["matmul_sbw", "relaxed", [1, 2, 3, 4, 5, 7, 20, 30, B], { tile: ["matmul_sbw_tile", [1, 2, 3, 4, 5, 7, 20, 30, 4, 100, 200, B, 8]], rowBytes: (n) => (n / 256) * 66, a6: B, cold: true, kernel: "matmul_sbr" }],
];
const more = (name) => KINDS.find(([kind]) => kind === name)[3];
const jobOf = (kind, said = {}) => [KIND[kind], 1, 2, 3, 4, 5, more(kind).a6 ?? 6, said.n ?? 7, 8, 9, said.count ?? 1, 100, 200, 300];

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
  for (const [name, from, args, { kernel = name }] of KINDS) {
    const { said, k, r } = both();
    runner(k, r)(jobOf(name), 20, 30);
    assert.deepEqual(said, [[from, kernel, args]], `${name}: a token's rows`);
    // (and from the control area's table: a job is read there as float64)
    const view = both();
    runner(view.k, view.r)(Float64Array.from(jobOf(name)), 20, 30);
    assert.deepEqual(view.said, [[from, kernel, args]], `${name}: a token's rows, the job read off a table of float64`);
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
          expected.push([from, more(name).kernel ?? name, moved]);
        }
      }
      assert.deepEqual(said, expected, `${name}, ${n} wide: four tokens go in blocks of ${step} rows, a token after another`);
      blocks += 1;
    }
  }
  console.log(`ok: count tokens: a tile's one call (${tiles} cases), or blocks of 16 KB of rows with every token in turn (${blocks} cases, the blocks as they were before the table)`);
}
{
  // T363.5: the signed tile spreads a row into bytes of the calling thread's own: each thread's are after the one's
  // before it, as many as a row n wide takes, and a thread past the room there is takes the other tile
  for (const [thread, n] of [[0, 256], [1, 256], [3, 5120], [SPREAD_THREADS - 1, 17408]]) {
    const { said, k, r } = both();
    runner(k, r, thread)(jobOf("matmul_sbw", { n, count: 4 }), 20, 30);
    assert.deepEqual(said, [["relaxed", "matmul_sbw_tile", [1, 2, 3, 4, 5, n, 20, 30, 4, 100, 200, B, 8 + thread * spreadBytes(n)]]], `thread ${thread}, a row ${n} wide`);
  }
  const { said, k, r } = both();
  runner(k, r, SPREAD_THREADS)(jobOf("matmul_sbw", { n: 512, count: 4 }), 20, 30);
  assert.deepEqual(said, [["relaxed", "matmul_sbr_tile", [1, 2, 3, 4, 5, 512, 20, 30, 4, 100, 200, B]]], "a thread past the scratch");
  assert.deepEqual([256, 5120, 17408].map(spreadBytes), [384, 6528, 22080], "the bytes a row is spread into: n + n / 4 + n / 64, to a line of the cache");
  assert.ok(spreadBytes(17408) >= 17408 + 17408 / 4 + 17408 / 64);
  // and the bytes of a block by the type's number, for the rows that go in blocks
  for (const [name, [type, bytes]] of Object.entries(BLOCK_TYPES)) {
    assert.equal(WORK.matmul_sb.rowBytes(768, type), 3 * bytes, `${name}: a row of three blocks`);
    assert.equal(WORK.matmul_sbr.rowBytes(768, type), 3 * bytes);
  }
  console.log(`ok: the signed tile's scratch is the calling thread's own (${SPREAD_THREADS} threads have room), and a block's bytes are its type's`);
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
  const kernelsOf = (from) => KINDS.filter(([, module, , { cold }]) => module === from && !cold).flatMap(([name, , , { tile }]) => [name, ...(tile ? [tile[0]] : [])]).sort();
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
  console.log("ok: the warm-up calls every kernel of every kind 4000 times (the relaxed ones where the thread has them), with its addresses in the scratch; not those of llama.cpp's blocks");
}

// ---- PACKED
{
  // Python's registry: the quantized dtypes by the name a plan's tensor has, the bytes of a row of each, and a matrix
  // packed by each with its weights as int8
  const { packed: said, blocks, codebooks } = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c", `
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from engine.dtypes import BLOCK, BLOCKS, CODEBOOKS, DTYPES, QUANTIZED
from convert.readers import SOURCES
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
    out[dtype.short] = {"group": dtype.group(n), "row": [dtype.stored_bytes(width, dtype.group(width))[0] for width in (128, 256, 1024)], "packed": packed.tolist(), "values": values.tolist(), "scale": dtype.scale}
blocks = {name: {"short": kind.short, "group": kind.group(256), "row": [kind.stored_bytes(width, BLOCK)[0] for width in (256, 1024)], "scale": kind.scale, "type": SOURCES[name].ggml} for name, kind in BLOCKS.items()}
print(json.dumps({"packed": out, "blocks": blocks, "codebooks": CODEBOOKS}))`, python()], { cwd: fileURLToPath(new URL("..", import.meta.url)) }).toString());
  const tabled = Object.keys(PACKED).filter((name) => PACKED[name].tabled !== undefined);
  assert.deepEqual(Object.keys(PACKED).filter((name) => !tabled.includes(name)).sort(), Object.keys(said).sort(), "the packings with scales of their own are the quantized dtypes of Python's registry (engine/dtypes.py), by the names a plan's tensors have");
  // T363.5: and the others llama.cpp's blocks, the kinds a row alone may have (engine/dtypes.py's BLOCKS): the name a
  // plan's tensor has, the type's number in a GGUF (the converter's SOURCES), a row's bytes, 256 weights to a block
  // with no scale after it, and one weight of it is never read alone
  assert.deepEqual(tabled.sort(), Object.keys(blocks).sort(), "the packings the kernels take with the engine's tables are llama.cpp's blocks of Python's registry");
  assert.deepEqual(Object.keys(BLOCK_TYPES).sort(), tabled);
  for (const [name, { short, group, row, scale, type }] of Object.entries(blocks)) {
    const packing = PACKED[name];
    assert.equal(short, name, `${name}: the name a plan's tensor has is the type's`);
    assert.deepEqual([packing.tabled, packing.group, packing.scale, [256, 1024].map((n) => packing.rowBytes(n))], [type, group, scale, row], `${name}: its type's number, a block's weights, no scale after it, a row's bytes`);
    assert.deepEqual(BLOCK_TYPES[name], [type, row[0]], `${name}: BLOCK_TYPES`);
    assert.ok(packing.value === undefined && typeof packing.row === "function", `${name}: a row is read by a kernel, and no weight alone`);
    const called = [];
    packing.row(standIn("plain", called), 700)(1, 2, 256);
    assert.deepEqual(called, [["plain", "sb_row", [1, 2, 700, 256, type]]], `${name}: a row's kernel`);
  }
  assert.ok(BLOCK_TABLES > codebooks, "the kernels' tables hold the checkpoint's codebooks and more");
  const stored = contract("plan.ts").literalsOf("Stored");
  assert.deepEqual([...stored].filter((name) => !Object.hasOwn(PACKED, name)).sort(), ["f16", "f32", "layers", "u8"], "contracts/plan.ts's Stored: the packings, float32 and float16 which are none, the codebooks' bytes, and a stack that is its layers");
  for (const [name, { group, row, packed, values, scale }] of Object.entries(said)) {
    const packing = PACKED[name];
    // (T533: ternary's scales are float16s)
    assert.equal(packing.scale, scale, `${name}: the bytes of a group's scale (engine/dtypes.py's DTYPES is the owner: put jobs.ts's PACKED right)`);
    assert.deepEqual([128, 256, 1024].map((n) => packing.rowBytes(n)), row, `${name}: the bytes of a row`);
    // (T375.5: of a row 256 wide, which every packing has in its whole groups: an int8 row's groups are smaller where it is narrower)
    assert.equal(packing.group, group, `${name}: the weights a scale is of (engine/dtypes.py's DTYPES is the owner: put jobs.ts's PACKED right)`);
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
  // T363.5: llama.cpp's blocks, all alike: signed 8-bit activations and their sums either way, 16 bytes a group of 32
  const summed = [["quantize_x", [1, 2, 3, 96, 0]], ["sums_x", [2, 1, 96]]];
  for (const name of Object.keys(BLOCK_TYPES)) FACTS[name] = { relaxed: "matmul_sbr", plain: "matmul_sb", signed: "matmul_sbw", sums: null, xs: 16, ready: [summed, summed] };
  assert.deepEqual(Object.keys(PACKED), Object.keys(FACTS));
  for (const [name, fact] of Object.entries(FACTS)) {
    const packing = PACKED[name];
    assert.deepEqual([packing.jobs.relaxed, packing.jobs.plain, packing.jobs.signed], [KIND[fact.relaxed], KIND[fact.plain], KIND[fact.signed]], `${name}: the kinds of its jobs`);
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
    assert.equal(WORK[kind].rowBytes(256, packing.tabled), name === "int6" ? 256 : packing.rowBytes(256), `${name}: a row in a block`);
  }
  // T375.5: how the GPU holds each packing, a second copy; and the bytes of a weight, in the checkpoint and on the GPU
  const GPU = { int8: { as: "int8", alone: true, dot: false }, int6: { as: "int8", alone: false, dot: false }, ternary: { as: "ternary", alone: true, dot: true } };
  for (const name of Object.keys(BLOCK_TYPES)) GPU[name] = null;  // (T363.5: none of llama.cpp's blocks on the GPU)
  assert.deepEqual(Object.fromEntries(Object.entries(PACKED).map(([name, { gpu }]) => [name, gpu])), GPU, "how the GPU holds each packing");
  assert.deepEqual(Object.values(PACKED).map(weightBytes), [1.125, 0.875, 0.265625, ...Object.values(BLOCK_TYPES).map(([, bytes]) => bytes / 256)], "the bytes of a weight: a group's values and its scale, over the group");
  // (the names the packings take the activations by: those of one name read what one ready() made)
  assert.deepEqual(Object.fromEntries(Object.entries(PACKED).map(([name, { takes }]) => [name, takes])),
    { int8: "biased", int6: "biased", ternary: "interleaved", ...Object.fromEntries(Object.keys(BLOCK_TYPES).map((name) => [name, "summed"])) }, "how each packing's kernels take the activations");
  assert.deepEqual(["IQ2_XXS", "Q4_K"].map(aloneTakes), [false, false], "no model of llama.cpp's blocks on the GPU alone");
  assert.deepEqual(["int8", "int6", "ternary", "f32", "float32", undefined, "toString"].map(gpuWeightBytes), [1.125, 1.125, 0.28125, 1.125, 1.125, 1.125, 1.125],
    "on the GPU: int6 widened to int8, ternary as it is (T533: its scales widened to float32); what the GPU does not take is counted as int8");
  assert.deepEqual(["int8", "int6", "ternary", "float32", "float16", "toString"].map(aloneTakes), [true, false, true, false, false, false], "the dtypes a model on the GPU alone may be of");
  assert.deepEqual(["int8", "int6", "ternary", "float32"].map((dtype) => cpuReadBytes(36e6, dtype)), [36e6, 36e6 * (1.125 / 0.875), 36e6 * (1.125 / 0.265625), 36e6],
    "what a CPU reads of a token, as int8 of the same weights (what /benchmark/'s CPU section measured the reading of)");
  // (the page's own copies, read on the main thread before the runtime is there: src/models/memory.js)
  const { SIX_OF_EIGHT, TERNARY_OF_EIGHT } = await import("../src/models/memory.js");
  assert.deepEqual([SIX_OF_EIGHT, TERNARY_OF_EIGHT], [weightBytes(PACKED.int6) / weightBytes(PACKED.int8), weightBytes(PACKED.ternary) / weightBytes(PACKED.int8)],
    "src/models/memory.js's sizes of six bits and of ternary against int8 are the packings' (jobs.ts's PACKED is the owner: put memory.js right)");
  assert.deepEqual(["int8", "int6", "ternary", "f32", "f16", "", undefined, "toString", "constructor"].map((name) => packingOf(name)),
    [PACKED.int8, PACKED.int6, PACKED.ternary, null, null, null, null, null, null], "packingOf(): a packing's line by its name, null for anything else");
  console.log("ok: each packing's jobs (relaxed and plain), corrections, activations made ready and bytes in xs; packingOf() knows the packings and nothing else");
}

// ---- the engine reads them: createForward() on Python's plan of a small model in each packing and in float32, its
// kernels the ones that write down their calls. What it asks of them is what the model's packing says, chosen as its
// matrices are made: the corrections' kernel, the kernel of a token's rows and of a block's, and the activations made
// ready for every token of a block (a frame apart) before the matrices of an input are multiplied.
{
  const { plans } = await import("./plans.mjs");
  const { createForward, keysInHalf } = await import(runtimeUrl("forward.js"));
  const { plansOf, planOf, FORM, empty } = plans(new URL("..", import.meta.url));
  const header = [256, 512, 3, 8, 4, 2000, 300], form = { ...FORM, qk_norm: true, head_dim: 64 }, LAYERS = header[2], BLOCK = 5;
  // (a second copy again: [relaxed, plain] each)
  const ASKED = {
    int8: { sums: ["int8_sums", null], one: ["matmul_q8r", "matmul_q8"], block: ["matmul_q8r_tile", "matmul_q8"], bias: [64, 0], interleaved: false },
    int6: { sums: ["six_sums", null], one: ["matmul_q6r", "matmul_q6"], block: ["matmul_q6r", "matmul_q6"], bias: [64, 0], interleaved: false },
    ternary: { sums: [null, null], one: ["matmul_t2r", "matmul_t2"], block: ["matmul_t2r_tile", "matmul_t2"], bias: [0, 0], interleaved: true },
    float32: { sums: [null, null], one: ["matmul_f32", "matmul_f32"], block: ["matmul_f32", "matmul_f32"], bias: [null, null], interleaved: false },
  };
  const named = (said, start) => [...new Set(said.map(([name]) => name).filter((name) => name.startsWith(start) || name.endsWith(start)))];
  let engines = 0;
  for (const p of plansOf(Object.keys(ASKED).map((dtype) => ({ header, form, dtype })))) {
    const asked = ASKED[p.dtype];
    [true, false].forEach((relaxed, i) => {
      const what = `${p.dtype}${relaxed ? ", relaxed" : ""}`, said = [];
      const kernels = () => new Proxy({}, { get: (_, name) => (...args) => { said.push([String(name), args]); return 0; } });
      const options = { ...p.form, dtype: p.dtype, int8: true, relaxed, halfKV: true, outliers: 8, gpu: false, shared: false };
      const memory = new WebAssembly.Memory({ initial: Math.ceil((64 + p.size) / 65536) + 1 });
      const engine = createForward({ memory, base: 64, size: p.size, kernels: { plain: empty, relaxed: relaxed ? empty : null, wide: false },
        plan: planOf(p, { relaxed, kvStart: 16, outliers: 8 }), halfKeys: keysInHalf(p.header, p.size, options), wrap: kernels });
      const made = said.splice(0);
      assert.deepEqual(named(made, "_sums"), asked.sums[i] ? [asked.sums[i]] : [], `${what}: the corrections, as the engine is made`);
      engine.forward(7, 0, true);
      const one = said.splice(0);
      engine.forwardMany(Array.from({ length: BLOCK }, (_, t) => 3 + t), 1);
      const block = said.splice(0);
      engine.release();
      assert.deepEqual(named(one, "matmul_"), [asked.one[i]], `${what}: the kernel of a token's rows`);
      assert.deepEqual(named(block, "matmul_"), [asked.block[i]], `${what}: the kernel of a block's rows`);
      assert.match(engine.backend, new RegExp(p.dtype === "float32" ? "^SIMD kernels, float32" : `^SIMD kernels, ${p.dtype}${relaxed ? ", relaxed SIMD" : ""}( |$)`), `${what}: what the status says`);
      // the corrections a relaxed kernel is handed are the ones the packing's kernel made for that matrix, a layer's
      // after the layer's before
      if (asked.sums[i]) {
        const sums = made.filter(([name]) => name === asked.sums[i]).map(([, [at, , groups]]) => ({ at, end: at + groups * 4, layers: new Set() }));
        for (const [name, args] of one) {
          if (name !== asked.one[i]) continue;
          const of = sums.find(({ at, end }) => args[5] >= at && args[5] < end);
          assert.ok(of, `${what}: a matrix's corrections are among the ones made`);
          assert.equal((args[5] - of.at) % 4, 0);
          of.layers.add(args[5]);
        }
        assert.deepEqual([...new Set(sums.map(({ layers }) => layers.size))].sort(), [1, LAYERS], `${what}: every layer of a matrix has corrections of its own (and the classifier its one)`);
      }
      // the activations made ready: four inputs a layer (and the classifier's for a token's logits), every token of a
      // block, each a frame after the one before, before the input's matrices
      const ready = (said) => said.filter(([name]) => name === "quantize_x");
      if (asked.bias[i] === null) {
        assert.deepEqual([ready(one), ready(block), named(one, "interleave"), named(block, "interleave")], [[], [], [], []], `${what}: float32 weights read the input as it is`);
      } else {
        assert.equal(ready(one).length, 4 * LAYERS + 1, `${what}: a token's inputs made ready`);
        assert.equal(ready(block).length, BLOCK * 4 * LAYERS, `${what}: a block's inputs made ready, every token's`);
        for (const [, args] of [...ready(one), ...ready(block)]) assert.equal(args[4], asked.bias[i], `${what}: the bias of the activations`);
        for (const said of [one, block]) {
          assert.equal(said.filter(([name]) => name === "interleave").length, asked.interleaved ? ready(said).length : 0, `${what}: the activations interleaved`);
          if (asked.interleaved) said.forEach(([name, args], at) => { if (name === "quantize_x") assert.deepEqual(said[at + 1], ["interleave", [args[0], args[1], args[3]]], `${what}: interleaved where they were quantized`); });
        }
        const first = block.findIndex(([name]) => name === "quantize_x"), tokens = block.slice(first).filter(([name]) => name === "quantize_x").slice(0, BLOCK).map(([, args]) => args);
        const frame = tokens[1][0] - tokens[0][0];
        assert.ok(frame > 0);
        tokens.forEach((args, t) => assert.deepEqual([args[0], args[1], args[2], args[3]], [tokens[0][0] + t * frame, tokens[0][1] + t * frame, tokens[0][2] + t * frame, tokens[0][3]], `${what}: token ${t} of a block made ready in its own frame`));
        // (and what the packing keeps of a group of activations in xs fits before the next token's frame begins)
        const [, [, x0]] = block.find(([name]) => name === "rmsnorm");
        for (const [, [xq, xs, , n]] of ready(block)) {
          const t = (xq - tokens[0][0]) / frame;
          assert.ok(Number.isInteger(t) && t >= 0 && t < BLOCK, `${what}: a token's activations are in a frame`);
          assert.ok(xs + Math.ceil(n / 32) * PACKED[p.dtype].xs <= x0 + (t + 1) * frame, `${what}: the activations' scales of ${n} values fit token ${t}'s frame`);
        }
        // (and all of them before the first of the input's matrices is multiplied)
        const multiplied = block.findIndex(([name]) => name.startsWith("matmul_"));
        assert.ok(block.slice(0, multiplied).filter(([name]) => name === "quantize_x").length === BLOCK, `${what}: a block's tokens are ready before its rows are multiplied`);
      }
      engines += 1;
    });
  }
  // What JavaScript reads of a packed tensor one weight at a time, on a checkpoint of made-up bytes (the kernels do
  // nothing, so what the engine wrote is still where it wrote it): a token's embedding row and the outlier channels'
  // columns of the classifier in an engine that multiplies packed weights, and a matrix widened to float32 in one
  // that does not (?without=int8). Each is the weight as its packing reads it, times the scale of its group.
  let read = 0;
  // (the packings a file's dtype may be: llama.cpp's blocks are a row's alone, tests/blocks-forward-check.mjs)
  for (const p of plansOf(Object.keys(PACKED).filter((name) => PACKED[name].tabled === undefined).map((dtype) => ({ header, form, dtype })))) {
    const packing = PACKED[p.dtype];
    for (const packed of [true, false]) {
      const what = `${p.dtype}, ${packed ? "multiplied packed" : "widened"}`, said = [], base = 64;
      const kernels = () => new Proxy({}, { get: (_, name) => (...args) => { said.push([String(name), args]); return 0; } });
      const memory = new WebAssembly.Memory({ initial: Math.ceil((base + p.size) / 65536) + 1 });
      const plan = { ...planOf(p, { relaxed: true, kvStart: 16, outliers: 8 }), int8: packed };
      // the checkpoint: every byte made up, then a scale for every group that says which group it is
      const U = new Uint8Array(memory.buffer), view = new DataView(memory.buffer);
      for (let i = 0; i < p.size; i++) U[base + i] = (i * 2654435761) >>> 24;
      const scaleOf = (g) => Math.fround((g % 61 + 1) / 64);
      for (const t of Object.values(plan.tensors)) {
        if (!Object.hasOwn(PACKED, t.kind)) continue;
        const groups = t.shape.reduce((a, b) => a * b, 1) / t.group;
        // (T533: a float16 where the packing keeps its scales so, ternary; every one of scaleOf()'s is a float16's too)
        for (let g = 0; g < groups; g++) {
          if (PACKED[t.kind].scale === 2) view.setFloat16(base + t.scales + g * 2, scaleOf(g), true);
          else view.setFloat32(base + t.scales + g * 4, scaleOf(g), true);
        }
      }
      const options = { ...p.form, dtype: p.dtype, int8: true, relaxed: true, halfKV: true, outliers: 8, gpu: false, shared: false };
      const engine = createForward({ memory, base, size: p.size, kernels: { plain: empty, relaxed: empty, wide: false }, plan, halfKeys: keysInHalf(p.header, p.size, options), wrap: kernels });
      const token = 7, dim = header[0], vocab = Math.abs(header[5]);
      engine.forward(token, 0, true);
      engine.release();
      const F = new Float32Array(memory.buffer), bytes = new Uint8Array(memory.buffer);
      const weight = (t, i) => Math.fround(packing.value(bytes, base + t.offset, i) * scaleOf((i / t.group) | 0));
      // the embedding's row: what the first norm reads
      const table = plan.tensors.token_embedding_table, [, [, x]] = said.find(([name]) => name === "rmsnorm");
      assert.equal(table.kind, p.dtype);
      const row = Array.from({ length: dim }, (_, i) => F[x / 4 + i]);
      assert.deepEqual(row, Array.from({ length: dim }, (_, i) => weight(table, token * dim + i)), `${what}: the embedding's row of a token`);
      assert.ok(new Set(row).size > 3, `${what}: a row of several values`);
      if (packed) {
        // the outlier channels' columns of the classifier (shared with the embedding here), a column a channel
        const [, [, columns, , count]] = said.find(([name]) => name === "add_columns");
        assert.equal(count, plan.outliers.length);
        plan.outliers.forEach((c, i) => {
          assert.deepEqual(Array.from({ length: vocab }, (_, v) => F[columns / 4 + i * vocab + v]), Array.from({ length: vocab }, (_, v) => weight(table, v * dim + c)), `${what}: the column of outlier channel ${c}`);
        });
        assert.deepEqual(named(said, "matmul_"), [Object.keys(KIND)[packing.jobs.relaxed]]);
      } else {
        // every matrix is float32 now: the first product's is the first layer's wq, widened where the engine put it
        assert.deepEqual(named(said, "matmul_"), ["matmul_f32"], `${what}: float32 products`);
        const wq = plan.tensors.wq, [rows, n] = wq.shape.slice(-2), [, [, , w, width]] = said.find(([name]) => name === "matmul_f32");
        assert.equal(width, n);
        assert.deepEqual(Array.from({ length: rows * n }, (_, i) => F[w / 4 + i]), Array.from({ length: rows * n }, (_, i) => weight(wq, i)), `${what}: a matrix widened to float32`);
      }
      read += 1;
    }
  }
  console.log(`ok: ${read} engines on made-up bytes: an embedding's row, the outlier channels' columns and a matrix widened to float32 are each weight as its packing reads it, times its group's scale`);
  console.log(`ok: ${engines} engines (int8, int6, ternary and float32, with relaxed SIMD and without) ask of their kernels what the packing says: the corrections, a token's kernel and a block's, and every token of a block made ready in its frame`);
}
await leave(0);
