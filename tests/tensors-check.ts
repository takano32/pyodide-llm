// tests/tensors-check.ts (T641.2): the readers of a model's tensors (src/runtime/forward/tensors.ts) by themselves,
// with stand-ins for their neighbours: no engine, no model, no kernels, no Pyodide. A few pages of memory whose
// checkpoint the test writes by hand, kernels that write down what they are asked, and made-up plans. Node only, well
// under a second, in the light suite.
//
//   node tests/tensors-check.ts
//
// What it sees: floats() of float32 (in place, nothing taken), float16, int8, int6, ternary and binary tensors (widened
// once to the numbers the packing's weights times the scale of their group are), a stack whose layers are each of a
// kind of their own, a derived tensor (copied once) and one the model has not (0); matrix() packed with and without
// relaxed SIMD (the corrections taken and made by the packing's kernel, or none), float32, a stack of its own kinds, the
// classifier shared with the embedding, the matrices on the GPU alone (the checkpoint's offsets, nothing taken), and
// llama.cpp's blocks (the tables and the type; the signed tile's scratch where there is one); the embedding's rows of
// every kind of table; the outlier columns of the classifier; and that the memory is taken in the order the names are
// first asked for.
// What it does not see: the kernels' numbers and the engine around it (tests/forward-check.ts, tests/plan-facts-check.ts
// and the net's calls hold those). The weights here are written by the test's own hand, not by the packings' table
// (jobs.ts's PACKED): a second copy of each layout, so that a reader that took the wrong line of it is seen.
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.ts";

const { tensorsOf, partsOf } = await import(runtimeUrl("forward/tensors.js"));
const { KIND } = await import(runtimeUrl("jobs.js"));

// ---- the stand-ins
// a memory: the checkpoint from 0 to base + size, alloc() takes 64-byte lines after it and writes down what it took
function roomOf(bytes = 1 << 16, top = 8192) {
  let buffer = new ArrayBuffer(bytes), F = new Float32Array(buffer), U = new Uint8Array(buffer), H = new Uint16Array(buffer);
  const taken = [], room = {
    taken,
    growing: false,
    alloc(size) {
      if (room.growing) {  // the engine's memory grows at a take: the views of before are cut off, the numbers are in the new ones
        const grown = new ArrayBuffer(bytes);
        new Uint8Array(grown).set(U);
        buffer = grown; F = new Float32Array(buffer); U = new Uint8Array(buffer); H = new Uint16Array(buffer);
      }
      const at = top;
      top += Math.ceil(size / 64) * 64;
      assert.ok(top <= bytes, "the test's memory is large enough");
      taken.push([at, size]);
      return at;
    },
    get F() { return F; },
    get U() { return U; },
    get H() { return H; },
  };
  return room;
}
// kernels that write down every call (their arguments) and do nothing
function kernelsOf() {
  const said = [];
  return { said, k: new Proxy({}, { get: (_, name) => (...args) => { said.push([name, ...args]); return 0; } }) };
}
const f16 = (x) => {  // float16 bits of a small number exact in it
  const f = new Float32Array([x]), u = new Uint32Array(f.buffer)[0];
  if (x === 0) return 0;
  return ((u >>> 16) & 0x8000) | ((((u >>> 23) & 0xff) - 112) << 10) | ((u >>> 13) & 0x3ff);
};
const base = 0;
// a plan of the tensors given (the rest of its numbers what the readers read)
const planOf = (tensors, more = {}) => ({ dim: 128, vocab_size: 4, int8: true, shared_classifier: false, tensors, ...more });

// ---- partsOf: a stack of its own kinds is its layers; any other tensor is itself
{
  const t = { kind: "int8", offset: 0, shape: [2, 4, 32], group: 32, scales: 0, layers: [{ kind: "int8", offset: 0, group: 32, scales: 0 }, { kind: "f32", offset: 64, group: 32, scales: 0 }] };
  assert.equal(partsOf(t), t.layers);
  const one = { kind: "f32", offset: 0, shape: [4], group: 32, scales: 0 };
  assert.deepEqual(partsOf(one), [one]);
}

// ---- floats(): the address of float32 values, each kind of the file's widened once
{
  const room = roomOf(), { U, F, H } = room;
  const n = 128;
  // float32 at 0; float16 at 1024; int8 at 2048 with float32 scales at 2304; int6 at 2560 (96 bytes) with scales 2688;
  // ternary at 3072 (32 bytes), float16 scale at 3200; binary at 3328 (16 bytes), float16 scale at 3456
  for (let i = 0; i < n; i++) F[i] = i / 8;
  for (let i = 0; i < n; i++) H[1024 / 2 + i] = f16(i - 64);
  for (let i = 0; i < n; i++) U[2048 + i] = (i - 64) & 255;
  F[2304 / 4] = 0.5; F[2304 / 4 + 1] = 2; F[2304 / 4 + 2] = -1; F[2304 / 4 + 3] = 0.25;
  const six = (i) => ((i * 7) % 64) - 32;  // a six-bit weight in -32..31
  for (let g = 0; g < n / 32; g++) {
    for (let j = 0; j < 32; j++) {
      const u = six(g * 32 + j) & 63, low = u & 15, top = u >> 4, at = 2560 + g * 24;
      if (j < 16) U[at + j] |= low; else U[at + j - 16] |= low << 4;
      U[at + 16 + (j % 8)] |= top << (2 * ((j / 8) | 0));
    }
    F[2688 / 4 + g] = g + 1;
  }
  const three = (i) => (i % 3) - 1;  // a ternary weight
  for (let i = 0; i < n; i++) U[3072 + (i >> 2)] |= (three(i) + 1) << (2 * (i & 3));
  H[3200 / 2] = f16(1.5);
  // binary: bit 4 h + p of byte c is weight 64 h + 4 c + p; the weights +1 where 64 h + 4 c + p is even
  const one = (i) => (i % 2 ? -1 : 1);
  for (let i = 0; i < n; i++) if (one(i) > 0) U[3328 + ((i >> 2) & 15)] |= 1 << ((((i >> 6) & 1) << 2) | (i & 3));
  H[3456 / 2] = f16(-2);
  const T = {
    rms_att_weight: { kind: "f32", offset: 0, shape: [n], group: 32, scales: 0 },
    rms_ffn_weight: { kind: "f16", offset: 1024, shape: [n], group: 32, scales: 0 },
    bq: { kind: "int8", offset: 2048, shape: [n], group: 32, scales: 2304 },
    bk: { kind: "int6", offset: 2560, shape: [n], group: 32, scales: 2688 },
    bv: { kind: "ternary", offset: 3072, shape: [n], group: 128, scales: 3200 },
    bo: { kind: "binary", offset: 3328, shape: [n], group: 128, scales: 3456 },
  };
  const derived = { "signs.128": new Uint8Array([1, 2, 3, 4]) };
  const { k } = kernelsOf();
  const { floats } = tensorsOf({ plan: planOf(T, { derived }), base, room, k, relaxed: null, tables: 0, spread: 0 });
  assert.equal(floats("rms_att_weight"), 0, "float32 is read where it is");
  assert.equal(room.taken.length, 0, "and takes nothing");
  assert.equal(floats("q_norm"), 0, "a tensor the model has not is 0");
  const expect = {
    rms_ffn_weight: (i) => i - 64,
    bq: (i) => (i - 64) * [0.5, 2, -1, 0.25][(i / 32) | 0],
    bk: (i) => six(i) * 4 * (((i / 32) | 0) + 1),
    bv: (i) => three(i) * 1.5,
    bo: (i) => one(i) * -2,
  };
  const order = [];
  for (const [name, value] of Object.entries(expect)) {
    const at = floats(name);
    order.push(at);
    assert.equal(at % 64, 0, `${name} widened to a line of its own`);
    for (let i = 0; i < n; i++) assert.equal(F[at / 4 + i], Math.fround(value(i)), `${name}[${i}]`);
    assert.equal(floats(name), at, `${name} is widened once`);
  }
  assert.deepEqual(room.taken.map(([at]) => at), order, "the memory is taken in the order the names were asked for");
  assert.ok(room.taken.every(([, bytes]) => bytes === n * 4), "n float32s each");
  const signs = floats("signs.128");
  assert.deepEqual([...U.subarray(signs, signs + 4)], [1, 2, 3, 4], "a derived tensor is copied");
  assert.equal(floats("signs.128"), signs, "once");
}

// ---- floats() of a stack whose layers are each of a kind of their own, and of llama.cpp's blocks (a row by its kernel)
{
  const room = roomOf(), { F, H } = room;
  for (let i = 0; i < 32; i++) F[i] = i;
  for (let i = 0; i < 32; i++) H[256 / 2 + i] = f16(-i - 1);
  const T = {
    rms_att_weight: { kind: "f16", offset: 0, shape: [2, 32], group: 32, scales: 0,
      layers: [{ kind: "f32", offset: 0, group: 32, scales: 0 }, { kind: "f16", offset: 256, group: 32, scales: 0 }] },
    rms_ffn_weight: { kind: "Q4_K", offset: 1024, shape: [2, 256], group: 256, scales: 0 },
  };
  const { k, said } = kernelsOf();
  const { floats } = tensorsOf({ plan: planOf(T), base, room, k, relaxed: null, tables: 4096, spread: 0 });
  const at = floats("rms_att_weight");
  for (let i = 0; i < 32; i++) assert.equal(F[at / 4 + i], i, "layer 0 as it is");
  for (let i = 0; i < 32; i++) assert.equal(F[at / 4 + 32 + i], -i - 1, "layer 1 widened from float16");
  const blocks = floats("rms_ffn_weight");
  const [type, bytes] = [said[0][5], 144];
  assert.deepEqual(said, [["sb_row", blocks, 1024, 4096, 256, type], ["sb_row", blocks + 1024, 1024 + bytes, 4096, 256, type]],
    "every row of llama.cpp's blocks by the kernel, on the engine's tables");
}

// ---- matrix(): packed with relaxed SIMD and without, float32, a stack of its own kinds, the shared classifier
{
  const n = 64, rows = 4, layers = 2;
  const T = {
    wq: { kind: "int8", offset: 0, shape: [layers, rows, n], group: 32, scales: 1024 },
    wk: { kind: "int8", offset: 0, shape: [layers, rows, n], group: 32, scales: 1024 },
    wv: { kind: "f32", offset: 2048, shape: [layers, rows, n], group: 32, scales: 0 },
    wo: { kind: "ternary", offset: 6144, shape: [layers, rows, 128], group: 128, scales: 6400 },
    w1: { kind: "int8", offset: 0, shape: [layers, rows, n], group: 32, scales: 1024,
      layers: [{ kind: "int8", offset: 0, group: 32, scales: 1024 }, { kind: "f16", offset: 7168, group: 32, scales: 0 }] },
    token_embedding_table: { kind: "int8", offset: 0, shape: [rows, n], group: 32, scales: 1024 },
  };
  for (const relaxed of [false, true]) {
    const room = roomOf(), { k, said } = kernelsOf(), r = relaxed ? kernelsOf().k : null;
    const { matrix } = tensorsOf({ plan: planOf(T, { shared_classifier: true }), base, room, k, relaxed: r, tables: 0, spread: 0 });
    const groups = layers * rows * n / 32;
    const wq = matrix("wq");
    assert.equal(matrix("wq"), wq, "made once for a name");
    assert.deepEqual([wq.rows, wq.n, wq.kind, wq.group], [rows, n, "int8", 32]);
    assert.equal(wq.job, relaxed ? KIND.matmul_q8r : KIND.matmul_q8, "the kind of the job, as the engine runs");
    if (relaxed) {
      const [[corrections, bytes]] = room.taken;
      assert.equal(bytes, groups * 4, "an int32 correction a group");
      assert.deepEqual(said, [["int8_sums", corrections, 0, groups]], "made by the packing's kernel");
      assert.deepEqual(wq.layer(1), [rows * n, 1024 + rows * (n / 32) * 4, corrections + rows * (n / 32) * 4]);
    } else {
      assert.equal(room.taken.length, 0, "no corrections without relaxed SIMD");
      assert.deepEqual(wq.layer(1), [rows * n, 1024 + rows * (n / 32) * 4, 0]);
    }
    assert.equal(matrix("wk").ready, wq.ready, "matrices packed alike take one ready()");
    const wo = matrix("wo");
    assert.notEqual(wo.ready, wq.ready, "ternary's activations are of another form");
    assert.deepEqual(wo.layer(1).slice(0, 2), [6144 + rows * 128 / 4, 6400 + rows * 2], "ternary's float16 scales");
    const wv = matrix("wv");
    assert.deepEqual([wv.kind, wv.job, wv.ready], ["f32", KIND.matmul_f32, null]);
    assert.deepEqual(wv.layer(1), [2048 + rows * n * 4], "float32 where it is");
    const w1 = matrix("w1");
    assert.equal(w1.kind, "layers");
    assert.throws(() => w1.layer(0), /is its layers/);
    assert.deepEqual([w1.layers[0].kind, w1.layers[1].kind], ["int8", "f32"], "each layer a matrix of its own kind");
    const widened = w1.layers[1].layer(0)[0];
    assert.equal(room.taken.at(-1)[0], widened, "the float16 layer widened by itself");
    assert.equal(matrix("w2"), null, "a matrix the model has not");
    const wcls = matrix("wcls");
    assert.deepEqual(wcls.layer(0).slice(0, 2), [0, 1024], "the shared classifier is the embedding");
  }
  // the plan of a float32 engine (plan.int8 false): every matrix float32
  const room = roomOf(), { k } = kernelsOf();
  const { matrix } = tensorsOf({ plan: planOf(T, { int8: false }), base, room, k, relaxed: null, tables: 0, spread: 0 });
  assert.equal(matrix("wq").kind, "f32", "widened where the engine does not multiply packed weights");
  assert.equal(room.taken[0][1], layers * rows * n * 4);
}

// ---- matrix() on the GPU alone: the checkpoint's offsets, nothing taken, and none of float32
{
  const T = {
    wq: { kind: "int8", offset: 100, shape: [2, 4, 64], group: 32, scales: 900 },
    wo: { kind: "f32", offset: 0, shape: [2, 4, 64], group: 32, scales: 0 },
    rms_att_weight: { kind: "int8", offset: 100, shape: [2, 64], group: 32, scales: 900 },
  };
  const room = roomOf(), { k } = kernelsOf();
  const { matrix } = tensorsOf({ plan: planOf(T), base: 4096, room, k, relaxed: null, direct: {}, tables: 0, spread: 0 });
  const wq = matrix("wq");
  assert.equal(wq.onGpu, true);
  assert.deepEqual(wq.layer(1), [100 + 4 * 64, 900 + 4 * 2 * 4], "offsets in the checkpoint, not addresses");
  assert.equal(room.taken.length, 0);
  assert.throws(() => matrix("wo"), /is of f32, which is not on the GPU alone/);
}

// ---- matrix() of llama.cpp's blocks: the tables and the type; the signed tile where there is a scratch for it
{
  const T = { wq: { kind: "Q4_K", offset: 0, shape: [1, 4, 256], group: 256, scales: 0 } };
  for (const [relaxed, spread, job] of [[null, 0, "matmul_sb"], [{ matmul_sbr: 1 }, 0, "matmul_sbr"], [{ matmul_sbr: 1 }, 512, "matmul_sbw"]]) {
    const room = roomOf(), { k } = kernelsOf();
    const { matrix } = tensorsOf({ plan: planOf(T), base: 64, room, k, relaxed, tables: 4096, spread });
    const wq = matrix("wq");
    assert.equal(wq.job, KIND[job], `${job}`);
    assert.deepEqual(wq.layer(0).slice(1), [4096, wq.layer(0)[2], spread]);
    assert.equal(wq.layer(0)[0], 64, "the values where they are");
    assert.equal(room.taken.length, 0, "no corrections");
  }
}

// ---- embedder(): a token's row of every kind of table, into float32 at the address given
{
  const dim = 128, vocab = 4;
  for (const kind of ["f32", "f16", "int8", "ternary", "Q4_K"]) {
    const room = roomOf(), { F, U, H } = room, { k, said } = kernelsOf();
    const value = (v, i) => ((v * 5 + i) % 7) - 3;
    for (let v = 0; v < vocab; v++) {
      for (let i = 0; i < dim; i++) {
        if (kind === "f32") F[v * dim + i] = value(v, i);
        if (kind === "f16") H[v * dim + i] = f16(value(v, i));
        if (kind === "int8") U[v * dim + i] = value(v, i) & 255;
        if (kind === "ternary") U[v * dim / 4 + (i >> 2)] |= ((value(v, i) % 2) + 1) << (2 * (i & 3));
      }
    }
    const scales = 4096;
    for (let g = 0; g < vocab * dim / 32; g++) F[scales / 4 + g] = g + 1;
    for (let g = 0; g < vocab; g++) H[(scales + 2048) / 2 + g] = f16(g + 0.5);
    const group = kind === "ternary" ? 128 : kind === "Q4_K" ? 256 : 32;
    const T = { token_embedding_table: { kind, offset: 0, shape: [vocab, dim], group, scales: kind === "ternary" ? scales + 2048 : scales } };
    const { embedder } = tensorsOf({ plan: planOf(T, { dim, vocab_size: vocab }), base, room, k, relaxed: null, tables: 4000, spread: 0 });
    const row = embedder(), to = 7168;
    assert.equal(room.taken.length, kind === "f16" ? 1 : 0, `${kind}: a float16 table is widened as the embedder is made, nothing else`);
    row(2, to);
    if (kind === "Q4_K") {
      assert.deepEqual(said, [["sb_row", to, 2 * 144 / 2, 4000, dim, said[0][5]]], "a row of blocks by its kernel");
      continue;
    }
    for (let i = 0; i < dim; i++) {
      const want = kind === "int8" ? value(2, i) * (2 * dim / 32 + ((i / 32) | 0) + 1)
        : kind === "ternary" ? (value(2, i) % 2) * 2.5 : value(2, i);
      assert.ok(F[to / 4 + i] === Math.fround(want), `${kind}: weight ${i} of token 2: ${F[to / 4 + i]}, not ${want}`);  // (=== : -0 and 0 alike)
    }
  }
}

// ---- outlierColumns(): the classifier's columns of the outlier channels, in float32
{
  const dim = 64, vocab = 3;
  const room = roomOf(), { F, U } = room, { k } = kernelsOf();
  for (let v = 0; v < vocab; v++) for (let i = 0; i < dim; i++) U[v * dim + i] = (v * 10 + i) & 255;
  for (let g = 0; g < vocab * dim / 32; g++) F[1024 / 4 + g] = g + 1;
  const T = { token_embedding_table: { kind: "int8", offset: 0, shape: [vocab, dim], group: 32, scales: 1024 } };
  const plan = planOf(T, { dim, vocab_size: vocab, shared_classifier: true });
  const { outlierColumns } = tensorsOf({ plan, base, room, k, relaxed: null, tables: 0, spread: 0 });
  assert.deepEqual(outlierColumns([]), { columns: 0, picked: 0 }, "none without outlier channels");
  assert.equal(room.taken.length, 0);
  const { columns, picked } = outlierColumns([3, 40]);
  assert.deepEqual(room.taken, [[columns, 2 * vocab * 4], [picked, 2 * 4]], "the columns, then where the channels are put aside");
  for (const [i, c] of [3, 40].entries()) {
    for (let v = 0; v < vocab; v++) assert.equal(F[columns / 4 + i * vocab + v], (v * 10 + c) * (v * 2 + ((c / 32) | 0) + 1), `channel ${c}, token ${v}`);
  }
  const alone = tensorsOf({ plan, base, room: roomOf(), k, relaxed: null, direct: {}, tables: 0, spread: 0 });
  assert.deepEqual(alone.outlierColumns([3]), { columns: 0, picked: 0 }, "none of a classifier on the GPU alone");
}

console.log("tensors-check: ok");

// ---- the memory grows at the take a widened copy is made with: the numbers are written in the views of after it
{
  const room = roomOf();
  for (let i = 0; i < 128; i++) room.H[1024 / 2 + i] = f16(i - 64);
  for (let i = 0; i < 128; i++) room.U[2048 + i] = (i - 64) & 255;
  room.F[2304 / 4] = 0.5;
  const tensors = {
    h: { kind: "f16", offset: 1024, shape: [128], group: 32, scales: 0 },
    q: { kind: "int8", offset: 2048, shape: [128], group: 128, scales: 2304 },
  };
  const { floats } = tensorsOf({ plan: planOf(tensors), base, room, k: kernelsOf().k, relaxed: null, tables: 0, spread: 0 });
  room.growing = true;
  const h = floats("h"), q = floats("q");
  for (let i = 0; i < 128; i++) {
    assert.equal(room.F[h / 4 + i], i - 64, `float16 ${i}, in the views after the memory grew`);
    assert.equal(room.F[q / 4 + i], (i - 64) * 0.5, `int8 ${i}, in the views after the memory grew`);
  }
}
console.log("tensors-check: ok (growing)");
