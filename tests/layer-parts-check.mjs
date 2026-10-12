// tests/layer-parts-check.mjs (T375.3): the parts the engine chooses as it is made, each by itself with stand-ins for
// its neighbours: the three kinds of layers (src/runtime/forward/attention.ts, linear.ts, convolution.ts), the two
// FFNs (ffn.ts) and the cache of the keys and values (cache.ts). No engine, no model, no kernels, no Pyodide: a few
// bytes of memory, kernels that write down what they are asked, and the phases and matmuls written down the same.
// Node only, under a second, in the light suite.
//
//   node tests/layer-parts-check.mjs
//
// What it sees of each kind: what it puts in a token's frame and says it keeps besides, of what sizes the model
// (T375.5: sizes.ts's Sizes, the plan's or the form's), and what it says of a form (which layers are its own, its
// widths); that it takes its tensors by their names and of the memory what it said it keeps; a step of several tokens (what goes out as matmuls and phases, in what order, at what
// addresses); that position 0's clear() empties its state and nothing else; that a kind which keeps keys and values
// has no clear(); and that a plan without a kind's numbers is refused as the kind is asked.
// What it does not see: the numbers (tests/plan-facts-check.mjs and tests/forward-check.mjs hold the engine to NumPy).
import assert from "node:assert/strict";
import { runtimeUrl } from "./tree.mjs";

const { attention } = await import(runtimeUrl("forward/attention.js"));
const { linear } = await import(runtimeUrl("forward/linear.js"));
const { convolution } = await import(runtimeUrl("forward/convolution.js"));
const { gatedFfn, plainFfn } = await import(runtimeUrl("forward/ffn.js"));
const { keyValueCache } = await import(runtimeUrl("forward/cache.js"));
const { LAYER_KINDS } = await import(runtimeUrl("forward/kinds.js"));

// ---- the stand-ins
// a memory of a few pages: alloc() takes 64-byte lines of it from 4096 on, and the views are read each time (the test
// swaps the buffer for a larger one as a WebAssembly memory does when it grows)
function memoryOf(bytes = 1 << 20) {
  let buffer = new ArrayBuffer(bytes), top = 4096;
  const taken = [];
  return {
    taken,
    alloc(size) {
      const at = top;
      top += Math.ceil(size / 64) * 64;
      if (top > buffer.byteLength) {
        const larger = new ArrayBuffer(2 * top);
        new Uint8Array(larger).set(new Uint8Array(buffer));
        buffer = larger;
      }
      taken.push([at, size]);
      return at;
    },
    get F() { return new Float32Array(buffer); },
    get U() { return new Uint8Array(buffer); },
    get H() { return new Uint16Array(buffer); },
  };
}
// what a part is handed: the tensors the model has (a matrix is its name, a vector an address of its own), the frame
// (every name an address, S apart), and everything asked of the kernels, of matmuls() and of phase() in said, in order
// (sizes: what sizes the model, as the kind's frame() and state() are asked with)
function coreOf(plan, { has = [], cache, sizes } = {}) {
  const said = [], room = memoryOf();
  const vectors = new Map(), names = ["x", "xb", "xb2", "q", "kNow", "vNow", "before", "hb", "hb2", "gate", "mixed", "conv", "z", "work", "att"];
  const at = Object.fromEntries(names.map((name, i) => [name, 1000 * (i + 1)]));
  const core = {
    plan, sizes, eps: 1e-5, room, at, S: 100000, cache, said,
    k: new Proxy({}, { get: (_, name) => (...args) => { said.push([name, ...args]); return 0; } }),
    matrix: (name) => (has.includes(name) ? { name } : null),
    floats: (name) => (has.includes(name) ? (vectors.has(name) ? vectors.get(name) : vectors.set(name, 7e6 + 1e5 * vectors.size).get(name)) : 0),
    matmuls: (input, count, list) => said.push(["matmuls", input, count, ...list.map(([m, out, stride, l]) => [m?.name, out, stride, l])]),
    phase: (jobs) => said.push(["phase", ...jobs]),
  };
  return { core, said, room, at, vector: (name) => vectors.get(name) };
}
const names = (said) => said.map(([name]) => name);
const S = 100000;

// ---- the cache of the keys and values
{
  for (const half of [false, true]) {
    const room = memoryOf(8192), asked = [];
    const k = new Proxy({}, { get: (_, name) => (...args) => asked.push([name, ...args]) });
    const cache = keyValueCache({ room, k, kvDim: 8, half, start: 4, seqLen: 30 });
    const KV = half ? 16 : 32;
    assert.equal(cache.KV, KV);
    assert.deepEqual([cache.hold(2), cache.hold(1)], [0, 2], "the kinds' layers have their places in turn");
    assert.equal(room.taken.length, 0, "nothing is taken of the memory before place()");
    cache.place();
    assert.deepEqual(room.taken, [[4096, 2 * 3 * 4 * KV]], "the keys and the values of three layers, four positions");
    assert.deepEqual([cache.capacity, cache.keys, cache.values], [4, 4096, 4096 + 3 * 4 * KV]);
    assert.throws(() => cache.hold(1), /has its room already/, "no layer takes a place once the cache has its room (it would have none)");
    assert.throws(() => cache.place(), /has its room already/, "and the cache takes its room once");
    // every byte of every layer's keys and values says whose it is, and is where it belongs after the cache grew twice
    const mark = () => {
      for (const [block, tag] of [[cache.keys, 10], [cache.values, 20]]) {
        for (let l = 0; l < 3; l++) for (let i = 0; i < 4 * KV; i++) room.U[block + l * cache.capacity * KV + i] = tag + l + (i % 7) * 30;
      }
    };
    const held = () => {
      for (const [block, tag] of [[cache.keys, 10], [cache.values, 20]]) {
        for (let l = 0; l < 3; l++) for (let i = 0; i < 4 * KV; i++) assert.equal(room.U[block + l * cache.capacity * KV + i], tag + l + (i % 7) * 30, `layer ${l}, byte ${i}`);
      }
    };
    mark();
    cache.grow(4);
    assert.equal(cache.capacity, 8, "twice the positions");
    assert.equal(cache.values, cache.keys + 3 * 8 * KV);
    held();
    cache.grow(25);
    assert.equal(cache.capacity, 26, "or up to the position asked for");
    held();
    cache.grow(100);
    assert.equal(cache.capacity, 30, "and no more than the context");
    held();
    // a float32 key and value go in as they are or through the kernel, and float16 ones the other way round
    room.U.fill(7, 100, 100 + 32);
    cache.put(cache.keys, cache.values, 100, 100);
    cache.putHalves(cache.keys + KV, cache.values + KV, 100, 100);
    assert.deepEqual(asked, half ? [["to_f16", cache.keys, 100, 8], ["to_f16", cache.values, 100, 8]]
      : [["from_f16", cache.keys + KV, 100, 8], ["from_f16", cache.values + KV, 100, 8]]);
    const copied = half ? cache.keys + KV : cache.keys, width = half ? 16 : 32;
    assert.ok(room.U.subarray(copied, copied + width).every((byte) => byte === 7), "the other is copied");
    const read = cache.read(0, 2);
    assert.deepEqual([read.keys.length, read.values.length], [3 * 2 * 8, 3 * 2 * 8], "read back as [layers][positions][kvDim]");
  }
  // a model on the GPU alone has no cache: no room for a position
  const room = memoryOf(), none = keyValueCache({ room, k: {}, kvDim: 8, half: true, start: 0, seqLen: 30 });
  none.hold(4);
  none.place();
  assert.deepEqual([none.capacity, room.taken[0][1]], [0, 0]);
  console.log("ok: the cache: the kinds' layers take their places, it takes its room when it is placed, every byte is kept as it grows (twice, to a position, to the context), in float32 and in float16");
}

// ---- the layers that attend
{
  const plan = { dim: 64, n_heads: 4, n_kv_heads: 2, head_size: 16, seq_len: 48, rotary: 16, widths: { q: 64, kv: 32 }, tensors: {} };
  const sizes = { dim: 64, heads: 4, seqLen: 48, widths: { q: 64, kv: 32 }, gate: false, numbers: {} };
  assert.deepEqual(attention.frame(sizes), { arrays: [], apart: [["att", 48 * 4 * 4]] }, "the scores of a token, apart from its frame");
  assert.deepEqual(attention.frame({ ...sizes, gate: true }).arrays, [["gate", 64 * 4]], "and a gate where the layers gate their output");
  // (T375.5) what footprint() reads of the kind: its layers keep keys and values, and nothing besides; and it says no
  // layer of a form is its own (a layer that no kind claims attends)
  assert.deepEqual([attention.cached, attention.state, attention.layers, attention.widths], [true, undefined, undefined, undefined]);
  const stand = (has, half = false) => {
    const room = memoryOf(), cacheSaid = [];
    const cache = keyValueCache({ room, k: new Proxy({}, { get: (_, name) => (...args) => cacheSaid.push([name, ...args]) }), kvDim: 32, half, start: 16, seqLen: 48 });
    cache.hold(5);  // (another kind's layers before these)
    const made = coreOf(plan, { has, cache, sizes }), layers = attention.make(made.core, 3);
    // (what says the kind's layers are cached is true: it took its places in the cache)
    assert.equal(cache.hold(0), 5 + 3);
    cache.place();
    return { ...made, layers, cache, cacheSaid };
  };
  // a Llama's layer: three matrices, RoPE, the cache, one phase of every token's heads, the output
  const { said, layers, cache, at, vector, cacheSaid } = stand(["wq", "wk", "wv", "wo", "freq_cis_real", "freq_cis_imag"], true);
  assert.equal(layers.clear, undefined, "keys and values can be written again at any position: nothing to clear");
  layers.layer(1, { kind: "attention", place: 1, rope: true })(3, 5);
  const KV = cache.KV, layerKeys = cache.keys + (5 + 1) * 16 * KV, layerValues = cache.values + (5 + 1) * 16 * KV;
  assert.deepEqual(names(said), ["matmuls", "rope", "rope", "rope", "rope", "rope", "rope", "phase", "matmuls"]);
  assert.deepEqual(said[0], ["matmuls", at.xb, 3, ["wq", at.q, S, 1], ["wk", at.kNow, S, 1], ["wv", at.vNow, S, 1]]);
  const cos = vector("freq_cis_real"), sin = vector("freq_cis_imag");
  assert.deepEqual(said[3], ["rope", at.q + S, cos + 6 * 8 * 4, sin + 6 * 8 * 4, 4, 16, 16], "the second token's q, at position 6");
  assert.deepEqual(said[4], ["rope", at.kNow + S, cos + 6 * 8 * 4, sin + 6 * 8 * 4, 2, 16, 16]);
  assert.deepEqual(cacheSaid.slice(2, 4), [["to_f16", layerKeys + 6 * KV, at.kNow + S, 32], ["to_f16", layerValues + 6 * KV, at.vNow + S, 32]], "into the layer's own keys and values, at the token's position");
  assert.deepEqual(said[7], ["phase", ...[0, 1, 2].map((t) => [4, at.xb + t * S, at.q + t * S, layerKeys, layerValues, at.att + t * 48 * 4 * 4, 5 + t, 2, 16, 4, 1, 0, 0, 0])],
    "the heads of every token as one phase, over float16 keys and values");
  assert.deepEqual(said[8], ["matmuls", at.xb, 3, ["wo", at.xb2, S, 1]]);
  // a layer RoPE leaves alone, on float32 keys and values
  const plain = stand(["wq", "wk", "wv", "wo", "freq_cis_real", "freq_cis_imag"]);
  plain.layers.layer(0, { kind: "attention", place: 0, rope: false })(2, 0);
  assert.deepEqual(names(plain.said), ["matmuls", "phase", "matmuls"]);
  assert.equal(plain.said[1][1][0], 3, "the attention over float32 keys and values");
  // Qwen2's biases, Qwen3's norms of the heads and a Qwen3.5's gate, each where the model has the tensor
  const all = stand(["wq", "wk", "wv", "wo", "wg", "bq", "bk", "bv", "q_norm", "k_norm", "freq_cis_real", "freq_cis_imag"]);
  all.layers.layer(2, { kind: "attention", place: 2, rope: true })(1, 9);
  assert.deepEqual(names(all.said), ["matmuls", "add_inplace", "add_inplace", "add_inplace", ...Array(6).fill("rmsnorm"), "rope", "rope", "phase", "gate", "matmuls"]);
  assert.equal(all.said[0].length, 3 + 4, "the gate is one more matrix of the same input");
  assert.deepEqual(all.said[1], ["add_inplace", all.at.q, all.vector("bq") + 2 * 64 * 4, 64]);
  assert.deepEqual(all.said[2], ["add_inplace", all.at.kNow, all.vector("bk") + 2 * 32 * 4, 32]);
  assert.deepEqual(all.said[5], ["rmsnorm", all.at.q + 16 * 4, all.at.q + 16 * 4, all.vector("q_norm") + 2 * 16 * 4, 16, 1e-5], "the second head of q");
  assert.deepEqual(all.said[13], ["gate", all.at.xb, all.at.xb, all.at.gate, 64]);
  console.log("ok: attention: its frame; a step's matmuls, biases, norms of the heads, RoPE, cache and phase in their order and at their addresses; no state to clear");
}

// ---- the linear-attention layers
{
  const numbers = { every: 2, key_heads: 2, value_heads: 4, key_dim: 8, value_dim: 8, conv: 3 };
  const plan = { dim: 16 }, sizes = { dim: 16, widths: { linear: { mixed: 64, keys: 16, read: 32 } }, numbers: { linear: numbers } };
  assert.deepEqual(linear.frame(sizes), { arrays: [["mixed", 256], ["conv", 256], ["z", 128], ["work", (8 + 32) * 4]], widest: 32 });
  assert.throws(() => linear.frame({ ...sizes, numbers: { linear: null } }), /not the numbers of the layers of its kind/);
  assert.throws(() => linear.make(coreOf(plan, { sizes: { ...sizes, widths: { linear: null } } }).core, 1), /not the numbers of the layers of its kind/);
  // (T375.5) what it says of a form, where there is no plan: every every-th layer attends, the others are its own; and
  // its widths, as Python's plan has them (tests/plan-keys-check.mjs holds both to Python's)
  assert.deepEqual(linear.layers({ linear: numbers }, 5), [true, false, true, false, true]);
  assert.deepEqual(linear.layers({ linear: { ...numbers, every: 4 } }, 8), [true, true, true, false, true, true, true, false]);
  assert.deepEqual([linear.layers({}, 5), linear.layers({ linear: null }, 5), linear.widths({})], [null, null, {}], "none of a form without its numbers");
  assert.deepEqual(linear.widths({ linear: numbers }), sizes.widths);
  assert.equal(linear.cached, undefined, "no keys and values");
  const { core, said, room, at, vector } = coreOf(plan, { sizes, has: ["wqkv", "wz", "wout", "wb", "wa", "conv", "dt_bias", "decay", "delta_norm"] });
  const layers = linear.make(core, 2);
  const stateBytes = 4 * 8 * 8 * 4, convBytes = 3 * 64 * 4;
  assert.deepEqual(room.taken.map(([, size]) => size), [2 * 2 * stateBytes, 2 * convBytes, 32, 32], "two states a layer, the convolution's rows, the two weights of the l2 norms");
  assert.deepEqual(linear.state(sizes, 2), room.taken.map(([, size]) => size), "what it says it keeps (footprint() counts that) is what it takes of the memory");
  const [[states], [rows], [qUnit], [kUnit]] = room.taken;
  assert.deepEqual([...room.F.subarray(qUnit / 4, qUnit / 4 + 8)], Array(8).fill(Math.fround(1 / 8)));
  assert.deepEqual([...room.F.subarray(kUnit / 4, kUnit / 4 + 8)], Array(8).fill(Math.fround(1 / Math.sqrt(8))));
  // three tokens through the second layer: the delta rule reads one of the layer's two states and writes the other, turn about
  const second = layers.layer(1, { kind: "linear", place: 1, rope: false });
  room.F.fill(3, (at.mixed + S) / 4, (at.mixed + S) / 4 + 64);  // (the second token's q, k and v)
  second(3, 0);
  const rules = said.filter(([name]) => name === "phase").map(([, job]) => job);
  assert.equal(rules.length, 3, "a phase a token");
  const mine = states + 2 * stateBytes;
  assert.deepEqual(rules.map(([kind, , now, next]) => [kind, now, next]), [[7, mine, mine + stateBytes], [7, mine + stateBytes, mine], [7, mine, mine + stateBytes]],
    "what a token wrote is what the next one reads, and never the state it reads itself (T120)");
  assert.deepEqual(rules[1].slice(4), [at.conv + S, at.work + S, 2, 8, 8, 4, 1, 0, 0, 0]);
  assert.deepEqual(said[0], ["matmuls", at.xb, 3, ["wqkv", at.mixed, S, 1], ["wz", at.z, S, 1]]);
  assert.deepEqual(said.at(-1), ["matmuls", at.xb, 3, ["wout", at.xb2, S, 1]]);
  const convolved = said.filter(([name]) => name === "convolve");
  assert.deepEqual(convolved[1], ["convolve", at.conv + S, vector("conv") + convBytes, rows + convBytes, 64, 3]);
  // (after the third token the rows hold the last three tokens' values, the oldest first: the second token's are in the middle)
  const row = (i) => [...room.F.subarray((rows + convBytes) / 4 + i * 64, (rows + convBytes) / 4 + (i + 1) * 64)];
  assert.ok(row(1).every((value) => value === 3) && row(0).every((value) => value === 0) && row(2).every((value) => value === 0), "the rows move up by one a token");
  // the first layer has its own states: its first token reads the first of them
  said.length = 0;
  layers.layer(0, { kind: "linear", place: 0, rope: false })(1, 3);
  assert.deepEqual(said.find(([name]) => name === "phase")[1].slice(2, 4), [states, states + stateBytes]);
  // position 0: the states and the rows are cleared, and nothing beside them
  room.F.fill(5, states / 4 - 1, (kUnit + 32) / 4 + 1);
  layers.clear();
  const F = room.F;
  assert.ok(F.subarray(states / 4, (states + 4 * stateBytes) / 4).every((value) => value === 0), "the states");
  assert.ok(F.subarray(rows / 4, (rows + 2 * convBytes) / 4).every((value) => value === 0), "the rows");
  assert.deepEqual([F[states / 4 - 1], F[qUnit / 4], F[kUnit / 4 + 8]], [5, 5, 5], "and nothing else");
  console.log("ok: linear attention: its frame and its numbers; two states a layer, read and written turn about; the convolution's rows; what position 0 clears");
}

// ---- the convolution layers
{
  const plan = { dim: 16 }, form = { convolution: { layers: "cac", taps: 3 } }, sizes = { dim: 16, numbers: form };
  assert.deepEqual(convolution.frame(sizes), { arrays: [["mixed", 3 * 16 * 4]] });
  assert.throws(() => convolution.frame({ dim: 16, numbers: { convolution: null } }), /not the numbers of the layers of its kind/);
  // (T375.5) what it says of a form: a letter a layer; no widths of its own, no keys and values
  assert.deepEqual([convolution.layers(form, 3), convolution.layers({}, 3)], [[true, false, true], null]);
  assert.deepEqual([convolution.widths, convolution.cached], [undefined, undefined]);
  const { core, said, room, at, vector } = coreOf(plan, { sizes, has: ["win", "wout", "conv"] });
  const layers = convolution.make(core, 2), bytes = 3 * 16 * 4;
  assert.deepEqual(room.taken, [[4096, 2 * bytes]], "taps rows of dim values a layer");
  assert.deepEqual(convolution.state(sizes, 2), [2 * bytes], "what it says it keeps is what it takes of the memory");
  const rows = 4096 + bytes;
  room.F.fill(9, (rows + 2 * 64) / 4, (rows + bytes) / 4);  // (the newest of the second layer's rows)
  layers.layer(1, { kind: "convolution", place: 1, rope: false })(2, 4);
  assert.deepEqual(said, [["matmuls", at.xb, 2, ["win", at.mixed, S, 1]],
    ["short_conv", at.xb, vector("conv") + bytes, rows, at.mixed, 16, 3], ["short_conv", at.xb + S, vector("conv") + bytes, rows, at.mixed + S, 16, 3],
    ["matmuls", at.xb, 2, ["wout", at.xb2, S, 1]]]);
  assert.ok(room.F.subarray(rows / 4, rows / 4 + 16).every((value) => value === 9), "the rows moved up by one a token: the newest is the oldest after two");
  room.F.fill(5, 4096 / 4 - 1, (4096 + 2 * bytes) / 4 + 1);
  layers.clear();
  assert.ok(room.F.subarray(4096 / 4, (4096 + 2 * bytes) / 4).every((value) => value === 0), "position 0 clears the rows");
  assert.deepEqual([room.F[4096 / 4 - 1], room.F[(4096 + 2 * bytes) / 4]], [5, 5], "and nothing else");
  console.log("ok: a short convolution: its frame and its numbers; a step's two matmuls and the kernel between them, a token at a time; the rows; what position 0 clears");
}

// ---- the two FFNs
{
  const plan = { dim: 16, hidden_dim: 48 };
  const gated = coreOf(plan, { has: ["w1", "w2", "w3"] });
  gatedFfn(gated.core)(3, 2);
  const at = gated.at;
  assert.deepEqual(gated.said, [["matmuls", at.xb, 2, ["w1", at.hb, S, 3], ["w3", at.hb2, S, 3]],
    ["swiglu", at.hb, at.hb, at.hb2, 48], ["swiglu", at.hb + S, at.hb + S, at.hb2 + S, 48],
    ["matmuls", at.hb, 2, ["w2", at.xb2, S, 3]], ["add_inplace", at.x, at.xb2, 16], ["add_inplace", at.x + S, at.xb2 + S, 16]]);
  const plain = coreOf(plan, { has: ["w1", "w2", "b1", "b2"] });
  plainFfn(plain.core)(3, 1);
  assert.deepEqual(plain.said, [["matmuls", at.xb, 1, ["w1", at.hb, S, 3]], ["gelu", at.hb, at.hb, plain.vector("b1") + 3 * 48 * 4, 48],
    ["matmuls", at.hb, 1, ["w2", at.xb2, S, 3]], ["add_inplace", at.x, at.xb2, 16], ["add_inplace", at.x, plain.vector("b2") + 3 * 16 * 4, 16]]);
  console.log("ok: the FFNs: a gated one (two matrices, swiglu, one more, into the stream) and GPT-2's (a matrix, gelu with its bias, one more, into the stream with its bias)");
}

// ---- the table
assert.deepEqual(Object.keys(LAYER_KINDS), ["attention", "linear", "convolution"], "the kinds, in the order they are made in (what they take of the memory is in that order)");
assert.deepEqual([LAYER_KINDS.attention, LAYER_KINDS.linear, LAYER_KINDS.convolution], [attention, linear, convolution]);
for (const [name, kind] of Object.entries(LAYER_KINDS)) assert.deepEqual([typeof kind.frame, typeof kind.make], ["function", "function"], name);
console.log("layer-parts-check: ok");
