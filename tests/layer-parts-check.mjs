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
  const vectors = new Map(), names = ["x", "xb", "xb2", "q", "kNow", "vNow", "before", "hb", "hb2", "gate", "mixed", "conv", "z", "work", "att", "tiled"];
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
  // T510.1: a layer that attends the last 5 positions of a context of 60 keeps a ring of 2 × (5 + 16) = 42 positions
  // (its period 21: the window and a prompt's block), taken first and never grown; the others keep every position, in
  // blocks after the rings that grow as before. A window whose ring would not be shorter than the context (22: 76 of
  // 60) keeps every position, and so does every layer of a model on the GPU alone (start 0)
  {
    const room = memoryOf(), cache = keyValueCache({ room, k: {}, kvDim: 8, half: false, start: 4, seqLen: 60 });
    assert.deepEqual([cache.hold(3, [0, 5, 22]), cache.hold(2, [5])], [0, 3], "the places in turn, whatever their windows");
    assert.deepEqual([0, 1, 2, 3, 4].map((place) => [cache.index(place), cache.ring(place)?.size ?? 0]), [[0, 0], [-1, 21], [1, 0], [-1, 21], [2, 0]],
      "a window of 5 in a ring of period 21 (a window and a block), the others in the blocks, by their index among those");
    cache.place();
    const KV = 32, first = cache.ring(1), second = cache.ring(3);
    assert.deepEqual(room.taken, [[4096, 2 * 4 * 21 * KV], [4096 + 2 * 4 * 21 * KV, 2 * 3 * 4 * KV]], "the rings first, all at once, then the blocks that grow");
    assert.deepEqual([first.keys, first.values, second.keys, second.values], [4096, 4096 + 42 * KV, 4096 + 84 * KV, 4096 + 126 * KV], "each ring's keys, then its values, 42 positions each");
    assert.deepEqual([cache.keys, cache.values], [4096 + 168 * KV, 4096 + 168 * KV + 3 * 4 * KV]);
    // a ring keeps its place as the blocks grow, and is read back where its positions are now: position p at p mod 21
    for (const [ring, tag] of [[first, 1], [second, 2]]) for (let p = 0; p < 21; p++) room.F.fill(tag * 100 + p, ring.keys / 4 + p * 8, ring.keys / 4 + p * 8 + 8);
    cache.grow(40);
    assert.deepEqual([cache.capacity, first.keys, cache.keys], [41, 4096, 4096 + 168 * KV]);
    const read = cache.read(19, 4);
    assert.deepEqual([1, 3].map((place) => [0, 1, 2, 3].map((t) => read.keys[(place * 4 + t) * 8])), [[119, 120, 100, 101], [219, 220, 200, 201]]);
    const all = keyValueCache({ room: memoryOf(), k: {}, kvDim: 8, half: false, start: 0, seqLen: 60 });
    all.hold(2, [5, 0]);
    assert.deepEqual([all.ring(0), all.index(0), all.index(1)], [null, 0, 1], "no ring on the GPU alone");
  }
  console.log("ok: the cache: the kinds' layers take their places, it takes its room when it is placed, every byte is kept as it grows (twice, to a position, to the context), in float32 and in float16; a window's ring, taken first and never grown");
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
  // T261: a local layer (a Gemma 3's): RoPE by the local layers' tables, and the cache handed over from the first
  // position of each token's window on, with the token's position counted from there. Four tokens at positions 5 to 8
  // and a window of 7: the first two attend from position 0, the third from 1, the fourth from 2
  {
    const local = stand(["wq", "wk", "wv", "wo", "freq_cis_real", "freq_cis_imag", "local_cis_real", "local_cis_imag"]);
    local.layers.layer(1, { kind: "attention", place: 1, rope: true, window: 7, local: true })(4, 5);
    const KV = local.cache.KV, keys = local.cache.keys + (5 + 1) * 16 * KV, values = local.cache.values + (5 + 1) * 16 * KV;
    const cos = local.vector("local_cis_real"), sin = local.vector("local_cis_imag");
    assert.ok(cos && sin && cos !== local.vector("freq_cis_real"));
    assert.deepEqual(local.said[3], ["rope", local.at.q + S, cos + 6 * 8 * 4, sin + 6 * 8 * 4, 4, 16, 16], "the second token's q, by the local tables");
    assert.deepEqual(local.said.at(-2), ["phase", ...[0, 1, 2, 3].map((t) => {
      const from = [0, 0, 1, 2][t];
      return [3, local.at.xb + t * S, local.at.q + t * S, keys + from * KV, values + from * KV, local.at.att + t * 48 * 4 * 4, 5 + t - from, 2, 16, 4, 1, 0, 0, 0];
    })], "each token's heads over the last 7 positions, its own among them");
    // the two facts apart: a window with the model's own tables, and the local tables with no window
    const windowed = stand(["wq", "wk", "wv", "wo", "freq_cis_real", "freq_cis_imag", "local_cis_real", "local_cis_imag"]);
    windowed.layers.layer(0, { kind: "attention", place: 0, rope: true, window: 2, local: false })(1, 9);
    assert.deepEqual(windowed.said[1].slice(2, 4), [windowed.vector("freq_cis_real") + 9 * 8 * 4, windowed.vector("freq_cis_imag") + 9 * 8 * 4]);
    assert.deepEqual(windowed.said.at(-2)[1].slice(3, 7), [windowed.cache.keys + 5 * 16 * KV + 8 * KV, windowed.cache.values + 5 * 16 * KV + 8 * KV, windowed.at.att, 1],
      "a window of 2 at position 9: from position 8, where the token is the second");
    windowed.said.length = 0;
    windowed.layers.layer(0, { kind: "attention", place: 0, rope: true, window: 0, local: true })(1, 9);
    assert.deepEqual(windowed.said[1].slice(2, 4), [windowed.vector("local_cis_real") + 9 * 8 * 4, windowed.vector("local_cis_imag") + 9 * 8 * 4]);
    assert.equal(windowed.said.at(-2)[1][6], 9, "no window: every position");
    // a window of one position: the token itself; and one longer than the text: every position
    windowed.said.length = 0;
    windowed.layers.layer(0, { kind: "attention", place: 0, rope: true, window: 1, local: false })(2, 3);
    assert.deepEqual(windowed.said.at(-2).slice(1).map((job) => job[6]), [0, 0]);
    windowed.said.length = 0;
    windowed.layers.layer(0, { kind: "attention", place: 0, rope: true, window: 40, local: false })(2, 3);
    assert.deepEqual(windowed.said.at(-2).slice(1).map((job) => job[6]), [3, 4]);
    // a plan with a local layer and no tables for it is refused as the layer is made
    assert.throws(() => stand(["wq", "wk", "wv", "wo", "freq_cis_real", "freq_cis_imag"]).layers.layer(0, { kind: "attention", place: 0, rope: true, window: 4, local: true }),
      /^Error: The plan has a local layer and no RoPE tables of the local layers\.$/);
  }
  assert.deepEqual(all.said[13], ["gate", all.at.xb, all.at.xb, all.at.gate, 64]);
  // T510.1: the layers of a kind whose facts give windows: those whose ring is shorter than the context (5 of 48: 42
  // positions) write each token's key and value twice into it, at p mod 21 and 21 after, and hand the kernels the run
  // of the ring from where each token's window begins. Four tokens at 19 to 22: slots 19, 20, 0 and 1; their windows
  // from 15 to 18
  {
    const room = memoryOf(), cacheSaid = [];
    const cache = keyValueCache({ room, k: new Proxy({}, { get: (_, name) => (...args) => cacheSaid.push([name, ...args]) }), kvDim: 32, half: true, start: 16, seqLen: 48 });
    const made = coreOf(plan, { has: ["wq", "wk", "wv", "wo", "freq_cis_real", "freq_cis_imag", "local_cis_real", "local_cis_imag"], cache, sizes });
    const facts = [0, 1, 2].map((place) => ({ kind: "attention", place, rope: true, window: place === 1 ? 5 : 0, local: place === 1 }));
    const layers = attention.make(made.core, 3, facts);
    assert.deepEqual([cache.ring(0), cache.ring(1)?.size, cache.index(2)], [null, 21, 1]);
    const steps = facts.map((fact) => layers.layer(fact.place, fact));
    cache.place();
    const ring = cache.ring(1), KV = cache.KV;
    steps[1](4, 19);
    assert.deepEqual(cacheSaid, [19, 20, 0, 1].flatMap((slot, t) => [slot, slot + 21].flatMap((at) => [["to_f16", ring.keys + at * KV, made.at.kNow + t * S, 32],
      ["to_f16", ring.values + at * KV, made.at.vNow + t * S, 32]])), "every key and value twice: at p mod 21, and 21 after");
    assert.deepEqual(made.said.at(-2), ["phase", ...[15, 16, 17, 18].map((from, t) => [4, made.at.xb + t * S, made.at.q + t * S, ring.keys + from * KV, ring.values + from * KV, made.at.att + t * 48 * 4 * 4, 4, 2, 16, 4, 1, 0, 0, 0])],
      "each token's heads over the run of the ring that holds its window, oldest first");
    // across the ring's seam: position 40's window from 36, slot 15 (36 mod 21) and on, past 20 into the second copy
    made.said.length = 0;
    steps[1](1, 23);
    steps[1](1, 40);
    assert.deepEqual(made.said.at(-2)[1].slice(3, 7), [ring.keys + 15 * KV, ring.values + 15 * KV, made.at.att, 4]);
    // a run may go back no further than a block before the last position written (24 back from 40), and begins again at 0
    assert.throws(() => steps[1](1, 23), /^Error: A layer of this model keeps the keys and values of its last 5 positions alone: position 23 is too far back after 40\.$/);
    steps[1](1, 24);
    steps[1](1, 0);
    steps[1](1, 1);
    // and the layers without a window keep every position, in the blocks, by their index among those
    cacheSaid.length = 0;
    steps[2](1, 3);
    assert.deepEqual([cacheSaid[0][1], cacheSaid[1][1]], [cache.keys + (1 * 16 + 3) * KV, cache.values + (1 * 16 + 3) * KV]);
  }
  console.log("ok: attention: its frame; a step's matmuls, biases, norms of the heads, RoPE, cache and phase in their order and at their addresses; no state to clear; a window's ring, written twice and handed over as one run");
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

  // T363.5: a model whose output matrix has its columns in llama.cpp's order of the value heads (the plan's tiled, the
  // form's): what the matrix reads is the value heads put in that order, every key head's first one, then every key
  // head's second (Hugging Face's head h * per + j at place j * key heads + h: here 0, 2, 1, 3), in an array of its
  // own in the token's frame; a model without it reads xb as it is, and has no such array
  const tiledSizes = { ...sizes, numbers: { linear: numbers, tiled: true } };
  assert.deepEqual(linear.frame(tiledSizes).arrays.at(-1), ["tiled", 128], "the array of what the output matrix reads");
  assert.equal(linear.frame({ ...sizes, numbers: { linear: numbers, tiled: null } }).arrays.length, 4);
  const tiled = coreOf({ ...plan, tiled: true }, { sizes: tiledSizes, has: ["wqkv", "wz", "wout", "wb", "wa", "conv", "dt_bias", "decay", "delta_norm"] });
  const step = linear.make(tiled.core, 1).layer(0, { kind: "linear", place: 0, rope: false });
  for (let t = 0; t < 2; t++) for (let i = 0; i < 32; i++) tiled.room.F[(tiled.at.xb + t * S) / 4 + i] = 100 * t + i;  // (value i of token t: head i >> 3)
  step(2, 0);
  assert.deepEqual(tiled.said.at(-1), ["matmuls", tiled.at.tiled, 2, ["wout", tiled.at.xb2, S, 0]], "the output matrix reads the array");
  for (let t = 0; t < 2; t++) {
    const read = [...tiled.room.F.subarray((tiled.at.tiled + t * S) / 4, (tiled.at.tiled + t * S) / 4 + 32)];
    assert.deepEqual(read, [0, 2, 1, 3].flatMap((head) => Array.from({ length: 8 }, (_, i) => 100 * t + 8 * head + i)), `token ${t}: the value heads in llama.cpp's order`);
  }
  console.log("ok: linear attention, tiled: the output matrix reads the value heads in llama.cpp's order, from an array of their own");
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
  const plan = { dim: 16, hidden_dim: 48, activation: "silu" };
  const gated = coreOf(plan, { has: ["w1", "w2", "w3"] });
  gatedFfn(gated.core)(3, 2);
  const at = gated.at;
  assert.deepEqual(gated.said, [["matmuls", at.xb, 2, ["w1", at.hb, S, 3], ["w3", at.hb2, S, 3]],
    ["swiglu", at.hb, at.hb, at.hb2, 48], ["swiglu", at.hb + S, at.hb + S, at.hb2 + S, 48],
    ["matmuls", at.hb, 2, ["w2", at.xb2, S, 3]], ["add_inplace", at.x, at.xb2, 16], ["add_inplace", at.x + S, at.xb2 + S, 16]]);
  // T261: a Gemma's: the gate's activation is a GELU's, by its own kernel, and what the FFN gives is normalized (the
  // layer's weights of the norm behind it) before it joins the stream
  const gemma = coreOf({ ...plan, activation: "gelu" }, { has: ["w1", "w2", "w3", "rms_ffn_post_weight"] });
  gatedFfn(gemma.core)(3, 2);
  assert.deepEqual(gemma.said, [["matmuls", at.xb, 2, ["w1", at.hb, S, 3], ["w3", at.hb2, S, 3]],
    ["geglu", at.hb, at.hb, at.hb2, 48], ["geglu", at.hb + S, at.hb + S, at.hb2 + S, 48],
    ["matmuls", at.hb, 2, ["w2", at.xb2, S, 3]],
    ["rmsnorm", at.xb2, at.xb2, gemma.vector("rms_ffn_post_weight") + 3 * 16 * 4, 16, 1e-5], ["add_inplace", at.x, at.xb2, 16],
    ["rmsnorm", at.xb2 + S, at.xb2 + S, gemma.vector("rms_ffn_post_weight") + 3 * 16 * 4, 16, 1e-5], ["add_inplace", at.x + S, at.xb2 + S, 16]]);
  // (the two apart: a GELU with no norm behind, a SiLU with one; and an activation the engine has no kernel for, or
  // none at all, is refused as the FFN is made)
  const gelu = coreOf({ ...plan, activation: "gelu" }, { has: ["w1", "w2", "w3"] });
  gatedFfn(gelu.core)(0, 1);
  assert.deepEqual(names(gelu.said), ["matmuls", "geglu", "matmuls", "add_inplace"]);
  const normed = coreOf(plan, { has: ["w1", "w2", "w3", "rms_ffn_post_weight"] });
  gatedFfn(normed.core)(0, 1);
  assert.deepEqual(names(normed.said), ["matmuls", "swiglu", "matmuls", "rmsnorm", "add_inplace"]);
  for (const activation of ["relu", undefined, "toString"]) {
    assert.throws(() => gatedFfn(coreOf({ ...plan, activation }, { has: ["w1", "w2", "w3"] }).core), /^Error: This engine has no gated FFN whose activation is /);
  }
  assert.throws(() => plainFfn(coreOf(plan, { has: ["w1", "w2", "b1", "b2"] }).core), /^Error: This engine has no FFN without a gate whose activation is "silu"\.$/);
  const plain = coreOf({ ...plan, activation: "gelu" }, { has: ["w1", "w2", "b1", "b2"] });
  plainFfn(plain.core)(3, 1);
  assert.deepEqual(plain.said, [["matmuls", at.xb, 1, ["w1", at.hb, S, 3]], ["gelu", at.hb, at.hb, plain.vector("b1") + 3 * 48 * 4, 48],
    ["matmuls", at.hb, 1, ["w2", at.xb2, S, 3]], ["add_inplace", at.x, at.xb2, 16], ["add_inplace", at.x, plain.vector("b2") + 3 * 16 * 4, 16]]);
  console.log("ok: the FFNs: a gated one (two matrices, swiglu or a Gemma's geglu, one more, the norm behind where the model has one, into the stream) and GPT-2's (a matrix, gelu with its bias, one more, into the stream with its bias)");
}

// ---- the table
assert.deepEqual(Object.keys(LAYER_KINDS), ["attention", "linear", "convolution"], "the kinds, in the order they are made in (what they take of the memory is in that order)");
assert.deepEqual([LAYER_KINDS.attention, LAYER_KINDS.linear, LAYER_KINDS.convolution], [attention, linear, convolution]);
for (const [name, kind] of Object.entries(LAYER_KINDS)) assert.deepEqual([typeof kind.frame, typeof kind.make], ["function", "function"], name);
console.log("layer-parts-check: ok");
