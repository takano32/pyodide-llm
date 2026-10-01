// T225's review: what /benchmark/'s layer check says (public/benchmark/gpu.js's judgeLayer) of a device that is JavaScript
// too, in Node, without a GPU:   node tests/layer-check.mjs
// The check's own data (layerCheckData: 33 heads of 64 and 3 of K and V at 2112 wide, 71 positions), the layer run in
// float64 by the check's own reference (layerReference) as a device would leave it, its keys and values rounded to
// float16 the way a device may choose: WGSL leaves to it which of the two neighbours a conversion gives (§15.7.6) and
// Direct3D takes the one toward zero (D3D11.3 3.2.2: the owner's NVIDIA PC on Windows did), where lavapipe and
// SwiftShader, the only devices CI has, round to the nearest. So CI cannot see a layer check that holds a device that
// rounds toward zero wrong again (T225's), nor the keys and values the reference takes from the device where they are
// a neighbour of its own: this does. The same numbers show what the check still catches (the lines of the check:
// LAYER_LINE 1e-3 of what the layer added, CACHE_LINE 2e-3 of the largest, the quantized scales 1e-3 and 3e-5).
//   The numbers of T225's review (300 draws of the check's data, a device that rounds toward zero, the check before it
// took the device's bits): the stream of a float form 3.4e-4 to 2.5e-3 (45% past its line), the scale of the DP4A
// form's quantized attention output 5.7e-4 to 1.2e-2 (97% past its line), the cache 4.5e-4 to 9.7e-4.
import assert from "node:assert/strict";

globalThis.onmessage = null;  // (the worker's file sets it as a module's plain assignment)
const { fromHalf, judgeLayer, layerCheckData, layerReference, load, toHalf } = await import("../public/benchmark/gpu.js");
await load();
const { GROUP, quantizedLikeCpu } = await import("../public/shaders.js");

// a seeded Math.random (mulberry32): the check's data is random, and this holds it to a number it was run on
const random = Math.random;
function seeded(seed) {
  let a = seed >>> 0;
  Math.random = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
seeded(20261001);
const draw = layerCheckData();
Math.random = random;
const { shape, pos, data } = draw;

// float16 bits of x as a device may round it: "nearest" (ties to even), "zero" (toward zero), "away" (away from it);
// "even" and "odd": toward zero in one lane of each pair and to the nearest in the other (what the first throwaway
// wrapper in CI did, 1 in 4 of all); "twice": toward zero and one float16 lower still (an error)
function rounded(x, how, i) {
  const n = toHalf(x), v = fromHalf(n), down = Math.abs(v) > Math.abs(x) ? n - 1 : n;
  switch (how) {
    case "nearest": return n;
    case "zero": return down;
    case "away": return Math.abs(v) < Math.abs(x) ? n + 1 : n;
    case "even": return i % 2 === 0 ? down : n;
    case "twice": return (down & 0x7fff) > 0 ? down - 1 : down;
  }
  throw new Error(how);
}
const POINTS = ["qkv", "o", "ffn", "down"];
const dequantized = ({ xq, xs }) => Float64Array.from(xq, (v, i) => v * xs[(i / GROUP) | 0]);

// what a GPU would leave of the layer: the stream h, the whole cache (the position's row the device's), on DP4A the
// vectors each matrix took (quantized from the device's own values), and the stages' q, att and g
function leave(form, { keys = "nearest", values = "nearest", keyScale = 1, valueScale = 1 } = {}) {
  const quantized = [];
  const inputs = form.dp4a ? (point, x) => {
    const q = quantizedLikeCpu(Float32Array.from(x));
    quantized[POINTS.indexOf(point)] = q;
    return dequantized(q);
  } : undefined;
  const out = layerReference(shape, pos, data, inputs, (which, x) => {
    const how = which === "keys" ? keys : values, scale = which === "keys" ? keyScale : valueScale;
    return Uint16Array.from(x, (v, i) => rounded(v * scale, how, i));
  });
  const cache = (own, row) => {
    const all = Uint16Array.from(own);
    all.set(row, pos * shape.kvDim);
    return all;
  };
  return { h: Float32Array.from(out.h), keys: cache(data.keys, out.keys), values: cache(data.values, out.values), quantized,
    q: Float32Array.from(out.stages.q), att: Float32Array.from(out.stages.att), g: Float32Array.from(out.stages.g) };
}
const FLOAT = { name: "llama.cpp, fused", dp4a: false, fused: true }, DP4A = { name: "DP4A, fused", dp4a: true, fused: true };
const judged = (form, got, normsApart = {}) => judgeLayer(shape, pos, data, form, got, normsApart);
// how the stages' line counts the keys and values
const counted = (verdict) => {
  const m = /K and V (\d+) to the nearest float16(?:, (\d+) toward zero, (\d+) away from it, (\d+) farther)?/.exec(verdict.stages);
  assert.ok(m, verdict.stages);
  return { same: +m[1], inward: +(m[2] ?? 0), outward: +(m[3] ?? 0), far: +(m[4] ?? 0) };
};

for (const form of [FLOAT, DP4A]) {
  const label = (what) => `${form.name}, ${what}`;
  // to the nearest: right to a float32 sum's difference, every key and value the nearest
  {
    const v = judged(form, leave(form));
    assert.ok(v.ok, label(`to the nearest: ${v.stages}`));
    assert.ok(v.stream < 1e-5 && v.cache === 0, label("the stream and the cache"));
    assert.deepEqual(counted(v), { same: 2 * shape.kvDim, inward: 0, outward: 0, far: 0 });
  }
  // toward zero (Direct3D's, T225): right, and the line says how it rounded: about half of the values toward zero, none away
  // from it or farther. (Before T225 the reference rounded to the nearest: the DP4A form's attention output was quantized
  // to a scale 5.7e-4 to 1.2e-2 off, and the stream of a float form 3.4e-4 to 2.5e-3: past their lines in 97% and 45% of draws)
  {
    const v = judged(form, leave(form, { keys: "zero", values: "zero" }));
    assert.ok(v.ok, label(`toward zero: ${v.stages}`));
    const c = counted(v);
    assert.ok(c.inward > 0.35 * 2 * shape.kvDim && c.inward < 0.65 * 2 * shape.kvDim && c.outward === 0 && c.far === 0, label(`counts ${JSON.stringify(c)}`));
    assert.ok(v.cache > 1e-4 && v.cache < 1e-3, label(`the cache, 1 ulp of the largest at the most: ${v.cache}`));
    // the same device, the keys alone or the values alone (the keys' shift of the scores is most of what the old check saw)
    for (const [which, modes] of [["keys", { keys: "zero" }], ["values", { values: "zero" }]]) assert.ok(judged(form, leave(form, modes)).ok, label(`${which} toward zero`));
  }
  // away from zero, and toward zero in one lane of each pair (what CI's first wrapper made of a device)
  {
    const v = judged(form, leave(form, { keys: "away", values: "away" }));
    assert.ok(v.ok && counted(v).inward === 0 && counted(v).outward > 0, label(`away from zero: ${v.stages}`));
  }
  assert.ok(judged(form, leave(form, { keys: "even", values: "even" })).ok, label("toward zero in the even lanes"));
  // errors the check has caught since T150: the keys and the values larger by 0.4% (4 float16 ulp at the most), one
  // float16 lower than a device that rounds toward zero (1 to 2 ulp off the value)
  for (const [what, modes, first] of [["the keys 0.4% larger", { keyScale: 1.004 }, "K and V"], ["the values 0.4% larger", { valueScale: 1.004 }, "K and V"],
    ["toward zero and one float16 lower still", { keys: "twice", values: "twice" }, "K and V"]]) {
    const v = judged(form, leave(form, modes));
    assert.ok(!v.ok, label(`${what} is not right: ${v.stages}`));
    assert.match(v.stages, new RegExp(`first to depart: ${first}`), label(what));
  }
}

// a float form's stream off by 1% (the layer's output wrong somewhere after the cache), the other positions of the
// cache written to, the DP4A form's quantized attention output 0.2% off in its scales
{
  const got = leave(FLOAT);
  got.h = got.h.map((v) => v * 1.01);
  const v = judged(FLOAT, got);
  assert.ok(!v.ok && v.stream > 1e-3, `the stream: ${v.stages}`);
  assert.match(v.stages, /first to depart: the stream/);
  const written = leave(FLOAT);
  written.keys[(pos - 1) * shape.kvDim + 3] ^= 1;
  const touched = judged(FLOAT, written);
  assert.ok(!touched.ok && touched.wroteOtherPositions, "another position's key written");
  const scaled = leave(DP4A);
  scaled.quantized[1].xs = scaled.quantized[1].xs.map((s) => s * 1.002);
  const w = judged(DP4A, scaled);
  assert.ok(!w.ok && w.quantized[1].wrong, `o's quantized scales 0.2% off: ${w.stages}`);
  assert.match(w.stages, /first to depart: o's quantizing/);
}

// the DP4A fused form against the one with the norms apart: the same to the bit passes; a scale 6 ulp apart (4 are
// allowed, and 7e-7 is far under the scale lines) does not
{
  const got = leave(DP4A), apart = {};
  assert.ok(judged({ ...DP4A, normApart: true }, got, apart).ok);
  const same = judged(DP4A, got, apart);
  assert.ok(same.ok && same.sameAsNormsApart.bitForBit && same.sameAsNormsApart.ulps === 0, "the same to the bit");
  const other = leave(DP4A);
  other.quantized[0].xs[1] *= 1 + 6 * 2 ** -23;  // (group 0 of the normed stream is all zeros: its scale is 0)
  const off = judged(DP4A, other, apart);
  assert.ok(!off.ok && off.sameAsNormsApart.ulps >= 5 && !off.sameAsNormsApart.ok, `6 ulp apart: ${JSON.stringify(off.sameAsNormsApart)}`);
}
console.log("ok");
