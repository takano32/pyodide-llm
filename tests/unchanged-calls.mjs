// unchanged-calls.mjs (T346)
// What forward.js asks of its kernels, as one hash a case: every call (the kernel's name and its arguments, the
// addresses with them) of an engine made with kernels that do nothing, for made-up models of every kind, dtype, memory
// and type of the keys and values, through single tokens and a prompt's blocks. A refactoring of forward.js, jobs.js or
// of Python's plan (llama2_numpy's Places and tensor order) that moves nothing leaves every hash as it was; and so does
// the plan itself, which is hashed too. tests/unchanged.mjs runs this on the working tree and on another commit's.
//   node tests/unchanged-calls.mjs <the root of a tree>        (PYTHON=.venv/bin/python)        -> one JSON object
// One thread: a second would take its chunks in an order of its own.
import crypto from "node:crypto";
import path from "node:path";
import { plans } from "./plans.mjs";
import { treeOf } from "./tree.mjs";

const root = path.resolve(process.argv[2] ?? ".");
const tree = treeOf(root);
const { createForward, footprint, keysInHalf } = await import(tree.runtimeUrl("forward.js"));
const { CONTROL_BYTES } = await import(tree.runtimeUrl("jobs.js"));
const { plansOf, planOf, FORM, empty } = plans(root);
const PAGE = 65536, MiB = 2 ** 20;
const LINEAR = { every: 4, key_heads: 8, value_heads: 16, key_dim: 128, value_dim: 128, conv: 4 };
const CONVOLUTION = { layers: "ccaccaca", taps: 4 };
const ROTATED = { block: 64, signs: { 256: "", 512: "" } };
const of = (name, header, form, dtypes) => dtypes.map((dtype) => ({ name, header, form: { ...FORM, ...form }, dtype }));
const shapes = [
  ...of("llama", [256, 512, 4, 8, 8, -2000, 300], {}, ["int8", "int6", "float32", "float16"]),
  ...of("llama, grouped-query", [256, 512, 4, 8, 2, 2000, 300], {}, ["int8", "float32"]),
  ...of("qwen2", [256, 512, 4, 8, 2, 2000, 300], { bias: true }, ["int8", "float32"]),
  ...of("qwen3", [256, 512, 4, 8, 4, 2000, 300], { qk_norm: true, head_dim: 64 }, ["int8", "int6", "float32", "ternary"]),
  ...of("ternary, a classifier of its own", [256, 768, 4, 8, 8, -2000, 300], {}, ["ternary"]),
  ...of("gpt2", [256, 1024, 3, 8, 8, 2000, 300], { arch: "gpt2" }, ["int8", "int6", "float32"]),
  ...of("neox", [256, 1024, 3, 8, 8, -2000, 300], { arch: "neox" }, ["int8", "float32"]),
  ...of("int8 kernels cannot run", [200, 400, 3, 4, 4, 1000, 300], {}, ["int8"]),
  ...of("qwen3.5", [256, 512, 8, 4, 2, 2000, 300], { arch: "qwen35", head_dim: 64, linear: LINEAR }, ["int8", "int6", "float32", "float16"]),
  ...of("qwen3.5, rotated", [256, 512, 8, 4, 2, 2000, 300], { arch: "qwen35", head_dim: 64, linear: LINEAR, rotated: ROTATED }, ["int8", "ternary"]),
  ...of("llama, rotated", [256, 512, 4, 8, 2, 2000, 300], { rotated: ROTATED }, ["int8", "ternary"]),
  ...of("lfm2", [256, 512, 8, 8, 2, -2000, 300], { arch: "lfm2", convolution: CONVOLUTION }, ["int8", "int6", "float32"]),
];
const sha = (text) => crypto.createHash("sha256").update(text).digest("hex").slice(0, 16);
const found = {};
for (const p of plansOf(shapes.map(({ name, ...shape }) => shape))) {
  const quantized = ["int8", "int6", "ternary"].includes(p.dtype);
  found[`plan: ${JSON.stringify([p.header, p.form, p.dtype])}`] = sha(JSON.stringify(p));
  for (const relaxed of [true, false]) for (const shared of [true, false]) {
    const options = { ...p.form, dtype: p.dtype, int8: true, relaxed, halfKV: quantized, outliers: 8, gpu: false, shared };
    const base = shared ? CONTROL_BYTES : 64, bound = footprint(p.header, p.size, options), halfKeys = keysInHalf(p.header, p.size, options);
    const memory = shared
      ? new WebAssembly.Memory({ initial: Math.ceil((base + p.size) / PAGE) + 1, maximum: Math.ceil((base + p.size + bound + 2 * MiB) / PAGE) + 2, shared: true })
      : new WebAssembly.Memory({ initial: Math.ceil((base + p.size) / PAGE) + 1 });
    const hash = crypto.createHash("sha256");
    let calls = 0;
    const kernels = () => new Proxy({}, { get: (_, name) => (...args) => { calls++; hash.update(`${String(name)}(${args.join(",")});`); return 0; } });
    const plan = planOf(p, { relaxed, kvStart: 16, outliers: 8 });
    const engine = createForward({ memory, base, size: p.size, kernels: { plain: empty, relaxed: relaxed ? empty : null, wide: false }, plan, halfKeys, wrap: kernels });
    const mark = (what) => hash.update(`|${what}|`);
    let pos = 0;
    // a prompt in blocks (the last needs logits, as generate() asks), then tokens one by one through two doublings
    // of the cache; then from position 0 again (a model with a state starts over there)
    for (const count of [16, 5, 1]) { mark(`block ${count} at ${pos}`); engine.forwardMany(Array.from({ length: count }, (_, i) => 3 + i), pos); pos += count; }
    for (; pos < 70; pos++) { mark(`token at ${pos}`); engine.forward(7, pos, pos % 3 !== 0); }
    for (pos = 0; pos < 3; pos++) { mark(`again at ${pos}`); engine.forward(9, pos, true); }
    mark(`footprint ${bound}, keys in half ${halfKeys}, the memory ${memory.buffer.byteLength}`);
    engine.release();
    found[`calls: ${JSON.stringify([p.header, p.form, p.dtype])}${relaxed ? "" : ", no relaxed SIMD"}, ${shared ? "shared" : "plain"}`] = `${hash.digest("hex").slice(0, 16)} (${calls} calls)`;
  }
}
console.log(JSON.stringify(found));
process.exit(0);
