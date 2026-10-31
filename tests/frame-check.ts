// T.622: a token's frame, array by array, as createForward() lays it out from the plan and as footprint() counts it from
// the header and the form (forward/memory.ts's frameOf(), which footprint() calls). The two make the packings whose
// activations' scales the frame holds (xs) each its own way (the engine from the plan's tensors, footprint() from the
// form's dtype and kinds), and memory-check.ts, in the full suite, holds what footprint() counts after the checkpoint only
// to a megabyte. Here, in the light suite, to the byte: every architecture and kind of layer of memory-check.ts's
// shapes (made small: the frame does not grow with the context or the vocabulary), in each dtype a file of it may be,
// with and without relaxed SIMD. The engine is forward.js's own, on kernels that do nothing (memory-check.ts's).
//   node tests/frame-check.ts [--forward <another forward.js, to see a broken one fail>]     (PYTHON=.venv/bin/python)
import assert from "node:assert/strict";
import path from "node:path";
import { plans } from "./plans.ts";
import { runtime, runtimeUrl } from "./tree.ts";

const root = new URL("..", import.meta.url);
const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : runtime("forward.js");
const { createForward } = await import(forwardFile);
const { frameOf } = await import(runtimeUrl("forward/memory.ts"));
const PAGE = 65536;

const { plansOf, planOf, FORM, empty, nothing } = plans(root);
// (memory-check.ts's shapes of the kinds of layers)
const CONVOLUTION_SMALL = { layers: "ccaccaca", taps: 4 };
const LINEAR_SMALL = { every: 4, key_heads: 8, value_heads: 16, key_dim: 128, value_dim: 128, conv: 4 };
const ROTATED_SMALL = { block: 64, signs: { 256: "", 512: "" } };

const ALL = ["int8", "int6", "float32", "float16"];
const shape = (name, header, form = {}, dtypes = ["int8", "int6"]) => dtypes.map((dtype) => ({ name, header, form: { ...FORM, ...form }, dtype }));
const shapes = [
  ...shape("llama, a key for every head, a classifier of its own", [256, 512, 4, 8, 8, -500, 64], {}, ALL),
  ...shape("llama, grouped-query", [256, 512, 4, 8, 2, 500, 64]),
  ...shape("qwen2, biases", [256, 512, 4, 8, 2, 500, 64], { bias: true }),
  ...shape("qwen3, heads of 64", [256, 512, 4, 8, 4, 500, 64], { qk_norm: true, head_dim: 64 }),
  ...shape("ternary qwen3", [256, 512, 4, 8, 4, 500, 64], { qk_norm: true, head_dim: 64 }, ["ternary"]),
  ...shape("gpt2", [256, 1024, 4, 8, 8, 500, 64], { arch: "gpt2" }, ["int8", "int6", "float32"]),
  ...shape("gpt-neox", [256, 1024, 4, 8, 8, -500, 64], { arch: "neox" }),
  // (rows that are no whole groups of 32: the int8 kernels do not run)
  ...shape("int8 kernels cannot run (dim 200)", [200, 400, 4, 4, 4, 500, 64], {}, ["int8"]),
  ...shape("qwen3.5", [256, 512, 8, 4, 2, 500, 64], { arch: "qwen35", head_dim: 64, linear: LINEAR_SMALL }, ALL),
  ...shape("qwen3.5 in a rotated basis", [256, 512, 8, 4, 2, 500, 64], { arch: "qwen35", head_dim: 64, linear: LINEAR_SMALL, rotated: ROTATED_SMALL }, ["int8", "float32"]),
  ...shape("ternary qwen3.5 in a rotated basis", [256, 512, 8, 4, 2, 500, 64], { arch: "qwen35", head_dim: 64, linear: LINEAR_SMALL, rotated: ROTATED_SMALL }, ["ternary"]),
  ...shape("llama in a rotated basis", [256, 512, 4, 8, 2, 500, 64], { rotated: ROTATED_SMALL }),
  ...shape("lfm2", [256, 512, 8, 8, 2, -500, 64], { arch: "lfm2", convolution: CONVOLUTION_SMALL }, ALL),
  ...shape("gemma3", [256, 512, 6, 4, 1, 500, 64], { arch: "gemma3", head_dim: 128 }, ALL),
  ...shape("gemma4, wide heads and a double FFN", [256, 512, 6, 4, 1, 500, 64], { arch: "gemma4", head_dim: 64, wide: { layers: [2, 5], head_dim: 128 }, double: [3, 4, 5] }, ALL),
  ...shape("gemma4, an embedding of every layer", [256, 512, 6, 4, 1, 500, 64], { arch: "gemma4", head_dim: 64, wide: { layers: [2, 5], head_dim: 128 },
    double: [4, 5], kv_shared: { layers: [4, 5], from: [3, 2] }, per_layer: 64 }, ALL),
];

let compared = 0;
plansOf(shapes.map(({ name, ...one }) => one)).forEach((p, n) => {
  for (const relaxed of [true, false]) {
    const where = `${shapes[n].name}, ${p.dtype}${relaxed ? ", relaxed SIMD" : ""}`;
    const memory = new WebAssembly.Memory({ initial: Math.ceil((64 + p.size) / PAGE) + 1 });
    const engine = createForward({ memory, base: 64, size: p.size, kernels: { plain: empty, relaxed: relaxed ? empty : null, wide: false }, plan: planOf(p, { relaxed, outliers: 8 }), wrap: nothing });
    const counted = frameOf(p.header, { ...p.form, dtype: p.dtype });
    assert.deepEqual(engine.frame, counted, `${where}: the frame the engine laid out (left) is not the frame footprint() counts (right)`);
    compared += 1;
  }
});
console.log(`frame-check: ${compared} engines lay out a token's frame array by array as footprint() counts it (${shapes.length} shapes and dtypes, with and without relaxed SIMD)`);
