// Where the time of one token goes, in the forward pass of public/forward.js (the page's, since T93), measured in
// Node: every kernel call is wrapped and timed (performance.now() resolves well below a microsecond here, and a
// token makes about a hundred calls), and the calls are summed by kind. What the wrapping itself costs is the
// difference between the wrapped and the plain run, and is shown. The classifier is the difference between a
// token with logits and one without. (Until T93 this measured the Python forward, ctypes calls included: those
// numbers are in AGENTS.md and kernels/README.md. Replacing kernels by functions that do nothing, as it did then,
// leaves NaN in the activations and slows what follows: that is why this one times the calls instead.)
//
//   node tests/profile.mjs [model id ...] [--positions 64]
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS } from "../src/models.js";
import { compileKernels, createForward, weightsMemory } from "../public/forward.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const positions = args.includes("--positions") ? Number(args[args.indexOf("--positions") + 1]) : 64;
const ids = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
const kernels = compileKernels(fs.readFileSync(`${root}public/simdkernel_plain.wasm`), fs.readFileSync(`${root}public/simdkernel_relaxed_plain.wasm`));
// the kinds of kernel calls, and the kernel names that go into each
const KINDS = { "matmuls": ["matmul_q8r", "matmul_q8", "matmul_f32"], "quantizing the input": ["quantize_x"],
  "attention": ["attention"], "norms": ["rmsnorm", "layernorm"], "rope, activation, adds": ["rope", "swiglu", "gelu", "add_inplace", "add_columns"] };
const kindOf = Object.fromEntries(Object.entries(KINDS).flatMap(([kind, names]) => names.map((name) => [name, kind])));
function timing(sums) {
  return (exports) => Object.fromEntries(Object.entries(exports).map(([name, f]) => [name, typeof f !== "function" || !kindOf[name] ? f : (...a) => {
    const t = performance.now();
    const result = f(...a);
    sums[kindOf[name]] += performance.now() - t;
    return result;
  }]));
}

const { pyodide: py } = await pyodideWithEngine();
console.log(`| model | ms per token | classifier | ${Object.keys(KINDS).join(" | ")} | JS between the calls | timing's own cost |\n|---|---:|---:|${"---:|".repeat(Object.keys(KINDS).length)}---:|---:|`);
for (const id of ids.length ? ids : ["llm-jp-3-150m", "tiny-lm", "stories15M"]) {
  const entry = MODELS.find((m) => m.id === id);
  const checkpoint = fs.readFileSync(root + entry.checkpoint);
  const { memory, base } = weightsMemory(checkpoint.length);
  new Uint8Array(memory.buffer).set(checkpoint, base);
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
  let plan;
  const outside = { size: checkpoint.length, read: (o, l) => new Uint8Array(memory.buffer, base + o, l).slice(),
    start: (p) => { plan = p.toJs({ dict_converter: Object.fromEntries }); return { backend: "", bind() {}, forward() {}, release() {} }; } };
  py.globals.set("OUTSIDE", outside);
  py.globals.set("OPTIONS", py.toPy(entry.options));
  py.runPython(`from llama2_numpy import Llama\nLlama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)`);
  // one engine per run, fresh; the KV cache of positions 0..15 warms up, 16.. is timed
  const run = (wrap, needLogits = true, timedFromHere = () => {}) => {
    const engine = createForward({ memory, base, size: checkpoint.length, kernels, plan, wrap });
    for (let pos = 0; pos < 16; pos++) engine.forward(1, pos, needLogits);
    timedFromHere();  // what the warm-up added to the sums goes
    const began = performance.now();
    for (let pos = 16; pos < 16 + positions; pos++) engine.forward(1, pos, needLogits);
    return (performance.now() - began) / positions;
  };
  const best = (f) => Math.min(f(), f(), f());
  const plain = best(() => run(undefined)), withoutLogits = best(() => run(undefined, false));
  let sums, timed = Infinity;
  for (let r = 0; r < 3; r++) {
    const these = Object.fromEntries(Object.keys(KINDS).map((kind) => [kind, 0]));
    const t = run(timing(these), true, () => Object.keys(these).forEach((kind) => { these[kind] = 0; }));
    if (t < timed) [timed, sums] = [t, these];
  }
  const per = Object.fromEntries(Object.entries(sums).map(([kind, ms]) => [kind, ms / positions]));
  const inKernels = Object.values(per).reduce((a, b) => a + b, 0);
  const share = (ms) => `${ms.toFixed(2)} (${((ms / plain) * 100).toFixed(0)}%)`;
  console.log(`| ${entry.name} | ${plain.toFixed(2)} | ${share(plain - withoutLogits)} | ${Object.keys(KINDS).map((kind) => share(per[kind])).join(" | ")} | ` +
    `${share(Math.max(0, plain - inKernels))} | ${(timed - plain).toFixed(2)} |`);
}
