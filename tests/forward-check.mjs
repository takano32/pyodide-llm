// T93 stage 1: the forward pass of public/forward.js against the engine's own, on the models of this directory
// (make models kernels). The same kernels in the same order: the logits must be the same to the bit, for every
// position of a greedy run. Then both, turn by turn, for speed.
//
//   node tests/forward-check.mjs [model id | <out> of tests/perplexity_prepare.py ...] [--rounds 3] [--positions 64]
//        [--without relaxed,int8,sampler]
import fs from "node:fs";
import path from "node:path";
import { loadPyodide } from "pyodide";
import { MODELS } from "../src/models.js";
import { compileKernels, external, weightsMemory } from "../public/forward.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
const rounds = option("--rounds", 3), positions = option("--positions", 64);
const ids = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
const without = args.includes("--without") ? args[args.indexOf("--without") + 1].split(",") : [];
// a model of src/models.js, or a converted Hugging Face model (<out>.bin, <out>.tokenizer.bin, <out>.json)
const modelOf = (id) => MODELS.find((m) => m.id === id) ?? { name: path.basename(id), checkpoint: path.resolve(`${id}.bin`),
  tokenizer: path.resolve(`${id}.tokenizer.bin`), options: JSON.parse(fs.readFileSync(`${id}.json`, "utf8")) };
const file = (f) => (path.isAbsolute(f) ? f : root + f);
const kernels = compileKernels(fs.readFileSync(`${root}public/simdkernel_plain.wasm`), fs.readFileSync(`${root}public/simdkernel_relaxed_plain.wasm`));

const py = await loadPyodide();
await py.loadPackage("numpy", { messageCallback: () => {} });
for (const f of ["llama2_numpy.py", "simdkernel.so", "simdkernel_relaxed.wasmlib"]) py.FS.writeFile(f, fs.readFileSync(`${root}public/${f}`));
py.runPython("import time, gc, numpy as np\nfrom llama2_numpy import Llama");
let failed = false;
for (const id of ids.length ? ids : ["stories260K", "stories15M", "tiny-lm", "llm-jp-3-150m"]) {
  const entry = modelOf(id);
  const checkpoint = fs.readFileSync(file(entry.checkpoint));
  const { memory, base } = weightsMemory(checkpoint.length);
  new Uint8Array(memory.buffer).set(checkpoint, base);
  py.FS.writeFile("model.bin", checkpoint);
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(file(entry.tokenizer)));
  py.globals.set("OPTIONS", py.toPy({ ...entry.options, disable: without }));
  py.globals.set("OUTSIDE", external({ memory, base, size: checkpoint.length, kernels }));
  py.runPython(`
old = Llama(open("model.bin", "rb").read(), open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", **OPTIONS)
new = Llama(None, open("tokenizer.bin", "rb").read(), kernels="simdkernel.so", external=OUTSIDE, **OPTIONS)
def run(llama, positions, keep=False):
    token, seen, began = llama.bos, [], time.perf_counter()
    for pos in range(positions):
        logits = llama.forward(token, pos)
        token = int(np.argmax(logits))
        if keep:
            seen.append(logits.copy())
    return seen, time.perf_counter() - began
a, _ = run(old, ${positions}, True)
b, _ = run(new, ${positions}, True)
same = all(np.array_equal(x, y) for x, y in zip(a, b))
largest = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
`);
  const same = py.globals.get("same"), largest = py.globals.get("largest");
  const backends = `${py.runPython("old.backend")} / ${py.runPython("new.backend")}`;
  const times = { old: [], new: [] };
  for (let r = 0; r < rounds; r++) {
    for (const which of ["old", "new"]) times[which].push(positions / py.runPython(`run(${which}, ${positions})[1]`));
  }
  const median = (xs) => [...xs].sort((p, q) => p - q)[xs.length >> 1];
  console.log(`${entry.name}: ${backends}; logits ${same ? "the same to the bit" : `DIFFER (largest ${largest})`} over ${positions} positions; ` +
    `Python ${median(times.old).toFixed(1)} (${times.old.map((v) => v.toFixed(0)).join(", ")}) against JS ${median(times.new).toFixed(1)} tok/s ` +
    `(${times.new.map((v) => v.toFixed(0)).join(", ")}) = ${(median(times.new) / median(times.old)).toFixed(2)}×`);
  failed ||= !same;
  py.runPython("del old, new; gc.collect()");
  py.globals.delete("OUTSIDE");
  py.FS.unlink("model.bin");
}
process.exit(failed ? 1 : 0);
