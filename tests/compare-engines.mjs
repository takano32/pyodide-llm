// The committed engine (git HEAD: llama2_numpy.py and forward.js) and the working copy, alternately in one Pyodide:
// 64 greedy tokens each, tok/s. Runs vary by 5-10% here, so old and new must be measured in the same process,
// turn by turn (AGENTS.md). Both run the forward pass in JavaScript, as the page does (T93); the kernels are the
// built ones of public/, the same for both.
//
//   node tests/compare-engines.mjs [model id of src/models.js | <out> of tests/perplexity_prepare.py] [rounds = 3]
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { loadPyodide } from "pyodide";

const [model, rounds = "3"] = process.argv.slice(2);
const root = new URL("../", import.meta.url).pathname;
const { MODELS } = await import(root + "src/models.js");
const entry = MODELS.find((m) => m.id === model) ?? { id: model, checkpoint: path.resolve(model + ".bin"), tokenizer: path.resolve(model + ".tokenizer.bin"),
  options: JSON.parse(fs.readFileSync(model + ".json", "utf8")), prompt: "Once upon a time" };
const file = (f) => (path.isAbsolute(f) ? f : root + f);
const head = (name) => execFileSync("git", ["show", `HEAD:${name}`], { cwd: root });
// forward.js of HEAD and of the working copy, as modules (a data: URL needs no file)
const moduleOf = (source) => import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const forwards = { old: await moduleOf(head("public/forward.js")), new: await moduleOf(fs.readFileSync(root + "public/forward.js")) };

const py = await loadPyodide();
await py.loadPackage("numpy", { messageCallback: () => {} });
for (const f of ["public/simdkernel.so", "public/simdkernel_relaxed.wasmlib", entry.checkpoint, entry.tokenizer]) py.FS.writeFile(f.split("/").pop(), fs.readFileSync(file(f)));
py.FS.writeFile("engine_new.py", fs.readFileSync(root + "public/llama2_numpy.py"));
py.FS.writeFile("engine_old.py", head("public/llama2_numpy.py"));
const checkpoint = fs.readFileSync(file(entry.checkpoint));
for (const which of ["old", "new"]) {
  const f = forwards[which];
  const kernels = f.compileKernels(fs.readFileSync(root + "public/simdkernel_plain.wasm"), fs.readFileSync(root + "public/simdkernel_relaxed_plain.wasm"));
  py.globals.set(`outside_${which}`, () => {
    const { memory, base } = f.weightsMemory(checkpoint.length);
    new Uint8Array(memory.buffer, base, checkpoint.length).set(checkpoint);
    return f.external({ memory, base, size: checkpoint.length, kernels });
  });
}
py.globals.set("OPTIONS", py.toPy(entry.options));
py.globals.set("TK", entry.tokenizer.split("/").pop());
py.runPython(`
import importlib, time, gc
tk = open(TK, "rb").read()
def measure(name):
    mod = importlib.import_module("engine_" + name)
    llama = mod.Llama(None, tk, kernels="simdkernel.so", external=globals()["outside_" + name](), **OPTIONS)
    for _ in llama.generate("${entry.prompt}", steps=8, temperature=0): pass
    t = time.perf_counter(); n = 0
    for _ in llama.generate("${entry.prompt}", steps=64, temperature=0): n += 1
    s = n / (time.perf_counter() - t)
    llama.release() if hasattr(llama, "release") else None
    del llama; gc.collect()
    return s
`);
const out = { old: [], new: [] };
for (let i = 0; i < Number(rounds); i++) for (const which of ["old", "new"]) out[which].push(py.runPython(`measure("${which}")`));
console.log(model, JSON.stringify(out, (k, v) => (typeof v === "number" ? Math.round(v * 10) / 10 : v)));
