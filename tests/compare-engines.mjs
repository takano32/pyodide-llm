// The committed engine (git HEAD) and the working copy, alternately in one Pyodide: 64 greedy tokens each, tok/s.
// Runs vary by 5-10% here, so old and new must be measured in the same process, turn by turn (AGENTS.md).
//
//   node tests/compare-engines.mjs [model id of src/models.js | <out> of tests/perplexity_prepare.py] [rounds = 3]
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { loadPyodide } from "pyodide";
const [model, rounds = "3"] = process.argv.slice(2);
const root = "/home/takano32/GitHub/pyodide-llm/";
const { MODELS } = await import(root + "src/models.js");
import path from "node:path";
const entry = MODELS.find((m) => m.id === model) ?? { id: model, checkpoint: path.resolve(model + ".bin"), tokenizer: path.resolve(model + ".tokenizer.bin"), options: JSON.parse(fs.readFileSync(model + ".json", "utf8")), prompt: "Once upon a time" };
const file = (f) => (path.isAbsolute(f) ? f : root + f);
const py = await loadPyodide();
await py.loadPackage("numpy", { messageCallback: () => {} });
for (const f of ["public/simdkernel.so", "public/simdkernel_relaxed.wasmlib", entry.checkpoint, entry.tokenizer]) py.FS.writeFile(f.split("/").pop(), fs.readFileSync(file(f)));
py.FS.writeFile("engine_new.py", fs.readFileSync(root + "public/llama2_numpy.py"));
py.FS.writeFile("engine_old.py", execFileSync("git", ["show", "HEAD:public/llama2_numpy.py"], { cwd: root }));
py.globals.set("OPTIONS", py.toPy(entry.options));
py.globals.set("CK", entry.checkpoint.split("/").pop()); py.globals.set("TK", entry.tokenizer.split("/").pop());
py.runPython(`
import importlib, time, gc
ck, tk = open(CK, "rb").read(), open(TK, "rb").read()
def measure(name):
    mod = importlib.import_module(name)
    llama = mod.Llama(ck, tk, kernels="simdkernel.so", **OPTIONS)
    for _ in llama.generate("${entry.prompt}", steps=8, temperature=0): pass
    t = time.perf_counter(); n = 0
    for _ in llama.generate("${entry.prompt}", steps=64, temperature=0): n += 1
    s = n / (time.perf_counter() - t)
    del llama; gc.collect()
    return s
`);
const out = { old: [], new: [] };
for (let i = 0; i < Number(rounds); i++) for (const which of ["old", "new"]) out[which].push(py.runPython(`measure("engine_${which}")`));
console.log(model, JSON.stringify(out, (k, v) => (typeof v === "number" ? Math.round(v * 10) / 10 : v)));
