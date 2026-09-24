// T93, part 1: how much faster is one token when the int8 matmuls are split over N threads that share the
// weights (SharedArrayBuffer), against the engine as the page runs it (Pyodide + simdkernel.so, one thread)?
//
//   node tests/threads-prototype/run.mjs <model id> [threads ...] [--rounds 3] [--positions 64]
//
// The weights are the engine's own arrays, copied into one shared WebAssembly memory, so the layout cannot
// differ from llama2_numpy.py. The kernels are the same AssemblyScript, built with a shared memory
// (kernels/build.py writes simdkernel_shared.wasm and simdkernel_relaxed_shared.wasm). The prototype must write
// the very tokens the engine writes (greedy): if it does not, it is wrong, and its speed means nothing.
import fs from "node:fs";
import path from "node:path";
import { Worker } from "node:worker_threads";
import { loadPyodide } from "pyodide";
import { MODELS } from "../../src/models.js";

const root = new URL("../../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
const [modelId] = args;
const counts = args.slice(1).filter((a, i, all) => /^\d+$/.test(a) && !/^--/.test(all[i - 1] ?? "")).map(Number);
const rounds = option("--rounds", 3), positions = option("--positions", 64);
const entry = MODELS.find((m) => m.id === modelId);
const kernelsDir = process.env.SHARED_KERNELS ?? path.join(root, "public");

// ---- the engine, as the page runs it
const py = await loadPyodide();
await py.loadPackage("numpy", { messageCallback: () => {} });
for (const f of ["public/llama2_numpy.py", "public/simdkernel.so", "public/simdkernel_relaxed.wasmlib", entry.checkpoint, entry.tokenizer]) {
  py.FS.writeFile(path.basename(f), fs.readFileSync(root + f));
}
py.globals.set("OPTIONS", py.toPy(entry.options));
py.globals.set("PROMPT", entry.prompt);
py.runPython(`
import time, numpy as np
from llama2_numpy import Llama
llama = Llama(open("${path.basename(entry.checkpoint)}", "rb").read(), open("${path.basename(entry.tokenizer)}", "rb").read(), kernels="simdkernel.so", **OPTIONS)
prompt = llama.tokenizer.encode(PROMPT, llama.specials)
def engine(positions):
    tokens, token = [], llama.bos
    began = time.perf_counter()
    for pos in range(positions):
        logits = llama.forward(token, pos)
        nxt = int(np.argmax(logits))
        token = prompt[pos] if pos < len(prompt) else nxt
        if pos >= len(prompt):
            tokens.append(nxt)
    return tokens, time.perf_counter() - began
def arrays():
    out = {}
    def matrix(name, t):
        values, scales = t
        out[name + ".values"] = np.ascontiguousarray(values)
        out[name + ".scales"] = np.ascontiguousarray(scales)
        out[name + ".corrections"] = np.ascontiguousarray((scales[..., 0] * values.sum(axis=-1, dtype=np.int32)).astype(np.float32))
    for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3", "wcls"):
        matrix(name, getattr(llama, name))
    out["embedding.values"], out["embedding.scales"] = (np.ascontiguousarray(a) for a in llama.token_embedding_table)
    for name, attr in (("attNorm", "rms_att_weight"), ("ffnNorm", "rms_ffn_weight"), ("finalNorm", "rms_final_weight"),
                       ("cos", "freq_cis_real"), ("sin", "freq_cis_imag")):
        out[name] = np.ascontiguousarray(getattr(llama, attr), dtype=np.float32)
    return out
`);
// the engine keeps its bytes; the file's copy in Pyodide's file system is not needed any more (memory is short here)
py.FS.unlink(path.basename(entry.checkpoint));
const cfg = py.runPython(`dict(dim=llama.dim, hidden=llama.hidden_dim, layers=llama.n_layers, heads=llama.n_heads,
  kvHeads=llama.n_kv_heads, headSize=llama.head_size, vocab=llama.vocab_size, bos=llama.bos, backend=llama.backend,
  prompt=len(prompt), relaxed="relaxed" in llama.backend)`).toJs({ dict_converter: Object.fromEntries });
if (!cfg.relaxed) throw new Error(`the engine runs ${cfg.backend}: this prototype only has the relaxed path`);
const prompt = py.runPython("prompt").toJs();

// ---- one shared memory: the control area, the weights, the activations
const capacity = positions + 1;
const tensors = py.runPython("arrays()");
let at = 4096;
const place = (bytes) => { const p = at; at += Math.ceil(bytes / 64) * 64; return p; };
const plan = {};
for (const name of tensors.keys()) plan[name] = place(tensors.get(name).nbytes);
const { dim, hidden, layers, kvHeads, headSize, vocab, heads } = cfg;
const L = { config: cfg, capacity, bos: cfg.bos,
  x: place(dim * 4), xb: place(dim * 4), xb2: place(dim * 4), q: place(dim * 4), hb: place(hidden * 4), hb2: place(hidden * 4),
  xq: place(Math.max(dim, hidden)), xs: place(Math.max(dim, hidden) / 32 * 4), att: place(capacity * heads * 4),
  logits: place(vocab * 4), keys: place(layers * capacity * kvHeads * headSize * 4), values: place(layers * capacity * kvHeads * headSize * 4) };
const pages = Math.ceil(at / 65536) + 1;
const memory = new WebAssembly.Memory({ initial: pages, maximum: pages, shared: true });
const bytes = new Uint8Array(memory.buffer);
for (const name of tensors.keys()) {
  const view = tensors.get(name).getBuffer("u8");
  bytes.set(view.data, plan[name]);
  view.release();
}
tensors.destroy();
for (const name of ["wq", "wk", "wv", "wo", "w1", "w2", "w3", "wcls", "embedding"]) {
  L[name] = { values: plan[`${name}.values`], scales: plan[`${name}.scales`], corrections: plan[`${name}.corrections`] };
}
Object.assign(L, { attNorm: plan.attNorm, ffnNorm: plan.ffnNorm, finalNorm: plan.finalNorm, cos: plan.cos, sin: plan.sin });
console.log(`${entry.name}: ${cfg.backend}; shared memory ${(memory.buffer.byteLength / 1e6).toFixed(0)} MB; prompt ${cfg.prompt} tokens, ${positions} positions`);

// ---- the prototype with N threads and a way of splitting
async function prototype(threads, mode) {
  new Int32Array(memory.buffer, 0, 1024).fill(0);
  const make = (share) => new Worker(new URL("./thread.mjs", import.meta.url), { workerData: { memory, share, threads, kernelsDir, layout: L, mode } });
  const coordinator = make(0);
  const helpers = Array.from({ length: threads - 1 }, (_, i) => make(i + 1));
  await new Promise((resolve) => coordinator.once("message", resolve));
  const run = () => new Promise((resolve) => { coordinator.once("message", resolve); coordinator.postMessage({ prompt, positions }); });
  await run();  // warm up
  const result = await run();
  const ctl = new Int32Array(memory.buffer, 0, 1024);
  Atomics.store(ctl, 1, 1); Atomics.add(ctl, 0, 1); Atomics.notify(ctl, 0);
  await Promise.all(helpers.map((h) => new Promise((resolve) => h.once("message", resolve))));
  await Promise.all([coordinator, ...helpers].map((w) => w.terminate()));
  return result;
}

const [expected] = py.runPython(`engine(${positions})`).toJs();  // warm up, and the tokens to match
const rows = {};
const note = (key, seconds) => (rows[key] ??= []).push(positions / seconds);
for (let round = 0; round < rounds; round++) {
  const [, seconds] = py.runPython(`engine(${positions})`).toJs();
  note("engine (Pyodide, 1 thread)", seconds);
  for (const threads of counts) {
    for (const mode of threads === 1 ? ["classifier"] : ["classifier", "all"]) {
      const { tokens, seconds: s } = await prototype(threads, mode);
      if (tokens.join() !== expected.join()) {
        throw new Error(`${threads} threads, ${mode}: tokens differ from the engine's\n${tokens.slice(0, 12)}\n${expected.slice(0, 12)}`);
      }
      note(`prototype, ${threads} thread${threads > 1 ? "s" : ""}${threads > 1 ? `, ${mode === "all" ? "every matmul" : "classifier only"}` : ""}`, s);
    }
  }
}
const base = rows["engine (Pyodide, 1 thread)"];
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
console.log("\n| run | tok/s (each round) | median | against the engine |\n|---|---|---:|---:|");
for (const [key, values] of Object.entries(rows)) {
  console.log(`| ${key} | ${values.map((v) => v.toFixed(1)).join(", ")} | ${median(values).toFixed(1)} | ${(median(values) / median(base)).toFixed(2)}× |`);
}
console.log(`\nthe prototype wrote the engine's tokens in every run (${expected.length} greedy tokens)`);
process.exit(0);
