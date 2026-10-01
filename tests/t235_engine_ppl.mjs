// t235_engine_ppl.mjs (T235's review, a probe for CI, not for main): the page's own forward pass (forward.js on the kernels,
// int8 weights, 7-bit activations, the checkpoint read from its file) on a Japanese text, under config.json's yarn and
// under a plain RoPE (rope_scaling: {}, what an entry's options would say): the perplexity of the first N tokens after
// a BOS, in one window, against transformers' prefix perplexities (run 36885501182: ja-fuji 1024 tokens 45.1127 / 42.2800).
//
//   node tests/t235_engine_ppl.mjs <out of tests/perplexity_prepare.py> <tokens> <window> <text file> ...
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";

const root = new URL("../", import.meta.url).pathname;
const [out, count, window, ...texts] = process.argv.slice(2);
const options = JSON.parse(fs.readFileSync(`${out}.json`, "utf8"));
const { pyodide } = await pyodideWithEngine();
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(`${out}.tokenizer.bin`));
// perplexity() of tests/perplexity.py (its main only runs where MODEL is a global)
pyodide.runPython(fs.readFileSync(`${root}tests/perplexity.py`, "utf8"));
pyodide.globals.set("FILE", path.resolve(`${out}.bin`));
pyodide.globals.set("TOKENS", Number(count));
pyodide.globals.set("WINDOW", Number(window));
for (const file of texts) {
  pyodide.globals.set("TEXT", fs.readFileSync(file, "utf8"));
  for (const [name, change] of [["yarn (config.json)", {}], ["plain RoPE (rope_scaling: {})", { rope_scaling: {} }]]) {
    pyodide.globals.set("OPTIONS", pyodide.toPy({ ...options, ...change }));
    const began = Date.now();
    const result = pyodide.runPython(`
import time
vocabulary = open("tokenizer.bin", "rb").read()
llama = kernel_llama_file(FILE, vocabulary, **OPTIONS)
tokens = llama.tokenizer.encode(TEXT)[:TOKENS]
value, counted = perplexity(llama, tokens, WINDOW)
text = f"{value:.4f} over {counted} targets, backend {llama.backend}"
llama.release()
del llama
text
`);
    console.log(`PPLENGINE ${path.basename(file)} ${name}: ${result} (${((Date.now() - began) / 1000).toFixed(0)} s)`);
  }
}
