// write.mjs (T249): what a Hugging Face model of the list writes for its own prompt on the page's engine
// (public/forward.js on the int8 kernels, tests/engine.mjs), greedy, for a person to read: whether a model added to
// the list writes sensible text is not something a number says. The checkpoint is read from its file straight into
// forward.js's memory (a 64-bit one where the model with its forward pass does not fit 4 GiB, as on the page).
//
//   node tests/write.mjs <out of tests/perplexity_prepare.py> <JSON of tests/write_options.py> [tokens = 32]
//
// SEEDS="1 2 3": after the greedy answer, one with the entry's own sampling (as the page writes) for each seed. A
// small model may loop when greedy and still write well as the page runs it.
//
// tests/write.sh fetches and converts a model and runs this (in CI: tests.yml's extra=).
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";
import { footprint, needsWide } from "../public/forward.js";
import { filled } from "../src/models.js";
import { leave } from "./leave.mjs";

const [out, pageFile, count = "32"] = process.argv.slice(2);
const page = JSON.parse(fs.readFileSync(pageFile, "utf8"));
const size = fs.statSync(`${out}.bin`).size;
const { arch, head_dim, dtype } = page.options;
const wide = needsWide(size, footprint(page.header, size, { dtype, arch, head_dim, halfKV: true, shared: true }));
const { pyodide } = await pyodideWithEngine({ wide });
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(`${out}.tokenizer.bin`));
const text = page.template ? filled(page.template, page.prompt) : page.prompt;
const { temperature, topp, repetition_penalty } = page.generation;
pyodide.globals.set("CHECKPOINT", `${out}.bin`);
pyodide.globals.set("OPTIONS", pyodide.toPy(page.options));
pyodide.globals.set("TEXT", text);
pyodide.globals.set("COUNT", Number(count));
pyodide.globals.set("SEEDS", pyodide.toPy((process.env.SEEDS ?? "").split(/\s+/).filter(Boolean).map(Number)));
pyodide.globals.set("SAMPLING", pyodide.toPy(Object.fromEntries(Object.entries({ temperature, topp, repetition_penalty }).filter(([, value]) => value !== undefined))));
const result = JSON.parse(pyodide.runPython(`
import json, time
llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **OPTIONS)
ids = llama.tokenizer.encode(TEXT, llama.specials)
began = time.perf_counter()
written = "".join(llama.generate(TEXT, steps=len(ids) + COUNT, temperature=0.0, echo=False))
stats = dict(llama.stats)
sampled = [[seed, "".join(llama.generate(TEXT, steps=len(ids) + COUNT, seed=seed, echo=False, **SAMPLING))] for seed in SEEDS]
json.dumps({"written": written, "prompt_tokens": len(ids), "sampled": stats["sampled"], "bos": llama.bos,
            "stops": sorted(llama.stop_tokens), "seconds": stats["seconds"], "others": sampled,
            "tokens_per_second": stats["tokens_per_second"]})
`));
const shown = (value) => JSON.stringify(value);
console.log(`written by ${page.id} (${page.name}; ${wide ? "a 64-bit" : "a 32-bit"} memory, int8 ${(size / 1e9).toFixed(2)} GB, bos ${result.bos}, ` +
  `stops ${result.stops.join(" ")}, ${result.prompt_tokens} tokens of prompt, ${result.sampled} sampled in ${result.seconds.toFixed(0)} s, ` +
  `${result.tokens_per_second.toFixed(1)} tok/s on one thread)`);
console.log(`written by ${page.id}: prompt ${shown(text)}`);
console.log(`written by ${page.id}: answer ${shown(result.written)}`);
for (const [seed, answer] of result.others) {
  console.log(`written by ${page.id}: answer with temperature ${temperature}, top-p ${topp}, penalty ${repetition_penalty} and seed ${seed} ${shown(answer)}`);
}
// (T357: what was written is out first: the answers of a long run are more than a pipe holds, tests/leave.mjs)
await leave(0);
