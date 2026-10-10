// logits.mjs (the review of T253): the page's engine's logits at the positions of a text, saved for
// tests/by_layer_logits.py to hold to transformers' own (a layer at a time): the whole depth of a model that no runner
// can hold in float32 (Granite 4.2 8B), at the very ids the page's engine was given, as perplexity (tests/start_check.*)
// cannot say whether the two agree where they differ by a percent.
//
//   OUT=.tmp/engine TEXT=.tmp/ja.txt TOKENS=96 WRITER=logits.mjs bash tests/write.sh hf-granite-4.2-8b
//   (or node tests/logits.mjs <out of tests/perplexity_prepare.py> <JSON of tests/write_options.py> [tokens = 96])
//
// The ids are the page's BOS and the first TOKENS of the text (the converter's tokenizer, through the engine's
// tokenizer); OUT.ids.json holds them and OUT.logits.f32 the float32 logits, one row (the vocabulary) for each of the
// positions, from llama.forward(token, position) one token at a time: the path a generated token takes (7-bit activations
// on the int8 kernels).
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";
import { footprint, needsWide } from "../public/forward.js";
import { leave } from "./leave.mjs";

const [out, pageFile, count = "96"] = process.argv.slice(2);
const page = JSON.parse(fs.readFileSync(pageFile, "utf8"));
const size = fs.statSync(`${out}.bin`).size;
const { arch, head_dim, dtype } = page.options;
const wide = needsWide(size, footprint(page.header, size, { dtype, arch, head_dim, halfKV: true, shared: true }));
const { pyodide } = await pyodideWithEngine({ wide });
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(`${out}.tokenizer.bin`));
const saveTo = process.env.OUT ?? ".tmp/engine";
pyodide.globals.set("CHECKPOINT", `${out}.bin`);
pyodide.globals.set("OPTIONS", pyodide.toPy(page.options));
pyodide.globals.set("TEXT", fs.readFileSync(process.env.TEXT, "utf8"));
pyodide.globals.set("COUNT", Number(process.env.TOKENS ?? count));
pyodide.runPython(`
import json, time
import numpy as np
llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **OPTIONS)
ids = [llama.bos] + llama.tokenizer.encode(TEXT, llama.specials)[:COUNT]
began = time.perf_counter()
rows = [np.asarray(llama.forward(token, position), dtype=np.float32).copy() for position, token in enumerate(ids)]
SECONDS = time.perf_counter() - began
IDS = json.dumps(ids)
import base64
LOGITS = base64.b64encode(np.stack(rows).astype(np.float32).tobytes()).decode()
`);
fs.mkdirSync(saveTo.replace(/[^/]*$/, "") || ".", { recursive: true });
fs.writeFileSync(`${saveTo}.logits.f32`, Buffer.from(pyodide.globals.get("LOGITS"), "base64"));
fs.writeFileSync(`${saveTo}.ids.json`, pyodide.globals.get("IDS"));
console.log(`logits ${page.id}: ${JSON.parse(pyodide.globals.get("IDS")).length} positions of the page's engine (${wide ? "a 64-bit" : "a 32-bit"} memory, int8 ${(size / 1e9).toFixed(2)} GB) ` +
  `in ${pyodide.globals.get("SECONDS").toFixed(0)} s, saved as ${saveTo}.logits.f32 (float32, a row of the vocabulary for each position) and ${saveTo}.ids.json`);
// (T357: what was written is out first: the answers of a long run are more than a pipe holds, tests/leave.mjs)
await leave(0);
