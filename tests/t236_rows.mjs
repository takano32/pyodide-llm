// t236_rows.mjs (T236's review, a probe for CI, not for main): the kernel rows of tests/perplexity.mjs (7-bit and 8-bit
// activations, the page's forward pass) for several texts and several converted checkpoints in one Pyodide, so that the
// GGUF's path and the original's are held to the same texts on the same CPU, and the spread of one row over the texts
// is there to read the difference of two rows against.
//
//   node tests/t236_rows.mjs '{"count": 1022, "paths": [{"name": "gguf", "out": ".tmp/g8"}, {"name": "original", "out": ".tmp/o8"}], "texts": ["a.txt", ...]}'
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";

const spec = JSON.parse(process.argv[2]);
const { pyodide } = await pyodideWithEngine();
pyodide.FS.writeFile("perplexity.py", fs.readFileSync(new URL("./perplexity.py", import.meta.url)));
spec.texts.forEach((file, i) => pyodide.FS.writeFile(`text${i}.txt`, fs.readFileSync(file)));
spec.paths.forEach((p, i) => pyodide.FS.writeFile(`tokenizer${i}.bin`, fs.readFileSync(`${p.out}.tokenizer.bin`)));
pyodide.globals.set("SPEC", JSON.stringify({ ...spec, names: spec.texts.map((file) => path.basename(file)),
  files: spec.paths.map((p) => path.resolve(`${p.out}.bin`)), options: spec.paths.map((p) => JSON.parse(fs.readFileSync(`${p.out}.json`, "utf8"))) }));
pyodide.runPython(`
import gc, json, math, time
import numpy as np
from perplexity import perplexity

spec = json.loads(SPEC)
for i, p in enumerate(spec["paths"]):
    vocabulary = open(f"tokenizer{i}.bin", "rb").read()
    for label, disable in (("7-bit", ()), ("8-bit", ("relaxed",))):
        llama = kernel_llama_file(spec["files"][i], vocabulary, **dict(spec["options"][i], disable=disable))
        for j, name in enumerate(spec["names"]):
            tokens = llama.tokenizer.encode(open(f"text{j}.txt").read())[:spec["count"]]
            began = time.perf_counter()
            value, count = perplexity(llama, tokens, min(512, llama.seq_len))
            print("T236ROWS", json.dumps({"path": p["name"], "row": label, "text": name, "tokens": count, "perplexity": value,
                                          "seconds": round(time.perf_counter() - began)}), flush=True)
        llama.release()
        del llama
        gc.collect()
`);
process.exit(0);
