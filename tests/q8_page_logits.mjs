// q8_page_logits.mjs (the review of T247, a probe for CI, not for main): the page's forward pass (forward.js on the int8 kernels,
// 7-bit and 8-bit activations) over given token ids, its logits at every position saved as raw float32 (positions x vocabulary),
// for tests/q8_page_compare.py to hold against transformers'.
//
//   node tests/q8_page_logits.mjs '{"out": ".tmp/g8", "ids": [248044, ...], "labels": ["7-bit", "8-bit"], "save": ".tmp/page", "wide": true}'
//   (<out>: the <out> of tests/perplexity_prepare.py; writes <save>-<label>-logits.f32)
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";

const spec = JSON.parse(process.argv[2]);
const { pyodide } = await pyodideWithEngine({ wide: Boolean(spec.wide) });
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(`${spec.out}.tokenizer.bin`));
pyodide.globals.set("SPEC", JSON.stringify({ file: path.resolve(`${spec.out}.bin`), options: JSON.parse(fs.readFileSync(`${spec.out}.json`, "utf8")),
  ids: spec.ids, labels: spec.labels ?? ["7-bit", "8-bit"] }));
const written = [];
pyodide.runPython(`
import gc, json, time
import numpy as np

spec = json.loads(SPEC)
vocabulary = open("tokenizer.bin", "rb").read()
for label, disable in (("7-bit", ()), ("8-bit", ("relaxed",))):
    if label not in spec["labels"]:
        continue
    llama = kernel_llama_file(spec["file"], vocabulary, **dict(spec["options"], disable=disable))
    began = time.perf_counter()
    logits = np.stack([np.array(llama.forward(token, pos), dtype=np.float32) for pos, token in enumerate(spec["ids"])])
    open(f"logits-{label}.f32", "wb").write(logits.tobytes())
    print("Q8PAGE", label, logits.shape, f"{time.perf_counter() - began:.0f} s", "min", float(logits.min()), "max", float(logits.max()), flush=True)
    llama.release()
    del llama
    gc.collect()
`);
for (const label of spec.labels ?? ["7-bit", "8-bit"]) {
  fs.writeFileSync(`${spec.save}-${label}-logits.f32`, pyodide.FS.readFile(`logits-${label}.f32`));
}
process.exit(0);
