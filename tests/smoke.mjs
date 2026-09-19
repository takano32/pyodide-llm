// Smoke test for the deployment: the Python engine under the latest Pyodide (in Node, no browser), with the
// models that `make models` has just produced. A broken engine, a broken conversion or an incompatible Pyodide
// release fails here instead of on the site.
//
//   make models && node tests/smoke.mjs
import fs from "node:fs";
import { loadPyodide, version } from "pyodide";

const root = new URL("../", import.meta.url).pathname;
const pyodide = await loadPyodide();
await pyodide.loadPackage("numpy", { messageCallback: () => {} });
for (const [file, name] of [["public/llama2_numpy.py", "llama2_numpy.py"], ["stories260K.bin", "stories260K.bin"],
                            ["tok512.bin", "tok512.bin"], ["tiny-lm.bin", "tiny-lm.bin"], ["tiny-lm.tokenizer.bin", "tiny-lm.tokenizer.bin"]]) {
  pyodide.FS.writeFile(name, fs.readFileSync(root + file));
}

const started = Date.now();
const report = pyodide.runPython(`
import sys
import llama2_numpy

def read(name):
    with open(name, "rb") as f:
        return f.read()

# a float32 model with grouped-query attention and llama2.c's tokenizer: greedy text is deterministic
stories = llama2_numpy.Llama(read("stories260K.bin"), read("tok512.bin"))
text = "".join(stories.generate("Once upon a time", steps=40))
expected = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park."
assert text.startswith(expected), f"stories260K wrote: {text!r}"

# the default model: converted from Hugging Face, quantized to int8, unigram tokenizer, sampled
tiny = llama2_numpy.Llama(read("tiny-lm.bin"), read("tiny-lm.tokenizer.bin"), dtype="int8",
                          tokenizer_kind="unigram", nfkc=True, stop_tokens=(1, 2))
assert tiny.tokenizer.encode("ＡＢＣ") == tiny.tokenizer.encode("ABC"), "NFKC normalization is off"
japanese = "".join(tiny.generate("昔々、", steps=12, temperature=0.7, repetition_penalty=1.3, seed=1))
assert japanese.startswith("昔々、") and len(japanese) > len("昔々、"), f"tiny-lm wrote: {japanese!r}"
assert japanese == "".join(tiny.generate("昔々、", steps=12, temperature=0.7, repetition_penalty=1.3, seed=1)), "a seed must reproduce"

f"Python {sys.version.split()[0]}: stories260K {stories.stats['tokens_per_second']:.0f} tok/s, tiny-lm {tiny.stats['tokens_per_second']:.0f} tok/s, {japanese!r}"
`);
console.log(`Pyodide ${version}, ${report} (${((Date.now() - started) / 1000).toFixed(1)}s)`);
console.log("ok");
