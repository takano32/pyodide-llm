// Where the page's conversion spends its time (T89): converts a Hugging Face model directory inside Pyodide in Node,
// fed in 8 MiB pieces as the worker feeds it (by the conduct of a conversion, as the worker's is: T374.2.1), under
// cProfile. No download: the files are on disk, so this is the converting alone, and the rest of a load in the browser
// is the fetching.
//
//   node tests/profile-convert.mjs <directory with config.json, model.safetensors and the tokenizer> [int8|int6|float32] [--numpy]
//   node tests/profile-convert.mjs <a Q8_0 .gguf file> [int8|int6|float32] [--numpy]
//   node tests/profile-convert.mjs <directory with config.json, the tokenizer and a Q8_0 .gguf> [...]
//       (T145: a GGUF with its original's vocabulary and configuration, T136's second stage; tests/hf_fetch.py makes it)
//
// int8 quantizes on the SIMD kernels, as the page does (T89: kernel_quantizer); int6 (T98) as the page does too.
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";
const [dir, dtype = "int8"] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const { pyodide: py } = await pyodideWithEngine();
// LAYOUT=<bytes>,<bytes> (T374.2.1): two buffers of those sizes made and held in Python, one before anything of the
// conversion and one once the conversion is made, before it is fed. Where the conversion's buffers lie beside one
// another is the same in every run of one tool, and another tool (or tree) lays them out otherwise: two tools that fed
// the same converter the same parts differed by one to three percent in CI, always the same way, until each run had
// its own layout (tests/abba-convert.sh gives every run two sizes of its own)
const [layoutBefore = 0, layoutBeside = 0] = (process.env.LAYOUT ?? "").split(",").map(Number);
py.runPython(`layout_before = bytes(${layoutBefore})`);
const convert = py.pyimport("llama2_convert");
const quantizeRows = py.pyimport("llama2_numpy").kernel_quantizer("simdkernel.so");
// the stored types' readers on the kernels, as the page does (T123: bfloat16; T136: GGUF's Q8_0; T273: the two ternary
// types); --numpy: NumPy's, to compare
const readers = process.argv.includes("--numpy") ? undefined : convert.kernel_readers("simdkernel.so");
const gguf = dir.endsWith(".gguf");
const folder = gguf ? path.dirname(dir) : dir;
const withVocabulary = !gguf && fs.readdirSync(dir).filter((name) => name.endsWith(".gguf")).sort()[0];
const tokenizer = ["tokenizer.json", "spiece.model", "tokenizer.model"].find((n) => fs.existsSync(`${folder}/${n}`));
// the model as the page lists it: the folder stands for its repository (and for the original's, T136's second stage)
const hf = gguf ? { weights: path.basename(dir) } : withVocabulary ? { weights: withVocabulary, vocabulary: { tokenizer } } : { weights: "model.safetensors", tokenizer };
// T374.2.1: by the conduct of a conversion (public/convert/conduct.py), answered from the folder as the worker answers
// it from huggingface.co (public/worker/conduct.js): a request and an answer for every file, and the parts to the
// conversion's own feed, which the request of a stream brings. (Before, this called the converter itself and fed it;
// tests/abba-convert.sh against a tree of before T374.2.1 therefore times what the conduct adds to a conversion.)
const opened = new Map();  // a file is opened once: a part costs one read, as it did
const open = (name) => {
  if (!opened.has(name)) {
    const fd = fs.openSync(`${folder}/${name}`, "r");
    opened.set(name, { fd, size: fs.fstatSync(fd).size });
  }
  return opened.get(name);
};
const sizeOf = (name) => open(name).size;
const range = (name, begin, end) => {
  const { fd, size } = open(name), bytes = new Uint8Array(Math.min(end, size) - begin);
  fs.readSync(fd, bytes, 0, bytes.length, begin);
  return bytes;
};
const there = (name, read) => (fs.existsSync(`${folder}/${name}`) ? read() : undefined);
let js = 0, timed = false;
const size = sizeOf(hf.weights);
const answers = {
  text: (name) => there(name, () => fs.readFileSync(`${folder}/${name}`, "utf8")),
  bytes: (name) => there(name, () => new Uint8Array(fs.readFileSync(`${folder}/${name}`))),
  range: (name, begin, end) => there(name, () => [range(name, begin, end), sizeOf(name)]),
  size: sizeOf,
  // the parts of a stream, 8 MiB each as the worker's first, to the conversion's own feed as the worker hands them
  stream(name, begin, end, before, total, feed) {
    // the clock (and the profiler) from the first part on: the head, the tokenizer and the template are read before
    if (!timed) {
      py.runPython(`layout_beside = bytes(${layoutBeside})`);
      py.runPython("import cProfile, pstats, io, time; profiler = cProfile.Profile(); began = time.perf_counter()");
      // PROFILE=0: the time alone (cProfile counts every call, which makes a change in the number of calls look larger)
      py.runPython(process.env.PROFILE === "0" ? "profiler.enable(); profiler.disable(); began = time.perf_counter()" : "profiler.enable()");
      timed = true;
    }
    for (let at = begin; at < end; at += 8 << 20) {
      const t = performance.now();
      const chunk = range(name, at, Math.min(at + (8 << 20), end));
      js += performance.now() - t;
      feed(chunk);
    }
    feed.destroy();
  },
};
const module = py.pyimport("convert.conduct"), listed = py.toPy(hf);
const steps = module.conduct.callKwargs(listed, { dtype, quantize_rows: quantizeRows, readers });
const taken = (step) => {
  const request = step.value.toJs({ depth: 1 });
  step.value.destroy();
  return request;
};
let request = taken(steps.next());
while (request[0] !== "done") {
  const [kind, , name, ...rest] = request;
  if (kind === "missing") throw new Error(`${folder} has no ${name}`);
  request = taken(steps.next(answers[kind](name, ...rest)));
}
// (the conduct finished the conversion before it said so)
console.log(py.runPython(`
profiler.disable()
total = time.perf_counter() - began
out = io.StringIO()
pstats.Stats(profiler, stream=out).sort_stats("tottime").print_stats(12)
f"converted {${size} / 1e6:.0f} MB in {total:.1f} s ({${size} / 1e6 / total:.0f} MB/s)\\n" + "\\n".join(line for line in out.getvalue().splitlines() if line.strip())[:3000]
`));
console.log(`reading the file in JS: ${(js / 1000).toFixed(1)} s`);
