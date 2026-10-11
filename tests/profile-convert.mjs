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
import { converted, fromFolder, listed } from "./conducting.mjs";
import { pyodideWithEngine } from "./engine.mjs";
const [dir, dtype = "int8"] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const { pyodide: py } = await pyodideWithEngine();
// LAYOUT=<bytes>,<bytes> (T374.2.1): two buffers of those sizes made and held in Python, one before anything of the
// conversion and one once the conversion is made, before it is fed. Where the conversion's buffers lie beside one
// another is the same in every run of one tool, and another tool (or tree) lays them out otherwise: with sizes of its
// own for every run (tests/abba-convert.sh), that is not taken for a difference of work. (It is not all there is to
// it: see TODO.md's T374.2.1 for what the runs of CI said with it.)
const [layoutBefore = 0, layoutBeside = 0] = (process.env.LAYOUT ?? "").split(",").map(Number);
py.runPython(`layout_before = bytes(${layoutBefore})`);
const convert = py.pyimport("llama2_convert");
const quantizeRows = py.pyimport("llama2_numpy").kernel_quantizer("simdkernel.so");
// the stored types' readers on the kernels, as the page does (T123: bfloat16; T136: GGUF's Q8_0; T273: the two ternary
// types); --numpy: NumPy's, to compare
const readers = process.argv.includes("--numpy") ? undefined : convert.kernel_readers("simdkernel.so");
// T374.2.1: by the conduct of a conversion (public/convert/conduct.py), answered from the folder as the worker answers
// it from huggingface.co (public/worker/conduct.js): a request and an answer for every file, and the parts to the
// conversion's own feed, which the request of a stream brings. (Before, this called the converter itself and fed it;
// tests/abba-convert.sh against a tree of before T374.2.1 therefore times what the conduct adds to a conversion.)
// T374.4: the answerer and what a path stands for are tests/conducting.mjs's (page-27b.mjs has the same), and the
// tokenizer is the conduct's choice of its own candidates.
const { hf, folder } = listed(dir);
let js = 0, size = 0;
const answerer = fromFolder(folder, {
  // the parts of a stream, 8 MiB each as the worker's first, to the conversion's own feed as the worker hands them
  part: 8 << 20,
  // the clock (and the profiler) from the first part on: the head, the tokenizer and the template are read before
  starting(total) {
    size = total;
    py.runPython(`layout_beside = bytes(${layoutBeside})`);
    py.runPython("import cProfile, pstats, io, time; profiler = cProfile.Profile(); began = time.perf_counter()");
    // PROFILE=0: the time alone (cProfile counts every call, which makes a change in the number of calls look larger)
    py.runPython(process.env.PROFILE === "0" ? "profiler.enable(); profiler.disable(); began = time.perf_counter()" : "profiler.enable()");
  },
  reading(ms) { js += ms; },
});
converted(py, hf, { dtype, quantize_rows: quantizeRows, readers }, answerer).destroy();
// (the conduct finished the conversion before it said so)
console.log(py.runPython(`
profiler.disable()
total = time.perf_counter() - began
out = io.StringIO()
pstats.Stats(profiler, stream=out).sort_stats("tottime").print_stats(12)
f"converted {${size} / 1e6:.0f} MB in {total:.1f} s ({${size} / 1e6 / total:.0f} MB/s)\\n" + "\\n".join(line for line in out.getvalue().splitlines() if line.strip())[:3000]
`));
console.log(`reading the file in JS: ${(js / 1000).toFixed(1)} s`);
