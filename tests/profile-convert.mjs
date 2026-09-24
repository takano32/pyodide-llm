// Where the page's conversion spends its time (T89): converts a Hugging Face model directory inside Pyodide in Node,
// fed in 8 MiB pieces as the worker feeds it, under cProfile. No download: the files are on disk, so this is the
// converting alone, and the rest of a load in the browser is the fetching.
//
//   node tests/profile-convert.mjs <directory with config.json, model.safetensors and the tokenizer> [int8|float32]
import fs from "node:fs";
import { loadPyodide } from "pyodide";
const [dir, dtype = "int8"] = process.argv.slice(2);
const py = await loadPyodide();
await py.loadPackage("numpy", { messageCallback: () => {} });
py.FS.writeFile("llama2_convert.py", fs.readFileSync("public/llama2_convert.py"));
const convert = py.pyimport("llama2_convert");
const weights = `${dir}/model.safetensors`;
const fd = fs.openSync(weights, "r"), size = fs.fstatSync(fd).size;
const range = (b, e) => { const x = new Uint8Array(e - b); fs.readSync(fd, x, 0, e - b, b); return x; };
const headerBytes = Number(new DataView(range(0, 8).buffer).getBigUint64(0, true));
const tokName = ["tokenizer.json", "spiece.model", "tokenizer.model"].find((n) => fs.existsSync(`${dir}/${n}`));
const conversion = convert.Conversion.callKwargs(new TextDecoder().decode(range(8, 8 + headerBytes)), 8 + headerBytes,
  fs.readFileSync(`${dir}/config.json`, "utf8"), new Uint8Array(fs.readFileSync(`${dir}/${tokName}`)), tokName, { dtype, start: 8 + headerBytes });
py.globals.set("conversion", conversion);
py.runPython("import cProfile, pstats, io, time; profiler = cProfile.Profile(); began = time.perf_counter(); profiler.enable()");
let js = 0;
for (let at = 8 + headerBytes; at < size; at += 8 << 20) {
  const t = performance.now(); const chunk = range(at, Math.min(at + (8 << 20), size)); js += performance.now() - t;
  conversion.feed(chunk);
}
conversion.finish();
console.log(py.runPython(`
profiler.disable()
total = time.perf_counter() - began
out = io.StringIO()
pstats.Stats(profiler, stream=out).sort_stats("tottime").print_stats(12)
f"converted {${size} / 1e6:.0f} MB in {total:.1f} s ({${size} / 1e6 / total:.0f} MB/s)\\n" + "\\n".join(line for line in out.getvalue().splitlines() if line.strip())[:3000]
`));
console.log(`reading the file in JS: ${(js / 1000).toFixed(1)} s`);
