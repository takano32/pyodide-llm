// Where the time of one token goes: runs tests/profile_token.py inside Pyodide, in Node or in a real browser, and
// prints Markdown tables. The models come from this directory (make models kernels).
//
//   node tests/profile.mjs [node|chromium|firefox] [model id ...]      (default: node, llm-jp-3-150m tiny-lm stories260K)
//   PROFILE_THREAD=worker node tests/profile.mjs chromium              inside a worker, as the page runs it
//
// Mind the memory: llm-jp-3-150m needs about 600 MB here (the file and the model). Keep `free -m` above 1.5 GB.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
let [where = "node", ...ids] = process.argv.slice(2);
const chosen = (ids.length ? ids : ["llm-jp-3-150m", "tiny-lm", "stories260K"]).map((id) => MODELS.find((model) => model.id === id));
const models = chosen.map((model) => ({ name: model.id, checkpoint: model.checkpoint, tokenizer: model.tokenizer, options: model.options, generation: model.generation }));
const files = ["public/llama2_numpy.py", "public/simdkernel.so", "public/simdkernel_relaxed.wasmlib", "tests/profile_token.py",
               ...new Set(models.flatMap((model) => [model.checkpoint, model.tokenizer]))];

let report, environment;
if (where === "node") {
  const { loadPyodide, version } = await import("pyodide");
  const pyodide = await loadPyodide();
  await pyodide.loadPackage("numpy", { messageCallback: () => {} });
  for (const file of files) {
    pyodide.FS.writeFile(path.basename(file), fs.readFileSync(root + file));
  }
  pyodide.globals.set("MODELS", pyodide.toPy(models));
  report = JSON.parse(pyodide.runPython(fs.readFileSync(`${root}tests/profile_token.py`, "utf8")));
  environment = `Pyodide ${version} in Node ${process.version}`;
} else {
  const playwright = await import("playwright-core");
  const { version } = await import("pyodide");
  const server = http.createServer((req, res) => {
    const file = path.join(root, decodeURIComponent(new URL(req.url, "http://localhost").pathname));
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(200, { "Content-Type": "text/html" });
      return res.end("<!doctype html><meta charset=utf-8><title>profile</title>");
    }
    res.writeHead(200, { "Content-Type": "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  }).listen(0);
  const engine = where;
  const browser = await playwright[engine].launch({ headless: true });
  const page = await browser.newPage();
  page.on("pageerror", (error) => console.error(String(error)));
  await page.goto(`http://localhost:${server.address().port}/`);
  // The page runs Pyodide in a worker, so that is measured too (PROFILE_THREAD=worker): a browser may give a worker
  // another core or another priority than its main thread.
  const inWorker = process.env.PROFILE_THREAD === "worker";
  const run = async ({ version, files, models, origin }) => {
    const { loadPyodide } = await import(`https://cdn.jsdelivr.net/pyodide/v${version}/full/pyodide.mjs`);
    const pyodide = await loadPyodide();
    await pyodide.loadPackage("numpy");
    let script;
    for (const file of files) {
      const data = new Uint8Array(await (await fetch(`${origin}/${file}`)).arrayBuffer());
      pyodide.FS.writeFile(file.split("/").pop(), data);
      if (file.endsWith("profile_token.py")) {
        script = new TextDecoder().decode(data);
      }
    }
    pyodide.globals.set("MODELS", pyodide.toPy(models));
    return pyodide.runPython(script);
  };
  const input = { version, files, models, origin: `http://localhost:${server.address().port}` };
  report = JSON.parse(inWorker
    ? await page.evaluate(({ source, input }) => new Promise((resolve, reject) => {
      const code = `const run = ${source}; onmessage = async ({ data }) => { try { postMessage({ result: await run(data) }); } catch (error) { postMessage({ error: String(error) }); } };`;
      const worker = new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })), { type: "module" });
      worker.onmessage = ({ data }) => (data.error ? reject(new Error(data.error)) : resolve(data.result));
      worker.onerror = (event) => reject(new Error(event.message));
      worker.postMessage(input);
    }), { source: run.toString(), input })
    : await page.evaluate(run, input));
  if (inWorker) {
    where = `a worker of ${where}`;
  }
  environment = `Pyodide ${version} in ${where} ${browser.version()}`;
  await browser.close();
  server.close();
}

const ms = (value) => value.toFixed(2);
console.log(`### One token, ${environment}\n`);
console.log("| model | calls | per call | kernels: layers | kernels: classifier | ctypes calls | Python and NumPy | sampling | token | tok/s | read per token |");
console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const m of report.models) {
  const share = (value) => `${ms(value)} ms (${Math.round((value / m.token_ms) * 100)}%)`;
  console.log(`| ${m.name} (${m.backend}) | ${m.calls} | ${m.call_us.toFixed(1)} us | ${share(m.kernels_layers_ms)} | ${share(m.kernels_classifier_ms)} | ${share(m.ctypes_ms)} | ${share(m.python_ms)} | ${share(m.sampling_ms)} | ${ms(m.token_ms)} ms | ${m.tokens_per_second.toFixed(0)} | ${m.weights_read_mb.toFixed(0)} MB (classifier ${m.classifier_read_mb.toFixed(0)}) |`);
}
console.log(`\n### Matmul throughput by matrix size (row length 768), ${environment}\n`);
console.log("| rows | int8 weights | matmul_q8 | matmul_q8r (relaxed SIMD) | float32 weights | matmul_f32 |");
console.log("|---:|---:|---:|---:|---:|---:|");
for (const row of report.bandwidth) {
  const rate = (name, bytes) => (row[name] ? `${row[name].toFixed(2)} G MAC/s (${(row[name] * bytes).toFixed(1)} GB/s)` : "-");
  console.log(`| ${row.rows} | ${row.int8_mb.toFixed(1)} MB | ${rate("matmul_q8", 1.125)} | ${rate("matmul_q8r", 1.125)} | ${row.float32_mb ? row.float32_mb.toFixed(1) + " MB" : "-"} | ${rate("matmul_f32", 4)} |`);
}
