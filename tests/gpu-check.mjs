// T135: the prompt on the GPU (public/gpu.js) against the CPU's (public/forward.js) and NumPy's (llama2_numpy.py), in
// a real browser, as WebGPU is nowhere else: Playwright's Chromium, whose WebGPU without a GPU is SwiftShader (the CPU
// in the GPU's place: its speed means nothing, its numbers are right or not).
//
//   node tests/gpu-check.mjs [model id | synthetic ...] [--engine chromium|chrome|msedge]
//
// Node reads each model with Pyodide as the page does and records two things: the plan that forward.js gets from
// Python (where every tensor is), and NumPy's answer for a prompt of 40 tokens: the keys and values of every layer at
// every position the prompt's blocks fill, and the logits of its last token. NumPy multiplies the int8 weights
// widened to float32 by float32 activations, which is what the GPU does; forward.js on the CPU quantizes the
// activations as well (7 bits with relaxed SIMD), so it is farther from both by design. The browser then runs
// forward.js in a worker (it waits in Atomics.wait, which a page may not) twice on the same memory: the prompt through
// forwardMany() and its last token through forward(), once on the CPU and once with the GPU's worker. The cache starts
// at 8 positions, so that both grow within the prompt (8, 16, 32, 64). Checked:
//   - the GPU took every token of the prompt (gpuTokens), the same again from position 0, and a block that begins
//     past the keys and values it holds went to the CPU;
//   - the keys and values it wrote back into the cache, against NumPy's: the worst row (a layer's keys or values of
//     one position) no more than GPU_LINE of the row's largest (see there for the number);
//   - the logits of the prompt's last token (the CPU's in both runs, on the GPU's keys and values in one): the same
//     most likely token as the run on the CPU, and no farther from NumPy's than LOGITS_LINE times that run (see there).
// "synthetic": a made-up int8 model with grouped-query attention (4 heads, 2 of keys and values; none of the models
// of this directory has it). The others are the models of this directory (make models kernels).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import * as playwright from "playwright-core";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? args.splice(args.indexOf(name), 2)[1] : value);
const engine = option("--engine", "chromium");
const ids = args.length ? args : ["synthetic", "stories15M", "tiny-lm", "llm-jp-3-150m"];
const COUNT = 40, KV_START = 8;
// Measured on this machine's SwiftShader (2026-09-26, T135, the four models below), the worst row of the keys and
// values against NumPy's: the GPU's 4.5e-4 to 4.7e-4 (the float16 of the cache: 2^-11 = 4.9e-4 of a value; the float32
// sums in another order are far below it), the CPU's 4.0e-2 to 1.4e-1 (its 7-bit activations). A GPU that is wrong
// lands far past the CPU's (broken on purpose: RoPE at the next position 1.13 to 1.16, no causal mask 1.86 to 6.97,
// the keys written back in the order [token][layer] 2.50 to 4.62). The line: 2e-3, four times the float16's own.
const GPU_LINE = 2e-3;
// The logits of the prompt's last token, the largest difference from NumPy's over the largest of NumPy's: the CPU's
// own run 1.6e-2 to 5.6e-2, the one on the GPU's keys and values 0.75 to 1.04 times that (closer: its keys and values
// are NumPy's but for the float16); broken on purpose 0.27 to 1.13, 4.8 times the CPU's and more. The line: no more
// than 1.5 times the CPU's, and the same most likely token.
const LOGITS_LINE = 1.5;

// ---- Node: the plans and NumPy's answers
const { pyodide: py } = await pyodideWithEngine();
py.runPython(`
import base64, struct, numpy as np, llama2_numpy, llama2_convert
from llama2_numpy import Llama

def synthetic(dim=64, hidden=128, layers=2, heads=4, kv_heads=2, vocab=320, seq_len=64, seed=0):
    """A made-up int8 checkpoint and its tokenizer.bin, as quantize.py writes one: grouped-query attention"""
    rng = np.random.default_rng(seed)
    header = (dim, hidden, layers, heads, kv_heads, vocab, seq_len)
    out = [struct.pack("<7i", *header)]
    for shape, is_matrix in llama2_convert.layout(*header):
        if is_matrix is None:
            continue  # the RoPE tables: an int8 file leaves them out
        values = (rng.standard_normal(shape) * 0.3).astype(np.float32)
        if not is_matrix:
            out.append((1.0 + values * 0.1).astype(np.float32).tobytes())
            continue
        q, scales = llama2_convert.quantize(values.reshape(-1, shape[-1]))
        out += [q.tobytes(), scales.tobytes()]
    pieces = [f"<{i}>".encode() for i in range(vocab)]
    tokenizer = struct.pack("<i", max(map(len, pieces))) + b"".join(struct.pack("<fi", 0.0, len(p)) + p for p in pieces)
    return b"".join(out), tokenizer

def answer(data, vocabulary, text, count, options):
    """NumPy's keys and values of the prompt's first count - 1 positions ([layers][positions][kv dim] each) and the
    logits of its last token, and the tokens"""
    numpy = Llama(data, vocabulary, **options)
    if text:
        tokens = ([numpy.bos] + list(numpy.tokenizer.encode(text)))[:count]
    else:
        tokens = [numpy.bos] + [int(t) for t in np.random.default_rng(1).integers(3, numpy.vocab_size, count - 1)]
    assert len(tokens) == count, f"the text is {len(tokens)} tokens long"
    for pos, token in enumerate(tokens[:-1]):
        numpy.forward(token, pos, need_logits=False)
    logits = numpy.forward(tokens[-1], count - 1)
    n = count - 1
    kv = lambda cache: np.ascontiguousarray(cache[:, :, :n, :].transpose(0, 2, 1, 3).reshape(numpy.n_layers, n, -1), dtype=np.float32)
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
    return {"tokens": tokens, "logits": b64(logits), "keys": b64(kv(numpy.key_cache)), "values": b64(kv(numpy.value_cache)),
            "header": list(struct.unpack_from("<7i", data, 0))}
`);

const TEXTS = {
  english: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. " +
    "One day, she saw a big red ball under a tree. She ran to the ball and kicked it high into the sky, and everyone laughed.",
  japanese: "富士山は静岡県と山梨県にまたがる活火山で、標高三七七六メートルの日本最高峰である。古くから信仰の対象とされ、" +
    "多くの和歌や絵画に描かれてきた。二〇一三年には世界文化遺産に登録され、毎年夏には多くの登山者が山頂を目指す。",
};
const cases = [];
const directory = path.join(root, ".tmp", "gpu-check");
fs.mkdirSync(directory, { recursive: true });
for (const id of ids) {
  let options, text;
  if (id === "synthetic") {
    py.runPython(`data, vocabulary = synthetic()`);
    options = { dtype: "int8" };
  } else {
    const entry = MODELS.find((m) => m.id === id);
    if (!entry) throw new Error(`no model ${id} in src/models.js`);
    py.FS.writeFile("model.bin", fs.readFileSync(root + entry.checkpoint));
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
    py.runPython(`data, vocabulary = open("model.bin", "rb").read(), open("tokenizer.bin", "rb").read()`);
    options = entry.options;
    text = /日本語/.test(entry.note) ? TEXTS.japanese : TEXTS.english;
  }
  if (options.dtype !== "int8") throw new Error(`${id} is ${options.dtype}: the GPU takes int8 weights`);
  py.globals.set("OPTIONS", py.toPy(options));
  py.globals.set("TEXT", text ?? "");
  const reference = py.runPython(`answer(data, vocabulary, TEXT, ${COUNT}, OPTIONS)`).toJs({ dict_converter: Object.fromEntries });
  // the plan forward.js gets from Python, recorded: Llama(external=) with a start() that keeps it
  let plan;
  const bytes = py.runPython("data").toJs();
  py.globals.set("recorder", {
    size: bytes.length,
    read: (offset, length) => bytes.slice(offset, offset + length),
    start: (given) => {
      plan = given.toJs({ dict_converter: Object.fromEntries });
      return { backend: "recorded", bind() {}, forward() {}, release() {} };
    },
  });
  py.runPython(`Llama(None, vocabulary, kernels="simdkernel.so", external=recorder, **OPTIONS).release()`);
  plan.kv_start = KV_START;
  for (const [name, value] of Object.entries(plan.derived)) plan.derived[name] = Buffer.from(value).toString("base64");
  const file = path.join(directory, `${id}.bin`);
  fs.writeFileSync(file, bytes);
  cases.push({ id, plan, checkpoint: `/case/${id}.bin`, file, reference });
}

// ---- the browser: a page that is cross-origin isolated (its own headers), a worker that runs forward.js
const HARNESS = /* js */ `
const search = "?v=gpu-check";
const { compileKernels, createForward, weightsMemory, footprint } = await import("/public/forward.js" + search);
const fetched = async (url) => new Uint8Array(await (await fetch(url)).arrayBuffer());
const b64 = (floats) => {
  const bytes = new Uint8Array(floats.buffer, floats.byteOffset, floats.byteLength);
  let text = "";
  for (let i = 0; i < bytes.length; i += 32768) text += String.fromCharCode(...bytes.subarray(i, i + 32768));
  return btoa(text);
};
const openGpu = () => new Worker("/public/gpu.js" + search, { type: "module" });
try {
  const kernels = compileKernels(await fetched("/public/simdkernel_shared.wasm"), await fetched("/public/simdkernel_relaxed_shared.wasm"));
  const results = [];
  for (const c of await (await fetch("/cases.json")).json()) {
    const plan = c.plan;
    for (const name of Object.keys(plan.derived)) plan.derived[name] = Uint8Array.from(atob(plan.derived[name]), (ch) => ch.charCodeAt(0));
    const checkpoint = await fetched(c.checkpoint), size = checkpoint.length, tokens = c.reference.tokens, n = tokens.length - 1;
    const { memory, base } = weightsMemory(size, { shared: true, after: footprint(c.reference.header, size, { dtype: "int8", halfKV: true, gpu: true, kvStart: plan.kv_start }) });
    new Uint8Array(memory.buffer, base, size).set(checkpoint);
    const run = async (gpu) => {
      const engine = createForward({ memory, base, size, kernels, plan, gpu });
      const note = gpu ? await engine.gpu : undefined;
      const began = performance.now();
      engine.forwardMany(tokens.slice(0, -1), 0);
      const promptMs = performance.now() - began, gpuTokens = engine.gpuTokens;
      engine.forward(tokens[n], n);
      const logits = engine.logits().slice(), { keys, values } = engine.keysAndValues(0, n);
      const out = { note, promptMs, gpuTokens, logits: b64(logits), keys: b64(keys), values: b64(values) };
      if (gpu) {
        // the same prompt again from position 0: the GPU's keys and values of the first run are written over
        engine.newGeneration();
        engine.forwardMany(tokens.slice(0, -1), 0);
        engine.forward(tokens[n], n);
        out.again = { gpuTokens: engine.gpuTokens, logits: b64(engine.logits().slice()) };
        // a block that begins past what the GPU holds (the CPU wrote position n) goes to the CPU
        engine.newGeneration();
        engine.forwardMany(tokens.slice(0, 2), n + 1);
        out.past = { gpuTokens: engine.gpuTokens };
      }
      engine.release();
      return out;
    };
    results.push({ id: c.id, cpu: await run(undefined), gpu: await run(openGpu) });
  }
  postMessage({ results });
} catch (error) {
  postMessage({ error: String(error?.stack ?? error) });
}
`;
// an icon of its own: a browser without one asks for /favicon.ico (a 404 in the console of the real Chrome)
const PAGE = `<!doctype html><meta charset="utf-8"><title>gpu-check</title><link rel="icon" href="data:,"><script type="module">
const worker = new Worker("/harness.js", { type: "module" });
worker.onmessage = ({ data }) => { window.__gpuCheck = data; };
worker.onerror = (event) => { window.__gpuCheck = { error: event.message ?? "the harness did not start" }; };
</script>`;
const types = { ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".html": "text/html; charset=utf-8" };
const server = http.createServer((req, res) => {
  const pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  const headers = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
  const send = (type, body) => {
    res.writeHead(200, { ...headers, "Content-Type": type });
    res.end(body);
  };
  const found = cases.find((c) => c.checkpoint === pathname);
  if (pathname === "/") return send(types[".html"], PAGE);
  if (pathname === "/harness.js") return send(types[".js"], HARNESS);
  if (pathname === "/cases.json") return send(types[".json"], JSON.stringify(cases.map(({ file, ...c }) => c)));
  if (found) return send("application/octet-stream", fs.readFileSync(found.file));
  const file = path.join(root, "public", pathname.replace(/^\/public\//, ""));
  if (!pathname.startsWith("/public/") || !fs.existsSync(file)) {
    res.writeHead(404, headers);
    return res.end();
  }
  send(types[path.extname(file)] ?? "application/octet-stream", fs.readFileSync(file));
}).listen(0);

// Chromium's WebGPU without a GPU: SwiftShader (as tests/bench-check.mjs has it)
const WEBGPU = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-webgpu-adapter=swiftshader"];
const browser = await playwright.chromium.launch({ ...(engine === "chromium" ? {} : { channel: engine }), args: WEBGPU });
const page = await browser.newPage();
const lines = [];
page.on("console", (message) => lines.push(`[${message.type()}] ${message.text()}`));
page.on("pageerror", (error) => lines.push(`[pageerror] ${error.message}`));
await page.goto(`http://localhost:${server.address().port}/`);
await page.waitForFunction(() => window.__gpuCheck, null, { timeout: 1800000 });
const outcome = await page.evaluate(() => window.__gpuCheck);
await Promise.race([browser.close(), new Promise((resolve) => setTimeout(resolve, 15000))]);
server.close();
if (lines.length) console.log(lines.join("\n"));
if (outcome.error) {
  console.error(`FAILED\n- ${outcome.error}`);
  process.exit(1);
}

// ---- the comparisons
const floats = (b64) => {
  const bytes = Buffer.from(b64, "base64");
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
};
// the worst row (width values) of got against want: the largest difference over the row's largest value
function worstRow(got, want, width) {
  let worst = 0;
  for (let at = 0; at < want.length; at += width) {
    let largest = 0, difference = 0;
    for (let i = at; i < at + width; i++) {
      largest = Math.max(largest, Math.abs(want[i]));
      difference = Math.max(difference, Math.abs(got[i] - want[i]));
    }
    worst = Math.max(worst, difference / (largest || 1));
  }
  return worst;
}
const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);
let failed = false;
for (const { id, cpu, gpu } of outcome.results) {
  const c = cases.find((entry) => entry.id === id), ref = c.reference, n = ref.tokens.length - 1;
  const [dim, , layers, heads, kvHeads] = ref.header, kvDim = (dim / heads) * kvHeads;
  const failures = [];
  if (gpu.note !== "prompts on WebGPU") failures.push(`the GPU did not take it: ${gpu.note}`);
  if (gpu.gpuTokens !== n || gpu.again?.gpuTokens !== n) failures.push(`the GPU took ${gpu.gpuTokens} and ${gpu.again?.gpuTokens} of ${n} tokens`);
  if (gpu.past?.gpuTokens !== 0) failures.push(`a block past the GPU's keys and values went to the GPU (${gpu.past?.gpuTokens} tokens)`);
  const kv = (run) => Math.max(worstRow(floats(run.keys), floats(ref.keys), kvDim), worstRow(floats(run.values), floats(ref.values), kvDim));
  const gpuKv = kv(gpu), cpuKv = kv(cpu);
  if (!(gpuKv <= GPU_LINE)) failures.push(`the keys and values of the GPU are ${gpuKv.toExponential(2)} from NumPy's (line ${GPU_LINE})`);
  const want = floats(ref.logits), largest = want.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
  const logitsError = (b64) => floats(b64).reduce((m, x, i) => Math.max(m, Math.abs(x - want[i])), 0) / largest;
  const cpuLogits = logitsError(cpu.logits), gpuLogits = logitsError(gpu.logits), againLogits = logitsError(gpu.again.logits);
  if (!(gpuLogits <= LOGITS_LINE * cpuLogits) || !(againLogits <= LOGITS_LINE * cpuLogits)) {
    failures.push(`the logits on the GPU's keys and values are ${gpuLogits.toExponential(2)} and ${againLogits.toExponential(2)} from NumPy's, the CPU's ${cpuLogits.toExponential(2)}`);
  }
  const best = argmax(floats(cpu.logits));
  if (argmax(floats(gpu.logits)) !== best || argmax(floats(gpu.again.logits)) !== best) failures.push("another most likely token than on the CPU");
  console.log(`${id} (${layers} layers, ${heads} heads, ${kvHeads} of keys and values, ${n} tokens): keys and values from NumPy's: ` +
    `GPU ${gpuKv.toExponential(2)}, CPU ${cpuKv.toExponential(2)}; logits: GPU ${gpuLogits.toExponential(2)} (again ${againLogits.toExponential(2)}), ` +
    `CPU ${cpuLogits.toExponential(2)}, most likely ${best} ${argmax(want) === best ? "as NumPy's" : `(NumPy's ${argmax(want)})`}; ` +
    `the prompt ${(gpu.promptMs / n).toFixed(2)} ms a token on the GPU (${gpu.note}), ${(cpu.promptMs / n).toFixed(2)} on the CPU` +
    (failures.length ? ` — FAILED\n  - ${failures.join("\n  - ")}` : ""));
  failed ||= failures.length > 0;
}
process.exit(failed ? 1 : 0);
