// The forward pass of public/forward.js (the page's) against the NumPy forward of llama2_numpy.py, on the models of
// this directory (make models kernels) or converted ones. Since T93 there is no other forward to hold it to.
//   float32: the kernels add in another order than NumPy, so the last bits differ; the most likely token must be the
//            same at every position, and no logit may differ by more than 1e-3 (measured 2026-09-25: 1.7e-5 to 3.7e-5).
//   int8: forward.js quantizes the activations too (7 bits with relaxed SIMD), NumPy does not, so the numbers
//         differ by design. Measured 2026-09-25 on NumPy's greedy text: at 128 positions the most likely token the
//         same at 93.8 to 100% and the perplexity -0.23 to +2.00% apart; at 64 positions llm-jp-3 150M was 87.5% and
//         +3.42% (fewer positions, more spread). The line: 85% or more and within 5%, at 128 positions (the
//         default). A real fault (a wrong order, a wrong scale) lands far outside: the agreement near nothing and
//         the perplexity a multiple.
// T108: the same text read by forward.js one token at a time and in blocks (forward_many), from a KV cache that
// starts small so that it grows within the blocks: the last logits must be the same to the bit.
// Then the speeds, both in turn. Runs in the deployment.
//
//   node tests/forward-check.mjs [model id | <out> of tests/perplexity_prepare.py ...] [--rounds 3] [--positions 128]
//        [--without relaxed,int8,sampler] [--plain] [--wide]
//
// The memory is shared, as the page's where it is cross-origin isolated; --plain: not shared, as the page's where it
// is not (the keys and values then stay float32, T110). --wide: a 64-bit memory and its kernels (T101), as the page
// has for a model past 4 GiB.
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const option = (name, value) => (args.includes(name) ? Number(args[args.indexOf(name) + 1]) : value);
const rounds = option("--rounds", 3), positions = option("--positions", 128);
const ids = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
const without = args.includes("--without") ? args[args.indexOf("--without") + 1].split(",") : [];
const modelOf = (id) => MODELS.find((m) => m.id === id) ?? { name: path.basename(id), checkpoint: path.resolve(`${id}.bin`),
  tokenizer: path.resolve(`${id}.tokenizer.bin`), options: JSON.parse(fs.readFileSync(`${id}.json`, "utf8")) };
const file = (f) => (path.isAbsolute(f) ? f : root + f);

// T98: six_sums (the corrections of int6 weights for matmul_q6r) against the sums of the int8 values the layout of
// llama2_numpy.pack6 holds, taken apart byte by byte here; the products rounded as forward.js rounds int8's
{
  const memory = new WebAssembly.Memory({ initial: 4 });
  const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/simdkernel_plain.wasm`)), { env: { memory } }).exports;
  const U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer);
  const groups = 2000, w = 4096, scales = w + groups * 24, out = scales + groups * 4;
  for (let i = 0; i < groups * 24; i++) U[w + i] = (Math.imul(i, 2654435761) >>> 7) & 255;
  for (let g = 0; g < groups; g++) F[scales / 4 + g] = Math.fround(0.001 + g * 1.37e-3);
  k.six_sums(out, w, scales, groups);
  for (let g = 0; g < groups; g++) {
    let sum = 0;
    for (let j = 0; j < 32; j++) {
      const at = w + g * 24, low = j < 16 ? U[at + j] & 15 : U[at + j - 16] >> 4;
      const top = (U[at + 16 + (j % 8)] >> (2 * ((j / 8) | 0))) & 3;
      sum += (((low | (top << 4)) << 2) << 24) >> 24;
    }
    if (Math.fround(F[scales / 4 + g] * sum) !== F[out / 4 + g]) throw new Error(`six_sums differs at group ${g}`);
  }
}

// T101: jobs.js says which arguments of each kernel are addresses (BigInt on a 64-bit memory): the same as the
// usize parameters of the kernels' source, every exported one
{
  const { ADDRESSES } = await import("../public/jobs.js");
  const source = {};
  for (const file of ["kernels/kernel.ts", "kernels/kernel_relaxed.ts"]) {
    for (const [, name, parameters] of fs.readFileSync(root + file, "utf8").matchAll(/export function (\w+)\(([^)]*)\)/g)) {
      source[name] = parameters.split(",").map((p, i) => [i, p.split(":")[1].trim()]).filter(([, type]) => type === "usize").map(([i]) => i);
    }
  }
  if (JSON.stringify(Object.keys(source).sort().map((n) => [n, source[n]])) !== JSON.stringify(Object.keys(ADDRESSES).sort().map((n) => [n, ADDRESSES[n]]))) {
    throw new Error("jobs.js's ADDRESSES is not the kernels' usize parameters");
  }
}

const { pyodide: py } = await pyodideWithEngine({ shared: !args.includes("--plain"), wide: args.includes("--wide") });
py.runPython("import time, gc, math, numpy as np\nfrom llama2_numpy import Llama");
let failed = false;
for (const id of ids.length ? ids : ["stories260K", "stories15M", "tiny-lm", "llm-jp-3-150m"]) {
  const entry = modelOf(id);
  py.FS.writeFile("model.bin", fs.readFileSync(file(entry.checkpoint)));
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(file(entry.tokenizer)));
  py.globals.set("OPTIONS", py.toPy({ ...entry.options, disable: without }));
  const verdict = py.runPython(`
data, vocabulary = open("model.bin", "rb").read(), open("tokenizer.bin", "rb").read()
page = kernel_llama(data, vocabulary, **OPTIONS)
numpy = Llama(data, vocabulary, **{k: v for k, v in OPTIONS.items() if k != "disable"})
int8 = "int8" in page.backend or "int6" in page.backend  # both quantize the activations (T98)
sequence, agree, largest, nll = [page.bos], 0, 0.0, [0.0, 0.0]
for pos in range(${positions}):
    a, b = page.forward(sequence[pos], pos).astype(np.float64), numpy.forward(sequence[pos], pos).astype(np.float64)
    largest = max(largest, float(np.abs(a - b).max()))
    agree += int(a.argmax() == b.argmax())
    following = int(b.argmax())  # NumPy's greedy text, which both read
    for i, logits in enumerate((a, b)):
        shifted = logits - logits.max()
        nll[i] -= shifted[following] - math.log(np.exp(shifted).sum())
    sequence.append(following)
agreement, change = agree / ${positions}, math.exp((nll[0] - nll[1]) / ${positions}) - 1
ok = (agreement >= 0.85 and abs(change) <= 0.05) if int8 else (agree == ${positions} and largest <= 1e-3)
import llama2_numpy
kv_start, llama2_numpy.KV_START = llama2_numpy.KV_START, 8
one, many = kernel_llama(data, vocabulary, **OPTIONS), kernel_llama(data, vocabulary, **OPTIONS)
llama2_numpy.KV_START = kv_start
fed = sequence[:${positions}]
for pos, token in enumerate(fed[:-1]):
    one.forward(token, pos, need_logits=False)
blocks = many.forward_many is not None
if blocks:
    for at in range(0, len(fed) - 1, 37):  # blocks that do not line up with forward.js's own BATCH
        many.forward_many(fed[at:min(at + 37, len(fed) - 1)], at)
else:
    for pos, token in enumerate(fed[:-1]):
        many.forward(token, pos, need_logits=False)
same = np.array_equal(one.forward(fed[-1], len(fed) - 1), many.forward(fed[-1], len(fed) - 1))
ok = ok and same
one.release(); many.release(); del one, many
def run(llama, positions):
    token, began = llama.bos, time.perf_counter()
    for pos in range(positions):
        token = int(np.argmax(llama.forward(token, pos)))
    return time.perf_counter() - began
(ok, f"{page.backend}: " + (f"most likely token the same at {agreement * 100:.1f}%, perplexity {change * 100:+.2f}% against NumPy"
     if int8 else f"most likely token the same at {agreement * 100:.1f}%, largest logit difference {largest:.2e} against NumPy")
     + (f"; the prompt in blocks {'the same to the bit' if same else 'DIFFERENT'}" if blocks else "; no blocks (NumPy)"))
`).toJs();
  const [ok, line] = verdict;
  const times = { numpy: [], page: [] };
  for (let r = 0; r < rounds; r++) for (const which of ["numpy", "page"]) times[which].push(positions / py.runPython(`run(${which}, ${positions})`));
  const median = (xs) => [...xs].sort((p, q) => p - q)[xs.length >> 1];
  console.log(`${entry.name}: ${line}${ok ? "" : " — FAILED"}; NumPy ${median(times.numpy).toFixed(1)} against forward.js ${median(times.page).toFixed(1)} tok/s`);
  failed ||= !ok;
  py.runPython("page.release(); del page, numpy; gc.collect()");
}
process.exit(failed ? 1 : 0);
