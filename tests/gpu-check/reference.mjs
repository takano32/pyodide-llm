// T357.3: the references of tests/gpu-check.mjs, a part of that tool: NumPy's answers for a model (the Python below,
// in Pyodide, or in the native Python for a model of <prefix>.json) and the plan forward.js gets from Python, together
// the case the harness runs. What it reads is given: the Pyodide (referencePython() makes it), the directory the
// cases' checkpoints are written into, a model's id.
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { pyodideWithEngine } from "../engine.mjs";
import { MODELS } from "../../src/models.js";
import { python as sources } from "../tree.mjs";
import { SYNTHETIC } from "./models.mjs";
import { COUNT, KV_START, GEN } from "./lines.mjs";

const root = new URL("../../", import.meta.url).pathname;

const PYTHON = `
import base64, gc, struct, numpy as np, llama2_numpy, llama2_convert
from llama2_numpy import Llama, external_tensors

class Half(np.ndarray):
    """T153: a cache that keeps what is written into it rounded to float16 (the GPU's and the CPU's cache, T110)"""
    def __setitem__(self, key, value):
        super().__setitem__(key, np.asarray(value, dtype=np.float32).astype(np.float16).astype(np.float32))

    def __array_wrap__(self, array, context=None, return_scalar=False):
        # T183: what is computed from the cache is a plain array (the attention's scores and output, then the residual
        # stream), or its later writes round as well: without this, stories15M's keys at 40 tokens were 7.8e-3 from
        # the review's engine that rounds the cache where it writes it (.tmp/t153-review/ref16), with it the same
        array = np.asarray(array).view(np.ndarray)
        return array[()] if return_scalar else array

class Q8(np.ndarray):
    """T187: a layer's matrix whose product takes its input quantized to 8 bits a group of 32, as the CPU's quantize_x
    makes it for matmul_q8 and the GPU's QUANTIZE for ORT's DP4A: the largest |value| / 127 the group's scale, each
    value times 1 / scale rounded half to even, in float32"""
    def __matmul__(self, x):
        x = np.asarray(x, dtype=np.float32)
        groups = x.reshape(-1, 32)
        scale = np.abs(groups).max(axis=1, keepdims=True) / np.float32(127)
        inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale > 0)
        return np.asarray(self).view(np.ndarray) @ (np.rint(groups * inverse) * scale).reshape(x.shape)

def answer(data, vocabulary, text, count, options, half=False, q8=False):
    """NumPy's keys and values of the prompt's first count - 1 positions ([layers][positions][kv dim] each) and the
    logits of its last token, and the tokens. half (T153): the cache rounded to float16, and nothing else (made whole
    at first, so that it never grows into an array of another class). q8 (T187): the inputs of the layers' matrices
    quantized to 8 bits (Q8)"""
    numpy = Llama(data, vocabulary, **options)
    if q8:
        for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3", "win", "wout"):  # (T545: an LFM2's matrices in and out too)
            if getattr(numpy, name, None) is not None:
                setattr(numpy, name, np.asarray(getattr(numpy, name)).view(Q8))
    if half:
        # (T545: of the attending layers alone, as the engine keeps them: an LFM2's convolution layers keep none)
        numpy.key_cache = np.zeros((len(numpy.key_cache), numpy.n_kv_heads, numpy.seq_len, numpy.head_size), dtype=np.float32).view(Half)
        numpy.value_cache = np.zeros_like(numpy.key_cache).view(Half)
    if text:
        tokens = ([numpy.bos] + list(numpy.tokenizer.encode(text)))[:count]
    else:
        tokens = [numpy.bos] + [int(t) for t in np.random.default_rng(1).integers(3, numpy.vocab_size, count - 1)]
    assert len(tokens) == count, f"the text is {len(tokens)} tokens long"
    for pos, token in enumerate(tokens[:-1]):
        numpy.forward(token, pos, need_logits=False)
    logits = numpy.forward(tokens[-1], count - 1)
    n = count - 1
    kv = lambda cache, at=0, m=n: np.ascontiguousarray(cache[:, :, at:at + m, :].transpose(0, 2, 1, 3).reshape(len(cache), m, -1), dtype=np.float32)
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
    out = {"tokens": tokens, "logits": b64(logits), "keys": b64(kv(numpy.key_cache)), "values": b64(kv(numpy.value_cache)),
           "header": list(struct.unpack_from("<7i", data, 0)), "attending": len(numpy.key_cache)}
    if half:
        return out
    # T152: GEN greedy steps after the prompt: the ids, the logits of each step, the keys and values of their positions
    greedy, rows = [], [np.array(logits, dtype=np.float32)]
    for step in range(${GEN}):
        greedy.append(int(np.argmax(rows[-1])))
        if step < ${GEN} - 1:
            rows.append(np.array(numpy.forward(greedy[-1], count + step), dtype=np.float32))
    return {**out, "greedy": greedy, "greedyLogits": b64(np.stack(rows)), "greedyKeys": b64(kv(numpy.key_cache, n, ${GEN})),
            "greedyValues": b64(kv(numpy.value_cache, n, ${GEN}))}

def answers(data, vocabulary, text, count, options):
    """answer() and beside it: T153, the keys and values of answer(half=True); T187, its logits, and the keys, values
    and logits of answer(half=True, q8=True) (the arithmetic of ORT's DP4A on the GPU: 8-bit inputs, the cache in
    float16)"""
    # (one engine at a time: Llama holds cycles, and Qwen3 0.6B widened to float32 is 2.4 GB)
    exact = answer(data, vocabulary, text, count, options)
    gc.collect()
    rounded = answer(data, vocabulary, text, count, options, half=True)
    gc.collect()
    eight = answer(data, vocabulary, text, count, options, half=True, q8=True)
    gc.collect()
    return {**exact, "keys16": rounded["keys"], "values16": rounded["values"], "logits16": rounded["logits"],
            "keys8": eight["keys"], "values8": eight["values"], "logits8": eight["logits"]}
`;
// Pyodide with the engine as the page has it, the Python above and the made-up models' maker
async function referencePython() {
  const { pyodide: py } = await pyodideWithEngine();
  py.runPython(PYTHON);
  // the made-up models are tests/made_up.py's quantized() (T357.2: the maker the unit tests have, laid out by the engine's
  // rows): the forms of models.mjs's SYNTHETIC are its arguments
  py.FS.writeFile("made_up.py", fs.readFileSync(path.join(root, "tests", "made_up.py")));
  py.runPython("import made_up");
  return py;
}

// each text three times over: long enough for COUNT tokens of every model here
const TEXTS = {
  english: "Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. " +
    "One day, she saw a big red ball under a tree. She ran to the ball and kicked it high into the sky, and everyone laughed.",
  japanese: "富士山は静岡県と山梨県にまたがる活火山で、標高三七七六メートルの日本最高峰である。古くから信仰の対象とされ、" +
    "多くの和歌や絵画に描かれてきた。二〇一三年には世界文化遺産に登録され、毎年夏には多くの登山者が山頂を目指す。",
};
// NumPy's answers for one model and its case (caseOf, below): a made-up model's name, <prefix>.json, or a model of
// src/models.js
function referenceCase(py, directory, id) {
  let options, text, run = {};
  const began = performance.now();
  if (SYNTHETIC[id]) {
    const [form, engineOptions] = SYNTHETIC[id];
    run = SYNTHETIC[id][2] ?? {};
    py.globals.set("FORM", py.toPy(form));
    py.runPython(`data, vocabulary = made_up.quantized(**FORM)`);
    options = { dtype: "int8", ...engineOptions };
  } else if (id.endsWith(".json")) {
    // T153: NumPy's answer in the native Python ($PYTHON, python3 by default): Qwen3 0.6B widened to float32 is 2.4
    // GB, which with the file's copies went past Pyodide's 4 GB and a 7.5 GB scope of the development machine
    const prefix = id.slice(0, -".json".length);
    options = JSON.parse(fs.readFileSync(id, "utf8"));
    text = TEXTS.english.repeat(3);
    const native = spawnSync(process.env.PYTHON ?? "python3", ["-c", `import sys, json\nsys.path.insert(0, ${JSON.stringify(sources())})\n${PYTHON}
data, vocabulary = open(sys.argv[1] + ".bin", "rb").read(), open(sys.argv[1] + ".tokenizer.bin", "rb").read()
print(json.dumps(answers(data, vocabulary, sys.argv[2], ${COUNT}, json.load(open(sys.argv[1] + ".json")))))`, prefix, text],
      { encoding: "utf8", maxBuffer: 1 << 30, stdio: ["ignore", "pipe", "inherit"] });
    if (native.status !== 0) throw new Error(`NumPy's answer for ${id} failed (${native.status ?? native.signal})`);
    const numpySeconds = (performance.now() - began) / 1000;
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${prefix}.tokenizer.bin`));
    py.runPython(`vocabulary = open("tokenizer.bin", "rb").read()`);
    return { ...caseOf(py, directory, id, options, JSON.parse(native.stdout), new Uint8Array(fs.readFileSync(`${prefix}.bin`))), numpySeconds };
  } else {
    const entry = MODELS.find((m) => m.id === id);
    if (!entry) throw new Error(`no model ${id} in src/models.js`);
    py.FS.writeFile("model.bin", fs.readFileSync(root + entry.checkpoint));
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
    py.runPython(`data, vocabulary = open("model.bin", "rb").read(), open("tokenizer.bin", "rb").read()`);
    options = entry.options;
    text = (/日本語/.test(entry.note) ? TEXTS.japanese : TEXTS.english).repeat(3);
  }
  if (!["int8", "int6", "ternary"].includes(options.dtype)) throw new Error(`${id} is ${options.dtype}: the GPU takes int8, int6 and ternary weights`);
  py.globals.set("OPTIONS", py.toPy(options));
  py.globals.set("TEXT", text ?? "");
  // (T560: a made-up model's run may say its own count of tokens)
  const reference = py.runPython(`answers(data, vocabulary, TEXT, ${run.count ?? COUNT}, OPTIONS)`).toJs({ dict_converter: Object.fromEntries });
  const numpySeconds = (performance.now() - began) / 1000;
  return { ...caseOf(py, directory, id, options, reference, py.runPython("data").toJs()), numpySeconds, ...run };
}
// the plan forward.js gets from Python (the vocabulary in Pyodide's globals), recorded: Llama(external=) with a start()
// that keeps it, and the case the browser runs
function caseOf(py, directory, id, options, reference, bytes) {
  const began = performance.now();
  let plan;
  py.globals.set("OPTIONS", py.toPy(options));
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
  // T156: a Llama of int8 whose steps the GPU takes goes on the GPU alone too: where its tensors are from its header
  // (T226: Qwen2 and Qwen3 with it; T232: and of ternary weights)
  let places;
  if ((options.arch ?? "llama") === "llama" && ["int8", "ternary"].includes(options.dtype)) {
    py.globals.set("HEADER", py.toPy([...new Int32Array(bytes.slice(0, 28).buffer)]));
    places = py.runPython(`external_tensors(HEADER, OPTIONS["dtype"], OPTIONS)`).toJs({ dict_converter: Object.fromEntries });
  }
  for (const [name, value] of Object.entries(plan.derived)) plan.derived[name] = Buffer.from(value).toString("base64");
  const name = path.basename(id), file = path.join(directory, `${name}.bin`);
  fs.writeFileSync(file, bytes);
  return { id, plan, places, headDim: options.head_dim ?? 0, arch: options.arch ?? "llama", dtype: options.dtype, checkpoint: `/case/${name}.bin`, file, reference,
    planSeconds: (performance.now() - began) / 1000 };
}

export { PYTHON, referencePython, referenceCase };
