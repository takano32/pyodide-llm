// T357.3: the references of tests/gpu-check.ts, a part of that tool: NumPy's answers for a model (the Python below,
// in Pyodide, or in the native Python for a model of <prefix>.json) and the plan forward.js gets from Python, together
// the case the harness runs. What it reads is given: the Pyodide (referencePython() makes it), the directory the
// cases' checkpoints are written into, a model's id.
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { pyodideWithEngine } from "../engine.ts";
import { MODELS } from "../../src/models.ts";
import { python as sources } from "../tree.ts";
import { SYNTHETIC } from "./models.ts";
import { COUNT, KV_START, GEN, NUDGES } from "./lines.ts";

const root = new URL("../../", import.meta.url).pathname;

const PYTHON = `
import base64, gc, struct, numpy as np, llama2_numpy, llama2_convert
import engine.tensors
from engine.dtypes import KINDS
from llama2_numpy import Llama, external_tensors

def read_in_place(read):
    """T334.5: a table every layer reads a row of (a Gemma 4's, engine/layout.py's ROWS), held as the file has it: views
    of the checkpoint in place of the copy engine/tensors.py's read_tensor() makes, so that the real E2B's (2.6 GB of
    int8, in a file the native Python maps: referenceCase() below) is no second copy in a runner of 16 GB. The same
    values: the copy is for a checkpoint that may be let go, which this one is not while a Llama reads it"""
    def reading(checkpoint, place, whole=True, copy=False):
        # (the table's own row first: a stack of layers of their own kinds has a kind of each, a tuple, which KINDS has
        # not: synthetic-q4k's, the full suite's, run 44633537717)
        if whole or place.row.name != "per_layer_table":
            return read(checkpoint, place, whole, copy)
        kind = KINDS[place.kind] if place.kind is not None else None
        if kind is None or kind.name != "int8":
            return read(checkpoint, place, whole, copy)
        shape = place.row.shape
        raw = np.frombuffer(checkpoint, dtype=np.uint8, count=place.scales - place.offset, offset=place.offset)
        scales = np.frombuffer(checkpoint, dtype=kind.scale_type, count=(place.offset + place.size - place.scales) // kind.scale, offset=place.scales)
        return kind.unpack(raw).reshape(*shape[:-1], -1, place.group), scales.reshape(*shape[:-1], -1, 1)
    return reading

engine.tensors.read_tensor = read_in_place(engine.tensors.read_tensor)

# T674: the generator of a run that rounds numbers a millionth off (answer(nudge=)), None for one that rounds them as
# they are. What such runs differ by from NumPy's is how far a rounding lands from it when another order of float32
# sums moves its numbers a little: the spread of the one sample E16, Q8 and H16 were (lines.ts). The same way as the
# reference of Saluki 27B's (tests/reference_saluki_numpy.py's nudged())
NUDGE = [None]

def nudged(x, groups=False):
    """x moved by a random amount of up to a millionth before it is rounded: of each value's size, or (groups: Q8's 8
    bits a group of 32) of its group's largest, the step of that rounding. x itself where no run nudges"""
    rng = NUDGE[0]
    x = np.asarray(x, dtype=np.float32)
    if rng is None:
        return x
    if groups:
        blocks = x.reshape(-1, 32)
        size = np.abs(blocks).max(axis=1, keepdims=True)
        return (blocks + rng.uniform(-1e-6, 1e-6, blocks.shape).astype(np.float32) * size).reshape(x.shape)
    return x + rng.uniform(-1e-6, 1e-6, x.shape).astype(np.float32) * np.abs(x)

class Half(np.ndarray):
    """T153: a cache that keeps what is written into it rounded to float16 (the GPU's and the CPU's cache, T110)"""
    def __setitem__(self, key, value):
        super().__setitem__(key, nudged(value).astype(np.float16).astype(np.float32))

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
        x = nudged(x, groups=True)
        groups = x.reshape(-1, 32)
        scale = np.abs(groups).max(axis=1, keepdims=True) / np.float32(127)
        inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale > 0)
        return np.asarray(self).view(np.ndarray) @ (np.rint(groups * inverse) * scale).reshape(x.shape)

class H16(np.ndarray):
    """T562: a layer's matrix whose weights (weight × scale) are rounded to float16 once, and whose product takes its
    input rounded to float16, with float32 sums: the arithmetic of llama.cpp's f16 tiles (every weight × scale and
    activation in the workgroup's memory as f16, shaders/prompt.ts's regTile), rounded to the nearest as SwiftShader
    and lavapipe round"""
    def __matmul__(self, x):
        x = nudged(x).astype(np.float16).astype(np.float32)
        return np.asarray(self).view(np.ndarray) @ x

# the matrices a layer multiplies by (T545: an LFM2's in and out; T271.2: a Qwen3.5's of q, k and v, of z and out, and
# its gate of the attention's output; T334.1: a Gemma 4's of its wide heads and wider FFN, and its layers' gates and
# projections of their inputs of a table of their own)
MATRICES = ("wq", "wk", "wv", "wo", "w1", "w2", "w3", "win", "wout", "wqkv", "wz", "wg", "wq_wide", "wk_wide", "wv_wide", "wo_wide",
            "w1_double", "w2_double", "w3_double", "per_layer_projection", "ple_gate", "ple_proj")

def answer(data, vocabulary, text, count, options, half=False, q8=False, h16=False, nudge=None):
    """NumPy's keys and values of the prompt's first count - 1 positions ([layers][positions][kv dim] each) and the
    logits of its last token, and the tokens. half (T153): the cache rounded to float16, and nothing else (made whole
    at first, so that it never grows into an array of another class). q8 (T187): the inputs of the layers' matrices
    quantized to 8 bits (Q8). h16 (T562): the layers' matrices and their inputs rounded to float16 (H16). nudge (T674):
    the seed of a run whose roundings (the cache's, Q8's, H16's inputs') take numbers a millionth off, nudged()"""
    NUDGE[0] = None if nudge is None else np.random.default_rng(nudge)
    try:
        return answered(data, vocabulary, text, count, options, half, q8, h16)
    finally:
        NUDGE[0] = None

def answered(data, vocabulary, text, count, options, half, q8, h16):
    numpy = Llama(data, vocabulary, **options)
    if h16:
        for name in MATRICES:
            if getattr(numpy, name, None) is not None:
                # (T334.5: rounded in place where the array is the Llama's own, so that the real E2B's 1.5 GB stack of
                # its wider FFN has no second float32 copy beside it)
                matrix = np.asarray(getattr(numpy, name), dtype=np.float32)
                if matrix.flags.writeable:
                    matrix[...] = matrix.astype(np.float16)
                else:
                    matrix = matrix.astype(np.float16).astype(np.float32)
                setattr(numpy, name, matrix.view(H16))
    if q8:
        # (T545: an LFM2's matrices in and out too; T271.2: a Qwen3.5's of q, k and v, of z and out, and its gate of the
        # attention's output; T334.1: a Gemma 4's of its wide heads and wider FFN, and its layers' gates and projections
        # of their inputs)
        for name in MATRICES:
            if getattr(numpy, name, None) is not None:
                setattr(numpy, name, np.asarray(getattr(numpy, name)).view(Q8))
    if half and isinstance(numpy.key_cache, list):
        # (T334.1: a Gemma 4's, a layer's of its own heads' size, none of a layer that reads another's)
        numpy.key_cache = [None if each is None else np.zeros((numpy.n_kv_heads, numpy.seq_len, each.shape[2]), dtype=np.float32).view(Half) for each in numpy.key_cache]
        numpy.value_cache = [None if each is None else np.zeros_like(each).view(Half) for each in numpy.key_cache]
    elif half:
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
    # (T334.1: a Gemma 4's cache is a list, an array a layer of its own heads' size and None for a layer that reads
    # another's: [layer][position][its own width] of the layers that keep theirs, one after another, as the page's
    # engine reads its own back; widths, each one's)
    own = [each for each in numpy.key_cache if each is not None] if isinstance(numpy.key_cache, list) else None
    kv = lambda cache, at=0, m=n: (np.concatenate([np.ascontiguousarray(each[:, at:at + m, :].transpose(1, 0, 2), dtype=np.float32).ravel() for each in cache if each is not None])
                                   if isinstance(cache, list) else np.ascontiguousarray(cache[:, :, at:at + m, :].transpose(0, 2, 1, 3).reshape(len(cache), m, -1), dtype=np.float32))
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
    out = {"tokens": tokens, "logits": b64(logits), "keys": b64(kv(numpy.key_cache)), "values": b64(kv(numpy.value_cache)),
           "header": list(struct.unpack_from("<7i", data, 0)), "attending": len(own) if own is not None else len(numpy.key_cache),
           **({"widths": [each.shape[0] * each.shape[2] for each in own]} if own is not None else {})}
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
    # (T562: and of answer(half=True, h16=True), the arithmetic of llama.cpp's f16 tiles)
    halves = answer(data, vocabulary, text, count, options, half=True, h16=True)
    gc.collect()
    # (T674: and each of the three again, NUDGES times (lines.ts), rounding numbers a millionth off)
    nudges = []
    for seed in range(1, ${NUDGES} + 1):
        each = {}
        for name, rounding in (("16", {}), ("8", {"q8": True}), ("H", {"h16": True})):
            run = answer(data, vocabulary, text, count, options, half=True, nudge=seed, **rounding)
            gc.collect()
            each["keys" + name], each["values" + name] = run["keys"], run["values"]
        nudges.append(each)
    return {**exact, "keys16": rounded["keys"], "values16": rounded["values"], "logits16": rounded["logits"],
            "keys8": eight["keys"], "values8": eight["values"], "logits8": eight["logits"], "keysH": halves["keys"], "valuesH": halves["values"],
            "nudged": nudges}
`;
// Pyodide with the engine as the page has it, the Python above and the made-up models' maker
async function referencePython() {
  const { pyodide: py } = await pyodideWithEngine();
  py.runPython(PYTHON);
  // the made-up models are tests/made_up.py's quantized() (T357.2: the maker the unit tests have, laid out by the engine's
  // rows): the forms of models.ts's SYNTHETIC are its arguments
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
// src/models.ts. language (T334.5): the text of a <prefix>.json's prompt, one of TEXTS (gpu-check --text)
function referenceCase(py, directory, id, language = "english") {
  if (!TEXTS[language]) throw new Error(`no text ${JSON.stringify(language)}: ${Object.keys(TEXTS).join(" or ")}`);
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
    text = TEXTS[language].repeat(3);
    const native = spawnSync(process.env.PYTHON ?? "python3", ["-c", `import sys, json\nsys.path.insert(0, ${JSON.stringify(sources())})\n${PYTHON}
import mmap
# (T334.5: the checkpoint mapped, not read: the real E2B's is 5.2 GB, whose pages the runner can let go and read again)
with open(sys.argv[1] + ".bin", "rb") as file:
    data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
vocabulary = open(sys.argv[1] + ".tokenizer.bin", "rb").read()
print(json.dumps(answers(data, vocabulary, sys.argv[2], ${COUNT}, json.load(open(sys.argv[1] + ".json")))))`, prefix, text],
      { encoding: "utf8", maxBuffer: 1 << 30, stdio: ["ignore", "pipe", "inherit"] });
    if (native.status !== 0) throw new Error(`NumPy's answer for ${id} failed (${native.status ?? native.signal})`);
    const numpySeconds = (performance.now() - began) / 1000;
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${prefix}.tokenizer.bin`));
    py.runPython(`vocabulary = open("tokenizer.bin", "rb").read()`);
    // (T334.5: the checkpoint read where it is, not whole into Node: the real E2B's 5.2 GB pass readFileSync()'s 2 GiB)
    return { ...caseOf(py, directory, id, options, JSON.parse(native.stdout), `${prefix}.bin`), numpySeconds };
  } else {
    const entry = MODELS.find((m) => m.id === id);
    if (!entry) throw new Error(`no model ${id} in src/models.ts`);
    py.FS.writeFile("model.bin", fs.readFileSync(root + entry.checkpoint));
    py.FS.writeFile("tokenizer.bin", fs.readFileSync(root + entry.tokenizer));
    py.runPython(`data, vocabulary = open("model.bin", "rb").read(), open("tokenizer.bin", "rb").read()`);
    options = entry.options;
    text = (/日本語/.test(entry.note) ? TEXTS.japanese : TEXTS.english).repeat(3);
  }
  if (!["int8", "int6", "ternary", "binary"].includes(options.dtype)) throw new Error(`${id} is ${options.dtype}: the GPU takes int8, int6, ternary and binary weights`);
  py.globals.set("OPTIONS", py.toPy(options));
  py.globals.set("TEXT", text ?? "");
  // (T560: a made-up model's run may say its own count of tokens)
  const reference = py.runPython(`answers(data, vocabulary, TEXT, ${run.count ?? COUNT}, OPTIONS)`).toJs({ dict_converter: Object.fromEntries });
  const numpySeconds = (performance.now() - began) / 1000;
  return { ...caseOf(py, directory, id, options, reference, py.runPython("data").toJs()), numpySeconds, ...run };
}
// the plan forward.js gets from Python (the vocabulary in Pyodide's globals), recorded: Llama(external=) with a start()
// that keeps it, and the case the browser runs. checkpoint: its bytes, or (T334.5) the file that holds them, read a
// stretch at a time and served from where it is
function caseOf(py, directory, id, options, reference, checkpoint) {
  const began = performance.now();
  let plan;
  const fd = typeof checkpoint === "string" ? fs.openSync(checkpoint, "r") : null;
  const size = fd === null ? checkpoint.length : fs.fstatSync(fd).size;
  const read = (offset, length) => {
    if (fd === null) return checkpoint.slice(offset, offset + length);
    const out = new Uint8Array(length);
    // (T334.5's review: readSync reads less than asked near 2 GiB and says so only by its count)
    const got = fs.readSync(fd, out, 0, length, offset);
    if (got !== length) throw new Error(`gpu-check: read ${got} of ${length} bytes at ${offset} of the checkpoint`);
    return out;
  };
  py.globals.set("OPTIONS", py.toPy(options));
  py.globals.set("recorder", {
    size,
    read,
    start: (given) => {
      plan = given.toJs({ dict_converter: Object.fromEntries });
      return { backend: "recorded", bind() {}, forward() {}, release() {} };
    },
  });
  py.runPython(`Llama(None, vocabulary, kernels="simdkernel.so", external=recorder, **OPTIONS).release()`);
  plan.kv_start = KV_START;
  // T156: a Llama of int8 whose steps the GPU takes goes on the GPU alone too: where its tensors are from its header
  // (T226: Qwen2 and Qwen3 with it; T232: and of ternary weights, T634.3 of binary ones; T271.4: and a Qwen3.5, whose form the GPU alone's plan
  // is made with: options, in the case)
  let places;
  // (T334.3: and a Gemma 3, a Gemma 4 and a PLaMo 3, whose GELU gate (a PLaMo 3's is a SiLU's) and windows are the GPU
  // alone's too; a Gemma 4's table of every layer's inputs stays apart from it)
  if (["llama", "qwen35", "gemma3", "gemma4", "plamo3"].includes(options.arch ?? "llama") && ["int8", "ternary", "binary"].includes(options.dtype)) {
    py.globals.set("HEADER", py.toPy([...new Int32Array(read(0, 28).buffer)]));
    places = py.runPython(`external_tensors(HEADER, OPTIONS["dtype"], OPTIONS)`).toJs({ dict_converter: Object.fromEntries });
  }
  for (const [name, value] of Object.entries(plan.derived)) plan.derived[name] = Buffer.from(value).toString("base64");
  const name = path.basename(id), file = fd === null ? path.join(directory, `${name}.bin`) : checkpoint;
  if (fd === null) fs.writeFileSync(file, checkpoint);
  else fs.closeSync(fd);
  return { id, plan, places, size, options, headDim: options.head_dim ?? 0, arch: options.arch ?? "llama", dtype: options.dtype, checkpoint: `/case/${name}.bin`, file, reference,
    planSeconds: (performance.now() - began) / 1000 };
}

export { PYTHON, referencePython, referenceCase };
