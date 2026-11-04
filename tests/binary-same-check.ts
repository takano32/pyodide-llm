// binary-same-check.ts (T.634): the same 1-bit model as the binary dtype and as the ternary one (the signs of every
// weight and one scale of its group of 128, as the 1-bit Bonsai's Q1_0 has them, which the page held as ternary before),
// both on the page's forward pass (src/runtime/forward.ts): the logits are the same to the bit, at every position, a
// token at a time and in blocks (forward_many), with relaxed SIMD and without. kernels/binary.ts says why: a binary
// weight's code is a ternary one's, so every group's integer sum is the same integer and every float32 operation after
// it the same in the same order.
//
//   node tests/binary-same-check.ts <out of tests/make_ternary.py: ternary ... signs> <the same model's: binary> [--positions 64]
//
// The light suite runs it on tests/make_ternary.py's made-up models (a Qwen3 and a hybrid in a rotated basis).
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.ts";

const args = process.argv.slice(2);
const [ternary, binary] = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
const positions = args.includes("--positions") ? Number(args[args.indexOf("--positions") + 1]) : 64;
const { pyodide: py } = await pyodideWithEngine();
const of = (out) => ({ file: path.resolve(`${out}.bin`), options: JSON.parse(fs.readFileSync(`${out}.json`, "utf8")) });
py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${ternary}.tokenizer.bin`));
py.globals.set("TERNARY", py.toPy(of(ternary)));
py.globals.set("BINARY", py.toPy(of(binary)));
py.globals.set("POSITIONS", positions);
const lines = py.runPython(`
import gc, numpy as np
vocabulary = open("tokenizer.bin", "rb").read()
assert TERNARY["options"]["dtype"] == "ternary" and BINARY["options"]["dtype"] == "binary", (TERNARY["options"]["dtype"], BINARY["options"]["dtype"])
# (T.634.2: one model at a time, its logits kept, so that a 27B's two checkpoints, 7.2 and 3.9 GB, are not in memory together)
def run(given, disable, tokens):
    m = kernel_llama_file(given["file"], vocabulary, **dict(given["options"], disable=disable))
    backend, out = m.backend, [np.asarray(m.forward(token, pos), dtype=np.float32).copy() for pos, token in enumerate(tokens)]
    block = None
    if m.forward_many is not None:
        m.forward_many(tokens[:-1], 0)
        block = np.asarray(m.forward(tokens[-1], len(tokens) - 1), dtype=np.float32).copy()
    m.release(); del m; gc.collect()
    return backend, out, block
lines = []
for label, disable in (("relaxed SIMD", ()), ("without relaxed SIMD", ("relaxed",))):
    rng = np.random.default_rng(634)
    tokens = None
    probe = kernel_llama_file(BINARY["file"], vocabulary, **dict(BINARY["options"], disable=disable))
    tokens = [probe.bos] + rng.integers(3, probe.vocab_size, POSITIONS - 1).tolist()
    probe.release(); del probe; gc.collect()
    (a_backend, a, a_block), (b_backend, b, b_block) = run(TERNARY, disable, tokens), run(BINARY, disable, tokens)
    assert "binary" in b_backend, b_backend
    for pos, (x, y) in enumerate(zip(a, b)):
        if not np.array_equal(x.view(np.uint32), y.view(np.uint32)):
            raise AssertionError(f"{label}: the logits at position {pos} differ, by {float(np.abs(x - y).max())} at most")
    blocks = a_block is not None and b_block is not None
    if blocks and not np.array_equal(a_block.view(np.uint32), b_block.view(np.uint32)):
        raise AssertionError(f"{label}: the logits after a block differ, by {float(np.abs(a_block - b_block).max())} at most")
    greedy = [int(x.argmax()) for x in b]
    lines.append(f"binary-same-check: {b_backend} and {a_backend} ({label}): the logits the same to the bit at {len(tokens)} positions{' and after a block of ' + str(len(tokens) - 1) if blocks else ''} (the most likely at the first eight: {greedy[:8]})")
lines
`).toJs();
for (const line of lines) console.log(line);
