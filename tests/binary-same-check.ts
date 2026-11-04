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
lines = []
for label, disable in (("relaxed SIMD", ()), ("without relaxed SIMD", ("relaxed",))):
    a = kernel_llama_file(TERNARY["file"], vocabulary, **dict(TERNARY["options"], disable=disable))
    b = kernel_llama_file(BINARY["file"], vocabulary, **dict(BINARY["options"], disable=disable))
    assert "binary" in b.backend or "ternary" not in b.backend, b.backend
    rng = np.random.default_rng(634)
    tokens = [a.bos] + rng.integers(3, a.vocab_size, POSITIONS - 1).tolist()
    for pos, token in enumerate(tokens):
        x, y = (np.asarray(m.forward(token, pos), dtype=np.float32).copy() for m in (a, b))
        if not np.array_equal(x.view(np.uint32), y.view(np.uint32)):
            raise AssertionError(f"{label}: the logits at position {pos} differ, by {float(np.abs(x - y).max())} at most")
    # (and a block: the prompt's tiles, from position 0 again)
    blocks = a.forward_many is not None and b.forward_many is not None
    if blocks:
        for m in (a, b):
            m.forward_many(tokens[:-1], 0)
        x, y = (np.asarray(m.forward(tokens[-1], len(tokens) - 1), dtype=np.float32).copy() for m in (a, b))
        if not np.array_equal(x.view(np.uint32), y.view(np.uint32)):
            raise AssertionError(f"{label}: the logits after a block differ, by {float(np.abs(x - y).max())} at most")
    lines.append(f"binary-same-check: {b.backend} and {a.backend} ({label}): the logits the same to the bit at {len(tokens)} positions{' and after a block of ' + str(len(tokens) - 1) if blocks else ''}")
    a.release(); b.release(); del a, b; gc.collect()
lines
`).toJs();
for (const line of lines) console.log(line);
