// ternary-compare.mjs (T231): the same ternary model as the ternary dtype and as int8 (its weights widened, which is
// how T235 ran it), both on the page's forward pass (public/forward.js), on the same tokens of a text: how far apart
// their logits are, how often their most likely token is the same, and the perplexity of each.
//
// What the two share and what they do not. The weights are the same values: ternary holds d, int8 127 times
// float32(d / 127), within 6e-8 of d (T235). Against int8 without relaxed SIMD (matmul_q8) the activations are the
// same int8 too (quantize_x with no bias), so every row's integer sums are the same integers, and a row differs only
// by float32's rounding of its scales and of its sum (a few times 1e-7 of the largest term, each of the two is within
// 2e-6 of the exact sum: tests/ternary-bench.mjs). That is one matrix: over the layers a difference of an ulp moves an
// activation across a rounding step of the next matrix's input now and then, as any change of the order of the sums
// does to an int8 model (AGENTS.md), so no bound of the logits follows, and this measures them. Against int8 with
// relaxed SIMD (matmul_q8r, what the page ran) the activations differ as well: 7 bits there, 8 here.
//
//   node tests/ternary-compare.mjs <out of perplexity_prepare.py: ternary> <the same model's: int8> <text file> [tokens = 256]
//
// For CI (the 1.7B is 0.5 GB and 1.9 GB): tests/page_ternary.sh runs it.
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";

const [ternary, int8, textFile, count = "256"] = process.argv.slice(2);
const { pyodide: py } = await pyodideWithEngine();
const of = (out) => ({ file: path.resolve(`${out}.bin`), options: JSON.parse(fs.readFileSync(`${out}.json`, "utf8")) });
py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${ternary}.tokenizer.bin`));
py.globals.set("TERNARY", py.toPy(of(ternary)));
py.globals.set("INT8", py.toPy(of(int8)));
py.globals.set("TEXT", fs.readFileSync(textFile, "utf8"));
py.globals.set("TOKENS", Number(count));
const lines = py.runPython(`
import gc, math, numpy as np
vocabulary = open("tokenizer.bin", "rb").read()
lines = []
for label, disable in (("int8, 8-bit activations (matmul_q8)", ("relaxed",)), ("int8, 7-bit activations (matmul_q8r)", ())):
    a = kernel_llama_file(TERNARY["file"], vocabulary, **TERNARY["options"])
    b = kernel_llama_file(INT8["file"], vocabulary, **dict(INT8["options"], disable=disable))
    tokens = [a.bos] + a.tokenizer.encode(TEXT)[:TOKENS]
    agree, largest, apart, size, nll = 0, 0.0, 0.0, 0.0, [0.0, 0.0]
    for pos in range(len(tokens) - 1):
        x, y = (np.asarray(m.forward(tokens[pos], pos), dtype=np.float64).copy() for m in (a, b))
        largest = max(largest, float(np.abs(x - y).max()))
        apart, size = apart + float(((x - y) ** 2).sum()), size + float((y ** 2).sum())
        agree += int(x.argmax() == y.argmax())
        for i, logits in enumerate((x, y)):
            shifted = logits - logits.max()
            nll[i] -= shifted[tokens[pos + 1]] - math.log(np.exp(shifted).sum())
    n = len(tokens) - 1
    lines.append(f"ternary-compare: {a.backend} against {b.backend} ({label}), {n} tokens: the logits {math.sqrt(apart / size):.4f} of "
                 f"int8's apart, the largest difference {largest:.3f}, the most likely token the same at {100 * agree / n:.1f}%, "
                 f"perplexity {math.exp(nll[0] / n):.3f} against {math.exp(nll[1] / n):.3f} ({100 * (math.exp((nll[0] - nll[1]) / n) - 1):+.2f}%)")
    a.release(); b.release(); del a, b; gc.collect()
lines
`).toJs();
for (const line of lines) console.log(line);
