// chat_fluency.mjs (T251's and T252's review): what the first token the page puts in front of a chat does to the model, on answers no
// model wrote, on the page's engine (forward.js on the int8 kernels with 7-bit activations, tests/engine.mjs). It is part 1 of
// tests/chat_nll.py (which holds the original in float32 with transformers and cannot take a 7B on a runner) on the engine
// of the page: the 24 chat turns written by hand (tests/fixtures/chat-answers.jsonl: 12 Japanese, 12 English, a prompt and a
// short answer each), the negative log likelihood of the answer's tokens and of the end of its turn
//   A  after the format as the real template makes it (nothing in front: the format's own first token begins), and
//   B  after the same ids with the page's start in front of them (what the page sends).
// The measure is a person's text, not the model's own greedy answer: a greedy answer is the likelier under the context it was
// written from, so it never says that context is the worse one (tests/answer_check.mjs reads it, for what the start does to
// what the model writes). The lines are `chat_fluency <id>: ...` (ci.mjs's --grep "chat_fluency").
//
//   END=<id of the end of a turn> node tests/chat_fluency.mjs <out of tests/perplexity_prepare.py> <JSON of tests/write_options.py>
//
// END: <|im_end|> of a ChatML format (Qwen 151645, Hermes 128039), <|eot_id|> of Llama 3's (128009): the token the answer ends
// with. START=<id>: the token in front in B (else the options' bos). FIRST=<id>: for an entry whose BOS is its format's own first
// token and whose template begins after it (a Qwen3 8B's, Hermes 3's since T252's review): A is then [FIRST] + the format, the real
// ids, and START=<the BOS to compare, the converter's> goes in front of those in B. PICK="0,3,5": only these pairs. FIXTURE=<file>.
// For a byte-level BPE vocabulary (an answer is encoded alone, as chat_nll.py does, and that is the same ids as after the format only
// where the format ends in a newline). tests/write.sh runs it (WRITER=chat_fluency.mjs, its TOKENS unused).
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";
import { runtimeUrl } from "./tree.mjs";
const { footprint, needsWide } = await import(runtimeUrl("forward.js"));
import { filled } from "../src/models.js";
import { leave } from "./leave.mjs";

const [out, pageFile] = process.argv.slice(2);
const page = JSON.parse(fs.readFileSync(pageFile, "utf8"));
const fixture = process.env.FIXTURE ?? new URL("./fixtures/chat-answers.jsonl", import.meta.url).pathname;
let pairs = fs.readFileSync(fixture, "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
if (process.env.PICK) pairs = process.env.PICK.split(",").map((at) => pairs[Number(at)]);
if (!page.template) throw new Error(`${page.id} has no format: this is for a chat model`);
if (!process.env.END) throw new Error("END=<the id of the end of a turn> is needed");
const size = fs.statSync(`${out}.bin`).size;
const { arch, head_dim, dtype } = page.options;
const wide = needsWide(size, footprint(page.header, size, { dtype, arch, head_dim, halfKV: true, shared: true }));
const { pyodide } = await pyodideWithEngine({ wide });
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(`${out}.tokenizer.bin`));
pyodide.globals.set("CHECKPOINT", `${out}.bin`);
pyodide.globals.set("OPTIONS", pyodide.toPy(page.options));
pyodide.globals.set("PAIRS", pyodide.toPy(pairs.map((pair) => ({ text: filled(page.template, pair.prompt), answer: pair.answer }))));
pyodide.globals.set("START", process.env.START ? Number(process.env.START) : -1);
pyodide.globals.set("FIRST", pyodide.toPy(process.env.FIRST ? [Number(process.env.FIRST)] : []));
pyodide.globals.set("END", Number(process.env.END));
const result = JSON.parse(pyodide.runPython(`
import json, math, time
import numpy as np

llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **dict(OPTIONS))
start = llama.bos if START < 0 else START

def log_probs(prefix, answer):
    """the log probabilities of the answer's tokens, each given the prefix and the answer before it (the forward pass one token a time)"""
    row, scored = prefix + answer, []
    for pos, token in enumerate(row[:-1]):
        logits = llama.forward(token, pos, need_logits=pos >= len(prefix) - 1)
        if pos >= len(prefix) - 1:
            logits = np.array(logits, dtype=np.float64)
            shifted = logits - logits.max()
            scored.append(shifted[row[pos + 1]] - math.log(np.exp(shifted).sum()))
    return -float(np.sum(scored))

began = time.perf_counter()
rows = []
for number, pair in enumerate(PAIRS):
    natural = list(FIRST) + llama.tokenizer.encode(pair["text"], llama.specials)
    answer = llama.tokenizer.encode(pair["answer"], ()) + [END]
    a = log_probs(natural, answer)
    b = log_probs([start] + natural, answer)
    rows.append({"tokens": len(answer), "A": a, "B": b, "japanese": any(ord(c) > 0x2E80 for c in pair["answer"])})
    print(f"pair {number + 1}/{len(PAIRS)} done, {time.perf_counter() - began:.0f} s", flush=True)
json.dumps({"rows": rows, "bos": start, "seconds": time.perf_counter() - began})
`));
const id = page.id;
console.log(`chat_fluency ${id}: ${result.rows.length} answers written by hand, A the real format${process.env.FIRST ? ` (it begins with ${process.env.FIRST})` : " with nothing in front"}, B with ${result.bos} in front, ` +
  `the end of a turn ${process.env.END}, ${result.seconds.toFixed(0)} s`);
const report = (name, rows) => {
  if (!rows.length) return;
  const tokens = rows.reduce((sum, row) => sum + row.tokens, 0);
  const a = rows.reduce((sum, row) => sum + row.A, 0) / tokens, b = rows.reduce((sum, row) => sum + row.B, 0) / tokens;
  const differences = rows.map((row) => row.B / row.tokens - row.A / row.tokens);
  const mean = differences.reduce((sum, value) => sum + value, 0) / differences.length;
  const sd = Math.sqrt(differences.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (differences.length - 1));
  console.log(`chat_fluency ${id}: ${name} (${rows.length} answers, ${tokens} tokens): nll A ${a.toFixed(4)}, B ${b.toFixed(4)} (B - A ${(b - a >= 0 ? "+" : "") + (b - a).toFixed(4)}, ` +
    `perplexity ${((Math.exp(b - a) - 1) * 100 >= 0 ? "+" : "") + ((Math.exp(b - a) - 1) * 100).toFixed(2)}%); B is worse on ${differences.filter((value) => value > 0).length} of ${rows.length}; ` +
    `a pair's B - A ${(mean >= 0 ? "+" : "") + mean.toFixed(4)}, standard error ${(sd / Math.sqrt(rows.length)).toFixed(4)}`);
};
report("ja", result.rows.filter((row) => row.japanese));
report("en", result.rows.filter((row) => !row.japanese));
report("all", result.rows);
// (T357: what was written is out first: the answers of a long run are more than a pipe holds, tests/leave.mjs)
await leave(0);
