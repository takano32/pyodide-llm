// draft_check.mjs (T100, stage 1): would a small model's drafts be accepted by a large one? Both read the same
// tokens (they must share a vocabulary) through the engine as the page runs it (tests/engine.mjs), and at every
// position their most likely next tokens are compared. Two texts: a real one (T85's), and the large model's own
// greedy continuation of its first tokens, which is closer to what speculative decoding would have to guess.
//
// A draft of k tokens is accepted up to the first one the large model would not have chosen, and the large model's
// pass gives one token more. So with the runs of agreement measured here, one pass of the large model moves on by
//   mean over positions of (min(run from there, k) + 1)
// tokens, which is the gain before the cost of that pass (stage 2) and of the drafting.
//
//   node tests/draft_check.mjs <small> <large> <text file> [tokens = 1500] [greedy = 256]
//   (<small> and <large>: a model id of src/models.js, or the <out> of tests/perplexity_prepare.py)
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const [smallId, largeId, textFile, tokensArg = "1500", greedyArg = "256"] = process.argv.slice(2);
const modelOf = (id) => {
  const entry = MODELS.find((m) => m.id === id);
  if (entry) return { name: entry.name, checkpoint: root + entry.checkpoint, tokenizer: root + entry.tokenizer, options: entry.options };
  return { name: path.basename(id), checkpoint: `${id}.bin`, tokenizer: `${id}.tokenizer.bin`, options: JSON.parse(fs.readFileSync(`${id}.json`, "utf8")) };
};
const small = modelOf(smallId), large = modelOf(largeId);
const { pyodide: py } = await pyodideWithEngine();
for (const [prefix, model] of [["small", small], ["large", large]]) {
  py.globals.set(`${prefix.toUpperCase()}_CHECKPOINT`, path.resolve(model.checkpoint));
  py.FS.writeFile(`${prefix}.tokenizer.bin`, fs.readFileSync(model.tokenizer));
  py.globals.set(`${prefix.toUpperCase()}_OPTIONS`, py.toPy(model.options));
}
py.globals.set("TEXT", fs.readFileSync(textFile, "utf8"));
const result = py.runPython(`
import json
import numpy as np
read = lambda name: open(name, "rb").read()
# the checkpoints go straight into forward.js's memories: gigabytes do not fit Pyodide's heap
small = kernel_llama_file(SMALL_CHECKPOINT, read("small.tokenizer.bin"), **SMALL_OPTIONS)
large = kernel_llama_file(LARGE_CHECKPOINT, read("large.tokenizer.bin"), **LARGE_OPTIONS)
tokens = large.tokenizer.encode(TEXT)[:${Number(tokensArg)}]
assert small.tokenizer.encode(TEXT)[:len(tokens)] == tokens, "the two models do not share a vocabulary"
WINDOW = 512

def agreement(sequence):
    """1 where both would choose the same next token, per position (the context starts anew every WINDOW)"""
    same = []
    for start in range(0, len(sequence), WINDOW - 1):
        piece = [large.bos] + sequence[start:start + WINDOW - 1]
        for pos in range(len(piece) - 1):
            a = int(np.argmax(small.forward(piece[pos], pos)))
            b = int(np.argmax(large.forward(piece[pos], pos)))
            same.append(int(a == b))
    return same

def greedy(first, count):
    """the large model's own continuation of the first tokens"""
    sequence = [large.bos] + first
    for pos in range(len(sequence) - 1):
        large.forward(sequence[pos], pos, need_logits=False)
    for pos in range(len(sequence) - 1, len(sequence) - 1 + count):
        sequence.append(int(np.argmax(large.forward(sequence[pos], pos))))
    return sequence[1:]

def summary(same):
    runs, run = [0] * len(same), 0
    for i in range(len(same) - 1, -1, -1):  # how many agree in a row from position i on
        run = run + 1 if same[i] else 0
        runs[i] = run
    return {"positions": len(same), "agree": sum(same) / len(same),
            "tokens_per_pass": {k: sum(min(r, k) + 1 for r in runs) / len(runs) for k in (1, 2, 4, 8)}}

text = summary(agreement(tokens))
continuation = greedy(tokens[:32], ${Number(greedyArg)})
own = summary(agreement(continuation)[32:])
json.dumps({"small": small.backend, "large": large.backend, "text": text, "greedy": own})
`);
const { text, greedy } = JSON.parse(result);
const row = (label, s) => `| ${label} | ${s.positions} | ${(s.agree * 100).toFixed(1)}% | ` +
  Object.values(s.tokens_per_pass).map((t) => t.toFixed(2)).join(" | ") + " |";
console.log(`\n${small.name} drafting for ${large.name}, ${path.basename(textFile)}\n`);
console.log("| text | positions | most likely token the same | tokens per pass, k = 1 | k = 2 | k = 4 | k = 8 |");
console.log("|---|---:|---:|---:|---:|---:|---:|");
console.log(row("the text", text));
console.log(row("the large model's greedy continuation", greedy));
