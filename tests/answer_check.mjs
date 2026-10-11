// answer_check.mjs (T250's review): what the token the page puts in front does to a chat model's own answers. The page begins every text
// with the BOS of its options; a Qwen3's real tokenizer puts nothing in front, and its format begins with <|im_start|>
// (tests/start_check.mjs found Shisa V2.1 Qwen3 8B 54% worse on plain Japanese with <|endoftext|> in front, 1.1% better with
// <|im_start|>). Plain text is not what a chat is, so this takes the answers: for each question the model writes greedily from the
// format as the real template makes it (nothing in front: its own <|im_start|> is the first token), then
//   - the same answer is read again with the page's BOS in front of the format: its perplexity there against the natural one, how
//     often the likeliest next token is the same, and the KL divergence of the page's next-token distribution from the natural one;
//   - the model writes greedily again with the page's BOS in front: where its answer first differs from the natural one, and both
//     answers' beginnings (what a visitor would see differ).
// The lines are `answer_check <id>: ...` (ci.mjs's --grep "answer_check"): a line for each question and one for all.
//
//   node tests/answer_check.mjs <out of tests/perplexity_prepare.py> <JSON of tests/write_options.py> [answer tokens = 60]
//
// PROMPTS='["…"]': the questions (else the entry's own and four more). START=<id>: the token in front in the page's reading (else
// the options' bos). The format's first token is the natural start: the entry's template must begin with a special token
// (<|im_start|>) that the options' specials name. tests/write.sh runs it (WRITER=answer_check.mjs).
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";
import { runtimeUrl } from "./tree.mjs";
const { footprint, needsWide } = await import(runtimeUrl("forward.js"));
import { filled } from "../src/models.js";
import { leave } from "./leave.mjs";

const [out, pageFile, count = "60"] = process.argv.slice(2);
const page = JSON.parse(fs.readFileSync(pageFile, "utf8"));
const size = fs.statSync(`${out}.bin`).size;
const { arch, head_dim, dtype } = page.options;
const wide = needsWide(size, footprint(page.header, size, { dtype, arch, head_dim, halfKV: true, shared: true }));
const { pyodide } = await pyodideWithEngine({ wide });
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(`${out}.tokenizer.bin`));
const prompts = process.env.PROMPTS ? JSON.parse(process.env.PROMPTS) :
  [page.prompt, "日本でいちばん高い山はどこですか？", "夏目漱石の代表作を 3 つ教えてください。", "おすすめの朝ごはんを 1 つ教えてください。", "新幹線について短く説明してください。"];
pyodide.globals.set("CHECKPOINT", `${out}.bin`);
pyodide.globals.set("OPTIONS", pyodide.toPy(page.options));
pyodide.globals.set("TEXTS", pyodide.toPy(prompts.map((prompt) => (page.template ? filled(page.template, prompt) : prompt))));
pyodide.globals.set("COUNT", Number(count));
pyodide.globals.set("START", process.env.START ? Number(process.env.START) : -1);
const result = JSON.parse(pyodide.runPython(`
import json, math, time
import numpy as np

llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **dict(OPTIONS))
start = llama.bos if START < 0 else START
stops = set(llama.stop_tokens) | {llama.bos}

def logits_from(row, first):
    """the logits of every position of row from first on (the forward pass one token a time, from position 0)"""
    rows = []
    for pos, token in enumerate(row):
        logits = llama.forward(token, pos, need_logits=pos >= first)
        if pos >= first:
            rows.append(np.array(logits, dtype=np.float64))
    return rows

def greedy(prefix):
    """what the model writes after prefix, greedily up to COUNT tokens or a stop token: the tokens and the logits that chose each"""
    logits = logits_from(prefix, len(prefix) - 1)[-1]
    written, rows = [], []
    while True:
        token = int(np.argmax(logits))
        if token in stops:
            break
        written.append(token)
        rows.append(logits)
        if len(written) == COUNT:
            break
        logits = np.array(llama.forward(token, len(prefix) + len(written) - 1), dtype=np.float64)
    return written, rows

def log_softmax(logits):
    shifted = logits - logits.max()
    return shifted - math.log(np.exp(shifted).sum())

def text_of(last, tokens):
    return b"".join(llama.tokenizer.decode(prev, token, llama.bos) for prev, token in zip([last] + tokens, tokens)).decode("utf-8", "replace")

began = time.perf_counter()
answers = []
for text in TEXTS:
    natural = llama.tokenizer.encode(text, llama.specials)  # the real template begins with its own first token: nothing in front
    paged = [start] + natural
    written, rows_natural = greedy(natural)
    if not written:
        continue
    rows_page = logits_from(paged + written, len(paged) - 1)[:len(written)]  # the same answer, the page's start in front
    nll_n = nll_p = agree = 0
    kl = 0.0
    for k, token in enumerate(written):
        ln, lp = log_softmax(rows_natural[k]), log_softmax(rows_page[k])
        nll_n -= ln[token]
        nll_p -= lp[token]
        agree += int(np.argmax(rows_natural[k]) == np.argmax(rows_page[k]))
        kl += float((np.exp(ln) * (ln - lp)).sum())
    written_page, _ = greedy(paged)  # what the model writes with the page's start in front
    same = 0
    while same < min(len(written), len(written_page)) and written[same] == written_page[same]:
        same += 1
    answers.append({"tokens": len(written), "nll_natural": float(nll_n), "nll_page": float(nll_p), "agree": agree, "kl": kl,
                    "text": text_of(natural[-1], written), "page_tokens": len(written_page), "same": same,
                    "page_text": text_of(natural[-1], written_page)})
    print(f"answer {len(answers)}/{len(TEXTS)} done, {time.perf_counter() - began:.0f} s", flush=True)
json.dumps({"answers": answers, "bos": start, "stops": sorted(stops), "seconds": time.perf_counter() - began})
`));
const id = page.id;
console.log(`answer_check ${id}: ${result.answers.length} answers of up to ${count} tokens written greedily from the real format (nothing in front), ` +
  `read again with ${result.bos} in front and written again with it in front, ${result.seconds.toFixed(0)} s`);
let tokens = 0, natural = 0, paged = 0, agree = 0, kl = 0, same = 0, differ = 0;
result.answers.forEach((row, k) => {
  tokens += row.tokens; natural += row.nll_natural; paged += row.nll_page; agree += row.agree; kl += row.kl; same += row.same;
  const alike = row.same === row.tokens && row.same === row.page_tokens;
  if (!alike) differ++;
  console.log(`answer_check ${id}: ${JSON.stringify(prompts[k])}: ${row.tokens} tokens, nll a token ${(row.nll_natural / row.tokens).toFixed(3)} natural, ` +
    `${(row.nll_page / row.tokens).toFixed(3)} with the start in front (${(Math.exp((row.nll_page - row.nll_natural) / row.tokens) * 100 - 100).toFixed(1)}% in perplexity), likeliest token the same ` +
    `${row.agree}/${row.tokens}, KL ${(row.kl / row.tokens).toFixed(4)} a token; written again with the start in front: ${alike ? "the same answer" : `the same for ${row.same} tokens, then another (${row.page_tokens} tokens)`}`);
  console.log(`answer_check ${id}:   natural: ${JSON.stringify(row.text.slice(0, 100))}`);
  if (!alike) console.log(`answer_check ${id}:   with the start: ${JSON.stringify(row.page_text.slice(0, 100))}`);
});
console.log(`answer_check ${id}: all: ${tokens} tokens, perplexity ${Math.exp(natural / tokens).toFixed(3)} natural, ${Math.exp(paged / tokens).toFixed(3)} with ${result.bos} in front ` +
  `(${(Math.exp((paged - natural) / tokens) * 100 - 100).toFixed(1)}%), likeliest token the same ${agree}/${tokens} (${(agree / tokens * 100).toFixed(1)}%), KL ${(kl / tokens).toFixed(4)} a token; ` +
  `${differ} of ${result.answers.length} answers written again differ, the same for ${same} of ${tokens} tokens`);
// (T357: what was written is out first: the answers of a long run are more than a pipe holds, tests/leave.mjs)
await leave(0);
