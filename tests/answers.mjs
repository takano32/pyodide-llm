// answers.mjs (the review of T253 and T254): what a model of the list writes for a dozen questions, whole, as the page
// samples it, for a person to read and for the few things a person counts: does an answer stop (a stop token of the
// model, not the end of the budget), is the language the question's, is anything come apart. tests/write.mjs shows
// one prompt's start and tests/degenerate.mjs counts breaks over many seeds; neither prints a whole answer to a list of
// questions, and a model that thinks first (<think> … </think>) needs the room to finish its thought, which
// the answers of 32 tokens never showed (T253 and T254 were read at one sentence).
//
//   SEEDS="1 2" PROMPTS='["…", "…"]' TOKENS=600 WRITER=answers.mjs bash tests/write.sh hf-granite-4.2-3b
//   (or node tests/answers.mjs <out of tests/perplexity_prepare.py> <JSON of tests/write_options.py> [tokens = 400])
//
// PROMPTS: a JSON list (else QUESTIONS below, six Japanese and six English; PICK="0,1,3": only these of them). SEEDS: the seeds of
// the sampled answers, as the entry's own sampling writes them (else "1"). GREEDY=1: and the greedy answer of each question first.
// TEMPERATURE=<number>: instead of the entry's. BOS=<token id>: begin every text with that token, and NOFORMAT=1: and no format, the bare question (what ?hf=<repository> gives a
// model whose chat template this converter cannot read: Granite 4.2's, which has an inline if and an empty list in it).
// The lines are `answers <id>: …` (ci.mjs's --grep answers).
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";
import { footprint, needsWide } from "../public/forward.js";
import { filled } from "../src/models.js";

const QUESTIONS = ["日本でいちばん高い山はどこですか？", "これからの流行りを3つ挙げてください。", "17 × 24 はいくつですか？途中の計算も書いてください。",
  "次の文を英語に訳してください。「今日は天気がいいので、散歩に行きます。」", "光合成とは何ですか？やさしく説明してください。", "夏目漱石の代表作を 2 つ挙げ、一言ずつ説明してください。",
  "What is the capital of Japan? Answer in one sentence.", "Give me three tips for sleeping better.", "What is 17 times 24? Show the steps.",
  "Translate into French: \"The weather is nice today, so I will go for a walk.\"", "Explain photosynthesis to a ten-year-old.",
  "Write a Python function that returns the n-th Fibonacci number, and say how it works in one sentence."];

const [out, pageFile, count = "400"] = process.argv.slice(2);
const page = JSON.parse(fs.readFileSync(pageFile, "utf8"));
const size = fs.statSync(`${out}.bin`).size;
const { arch, head_dim, dtype } = page.options;
const wide = needsWide(size, footprint(page.header, size, { dtype, arch, head_dim, halfKV: true, shared: true }));
const { pyodide } = await pyodideWithEngine({ wide });
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(`${out}.tokenizer.bin`));
// PICK="0,1,3,6,7,9": only these of QUESTIONS (an 8B at one thread writes a token a second: a dozen answers are a long job)
const prompts = process.env.PROMPTS ? JSON.parse(process.env.PROMPTS)
  : process.env.PICK ? process.env.PICK.split(",").map((at) => QUESTIONS[Number(at)]) : QUESTIONS;
const seeds = (process.env.SEEDS ?? "1").split(/\s+/).filter(Boolean).map(Number);
const { topp, repetition_penalty, top_k, min_p, presence_penalty } = page.generation;
// TEMPERATURE=0.7: the entry's temperature replaced (what a lower temperature does to the same questions and seeds)
const temperature = process.env.TEMPERATURE ? Number(process.env.TEMPERATURE) : page.generation.temperature;
const bare = Boolean(process.env.NOFORMAT);
pyodide.globals.set("CHECKPOINT", `${out}.bin`);
pyodide.globals.set("OPTIONS", pyodide.toPy({ ...page.options, ...(process.env.BOS ? { bos: Number(process.env.BOS) } : {}) }));
pyodide.globals.set("TEXTS", pyodide.toPy(prompts.map((prompt) => (page.template && !bare ? filled(page.template, prompt) : prompt))));
pyodide.globals.set("COUNT", Number(count));
pyodide.globals.set("SEEDS", pyodide.toPy(seeds));
pyodide.globals.set("GREEDY", Boolean(process.env.GREEDY));
// SAMPLING=temperature=0.6,topp=0.95,top_k=0,presence_penalty=0: settings over the entry's (T274: the same
// questions and seeds with another sampler)
const sampling = { ...Object.fromEntries(Object.entries({ temperature, topp, repetition_penalty, top_k, min_p, presence_penalty }).filter(([, v]) => v !== undefined)),
  ...Object.fromEntries((process.env.SAMPLING ?? "").split(",").filter(Boolean).map((pair) => [pair.split("=")[0], Number(pair.split("=")[1])])) };
pyodide.globals.set("SAMPLING", pyodide.toPy(sampling));
const result = JSON.parse(pyodide.runPython(`
import json, re, time
llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **OPTIONS)
JAPANESE = re.compile(r"[\\u3040-\\u30ff\\u3400-\\u9fff\\uff00-\\uffef]")
KANA = re.compile(r"[\\u3040-\\u30ff]")
LETTERS = re.compile(r"[^\\W\\d_]", re.UNICODE)

def judge(answer, text):
    letters = LETTERS.findall(answer)
    windows = [answer[i:i + 5] for i in range(max(0, len(answer) - 4))]
    return {"loop": len(windows) >= 20 and 1 - len(set(windows)) / len(windows) > 0.5,
            # the share of kana, not of kanji: a Chinese answer to a Japanese question is all kanji and counts as Japanese by them
            # (MiniCPM5's, in the review of T254), and a Japanese one is half kana
            "japanese": sum(1 for c in letters if KANA.match(c)) / len(letters) if letters else 0.0,
            # asked in Japanese: four letters of kana or kanji in what was typed (the format's own words are English)
            "asked_in_japanese": sum(1 for c in text if JAPANESE.match(c)) >= 4, "unk": "<unk>" in answer,
            "thought": "</think>" in answer, "thinks": text.rstrip().endswith("<think>") or "<think>" in answer}

def write(text, **sampling):
    ids = llama.tokenizer.encode(text, llama.specials)
    began = time.perf_counter()
    answer = "".join(llama.generate(text, steps=len(ids) + COUNT, echo=False, **sampling))
    stats = dict(llama.stats)
    # a stop token ends the run with one more sampling step than there are tokens after the prompt
    stopped = stats["sampled"] > stats["tokens"] - stats["prompt_tokens"]
    return {"answer": answer, "tokens": stats["sampled"] - (1 if stopped else 0), "stopped": stopped, "seconds": time.perf_counter() - began,
            "prompt_tokens": len(ids)}

rows = []
for p, text in enumerate(TEXTS):
    runs = ([("greedy", write(text, temperature=0.0))] if GREEDY else []) + [(f"seed {seed}", write(text, seed=seed, **SAMPLING)) for seed in SEEDS]
    for kind, run in runs:
        rows.append({"prompt": p, "kind": kind, **run, **judge(run["answer"], TEXTS[p])})
json.dumps({"rows": rows, "bos": llama.bos, "stops": sorted(llama.stop_tokens)})
`));
const id = page.id;
const sampled = result.rows.filter((row) => row.kind !== "greedy");
console.log(`answers ${id}: ${prompts.length} questions x ${seeds.length} seeds${process.env.GREEDY ? " and greedy" : ""}, up to ${count} tokens, temperature ${temperature}, ` +
  `sampling ${JSON.stringify(sampling)}, bos ${result.bos}, stops ${result.stops.join(" ")}, ${wide ? "a 64-bit" : "a 32-bit"} memory, int8 ${(size / 1e9).toFixed(2)} GB` +
  `${bare ? ", NO FORMAT (the bare question)" : ""}`);
for (const row of result.rows) {
  console.log(`answers ${id}: [${row.kind}] ${JSON.stringify(prompts[row.prompt])} -> ${row.tokens} tokens in ${row.seconds.toFixed(0)} s, ` +
    `${row.stopped ? "STOPPED (a stop token)" : `CUT at the budget of ${count}`}, ${row.thought ? "thought finished (</think>)" : row.thinks ? "thought not finished" : "no thought"}, ` +
    `${row.asked_in_japanese ? `${(row.japanese * 100).toFixed(0)}% of its letters kana (Japanese is about half, Chinese none)` : "asked in English"}${row.loop ? ", LOOPS" : ""}${row.unk ? ", <unk>" : ""}`);
  console.log(`answers ${id}: [${row.kind}] answer ${JSON.stringify(row.answer)}`);
}
const share = (key) => `${sampled.filter((row) => row[key]).length}/${sampled.length}`;
console.log(`answers ${id}: all: stopped ${share("stopped")}, thought finished ${share("thought")}, loops ${share("loop")}, <unk> ${share("unk")}, ` +
  `a Japanese question answered in Japanese (a fifth of its letters kana) ${sampled.filter((row) => row.asked_in_japanese && row.japanese >= 0.2).length}/${sampled.filter((row) => row.asked_in_japanese).length}, ` +
  `${(sampled.reduce((sum, row) => sum + row.tokens, 0) / Math.max(1, sampled.length)).toFixed(0)} tokens written on average`);
process.exit(0);
