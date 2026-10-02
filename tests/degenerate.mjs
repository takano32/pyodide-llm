// degenerate.mjs (T249's review): how often a model of the list, sampled as the page samples it, writes something that has
// come apart, over many seeds and a few openings. tests/write.mjs shows what a model writes for one prompt, for a person to
// read; whether a model that wrote `<unk>` under greedy (rinna's japanese-gpt2 xsmall) does it as the page runs it is a
// question of how often, which three seeds do not answer. The arguments are write.mjs's, and tests/write.sh runs this in
// its place with WRITER=degenerate.mjs:
//
//   SEEDS="1 2 … 64" PROMPTS='["これからの流行りは", "富士山は、"]' WRITER=degenerate.mjs TOKENS=200 bash tests/write.sh hf-japanese-gpt2-xsmall
//
// (BOS=<token id> begins every text with that token instead of the entry's: how another start compares. BAN=unknown, or
// token ids "5,7": those are never drawn: how the answers look without the unknown piece.)
// For each prompt (PROMPTS: a JSON list, else the entry's own and two more openings) and seed it writes up to TOKENS
// tokens with the entry's sampling (until a stop token, as the page does) and counts what a reader sees come apart:
//   unk     the answer has <unk> in it (the model wrote the piece that stands for what its vocabulary lacks)
//   loop    more than half of its 5-character windows are ones it wrote before (it repeats itself)
//   other   less than half of its letters are in the prompt's script: Japanese (kana, kanji, full-width) where the prompt has any,
//           else Latin (T251's review: it was Japanese whatever the prompt, so every answer of an English model came out "other")
// and the line has the mean share of repeated 5-character windows besides (the loop's measure as a number, for models that loop
// less than all the way). The lines it prints are `degenerate <id>: <prompt> ...` for each prompt and a total (ci.mjs's --grep "degenerate").
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.mjs";
import { footprint, needsWide } from "../public/forward.js";
import { MODELS, filled } from "../src/models.js";

// <out> and the JSON as tests/write.sh gives them; or the id of a model of this site (tiny-lm: the baseline of a model of
// the list), whose checkpoint is in the repository's folder after `make models`
const [out, pageFile, count = "200"] = process.argv.slice(2);
const site = MODELS.find((entry) => entry.id === out && entry.checkpoint);
const files = site ? { checkpoint: `${new URL("../", import.meta.url).pathname}${site.checkpoint}`, tokenizer: `${new URL("../", import.meta.url).pathname}${site.tokenizer}` }
  : { checkpoint: `${out}.bin`, tokenizer: `${out}.tokenizer.bin` };
const size = fs.statSync(files.checkpoint).size;
const head = Buffer.alloc(28), fd = fs.openSync(files.checkpoint, "r");
fs.readSync(fd, head, 0, 28, 0);
fs.closeSync(fd);
const page = site ? { id: site.id, options: site.options, template: site.template, prompt: site.prompt, generation: site.generation,
  header: Array.from({ length: 7 }, (_, i) => head.readInt32LE(4 * i)) } : JSON.parse(fs.readFileSync(pageFile, "utf8"));
const { arch, head_dim, dtype } = page.options;
const wide = needsWide(size, footprint(page.header, size, { dtype, arch, head_dim, halfKV: true, shared: true }));
const { pyodide } = await pyodideWithEngine({ wide });
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(files.tokenizer));
const prompts = process.env.PROMPTS ? JSON.parse(process.env.PROMPTS) : [page.prompt, "富士山は、", "昔々、あるところに"];
const seeds = (process.env.SEEDS ?? "1 2 3 4 5 6 7 8 9 10").split(/\s+/).filter(Boolean).map(Number);
const { temperature, topp, repetition_penalty } = page.generation;
pyodide.globals.set("CHECKPOINT", files.checkpoint);
// BOS=<token id>: the token that begins every text, where the entry's says another (how the entry's choice compares)
pyodide.globals.set("OPTIONS", pyodide.toPy({ ...page.options, ...(process.env.BOS ? { bos: Number(process.env.BOS) } : {}) }));
pyodide.globals.set("TEXTS", pyodide.toPy(prompts.map((prompt) => (page.template ? filled(page.template, prompt) : prompt))));
pyodide.globals.set("COUNT", Number(count));
pyodide.globals.set("SEEDS", pyodide.toPy(seeds));
// BAN=unknown (or token ids, "5,7"): those tokens are never drawn (-inf written into their logits after the forward pass, where
// the sampling reads them): how the answers look if the page left the unknown piece out of what a model may write
pyodide.globals.set("BAN", pyodide.toPy((process.env.BAN ?? "").split(",").filter(Boolean)));
pyodide.globals.set("SAMPLING", pyodide.toPy(Object.fromEntries(Object.entries({ temperature, topp, repetition_penalty }).filter(([, v]) => v !== undefined))));
const result = JSON.parse(pyodide.runPython(`
import json, re, time
llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **OPTIONS)
banned = [llama.tokenizer.unknown if name == "unknown" else int(name) for name in BAN]
if banned:
    import numpy as np
    drawn_from = llama.forward
    def forward(token, pos, need_logits=True):
        logits = drawn_from(token, pos, need_logits=need_logits)
        if need_logits:
            logits[banned] = -np.inf
        return logits
    llama.forward = forward
JAPANESE =re.compile(r"[\\u3040-\\u30ff\\u3400-\\u9fff\\uff00-\\uffef]")
LATIN = re.compile(r"[A-Za-z]")
LETTERS = re.compile(r"[^\\W\\d_]", re.UNICODE)

def judge(answer, japanese):
    letters = LETTERS.findall(answer)
    windows = [answer[i:i + 5] for i in range(max(0, len(answer) - 4))]
    repeated = 1 - len(set(windows)) / len(windows) if windows else 0.0
    script = JAPANESE if japanese else LATIN  # the script of the prompt (T251's review: the check was Japanese whatever the prompt)
    return {"unk": "<unk>" in answer,
            "loop": len(windows) >= 20 and repeated > 0.5,
            "other": len(letters) >= 10 and sum(1 for c in letters if script.match(c)) / len(letters) < 0.5,
            "repeated": repeated, "chars": len(answer), "unks": answer.count("<unk>")}

rows = []
began = time.perf_counter()
for p, text in enumerate(TEXTS):
    ids = llama.tokenizer.encode(text, llama.specials)
    for seed in SEEDS:
        answer = "".join(llama.generate(text, steps=len(ids) + COUNT, seed=seed, echo=False, **SAMPLING))
        rows.append({"prompt": p, "seed": seed, "answer": answer, "tokens": llama.stats.get("sampled", 0), **judge(answer, bool(JAPANESE.search(text)))})
json.dumps({"rows": rows, "seconds": time.perf_counter() - began, "bos": llama.bos, "stops": sorted(llama.stop_tokens)})
`));
const id = page.id;
console.log(`degenerate ${id}: ${seeds.length} seeds x ${prompts.length} openings, up to ${count} tokens, temperature ${temperature}, top-p ${topp}, ` +
  `penalty ${repetition_penalty}, bos ${result.bos}, stops ${result.stops.join(" ")}, ${result.seconds.toFixed(0)} s`);
const share = (rows, key) => `${rows.filter((row) => row[key]).length}/${rows.length}`;
const line = (name, rows) => console.log(`degenerate ${id}: ${name}: <unk> in ${share(rows, "unk")}, loop in ${share(rows, "loop")}, ` +
  `less than half in the prompt's script in ${share(rows, "other")}, any of the three in ${rows.filter((row) => row.unk || row.loop || row.other).length}/${rows.length}; ` +
  `${(rows.reduce((sum, row) => sum + row.repeated, 0) / rows.length * 100).toFixed(1)}% of the 5-character windows repeated on average, ` +
  `${(rows.reduce((sum, row) => sum + row.unks, 0) / rows.reduce((sum, row) => sum + Math.max(1, row.tokens), 0) * 100).toFixed(2)} <unk> a 100 tokens, ` +
  `${(rows.reduce((sum, row) => sum + row.tokens, 0) / rows.length).toFixed(0)} tokens written on average`);
prompts.forEach((prompt, p) => line(JSON.stringify(prompt), result.rows.filter((row) => row.prompt === p)));
line("all", result.rows);
// what the worst ones say, to read: the first three that have <unk> or loop, and the first that is none of them
const shown = (row) => `degenerate ${id}: ${JSON.stringify(prompts[row.prompt])} seed ${row.seed}: ${JSON.stringify(row.answer.slice(0, 160))}`;
for (const row of result.rows.filter((row) => row.unk || row.loop || row.other).slice(0, 3)) console.log(shown(row));
for (const row of result.rows.filter((row) => !row.unk && !row.loop && !row.other).slice(0, 2)) console.log(shown(row) + " (fine)");
process.exit(0);
