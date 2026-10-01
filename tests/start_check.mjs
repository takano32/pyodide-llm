// start_check.mjs (T249's review): what the token in front of a text costs a model, on the page's engine (forward.js on
// the int8 kernels with 7-bit activations, tests/engine.mjs): the same measure as tests/start_check.py, which does it
// on the original in float32 with transformers and cannot take a model whose float32 weights do not fit a runner (an
// 8B). Every window of W tokens is scored from its second token under each start, so that each is held to the same
// targets (T131's measure); per start the perplexity and the nats per character, which a model of another
// vocabulary can be set beside (tiny-lm: xsmall's place in the list is a question of how much better it is).
//
//   node tests/start_check.mjs <the <out> of tests/perplexity_prepare.py, or the id of a model of this site>
//        [<the JSON of tests/write_options.py>] [--starts none,1,151643] [--tokens 1500] [--window 512] [--text <file>]
//
// --starts: the token ids to begin a window with, "none" for no token (the default: none and the model's own BOS).
// The JSON of write_options.py gives the options the page gives the model (the entry's over the converter's), where
// an <out>'s own .json has the converter's alone. A prepared <out> is read from its file straight into forward.js's
// memory (a model of gigabytes), on a 64-bit memory where it with its forward pass does not fit 4 GiB, as
// tests/write.mjs; a model of this site is copied in. The text, if none is given, is the beginning of three Japanese
// Wikipedia articles (tests/wikipedia.mjs: fetched when this runs).
// tests/write.sh runs it too (WRITER=start_check.mjs, with <out> and the JSON as it gives them; its TOKENS is
// --tokens, and STARTS, WINDOW and TEXT stand for the flags): an 8B is fetched and converted there, and takes 2 tokens
// a second (one thread), so 512 tokens in windows of 256 under four starts is a quarter of an hour.
import fs from "node:fs";
import path from "node:path";
import { pyodideWithEngine } from "./engine.mjs";
import { footprint, needsWide } from "../public/forward.js";
import { ARTICLES, wikipediaText } from "./wikipedia.mjs";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
// the model (and a JSON), then the flags: each with the value that follows
const args = process.argv.slice(2), given = {}, named = [];
for (let i = 0; i < args.length; i++) {
  if (args[i].startsWith("--")) given[args[i].slice(2)] = args[++i];
  else named.push(args[i]);
}
const [id, pageFile, tokensGiven] = named;  // tests/write.sh's order: <out>, the JSON, TOKENS
const value = (name, fallback) => given[name] ?? process.env[name.toUpperCase()] ?? fallback;
const site = MODELS.find((entry) => entry.id === id);
const model = site ?? (fs.existsSync(`${id}.json`) &&
  { id: path.basename(id), checkpoint: path.resolve(`${id}.bin`), tokenizer: path.resolve(`${id}.tokenizer.bin`),
    options: JSON.parse(fs.readFileSync(`${id}.json`, "utf8")) });
if (!model) throw new Error(`${id} is neither a model of src/models.js nor the <out> of tests/perplexity_prepare.py`);
const page = pageFile?.endsWith(".json") ? JSON.parse(fs.readFileSync(pageFile, "utf8")) : null;
const options = page?.options ?? model.options;
const fromFile = !site;
const local = (file) => (path.isAbsolute(file) ? file : root + file);
const starts = value("starts", "none").split(",").map((token) => (token === "none" ? null : Number(token)));
const text = value("text") ? fs.readFileSync(value("text"), "utf8") : await wikipediaText("ja", ARTICLES.ja);

// the checkpoint's own header (7 int32: the legacy format) says what the forward pass puts after it
let wide = false;
if (fromFile) {
  const size = fs.statSync(local(model.checkpoint)).size, head = Buffer.alloc(28), fd = fs.openSync(local(model.checkpoint), "r");
  fs.readSync(fd, head, 0, 28, 0);
  fs.closeSync(fd);
  const header = Array.from({ length: 7 }, (_, i) => head.readInt32LE(4 * i));
  const { arch, head_dim, dtype = "float32" } = options;
  wide = needsWide(size, footprint(header, size, { dtype, arch, head_dim, halfKV: true, shared: true }));
}
const { pyodide } = await pyodideWithEngine({ wide });
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(local(model.tokenizer)));
if (!fromFile) pyodide.FS.writeFile("checkpoint.bin", fs.readFileSync(local(model.checkpoint)));
pyodide.globals.set("CHECKPOINT", fromFile ? local(model.checkpoint) : "checkpoint.bin");
pyodide.globals.set("FROM_FILE", fromFile);
pyodide.globals.set("OPTIONS", pyodide.toPy(options));
pyodide.globals.set("TEXT", text);
pyodide.globals.set("TOKENS", Number(given.tokens ?? tokensGiven ?? process.env.TOKENS ?? "1500"));
pyodide.globals.set("WINDOW", Number(value("window", "512")));
pyodide.globals.set("STARTS_JSON", JSON.stringify(starts));  // (a null of JavaScript is no None in Python)
const result = JSON.parse(pyodide.runPython(`
import json, math, time
import numpy as np

options = dict(OPTIONS)
STARTS = json.loads(STARTS_JSON)
vocabulary = open("tokenizer.bin", "rb").read()
if FROM_FILE:
    llama = kernel_llama_file(CHECKPOINT, vocabulary, **options)
else:
    llama = kernel_llama(open(CHECKPOINT, "rb").read(), vocabulary, **options)
every = llama.tokenizer.encode(TEXT, llama.specials)
ids = every[:TOKENS]
characters = len(TEXT) * len(ids) / len(every)
window = min(WINDOW, llama.seq_len - 1)
windows = [w for w in (ids[i:i + window] for i in range(0, len(ids), window)) if len(w) >= 64]
labels = [("none" if s is None else str(s)) for s in STARTS]
totals, scored, began = [0.0] * len(STARTS), 0, time.perf_counter()
for number, targets in enumerate(windows):
    scored += len(targets) - 1
    for k, start in enumerate(STARTS):
        row = ([] if start is None else [start]) + targets
        # row[1:] are the targets but the first without a start; with one the first is row[1]: scored from the second
        for pos in range(len(row) - 1):
            logits = np.asarray(llama.forward(row[pos], pos), dtype=np.float64)
            logits -= logits.max()
            if start is None or pos >= 1:
                totals[k] -= logits[row[pos + 1]] - math.log(np.exp(logits).sum())
    print(f"window {number + 1}/{len(windows)} done, {time.perf_counter() - began:.0f} s", flush=True)
pieces = [llama.tokenizer.vocab[s].decode("utf-8", "replace") if s is not None else "" for s in STARTS]
json.dumps({"tokens": len(ids), "characters": characters, "window": window, "windows": len(windows), "scored": scored,
            "bos": llama.bos, "stops": sorted(llama.stop_tokens), "labels": labels, "pieces": pieces, "totals": totals,
            "seconds": time.perf_counter() - began})
`));
const perCharacter = result.characters * result.scored / result.tokens;
const name = site?.id ?? page?.id ?? path.basename(id);
console.log(`start_check ${name}: ${result.tokens} tokens of about ${result.characters.toFixed(0)} characters, windows of ${result.window}, ` +
  `the page's engine${wide ? " (a 64-bit memory)" : ""}; BOS ${result.bos}, stops ${result.stops.join(" ")}, ${result.seconds.toFixed(0)} s`);
console.log("\n| start | token | perplexity | against none | nats per character |\n|---|---|---:|---:|---:|");
const none = result.totals[result.labels.indexOf("none")];
result.labels.forEach((label, k) => {
  const total = result.totals[k];
  console.log(`| ${label} | ${JSON.stringify(result.pieces[k])} | ${Math.exp(total / result.scored).toFixed(3)} | ` +
    `${Number.isFinite(none) ? `${((Math.exp((total - none) / result.scored) - 1) * 100).toFixed(2)}%` : ""} | ${(total / perCharacter).toFixed(4)} |`);
});
console.log(`start_check ${name}: scored ${result.scored} targets in ${result.windows} windows`);
process.exit(0);
