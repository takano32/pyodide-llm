// What quantizing the activations costs: the perplexity of a model with NumPy, with 8-bit and with 7-bit activations
// in the kernels, on the same text. Runs tests/perplexity.py inside Pyodide in Node.
//
//   node tests/perplexity.mjs [model id = llm-jp-3-150m] [tokens = 1500] [text file | Japanese Wikipedia title ...]
//
// The model id may instead be the <out> of tests/perplexity_prepare.py (a Hugging Face model converted here, with
// <out>.bin, <out>.tokenizer.bin and the options in <out>.json).
//
// --file (T229): the checkpoint is read from its file straight into forward.js's memory, never into Pyodide's, and
// the NumPy row is left out (a model whose weights widened to float32 pass Pyodide's 4 GiB: tests/perplexity_native.py
// has that row). --numpy=<its perplexity>: what the rows are held against then. --wide (T247): a 64-bit memory and its
// kernels, for a checkpoint that with what forward.js puts after it passes 4 GiB (Qwen3.5 4B and 9B). --rows=0,2: only
// those rows, counted from 0 in the order of tests/perplexity.py (a model of 10 GB takes a process for each row: the
// memory of one row is not given back before the next row's is made).
//
// Without a text file the text is fetched from Japanese Wikipedia (plain-text extracts; nothing of it is stored in
// this repository). The NumPy row takes minutes: it runs at a tenth of the speed.
import fs from "node:fs";
import path from "node:path";
import { version } from "pyodide";
import { pyodideWithEngine } from "./engine.mjs";
import { ARTICLES, wikipediaText } from "./wikipedia.mjs";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const flags = process.argv.slice(2).filter((arg) => arg.startsWith("--"));
const [id = "llm-jp-3-150m", count = "1500", ...sources] = process.argv.slice(2).filter((arg) => !arg.startsWith("--"));
const fromFile = flags.includes("--file"), numpy = Number(flags.find((flag) => flag.startsWith("--numpy="))?.split("=")[1]);
const rows = flags.find((flag) => flag.startsWith("--rows="))?.split("=")[1].split(",").map(Number);
// --max-change=<percent> (T229's review): exit 1 if a row's perplexity is further than this from NumPy's, up or down; without it
// the table is only to be read (a Qwen3.5's int8 path whose gate was left out read as such for nobody: 25.5 on the table
// is +1.9%, 28 or 700 is a fault, and no line said so)
const maxChange = Number(flags.find((flag) => flag.startsWith("--max-change="))?.split("=")[1]);
const model = MODELS.find((entry) => entry.id === id) ?? (fs.existsSync(`${id}.json`) &&
  { id: path.basename(id), checkpoint: path.resolve(`${id}.bin`), tokenizer: path.resolve(`${id}.tokenizer.bin`),
    options: JSON.parse(fs.readFileSync(`${id}.json`, "utf8")) });
if (!model) throw new Error(`${id} is neither a model of src/models.js nor the <out> of tests/perplexity_prepare.py`);
const local = (file) => (path.isAbsolute(file) ? file : root + file);
const titles = sources.length ? sources : ARTICLES.ja;

let text = "";
for (const source of titles) {
  text += fs.existsSync(source) ? fs.readFileSync(source, "utf8") : await wikipediaText("ja", [source]);
}

const { pyodide } = await pyodideWithEngine({ wide: flags.includes("--wide") });
for (const file of fromFile ? [model.tokenizer] : [model.checkpoint, model.tokenizer]) {
  pyodide.FS.writeFile(path.basename(file), fs.readFileSync(local(file)));
}
const name = (file) => path.basename(file);
pyodide.globals.set("MODEL", pyodide.toPy({ checkpoint: name(model.checkpoint), tokenizer: name(model.tokenizer), options: model.options,
  ...(fromFile ? { file: local(model.checkpoint) } : {}), ...(rows ? { rows } : {}) }));
pyodide.globals.set("TEXT", text);
pyodide.globals.set("TOKENS", Number(count));
pyodide.globals.set("WINDOW", 512);
pyodide.runPython(fs.readFileSync(`${root}tests/perplexity.py`, "utf8"));
const results = JSON.parse(pyodide.globals.get("RESULT"));

const reference = fromFile ? numpy : results.at(-1).perplexity;  // (NaN with --file and no --numpy: no change said)
console.log(`### Perplexity of ${model.id}, ${results[0].tokens} tokens of ${titles.join(", ")} (Pyodide ${version} in Node)\n`);
console.log("| computation | perplexity | against NumPy | time |");
console.log("|---|---:|---:|---:|");
let apart = [];
for (const row of results) {
  const change = (row.perplexity / reference - 1) * 100;
  const against = Number.isFinite(change) ? `${change < 0 ? "-" : "+"}${Math.abs(change).toFixed(2)}%` : "";
  console.log(`| ${row.variant} | ${row.perplexity.toFixed(3)} | ${against} | ${row.seconds.toFixed(0)} s |`);
  if (Number.isFinite(maxChange) && !(Math.abs(change) <= maxChange)) apart.push(`${row.variant} (${change.toFixed(2)}%)`);
}
if (apart.length) {
  console.log(`\nFAILED: further than ${maxChange}% from NumPy's perplexity: ${apart.join("; ")}`);
  process.exit(1);
}
