// What quantizing the activations costs: the perplexity of a model with NumPy, with 8-bit and with 7-bit activations
// in the kernels, on the same text. Runs tests/perplexity.py inside Pyodide in Node.
//
//   node tests/perplexity.mjs [model id = llm-jp-3-150m] [tokens = 1500] [text file | Japanese Wikipedia title ...]
//
// Without a text file the text is fetched from Japanese Wikipedia (plain-text extracts; nothing of it is stored in
// this repository). The NumPy row takes minutes: it runs at a tenth of the speed.
import fs from "node:fs";
import path from "node:path";
import { loadPyodide, version } from "pyodide";
import { MODELS } from "../src/models.js";

const root = new URL("../", import.meta.url).pathname;
const [id = "llm-jp-3-150m", count = "1500", ...sources] = process.argv.slice(2);
const model = MODELS.find((entry) => entry.id === id);
const titles = sources.length ? sources : ["富士山", "夏目漱石", "新幹線"];

let text = "";
for (const source of titles) {
  if (fs.existsSync(source)) {
    text += fs.readFileSync(source, "utf8");
    continue;
  }
  const url = `https://ja.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exsectionformat=plain&format=json&titles=${encodeURIComponent(source)}`;
  const pages = (await (await fetch(url, { headers: { "User-Agent": "pyodide-llama-py perplexity measurement" } })).json()).query.pages;
  // the beginning of each article: prose, before the lists and tables of the later sections
  text += Object.values(pages)[0].extract.slice(0, 6000) + "\n";
}

const pyodide = await loadPyodide();
await pyodide.loadPackage("numpy", { messageCallback: () => {} });
for (const file of ["public/llama2_numpy.py", "public/simdkernel.so", "public/simdkernel_relaxed.wasmlib", model.checkpoint, model.tokenizer]) {
  pyodide.FS.writeFile(path.basename(file), fs.readFileSync(root + file));
}
pyodide.globals.set("MODEL", pyodide.toPy({ checkpoint: model.checkpoint, tokenizer: model.tokenizer, options: model.options }));
pyodide.globals.set("TEXT", text);
pyodide.globals.set("TOKENS", Number(count));
pyodide.globals.set("WINDOW", 512);
const results = JSON.parse(pyodide.runPython(fs.readFileSync(`${root}tests/perplexity.py`, "utf8")));

const reference = results.at(-1).perplexity;
console.log(`### Perplexity of ${model.id}, ${results[0].tokens} tokens of ${titles.join(", ")} (Pyodide ${version} in Node)\n`);
console.log("| computation | perplexity | against NumPy | time |");
console.log("|---|---:|---:|---:|");
for (const row of results) {
  const change = (row.perplexity / reference - 1) * 100;
  console.log(`| ${row.variant} | ${row.perplexity.toFixed(3)} | ${change < 0 ? "-" : "+"}${Math.abs(change).toFixed(2)}% | ${row.seconds.toFixed(0)} s |`);
}
