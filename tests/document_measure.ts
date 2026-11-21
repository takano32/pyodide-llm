// document_measure.ts (T666): the head of an English book translated by PLaMo 3 Translate a chunk at a time
// (src/python/engine/document.py's translate_document()), at each size of chunk and of history asked, on the page's
// engine in Node; per chunk its tokens, its translation's, and whether it repeats itself, stopped far too short, ran
// out of room or lost paragraphs (tests/document_measure.py, run in Pyodide, counts them). The measure that
// tanakh/plamo-translate-desktop chose its sizes by: the chunks that repeat and the paragraphs dropped.
//
//   python3 tests/hf_fetch.py hf-plamo-3-translate-0.8b-en-ja <downloads>   (prints <folder>)
//   python3 tests/perplexity_prepare.py <folder> <out> int8 --blocks --entry hf-plamo-3-translate-0.8b-en-ja
//   node tests/document_measure.ts <out> [chunk/history ...] (e.g. 640/1280 320/1280)
//
// CHARACTERS=<n>: how much of the book's text (16000), from the line of FROM ("PREFACE.": the front matter before it is
// a title page and a list of illustrations); BOOK=<url>: another plain text
// (Pride and Prejudice, from a mirror of Project Gutenberg: www.gutenberg.org answers 503 to the machines of clouds). TARGET=<code>: the language translated into (ja). Lines begin `document:`.
import fs from "node:fs";
import { pyodideWithEngine } from "./engine.ts";
import { fetched } from "./fetching.ts";
import { leave } from "./leave.ts";
import { runtimeUrl } from "./tree.ts";
const { footprint, needsWide } = await import(runtimeUrl("forward.js"));

const [out, ...sizes] = process.argv.slice(2);
const asked = (sizes.length ? sizes : ["640/1280"]).map((size) => size.split("/").map(Number));
const book = (await fetched(process.env.BOOK ?? "https://gutenberg.pglaf.org/1/3/4/1342/1342-0.txt")).toString("utf8").replace(/\r\n/g, "\n");
const start = book.indexOf(process.env.FROM ?? "PREFACE.");
const body = book.slice(start < 0 ? 0 : book.lastIndexOf("\n", start) + 1);
const text = body.slice(0, Number(process.env.CHARACTERS ?? 16000)).replace(/\S*$/, "");
const options = JSON.parse(fs.readFileSync(`${out}.json`, "utf8"));
const size = fs.statSync(`${out}.bin`).size;
const header = [...new Int32Array(fs.readFileSync(`${out}.bin`).buffer.slice(0, 28))];
const wide = needsWide(size, footprint(header, size, { dtype: options.dtype, arch: options.arch, head_dim: options.head_dim, halfKV: true, shared: true }));
const { pyodide } = await pyodideWithEngine({ wide });
pyodide.FS.writeFile("tokenizer.bin", fs.readFileSync(`${out}.tokenizer.bin`));
pyodide.FS.writeFile("document_measure.py", fs.readFileSync(new URL("./document_measure.py", import.meta.url)));
pyodide.globals.set("CHECKPOINT", `${out}.bin`);
pyodide.globals.set("OPTIONS", pyodide.toPy(options));
pyodide.globals.set("TEXT", text);
pyodide.globals.set("TARGET", process.env.TARGET ?? "ja");
pyodide.runPython(`
import sys
sys.path.insert(0, ".")
from document_measure import measure
llama = kernel_llama_file(CHECKPOINT, open("tokenizer.bin", "rb").read(), **OPTIONS)
`);
console.log(`document: ${text.length} characters of the book, ${pyodide.runPython("len(llama.tokenizer.encode(TEXT, literal=True))")} tokens, ` +
  `${text.split(/\n[ \t\r]*\n/).filter((part) => part.trim()).length} paragraphs, context ${pyodide.runPython("llama.seq_len")}, into ${process.env.TARGET ?? "ja"}`);
const rows = [];
for (const [chunk, history] of asked) {
  pyodide.globals.set("CHUNK", chunk);
  pyodide.globals.set("HISTORY", history);
  const result = JSON.parse(pyodide.runPython("import json; json.dumps(measure(llama, TEXT, TARGET, CHUNK, HISTORY), ensure_ascii=False)"));
  for (const row of result.chunks) {
    console.log(`document ${chunk}/${history}: chunk ${row.index + 1}/${row.of}, ${row.source_tokens} -> ${row.tokens} tokens` +
      `${row.repeats ? ", REPEATS" : ""}${row.short ? ", SHORT" : ""}${row.lost ? `, ${row.lost} paragraphs LOST` : ""}: ${JSON.stringify(row.translation.slice(0, 120))}`);
  }
  const sum = (key) => result.chunks.reduce((total, row) => total + Number(row[key]), 0);
  const summary = `${result.chunks.length} chunks, ${sum("repeats")} repeat, ${sum("short")} short, ${result.unfinished} ran out of room, ` +
    `${sum("lost")} paragraphs lost, ${result.prompt_tokens} tokens of prompts, ${result.sampled} drawn, ${result.seconds.toFixed(0)} s`;
  rows.push(`document ${chunk}/${history}: ${summary}`);
  console.log(`document ${chunk}/${history}: ${summary}`);
  if (process.env.WRITE) fs.writeFileSync(`${process.env.WRITE}.${chunk}-${history}.txt`, result.translation);
}
for (const row of rows) console.log(row);
await leave(0);
