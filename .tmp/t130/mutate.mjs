// mutants of public/forward.js for tests/memory-check.mjs: each must fail it. Written to .tmp/t130/mut/ with jobs.js beside.
import fs from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";

const root = new URL("../../", import.meta.url).pathname;
const out = path.join(root, ".tmp/t130/mut");
fs.mkdirSync(out, { recursive: true });
fs.copyFileSync(path.join(root, "public/jobs.js"), path.join(out, "jobs.js"));
const source = fs.readFileSync(path.join(root, "public/forward.js"), "utf8");

const growLoops = `    for (let l = layers - 1; l >= 0; l--) U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
    for (let l = layers - 1; l > 0; l--) U.copyWithin(keys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);`;
const oldGrow = `    const newKeys = alloc(layers * newLayer), newValues = alloc(layers * newLayer);
    for (let l = 0; l < layers; l++) {
      U.copyWithin(newKeys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);
      U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
    }
    // move both down onto the old blocks, which were the last thing in memory
    const start = keys;
    U.copyWithin(start, newKeys, newValues + layers * newLayer);
    keys = start;
    values = start + (newValues - newKeys);
    top = align(values + layers * newLayer);
    capacity = larger;`;
const mutants = [
  ["grow-ascending", growLoops, `    for (let l = 0; l < layers; l++) U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
    for (let l = 1; l < layers; l++) U.copyWithin(keys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);`],
  ["grow-keys-first", growLoops, `    for (let l = layers - 1; l > 0; l--) U.copyWithin(keys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);
    for (let l = layers - 1; l >= 0; l--) U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);`],
  ["grow-drops-last-position", `U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
    for (let l = layers - 1; l > 0; l--)`, `U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer - KV);
    for (let l = layers - 1; l > 0; l--)`],
  ["grow-keys-first-layer-moved-too", growLoops, `    for (let l = layers - 1; l >= 0; l--) U.copyWithin(newValues + l * newLayer, values + l * oldLayer, values + (l + 1) * oldLayer);
    for (let l = layers - 1; l > 1; l--) U.copyWithin(keys + l * newLayer, keys + l * oldLayer, keys + (l + 1) * oldLayer);`],
  ["grow-no-room", "    alloc(2 * layers * (newLayer - oldLayer));\n", "    alloc(layers * (newLayer - oldLayer));\n"],
  ["grow-before-t130", `    alloc(2 * layers * (newLayer - oldLayer));
    const newValues = keys + layers * newLayer;
${growLoops}
    values = newValues;
    capacity = larger;
  }`, oldGrow + "\n  }"],
  ["footprint-peak-of-the-old-growth", "keys = direct ? 0 : seqLen * layers * 2 * kvDim;", "keys = direct ? 0 : 1.5 * seqLen * layers * 2 * kvDim;"],
  ["footprint-no-megabyte", "const others = Math.ceil(bytes) + 2 ** 20,", "const others = Math.ceil(bytes),"],
  ["footprint-no-outlier-columns", "Math.min(outliers, dim) * (vocab + 1) * 4", "0"],
  ["footprint-no-corrections", "(relaxed ? weights / 8 : 0)", "0"],
  ["footprint-no-rope-tables", "  if (quantized || arch === \"gpt2\") bytes += seqLen * headSize * 4;\n", ""],
  ["footprint-gpt2-before-the-review", "  if (quantized && arch === \"gpt2\") bytes += seqLen * dim * 4;\n  if (quantized || arch === \"gpt2\") bytes += seqLen * headSize * 4;", "  if (quantized) bytes += arch === \"gpt2\" ? seqLen * dim * 4 : seqLen * headSize * 4;"],
  ["footprint-no-logits", "+ vocab * 4;\n  if (gpu)", ";\n  if (gpu)"],
  ["footprint-half-where-int8-does-not-run", "const half = halfKV && onInt8 && (", "const half = halfKV && ("],
  ["footprint-plain-rule-on-shared", "(shared ? kvHeads >= heads || past : past && !needsWide(size, others + 2 * keys))", "(past && !needsWide(size, others + 2 * keys))"],
  ["footprint-shared-rule-on-plain", "(shared ? kvHeads >= heads || past : past && !needsWide(size, others + 2 * keys))", "(kvHeads >= heads || past)"],
];
const results = [];
for (const [name, find, replace] of mutants) {
  const count = source.split(find).length - 1;
  if (count !== 1) { console.log(`${name}: the text to change is there ${count} times`); continue; }
  const file = path.join(out, `forward-${name}.js`);
  fs.writeFileSync(file, source.replace(find, replace));
  let verdict;
  try {
    execFileSync("node", [path.join(root, "tests/memory-check.mjs"), "--forward", file], { cwd: root, stdio: ["ignore", "pipe", "pipe"], timeout: 120000 });
    verdict = "PASSED (not caught)";
  } catch (error) {
    const text = (error.stderr?.toString() ?? "") + (error.stdout?.toString() ?? "");
    const line = text.split("\n").find((l) => /AssertionError|Error:/.test(l)) ?? text.split("\n")[0];
    verdict = `fails: ${line.replace(/^.*?(AssertionError \[ERR_ASSERTION\]: |Error: )/, "").slice(0, 190)}`;
  }
  console.log(`${name.padEnd(40)} ${verdict}`);
  results.push([name, verdict]);
}
console.log(`\n${results.filter(([, v]) => v.startsWith("fails")).length} of ${results.length} mutants fail the check`);
