// T232's review (the throwaway branch t232-review-probe only): mutants of the ternary GPU path that the three made-up
// ternary models of dim 128 could not see, `node tests/t232-review-mutants.mjs <name>` changes the working tree for that
// mutant (git checkout restores it); without a name it lists them. Every text must be found exactly once.
import fs from "node:fs";

const SHADERS = "public/shaders.js", GPU = "public/gpu.js", FORWARD = "public/forward.js";
const COLUMN = "        sum += picked[k] * (f32(i32(code) - 1) * scales[row * (outliers.n / 128u) + channel / 128u]);";
const MUTANTS = {
  // the outlier columns' scale of the first group of the row, whichever group the channel is in
  "columns-group": [[SHADERS, COLUMN, "        sum += picked[k] * (f32(i32(code) - 1) * scales[row * (outliers.n / 128u)]);"]],
  // the outlier columns' scales a row apart by one, not by the row's n / 128
  "columns-row-stride": [[SHADERS, COLUMN, "        sum += picked[k] * (f32(i32(code) - 1) * scales[row + channel / 128u]);"]],
  // the embedding's scale of the first group of the row, whichever group the word is in
  "embed-group": [[SHADERS, "        let d = scales[(row * n + word * 16u) / 128u];", "        let d = scales[row * (n / 128u)];"]],
  // the embedding's scales a row apart by the one of a dim of 128
  "embed-row-stride": [[SHADERS, "        let d = scales[(row * n + word * 16u) / 128u];", "        let d = scales[(row * 128u + word * 16u) / 128u];"]],
  // a step's embedding from the classifier's pieces (the same where the two tables are one)
  "embed-from-classifier": [[GPU, "  const list = !embed ? [] : m.tables.embedding.map((piece, i) =>", "  const list = !embed ? [] : m.tables.classifier.map((piece, i) =>"]],
  // the embedding apart from the classifier read as int8 weights (its flag lost)
  "embed-flag": [[FORWARD, "ternary: embedding.kind === \"ternary\", at: [base + embedding.offset, base + embedding.scales] },", "ternary: false, at: [base + embedding.offset, base + embedding.scales] },"]],
  // the outlier channels taken from the first four channels' words only (the second vec4 of the Outliers read as the first)
  "take-first-four": [[SHADERS, "    let channel = outliers.channels[k / 4u][k % 4u];\n    picked[k] = x[channel];", "    let channel = outliers.channels[0u][k % 4u];\n    picked[k] = x[channel];"]],
  // a token's matrix: the scale of the group of 128 by the thread's own column, not by the pass's (the same in the first
  // pass of 32 groups, and a row of more than 32 groups has a second)
  "token-scale-local": [[SHADERS, "                let own_scale_b = scales_b[b_global * K128 + k_offset / 4u];", "                let own_scale_b = scales_b[b_global * K128 + local_col / 4u];"]],
  // the last of the outlier channels left out of the columns
  "columns-last-dropped": [[SHADERS, "    for (var k = 0u; k < outliers.count; k++) {\n        let channel = outliers.channels[k / 4u][k % 4u];", "    for (var k = 0u; k + 1u < outliers.count; k++) {\n        let channel = outliers.channels[k / 4u][k % 4u];"]],
};

const name = process.argv[2];
if (!name) {
  console.log(Object.keys(MUTANTS).join(" "));
  process.exit(0);
}
if (!MUTANTS[name]) throw new Error(`no mutant ${name}`);
for (const [file, find, replace] of MUTANTS[name]) {
  const text = fs.readFileSync(file, "utf8"), count = text.split(find).length - 1;
  if (count !== 1) throw new Error(`${name}: its text is in ${file} ${count} times, not once`);
  fs.writeFileSync(file, text.replace(find, () => replace));
}
console.log(`mutant ${name} applied`);
