// T.357.3: the tables of tests/gpu-check.mjs, a part of that tool: what a person reads in the log of a model.
//
// T.183: what a person reads to judge it, in the log of CI (the development machine does not run WebGPU's tests): E16,
// how far NumPy's answer moves when nothing but its cache is rounded to float16 (answer(half=True), T.153's review), and
// for each model a table of the keys and values by layer, a column E16 and one a run (the CPU's, and the GPU's lettered
// A, B...) against NumPy's, another against NumPy's with its cache in float16, a line a run with its form, its line and
// the ratio to it, how many E16 it is and its most likely token, and the seconds of every step. T.187: Q8 (NumPy's
// with the inputs of the layers' matrices in 8 bits as well) beside E16 and a table against it, the first layer's
// scale, the logits' KL divergence and the difference of the ten most likely (in place of the logits' ratio to the
// CPU's).
import { floats, worstRow } from "./measures.mjs";

// T.183: what a person reads in the log of CI to judge the numbers (Markdown: gpu-prompt.yml puts the log in the run's
// summary too). A table of the keys and values by layer against NumPy's, with E16's column (NumPy's with its cache in
// float16) and T.187's Q8 (and its matrices' inputs in 8 bits) beside the runs', where a layer that jumps shows; the same
// against NumPy's with its cache in float16 (what is left is the arithmetic, not the cache); a line a run: its form, its
// line and the ratio to it, how many E16 and Q8, the first layer's scale, its logits' KL and the largest difference of
// NumPy's ten most likely, its most likely token; the seconds of every step. The runs are the GPU's worker's first (the
// forms it chose), then one a form of the matrices (T.147), then the attention without subgroups.
function layerTables(c, cpu, runs, { e16, q8, cpuLogits, logits, scale }) {
  const ref = c.reference, n = ref.tokens.length - 1;
  const [dim, , every, heads, kvHeads] = ref.header, layers = ref.attending ?? every, kvDim = (c.headDim || dim / heads) * kvHeads;
  const exact = { keys: ref.keys, values: ref.values }, half = { keys: ref.keys16, values: ref.values16 };
  const eight = { keys: ref.keys8, values: ref.values8 };
  // (T.562: H16's distance, the f16 tiles' yardstick)
  const h16 = ref.keysH ? Math.max(...Array.from({ length: layers }, (_, l) => {
    const layer = (b64) => floats(b64).subarray(l * n * kvDim, (l + 1) * n * kvDim);
    return Math.max(worstRow(layer(ref.keysH), layer(ref.keys), kvDim), worstRow(layer(ref.valuesH), layer(ref.values), kvDim));
  })) : NaN;
  const byLayer = (got, want) => Array.from({ length: layers }, (_, l) => {
    const layer = (b64) => floats(b64).subarray(l * n * kvDim, (l + 1) * n * kvDim);
    return Math.max(worstRow(layer(got.keys), layer(want.keys), kvDim), worstRow(layer(got.values), layer(want.values), kvDim));
  });
  const e = (x) => x.toExponential(1), letter = (i) => String.fromCharCode(65 + i);
  const columns = [["CPU", cpu], ...runs.map((run, i) => [letter(i), run])];
  const tables = [[`against NumPy's (E16: NumPy's with its cache in float16; Q8: and the matrices' inputs in 8 bits)`, exact,
    [["E16", half], ["Q8", eight]]], ["against NumPy's with its cache in float16", half, []],
    ["against Q8 (T.187: what the packed shaders are held to)", eight, []]];
  for (const [title, want, more] of tables) {
    const table = [...more, ...columns].map(([, got]) => byLayer(got, want));
    console.log(`\nkeys and values of ${c.id} by layer, ${title}: the worst row, its largest difference over its largest value\n`);
    console.log(`| layer | ${[...more, ...columns].map(([name]) => name).join(" | ")} |`);
    console.log(`|---:|${table.map(() => "---:").join("|")}|`);
    for (let l = 0; l < layers; l++) console.log(`| ${l} | ${table.map((column) => e(column[l])).join(" | ")} |`);
    console.log(`| all | ${table.map((column) => e(Math.max(...column))).join(" | ")} |`);
  }
  console.log("");
  const all = (run) => Math.max(...byLayer(run, exact));
  const said = (x) => `logits KL ${e(x.kl)}, the ten most likely ${e(x.ten)}, most likely ${x.token}`;
  console.log(`- CPU: forward.js on the CPU: ${e(all(cpu))}, ${(all(cpu) / e16).toFixed(1)} E16, ${(all(cpu) / q8).toFixed(2)} Q8, ` +
    `the first layer's scale ${e(scale(cpu))}, ${said(cpuLogits)}`);
  runs.forEach((run, i) => {
    console.log(`- ${letter(i)}: ${run.form ?? "no form"}, ${run.attention ?? "no attention"}: ${e(all(run))} ` +
      `(all at once ${run.again ? e(all(run.again)) : "-"}), the worst layer ${run.layer} at ${run.ratio.toFixed(2)} of its line ${e(run.line)}, ` +
      `${(all(run) / e16).toFixed(1)} E16, ${(all(run) / q8).toFixed(2)} Q8, ${(all(run) / h16).toFixed(2)} H16, ${e(Math.max(...byLayer(run, half)))} from NumPy's with its cache in float16, ` +
      `${e(Math.max(...byLayer(run, eight)))} from Q8, the first layer's scale ${e(scale(run))} (against Q8 ${e(scale(run, eight))}), ${said(logits(run.logits))}${logits(run.logits).token === cpuLogits.token ? " as the CPU's" : ""}`);
  });
  const seconds = { "NumPy (with its cache in float16, and Q8)": c.numpySeconds, "the plan": c.planSeconds, CPU: cpu.seconds };
  runs.forEach((run, i) => { seconds[`${letter(i)} (its GPU ready in ${run.readySeconds.toFixed(1)})`] = run.seconds; });
  console.log(`\nseconds of ${c.id}: ${Object.entries(seconds).map(([step, s]) => `${step} ${s.toFixed(1)}`).join(", ")}\n`);
}

export { layerTables };
