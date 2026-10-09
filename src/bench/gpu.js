// The GPU section's tables: the check's verdicts, a matrix × a vector (T149), a token, a layer (T150, T175).
// (T353: a part of src/bench.js, which is the window that exports every name of these)
import { number, threadsOf, times, noRatios, tableCell, unmeasured } from "./cells.js";

/** T186: the numbers under a layer's verdict in the check (T175), in one short line for the page and the report: its
 * quantized vectors held to quantize_x (quantized: [{point, wrong, scale, apart, of}]; the worst scale's relative
 * difference and the values off by 1 of all four, or the first that was wrong) and, for the DP4A fused form, the one
 * with the norms apart (sameAsNormsApart: {ulps, apart, stream, bitForBit}). T225: then the worker's own line of how
 * each stage of the layer held (stages: short where the form is ok, every stage and the first that departed where it
 * is not). "" where the verdict has none of them. */
export function layerCheckNumbers({ quantized, sameAsNormsApart: apart, stages } = {}) {
  const parts = [];
  if (quantized?.length) {
    const wrong = quantized.find((q) => q.wrong);
    const sum = (key) => quantized.reduce((total, q) => total + q[key], 0);
    parts.push(wrong ? `quantized: ${wrong.point} ${wrong.wrong}`
      : `quantized: scales ${Math.max(...quantized.map((q) => q.scale)).toExponential(1)}, ${sum("apart")} of ${sum("of")} off by 1`);
  }
  if (apart) {
    parts.push(`norms apart: ${apart.bitForBit ? "bit for bit" : `${apart.ulps} ulp, ${apart.apart} off by 1, stream ${apart.stream.toExponential(1)}`}`);
  }
  if (stages) parts.push(tableCell(stages));
  return parts.join("; ");
}

/** A verdict of the GPU's check of its shaders against JavaScript, as the GPU section's line of them and the list of
 * warnings (T227) write it. A tiled shader the device refused says so (T146); a packed one also says how many of the
 * values the GPU quantized came out apart from the CPU's; a layer, its quantized vectors and the fused form against
 * the norms apart (T186); the tokens generated, the first that differed, and (T225) the worker's line of how each
 * run's steps held (steps). */
export function checkVerdict([name, v]) {
  // (a device's validation error has line breaks, and with them the verdict is lines of its own in the report)
  if (v.error) return `${name} FAILED (${tableCell(v.error)})`;
  const verdict = `${name} ${v.ok ? "ok" : "WRONG"}`;
  if (v.tokens !== undefined) return `${verdict} (${v.tokens} tokens, ${v.edge} next to a border${v.problems ? `: ${tableCell(v.problems[0])}` : ""}${v.steps ? `; ${tableCell(v.steps)}` : ""})`;
  const apart = v.apart === undefined ? "" : `, quantized ${v.far ? "far apart" : `${number(100 * v.apart, 2)}% apart by 1`}`;
  const numbers = layerCheckNumbers(v);
  return `${verdict} (worst ${v.worstRelative.toExponential(1)}${apart}${numbers ? `; ${numbers}` : ""})`;
}

/** Whether the check of the shaders found the one a token's row ran with wrong (or could not run it): widened or
 * packed, and the choosing on the GPU where it chose. */
function tokenWrong(t, check) {
  if (!check) return false;
  const bad = (key) => check[key] !== undefined && !check[key].ok;
  return bad(t.kind === "packed" ? "packed" : "widen") || (Boolean(t.sample) && bad("argmax"));
}

/** T149: the GPU section's table of an int8 matrix times a vector: a row a shader (T134's two and those of llama.cpp and
 * ONNX Runtime), a column a shape (bandwidths: the steps named "bandwidth: <shape>", {name, result: {rows, cpu,
 * quantize}} or {name, error}), with each GB/s's share of what a loop that only reads a buffer reads (ceilings: T168's).
 * No share where the device was lost (its later times are no GPU's, and too fast: past 100%), on a fallback adapter,
 * or where that read was unsteady. A shader the check found WRONG says so and is never the fastest, nor is a row
 * measured again at the end. gpu: { lost }. */
/** Of a shape's matrix × vector rows, the fastest the check did not find WRONG (not one measured again, nor unsteady) */
function fastestMatVec(s, check) {
  const wrong = (row) => Boolean(check && check[row.check] && !check[row.check].ok);
  return (s.result?.rows ?? []).filter((row) => row.GBps && !row.again && !row.unsteady && !wrong(row)).sort((a, b) => b.GBps - a.GBps)[0];
}
const QUANTIZED_A_TOKEN = 16 * 4 + 1;  // Llama 3.2 1B: q, o, gate and down a layer read an input of their own, and the classifier
export function matVecTable(bandwidths, check, ceilings, gpu = {}) {
  const percent = (part, whole) => `${number((100 * part) / whole)}%`;
  const reads = !gpu.lost && ceilings && !ceilings.fallback && !ceilings.global?.unsteady ? ceilings.global?.GBps : undefined;
  const wrong = (row) => Boolean(check && check[row.check] && !check[row.check].ok);
  const shape = (s) => s.name.replace(/^bandwidth: /, "");
  const shaders = [...new Map(bandwidths.flatMap((s) => (s.result?.rows ?? []).map((row) => [row.shader, row]))).values()];
  const cell = (s, name) => {
    const row = s.result?.rows?.find((one) => one.shader === name);
    if (s.error || !row) return s.error ? tableCell(unmeasured(s.error)) : "";
    if (row.none || row.error) return tableCell(row.none ?? `failed: ${row.error}`);
    return `${row.unsteady ? "unsteady: " : ""}${number(row.GBps)} GB/s${reads ? ` (${percent(row.GBps, reads)})` : ""}`;
  };
  const fastest = (s) => {
    const best = fastestMatVec(s, check);
    return best ? `${shape(s)} ${best.shader}, ${number(best.GBps)} GB/s` : "";
  };
  const quantized = bandwidths.filter((s) => s.result?.quantize?.msEach);
  const wide = quantized.find((s) => s.name.includes("1B"));
  const quantizing = quantized.length ? "**Quantizing the vector** of a packed row (QUANTIZE, one dispatch; not in the packed rows): " +
    quantized.map((s) => `${shape(s)} ${number(s.result.quantize.msEach, 3)} ms`).join(", ") +
    (wide ? `; a token of Llama 3.2 1B quantizes ${QUANTIZED_A_TOKEN} vectors (16 of them 8192 wide), ${number(QUANTIZED_A_TOKEN * wide.result.quantize.msEach, 2)} ms or more` : "") : "";
  return ["**An int8 matrix × vector** (a generated token's). A shader's name says whose form it takes and the rows a workgroup takes. " +
    "Each is timed as a submission of 2n of it less one of n: what waiting for a submission costs is not in it (a token pays it once, in the table of what it costs besides the weights below), " +
    "and each matrix is read from copies of it that make 128 MiB in turn, as a token reads it once, not from the GPU's caches: not to be compared with reports from before T149. " +
    `The packed rows take the vector quantized already${reads ? `. In parentheses, the share of what a loop that only reads a buffer reads, ${number(reads)} GB/s below` : ""}.`,
    ...(gpu.lost ? ["No share of the buffer's reads: the device was lost, and the times after it are no GPU's."] : []), "",
    `| shader | ${bandwidths.map(shape).join(" | ")} |`, `|---|${bandwidths.map(() => "---:|").join("")}`,
    ...shaders.map((row) => `| ${row.shader}${wrong(row) ? " (WRONG in the check)" : ""} | ${bandwidths.map((s) => cell(s, row.shader)).join(" | ")} |`),
    // Safari's kernel on one thread, for scale; the page's forward on the CPU is the CPU section's (T157)
    `| CPU matmul_q8, one thread (not the page's forward) | ${bandwidths.map((s) => (s.result?.cpu ? `${number(s.result.cpu.GBps)} GB/s` : "")).join(" | ")} |`,
    ...(quantizing ? ["", quantizing] : []),
    "", `**Fastest on the GPU** (of the shaders the check found right): ${bandwidths.map(fastest).filter(Boolean).join("; ") || "nothing measured"}`];
}

/**
 * The GPU section's table of a token, with the CPU beside it. steps: the GPU section's steps whose name begins with
 * "a token of " ({name, result: {kind, sample, GB, dispatches, msPerToken, tokPerSecond}} or {name, error});
 * baseline: cpuBaseline(); gpu: { fallback, lost, check } (check: the shaders against JavaScript).
 * The CPU section's model (two layers of Llama 3.2 1B's width) is not the GPU's, so the CPU's speed on each model is
 * an estimate: its weights read at the GB/s the CPU section measured (a token of the model page reads its weights once,
 * T93). GPU ÷ CPU is then the GPU's tok/s over that.
 */
export function tokenTable(steps, baseline, gpu = {}) {
  const cpu = baseline.token, none = noRatios(gpu);
  const lines = [cpu ? `The CPU (an estimate): each model's weights at the CPU section's fastest, ${number(cpu.GBps)} GB/s with ${threadsOf(cpu)}. ` +
      "Its model is as wide as Llama 3.2 1B; a narrower one such as llm-jp-3 150M runs slower than this says. Above 1× the GPU is faster. " +
      "Each number is one run of this page (the same device has differed by more than twice from one run to another)."
    : `The CPU: not measured (${baseline.why}).`,
    ...(none && cpu ? [`GPU ÷ CPU: ${none}.`] : []), "",
    "| a token (weights, dispatches, logits back) | GB | dispatches | GPU ms | GPU tok/s | CPU tok/s (estimate) | GPU ÷ CPU |",
    "|---|---:|---:|---:|---:|---:|---:|"];
  for (const s of steps) {
    const t = s.result ?? {};
    const name = s.name.replace(/^a token of /, "");
    if (s.error || t.error) {
      lines.push(`| ${name} | | | ${tableCell(unmeasured(s.error ?? t.error))} | | | |`);
      continue;
    }
    const cpuTok = cpu && Number.isFinite(t.GB) && t.GB > 0 ? cpu.GBps / t.GB : undefined;
    const wrong = tokenWrong(t, gpu.check);
    const ratio = none || wrong || cpuTok === undefined ? "" : times(t.tokPerSecond / cpuTok);
    lines.push(`| ${name}${wrong ? " (WRONG in the check)" : ""} | ${number(t.GB, 2)} | ${t.dispatches} | ${number(t.msPerToken)} | ` +
               `${number(t.tokPerSecond)} | ${cpuTok === undefined ? "" : number(cpuTok)} | ${ratio} |`);
  }
  return lines;
}

/**
 * T150: the GPU section's table of one layer of a token: its fourteen steps each a dispatch of their own and the same
 * fused into five (public/shaders.js's fusedMatVec), on llama.cpp's matrix × vector; T175: the same on ONNX Runtime's
 * DP4A for small M, its vector quantized before each matrix: separate steps (eighteen), fused but for the norms
 * (eleven) and fused (nine: fusedDp4aMatVec, the norm with the quantizer). step: {name, result: {model, pos, layers,
 * GB, rows: [{form, check, base, fused, normApart, subgroups, dispatches, msPerLayer, GBps, unsteady}, or {form, check,
 * error}, or {form, none}]}} or {name, error}; check: the shaders against JavaScript (a form's verdict under
 * row.check); ceilings: T168's (the buffer's reads, for the share of it the layer's weights are read at); gpu: {
 * fallback, lost }. "Faster than the separate steps" beside a fused row, against the separate steps of the same matrix
 * × vector (base) and reduction (measured in turn with it): not where noRatios() says none, nor for a row the check
 * found WRONG or that was unsteady. The share of the reads: not after a lost device, on a fallback adapter, or where
 * that read was unsteady (as matVecTable). none: a form this device cannot run (no packed int8 dot), and why.
 */
export function layerTable(step, check, ceilings, gpu = {}) {
  if (!step) return [];
  if (step.error || !step.result) return [`**A layer of a token**: ${tableCell(unmeasured(step.error))}`];
  const r = step.result, none = noRatios(gpu);
  const reads = !gpu.lost && ceilings && !ceilings.fallback && !ceilings.global?.unsteady ? ceilings.global?.GBps : undefined;
  const wrong = (row) => Boolean(check && check[row.check] && !check[row.check].ok);
  const usable = (row) => Number.isFinite(row?.msPerLayer) && row.msPerLayer > 0 && !row.unsteady && !wrong(row);
  const faster = (row) => {
    if (none || !row.fused || !usable(row)) return "";
    const separate = r.rows.find((one) => !one.fused && one.subgroups === row.subgroups && one.base === row.base);
    return usable(separate) ? times(separate.msPerLayer / row.msPerLayer) : "";
  };
  // two significant digits under 1 GB/s (a fallback adapter reads a layer at a few hundredths)
  const GBps = (row) => (Number.isFinite(row.GBps) ? `${row.GBps < 1 ? row.GBps.toPrecision(2) : number(row.GBps)}${reads ? ` (${number((100 * row.GBps) / reads)}%)` : ""}` : "");
  return [`**A layer of a token** (${r.model}'s width, at position ${r.pos}, ${number(r.GB * 1000, 1)} MB of weights): on llama.cpp's matrix × vector, its fourteen steps each a dispatch of its own ` +
    "(the norm, q, k, v, RoPE and the cache, the attention, o, the residual's add, the norm, gate, up, SwiGLU, down, the add) and the same fused into five " +
    "(q, k and v with the norm, RoPE and the cache; the attention; o with the add; gate and up with the norm and SwiGLU; down with the add); " +
    "on ONNX Runtime's DP4A, the vector quantized to 8 bits before each matrix, as separate steps (eighteen), fused but for the two norms (eleven), " +
    "and fused (nine: the norm with its quantizing, q, k and v with RoPE and the cache, the attention, its quantizing, o with the add, the norm with its quantizing, " +
    "gate and up with SwiGLU, its quantizing, down with the add). The forms are timed in turn, each as a submission of 2n layers less one of n, the weights read from copies of them in turn, as the matrix × vector is. " +
    "The attention is the prompt's tiles (one row of their four used for a token), in float32 and without subgroups (the engine's own take f16 and subgroups where the device has them: " +
    "the table of a token's attention alone, under the steps', has both); the fused forms a token runs are there again with llama.cpp's flash_attn_vec for it " +
    "(T224: its decode form, the positions split over more workgroups a head, then reduced; with subgroups where there are), one dispatch more where it takes two parts or more. " +
    `GB/s: the layer's weights over its time${reads ? `; in parentheses, the share of what a loop that only reads a buffer reads, ${number(reads)} GB/s below` : ""}.`,
    ...(none ? [`Faster than the separate steps: ${none}.`] : []), ...(gpu.lost ? ["No share of the buffer's reads: the device was lost."] : []), "",
    `| a layer | dispatches | GPU ms | GB/s | its ${r.layers} layers, ms | faster than the separate steps |`, "|---|---:|---:|---:|---:|---:|",
    ...r.rows.map((row) => (row.none ? `| ${tableCell(row.form)} | ${tableCell(`not here: ${row.none}`)} | | | | |`
      : row.error ? `| ${tableCell(row.form)} | ${tableCell(`failed: ${row.error}`)} | | | | |`
      : `| ${tableCell(row.form)}${wrong(row) ? " (WRONG in the check)" : ""} | ${row.dispatches} | ${row.unsteady ? "unsteady: " : ""}${number(row.msPerLayer, 2)} | ` +
        `${GBps(row)} | ${number(row.msPerLayer * r.layers)} | ${faster(row)} |`))];
}

export { tokenWrong, fastestMatVec };
