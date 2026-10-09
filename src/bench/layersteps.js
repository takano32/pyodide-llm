// The GPU section's table of where a layer's time goes (T202, T208), and its attention by the cache's length.
// (T353: a part of src/bench.js, which is the window that exports every name of these)
import { number, times, noRatios, tableCell, unmeasured } from "./cells.js";

// T202: a time of layerSteps() in ms (undefined where it failed or is not a positive number)
const stepMs = (one) => (one && !one.error && Number.isFinite(one.ms) && one.ms > 0 ? one.ms : undefined);
const less = (a, b) => (a === undefined || b === undefined ? undefined : a - b);
// T202: a form's layer (r: layerSteps()'s result) and its parts in ms: the steps one by one (steps) as the matrices
// with what they write (matrices), the attention and the norms and quantizing (small); the four matrices alone
// (alone); the chain (the layer less its steps). undefined where a part was not timed.
function layerSplit(r, form) {
  const byName = new Map(r.steps.map((one) => [one.step, one]));
  const sum = (kinds) => form.steps.filter(({ step }) => kinds.includes(byName.get(step)?.kind))
    .reduce((total, { step, count }) => (total === undefined || stepMs(byName.get(step)) === undefined ? undefined : total + count * byName.get(step).ms), 0);
  const aloneRows = r.steps.filter((one) => one.kind === "alone");
  const alone = aloneRows.length && aloneRows.every((one) => stepMs(one) !== undefined) ? aloneRows.reduce((total, one) => total + one.ms, 0) : undefined;
  const layer = stepMs(form), steps = sum(["matrix", "attention", "small"]);
  return { layer, steps, matrices: sum(["matrix"]), attention: sum(["attention"]), small: sum(["small"]), alone, chain: less(layer, steps) };
}

/**
 * T202: the GPU section's table of where a layer's time goes (public/benchmark/gpu.js's layerSteps()): each step of
 * the fused forms a token would run on this device alone, the four matrices alone, a dispatch of one workgroup, and
 * the whole layer, timed in turn. step: {name, result: {model, pos, layers, GB, dp4a, forms: [{form, check, normApart,
 * dispatches, steps: [{step, count}], ms, unsteady} or {form, check, error}], steps: [{step, kind ("matrix",
 * "attention", "small", "alone", "floor"), matrix, ms, unsteady} or {step, kind, error}]}} or {name, error}; check:
 * the shaders against JavaScript (a form's verdict under its check); ceilings: T168's (the share of the buffer's reads
 * the matrices alone read the weights at); gpu: {fallback, lost}. Under the table, for each form: the layer, its
 * steps' sum and what the chain costs beyond it, the matrices alone and the layer less them split into what fusing adds
 * to the matrices (their writes; the norm on the read and gate with up in one workgroup where so), the attention, the
 * norms and quantizing, and the chain. cycleMB: the fewest MB of weights read before the same ones again. T208: chosen,
 * how the forms were chosen ({by: "layer", fastest}: the layer table's fastest fused form the check found right, and
 * beside it the same with the norms folded or apart; {by: "packed", why}: as T202, by the packed int8 dot); caches,
 * the attention's ({count, MB}: the step alone reads them in turn); timestamps, the whole layer by the GPU's own clock
 * ({layers, rounds, forms: [{form, ms}]}, {none} or {error}), one line under the forms'.
 */
export function layerStepsTable(step, check, ceilings, gpu = {}) {
  if (!step) return [];
  const title = "**Where a layer's time goes**";
  if (step.error || !step.result) return [`${title}: ${tableCell(unmeasured(step.error))}`];
  const r = step.result, none = noRatios(gpu);
  const reads = !gpu.lost && ceilings && !ceilings.fallback && !ceilings.global?.unsteady ? ceilings.global?.GBps : undefined;
  const wrong = (form) => Boolean(check && check[form.check] && !check[form.check].ok);
  const microseconds = (one) => (one?.error ? `failed: ${tableCell(one.error)}` : `${one?.unsteady ? "unsteady: " : ""}${stepMs(one) === undefined ? "?" : number(1000 * one.ms, 1)}`);
  const floor = r.steps.find((one) => one.kind === "floor"), forms = r.forms.filter((form) => !form.error);
  const GBps = (took) => (took === undefined ? "?" : `${number(r.GB / (took / 1000))} GB/s${reads ? `, ${number((100 * r.GB) / (took / 1000) / reads)}% of the buffer's reads (${number(reads)} GB/s)` : ""}`);
  const named = forms.map((form) => `"${tableCell(form.form)}"`).join(" and ");
  const which = r.chosen?.by === "layer"
    ? `the layer a token would run here: the fastest fused layer of the table above that the check found right, "${tableCell(r.chosen.fastest)}", as the engine chooses it (T152)` +
      `${forms.length > 1 ? `, and beside it the same with its norms ${forms.find((form) => form.form !== r.chosen.fastest)?.normApart ? "apart" : "folded in"}` : ""}`
    : `the layers a token would run here by the packed int8 dot alone (${r.chosen?.why ? `${tableCell(r.chosen.why)}: ` : ""}` +
      `${r.dp4a ? "ONNX Runtime's DP4A, the packed int8 dot is here" : "llama.cpp's matrix × vector, no packed int8 dot here"})`;
  const lines = [`${title} (T202, T208; ${r.model}'s width at position ${r.pos}, ${number(r.GB * 1000, 1)} MB of weights a layer, ${which}${named ? `: ${named}` : ""}). ` +
    "Each step of a layer alone, as a submission of 2n of its one dispatch less one of n; the four matrices alone (the plain matrix × vector over all of a matrix's rows, " +
    "nothing folded into what it writes: what the matrices themselves take); a dispatch of one workgroup that does next to nothing (what a step costs for being a dispatch in a chain); " +
    "and the whole layer: all timed in turn, the residual stream written back before every submission. " +
    `The layer reads its copies of the weights in turn, a matrix alone or a step with a matrix the ranges of its size of all of them in turn${r.spares ? ` and of ${r.spares} spare matrices of random weights` : ""}` +
    `${Number.isFinite(r.cycleMB) ? ` (none read again before ${number(r.cycleMB, 0)} MB of other weights` : " (none read again soon"}: not from the GPU's caches). ` +
    "Each step alone, not the layer less that step: a small step is 1 or 2% of a layer, about what a layer's time moves from one pair of submissions to the next, " +
    "so the difference of two layers could not tell it; alone it is repeated until a submission takes long enough. What the steps alone leave out is the layer less their sum: " +
    "what a chain of different dispatches, each waiting for the one before, costs beyond each on its own. " +
    "The fused matrices less the matrices alone is what fusing adds to them: the writes folded in (RoPE and the cache, the residual's add, SwiGLU), in llama.cpp's fused form the norm folded into the read, " +
    "and gate with up in one workgroup, which can take less than the two alone (so this can come out less than nothing). " +
    (r.caches?.count > 1
      ? `The attention alone reads ${r.caches.count} copies of the cache (the keys and values of the positions up to this one) in turn, ${number(r.caches.MB, 0)} MB, so not from the GPU's caches, as a token's layers each read their own; ` +
        "in the whole layer it reads one cache, with a copy of the layer's weights read between two reads of it, as in the layer table."
      : "The attention reads one cache (the positions up to this one) every time, in the GPU's caches as in the layer table."),
  ...(none ? [`Read nothing from these times: ${none.replace(/^none:? /, "")}.`] : []), "",
  `| step | µs each | ${forms.map((form) => `in "${tableCell(form.form)}"${wrong(form) ? " (WRONG in the check)" : ""}, µs a layer`).join(" | ")} |`,
  `|---|---:|${forms.map(() => "---:|").join("")}`,
  ...r.steps.map((one) => `| ${tableCell(one.step)} | ${microseconds(one)} | ${forms.map((form) => {
    const count = form.steps.find((s) => s.step === one.step)?.count;
    return count ? `${count} × = ${stepMs(one) === undefined ? "?" : number(1000 * count * one.ms, 1)}` : "";
  }).join(" | ")} |`), ""];
  for (const form of r.forms) {
    if (form.error) {
      lines.push(`- "${tableCell(form.form)}": failed: ${tableCell(form.error)}`);
      continue;
    }
    const { layer, steps, matrices, attention, small, alone: aloneMs, chain } = layerSplit(r, form);
    const signed = (value) => (value === undefined ? "?" : `${value < 0 ? "−" : ""}${number(Math.abs(value), 2)}`);
    lines.push(`- "${tableCell(form.form)}"${wrong(form) ? " (WRONG in the check)" : ""}, ${form.dispatches} dispatches: the layer ${form.unsteady ? "(unsteady) " : ""}${signed(layer)} ms. ` +
      `Its steps one by one ${signed(steps)} ms (the matrices with what they write ${signed(matrices)}, the attention ${signed(attention)}, the norms and quantizing ${signed(small)}); ` +
      `the layer less them, what the chain costs beyond each step alone: ${signed(chain)} ms. ` +
      `The matrices alone ${signed(aloneMs)} ms (${GBps(aloneMs)}); the layer less them, ${signed(less(layer, aloneMs))} ms, is ` +
      `what fusing adds to the matrices ${signed(less(matrices, aloneMs))} + the attention ${signed(attention)} + the norms and quantizing ${signed(small)} + the chain ${signed(chain)}. ` +
      `A dispatch of one workgroup takes ${microseconds(floor)} µs: ${form.dispatches} of them ${stepMs(floor) === undefined ? "?" : number(form.dispatches * floor.ms, 2)} ms.`);
  }
  // T202's review: how far this run's times move. Two forms of the same matrices and attention (DP4A's) differ only in
  // their norms and quantizing, so their layers less those should come out the same; on lavapipe in CI (steady by the
  // pairs) they came out 0.4 and 4.2 ms apart in 48, where the chain was −2.4 to 2.7
  const bigSteps = (form) => form.steps.filter(({ step }) => ["matrix", "attention"].includes(r.steps.find((one) => one.step === step)?.kind))
    .map(({ step, count }) => `${step} ${count}`).sort().join("\n");
  if (forms.length === 2 && bigSteps(forms[0]) === bigSteps(forms[1])) {
    const [a, b] = forms.map((form) => { const s = layerSplit(r, form); return less(s.layer, s.small); });
    if (a !== undefined && b !== undefined) {
      lines.push("", `The two forms run the same matrices and attention and differ only in their norms and quantizing, so their layers less those should be the same: ` +
        `${number(a, 2)} and ${number(b, 2)} ms, ${number(Math.abs(a - b), 2)} apart. That is about how far this run's times move: read the chain, and any part, only where it is larger.`);
    }
  }
  if (r.steps.some((one) => one.unsteady) || r.forms.some((form) => form.unsteady)) lines.push("", "Unsteady: the pairs of a submission of n and of 2n were not about twice each other, so those times are rough.");
  // T208: the whole layer once more by the GPU's own clock
  const t = r.timestamps;
  if (t) {
    const head = "The whole layer by the GPU's own clock (timestamp-query), a check of the times above";
    if (t.none || t.error) lines.push("", `${head}: ${t.none ? "not here" : "failed"}: ${tableCell(t.none ?? t.error)}.`);
    else {
      const ms = (value) => (Number.isFinite(value) && value > 0 ? number(value, 2) : "?");
      const got = t.forms.map(({ form, ms: mean, span }) => {
        const above = stepMs(r.forms.find((one) => one.form === form));
        return `"${tableCell(form)}" ${ms(mean)} ms (first to last ${ms(span)}; above: ${above === undefined ? "?" : number(above, 2)})`;
      });
      lines.push("", `${head}: ${got.join(", ")} a layer, each the median of ${t.rounds} submissions of ${t.layers} layers, each layer a compute pass that writes a timestamp as it begins and as it ends: ` +
        `the mean of the passes, and in parentheses from the first pass's beginning to the last one's end over the ${t.layers} layers. ` +
        "The times above are a submission of 2n layers less one of n, by the page's clock; these hold only what runs on the GPU. " +
        "Chrome cuts every timestamp down to a multiple of 65.5 µs (its own words say 100 µs; unless its developer features are on), so a layer's time is off by up to that, " +
        `the mean of ${t.layers} by 6 µs or less where the passes start anywhere on that clock's steps, and first to last by ${number(65.5 / t.layers, 0)} µs or less whatever they do. ` +
        "First to last also holds what comes between two passes; a mean above it says the passes overlapped (one stamped as begun before the one before it ended) and is not the layers' work. " +
        "Where these agree with the times above to about 0.1 ms, the times above are the layers' work; where these are well below, the times above hold something besides it (what comes between one layer and the next, or of the submissions).");
    }
  }
  lines.push(...attentionLengthsLines(r.lengths, none, reads));
  return lines;
}

/**
 * T224: under the steps' table, a token's attention alone at 128, 1024 and 4096 positions (public/benchmark/gpu.js's
 * attentionLengths()): the prompt's tiles, and llama.cpp's flash_attn_vec with subgroups and with the lanes of the
 * workgroup standing for one. lengths: { positions, rows: [{ attention, tiles, times: [{ ms, n, ratio, unsteady } or
 * { error } or { none } a length] }], base, MB } or { error }; none: noRatios()'s (no "faster than the tiles" then). The
 * vec rows say how many times faster they are than the tiles at the same length (neither unsteady): T224's review, the
 * tiles of row base, the engine's here where there are two (f32 without subgroups as the layer rows, and f16 and
 * subgroups as the engine makes them where the device has them). bytes (a result of the review's, optional): the bytes
 * of the keys and values of each length read once; every cell then says GB/s of them, and with reads (the GB/s of a loop
 * that only reads a buffer: T168's, the layer steps' `reads`) their share: a head of q reads its keys and values
 * again for every head of q that shares them (4 on Llama 3.2 1B's), so past 100% the caches serve the rereads.
 */
function attentionLengthsLines(lengths, none, reads) {
  if (!lengths) return [];
  const head = "**A token's attention alone, by the positions it reads** (T224)";
  if (lengths.error) return ["", `${head}: failed: ${tableCell(lengths.error)}`];
  const tiles = lengths.rows[lengths.base ?? 0], two = lengths.rows.filter((row) => row.tiles).length > 1;
  const cell = (one, i, row) => {
    if (one?.none) return tableCell(`not here: ${one.none}`);
    if (one?.error || !(Number.isFinite(one?.ms) && one.ms > 0)) return `failed: ${tableCell(one?.error ?? "no time")}`;
    const base = tiles.times[i], steady = !one.unsteady && !base?.unsteady && Number.isFinite(base?.ms) && base.ms > 0;
    const faster = row !== tiles && !row.tiles && !none && steady ? `${times(base.ms / one.ms)} the ${two ? "engine's " : ""}tiles` : "";
    const rate = lengths.bytes?.[i] > 0 && !one.unsteady ? lengths.bytes[i] / (one.ms / 1000) / 1e9 : undefined;
    const reading = rate === undefined ? "" : `${number(rate)} GB/s${reads ? `, ${number((100 * rate) / reads, 0)}% of the buffer's reads` : ""}`;
    const notes = [faster, reading].filter(Boolean).join("; ");
    return `${one.unsteady ? "unsteady: " : ""}${number(1000 * one.ms, 1)}${notes ? ` (${notes})` : ""}`;
  };
  return ["", `${head}: Llama 3.2 1B's heads (32 of q on 8 of keys and values, 64 each), a token that reads 128, 1024 and 4096 positions of the cache, ` +
    "each attention as a submission of 2n of it less one of n, all in turn at a length, each on the next of copies of the cache " +
    `(${number(lengths.MB, 0)} MB of them at the most: not from the GPU's caches, as a token's layers each read their own). ` +
    "The prompt's tiles run a workgroup a head at any length and use one row of their four for a token; llama.cpp's flash_attn_vec (its decode form) splits the positions " +
    "over more workgroups a head as they grow (up to the least subgroup, or 32 where the lanes of a workgroup stand for one), then a second dispatch reduces the parts. " +
    (two ? "The tiles are here twice: in float32 without subgroups, as the layer table and the steps above run them, and as the engine makes them on this device (f16, subgroups), " +
      "which is what a token's attention is chosen against, and what the vec rows are read against. " : "") +
    (lengths.bytes ? "GB/s: the keys and values of the length read once over the time" + (reads ? `, and its share of what a loop that only reads a buffer reads (${number(reads)} GB/s)` : "") +
      " (each of the 4 heads of q that share a head of keys and values reads them again, so what the GPU's caches serve of the rereads can take a share past 100%). " : "") +
    "The engine times the ones right on the device at 128 and 2048 positions and takes the fastest.",
  ...(none ? [`Faster than the tiles: ${none}.`] : []), "",
  `| attention | ${lengths.positions.map((p) => `${p} positions, µs`).join(" | ")} |`, `|---|${lengths.positions.map(() => "---:|").join("")}`,
  ...lengths.rows.map((row) => `| ${tableCell(row.attention)} | ${lengths.positions.map((_, i) => cell(row.times[i], i, row)).join(" | ")} |`)];
}

export { stepMs, less, layerSplit };
