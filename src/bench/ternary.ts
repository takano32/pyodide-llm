// T270: the GPU section's table of ternary weights (T232's Ternary Bonsai) beside int8's, and its line in the summary.
// (A part of src/bench.ts, which is the window that exports every name of these)
import { number, times, tableCell, unmeasured } from "./cells.ts";
import type { Ratios, Step } from "./cells.ts";
import type { GpuCeilings, TernaryFound, TernaryRow } from "../runtime/contracts/benchmark.ts";

// the tables of the step, in its order, and what a row of each is
const WHAT: Record<TernaryRow["what"], string> = { matrix: "a token's matrix × vector", tile: "a prompt's tile", layer: "a layer of a token" };
// the two forms of a ternary matrix × vector the summary holds against each other (gpu/ternary.ts's names)
const SIGNED = "ternary, the codes made signed (engine)", SUMS = "ternary, the codes as they are less Σa";

// a ternary tile's int8 row: the same tile with or without subgroups ("ORT DP4A 64×64, subgroups, ternary" and
// "ORT DP4A 64×64, subgroups, int8")
const sameTile = (int8: string, ternary: string) => int8.replace(/, int8$/, "") === ternary.replace(/, ternary$/, "");
const measured = (row: TernaryRow | undefined) => (row && !row.none && !row.wrong && !row.error && row.ms! > 0 ? row : undefined);
// the rows of one table and shape, by their form
const rowOf = (rows: TernaryRow[], { what, shape }: Pick<TernaryRow, "what" | "shape">, form: (name: string) => boolean) =>
  rows.find((row) => row.what === what && row.shape === shape && form(row.form));

/** How much faster the codes as they are less Σa run than the engine's form (the codes made signed), a shape each, as
 * "1.1× on Llama 3.2 1B w1": above 1 the form the engine did not take is faster here (T232's choice to overturn). */
export function ternaryForms(r: TernaryFound | undefined): string[] {
  const rows = r?.rows ?? [];
  return [...new Set(rows.filter((row) => row.what === "matrix").map((row) => row.shape))].flatMap((shape) => {
    const signed = measured(rowOf(rows, { what: "matrix", shape }, (form) => form === SIGNED)), sums = measured(rowOf(rows, { what: "matrix", shape }, (form) => form === SUMS));
    return signed && sums && !signed.unsteady && !sums.unsteady ? [`${times(signed.ms! / sums.ms!)} on ${shape}`] : [];
  });
}

/**
 * The table of the step "ternary" (s: {name, result: {rows, pos, none}} or {name, error}): a row a form of a table
 * (a token's matrix × vector on two shapes, a prompt's tile of 64 tokens, a layer of a token), its ms a dispatch, the
 * weights it multiplies a second (times its tokens: a multiply-add each), the GB/s of the weights and their scales it
 * reads, with their share of what a loop that only reads a buffer reads (ceilings: T168's; none where the device was
 * lost, on a fallback adapter, or where that read was unsteady), and how many times as fast as the int8 row of the same
 * table and shape. gpu: { fallback, lost }.
 */
export function ternaryTable(s: Step<TernaryFound> | undefined, ceilings: GpuCeilings | undefined, gpu: Ratios = {}): string[] {
  const reads = !gpu.lost && ceilings && !ceilings.fallback && !ceilings.global?.unsteady ? ceilings.global?.GBps : undefined;
  const head = ["**Ternary weights** (T270): Ternary Bonsai's weights, two bits a weight (-1, 0 or +1 times a scale a group of 128), beside int8 on the same shapes. " +
    "The GPU multiplies them by the packed int8 dot as the int8 rows do, the codes spread to a byte each as they are read. " +
    "Two forms of a token's matrix × vector: the engine's, which makes every code signed as it reads it (an add and an xor a word of 16 codes), " +
    "and the form Prism ML's llama.cpp and this project's CPU kernel take, which keeps the codes as they are (0, 1 and 2) and takes the vector's sum off the dot (a subtraction a row and group, the sums made once as the vector is loaded). " +
    "Which is faster depends on the device: the engine takes the first until a report says otherwise. " +
    "The prompt's tile and the layer are the engine's forms (ORT's DP4A, T175's fused layer) on int8 and on ternary weights. " +
    "Each is timed as a submission of 2n of it less one of n, in turn with the other rows of its table, on copies of its weights that make 128 MiB (not from the GPU's caches). " +
    "G weights/s counts a multiply-add for each weight and token; GB/s the weights and their scales read" +
    `${reads ? `, with its share of what a loop that only reads a buffer reads (${number(reads)} GB/s)` : ""}; × int8, how many times as fast as the int8 row of the same table and shape.`];
  if (!s || s.error) return [...head, "", tableCell(unmeasured(s?.error))];
  const r = s.result!;
  if (r.none) return [...head, "", tableCell(r.none)];
  const cell = (row: TernaryRow) => {
    if (row.none) return [tableCell(row.none), "", "", ""];
    if (row.wrong) return [tableCell(`WRONG in the check: ${row.wrong}`), "", "", ""];
    if (row.error) return [tableCell(`failed: ${row.error}`), "", "", ""];
    const seconds = row.ms! / 1000, GBps = row.bytes! / seconds / 1e9;
    // the int8 row of the same table and shape (of a tile, the same tile)
    const base = measured(r.rows.find((one) => one.what === row.what && one.shape === row.shape && !one.ternary && (row.what !== "tile" || sameTile(one.form, row.form))));
    return [`${row.unsteady ? "unsteady: " : ""}${number(row.ms, 3)}`, number((row.weights! * row.tokens) / seconds / 1e9),
      `${number(GBps)}${reads ? ` (${number((100 * GBps) / reads, 0)}%)` : ""}`, row.ternary && base ? times(base.ms! / row.ms!) : ""];
  };
  const lines = [...head, "", "| table | shape | form | ms | G weights/s | GB/s | × int8 |", "|---|---|---|---:|---:|---:|---:|",
    ...r.rows.map((row) => `| ${WHAT[row.what]}${row.what === "tile" ? `, ${row.tokens} tokens` : ""} | ${row.shape} | ${row.form} | ${cell(row).join(" | ")} |`)];
  const forms = ternaryForms(r);
  lines.push("", `**The codes as they are less Σa against the engine's form** (above 1×: faster than the engine's here): ${forms.length ? forms.join(", ") : "not measured"}`);
  return lines;
}
