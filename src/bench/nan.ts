// T263: the GPU section's table of NaN and infinities (the step "nan", src/runtime/benchmark/gpu/nan.ts), and its line
// in the summary: what each case made on this device, what it should make, and which of the engine's marks stands on it.
// (A part of src/bench.ts, which is the window that exports every name of these)
import { tableCell, unmeasured } from "./cells.ts";
import type { Step } from "./cells.ts";
import type { NanFound, NanGot } from "../runtime/contracts/benchmark.ts";

// The marks of the engine that stand on a NaN or an infinity coming through, as the table names them. The engine sees
// what is no finite number by its bits, never by a float's comparison (WGSL lets an implementation assume there is
// none, §15.7): T219's sampler refuses a step whose logits hold a NaN or +Inf or are all −Inf, T241's quantizing gives a
// group with one a NaN scale, and the CPU refuses the keys and values of a request where a float16 of them has its
// exponent all ones (forward/staging.ts's stagingFinite()). Each holds only where what a layer made reaches it.
const STREAM = "a NaN or an infinity of a layer reaches the logits, which T219's sampler refuses";
const ALL_LOW = "T219's sampler refuses logits that are all −Inf";
const SCALE = "T241: a group's NaN scale times its integer sum (0 where its values are) stays NaN";
const SOFTMAX = "the attention carries a NaN or an infinite score through its softmax";
const FFN = "the FFN carries a gate that is no finite number";
const BITS = "T219's sampler and T241's quantizing see what is no finite number by its bits";
const KV = "the CPU refuses keys and values the GPU wrote where a float16 of them is no finite number (its exponent all ones)";
const KV_READ = "the attention reads the cache's float16 keys and values so";
const F16 = "llama.cpp's f16 tiles of a prompt and the f16 attention carry a NaN";
// what is no finite number, either of them
const NOT_FINITE = ["NaN", "+Inf", "−Inf"];
// a case by its name (shaders/bench.ts's NAN_CASES has a row of each: tests/bench-nan-check.ts holds the two lists to one
// another): what carries (any of wants, as nanClass() says what a case made), and the mark that stands on it, where one
// does. A case with no mark is there for what the device does (a NaN from max(), clamp()...): the engine stands on
// none of it, and it says no warning whatever it made.
export const NAN_READINGS: Record<string, { wants: string[]; relied?: string }> = {
  "a NaN loaded and stored": { wants: ["NaN"], relied: STREAM },
  "+Inf loaded and stored": { wants: ["+Inf"], relied: STREAM },
  "−Inf loaded and stored": { wants: ["−Inf"], relied: ALL_LOW },
  "1.5 + NaN": { wants: ["NaN"], relied: STREAM },
  "NaN × 0": { wants: ["NaN"], relied: SCALE },
  "+Inf × 0": { wants: ["NaN"] },
  "+Inf − +Inf": { wants: ["NaN"] },
  // (exp() of a NaN alone is no mark's: SwiftShader makes it +Inf, and its softmax and SiLU carry the NaN all the same,
  // +Inf against +Inf or a NaN over +Inf: the rows of those are the marks', bench.yml's run 45879339792)
  "exp(NaN)": { wants: ["NaN"] },
  "exp(+Inf)": { wants: ["+Inf"] },
  "exp(−Inf)": { wants: ["0"] },
  "max(NaN, 1)": { wants: ["NaN"] },
  "max(1, NaN)": { wants: ["NaN"] },
  "min(NaN, 1)": { wants: ["NaN"] },
  "clamp(NaN, −9, 9)": { wants: ["NaN"] },
  "tanh(NaN)": { wants: ["NaN"] },
  "SiLU of a NaN gate": { wants: ["NaN"], relied: FFN },
  "SiLU of a −Inf gate": { wants: NOT_FINITE, relied: FFN },
  "GEGLU of a NaN gate": { wants: ["NaN"], relied: FFN },
  "softmax of the scores 1 and NaN": { wants: ["NaN"], relied: SOFTMAX },
  "softmax of the scores NaN and 1": { wants: ["NaN"], relied: SOFTMAX },
  "softmax of the scores 1 and +Inf": { wants: NOT_FINITE, relied: SOFTMAX },
  "NaN × 2 found by its bits": { wants: ["yes"], relied: BITS },
  "+Inf × 2 found by its bits": { wants: ["yes"], relied: BITS },
  "3e38 × 1 taken by its bits for a finite number": { wants: ["yes"], relied: BITS },
  "NaN × 2 found by x != x": { wants: ["yes"] },
  "pack2x16float(NaN)": { wants: NOT_FINITE, relied: KV },
  "pack2x16float(+Inf)": { wants: NOT_FINITE, relied: KV },
  "pack2x16float(−Inf)": { wants: NOT_FINITE, relied: KV },
  "pack2x16float(1e6), past float16's largest": { wants: ["+Inf"] },
  "unpack2x16float of a float16 NaN": { wants: ["NaN"], relied: KV_READ },
  "unpack2x16float of a float16 +Inf": { wants: ["+Inf"], relied: KV_READ },
  "f16(NaN)": { wants: NOT_FINITE, relied: F16 },
  "f16(+Inf)": { wants: NOT_FINITE, relied: F16 },
  "f16(70000), past float16's largest": { wants: ["+Inf"] },
  "a float16 NaN × 1 in f16": { wants: ["NaN"], relied: F16 },
  "a float16 NaN + 1 in f16": { wants: ["NaN"], relied: F16 },
  "f32 of a float16 NaN": { wants: ["NaN"], relied: F16 },
};

// a finite number as the table writes it (six significant digits at the most: 0, 1, 65504, 3e+38)
const shown = (value: number) => String(Number(value.toPrecision(6))).replace(/^-/, "−");
/** What a case made, by its kind: an f32's bits or a float16's as "NaN", "+Inf", "−Inf" or the number; a flag as "yes"
 * (1) or "no" (0), else its word in hexadecimal. By the bits alone: nothing here is a float's comparison. */
export function nanClass(kind: NanGot["kind"], word: number): string {
  if (kind === "flag") return word === 1 ? "yes" : word === 0 ? "no" : `0x${(word >>> 0).toString(16)}`;
  if (kind === "half") {
    const exponent = (word >> 10) & 31, fraction = word & 1023, negative = (word & 0x8000) !== 0;
    if (exponent === 31) return fraction ? "NaN" : negative ? "−Inf" : "+Inf";
    const value = (exponent ? (1 + fraction / 1024) * 2 ** (exponent - 15) : (fraction / 1024) * 2 ** -14) * (negative ? -1 : 1);
    return shown(value);
  }
  const magnitude = (word & 0x7fffffff) >>> 0;
  if (magnitude > 0x7f800000) return "NaN";
  if (magnitude === 0x7f800000) return word >>> 31 ? "−Inf" : "+Inf";
  return shown(new Float32Array(new Uint32Array([word >>> 0]).buffer)[0]);
}

// a case read: what it made (and what its f32 store held, where that is something else), and whether that carries
type NanRead = { got: NanGot; made?: string; stored?: string; carries: boolean; wants: string[]; relied?: string };
const readOf = (got: NanGot): NanRead => {
  const { wants = [], relied } = NAN_READINGS[got.name] ?? {};
  if (got.got === undefined) return { got, carries: false, wants, relied };
  const made = nanClass(got.kind, got.got), stored = got.stored === undefined ? undefined : nanClass("float", got.stored);
  return { got, made, stored: stored === made ? undefined : stored, carries: wants.includes(made) && (stored === undefined || wants.includes(stored)), wants, relied };
};

/** The marks of the engine that do not hold on this device: the cases that ran, that a mark stands on and that did not
 * carry, as "1.5 + NaN (made 0)". [] where all held (or nothing ran). */
export function nanBroken(found: NanFound | undefined): string[] {
  return (found?.cases ?? []).map(readOf).filter((read) => read.made !== undefined && read.relied && !read.carries)
    .map((read) => `${read.got.name} (made ${read.made}${read.stored ? `, stored ${read.stored}` : ""})`);
}

/** The table of the step "nan" (s: {name, result: {cases}} or {name, error}): a row a case, what it should make, what it
 * made, and the engine's mark that stands on it. A case a mark stands on that did not carry says "failed" (warnings()
 * lists the row): that mark does not hold on this device. A case no mark stands on says what the device did, and no
 * warning. */
export function nanTable(s: Step<NanFound> | undefined): string[] {
  const head = "**NaN and infinities** (T263): each case once on this device, a thread a case, what it made read back as its bits. " +
    "The engine sees a value that is no finite number by its bits, never by comparing floats (WGSL lets a device assume there is none), " +
    "and its marks hold only where a NaN or an infinity comes through the arithmetic, the stores and the conversions to float16 as one: " +
    "a mark that does not hold here says so in its row, and is among the warnings. The rows no mark stands on are there for what this device does: " +
    "a device may give either there (max() of a NaN, clamp(), x != x), and they are no warning.";
  if (!s?.result) return [head, "", unmeasured(s?.error)];
  const lines = [head, "", "| case | carries where it makes | it made | what stands on it |", "|---|---|---|---|"];
  for (const read of s.result.cases.map(readOf)) {
    const { got, made, stored, carries, wants, relied } = read;
    const cell = got.none ?? (got.error ? `failed: ${got.error}` : `${made}${stored ? ` (stored as ${stored})` : ""}`);
    const mark = made === undefined ? (relied ?? "") : relied ? (carries ? relied : `failed: ${relied}, not here`) : carries ? "nothing: as IEEE 754 has it" : "nothing";
    lines.push(`| ${tableCell(got.name)} | ${wants.join(" or ")} | ${tableCell(cell)} | ${tableCell(mark)} |`);
  }
  return lines;
}

/** The summary's line: how many of the cases a mark stands on carried, and those that did not; undefined where none ran. */
export function nanSummary(s: Step<NanFound> | undefined): string | undefined {
  const reads = (s?.result?.cases ?? []).map(readOf).filter((read) => read.made !== undefined);
  if (!reads.length) return undefined;
  const relied = reads.filter((read) => read.relied), broken = nanBroken(s!.result);
  const others = reads.filter((read) => !read.relied);
  return `GPU, NaN and infinities: ${broken.length ? `${relied.length - broken.length} of the ${relied.length} cases the engine stands on carried, failed: ${broken.join(", ")}`
    : `the ${relied.length} cases the engine stands on all carried`}; of the ${others.length} others ${others.filter((read) => read.carries).length} as IEEE 754 has them`;
}
