// The CPU section's table (T134, T158), and what it leaves for the model page to estimate a CPU's speed from (T157).
// (T353: a part of src/bench.js, which is the window that exports every name of these)
import { number, tableCell, unmeasured } from "./cells.js";

// T157: the GPU section of /benchmark/ against the CPU section's forward pass (forward.js itself, relaxed SIMD and the
// software threads, as the model page runs it). Before T157 the CPU of the GPU's token table was one thread of
// matmul_q8 doubled, which read low: 1.44× on the owner's Android where the CPU section's numbers make about 1.0×.
const STATES = { none: "found nothing to run on here", wrong: "computed something wrong", error: "failed" };

/** The counts of software threads the CPU section measures: 1, 2, 4 and on, doubling, up to the logical cores, and
 * the cores themselves where they are no power of two (T157's review: the owner's Android has 8, and 2 to 4 threads
 * still gave 1.41 to 1.46 times). */
export function threadCounts(cores) {
  const most = Number.isInteger(cores) && cores > 0 ? cores : 4;
  const counts = [];
  for (let n = 1; n <= most; n *= 2) counts.push(n);
  if (counts.at(-1) !== most) counts.push(most);
  return counts;
}

/** A CPU ceiling's value, where it was measured and steady (T163) */
const steadyCeiling = (ceiling, key) => (ceiling && !ceiling.error && !ceiling.none && !ceiling.unsteady && ceiling[key] > 0 ? ceiling[key] : undefined);

// what a CPU row reads a token, GB/s (before T163's review a result had no tokenMegabytes: the checkpoint's bytes),
// and a prompt's G MAC/s (a multiply-add for each weight of the layers and token)
const tokenReads = (r, row) => row.GBps * (r.tokenMegabytes ?? r.megabytes) / r.megabytes;
const promptGMACsOf = (r, row) => (r.layerWeights && row.promptMsPerToken > 0 ? r.layerWeights / (row.promptMsPerToken / 1000) / 1e9 : undefined);

/**
 * The CPU section's Markdown (the lines), from its result r ({backend, shared, megabytes, tokenMegabytes, layerWeights,
 * rows, ceilings}): the forward pass at each count of software threads, and T163's ceilings beside it. A token's share
 * of reading alone with the same count counts all it reads (tokenMegabytes: with relaxed SIMD matmul_q8r's corrections
 * too), while its GB/s column stays the checkpoint's (T157 turns that into other models' speeds). A prompt's G MAC/s
 * with one thread is held against relaxed_dot with its two loads, on one thread; with more threads it is not (threads
 * on cores of different speeds do not add up to a count times one). That loop is the one-token kernel's form (a weight
 * and an activation loaded for each dot); a prompt's tile (T159, four rows by four tokens) loads half a vector of data
 * a dot, so the share can pass 100%, and the text says where its bound lies (T198). No share against a ceiling that
 * failed or was unsteady.
 */
export function cpuTable(r) {
  const c = r.ceilings ?? {};
  const share = (value, whole) => (whole && Number.isFinite(value) ? ` (${number((100 * value) / whole, 0)}%)` : "");
  const readAt = (threads) => steadyCeiling((c.read ?? []).find((one) => one.threads === threads), "GBps");
  const dot = steadyCeiling(c.dot, "GMACs");
  const read = (row) => tokenReads(r, row);
  const promptGMACs = (row) => promptGMACsOf(r, row);
  const ceiling = (one, key, unit) => {
    if (!one) return "not measured";
    if (one.none) return tableCell(one.none === "no relaxed SIMD in this browser" ? "not in this browser" : one.none);
    if (one.error) return tableCell(`failed: ${one.error}`);
    return `${one.unsteady ? "unsteady: " : ""}${number(one[key])} ${unit}`;
  };
  const lines = [`${r.backend}, ${number(r.megabytes, 0)} MB a token${r.shared ? "" : " (not cross-origin isolated: one thread only)"}. ` +
    "In parentheses, the share of this device's ceiling below: a token's reads (with the kernel's corrections) of reading alone with as many threads, " +
    "a prompt's G MAC/s (a multiply-add for each weight of the layers and token) of relaxed_dot with its two loads, one thread. " +
    "A token reads each weight once, so its bound is reading. A prompt's 16 tokens go through tiles of four rows by four tokens (T159): " +
    "for each group of 32 weights a tile loads the four rows' weights once for the four tokens and the four tokens' activations once for the four rows, " +
    "32 relaxed_dots on 16 loads of data (and 12 of scales and corrections), where the loop below loads two for each dot. " +
    "So a prompt can pass 100% of that loop. Its bound lies between that loop and relaxed_dot on registers alone, " +
    "and below the second by the tile's own work: each row, token and group's sum is turned, scaled and added (on arm64, 32 of the 96 vector instructions a tile spends on a group are dots).", "",
    "| software threads | ms a token | GB/s | tok/s | ms a token of a prompt, 16 at once | G MAC/s of the prompt |", "|---:|---:|---:|---:|---:|---:|",
    ...r.rows.map((row) => (row.none ? `| ${row.asked} | ${tableCell(row.none)} | | | | |`
      : `| ${row.threads} | ${number(row.msPerToken)} | ${number(row.GBps)}${share(read(row), readAt(row.threads))} | ${number(1000 / row.msPerToken)} | ` +
        `${number(row.promptMsPerToken, 2)} | ${number(promptGMACs(row))}${row.threads === 1 ? share(promptGMACs(row), dot) : ""} |`))];
  lines.push("", "**Ceilings** (T163): loops of one kind of instruction, each timed as 2n passes less n. " +
    "Reading alone reads the model's weights above, a megabyte at a time taken in turn by the threads. " +
    "relaxed_dot with its two loads reads 8 KB that stay in the first cache, a weight and an activation for each dot, as the one-token kernel (matmul_q8r) does. " +
    "On registers alone it is the instruction's own rate, which no kernel that loads its weights and tokens reaches; a prompt's tiles, with half a load of data for each dot, lie between the two.");
  if (c.error) return [...lines, "", tableCell(unmeasured(c.error))];
  lines.push("", "| loop | software threads | ceiling |", "|---|---:|---:|",
    ...(c.read ?? []).map((one) => `| reading alone | ${one.threads} | ${ceiling(one, "GBps", "GB/s")} |`),
    `| relaxed_dot with its two loads (int8) | 1 | ${ceiling(c.dot, "GMACs", "G MAC/s")} |`,
    `| relaxed_dot, registers only | 1 | ${ceiling(c.dotRegisters, "GMACs", "G MAC/s")} |`,
    `| f32 multiply + add, registers only | 1 | ${ceiling(c.fma, "GMACs", "G MAC/s")} |`);
  return lines;
}

/**
 * The CPU the GPU is held against: of the CPU section's rows, the fastest count of software threads for a token and,
 * each on its own, for a prompt's tokens 16 at once. section: the page's result of the CPU section ({status, data}),
 * or undefined where it has not run. Where it gives nothing to hold against, { why } says so, and no estimate stands
 * in for it.
 */
/** T156: where /benchmark/ keeps its CPU section's fastest reading of the weights (cpuBaseline().token: { GBps,
 * threads }) for the model page, which holds a model on the GPU alone against it (the same estimate as the GPU
 * section's GPU ÷ CPU, T157) */
export const CPU_SPEED_KEY = "benchmark:cpu";
/** T156: where the model page keeps how it is used (the tokens of prompts and written, each generation adding its own
 * to four fifths of what was there: the last few count most), which a model on the GPU alone is weighed by */
export const USAGE_KEY = "gpu:usage";
export const USAGE_DECAY = 0.8;
/** T156: the use kept (USAGE_KEY's value, or nothing) after a generation of prompt tokens and written ones */
export const usedAfter = (kept, prompt, written) => ({ prompt: (kept?.prompt ?? 0) * USAGE_DECAY + prompt, written: (kept?.written ?? 0) * USAGE_DECAY + written });
export function cpuBaseline(section) {
  if (!section) return { why: "run the CPU section for it" };
  if (section.status !== "ok") return { why: `the CPU section ${STATES[section.status] ?? section.status}` };
  const rows = (section.data?.rows ?? []).filter((row) => Number.isFinite(row.msPerToken) && row.msPerToken > 0);
  const fastest = (key) => rows.filter((row) => Number.isFinite(row[key]) && row[key] > 0).sort((a, b) => a[key] - b[key])[0];
  const token = fastest("msPerToken"), prompt = fastest("promptMsPerToken");
  if (!token) return { why: "the CPU section measured no token" };
  const isolated = section.data?.shared !== false;
  return { token: { threads: token.threads, msPerToken: token.msPerToken, GBps: token.GBps, isolated },
           ...(prompt ? { prompt: { threads: prompt.threads, msPerToken: prompt.promptMsPerToken, isolated } } : {}) };
}

export { STATES, steadyCeiling, tokenReads, promptGMACsOf };
