// T192: one line a section for the short report and the page's head.
// (T353: a part of src/bench.js, which is the window that exports every name of these)
import { number, threadsOf, times, timesFaster, noRatios } from "./cells.js";
import { steadyCeiling, tokenReads, promptGMACsOf } from "./cpu.js";
import { tokenWrong, fastestMatVec } from "./gpu.js";
import { stepMs, less, layerSplit } from "./layersteps.js";

// ---- T185: a line of the summary for each section (shortReport()), from the data the section's tables are made of.
// Each says the fastest and the ceilings; the tables with every row are in the whole report.

const yesNo = (value) => (value ? "yes" : "no");

/** The device section's line: what the head's line does not say (the cores and the memory are there) */
export function deviceSummary(r) {
  return [`Browser: SIMD ${yesNo(r.simd)}, relaxed SIMD ${yesNo(r.relaxedSimd)}, 64-bit memory ${yesNo(r.memory64)}, ` +
    `cross-origin isolated ${yesNo(r.crossOriginIsolated && r.sharedMemory)}, WebGPU in a worker ${yesNo(r.webgpu)}, ` +
    `private file system ${yesNo(r.opfs && r.syncHandle)}`];
}

/** The CPU section's line: its fastest token and prompt (cpuTable()'s rows), and the ceilings */
export function cpuSummary(r) {
  const c = r.ceilings ?? {};
  const rows = (r.rows ?? []).filter((row) => row.msPerToken > 0);
  const token = [...rows].sort((a, b) => a.msPerToken - b.msPerToken)[0];
  const prompt = rows.filter((row) => row.promptMsPerToken > 0).sort((a, b) => a.promptMsPerToken - b.promptMsPerToken)[0];
  if (!token) return ["CPU: no token measured"];
  const isolated = r.shared !== false;
  const readAlone = steadyCeiling((c.read ?? []).find((one) => one.threads === token.threads), "GBps");
  const parts = [`${number(token.msPerToken)} ms a token with ${threadsOf({ threads: token.threads, isolated })}, ${number(token.GBps)} GB/s` +
    (readAlone ? ` (${number((100 * tokenReads(r, token)) / readAlone, 0)}% of reading alone)` : "")];
  if (prompt) parts.push(`a prompt ${number(promptGMACsOf(r, prompt))} G MAC/s with ${prompt.threads}`);
  const reads = (c.read ?? []).map((one) => steadyCeiling(one, "GBps")).filter(Boolean);
  const ceilings = [reads.length && `reading alone ${number(Math.max(...reads))} GB/s`,
    steadyCeiling(c.dot, "GMACs") && `relaxed_dot ${number(c.dot.GMACs)} G MAC/s`,
    steadyCeiling(c.fma, "GMACs") && `f32 ${number(c.fma.GMACs)} G MAC/s`].filter(Boolean);
  if (ceilings.length) parts.push(`ceilings: ${ceilings.join(", ")}`);
  return [`CPU: ${parts.join("; ")}`];
}

/**
 * The GPU section's lines, from its steps (as the page's gpuMarkdown() takes them): the adapter and the check; a token
 * of each model with the CPU beside it (tokenTable()'s estimate) and the fastest matrix × vector of the widest shape;
 * the fastest layer and the gain of several tokens a submission; the fastest prompt at the most tokens. No ratio where
 * noRatios() says none or the check found the row WRONG; a step that failed or was not measured has no line (the whole
 * report says why). baseline: cpuBaseline(); gpu: { lost }.
 */
export function gpuSummary(steps, baseline = {}, gpu = {}) {
  const find = (name) => steps.find((s) => s.name === name);
  const a = find("the adapter")?.result ?? {};
  const checkStep = find("the shaders against JavaScript"), check = checkStep?.result;
  const none = noRatios({ fallback: a.fallback, lost: gpu.lost });
  const verdicts = Object.entries(check ?? {});
  const bad = verdicts.filter(([, v]) => v.error || !v.ok).map(([k, v]) => `${k} ${v.error ? "FAILED" : "WRONG"}`);
  const sizes = a.subgroupSizes ? ` (${a.subgroupSizes[0] === a.subgroupSizes[1] ? a.subgroupSizes[0] : a.subgroupSizes.join(" to ")} wide)` : "";
  const lines = [`GPU: ${a.adapter ?? "?"}${a.fallback ? " (a fallback adapter: nothing timed)" : ""}; packed int8 dot ${yesNo(a.packed)}, ` +
    `shader-f16 ${yesNo(a.features?.includes("shader-f16"))}, subgroups ${yesNo(a.features?.includes("subgroups"))}${sizes}; ` +
    (check ? `the check ${verdicts.length - bad.length} of ${verdicts.length} ok${bad.length ? ` (${bad.join(", ")})` : ""}` : `not checked (${checkStep?.error ?? "not run"})`) +
    (gpu.lost ? `; the device was lost (${gpu.lost})` : "")];
  // a token of each model, widened (the table's first row of each): its tok/s and the CPU's estimate beside it
  const cpu = baseline.token;
  const tokens = steps.filter((s) => /^a token of [^,]+$/.test(s.name) && s.result?.tokPerSecond > 0).map((s) => {
    const t = s.result, cpuTok = cpu && t.GB > 0 ? cpu.GBps / t.GB : undefined;
    const ratio = none || tokenWrong(t, check) || !cpuTok ? "" : ` (${times(t.tokPerSecond / cpuTok)} the CPU)`;
    return `${s.name.replace(/^a token of /, "")} ${number(t.tokPerSecond)} tok/s${ratio}`;
  });
  const widest = steps.filter((s) => s.name.startsWith("bandwidth: ")).at(-1);
  const best = widest && fastestMatVec(widest, check);
  const reads = find("the device's ceilings")?.result?.global;
  const share = best && !gpu.lost && !a.fallback && reads?.GBps > 0 && !reads.unsteady ? ` (${number((100 * best.GBps) / reads.GBps, 0)}% of reading a buffer)` : "";
  if (tokens.length || best) {
    lines.push(`GPU, a token: ${[...tokens, best && `the fastest matrix × vector of ${widest.name.replace(/^bandwidth: /, "")} ${best.shader}, ${number(best.GBps)} GB/s${share}`].filter(Boolean).join("; ")}`);
  }
  const layer = find("a layer of a token")?.result;
  const fastestLayer = (layer?.rows ?? []).filter((row) => row.msPerLayer > 0 && !row.unsteady && !(check?.[row.check] && !check[row.check].ok))
    .sort((x, y) => x.msPerLayer - y.msPerLayer)[0];
  const generated = find("tokens generated on the GPU")?.result;
  const one = generated?.rows?.find((row) => row.perSubmission === 1);
  const most = generated?.rows?.filter((row) => row.perSubmission > 1 && row.msPerToken > 0).sort((x, y) => x.msPerToken - y.msPerToken)[0];
  const samplingWrong = ["sampling", "tokens on the GPU"].some((key) => check?.[key] && !check[key].ok);
  const layerParts = [fastestLayer && `the fastest layer ${fastestLayer.form}, ${fastestLayer.dispatches} dispatches, ${number(fastestLayer.msPerLayer, 2)} ms`,
    one && most && `generated ${number(one.msPerToken, 2)} ms a token one a submission, ${number(most.msPerToken, 2)} with ${most.perSubmission}` +
      (none || samplingWrong ? "" : ` (${times(one.msPerToken / most.msPerToken)})`)].filter(Boolean);
  if (layerParts.length) lines.push(`GPU, ${layerParts.join("; ")}`);
  // T202: where the time of the fastest layer the steps' table broke down goes
  const split = find("the steps of a layer")?.result;
  const splitForm = (split?.forms ?? []).filter((form) => stepMs(form) !== undefined && !form.unsteady && !(check?.[form.check] && !check[form.check].ok))
    .sort((x, y) => x.ms - y.ms)[0];
  if (splitForm && !none) {
    const parts = layerSplit(split, splitForm), two = (value) => (value === undefined ? "?" : number(value, 2));
    lines.push(`GPU, where the time of a layer ${splitForm.form} goes: ${two(parts.layer)} ms, the matrices alone ${two(parts.alone)}, fusing adds ${two(less(parts.matrices, parts.alone))}, ` +
      `the attention ${two(parts.attention)}, the norms and quantizing ${two(parts.small)}, the chain ${two(parts.chain)}`);
  }
  const prompt = find("a prompt all at once")?.result;
  const promptTokens = Math.max(0, ...(prompt?.rows ?? []).filter((row) => row.tokens).map((row) => row.tokens));
  const fastestPrompt = (prompt?.rows ?? []).filter((row) => row.tokens === promptTokens && !row.again && row.msPerToken > 0 && !(check?.[row.shader] && !check[row.shader].ok))
    .sort((x, y) => x.msPerToken - y.msPerToken)[0];
  if (fastestPrompt) {
    const ratio = none ? "" : timesFaster(baseline.prompt?.msPerToken, fastestPrompt.msPerToken);
    lines.push(`GPU, a prompt of ${promptTokens} tokens: ${fastestPrompt.shader}, ${number(fastestPrompt.GFLOPS, 0)} GFLOPS${ratio ? ` (${ratio} the CPU)` : ""}`);
  }
  return lines;
}

/** The storage section's line: the writes' MB/s and the read back */
export function storageSummary(r) {
  const mb = (r.mib * 2 ** 20) / 1e6, rate = (part) => number(mb / part.seconds, 0);
  return [`Storage: writes ${rate(r.sequential)} MB/s in order, ${rate(r.scattered)} far apart, ${rate(r.scatteredFlushEach)} with a flush each piece; ` +
    `read back ${rate(r.read)} MB/s${r.read.wrong ? ` (${r.read.wrong} pieces WRONG)` : ""}`];
}

/** The line section's line: the two fetches and the paced reads */
export function lineSummary(r) {
  const fetched = (f) => (f?.error ? "failed" : f ? `${number(f.MBps)} MB/s (first byte ${number(f.firstByteMs, 0)} ms)` : "not measured");
  const paced = (r.paced ?? []).map((p) => (p.slower ? "slower" : p.error ? "failed" : number(p.MBps, 2)));
  return [`Line: this site ${fetched(r.site)}, huggingface.co ${fetched(r.hf)}` +
    (paced.length ? `; huggingface.co read no faster than ${r.paced.map((p) => p.rate).join(", ")} MB/s: ${paced.join(", ")}` : "")];
}
