// The GPU section's table of generated tokens and of the sampling (T151, T191).
// (T353: a part of src/bench.js, which is the window that exports every name of these)
import { number, times, noRatios, tableCell, unmeasured } from "./cells.js";

/**
 * T151: the GPU section's table of tokens generated on the GPU (public/shaders.js's EMBED, fusedMatVec and SAMPLE: the
 * sampling on the GPU too, the state carried from token to token there). step: {name, result: {model, layers, vocab,
 * GB, dispatches, chunkDispatches, tokens, settings, work: {ms, unsteady}, chunkWork, rows: [{perSubmission, msPerToken,
 * fixedMs, chunks}], sampling: {vocab, msEach, over, unsteady, chunks: {msEach, unsteady, pick} (pick: its last stage
 * alone, the same form), flat: {msEach, over, unsteady, chunks}}}} or {name, error} (a time {error} where it failed);
 * check: the shaders against JavaScript ("sampling", "sampling in chunks"
 * and "tokens on the GPU"); gpu: { fallback, lost }. T191: each row also the ms a token with the sampling in chunks
 * (shaders.js's SAMPLER_STAGES, many workgroups) where the rest is SAMPLE's one workgroup. A row a way of submitting: one token a submission, each read back as it comes, and
 * several in one submission, read back once. "A submission besides its tokens" (F) is what a submission costs past its
 * tokens' work (the wait and the reading back: a row's ms a token times its tokens, less a token's work, the
 * difference of 2n tokens in one submission and n), "of it a token" F over the row's tokens (what is left of F a
 * token), and "faster than one a submission" the first row's ms a token
 * over the row's: neither where noRatios() says none, where the check found the sampling or the run WRONG, nor
 * (the first) where a token's work was unsteady. A submission's cost below 0 (a token's work read higher than the
 * row's time allows) is "under the noise".
 */
export function generateTable(step, check, gpu = {}) {
  if (!step) return [];
  if (step.error || !step.result) return [`**Tokens generated on the GPU**: ${tableCell(unmeasured(step.error))}`];
  const r = step.result, none = noRatios(gpu);
  const wrong = ["sampling", "tokens on the GPU"].some((key) => check?.[key] && !check[key].ok);
  const chunksWrong = check?.["sampling in chunks"] && !check["sampling in chunks"].ok;
  const derived = !none && !wrong;
  const one = r.rows.find((row) => row.perSubmission === 1);
  // a submission's cost besides its tokens, or its share a token (what N tokens a submission leave of it: F / N)
  const fixed = (row, ms) => (derived && Number.isFinite(ms) ? (ms < 0 ? "under the noise" : number(ms, 2)) : "");
  const s = r.settings ?? {};
  return [`**Tokens generated on the GPU** (${tableCell(r.model)}: ${r.layers} layers, a vocabulary of ${r.vocab}, ${number(r.GB, 2)} GB of weights a token; ` +
    `a token is ${r.dispatches} dispatches: its row of the embedding, the layers as ${r.layer ? `the layer table's "${tableCell(r.layer)}"` : "T150's fused"}, the classifier, and the sampling on the GPU in one workgroup (T151), ` +
    `penalty ${s.penalty}, temperature ${s.temperature}, top-p ${s.topp}). The same ${r.tokens} tokens each way, the ways in turn; ` +
    "a token's work alone is a submission of 2n tokens less one of n. A submission besides its tokens: what it costs past their work (submitting, waiting, reading the ids back)." +
    " With the sampling in chunks: the same tokens sampled by a workgroup a chunk of the vocabulary" +
    `${r.chunkDispatches ? ` (${r.chunkDispatches} dispatches a token)` : ""}, the ways and the samplers in turn (T191).` +
    `${wrong ? " The check found the sampling on the GPU WRONG." : ""}${chunksWrong ? " The check found the sampling in chunks WRONG." : ""}`,
    ...(none ? [`Besides its tokens and faster: ${none}.`] : []), "",
    "| tokens a submission | GPU ms a token | with the sampling in chunks, ms | a submission besides its tokens, ms | of it a token, ms | faster than one a submission |",
    "|---|---:|---:|---:|---:|---:|",
    ...r.rows.map((row) => `| ${row.perSubmission === 1 ? "1, each read back as it comes" : `${row.perSubmission}, read back once`}${wrong ? " (WRONG in the check)" : ""} | ` +
      `${number(row.msPerToken, 2)} | ${Number.isFinite(row.chunks) ? `${number(row.chunks, 2)}${chunksWrong ? " (WRONG in the check)" : ""}` : ""} | ` +
      `${fixed(row, row.fixedMs)} | ${fixed(row, row.fixedMs / row.perSubmission)} | ` +
      `${derived && row !== one && one ? times(one.msPerToken / row.msPerToken) : ""} |`),
    "", `A token's work: ${r.work ? `${r.work.unsteady ? "unsteady: " : ""}${number(r.work.ms, 2)} ms` : "not measured here"}` +
    `${r.chunkWork ? ` (with the sampling in chunks ${r.chunkWork.unsteady ? "unsteady: " : ""}${number(r.chunkWork.ms, 2)} ms)` : ""}; ` +
    `the sampling alone (made-up logits of ${r.sampling?.vocab ?? "Llama 3's"} tokens, the same settings but no penalty: ` +
    `it would change them in place, sampling after sampling): ${r.sampling ? samplingCell(r.sampling) : "not measured here"}` +
    `${r.sampling?.flat ? `; on flat logits, ${samplingCell(r.sampling.flat)}` : ""}.`];
}
// the ms of the sampling alone, with how many tokens were over the nucleus's floor (what SAMPLE gathers and reads each
// round of its searches: T151), in one workgroup and (T191) in chunks, and of those the last stage alone (T191's
// review: one workgroup's nucleus and draw, SAMPLE's search; the rest of SAMPLE is its reading of the vocabulary)
const samplingTime = (s) => (s.error ? `failed: ${tableCell(s.error)}` : `${s.unsteady ? "unsteady: " : ""}${number(s.msEach, 3)} ms`);
const samplingCell = (s) => `${s.chunks ? "one workgroup " : ""}${samplingTime(s)}` +
  `${Number.isFinite(s.over) ? ` (${s.over} tokens over the floor)` : ""}` +
  `${s.chunks ? `, in chunks ${samplingTime(s.chunks)}` : ""}` +
  `${s.chunks?.pick ? ` (of it the last stage, one workgroup's nucleus and draw: ${samplingTime(s.chunks.pick)})` : ""}`;
