// forward/alone.js (T349): a model on the GPU alone (T156, T210): what the GPU holds of it, whether it goes there
// (weightsPlace, gpuOnlyUnfit) and stays there (aloneVerdict), and its weights as they arrive, the layers' matrices and
// the tables to the GPU's worker and the rest to this memory without the holes they leave (gpuOnlyWeights, gpuOnlyPlan).
// A module of public/forward.js, which asks for it with its own ?v=<build> and exports its names as before; it reads
// its neighbour the same way.

const { BETTER } = await import(new URL(`choice.js${new URL(import.meta.url).search}`, import.meta.url));

// ---- T156: a model on the GPU alone. Where the layers could not be held twice (in this memory and on the GPU), the
// worker decides before a byte comes that the GPU alone takes them (the owner, 2026-09-27: "大きいモデルは最初から GPU
// だけ", T156's B): the layers' matrices (T210: and the tables) go to the GPU's worker as they arrive and are never in
// this memory, the rest (the norms, the header) comes here, packed without the holes they leave (place()). Nothing
// runs on the CPU then, and it keeps none of the keys and values (T210); a GPU that fails means the model is loaded
// again on the CPU (the worker).

// T232: the bytes of a weight on the GPU: an int8 value and a float32 scale a group of 32 (1.125; int6 is widened to
// that, T155), or ternary (T230) as the checkpoint holds it, two bits and a float32 scale a group of 128 (0.28125)
export const GPU_WEIGHT_BYTES = 1 + 4 / 32, GPU_TERNARY_BYTES = 1 / 4 + 4 / 128;
/** T156: the bytes the GPU holds of a Llama with this header and form (FORM): its layers' matrices (int8 values and a
 * float32 scale a group of 32: 1.125 bytes a weight, int6 widened as well, T155; T232, dtype "ternary": 0.28125), its two norms a layer, its own keys
 * and values (float16, the whole context), and for a generation's steps (T152) the classifier (and the embedding where
 * it is another table), RoPE's table of every position and the sampling's three arrays of the vocabulary. forward.js
 * counts the same of a model it runs (layersOnGpu, tokensUnfit). */
export function gpuBytes(header, { head_dim = 0, arch = "llama", dtype = "int8" } = {}) {
  const [dim, hidden, layers, heads, kvHeads, signedVocab, seqLen] = header;
  const vocab = Math.abs(signedVocab), headSize = head_dim || dim / heads, qDim = heads * headSize, kvDim = kvHeads * headSize;
  const each = dtype === "ternary" ? GPU_TERNARY_BYTES : GPU_WEIGHT_BYTES;
  // (GPT-2's and GPT-NeoX's FFN has no gate: two matrices)
  const matrices = qDim * dim + 2 * kvDim * dim + dim * qDim + (arch === "llama" ? 3 : 2) * hidden * dim;
  const layerBytes = layers * matrices * each + layers * 2 * dim * 4 + 2 * layers * seqLen * kvDim * 2;
  const table = vocab * dim * each;
  return layerBytes + table * (signedVocab > 0 ? 1 : 2) + seqLen * headSize * 4 + 3 * vocab * 4;
}

// the layers' matrices of a Llama, which the GPU alone holds (T156)
const LAYER_MATRICES = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"];
// T210: and its tables, the embedding and a classifier of its own (wcls is the embedding's tensor where it is shared):
// the GPU embeds a prompt's rows too (shaders.js's EMBED_ROWS), and nothing of them is here
const GPU_ALONE = [...LAYER_MATRICES, "token_embedding_table", "wcls"];
/** T156: the stretches of the checkpoint the GPU alone holds, the layers' matrices and (T210) the tables ([start, end)
 * of each, in file order: every one is its values and then its scales, llama2_numpy's Tensor), from the tensors
 * llama2_numpy.external_tensors() places */
export function gpuHoles(tensors) {
  const holes = new Map(GPU_ALONE.filter((name) => tensors[name]).map((name) => {
    const t = tensors[name], values = t.shape.reduce((a, b) => a * b, 1);
    return [t.offset, [t.offset, t.scales + (values / t.group) * 4]];
  }));
  return [...holes.values()].sort((a, b) => a[0] - b[0]);
}
/** T156: where an offset of the checkpoint (outside the holes) is in the memory that holds the rest: less the holes
 * before it */
export const placer = (holes) => (offset) => holes.reduce((at, [start, end]) => (end <= offset ? at - (end - start) : at), offset);

/** T156: whether a model goes on the GPU alone, from what is known before its bytes come (the owner's B, 2026-09-27;
 * TODO.md's T156 has the numbers): both, as before (T148: the layers in this memory and on the GPU, and each block and
 * step where it is faster), where both fit half of what the device says it has; else the GPU alone where the model can
 * be (eligible: a Llama the GPU's steps take, T152) and it fits; else the CPU (and the prompts on the GPU where the
 * layers alone fit, gpuRoom, as before). deviceMemory: navigator.deviceMemory, 4 where the browser does not say
 * (Safari, Firefox); Chromium says 8 for 8 GB or more: both are then held to BOTH_ON_8 (6.5 GiB, the owner's choice,
 * 2026-09-27: the models of the list up to 2B both as before, llm-jp-3.1 1.8B's 6.45 GB = 6.01 GiB with the keys and
 * values of 4096 positions on either side too (at 6 GiB it was 11 MB past: the second review of T156); 3B and larger
 * (Qwen2.5 3B's 7.2 GB = 6.70 GiB, T153's risk on an 8 GB phone) not), and the GPU alone to nothing, as the CPU alone is (a 7B's 9.2 GB runs on the CPU
 * there, T132: the GPU alone takes about as much as the CPU alone, the matrices once either way).
 * cpu: the checkpoint and what the forward pass puts after it (footprint); gpuOnly: the checkpoint without the
 * matrices and what the forward pass puts after that (footprint with direct); gpu: gpuBytes. Returns { mode: "both" |
 * "gpu" | "cpu", gpuRoom } */
export const BOTH_ON_8 = 6.5 * 2 ** 30;
export function weightsPlace({ cpu, gpuOnly, gpu, deviceMemory = 4, eligible = false, forced = false }) {
  const room = deviceMemory >= 8 ? BOTH_ON_8 : (deviceMemory * 2 ** 30) / 2;
  if (forced && eligible) return { mode: "gpu" };
  if (cpu + gpu <= room) return { mode: "both", gpuRoom: room - cpu };
  if (eligible && (deviceMemory >= 8 || gpuOnly + gpu <= room)) return { mode: "gpu" };
  return { mode: "cpu", gpuRoom: room - cpu };
}

/** T156: the multiply-adds of a Llama's layers a token of a prompt makes (every weight of its seven matrices once) */
export function layerWeightsOf(header, { head_dim = 0 } = {}) {
  const [dim, hidden, layers, heads, kvHeads] = header;
  const headSize = head_dim || dim / heads, qDim = heads * headSize, kvDim = kvHeads * headSize;
  return layers * (2 * qDim * dim + 2 * kvDim * dim + 3 * hidden * dim);
}
// T156 (the owner, 2026-09-27: the prompt counts too): the tokens of a prompt against the tokens written, where the
// page has not kept how they are used here yet: as many (a chat's question with its template and an answer of about
// the same length; no measurement of the site's visitors says otherwise)
export const USAGE_UNKNOWN = { prompt: 1, written: 1 };
/** T156: whether the CPU would be faster than the GPU for a model on the GPU alone (then it is loaded again on the
 * CPU), by the time of what the page does: usage.prompt tokens of prompts and usage.written tokens written (the page's
 * recent use, decayed, or USAGE_UNKNOWN). The CPU's side is /benchmark/'s CPU section (cpu: { GBps, promptGMACs }): a
 * token written reads the checkpoint once (size bytes at GBps, T157), a prompt's token makes layerWeights multiply-adds
 * at promptGMACs (its model is two layers as wide as Llama 3.2 1B; a narrower model runs slower than that says,
 * T157's review: 1.09 to 1.21 on llm-jp-3 150M's shape). The GPU's side is its own (gpu: { stepMs, promptMs }: a step
 * of a run of GPU_TOKENS, a token of a block of GPU_BLOCK, as gpu.js timed them as it started). A side not known on
 * either leaves its part out; nothing known of the CPU: the GPU stays. Returns { cpuFaster, cpu, gpu } (ms of that use).
 * T232: size is what an int8 model's token reads, 1.125 bytes a weight. A ternary model's checkpoint is a quarter of
 * that, and its CPU kernel is bound by its arithmetic, not by the reading (T231: 1.0 to 2.0 times the int8 kernel's
 * weights a second on the CPUs measured): the caller hands the bytes of the same weights as int8 (cpuReadBytes), so
 * that the CPU is not taken for four times as fast as it is. */
export const cpuReadBytes = (size, dtype) => (dtype === "ternary" ? size * (GPU_WEIGHT_BYTES / GPU_TERNARY_BYTES) : size);
export function aloneVerdict({ size, layerWeights, cpu = {}, gpu = {}, usage = USAGE_UNKNOWN }) {
  const parts = [];
  if (cpu.GBps > 0 && gpu.stepMs > 0) parts.push([usage.written, size / (cpu.GBps * 1e6), gpu.stepMs]);
  if (cpu.promptGMACs > 0 && gpu.promptMs > 0) parts.push([usage.prompt, layerWeights / (cpu.promptGMACs * 1e6), gpu.promptMs]);
  const onCpu = parts.reduce((sum, [n, ms]) => sum + n * ms, 0), onGpu = parts.reduce((sum, [n, , ms]) => sum + n * ms, 0);
  return { cpuFaster: parts.length > 0 && onCpu < BETTER * onGpu, cpu: onCpu, gpu: onGpu };
}
/** T156: whether a verdict the page kept (alone: { key, cpu }, the device's key and /benchmark/'s CPU then) still holds:
 * the same device, browser and shaders (the key, T148's), and the same CPU reading of /benchmark/ (a new run of its
 * CPU section, or none now, asks again) */
export const aloneHolds = (alone, key, cpu) => Boolean(alone && key && alone.key === key && cpu &&
  alone.cpu?.GBps === cpu.GBps && alone.cpu?.promptGMACs === cpu.promptGMACs);

// T220: where the GPU's device binds a buffer (minStorageBufferOffsetAlignment). gpu.js asks its device for the
// adapter's limits of size but not for this one, so the device has WebGPU's default, 256, whatever smaller value the
// adapter says it could give: gpuOnlyUnfit judges by the device's value, the one gpu.js's tokensLayout and piecesOf
// read (asking the device for the adapter's value instead would move where every model's pieces start on the GPU, the
// prompt's too, for no model of the list: they all start on 256, T213)
const BINDS_AT = 256;
/** T156: why a model cannot go on the GPU alone, from its header, dtype and form before its bytes come, and the
 * adapter the worker asked for ({ fallback, limits }); null where it can. A Llama whose steps the GPU takes (T152;
 * T226: Qwen2 and Qwen3 with it, whose biases of q, k and v and norms of the heads are vectors that stay in this
 * memory as the norms' weights do; not GPT-2 or GPT-NeoX: llama2_numpy.external_tensors() places a Llama's tensors
 * alone before the model is built, and no such model of the list is too large to hold twice), int8 (six bits are for a device short of memory, where
 * the widened int8 on the GPU, 1.125 bytes a weight against six bits' 0.875, would not fit either: T155's review) or
 * T232 ternary (where the browser's WGSL has the packed int8 dot: adapter.packed),
 * heads of a multiple of 4, and q, k and v and gate and up each one range of a buffer the device binds (gpu.js's
 * tokensLayout says the last word: a GPU that refuses then means the model is loaded again on the CPU). */
// T219: what a model on the GPU alone says where the GPU sampled an id outside the vocabulary (its logits were not
// finite numbers, most likely: the words of T195's NOT_FINITE, llama2_numpy.py), since no CPU can take the step again
export const OUTSIDE_VOCABULARY = "The model computed logits on the GPU that are not finite numbers (NaN or infinity), so no token can be drawn: its weights are broken or its numbers overflowed.";
export function gpuOnlyUnfit(header, dtype, { arch = "llama", head_dim = 0, rotated = null } = {}, adapter, force = {}) {
  const [dim, hidden, , heads, kvHeads] = header;
  const headSize = head_dim || dim / heads, qDim = heads * headSize, kvDim = kvHeads * headSize;
  if (!adapter) return "no GPU adapter here";
  if (adapter.fallback && !force.fallback) return "a fallback adapter";
  if (arch !== "llama") return "GPT-2 and GPT-NeoX (and a Qwen3.5 and an LFM2) are not placed on the GPU alone: only a Llama's tensors are, by external_tensors()";
  if (rotated) return "a rotated basis is not on the GPU yet";  // T237
  const ternary = dtype === "ternary";
  if (dtype !== "int8" && !ternary) return `${dtype} weights stay on the CPU`;
  if (ternary && !adapter.packed) return "ternary weights need the packed int8 dot product of WGSL, which this browser lacks";
  if (headSize % 4) return "heads of a size that is no multiple of 4";
  const { maxStorageBufferBindingSize, maxBufferSize } = adapter.limits;
  const binds = Math.min(maxStorageBufferBindingSize, maxBufferSize);
  // (a joined matrix's parts start where the device binds, the scales an eighth of the values' bytes on; T232: a
  // ternary weight is a quarter of a byte)
  const bytes = (weights) => (ternary ? weights / 4 : weights);
  const starts = [qDim * dim, (qDim + kvDim) * dim, hidden * dim].map(bytes);
  if (starts.some((values) => values % BINDS_AT || (values / 8) % BINDS_AT)) return "q, k and v or gate and up would not start where this GPU binds a buffer";
  if ([(qDim + 2 * kvDim) * dim, 2 * hidden * dim, dim * Math.max(qDim, hidden)].some((weights) => bytes(weights) > binds)) return "a layer's matrices are past a buffer of this GPU";
  return null;
}

// T156: the GPU's worker takes the layers' bytes as they come, and says in flow[0] how many it has put on the GPU: the
// loops that write wait (room()) where they are more than FLOW_BYTES ahead (a kept model read from the disk comes
// faster than a GPU takes it, and what waits in the messages is memory), and give up where nothing was taken for
// FLOW_STALL_MS. They wait without blocking: a worker made from a worker may not even start while the one that made
// it blocks (Chromium, gpu-check's first run: the GPU's worker took nothing for 60 s)
const FLOW_BYTES = 64 * 2 ** 20, FLOW_STALL_MS = 60000;
/** T156: the weights of a model on the GPU alone. worker: the GPU's worker (gpu.js, told to open with gpuOnlyPlan()),
 * tensors: llama2_numpy.external_tensors(), size: the checkpoint's bytes. Returns { holes, place, stored, flow,
 * write(offset, bytes), room(), drained() }: write() puts the bytes outside the holes into memory at base +
 * place(offset), and posts those in them to the worker (a copy each); room() resolves once the worker is no more than
 * FLOW_BYTES behind, drained() once it has all of them; stored is what memory holds (the checkpoint less the holes). */
export function gpuOnlyWeights({ memory, base, size, tensors, worker }) {
  const holes = gpuHoles(tensors), place = placer(holes);
  const stored = size - holes.reduce((sum, [start, end]) => sum + (end - start), 0);
  const flow = new SharedArrayBuffer(8), taken = new BigInt64Array(flow);
  let sent = 0n;
  const wait = async (ahead) => {
    let seen = Atomics.load(taken, 0), moved = performance.now();
    while (sent - seen > ahead) {
      if (Atomics.waitAsync) await Atomics.waitAsync(taken, 0, seen, 1000).value;
      else await new Promise((resolve) => setTimeout(resolve, 10));
      const now = Atomics.load(taken, 0);
      if (now !== seen) [seen, moved] = [now, performance.now()];
      else if (performance.now() - moved > FLOW_STALL_MS) throw new Error(`the GPU took no weights for ${FLOW_STALL_MS / 1000} s`);
    }
  };
  const post = (offset, chunk, from, to) => {
    const bytes = chunk.slice(from, to);
    worker.postMessage({ type: "weights", offset: offset + from, bytes }, [bytes.buffer]);
    sent += BigInt(to - from);
  };
  return {
    holes, place, stored, flow,
    write(offset, chunk) {
      let at = 0;
      for (const [start, end] of holes) {
        if (end <= offset + at || start >= offset + chunk.length) continue;
        const from = Math.max(start - offset, at), to = Math.min(end - offset, chunk.length);
        if (from > at) new Uint8Array(memory.buffer, base + place(offset + at), from - at).set(chunk.subarray(at, from));
        post(offset, chunk, from, to);
        at = to;
      }
      if (at < chunk.length) new Uint8Array(memory.buffer, base + place(offset + at), chunk.length - at).set(chunk.subarray(at));
    },
    room: () => wait(BigInt(FLOW_BYTES)),
    /** every byte posted is on the GPU (or was dropped by a GPU that failed) */
    drained: () => wait(0n),
  };
}
/** T156: what the GPU's worker opens with (gpu.js's open(): the device, and a buffer for every piece of every layer's
 * matrices, before a byte comes): each matrix's rows and length and, a layer each, where its values and scales start in
 * the checkpoint (the offsets the bytes come with). T210: and the tables' ({ classifier, embedding }, the embedding
 * null where the classifier is it: { rows, n, at: [values, scales] } each, in the checkpoint too) */
export function gpuOnlyPlan(header, tensors, force = {}, remembered) {
  const layers = header[2];
  // (T232, ternary: a weight is two bits of the values, gpu.js's rowBytes)
  const matrices = Object.fromEntries(LAYER_MATRICES.map((name) => {
    const t = tensors[name], [, rows, n] = t.shape, perLayer = rows * n, ternary = t.kind === "ternary";
    return [name, { rows, n, ternary, layers: Array.from({ length: layers }, (_, l) =>
      [t.offset + l * (ternary ? perLayer / 4 : perLayer), t.scales + (l * perLayer / t.group) * 4]) }];
  }));
  const table = (t) => ({ rows: t.shape[0], n: t.shape[1], ternary: t.kind === "ternary", at: [t.offset, t.scales] });
  const embedding = tensors.token_embedding_table, classifier = tensors.wcls ?? embedding;
  const tables = { classifier: table(classifier), embedding: classifier.offset === embedding.offset ? null : table(embedding) };
  return { layers, matrices, tables, force, remembered };
}

// (T349) what forward.js and the other modules read besides, which was not exported where it was one file
export { GPU_ALONE };
