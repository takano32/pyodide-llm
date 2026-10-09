// forward/memory.js (T349): the memory a model takes: what the forward pass puts after a checkpoint (footprint, with
// the frame of a token and the states of the layers that keep one), the type of the keys and values (keysInHalf),
// whether it fits a 32-bit or a 64-bit memory, and the WebAssembly memory itself (weightsMemory).
// A module of public/forward.js, which asks for it with its own ?v=<build> and exports its names as before; it reads
// jobs.js and its neighbour the same way.

const { CONTROL_BYTES, BATCH } = await import(new URL(`../jobs.js${new URL(import.meta.url).search}`, import.meta.url));
const { GPU_BLOCK } = await import(new URL(`choice.js${new URL(import.meta.url).search}`, import.meta.url));

const PAGE = 65536;

const align = (n, to = 64) => Math.ceil(n / to) * to;

// T101: a 64-bit memory (Memory64) holds more than 4 GiB. Chrome and Firefox have it, shipping Safari not. Its sizes
// are BigInt; the pages of a 32-bit memory stop at 65536, those of a 64-bit one here at 262144 (16 GiB, Chrome's).
const PAGES_32 = 65536, PAGES_64 = 262144;
/** Whether this browser makes 64-bit memories. */
export function memory64() {
  try {
    new WebAssembly.Memory({ initial: 1n, address: "i64" });
    return true;
  } catch {
    return false;
  }
}

// T229: a Qwen3.5's layers (llama2_numpy.py has the computation, above linear_form()). linear: the numbers of its
// linear-attention layers (FORM's "linear": every, key_heads, value_heads, key_dim, value_dim, conv), null for the
// models without them.
// T260: an LFM2's layers (llama2_numpy.py has the computation, above convolution_form()). convolution: its convolution
// layers (FORM's "convolution": layers, a letter a layer, "c" for a convolution layer and "a" for one that attends, and
// taps), null for the models without them.
/** for every layer [whether it keeps a state in place of keys and values (a linear-attention layer, a convolution
 * layer), its place among the layers of its kind]: where its tensors are in the file's stacks, and its keys and values
 * or its state here (llama2_numpy.layer_slots) */
const layerSlots = (layers, linear, convolution = null) => {
  const counts = [0, 0];
  return Array.from({ length: layers }, (_, l) => {
    const kind = (convolution ? convolution.layers[l] === "c" : linear && (l + 1) % linear.every !== 0) ? 1 : 0;
    return [kind === 1, counts[kind]++];
  });
};
/** the layers that attend over all positions (and keep keys and values): all of them without linear or convolution ones */
const attendingLayers = (layers, linear, convolution = null) =>
  layerSlots(layers, linear, convolution).filter(([stateful]) => !stateful).length;
/** of a linear-attention layer: the values the convolution runs over (q, k and v), those of q or of k, those of v */
const linearWidths = (linear) => {
  const keys = linear.key_heads * linear.key_dim, read = linear.value_heads * linear.value_dim;
  return { mixed: 2 * keys + read, keys, read };
};
/** the bytes of the state of the linear-attention layers: a matrix (key_dim, value_dim) a value head, twice (the
 * delta rule reads one and writes the other, so that a phase run again computes the same, T120), and the last conv
 * tokens' q, k and v before the convolution; and the two weights the l2 norm of a head of q and of k is taken with */
const linearStateBytes = (layers, linear) => {
  const lines = layers - attendingLayers(layers, linear), { mixed } = linearWidths(linear);
  return lines * (2 * linear.value_heads * linear.key_dim * linear.value_dim + linear.conv * mixed) * 4 + 2 * align(linear.key_dim * 4);
};
/** T260: the bytes of the state of an LFM2's convolution layers: the last taps tokens' values before the convolution
 * (dim of them a token; the token under way is the last) */
const convolutionStateBytes = (layers, dim, convolution) =>
  (layers - attendingLayers(layers, null, convolution)) * convolution.taps * dim * 4;

// The arrays of one token's frame (see createForward), in their order, and the bytes of each. qDim: the width of q
// and of the attention's output (into xb), heads times the head size: dim, except where a head has another size (T124).
// T229, a Qwen3.5: the gate of a full-attention layer's output; of a linear-attention layer q, k and v before and
// after the convolution, z, and the delta rule's work (beta and decay of every value head, then its delta); xb holds
// what the delta rule reads (as wide as v)
// T237, a model in a rotated basis: xr, what a matrix reads of its input (the widest of them)
// ternary (T231): after the scales of the activations' groups of 32 (xs) come as many int32, minus each group's sum
// (kernels' interleave)
// T260, an LFM2: what a convolution layer's matrix in gives (mixed: the gates B and C and what B multiplies, 3 dim)
const frameArrays = (dim, hidden, kvDim, qDim = dim, linear = null, rotated = null, ternary = false, convolution = null) => {
  const { mixed = 0, read = 0 } = linear ? linearWidths(linear) : {};
  const D = dim * 4, HD = hidden * 4, KF = kvDim * 4, QD = Math.max(dim, qDim, read) * 4, XQ = Math.max(dim, hidden, qDim, read);
  return [["x", D], ["xb", QD], ["xb2", D], ["q", qDim * 4], ["kNow", KF], ["vNow", KF], ["before", D], ["hb", HD],
    ["hb2", HD], ["xq", XQ], ["xs", Math.ceil(XQ / 32) * (ternary ? 8 : 4)],
    ...(linear ? [["gate", qDim * 4], ["mixed", mixed * 4], ["conv", mixed * 4], ["z", read * 4],
      ["work", (2 * linear.value_heads + read) * 4]] : []),
    ...(convolution ? [["mixed", 3 * dim * 4]] : []),
    ...(rotated ? [["xr", XQ * 4]] : [])];
};
/** T237: the widths of what a model's matrices read (llama2_numpy.rotated_widths): the residual stream, an
 * attention's output (and a linear-attention layer's), the FFN's inside. A rotated basis has signs for each. */
const rotatedWidths = (dim, hidden, qDim, linear) =>
  [...new Set([dim, qDim, hidden, ...(linear ? [linearWidths(linear).read] : [])])];
const frameBytes = (arrays) => arrays.reduce((size, [, bytes]) => size + align(bytes), 0);

/** T115: the most bytes the forward pass puts after a checkpoint of size bytes: at the end of its whole context,
 * the KV cache grown to it in place (T130). An upper bound, a little above what createForward allocates
 * (tests/forward-check.mjs holds the two together).
 * header: the 7 ints of the legacy format. dtype: the file's ("float32", "float16", "int8", "int6", "ternary"). int8: the int8
 * kernels compute on the weights (not with ?without=int8, which widens them to float32); relaxed: with relaxed SIMD
 * (an int32 correction a group, T197); halfKV: the keys and values may be float16 (an int8 model, not ?without=kv16;
 * whether they are is keysInHalf's, T160); shared: on a shared memory (T110, where there are software threads).
 * outliers is llama2_numpy's OUTLIER_CHANNELS. arch and head_dim are of the form
 * (llama2_numpy.FORM, which a model's options carry: the caller passes them in as they are, T144): head_dim is the
 * size of a head where it is not dim / heads (T124), 0 where it is. gpu (T135): the page asked for the prompt on the
 * GPU, whose keys and values of a block come back through a place of their own. direct (T156, T210): a model on the
 * GPU alone, whose matrices, tables and keys and values are all there. linear (T229): the form's, the
 * linear-attention layers of a Qwen3.5, which keep a state of a fixed size and no keys and values. rotated (T237):
 * the form's, a rotated basis (its signs for every width, and a place in the frame for a rotated input). convolution
 * (T260): the form's, the convolution layers of an LFM2, which keep the last taps tokens and no keys and values. */
export function footprint(header, size, { dtype = "float32", arch = "llama", int8 = true, relaxed = true, halfKV = false,
  shared = false, outliers = 8, head_dim = 0, gpu = false, direct = false, linear = null, rotated = null,
  convolution = null } = {}) {
  const [dim, hidden, layers, heads, kvHeads, signedVocab, seqLen] = header;
  const vocab = Math.abs(signedVocab), headSize = head_dim || dim / heads, kvDim = kvHeads * headSize, qDim = heads * headSize;
  const ternary = dtype === "ternary", quantized = dtype === "int8" || dtype === "int6" || ternary, six = dtype === "int6";
  // the int8 kernels take rows of whole groups of 32 (llama2_numpy widens the others; a ternary file's rows are whole
  // groups of 128)
  const onInt8 = int8 && dim % 32 === 0 && qDim % 32 === 0 && kvDim % 32 === 0 && hidden % 32 === 0 &&
    (!linear || linearWidths(linear).read % 32 === 0);
  let bytes = 0;
  // what the file holds in another form, for its matrices (not the tables that are no matrix multiplied: an
  // embedding apart from the classifier, GPT-2's positions): the corrections of relaxed SIMD, one int32 a group,
  // as many as the scales (a ninth of an int8 file, a seventh of an int6 one), and the float32 columns of the
  // outlier channels (T92); or, off the int8 kernels, every weight widened to float32
  const tables = (signedVocab < 0 ? vocab * dim : 0) + (arch === "gpt2" ? seqLen * dim : 0);
  // (T156, direct: the layers' matrices are on the GPU alone; T210: so are the embedding and the classifier, and none
  // of their corrections or outlier columns is here)
  // (T231: ternary weights, 36 bytes a group of 128, have no corrections: their kernels take the activations signed)
  const weights = !quantized ? 0 : size * (ternary ? 128 / 36 : six ? 32 / 28 : 32 / 36) - tables, matrices = quantized && !direct;
  if (matrices && onInt8) bytes += (relaxed && !ternary ? weights / 8 : 0) + Math.min(outliers, dim) * (vocab + 1) * 4;
  else if (matrices) bytes += weights * 4;
  else if (dtype === "float16") bytes += size * 2;
  // what a quantized file leaves out: GPT-2's positions widened, and the RoPE tables Python computes (two of seqLen ×
  // headSize / 2 float32; GPT-2 has them too, of zeros, at any dtype: it has no RoPE; T130's review)
  if (quantized && arch === "gpt2") bytes += seqLen * dim * 4;
  if (quantized || arch === "gpt2") bytes += seqLen * headSize * 4;
  // the frames of BATCH tokens, their attention scores, the logits; the keys and values of a block from the GPU (in
  // float16) and its rows
  bytes += BATCH * (frameBytes(frameArrays(dim, hidden, kvDim, qDim, linear, rotated, ternary, convolution)) + align(seqLen * heads * 4)) + vocab * 4;
  // T237: the signs of a rotated basis, a float32 for every value of every width
  if (rotated) bytes += rotatedWidths(dim, hidden, qDim, linear).reduce((sum, width) => sum + align(width * 4), 0);
  if (gpu) bytes += 2 * layers * GPU_BLOCK * kvDim * 2 + GPU_BLOCK * dim * 4;
  // T229: the state of the linear-attention layers, whatever the context; keys and values of the others alone
  if (linear) bytes += linearStateBytes(layers, linear);
  if (convolution) bytes += convolutionStateBytes(layers, dim, convolution);  // T260
  // the KV cache, doubled in place up to the whole context (T130: createForward's grow() moves the blocks up into the
  // room it adds; before, the smaller blocks were still there next to the larger ones at each step, 1.5 times the
  // context at the last), and a megabyte for the alignment of every array (T210, direct: the keys and values are the
  // GPU's alone)
  const others = Math.ceil(bytes) + 2 ** 20, keys = direct ? 0 : seqLen * attendingLayers(layers, linear, convolution) * 2 * kvDim;
  // The type of the keys and values: float16 on a shared memory where every head has keys of its own (T110: several
  // threads wait on the memory, and read half of it). T160: a grouped-query model's are widened for every head of
  // their group, g = heads / kvHeads times, which float16 saves nothing of (Qwen2.5 0.5B, g = 7: float32 1.42 to 1.44
  // times as fast on one thread at position 2000, 1.15 to 1.26 on four, CI's x86-64 and arm64; TODO.md's T160), and
  // one thread waits on the arithmetic of the widening (T110): float32 there. But float16 on a shared memory wherever
  // float32 would not fit a 32-bit one (the owner, 2026-09-27: Llama 3.2 3B and the 7B models, on a 64-bit memory,
  // keep their memory), and T130, on a memory that is not shared (a page not cross-origin isolated, or a shared one
  // refused) only where float16 keeps on a 32-bit memory a model that float32 would take past 4 GiB (Llama 3.2 3B's
  // int8 on Safari, without relaxed SIMD: 3.82 GiB, 4.26 in float32). A model past 4 GiB either way keeps float32
  // there (the owner, 2026-09-28: one thread's long contexts stay fast), unless float32 would not fit even a 64-bit
  // memory (16 GiB) where float16 does: the plain memory that a refused shared one leaves must hold what the worker sized
  // the shared one for, which pastWide() was asked with (T130's review: Pythia 12B's int8 of ?hf= is 15.2 GiB with
  // float16 keys and values and 16.6 with float32). keysInHalf tells which.
  const past = needsWide(size, others + 4 * keys);
  // (and only where the int8 kernels run, as the engine's half_kv says: a file whose rows are not whole groups of 32 is
  // widened to float32 and keeps float32 keys and values; T130's review)
  const half = halfKV && onInt8 &&
    (shared ? kvHeads >= heads || past : past && (!needsWide(size, others + 2 * keys) || pastWide(size, others + 4 * keys)));
  return others + keys * (half ? 2 : 4);
}
/** T160: whether the keys and values of a model that may keep them in float16 (footprint's halfKV) do, as footprint
 * counts them. The worker hands this to createForward: what it sized the memory for. */
export const keysInHalf = (header, size, options = {}) =>
  Boolean(options.halfKV) && footprint(header, size, options) < footprint(header, size, { ...options, halfKV: false });

/** Whether a checkpoint of size bytes and the forward pass after it (footprint) pass the 4 GiB of a 32-bit memory.
 * A model that fits stays there: a 64-bit memory runs the kernels about a tenth slower (T101, measured). */
export const needsWide = (size, after) => CONTROL_BYTES + size + after > PAGES_32 * PAGE;
/** T129 (7): whether they pass even a 64-bit memory (16 GiB here, Chrome's): no memory holds such a model. */
export const pastWide = (size, after) => CONTROL_BYTES + size + after > PAGES_64 * PAGE;
/** T133: the dtype of a model converted with none asked for, from its int8 size and what the forward pass puts after
 * it (footprint, as int8): int8 where that fits a 32-bit memory, or where the browser has a 64-bit one (wide: Chrome
 * and Firefox; about a tenth slower, against six bits' half the speed and +1.4 to 1.7% of perplexity, T98); else six
 * bits (7/9 of int8's memory: Safari). */
export const automaticDtype = (int8, after, wide) => (wide || !needsWide(int8, after) ? "int8" : "int6");
/** memory.grow(pages), in the number type of the memory (wide: 64-bit) */
export function growMemory(memory, pages, wide) {
  memory.grow(wide ? BigInt(pages) : pages);
}

/** A memory with room for a checkpoint of size bytes at base; the forward pass allocates after it. shared (stage 2):
 * a SharedArrayBuffer for the helper threads, only where the page is cross-origin isolated; its first 8 KiB are the
 * control area of the helpers. A shared memory needs a maximum: as much as the browser grants, less if it refuses.
 * maximum (pages): what to ask for first; else what this model needs (after: what the forward pass puts after the
 * checkpoint, footprint(); three times the file where it is not said) and a gigabyte more. The worker keeps the
 * memory for the models that fit under that maximum (T96): a browser reserves address space for each WebAssembly
 * memory whatever its maximum, and Chromium refused the third one of a page. spare (bytes): what is asked for besides
 * the model, the gigabyte for the next one unless it is said (T242: no model follows in /benchmark/'s worker). */
export function weightsMemory(size, { shared = false, maximum, wide = false, after = 3 * size, spare = 2 ** 30 } = {}) {
  const base = shared ? CONTROL_BYTES : 64;
  const initial = Math.ceil((base + size) / PAGE) + 1;
  // a 64-bit memory (T101) says its sizes in BigInt
  const describe = (pages) => (wide ? { initial: BigInt(initial), ...(pages ? { maximum: BigInt(pages) } : {}), address: "i64" }
    : { initial, ...(pages ? { maximum: pages } : {}) });
  if (!shared) return { memory: new WebAssembly.Memory(describe()), base };
  // what the model needs, and a gigabyte for the next one to fit as well (T96); less if the browser refuses, never more
  // (T242's review: where no gigabyte is asked for, the model alone may be under both of the others, and a browser that
  // refused it would be asked for a gigabyte: the reservation a Windows WebKit's page went down in)
  const most = maximum ?? Math.min(wide ? PAGES_64 : PAGES_32, Math.ceil((base + size + after + spare) / PAGE));
  for (const pages of [most, ...[initial + 16384, initial + 4096].filter((fewer) => fewer < most)]) {
    try {
      const memory = new WebAssembly.Memory({ ...describe(Math.max(pages, initial)), shared: true });
      memory.maximum = Math.max(pages, initial);  // the worker keeps the memory as long as the next model fits (T96)
      memory.limited = pages < most;  // the browser gave less than was asked for: a new memory would not get more
      return { memory, base };
    } catch {
      // too much address space for this browser: ask for less
    }
  }
  throw new Error("This browser gives no shared WebAssembly memory for this model.");
}

// (T349) what forward.js and the other modules read besides, which was not exported where it was one file
export { PAGE, align, layerSlots, attendingLayers, linearWidths, frameArrays, rotatedWidths, frameBytes };
