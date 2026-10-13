// forward/memory.ts (T349): the memory a model takes: what the forward pass puts after a checkpoint (footprint, with
// the frame of a token and the states of the layers that keep one), the type of the keys and values (keysInHalf),
// whether it fits a 32-bit or a 64-bit memory, and the WebAssembly memory itself (weightsMemory).
// A module of src/runtime/forward.ts, which exports its names as before.

import { CONTROL_BYTES, BATCH, BLOCK_TABLES, SPREAD_THREADS, packingOf, spreadBytes } from "../jobs.ts";
import { GPU_BLOCK } from "./choice.ts";
import { architectureOf } from "./architectures.ts";
import { LAYER_KINDS } from "./kinds.ts";
import { kindsHad, sizesOfForm } from "./sizes.ts";
import type { Kinds } from "./kinds.ts";
import type { Arch, Dtype, Form, Rotated, Stateful } from "../contracts/form.ts";

// (T368.4, the types) a checkpoint's header: the 7 ints of the legacy format
export type Header = number[];
// A memory weightsMemory() made: a shared one says the pages it may grow to, and whether the browser gave fewer than
// were asked for (T96)
export type Memory = WebAssembly.Memory & { maximum?: number; limited?: boolean };
// what a 64-bit memory is made with (T101), which the WebAssembly types of TypeScript's lib do not know yet
type Wide = { initial: bigint; maximum?: bigint; address: "i64" };
// What sizes the forward pass besides the header (footprint(), keysInHalf()): the dtype and the form's part of it, and
// how the engine is to run (footprint() below says each)
export type Footprint = {
  dtype?: Dtype;
  arch?: Arch;
  int8?: boolean;
  relaxed?: boolean;
  halfKV?: boolean;
  shared?: boolean;
  outliers?: number;
  head_dim?: number;
  gpu?: boolean;
  direct?: boolean;
  rotated?: Rotated | null;
  kinds?: Form["kinds"];
  weights?: number;
} & Stateful;

const PAGE = 65536;

const align = (n: number, to = 64) => Math.ceil(n / to) * to;

// T101: a 64-bit memory (Memory64) holds more than 4 GiB. Chrome and Firefox have it, shipping Safari not. Its sizes
// are BigInt; the pages of a 32-bit memory stop at 65536, those of a 64-bit one here at 262144 (16 GiB, Chrome's).
const PAGES_32 = 65536, PAGES_64 = 262144;
/** Whether this browser makes 64-bit memories. */
export function memory64() {
  try {
    new WebAssembly.Memory({ initial: 1n, address: "i64" } satisfies Wide as unknown as WebAssembly.MemoryDescriptor);
    return true;
  } catch {
    return false;
  }
}

// The arrays of one token's frame (see createForward), in their order, and the bytes of each. qDim: the width of q
// and of the attention's output (into xb), heads times the head size: dim, except where a head has another size (T124).
// T229, a Qwen3.5: the gate of a full-attention layer's output; of a linear-attention layer q, k and v before and
// after the convolution, z, and the delta rule's work (beta and decay of every value head, then its delta); xb holds
// what the delta rule reads (as wide as v)
// T237, a model in a rotated basis: xr, what a matrix reads of its input (the widest of them)
// scaleBytes: what a group of 32 activations takes in xs, as the packing of the weights says (jobs.ts's PACKED: 4, a
// float32 scale; ternary's 8, T231: after the scales come as many int32, minus each group's sum, kernels' interleave)
// T260, an LFM2: what a convolution layer's matrix in gives (mixed: the gates B and C and what B multiplies, 3 dim)
// (rotated: the form's rotated basis or the plan's block of one: whether there is one is all that is read. Framed: an
// array of a token's frame, its name and its bytes)
// T375.3: the frame is the arrays every model has, then what each kind of the model's layers puts in it (in the
// order of the table of the kinds: forward/kinds.ts), then a rotated basis's. A kind says its arrays and the widest
// input its matrices read where that is wider than the model's own (widest: xb and xq hold it); the engine hands in
// what the kinds say (each asked with what sizes the model, sizes.ts): the engine of the plan, footprint() below of the
// header and the form
export type Framed = [string, number];
export type KindFrame = { arrays: Framed[]; widest?: number };
const frameArrays = (dim: number, hidden: number, kvDim: number, qDim = dim, kinds: KindFrame[] = [], rotated: Rotated | number | null = null, scaleBytes = 4): Framed[] => {
  const read = Math.max(0, ...kinds.map(({ widest = 0 }) => widest));
  const D = dim * 4, HD = hidden * 4, KF = kvDim * 4, QD = Math.max(dim, qDim, read) * 4, XQ = Math.max(dim, hidden, qDim, read);
  return [["x", D], ["xb", QD], ["xb2", D], ["q", qDim * 4], ["kNow", KF], ["vNow", KF], ["before", D], ["hb", HD],
    ["hb2", HD], ["xq", XQ], ["xs", Math.ceil(XQ / 32) * scaleBytes],
    ...kinds.flatMap(({ arrays }) => arrays),
    ...(rotated ? [["xr", XQ * 4]] satisfies Framed[] : [])];
};
const frameBytes = (arrays: Framed[]) => arrays.reduce((size, [, bytes]) => size + align(bytes), 0);

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
 * GPU alone, whose matrices, tables and keys and values are all there. rotated (T237): the form's, a rotated basis
 * (its signs for every width, and a place in the frame for a rotated input). And the form's numbers of the layers that
 * keep a state and no keys and values, each under its kind's name (linear, T229: the linear-attention layers of a
 * Qwen3.5; convolution, T260: the convolution layers of an LFM2), which the kinds read.
 * T363.5: kinds, the form's: the rows the file stores otherwise than its dtype does (a kind's name, or one for each
 * layer of a stack). What their kernels need beside them is counted by their lines of PACKED (the tables and the
 * scratch of llama.cpp's blocks, the room their activations take). weights: of a file with kinds, how many weights of
 * its matrices are still the dtype's own (Python counts them of the rows: llama2_numpy's dtype_weights(); there are no
 * rows here): what relaxed SIMD's corrections are for. Left out, every weight of the file is taken for one: an upper
 * bound, a ninth of the file too high where all the matrices are blocks.
 * T375.5: nothing here names an architecture, a dtype or a kind of layer. What the layers put in a token's frame and
 * keep besides is their kinds' (kinds: forward/kinds.ts's table, the engine's own: frame(), state(), cached), asked
 * with what sizes the model as the form says it (sizes.ts's sizesOfForm()); what the architecture's name says is its
 * line of architectures.ts; how the weights are packed is the dtype's line of jobs.ts's PACKED. */
export function footprint(header: Header, size: number, options: Footprint = {}, kinds: Kinds = LAYER_KINDS) {
  const { dtype = "float32", arch = "llama", int8 = true, relaxed = true, halfKV = false, shared = false, outliers = 8, head_dim = 0, gpu = false,
    direct = false, rotated = null } = options;
  const [dim, hidden, layers, heads, kvHeads, signedVocab, seqLen] = header;
  const vocab = Math.abs(signedVocab), headSize = head_dim || dim / heads;
  // (the options as the form: the numbers of the layers that keep a state are in them under their kinds' names, which
  // the kinds read and this does not)
  const sizes = sizesOfForm(header, options, kinds), { q: qDim, kv: kvDim } = sizes.widths;
  const { positions, rope } = architectureOf(arch), packing = packingOf(dtype);
  // (T363.5) the packings the model's rows are in: the dtype's, and those of the kinds its form gives rows
  const own = Object.values(options.kinds ?? {}).flat().map((kind) => packingOf(kind)).filter((one) => one !== null);
  const packings = packing ? [packing, ...own] : own;
  // the kinds of layers this model has, and what each puts in a token's frame
  const had = kindsHad(sizes, kinds), frames = had.map(({ kind }) => kinds[kind].frame(sizes));
  // the int8 kernels take rows of whole groups of 32 (llama2_numpy widens the others; a ternary file's rows are whole
  // groups of 128): the model's own widths and the widest a kind's matrices read
  const onInt8 = int8 && [dim, qDim, kvDim, hidden, ...frames.map(({ widest = 0 }) => widest)].every((width) => width % 32 === 0);
  let bytes = 0;
  // what the file holds in another form, for its matrices (not the tables that are no matrix multiplied: an
  // embedding apart from the classifier, a table of positions): the corrections of relaxed SIMD where the packing's
  // relaxed kernel takes them, one int32 a group, as many as the scales (a ninth of an int8 file, a seventh of an int6
  // one; T231: none of ternary weights, whose kernels take the activations signed), and the float32 columns of the
  // outlier channels (T92); or, off the int8 kernels, every weight widened to float32
  const tables = (signedVocab < 0 ? vocab * dim : 0) + (positions ? seqLen * dim : 0);
  // (T156, direct: the layers' matrices are on the GPU alone; T210: so are the embedding and the classifier, and none
  // of their corrections or outlier columns is here)
  // (the weights of a packed file: its bytes, by the weights a byte holds: a group's values and its scale)
  if (packing && !direct) {
    const weights = options.weights ?? size * (packing.group / (packing.rowBytes(packing.group) + packing.scale)) - tables;
    // (T363.5: no outlier channels of a classifier in a kind of the form's that no weight is read of alone: plan.py)
    const classifier = options.kinds?.[signedVocab < 0 ? "wcls" : "token_embedding_table"];
    const columns = classifier === undefined || (typeof classifier === "string" && Boolean(packingOf(classifier)?.value)) ? Math.min(outliers, dim) * (vocab + 1) * 4 : 0;
    bytes += onInt8 ? (relaxed && packing.sums ? (weights / packing.group) * 4 : 0) + columns : weights * 4;
  } else if (dtype === "float16") bytes += size * 2;  // (no packing: the whole file widened to float32)
  // what a packed file leaves out: the positions widened, and the RoPE tables Python computes (two of seqLen ×
  // headSize / 2 float32; a model without RoPE has them too, of zeros, at any dtype; T130's review)
  if (packing && positions) bytes += seqLen * dim * 4;
  if (packing || !rope) bytes += seqLen * headSize * 4;
  // the frames of BATCH tokens and what the kinds keep of a token apart from its frame (the scores of its
  // attention), the logits; the keys and values of a block from the GPU (in float16) and its rows
  const apart = frames.flatMap(({ apart = [] }) => apart).reduce((sum, [, each]) => sum + align(each), 0);
  const inFrame = frameArrays(dim, hidden, kvDim, qDim, frames, rotated, packings.length ? Math.max(...packings.map(({ xs }) => xs)) : undefined);
  bytes += BATCH * (frameBytes(inFrame) + apart) + vocab * 4;
  // T363.5: the tables of the kernels of llama.cpp's blocks, the bytes the engine asks its relaxed SIMD with, and the
  // scratch a thread spreads a row into (as wide as the widest input: an upper bound, taken where the engine's relaxed
  // SIMD multiplies signed bytes)
  if (own.some(({ tabled }) => tabled !== undefined) && onInt8) {
    bytes += align(BLOCK_TABLES) + (relaxed ? 64 + SPREAD_THREADS * spreadBytes(inFrame.find(([name]) => name === "xq")![1]) : 0);
  }
  // T237: the signs of a rotated basis, a float32 for every value of every width
  if (rotated) bytes += sizes.widths.rotated.reduce((sum, width) => sum + align(width * 4), 0);
  if (gpu) bytes += 2 * layers * GPU_BLOCK * kvDim * 2 + GPU_BLOCK * dim * 4;
  // what the kinds' layers keep besides, whatever the context (T229: a linear-attention layer's state; T260: a
  // convolution layer's rows), each as the engine takes it of the memory
  bytes += had.flatMap(({ kind, count }) => kinds[kind].state?.(sizes, count) ?? []).reduce((sum, each) => sum + align(each), 0);
  // the layers that keep keys and values (all of them where none keeps a state in their place)
  const cached = had.reduce((sum, { kind, count }) => sum + (kinds[kind].cached ? count : 0), 0);
  // the KV cache, doubled in place up to the whole context (T130: createForward's grow() moves the blocks up into the
  // room it adds; before, the smaller blocks were still there next to the larger ones at each step, 1.5 times the
  // context at the last), and a megabyte for the alignment of every array (T210, direct: the keys and values are the
  // GPU's alone)
  const others = Math.ceil(bytes) + 2 ** 20, keys = direct ? 0 : seqLen * cached * 2 * kvDim;
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
  // (and only where the int8 kernels run on packed weights, as the engine's half_kv says: a file whose rows are not
  // whole groups of 32 is widened to float32 and keeps float32 keys and values, T130's review; T375.5: and a float
  // model's are float32, which the caller had to know before)
  const half = halfKV && Boolean(packing) && onInt8 &&
    (shared ? kvHeads >= heads || past : past && (!needsWide(size, others + 2 * keys) || pastWide(size, others + 4 * keys)));
  return others + keys * (half ? 2 : 4);
}
/** T363.5: whether a form's kinds (the options of a model) have rows of llama.cpp's blocks: such a model runs on the
 * kernels of those, which the worker fetches for it (and for no other) */
export const takesBlocks = (kinds: Form["kinds"] | undefined) => Object.values(kinds ?? {}).flat().some((kind) => packingOf(kind)?.tabled !== undefined);

/** T160: whether the keys and values of a model that may keep them in float16 (footprint's halfKV) do, as footprint
 * counts them. The worker hands this to createForward: what it sized the memory for. */
export const keysInHalf = (header: Header, size: number, options: Footprint = {}) =>
  Boolean(options.halfKV) && footprint(header, size, options) < footprint(header, size, { ...options, halfKV: false });

/** Whether a checkpoint of size bytes and the forward pass after it (footprint) pass the 4 GiB of a 32-bit memory.
 * A model that fits stays there: a 64-bit memory runs the kernels about a tenth slower (T101, measured). */
export const needsWide = (size: number, after: number) => CONTROL_BYTES + size + after > PAGES_32 * PAGE;
/** T129 (7): whether they pass even a 64-bit memory (16 GiB here, Chrome's): no memory holds such a model. */
export const pastWide = (size: number, after: number) => CONTROL_BYTES + size + after > PAGES_64 * PAGE;
/** T133: the dtype of a model converted with none asked for, from its int8 size and what the forward pass puts after
 * it (footprint, as int8): int8 where that fits a 32-bit memory, or where the browser has a 64-bit one (wide: Chrome
 * and Firefox; about a tenth slower, against six bits' half the speed and +1.4 to 1.7% of perplexity, T98); else six
 * bits (7/9 of int8's memory: Safari). */
export const automaticDtype = (int8: number, after: number, wide: boolean) => (wide || !needsWide(int8, after) ? "int8" : "int6");
/** memory.grow(pages), in the number type of the memory (wide: 64-bit) */
export function growMemory(memory: WebAssembly.Memory, pages: number, wide: boolean) {
  memory.grow((wide ? BigInt(pages) : pages) as number);
}

/** A memory with room for a checkpoint of size bytes at base; the forward pass allocates after it. shared (stage 2):
 * a SharedArrayBuffer for the helper threads, only where the page is cross-origin isolated; its first 8 KiB are the
 * control area of the helpers. A shared memory needs a maximum: as much as the browser grants, less if it refuses.
 * maximum (pages): what to ask for first; else what this model needs (after: what the forward pass puts after the
 * checkpoint, footprint(); three times the file where it is not said) and a gigabyte more. The worker keeps the
 * memory for the models that fit under that maximum (T96): a browser reserves address space for each WebAssembly
 * memory whatever its maximum, and Chromium refused the third one of a page. spare (bytes): what is asked for besides
 * the model, the gigabyte for the next one unless it is said (T242: no model follows in /benchmark/'s worker). */
export function weightsMemory(size: number, { shared = false, maximum, wide = false, after = 3 * size, spare = 2 ** 30 }:
  { shared?: boolean; maximum?: number; wide?: boolean; after?: number; spare?: number } = {}): { memory: Memory; base: number } {
  const base = shared ? CONTROL_BYTES : 64;
  const initial = Math.ceil((base + size) / PAGE) + 1;
  // a 64-bit memory (T101) says its sizes in BigInt
  const describe = (pages?: number): WebAssembly.MemoryDescriptor => (wide ? { initial: BigInt(initial), ...(pages ? { maximum: BigInt(pages) } : {}), address: "i64" } satisfies Wide as unknown as WebAssembly.MemoryDescriptor
    : { initial, ...(pages ? { maximum: pages } : {}) });
  if (!shared) return { memory: new WebAssembly.Memory(describe()), base };
  // what the model needs, and a gigabyte for the next one to fit as well (T96); less if the browser refuses, never more
  // (T242's review: where no gigabyte is asked for, the model alone may be under both of the others, and a browser that
  // refused it would be asked for a gigabyte: the reservation a Windows WebKit's page went down in)
  const most = maximum ?? Math.min(wide ? PAGES_64 : PAGES_32, Math.ceil((base + size + after + spare) / PAGE));
  for (const pages of [most, ...[initial + 16384, initial + 4096].filter((fewer) => fewer < most)]) {
    try {
      const memory: Memory = new WebAssembly.Memory({ ...describe(Math.max(pages, initial)), shared: true });
      memory.maximum = Math.max(pages, initial);  // the worker keeps the memory as long as the next model fits (T96)
      memory.limited = pages < most;  // the browser gave less than was asked for: a new memory would not get more
      return { memory, base };
    } catch {
      // too much address space for this browser: ask for less
    }
  }
  throw new Error("This browser gives no shared WebAssembly memory for this model.");
}

// (T349) what forward.ts and the other modules read besides, which was not exported where it was one file
// half to float, exactly (the same as NumPy's astype(float32))
function halfToFloat(h: number) {
  const sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

export { PAGE, align, frameArrays, frameBytes, halfToFloat };
