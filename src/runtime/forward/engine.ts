// forward/engine.ts (T349): createForward(), one model's forward pass: its memory after the checkpoint, its tensors,
// the frames of the activations, the round of the layers, and the object Python and the worker call. The software
// threads are threads.ts's and the GPU's side gpuside.ts's: each is handed what it reads of this function. T375.3:
// what a layer does is its kind's (kinds.ts's table: attention.ts, linear.ts, convolution.ts), the FFN is ffn.ts's and
// the cache of the keys and values cache.ts's: each is chosen or made here once, and a token's path calls what it gave.
// A module of src/runtime/forward.ts.

import { BATCH, BLOCK_TABLES, KIND, SPREAD_THREADS, addressed, packingOf, runner, spreadBytes } from "../jobs.ts";
import { GPU_BLOCK, GPU_TOKENS } from "./choice.ts";
import { growMemory, PAGE, align, frameArrays, frameBytes, halfToFloat } from "./memory.ts";
import { LAYER_KINDS } from "./kinds.ts";
import { kindsHad, sizesOfPlan } from "./sizes.ts";
import { keyValueCache } from "./cache.ts";
import { gatedFfn, plainFfn } from "./ffn.ts";
import { perLayer } from "./perlayer.ts";
import { GPU_ALONE } from "./alone.ts";
import { controlArea } from "./control.ts";
import { softwareThreads } from "./threads.ts";
import { gpuFit, gpuSide } from "./gpuside.ts";
import type { Kernels, Packed, Packing, Ready } from "../jobs.ts";
import type { Direct } from "./alone.ts";
import type { CpuSide, GpuWorker, Held, Placed, Shape, Taken } from "./gpuside.ts";
import type { Spawn } from "./threads.ts";
import type { Rows } from "./perlayer.ts";
import type { Force, Remembered } from "../contracts/gpu/protocol.ts";
import type { Core, Kinds, Product, Room } from "./kinds.ts";
import type { Derived, Part, Plan, Tensor, TensorName } from "../contracts/plan.ts";

// ---- (T368.4, the types)
// The kernels of one build as modules (forward.ts's compileKernels()): relaxed: null where the browser has no relaxed
// SIMD; wide: the build for a 64-bit memory; blocks (T363.5): the same build of the kernels of llama.cpp's blocks
// (blockkernel_*.wasm), which an engine has only for a model with such rows (forward.ts's blockKernels())
export type Compiled = { plain: WebAssembly.Module; relaxed: WebAssembly.Module | null; wide?: boolean; blocks?: Blocks | null };
export type Blocks = { plain: WebAssembly.Module; relaxed: WebAssembly.Module | null };
// what createForward() is made with for the tests alone, to hold every form of a kernel to the others: signed, whether
// the tile of the signed dot product is taken (left out: where the engine's relaxed SIMD multiplies so, signed_dots)
export type Trying = { signed?: boolean };
// What an engine is made of (the comments above createForward() say each)
export type Made = {
  memory: WebAssembly.Memory; base: number; size: number; kernels: Compiled; plan: Plan;
  spawn?: Spawn | null; gpu?: () => GpuWorker; gpuRoom?: number; memoryUnsaid?: boolean; gpuRemembered?: Remembered; gpuForce?: Force;
  halfKeys?: boolean; direct?: Direct; rows?: Rows;
  wrap?: (exports: Kernels) => Kernels; stalledMs?: number; clock?: () => number; kinds?: Kinds; trying?: Trying;
};
// The array forward() fills with the logits: a Float32Array, or Python's (a PyProxy of a NumPy array: copy() keeps
// it past the call, getBuffer() gives its floats, destroy() lets it go)
type Bound = Float32Array & { copy?(): Bound; getBuffer?(type: string): { data: Float32Array; release(): void }; destroy?(): void };
// ids as Python hands them (a PyProxy of a list, which toJs() copies) or as JavaScript does
type Ids = Iterable<number> & { toJs?(): number[] };

// T120: no phase comes near this without progress (the longest token measured, Qwen2.5 7B's on CI, took 286 ms in
// all): a count that has not moved for so long means a software thread the browser stopped in the middle of its chunk
const STALLED_MS = 10000;

/** spawn (stage 2, a shared memory only): starts one helper thread with { memory, plain, relaxed } and resolves once
 * it is ready; the result has terminate(). Without it, or on a memory that is not shared, everything runs here. */
/** gpu (T135; T148: wherever the worker has WebGPU): makes the GPU's worker (gpu.js, not started: a Worker, or what
 * posts and listens as one). A prompt's blocks then go through the layers there where the model allows it and the GPU
 * is faster; see forwardMany. gpuRoom (T148): the bytes this device can give the GPU's copy of the layers (the
 * worker's reckoning from navigator.deviceMemory; none said: no limit). memoryUnsaid (T205): the browser does not say
 * how much memory the device has (no navigator.deviceMemory: Safari, Firefox); a generation's steps then stay on the
 * CPU. gpuRemembered: what the page kept of the GPU's shaders on an earlier visit ({ key, matrices, attention },
 * gpu.js). */
/** wrap (tests/profile.mjs only): gets the kernels' exports and returns what to call instead, to time the forward
 * pass with some kernels replaced by functions that do nothing. */
/** stalledMs (tests only): how long a phase may make no progress before its software threads are given up (T120) */
/** clock (tests only, T199): the time the threads' search reads, in ms (a made-up one slows a block of its choice) */
/** halfKeys (T160): keysInHalf's answer for this model on the memory it got (T130: a shared one refused, the
 * worker asks again for the plain one). Left out (the tests, the benchmark's own model): float16 on a shared memory
 * where every head has keys of its own, which keysInHalf answers for every model that fits a 32-bit memory with
 * float32 keys and values. */
/** gpuForce (tests only, T147): { matrices, attention }, the names of the GPU's shaders to take (shaders.js's
 * promptForms, gpu.js's attentions), without timing the others. T148: fallback, a fallback adapter taken as a GPU
 * (SwiftShader and lavapipe: the only WebGPU of CI and the development machine); always, every block the GPU can take
 * goes there, whatever the CPU's time (a fallback adapter is far slower than the CPU); quick (the page's tests), the
 * first right shader of the matrices untimed, and no block timed (SwiftShader timing Llama 3.2 1B's took more than
 * gpu.js's STEP_MS in CI, T147); pieceBytes (T155), the most bytes of a piece of a matrix on the GPU, so that a small
 * model goes in pieces as a matrix past a buffer of the device does */
/** kinds (tests only, T375.3): the table of the kinds of layers there is a step for (forward/kinds.ts's LAYER_KINDS):
 * a test hands in one with a kind of its own */
/** trying (tests only, T363.5): the forms of a kernel to take whatever the engine would choose (Trying) */
/** rows (T329.2): what reads a token's row of a table every layer has an input of (a Gemma 4's embedding of every
 * layer: perlayer.ts's Rows), at once. Left out, the row is copied from the checkpoint in this memory; the worker hands
 * in one that reads it from the kept checkpoint where the memory holds no table (T329.2.2) */
/** direct (T156, external()): the layers' matrices (T210: and the tables) are on the GPU alone: nothing runs on the CPU,
 * every block and step goes to the GPU (as gpuForce.always), and one the GPU cannot take throws (the worker loads the
 * model again on the CPU). T210: no cache of keys and values here either; the GPU embeds a prompt's tokens itself */
export function createForward({ memory, base, size, kernels, plan, spawn, gpu, gpuRoom, memoryUnsaid, gpuRemembered, gpuForce = {},
  halfKeys, direct, rows, wrap = (exports) => exports, stalledMs = STALLED_MS, clock = () => performance.now(), kinds = LAYER_KINDS, trying = {} }: Made) {
  const { dim, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: headSize, vocab_size: vocab,
    seq_len: seqLen } = plan;
  // T375.2: what the layers of one architecture differ from another's by is read here once, as the facts Python's plan
  // says (engine/plan.py, from engine/layout.py's table of the architectures): nothing below compares an
  // architecture's name, and nothing here works out again what the plan holds. What a tensor's presence says is read
  // off the tensors (a bias, the norms of a head, a gate, the table of positions: floats() and matrix() answer 0 and
  // null for one the model has not).
  // The widths that are not in the header: q and the attention's output (the heads together), k and v
  const hidden = plan.hidden_dim, { q: qDim, kv: kvDim } = plan.widths;
  // the kind of the norms (LayerNorms, else RMSNorms) and of the FFN (gated, else GPT-2's) are two facts, though every
  // architecture so far has LayerNorms with GPT-2's FFN or RMSNorms with a gated one; and GPT-NeoX's parallel residual
  const layerNorm = plan.layer_norm, parallel = plan.parallel_residual;
  // every layer's facts: its kind (one that attends, and keeps keys and values; or one that keeps a state: T229, a
  // Qwen3.5's linear-attention layer; T260, an LFM2's convolution layer), its place among the layers of its kind (the
  // layer itself where all attend), and whether RoPE turns its q and k (T255: none of a GPT-2's, not the ones a
  // SmolLM3 leaves alone). A plan that says nothing of its layers is no plan of this engine's, and a kind this engine
  // has no step for is said here, before a token runs.
  // T375.3: what a kind of layer is, is in the table of the kinds (forward/kinds.ts) and nowhere here: the kinds this
  // model has, in the table's order, each with the number of its layers
  const facts = plan.layers;
  if (!Array.isArray(facts) || facts.length !== layers) throw new Error(`The plan says the facts of ${Array.isArray(facts) ? facts.length : "no"} layers, and the model has ${layers}.`);
  const strange = facts.find(({ kind }) => !Object.hasOwn(kinds, kind));
  if (strange) throw new Error(`This engine has no layer of the kind ${JSON.stringify(strange.kind)}.`);
  // T375.5: what sizes the model, as the plan says it (sizes.ts): the kinds are asked with it for what they put in a
  // token's frame and keep besides, as footprint() asks them before a model is loaded
  const sizes = sizesOfPlan(plan), had = kindsHad(sizes, kinds);
  // T255: the layers RoPE leaves alone, as the GPU's plan lists them
  const unturned = Array.from(plan.unturned || []);
  // T237: the block of a rotated basis (0: the model's own basis). llama2_numpy.py has the definition above hadamard()
  const rotated = plan.rotated || 0;
  const imports = { env: { memory } };
  const wide = Boolean(kernels.wide);  // T101: a 64-bit memory, whose kernels take their addresses as BigInt
  // a page that is not cross-origin isolated has no SharedArrayBuffer to ask about: its memory is not shared
  const sharedMemory = typeof SharedArrayBuffer !== "undefined" && memory.buffer instanceof SharedArrayBuffer;
  const T = plan.tensors, derived: Derived = plan.derived ?? {};
  // T363.5: how every tensor of the file is packed, a stack's layers each: the packings there are (jobs.ts's PACKED)
  const partsOf = (t: Tensor): Part[] => t.layers ?? [t as Part];
  const packings = Object.values(T).flatMap(partsOf).map(({ kind }) => packingOf(kind)).filter((packing) => packing !== null);
  // the kernels of llama.cpp's blocks, of modules of their own, on this memory beside the others where the model has
  // rows of such a type; nowhere else
  const blocked = packings.find(({ tabled }) => tabled !== undefined);
  if (blocked && !kernels.blocks) throw new Error("This engine was made without the kernels of llama.cpp's blocks, and the model has rows of them.");
  const blocks = blocked ? kernels.blocks! : null;
  const instance = (module: WebAssembly.Module) => new WebAssembly.Instance(module, imports).exports;
  const k = wrap(addressed(blocks ? { ...instance(kernels.plain), ...instance(blocks.plain) } : instance(kernels.plain), wide));
  const runsRelaxed = plan.int8 && plan.relaxed && kernels.relaxed;
  const relaxed = runsRelaxed ? wrap(addressed(blocks?.relaxed ? { ...instance(kernels.relaxed!), ...instance(blocks.relaxed) } : instance(kernels.relaxed!), wide)) : null;

  // ---- memory: the checkpoint at base, everything else after it; views are made again after the memory grows
  let top = align(base + size);
  // (F!: of the four, F alone is read outside a function, after views() made them)
  let buffer: ArrayBufferLike | undefined, F!: Float32Array, U: Uint8Array, H: Uint16Array;
  const views = () => {
    if (buffer !== memory.buffer) {
      buffer = memory.buffer;
      F = new Float32Array(buffer); U = new Uint8Array(buffer); H = new Uint16Array(buffer);
    }
  };
  const alloc = (bytes: number) => {
    const at = top;
    top = align(at + bytes);
    if (top > memory.buffer.byteLength) growMemory(memory, Math.ceil((top - memory.buffer.byteLength) / PAGE), wide);
    views();
    return at;
  };
  views();

  // (T375.3) the memory as the parts are handed it: the views as they are now, read where they are used
  const room: Room = {
    alloc,
    get F() { return F; },
    get U() { return U; },
    get H() { return H; },
  };

  const count = (t: Tensor) => t.shape.reduce((a, b) => a * b, 1);
  // T363.5: what the kernels of llama.cpp's blocks read besides a matrix: their tables, made here once of the
  // checkpoint's codebooks (0: the model has no such rows); and for the tile of the signed dot product, taken where
  // this engine's relaxed SIMD multiplies signed bytes by signed bytes exactly (every pair of bytes is asked, once),
  // the bytes each thread spreads a row into, for the widest row there is
  let tables = 0, spread = 0;
  if (blocked) {
    tables = alloc(BLOCK_TABLES);
    if (!T.codebooks || k.sb_tables(tables, base + T.codebooks.offset) !== BLOCK_TABLES) throw new Error("The kernels of llama.cpp's blocks did not make their tables of this checkpoint.");
    if (relaxed?.matmul_sbw_tile && (trying.signed ?? signedDots(relaxed, alloc(64)))) {
      const widest = Math.max(...Object.values(T).filter((t) => partsOf(t).some(({ kind }) => packingOf(kind)?.tabled !== undefined)).map((t) => t.shape[t.shape.length - 1]));
      spread = alloc(SPREAD_THREADS * spreadBytes(widest));
    }
  }
  // Whether the relaxed dot product of this engine multiplies signed bytes by signed bytes exactly, for every pair of
  // bytes: what matmul_sbw_tile needs (kernels/README.md; arm64's does, x86's takes the second for unsigned). at: 48 bytes
  function signedDots(r: Kernels, at: number) {
    const I8 = new Int8Array(memory.buffer), I32 = new Int32Array(memory.buffer);
    for (let x = -128; x < 128; x++) {
      for (let y0 = -128; y0 < 128; y0 += 16) {
        let want = 0;
        for (let j = 0; j < 16; j++) {
          I8[at + j] = x;
          I8[at + 16 + j] = y0 + j;
          want += x * (y0 + j);
        }
        r.signed_dots(at + 32, at, at + 16);
        if (I32[at / 4 + 8] + I32[at / 4 + 9] + I32[at / 4 + 10] + I32[at / 4 + 11] !== want) return false;
      }
    }
    return true;
  }
  // a float32 copy of a tensor: float16 converted; a packed one (int8, int6, ternary: jobs.ts's PACKED says how each
  // weight is read) times the scale of its group (Math.fround is the float32 product NumPy computes); T363.5: a stack
  // whose layers are each of a kind of their own layer by layer, and rows of llama.cpp's blocks by their kernel
  function widen(t: Tensor) {
    const n = count(t), at = alloc(n * 4), parts = partsOf(t), each = n / parts.length, width = t.shape[t.shape.length - 1];
    parts.forEach((part, l) => {
      const to = at + l * each * 4, packing = packingOf(part.kind);
      if (part.kind === "f32") {
        F.copyWithin(to / 4, (base + part.offset) / 4, (base + part.offset) / 4 + each);
      } else if (part.kind === "f16") {
        for (let i = 0; i < each; i++) F[to / 4 + i] = halfToFloat(H[(base + part.offset) / 2 + i]);
      } else if (packing?.row) {
        const row = packing.row(k, tables), bytes = packing.rowBytes(width);
        for (let r = 0; r < each / width; r++) row(to + r * width * 4, base + part.offset + r * bytes, width);
      } else if (packing?.value) {
        const g = part.group, values = base + part.offset, scaleAt = scaleOf(packing, base + part.scales);
        for (let i = 0; i < each; i++) F[to / 4 + i] = Math.fround(packing.value(U, values, i) * scaleAt((i / g) | 0));
      } else {
        throw new Error(`This engine reads no weights stored as ${JSON.stringify(part.kind)}.`);
      }
    });
    return at;
  }
  // T533: the scale of group g of a packed tensor whose scales start at address scales: a float32, or a float16
  // (ternary's: packing.scale says which), widened exactly. Of the views as they are when it is called
  function scaleOf(packing: Packing, scales: number) {
    return packing.scale === 2 ? (g: number) => halfToFloat(H[scales / 2 + g]) : (g: number) => F[scales / 4 + g];
  }
  const widened = new Map<string, number>();
  // the address of a float32 tensor, widened once if the file does not hold float32
  function floats(name: TensorName | keyof Derived) {
    if (name in derived) {
      if (!widened.has(name)) {
        const bytes = derived[name as keyof Derived]!, at = alloc(bytes.length);
        U.set(bytes, at);
        widened.set(name, at);
      }
      return widened.get(name)!;
    }
    const t = T[name as TensorName];
    if (!t) return 0;
    if (t.kind === "f32") return base + t.offset;
    if (!widened.has(name)) widened.set(name, widen(t));
    return widened.get(name)!;
  }
  // a matrix: packed (values, scales, and the corrections of relaxed SIMD where its packing has them) when plan.int8,
  // else float32. T375.4: how a packing is multiplied is its line of jobs.ts's PACKED, read here once: the matrix keeps
  // the kind of its job (with relaxed SIMD or without, as this engine runs) and what makes its input ready.
  // T375.3: made once for a name, as floats() widens once: what the engine names for the GPU's side and what a kind of
  // layer takes for itself are the same thing
  const matrices = new Map<TensorName, Held | null>();
  function matrix(name: TensorName): Held | null {
    if (!matrices.has(name)) matrices.set(name, made(name));
    return matrices.get(name)!;
  }
  // (the activations as a packing's kernels take them: one function for a name, so that matrices packed alike in that
  // are told by it)
  const readies = new Map<string, Ready>();
  const readyOf = (packing: Packing) => {
    if (!readies.has(packing.takes)) readies.set(packing.takes, packing.ready(k, Boolean(relaxed)));
    return readies.get(packing.takes)!;
  };
  function made(name: TensorName): Held | null {
    const source: TensorName = plan.shared_classifier && name === "wcls" ? "token_embedding_table" : name;
    const t = T[source];
    if (!t) return null;
    const [rows, n] = t.shape.slice(-2);
    // T363.5: a stack with a kind for each layer is its layers, each a matrix of its own (matmuls() takes the layer's)
    if (t.layers) {
      const layers = t.layers.map((part) => held({ ...part, shape: [rows, n] }, source, rows, n));
      return { rows, n, kind: "layers", layers, layer: () => { throw new Error(`The matrix ${source} is its layers, each of a kind of its own.`); } };
    }
    return held(t, source, rows, n);
  }
  function held(t: Tensor, source: TensorName, rows: number, n: number): Held {
    // (scaled: the bytes of a layer's scales, T533: float16s for ternary; corrected: of its corrections, int32s)
    const packing = packingOf(t.kind), scaled = rows * (n / t.group) * (packing?.scale ?? 4), corrected = rows * (n / t.group) * 4;
    // T156: on the GPU alone: where each layer's values and scales start in the checkpoint (what gpu.ts was opened
    // with), for the plan the GPU's worker is started with; the CPU never multiplies by it (T210: nor by the classifier)
    if (direct && GPU_ALONE.includes(source)) {
      const rowBytes = packing!.rowBytes(n);
      return { rows, n, kind: t.kind as Packed, group: t.group, onGpu: true, layer: (l) => [t.offset + l * rows * rowBytes, t.scales + l * scaled] };
    }
    if (packing && plan.int8) {
      const rowBytes = packing.rowBytes(n), values = base + t.offset, scales = base + t.scales, groups = count(t) / t.group;
      // (T363.5: llama.cpp's blocks: the tables and the type where the others have scales and corrections, and last
      // the scratch of the signed tile, taken where there is one)
      if (packing.tabled !== undefined) {
        const type = packing.tabled, { jobs } = packing;
        const job = !relaxed?.matmul_sbr ? jobs.plain : spread && jobs.signed !== undefined ? jobs.signed : jobs.relaxed;
        return { rows, n, kind: t.kind as Packed, group: t.group, layer: (l) => [values + l * rows * rowBytes, tables, type, spread], job, ready: readyOf(packing) };
      }
      let corrections = 0;
      if (relaxed && packing.sums) {
        // relaxed SIMD multiplies by 7-bit unsigned activations with a bias of 64, which this takes out again:
        // dot(w, q - 64) = dot(w, q) - 64 * sum(w). -64 * sum of the group, an int32 (T197: the kernels add it to the
        // group's integer sum; before it the float32 scale * sum)
        corrections = alloc(groups * 4);
        // the same numbers as a sum in JavaScript, a kernel's speed (T98, T123: 7B spent 266 s here one value at a
        // time). Groups of 32: relaxed runs only where every row is whole groups
        k[packing.sums](corrections, values, groups);
      }
      // (a layer's values, scales and corrections: 0 where the packing has none)
      const layer = (l: number) => [values + l * rows * rowBytes, scales + l * scaled, corrections && corrections + l * corrected];
      return { rows, n, kind: t.kind as Packed, group: t.group, layer, job: relaxed ? packing.jobs.relaxed : packing.jobs.plain, ready: readyOf(packing) };
    }
    // (a layer of a stack that is float32 by itself: widened or taken as it is, where it is)
    if (t !== T[source]) return { rows, n, kind: "f32", job: KIND.matmul_f32, ready: null, layer: ((w) => () => [w])(t.kind === "f32" ? base + t.offset : widen(t)) };
    const w = floats(source);  // float32 as it is, or widened once (shared with the embedding when it is the same table)
    return { rows, n, kind: "f32", job: KIND.matmul_f32, ready: null, layer: (l) => [w + l * rows * n * 4] };
  }

  // ---- the activations: one frame per token, BATCH of them for a prompt (T108). A frame holds what one token needs,
  // in the order it always had, and the next token's frame follows: every array of token t is t * S bytes after token
  // 0's. (One block per array, BATCH tokens long, put token 0's arrays 32 KB apart for tiny-lm, and they fought for
  // the same lines of the cache: 3% slower on one token at a time.)
  // T110: an int8 model keeps its keys and values in float16 (half the bytes attention reads) where the memory is
  // shared, that is where there are software threads: several threads wait on the memory, and reading half of it
  // made a long context 1.2 times as fast with 4; one thread waits on the arithmetic, and widening every key and
  // value made it 1.6 to 1.8 times as slow. A float32 model, the one held to NumPy's numbers, stays in float32.
  // T160: a grouped-query model's too, unless that alone would not fit a 32-bit memory (keysInHalf, halfKeys). T130: and
  // on a memory that is not shared, float16 where float32 would not fit (halfKeys). The GPU's keys and values stay
  // float16 either way: the cache's putHalves() widens them into a float32 cache.
  const halfKV = Boolean(plan.half_kv) && (halfKeys ?? (sharedMemory && kvHeads >= heads));
  const D = dim * 4;
  // T375.3: a frame is the arrays every model has, what each kind of its layers puts there, and a rotated basis's
  // (memory.ts's frameArrays()); what a kind keeps of a token apart from its frame follows the frames, BATCH of each
  const framed = had.map(({ kind }) => kinds[kind].frame(sizes));
  // (xs: the most a group of 32 activations takes there by any packing of the model's)
  // (T331: as wide as the widest FFN, sizes.hidden)
  const inFrame = frameArrays(dim, sizes.hidden, kvDim, qDim, framed, rotated, packings.length ? Math.max(...packings.map(({ xs }) => xs)) : undefined), S = frameBytes(inFrame);
  const named = [...inFrame, ...framed.flatMap(({ apart = [] }) => apart)].map(([name]) => name);
  const twice = named.find((name, i) => named.indexOf(name) !== i);
  if (twice) throw new Error(`Two kinds of this model's layers keep an array called ${JSON.stringify(twice)} for a token.`);
  const frames = alloc(BATCH * S), at: Record<string, number> = {};
  inFrame.reduce((offset, [name, bytes]) => { at[name] = frames + offset; return offset + align(bytes); }, 0);
  for (const { apart = [] } of framed) for (const [name, bytes] of apart) at[name] = alloc(BATCH * bytes);
  const { x, xb, xb2, before, xq, xs, xr } = at;
  const logits = alloc(vocab * 4);
  // the tensors the layers and the GPU's side read, by their names (T375.3: in the order they were always asked for,
  // which is the order of what they take of the memory: a kind of layer and the FFN then take theirs by the same names)
  const wq = matrix("wq"), wk = matrix("wk"), wv = matrix("wv"), wo = matrix("wo");
  const w1 = matrix("w1"), w2 = matrix("w2"), w3 = matrix("w3"), wcls = matrix("wcls");
  const attW = floats("rms_att_weight"), ffnW = floats("rms_ffn_weight"), finalW = floats("rms_final_weight");
  // T261: the norms behind the attention and behind the FFN (0: the model has none; the FFN's is ffn.ts's to read)
  const attPost = floats("rms_att_post_weight"), ffnPost = floats("rms_ffn_post_weight");
  const attB = floats("ln_att_bias"), ffnB = floats("ln_ffn_bias"), finalB = floats("ln_final_bias");
  const bo = floats("bo"), b1 = floats("b1"), b2 = floats("b2"), bq = floats("bq"), bk = floats("bk"), bv = floats("bv");
  const qNorm = floats("q_norm"), kNorm = floats("k_norm");  // T124: Qwen3 normalizes every head of q and k
  const eps = plan.rms_norm_eps ?? 1e-5;  // the epsilon of every RMSNorm (T124: Qwen3's is 1e-6)
  // T331: the number every layer ends by multiplying the stream by (0: the model has none), and the cap of the logits
  // (softcap * tanh(logits / softcap); 0: none)
  const layerScale = floats("layer_scale"), softcap = plan.softcap;
  const cosTable = floats("freq_cis_real"), sinTable = floats("freq_cis_imag");
  const positions = floats("positions");  // GPT-2's learned table of positions (0: the model has none)
  const embedding = T.token_embedding_table!;
  const embeddingRows = embedding.kind === "f16" ? floats("token_embedding_table") : 0;
  // (a packed table's rows are read a weight at a time: jobs.ts's PACKED says how; T363.5: a row of llama.cpp's blocks
  // by its kernel)
  const embeddingPacked = packingOf(embedding.kind), embeddingRow = embeddingPacked?.row?.(k, tables), embeddingBytes = embeddingPacked?.rowBytes(dim) ?? 0;
  const embeddingValue = embeddingPacked?.value;
  // T375.3: the norm of a layer's input, of the FFN's and the last one, chosen once: LayerNorms (GPT-2, GPT-NeoX; b:
  // their biases) or RMSNorms
  const norm: (out: number, input: number, w: number, b: number) => void = layerNorm
    ? (out, input, w, b) => { k.layernorm(out, input, w, b, dim); }
    : (out, input, w) => { k.rmsnorm(out, input, w, dim, eps); };

  const runRows = runner(k, relaxed);

  // T378.2: the control area of this memory (forward/control.ts), through which this thread meets the software threads
  // and the GPU's worker: it is neither's, and both parts below are handed it
  const control = controlArea({ memory, sharedMemory, stalledMs });
  // T349: the phases and the software threads (forward/threads.ts): their functions by name. T378.1: their state is
  // theirs alone: what happens here is told through those functions, and threads.count, .lost and .searching are read.
  // No software thread where the model is on the GPU alone (T156: nothing runs on the CPU). T378.2: the two kernels a
  // thread is started with are chosen here, as this thread's are (relaxed above)
  const threads = softwareThreads({ memory, wide, kernels: { plain: kernels.plain, relaxed: plan.int8 && plan.relaxed ? kernels.relaxed : null, blocks },
    spawn: direct ? null : spawn, control, runRows,
    // (a function: the GPU's side is made after the threads, which it reads)
    gpuGettingReady: () => onGpu.gettingReady() });
  const { phase, searchLog, use, find, generationBegan, tokenBegan, tokenEnded, stop } = threads;

  // T375.3: the cache of the keys and values (forward/cache.ts): the layers that keep theirs there say so as they are
  // made, and it takes its room last, below. T210: none on the GPU alone, whose keys and values are the GPU's alone
  // (no step runs here, and one that fails is loaded again on the CPU)
  const cache = keyValueCache({ room, k, kvDim, half: halfKV, start: direct ? 0 : Math.min(plan.kv_start, seqLen), seqLen });
  const { KV } = cache;
  // T375.3: the layers, made by their kinds (the table's order), and the FFN: what each is handed is core (kinds.ts
  // says what is in it). Every layer's step is then in steps, which run() calls; the kinds that keep a state from
  // token to token are in keeping (follow())
  // (T363.5: how the matmuls of one input go out is chosen here once: a model whose rows are all of the file's dtype,
  // every model before T363, takes the one it always did)
  const matmuls = plan.own_kinds ? matmulsByKind : matmulsAlike;
  const core: Core = { plan, sizes, eps, k, room, floats, matrix, at, S, matmuls, phase, cache };
  const parts = Object.fromEntries(had.map(({ kind, count }) => [kind, kinds[kind].make(core, count, facts.filter((fact) => fact.kind === kind))]));
  const steps = facts.map((fact) => parts[fact.kind].layer(fact.place, fact));
  const keeping = Object.values(parts).filter((part) => part.clear);
  // T545: the state the GPU's worker goes on from, of the kind whose layers it has a step for (an LFM2's convolution
  // layers: null where the model has none), and how many layers keep keys and values in the cache (the GPU's own and
  // what comes back from it are theirs: every layer, but for the layers of such a kind)
  const gpuState = Object.values(parts).find((part) => part.rows)?.rows ?? null;
  const attending = had.filter(({ kind }) => kinds[kind].cached).reduce((sum, { count }) => sum + count, 0);
  const ffn = (plan.gated_ffn ? gatedFfn : plainFfn)(core);
  // T329.2: every layer's input of a table of its own (null: the model has none), its rows read by rows, or copied from
  // the checkpoint here
  const perLayered = perLayer(core, rows ?? ((offset, bytes, to) => { U.copyWithin(to, base + offset, base + offset + bytes); }));
  // the position that comes next for the layers that keep a state, -1 once a run stopped half way
  let stateAt = 0;

  // T237: the signs of a rotated basis for every width a matrix reads (Python's, each times 1 / sqrt(block))
  const signs: Record<number, number> | null = rotated ? Object.fromEntries(plan.widths.rotated.map((width) => {
    if (!(`signs.${width}` in derived)) throw new Error(`The rotated basis has no signs for an input ${width} wide.`);
    return [width, floats(`signs.${width}`)];
  })) : null;

  // the outlier channels of the classifier's input (T92): their columns in float32, multiplied apart (T210: not of a
  // classifier on the GPU alone, which is not here; T226: the GPU multiplies a float classifier for a model with them,
  // so such a model stays on the GPU alone and needs no columns)
  const channels = plan.outliers ?? [];
  let columns = 0, picked = 0;
  if (channels.length && !direct) {
    const t = (plan.shared_classifier ? T.token_embedding_table : T.wcls)!;
    columns = alloc(channels.length * vocab * 4);
    picked = alloc(channels.length * 4);
    const groups = dim / t.group, values = base + t.offset, packing = packingOf(t.kind)!, value = packing.value!;
    const scaleAt = scaleOf(packing, base + t.scales);
    channels.forEach((c, i) => {
      for (let v = 0; v < vocab; v++) {
        F[columns / 4 + i * vocab + v] = Math.fround(value(U, values, v * dim + c) * scaleAt(v * groups + ((c / t.group) | 0)));
      }
    });
  }
  // T378.2: what the GPU's side (forward/gpuside.ts) reads of this model, as two things made once: the model's facts,
  // and where its tensors are in the memory (addresses, and the matrices as matrix() made them)
  const shape: Shape = { dim, hidden, layers, heads, kvHeads, headSize, kvDim, qDim, vocab, seqLen, D, KV, halfKV, eps, rotary: plan.rotary, kvStart: plan.kv_start,
    turning: facts.some(({ rope }) => rope), layerNorm, gatedFfn: plan.gated_ffn, activation: plan.activation,
    local: facts.some(({ window, local }) => window > 0 || local), parallel, unturned, linear: plan.linear ?? null, convolution: plan.convolution ?? null,
    // (T545: a kind that keeps a state the GPU's worker has no step for)
    stateful: keeping.some((part) => !part.rows), rotated, sharedClassifier: plan.shared_classifier,
    // T545: every layer's kind and its place among the layers of its kind, the layers that keep keys and values in the
    // cache, and the state the GPU's worker goes on from
    kinds: facts.map(({ kind }) => kind), places: facts.map(({ place }) => place), attending, state: gpuState,
    // T261.4: every layer's window and the layers the local tables turn (the GPU's plan has them), and what a Gemma 4's
    // layers have besides, which the GPU's have not
    windows: facts.map(({ window }) => window), locals: facts.flatMap(({ local }, l) => (local ? [l] : [])),
    extras: Boolean(plan.value_norm || softcap || layerScale || perLayered) ||
      facts.some(({ wide, double, kept, reads, place }) => wide || double || kept !== place || reads !== place) };
  const weights: Placed = { base, embedding, wq, wk, wv, wo, w1, w2, w3, wcls, attW, ffnW, attB, ffnB, finalW, finalB, bo, b1, b2, bq, bk, bv, qNorm, kNorm,
    cosTable, sinTable, positions, channels, attPost, ffnPost, localCos: floats("local_cis_real"), localSin: floats("local_cis_imag"),
    // (T545: an LFM2's matrices in and out and its taps, which its kind made: null and 0 where there are none. A Qwen3.5's
    // linear-attention layers have a wout and taps of their own, which are no convolution layer's)
    win: plan.convolution ? matrix("win") : null, wout: plan.convolution ? matrix("wout") : null, conv: plan.convolution ? floats("conv") : 0 };
  // T349: what the GPU would hold of this model, and why it would not take it
  const fit = gpuFit({ shape, weights, device: { gpuRoom, memoryUnsaid, sharedMemory }, direct });

  // T135: where the GPU's worker puts the keys and values of a prompt's block (float16, T147: the keys
  // [layers][GPU_BLOCK][kvDim], then the values the same), for the cache below, and where it reads the block's rows
  // (dense). Only where the page asked for the GPU and it can take this model
  const gpuWhyNot = gpu ? fit.unfit() : null;
  // (T210: a model on the GPU alone has its keys and values read back there only for the tests, keysAndValues; and the
  // GPU embeds its rows itself)
  // (T545: the attending layers' alone, and after them the state the GPU's worker leaves of a block, where it has one)
  const stagingKeys = 2 * attending * GPU_BLOCK * kvDim * 2;
  const staging = gpu && !gpuWhyNot ? alloc(stagingKeys + (gpuState?.bytes ?? 0)) : 0;
  const gpuRows = staging && !direct ? alloc(GPU_BLOCK * D) : 0;
  // T152: why a generation's steps stay on the CPU where the prompt's blocks may go to the GPU (else null), and where
  // the GPU's worker writes the ids of the steps it took ([sampled, id, ...])
  const tokensWhyNot = staging ? fit.tokensUnfit() : null;
  // (T219: and after the ids, the word that says the step after them was refused: its logits were not finite)
  const gpuIds = staging && !tokensWhyNot ? alloc((2 + GPU_TOKENS) * 4) : 0;

  // the cache of the keys and values takes its room: last in the memory, where it grows (forward/cache.ts)
  cache.place();
  // the job (jobs.ts) of a matrix by count tokens of an input: what phase() shares out. Its kind is the matrix's (made());
  // a packed matrix reads the activations made ready in xq and xs, a float32 one the input as it is
  const jobOf = (m: Held, out: number, outStride: number, input: number, l: number, count: number) => {
    const [w, s, c] = m.layer(l);
    return m.ready ? [m.job!, out, xq, xs, w, s, c, m.n, 0, m.rows, count, outStride, S, S] : [m.job!, out, input, 0, w, 0, 0, m.n, 0, m.rows, count, outStride, S, 0];
  };
  // T363.5: the same of a model whose rows have kinds of their own (the plan's own_kinds): a stack may be its layers,
  // each a matrix of a kind of its own, and the matrices of one input may take their activations in different forms.
  // Those that take them alike go out as one phase, after the activations are made ready for them; then the next
  function matmulsByKind(input: number, count: number, list: Product[]) {
    const helds = list.map(([m, out, outStride, l]): [Held, number, number, number] => (m!.layers ? [m!.layers[l], out, outStride, 0] : [m!, out, outStride, l]));
    const { n } = helds[0][0];
    if (rotated) {
      for (let t = 0; t < count; t++) k.rotate(xr + t * S, input + t * S, signs![n], n, rotated);
      input = xr;
    }
    const readied = helds.map(([m]) => m.ready ?? null);
    readied.forEach((ready, first) => {
      if (readied.indexOf(ready) !== first) return;
      if (ready) for (let t = 0; t < count; t++) ready(xq + t * S, xs + t * S, input + t * S, n);
      phase(helds.filter((_, i) => readied[i] === ready).map(([m, out, outStride, l]) => {
        const [w, s, c, last = 0] = m.layer(l);
        return m.ready ? [m.job!, out, xq, xs, w, s, c, m.n, last, m.rows, count, outStride, S, S] : [m.job!, out, input, 0, w, 0, 0, m.n, 0, m.rows, count, outStride, S, 0];
      }));
    });
  }
  // matmuls of one input (count tokens of it, a frame apart): [matrix, output, output stride, layer] each
  // T237: in a rotated basis the matrices read R of the input, turned here once for all of them (a kernel on this
  // thread: 5 blocks of 1024 take a few microseconds, a thousandth of the rows they are multiplied by)
  function matmulsAlike(input: number, count: number, list: Product[]) {
    const { n, ready } = list[0][0]!;
    if (rotated) {
      for (let t = 0; t < count; t++) k.rotate(xr + t * S, input + t * S, signs![n], n, rotated);
      input = xr;
    }
    // the activations as the first matrix's kernels take them (the matrices of one input are packed alike)
    if (ready) for (let t = 0; t < count; t++) ready(xq + t * S, xs + t * S, input + t * S, n);
    phase(list.map(([m, out, outStride, l]) => jobOf(m!, out, outStride, input, l, count)));
  }

  // the embedding rows of tokens at positions pos0, pos0 + 1, ... into x, a frame apart (or into rows, stride apart)
  function embed(tokens: number[], pos0: number, rows = x, stride = S) {
    for (let t = 0; t < tokens.length; t++) {
      const row = tokens[t] * dim, to = (rows + t * stride) / 4;
      if (embeddingValue) {
        const g = embedding.group, values = base + embedding.offset, scaleAt = scaleOf(embeddingPacked!, base + embedding.scales);
        for (let i = 0; i < dim; i++) {
          F[to + i] = Math.fround(embeddingValue(U, values, row + i) * scaleAt(((row + i) / g) | 0));
        }
      } else if (embeddingRow) {
        embeddingRow(rows + t * stride, base + embedding.offset + tokens[t] * embeddingBytes, dim);
      } else {
        const from = embedding.kind === "f32" ? base + embedding.offset : embeddingRows;
        F.copyWithin(to, from / 4 + row, from / 4 + row + dim);
      }
      if (positions) k.add_inplace(rows + t * stride, positions + (pos0 + t) * D, dim);
      // T237: the table of a rotated basis holds rotated rows
      if (rotated) k.unrotate(rows + t * stride, rows + t * stride, signs![dim], dim, rotated);
    }
  }
  // T147: the keys and values of count positions from pos that the GPU's worker put into staging (float16, [keys,
  // values][layer][GPU_BLOCK positions]), into the cache
  // (T545: by the cache's places, the attending layers in order: every layer where every layer attends; and the
  // staging place lays out as many)
  function fromStaging(pos: number, count: number) {
    const { capacity } = cache;
    for (let l = 0; l < attending; l++) {
      // T261.4: a layer whose window a ring holds (T510.1): each position at its two places there, and the ring told of
      // the positions written, as the layer itself tells it (attention.ts: a run that would go back past what the ring
      // still holds is refused there)
      const ring = cache.ring(l);
      if (ring) {
        const twice = ring.size * ring.row;
        for (let t = 0; t < count; t++) {
          const slot = ((pos + t) % ring.size) * ring.row, key = staging + (l * GPU_BLOCK + t) * kvDim * 2, value = staging + ((attending + l) * GPU_BLOCK + t) * kvDim * 2;
          cache.putHalves(ring.keys + slot, ring.values + slot, key, value);
          cache.putHalves(ring.keys + slot + twice, ring.values + slot + twice, key, value);
        }
        ring.latest = pos === 0 ? count - 1 : Math.max(ring.latest, pos + count - 1);
        continue;
      }
      // (a layer's block among those that keep every position: l * KV a position in, where no ring is)
      const offset = capacity * cache.offset(cache.index(l)), layerKeys = cache.keys + offset, layerValues = cache.values + offset;
      for (let t = 0; t < count; t++) {
        cache.putHalves(layerKeys + (pos + t) * KV, layerValues + (pos + t) * KV,
          staging + (l * GPU_BLOCK + t) * kvDim * 2, staging + ((attending + l) * GPU_BLOCK + t) * kvDim * 2);
      }
    }
  }
  // T243: whether the keys and values of count positions in staging are all finite numbers. The CPU's readers of a
  // float16 (the kernels' halves4, and so from_f16 and attention_f16) make a finite number of a NaN and of an infinity
  // (65536 and more), so that what a GPU computed wrong would be read from the cache as numbers ever after, by the CPU's
  // steps and its logits (T195 sees nothing then). They are looked at by their bits (finite_f16: the exponent's five)
  // before anything of them is written: one pass of a kernel over what fromStaging reads, none in the attention's loops
  function stagingFinite(count: number) {
    for (let part = 0; part < 2 * attending; part++) {
      if (!k.finite_f16(staging + part * GPU_BLOCK * kvDim * 2, count * kvDim)) return false;
    }
    return true;
  }
  // one token (count 1) or up to BATCH tokens of a prompt at positions pos0, pos0 + 1, ... (T108). Every token is
  // computed as it would be alone: the same kernels on the same numbers, only the matmuls of a layer go out once for
  // all of them. The logits, if asked for, are the last token's.
  function run(tokens: number[], pos0: number, needLogits: boolean) {
    // T156: the layers are on the GPU alone: a block or a step it did not take cannot run here
    if (direct) throw onGpu.notHere();
    const count = tokens.length;
    if (pos0 + count - 1 >= cache.capacity) cache.grow(pos0 + count - 1);
    views();
    cpuWrote(pos0);  // T135: from here on the cache holds what the GPU does not
    if (keeping.length) follow(pos0);
    embed(tokens, pos0);
    if (perLayered) perLayered.inputs(tokens, count);
    for (let l = 0; l < layers; l++) {
      for (let t = 0; t < count; t++) {
        norm(xb + t * S, x + t * S, attW + l * D, attB + l * D);
        if (parallel) F.copyWithin((before + t * S) / 4, (x + t * S) / 4, (x + t * S) / 4 + dim);  // GPT-NeoX reads this layer's input twice
      }
      // the layer itself, whatever its kind: from xb into xb2
      steps[l](count, pos0);
      for (let t = 0; t < count; t++) {
        if (attPost) k.rmsnorm(xb2 + t * S, xb2 + t * S, attPost + l * D, dim, eps);  // T261: a Gemma normalizes what joins the stream
        k.add_inplace(x + t * S, xb2 + t * S, dim);
      }
      // the FFN reads the norm of x (GPT-NeoX's parallel residual: of this layer's input), after the bias of the
      // output projection where the model has one (GPT-2 and GPT-NeoX)
      for (let t = 0; t < count; t++) {
        if (bo) k.add_inplace(x + t * S, bo + l * D, dim);
        norm(xb + t * S, (parallel ? before : x) + t * S, ffnW + l * D, ffnB + l * D);
      }
      ffn(l, count);
      if (perLayered) perLayered.layer(l, count);
      if (layerScale) {
        const scale = F[layerScale / 4 + l];
        for (let t = 0; t < count; t++) for (let i = (x + t * S) / 4, end = i + dim; i < end; i++) F[i] *= scale;
      }
    }
    if (keeping.length) stateAt = pos0 + count;
    if (!needLogits) return;
    const last = x + (count - 1) * S;
    norm(xb, last, finalW, finalB);
    channels.forEach((c, i) => {  // T92: the outlier channels are multiplied apart
      F[picked / 4 + i] = F[xb / 4 + c];
      F[xb / 4 + c] = 0;
    });
    matmuls(xb, 1, [[wcls, logits, 0, 0]]);
    if (channels.length) k.add_columns(logits, columns, picked, channels.length, vocab);
    if (softcap) {
      // (in float32, as NumPy: the quotient, its tanh and the product each rounded)
      for (let i = logits / 4, end = i + vocab; i < end; i++) F[i] = Math.fround(softcap * Math.fround(Math.tanh(Math.fround(F[i] / softcap))));
    }
  }
  const forward = (token: number, pos: number, needLogits: boolean) => run([token], pos, needLogits);

  // T229: the state of the layers that keep one is what the tokens before this position left: position 0 clears it
  // (every kind its own), and any other has to be the one that comes next (llama2_numpy's follow()). Keys and values
  // could be written again at any position; a state cannot, and tokens out of turn would compute on the wrong one
  // without a word. A run that stopped half way leaves it no token's: only position 0 goes on from there.
  function follow(pos: number) {
    if (pos === 0) {
      for (const part of keeping) part.clear!();
    } else if (pos !== stateAt) {
      throw new Error(`This model keeps a state from token to token: position ${stateAt < 0 ? 0 : stateAt} comes next ` +
        `(or 0, to begin again), not ${pos}.`);
    }
    stateAt = -1;
  }
  // T378.2: what the GPU's side is handed of the CPU's side: the memory and its views, the cache of the keys and values as
  // it is now (T349: getters, read each time: the cache moves as it grows) and grown, and what the CPU does for a block
  const cpuSide: CpuSide = {
    memory, views, grow: cache.grow, embed, fromStaging, stagingFinite, run,
    // T545: whether a block at pos may go on from the state the layers that keep one have (follow()'s rule: position 0,
    // or the one that comes next), and that state taken from what the GPU's worker left of a block after its keys and
    // values, the tokens up to next there now
    follows: (pos: number) => !keeping.length || pos === 0 || pos === stateAt,
    stateFromStaging(next: number) {
      if (gpuState) U.copyWithin(gpuState.at, staging + stagingKeys, staging + stagingKeys + gpuState.bytes);
      if (keeping.length) stateAt = next;
    },
    get H() { return H; },
    get capacity() { return cache.capacity; },
    // T261.4.2: a ring's (T510.1), or the layer's block among those that keep every position (as fromStaging finds them)
    places: () => Array.from({ length: attending }, (_, l) => {
      const ring = cache.ring(l), offset = ring ? 0 : cache.capacity * cache.offset(cache.index(l));
      return ring ? { keys: ring.keys, values: ring.values, ring: ring.size } : { keys: cache.keys + offset, values: cache.values + offset, ring: 0 };
    }),
  };
  // what the GPU is to take of this model (gpuFit()'s answers), and the places above where it answers
  const taken: Taken = { matrices: fit.matrices, vectors: fit.vectors, whyNot: gpuWhyNot, tokensWhyNot, staging, rows: gpuRows, ids: gpuIds };

  // T349: the GPU's side (forward/gpuside.ts): its functions by name. T378.2: its state is its own: what happens here is
  // told through those functions, and what the page is shown of it is read (onGpu.status, .ready, ...)
  const onGpu = gpuSide({ shape, weights, taken, cpuSide, control, threads,
    options: { gpu, remembered: gpuRemembered, force: gpuForce, direct, stalledMs } });
  const { cpuWrote, cpuStepBegan, cpuStepEnded, generateMany, promptBlock, tokenBlock } = onGpu;

  let bound: Bound | null = null;
  // (T364.3: a model with rows of llama.cpp's blocks is said by those, the owner's words, whatever its other rows are)
  const said = blocked ? "llama.cpp's blocks" : wo && !wo.layers && wo.kind !== "f32" ? wo.kind : "int8";
  const backend = (plan.int8 ? `SIMD kernels, ${said}${relaxed ? ", relaxed SIMD" : ""}` : "SIMD kernels, float32") +
    (wide ? ", 64-bit memory" : "");  // T101: the status line says so, as it says every other way the model runs
  return {
    backend,
    /** Use n threads from the next token on (stage 2): starts the helpers that are missing. 1 on a memory that is
     * not shared. Resolves to the number in use. */
    setThreads: use,
    /** Find the number of threads while generating (see threads.js): from a hint, or from a count remembered from an
     * earlier visit, which is then only checked against its neighbours now and then (every recheck generations).
     * chose(count) is told the answer, compared(verdict) every comparison on the way. The helpers of the starting count are started (and warmed) before this
     * resolves, so the first tokens do not wait for them. */
    findThreads: find,
    /** the page starts a generation: now and then the remembered count is checked against its neighbours again (T223:
     * and first on the first generation of a load with no GPU getting ready, where the count in use is remembered from
     * an earlier visit or was found while the GPU got ready) */
    newGeneration() {
      generationBegan();
      onGpu.generationBegan();
    },
    get searching() {
      return threads.searching;
    },
    searchLog,
    get threads() {
      return threads.count;
    },
    /** the helper threads end; this engine runs on its own again */
    stopThreads: stop,
    /** T120: whether a software thread stopped under this engine, which then went on with one */
    get lostThreads() {
      return threads.lost;
    },
    /** the float32 array of Python's that forward() fills with the logits */
    bind(array: Bound) {
      bound = array.copy ? array.copy() : array;
    },
    /** T108: tokens of a prompt at positions pos, pos + 1, ...: the same as forward() for each of them in turn without
     * logits, BATCH at a time through the layers. T135: on the GPU where it is on (see gpuside.js), GPU_BLOCK at a time */
    forwardMany(tokens: Ids, pos: number) {
      onGpu.prompt(tokens.toJs ? tokens.toJs() : [...tokens], pos);
    },
    /** T147: how many tokens of a prompt Python hands forwardMany() at once: GPU_BLOCK where the GPU takes a whole
     * block (T148: where it is faster), else BATCH (a longer call keeps the worker from answering for longer, and the
     * CPU gains nothing from it) */
    get promptBlock() {
      return promptBlock();
    },
    /** T147: the shaders the GPU multiplies a prompt's matrices and runs its attention with (tests) */
    get gpuForm() {
      return onGpu.chosen?.matrices;
    },
    get gpuAttention() {
      return onGpu.chosen?.attention;
    },
    /** T135: where the worker has WebGPU, a promise of what the status line says of the GPU (gpuLine: "WebGPU where it
     * is faster than the CPU", or on the CPU and why), settled once the layers are on the GPU or it is known that they
     * will not be. T148: nothing waits for it */
    gpu: onGpu.note,
    /** T148: what the status line says of the GPU now (it changes with the times of the prompts), and what the GPU
     * chose as it started ({ matrices, attention, key, remembered, seconds }: the page shows it and remembers it) */
    get gpuStatus() {
      return onGpu.status;
    },
    get gpuReady() {
      return onGpu.ready;
    },
    /** T184: why the prompts stay on the CPU (no WebGPU, a fallback adapter, a model the GPU does not take yet, a
     * failure), or null while none is known */
    get gpuWhyNot() {
      return onGpu.whyNot;
    },
    /** T184 (the benchmark's page path): every block of a prompt on "cpu" or on "gpu" (where the GPU is on and holds
     * the keys and values before it, as gpuForce.always), or null: each where it is faster (promptTimes) */
    set gpuSide(side: string | null) {
      onGpu.put(side);
    },
    /** T135: the tokens of a prompt that went through the GPU since the generation began */
    get gpuTokens() {
      return onGpu.promptTokens;
    },
    /** T152: the steps of a generation the GPU took since it began */
    get gpuSampled() {
      return onGpu.sampled;
    },
    /** T152: why a generation's steps stay on the CPU (as gpuWhyNot, and the model's layer of a token, or the device's
     * memory for the classifier), or null */
    get gpuTokensWhyNot() {
      return onGpu.stepsWhyNot;
    },
    /** T152: whether this model's steps were asked of the GPU (the model and the memory allow them: tests) */
    get gpuTokensPlanned() {
      return Boolean(gpuIds);
    },
    /** T152: how many steps of a generation Python hands generateMany() at once now: GPU_TOKENS where the GPU takes
     * them (it is faster, or it is timed again), 0 where the CPU takes the step (the forward pass, and the sampling in
     * Python). Not while the threads are searched: that times the CPU's tokens */
    get tokenBlock() {
      return tokenBlock();
    },
    /** T152: count steps of generate() on the GPU, or undefined where it did not take them (gpuside.js's generateMany) */
    generateMany,
    forward(token: number, pos: number, needLogits = true) {
      if (tokenBegan(needLogits)) {
        // a token of the search for the number of threads (threads.ts), timed for it
        const began = clock();
        forward(token, pos, needLogits);
        tokenEnded(clock() - began);
      } else if (cpuStepBegan(needLogits)) {
        // T152: a step on the CPU, timed against the GPU's
        const began = performance.now();
        forward(token, pos, needLogits);
        cpuStepEnded(performance.now() - began);
      } else {
        forward(token, pos, needLogits);
      }
      if (!needLogits || !bound) return;
      const view = bound.getBuffer ? bound.getBuffer("f32") : { data: bound, release() {} };
      view.data.set(new Float32Array(memory.buffer, logits, vocab));
      view.release();
    },
    /** Python's array goes back (Llama.release()), the helper threads end, and the GPU's worker (T135). T205: resolves
     * once the GPU's worker has let go of its buffers and its device, or after GPU_END_MS (the worker is terminated
     * then), true where it said so: the next model is read after it (an iPhone's tab went down where the GPU of the
     * one before still held up to 189 MB as the next came from the cache) */
    release() {
      bound?.destroy?.();
      bound = null;
      this.stopThreads();
      return onGpu.release();
    },
    /** the logits in this memory, for callers without Python (tests) */
    logits: () => new Float32Array(memory.buffer, logits, vocab),
    /** the keys and values in the cache at positions from .. from + count - 1, in float32, [layers][count][kvDim] each
     * (T229: the layers that attend)
     * (tests: T135 holds what the GPU wrote back to what the CPU computes). T210: on the GPU alone, the GPU's own, read
     * back from it (GPU_BLOCK positions a request) */
    keysAndValues(from: number, count: number) {
      if (direct) return onGpu.keysAndValues(from, count);
      views();
      return cache.read(from, count);
    },
    memoryBytes: () => memory.buffer.byteLength,
  };
}
// (T368.4) the engine of one model, as Python and the worker call it
export type Engine = ReturnType<typeof createForward>;
