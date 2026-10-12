// forward/engine.ts (T349): createForward(), one model's forward pass: its memory after the checkpoint, its tensors,
// the frames of the activations, the round of the layers, and the object Python and the worker call. The software
// threads are threads.ts's and the GPU's side gpuside.ts's: each is handed what it reads of this function. T375.3:
// what a layer does is its kind's (kinds.ts's table: attention.ts, linear.ts, convolution.ts), the FFN is ffn.ts's and
// the cache of the keys and values cache.ts's: each is chosen or made here once, and a token's path calls what it gave.
// A module of src/runtime/forward.ts.

import { BATCH, addressed, runner } from "../jobs.ts";
import { GPU_BLOCK, GPU_TOKENS } from "./choice.ts";
import { growMemory, PAGE, align, frameArrays, frameBytes, halfToFloat } from "./memory.ts";
import { LAYER_KINDS } from "./kinds.ts";
import { keyValueCache } from "./cache.ts";
import { gatedFfn, plainFfn } from "./ffn.ts";
import { GPU_ALONE } from "./alone.ts";
import { controlArea } from "./control.ts";
import { softwareThreads } from "./threads.ts";
import { gpuFit, gpuSide } from "./gpuside.ts";
import type { Kernels } from "../jobs.ts";
import type { Direct } from "./alone.ts";
import type { CpuSide, GpuWorker, Held, Placed, Shape, Taken } from "./gpuside.ts";
import type { Spawn } from "./threads.ts";
import type { Force, Remembered } from "../contracts/gpu/protocol.ts";
import type { Core, Kinds, Product, Room } from "./kinds.ts";
import type { Derived, Plan, Tensor, TensorName } from "../contracts/plan.ts";

// ---- (T368.4, the types)
// The kernels of one build as modules (forward.ts's compileKernels()): relaxed: null where the browser has no relaxed
// SIMD; wide: the build for a 64-bit memory
export type Compiled = { plain: WebAssembly.Module; relaxed: WebAssembly.Module | null; wide?: boolean };
// What an engine is made of (the comments above createForward() say each)
export type Made = {
  memory: WebAssembly.Memory; base: number; size: number; kernels: Compiled; plan: Plan;
  spawn?: Spawn | null; gpu?: () => GpuWorker; gpuRoom?: number; memoryUnsaid?: boolean; gpuRemembered?: Remembered; gpuForce?: Force;
  halfKeys?: boolean; direct?: Direct;
  wrap?: (exports: Kernels) => Kernels; stalledMs?: number; clock?: () => number; kinds?: Kinds;
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
/** direct (T156, external()): the layers' matrices (T210: and the tables) are on the GPU alone: nothing runs on the CPU,
 * every block and step goes to the GPU (as gpuForce.always), and one the GPU cannot take throws (the worker loads the
 * model again on the CPU). T210: no cache of keys and values here either; the GPU embeds a prompt's tokens itself */
export function createForward({ memory, base, size, kernels, plan, spawn, gpu, gpuRoom, memoryUnsaid, gpuRemembered, gpuForce = {},
  halfKeys, direct, wrap = (exports) => exports, stalledMs = STALLED_MS, clock = () => performance.now(), kinds = LAYER_KINDS }: Made) {
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
  const had = Object.keys(kinds).map((kind) => ({ kind, count: facts.filter((fact) => fact.kind === kind).length })).filter(({ count }) => count);
  // T255: the layers RoPE leaves alone, as the GPU's plan lists them
  const unturned = Array.from(plan.unturned || []);
  // T237: the block of a rotated basis (0: the model's own basis). llama2_numpy.py has the definition above hadamard()
  const rotated = plan.rotated || 0;
  const imports = { env: { memory } };
  const wide = Boolean(kernels.wide);  // T101: a 64-bit memory, whose kernels take their addresses as BigInt
  // a page that is not cross-origin isolated has no SharedArrayBuffer to ask about: its memory is not shared
  const sharedMemory = typeof SharedArrayBuffer !== "undefined" && memory.buffer instanceof SharedArrayBuffer;
  const k = wrap(addressed(new WebAssembly.Instance(kernels.plain, imports).exports, wide));
  const relaxed = plan.int8 && plan.relaxed && kernels.relaxed ? wrap(addressed(new WebAssembly.Instance(kernels.relaxed, imports).exports, wide)) : null;
  const bias = relaxed ? 64 : 0;

  // ---- memory: the checkpoint at base, everything else after it; views are made again after the memory grows
  let top = align(base + size);
  // (F!: of the four, F alone is read outside a function, after views() made them)
  let buffer: ArrayBufferLike | undefined, F!: Float32Array, I: Int8Array, U: Uint8Array, H: Uint16Array;
  const views = () => {
    if (buffer !== memory.buffer) {
      buffer = memory.buffer;
      F = new Float32Array(buffer); I = new Int8Array(buffer); U = new Uint8Array(buffer); H = new Uint16Array(buffer);
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

  const T = plan.tensors, derived: Derived = plan.derived ?? {};
  const count = (t: Tensor) => t.shape.reduce((a, b) => a * b, 1);
  // weight i of an int8, int6 or ternary tensor, as the int8 it is: int6 (T98) unpacked from its group of 24 bytes,
  // the layout of llama2_numpy.pack6 (six bits, then two zero bits); ternary (T230) its two bits less one, the layout
  // of llama2_numpy.pack_ternary
  function weightAt(t: Tensor, i: number) {
    if (t.kind === "int8") return I[base + t.offset + i];
    if (t.kind === "ternary") return ((U[base + t.offset + (i >> 2)] >> (2 * (i & 3))) & 3) - 1;
    const group = base + t.offset + ((i / 32) | 0) * 24, j = i % 32;
    const low = j < 16 ? U[group + j] & 15 : U[group + j - 16] >> 4;
    const top = (U[group + 16 + (j % 8)] >> (2 * ((j / 8) | 0))) & 3;
    return (((low | (top << 4)) << 2) << 24) >> 24;  // the byte as a signed int8
  }
  // a float32 copy of a tensor: float16 converted, int8, int6 and ternary times the scale of their group (Math.fround is the
  // float32 product NumPy computes)
  function widen(t: Tensor) {
    const n = count(t), at = alloc(n * 4);
    if (t.kind === "f16") {
      for (let i = 0; i < n; i++) F[at / 4 + i] = halfToFloat(H[(base + t.offset) / 2 + i]);
    } else {
      const g = t.group;
      for (let i = 0; i < n; i++) F[at / 4 + i] = Math.fround(weightAt(t, i) * F[(base + t.scales) / 4 + ((i / g) | 0)]);
    }
    return at;
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
  // a matrix: int8 or int6 (values, scales, corrections) or ternary (values, scales) when plan.int8, else float32.
  // T375.3: made once for a name, as floats() widens once: what the engine names for the GPU's side and what a kind of
  // layer takes for itself are the same thing
  const matrices = new Map<TensorName, Held | null>();
  function matrix(name: TensorName): Held | null {
    if (!matrices.has(name)) matrices.set(name, made(name));
    return matrices.get(name)!;
  }
  function made(name: TensorName): Held | null {
    const source: TensorName = plan.shared_classifier && name === "wcls" ? "token_embedding_table" : name;
    const t = T[source];
    if (!t) return null;
    const [rows, n] = t.shape.slice(-2);
    // T156: on the GPU alone: where each layer's values and scales start in the checkpoint (what gpu.ts was opened
    // with), for the plan the GPU's worker is started with; the CPU never multiplies by it (T210: nor by the classifier)
    if (direct && GPU_ALONE.includes(source)) {
      const ternary = t.kind === "ternary";
      return { rows, n, int8: true, six: false, ternary, group: t.group, onGpu: true,
        layer: (l) => [t.offset + l * rows * (ternary ? n / 4 : n), t.scales + l * rows * (n / t.group) * 4] };
    }
    if ((t.kind === "int8" || t.kind === "int6" || t.kind === "ternary") && plan.int8) {
      const six = t.kind === "int6", ternary = t.kind === "ternary", rowBytes = ternary ? n / 4 : six ? n / 32 * 24 : n;
      const values = base + t.offset, scales = base + t.scales, groups = count(t) / t.group;
      let corrections = scales;
      if (relaxed && !ternary) {
        // relaxed SIMD multiplies by 7-bit unsigned activations with a bias of 64, which this takes out again:
        // dot(w, q - 64) = dot(w, q) - 64 * sum(w). -64 * sum of the group, an int32 (T197: the kernels add it to the
        // group's integer sum; before it the float32 scale * sum)
        corrections = alloc(groups * 4);
        // the same numbers as a sum in JavaScript, a kernel's speed (T98, T123: 7B spent 266 s here one value at a
        // time). Groups of 32: relaxed runs only where every row is whole groups
        (six ? k.six_sums : k.int8_sums)(corrections, values, groups);
      }
      const layer = (l: number) => [values + l * rows * rowBytes, scales + l * rows * (n / t.group) * 4, corrections + l * rows * (n / t.group) * 4];
      return { rows, n, int8: true, six, ternary, group: t.group, layer };
    }
    const w = floats(source);  // float32 as it is, or widened once (shared with the embedding when it is the same table)
    return { rows, n, int8: false, layer: (l) => [w + l * rows * n * 4] };
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
  const framed = had.map(({ kind }) => kinds[kind].frame(plan));
  const inFrame = frameArrays(dim, hidden, kvDim, qDim, framed, rotated, T.wo?.kind === "ternary"), S = frameBytes(inFrame);
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
  const attB = floats("ln_att_bias"), ffnB = floats("ln_ffn_bias"), finalB = floats("ln_final_bias");
  const bo = floats("bo"), b1 = floats("b1"), b2 = floats("b2"), bq = floats("bq"), bk = floats("bk"), bv = floats("bv");
  const qNorm = floats("q_norm"), kNorm = floats("k_norm");  // T124: Qwen3 normalizes every head of q and k
  const eps = plan.rms_norm_eps ?? 1e-5;  // the epsilon of every RMSNorm (T124: Qwen3's is 1e-6)
  const cosTable = floats("freq_cis_real"), sinTable = floats("freq_cis_imag");
  const positions = floats("positions");  // GPT-2's learned table of positions (0: the model has none)
  const embedding = T.token_embedding_table!;
  const embeddingRows = embedding.kind === "f16" ? floats("token_embedding_table") : 0;
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
  const threads = softwareThreads({ memory, wide, kernels: { plain: kernels.plain, relaxed: plan.int8 && plan.relaxed ? kernels.relaxed : null },
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
  const core: Core = { plan, eps, k, room, floats, matrix, at, S, matmuls, phase, cache };
  const parts = Object.fromEntries(had.map(({ kind, count }) => [kind, kinds[kind].make(core, count)]));
  const steps = facts.map((fact) => parts[fact.kind].layer(fact.place, fact));
  const keeping = Object.values(parts).filter((part) => part.clear);
  const ffn = (plan.gated_ffn ? gatedFfn : plainFfn)(core);
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
    const groups = dim / t.group;
    channels.forEach((c, i) => {
      for (let v = 0; v < vocab; v++) {
        F[columns / 4 + i * vocab + v] = Math.fround(weightAt(t, v * dim + c) * F[(base + t.scales) / 4 + v * groups + ((c / t.group) | 0)]);
      }
    });
  }
  // T378.2: what the GPU's side (forward/gpuside.ts) reads of this model, as two things made once: the model's facts,
  // and where its tensors are in the memory (addresses, and the matrices as matrix() made them)
  const shape: Shape = { dim, hidden, layers, heads, kvHeads, headSize, kvDim, qDim, vocab, seqLen, D, KV, halfKV, eps, rotary: plan.rotary, kvStart: plan.kv_start,
    turning: facts.some(({ rope }) => rope), layerNorm, parallel, unturned, linear: plan.linear ?? null, convolution: plan.convolution ?? null, stateful: keeping.length > 0,
    rotated, sharedClassifier: plan.shared_classifier };
  const weights: Placed = { base, embedding, wq, wk, wv, wo, w1, w2, w3, wcls, attW, ffnW, attB, ffnB, finalW, finalB, bo, b1, b2, bq, bk, bv, qNorm, kNorm,
    cosTable, sinTable, positions, channels };
  // T349: what the GPU would hold of this model, and why it would not take it
  const fit = gpuFit({ shape, weights, device: { gpuRoom, memoryUnsaid, sharedMemory }, direct });

  // T135: where the GPU's worker puts the keys and values of a prompt's block (float16, T147: the keys
  // [layers][GPU_BLOCK][kvDim], then the values the same), for the cache below, and where it reads the block's rows
  // (dense). Only where the page asked for the GPU and it can take this model
  const gpuWhyNot = gpu ? fit.unfit() : null;
  // (T210: a model on the GPU alone has its keys and values read back there only for the tests, keysAndValues; and the
  // GPU embeds its rows itself)
  const staging = gpu && !gpuWhyNot ? alloc(2 * layers * GPU_BLOCK * kvDim * 2) : 0;
  const gpuRows = staging && !direct ? alloc(GPU_BLOCK * D) : 0;
  // T152: why a generation's steps stay on the CPU where the prompt's blocks may go to the GPU (else null), and where
  // the GPU's worker writes the ids of the steps it took ([sampled, id, ...])
  const tokensWhyNot = staging ? fit.tokensUnfit() : null;
  // (T219: and after the ids, the word that says the step after them was refused: its logits were not finite)
  const gpuIds = staging && !tokensWhyNot ? alloc((2 + GPU_TOKENS) * 4) : 0;

  // the cache of the keys and values takes its room: last in the memory, where it grows (forward/cache.ts)
  cache.place();
  // the job (jobs.ts) of a matrix by count tokens of an input: what phase() shares out
  const jobOf = (m: Held, out: number, outStride: number, input: number, l: number, count: number) => {
    const [w, s, c] = m.layer(l);
    if (!m.int8) return [2, out, input, 0, w, 0, 0, m.n, 0, m.rows, count, outStride, S, 0];
    const kind = m.ternary ? (relaxed ? 8 : 9) : m.six ? (relaxed ? 5 : 6) : (relaxed ? 0 : 1);
    return [kind, out, xq, xs, w, s, relaxed && !m.ternary ? c : 0, m.n, 0, m.rows, count, outStride, S, S];
  };
  // matmuls of one input (count tokens of it, a frame apart): [matrix, output, output stride, layer] each
  // T237: in a rotated basis the matrices read R of the input, turned here once for all of them (a kernel on this
  // thread: 5 blocks of 1024 take a few microseconds, a thousandth of the rows they are multiplied by)
  function matmuls(input: number, count: number, list: Product[]) {
    if (rotated) {
      const n = list[0][0]!.n;
      for (let t = 0; t < count; t++) k.rotate(xr + t * S, input + t * S, signs![n], n, rotated);
      input = xr;
    }
    const [first] = list[0];
    if (first!.ternary) {
      // T231: the ternary kernels take the activations signed, in all 8 bits, and laid out as the weights' planes are
      for (let t = 0; t < count; t++) {
        k.quantize_x(xq + t * S, xs + t * S, input + t * S, first!.n, 0);
        k.interleave(xq + t * S, xs + t * S, first!.n);
      }
    } else if (first!.int8) {
      for (let t = 0; t < count; t++) k.quantize_x(xq + t * S, xs + t * S, input + t * S, first!.n, bias);
    }
    phase(list.map(([m, out, outStride, l]) => jobOf(m!, out, outStride, input, l, count)));
  }

  // the embedding rows of tokens at positions pos0, pos0 + 1, ... into x, a frame apart (or into rows, stride apart)
  function embed(tokens: number[], pos0: number, rows = x, stride = S) {
    for (let t = 0; t < tokens.length; t++) {
      const row = tokens[t] * dim, to = (rows + t * stride) / 4;
      if (embedding.kind === "int8" || embedding.kind === "int6" || embedding.kind === "ternary") {
        const g = embedding.group;
        for (let i = 0; i < dim; i++) {
          F[to + i] = Math.fround(weightAt(embedding, row + i) * F[(base + embedding.scales) / 4 + (((row + i) / g) | 0)]);
        }
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
  function fromStaging(pos: number, count: number) {
    const { capacity } = cache;
    for (let l = 0; l < layers; l++) {
      const layerKeys = cache.keys + l * capacity * KV, layerValues = cache.values + l * capacity * KV;
      for (let t = 0; t < count; t++) {
        cache.putHalves(layerKeys + (pos + t) * KV, layerValues + (pos + t) * KV,
          staging + (l * GPU_BLOCK + t) * kvDim * 2, staging + ((layers + l) * GPU_BLOCK + t) * kvDim * 2);
      }
    }
  }
  // T243: whether the keys and values of count positions in staging are all finite numbers. The CPU's readers of a
  // float16 (the kernels' halves4, and so from_f16 and attention_f16) make a finite number of a NaN and of an infinity
  // (65536 and more), so that what a GPU computed wrong would be read from the cache as numbers ever after, by the CPU's
  // steps and its logits (T195 sees nothing then). They are looked at by their bits (finite_f16: the exponent's five)
  // before anything of them is written: one pass of a kernel over what fromStaging reads, none in the attention's loops
  function stagingFinite(count: number) {
    for (let part = 0; part < 2 * layers; part++) {
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
    for (let l = 0; l < layers; l++) {
      for (let t = 0; t < count; t++) {
        norm(xb + t * S, x + t * S, attW + l * D, attB + l * D);
        if (parallel) F.copyWithin((before + t * S) / 4, (x + t * S) / 4, (x + t * S) / 4 + dim);  // GPT-NeoX reads this layer's input twice
      }
      // the layer itself, whatever its kind: from xb into xb2
      steps[l](count, pos0);
      for (let t = 0; t < count; t++) k.add_inplace(x + t * S, xb2 + t * S, dim);
      // the FFN reads the norm of x (GPT-NeoX's parallel residual: of this layer's input), after the bias of the
      // output projection where the model has one (GPT-2 and GPT-NeoX)
      for (let t = 0; t < count; t++) {
        if (bo) k.add_inplace(x + t * S, bo + l * D, dim);
        norm(xb + t * S, (parallel ? before : x) + t * S, ffnW + l * D, ffnB + l * D);
      }
      ffn(l, count);
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
    get H() { return H; },
    get capacity() { return cache.capacity; },
    get keys() { return cache.keys; },
    get values() { return cache.values; },
  };
  // what the GPU is to take of this model (gpuFit()'s answers), and the places above where it answers
  const taken: Taken = { matrices: fit.matrices, vectors: fit.vectors, whyNot: gpuWhyNot, tokensWhyNot, staging, rows: gpuRows, ids: gpuIds };

  // T349: the GPU's side (forward/gpuside.ts): its functions by name. T378.2: its state is its own: what happens here is
  // told through those functions, and what the page is shown of it is read (onGpu.status, .ready, ...)
  const onGpu = gpuSide({ shape, weights, taken, cpuSide, control, threads,
    options: { gpu, remembered: gpuRemembered, force: gpuForce, direct, stalledMs } });
  const { cpuWrote, cpuStepBegan, cpuStepEnded, generateMany, promptBlock, tokenBlock } = onGpu;

  let bound: Bound | null = null;
  const backend = (plan.int8 ? `SIMD kernels, ${["int6", "ternary"].includes(T.wo?.kind!) ? T.wo!.kind : "int8"}${relaxed ? ", relaxed SIMD" : ""}` : "SIMD kernels, float32") +
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
