// jobs.ts (T.109): what a job of a phase is, for the one that hands them out (forward.ts, in the model's worker) and
// for the software threads that take them (helper.ts). One place for the layout of the control area at the start of
// a shared memory, the kinds of kernel calls, and how the rows of a job are run, so that the two never drift apart.
//
// TypeScript of which only the types are erased (T.368.1: what runs is these lines). forward.ts and helper.ts import it, and a bundler links it into each (T.367.3: the two
// then hold the same text because they are of one build).

import { PYTHON_BLOCKS } from "./built/facts/blocks.ts";
/** the control area: the first bytes of a shared memory, as 32-bit words (the helpers', then the GPU's: GPU_DONE) */
export const CONTROL_BYTES = 8192;
// GEN: the generation of the phase (odd while the coordinator rewrites the jobs); QUIT: the helpers end; COUNTER:
// the next chunk to take; FINISHED: chunks done; ACTIVE: helpers inside a phase; TOTAL: chunks in all
export const GEN = 0, QUIT = 1, COUNTER = 2, FINISHED = 3, ACTIVE = 4, TOTAL = 5;
// WAKE + share: each helper's own word. The coordinator writes the generation there and wakes exactly the helpers it
// wants, always the same ones. (With one word for all, Atomics.notify wakes whoever has slept longest, so a
// different, cold helper took every phase.)
export const WAKE = 256;
// JOBS: the word that says how many jobs the phase has. The jobs themselves are a table of float64 from byte
// JOB_TABLE, JOB numbers each: a double holds an address exactly up to 2^53, where an int32 cut the addresses of a
// model past 4 GiB of a 64-bit memory (T.101: every helper computed on the wrong bytes, only the coordinator right)
export const JOBS = 512, JOB = 16, JOB_TABLE = 2056;
/** T.108: the most tokens that go through the layers together, and so the most jobs a phase has */
export const BATCH = 16;
/** scratch for a helper's warm-up, after the last job */
export const SCRATCH = 4608;
// T.135: the words of the GPU's worker (gpu.ts), which runs the blocks of a prompt that forward.ts hands it while
// forward.ts waits: GPU_DONE, the number of the last request it finished; GPU_FAILED, whether that one failed;
// GPU_BEAT, counted up while it works (a worker that stopped counting has stopped); GPU_WANTED, the request forward.ts
// waits for (T.147: 0 once it gave the GPU up, and the GPU's worker writes nothing for a request no longer wanted)
export const GPU_DONE = 1280, GPU_FAILED = 1281, GPU_BEAT = 1282, GPU_WANTED = 1283;
if (JOB_TABLE <= JOBS * 4 || JOB_TABLE % 8 || JOB_TABLE + BATCH * JOB * 8 > SCRATCH || SCRATCH + 448 > GPU_DONE * 4 ||
    (GPU_WANTED + 1) * 4 > CONTROL_BYTES) {
  throw new Error("the control area does not hold its jobs, the scratch and the GPU's words: see jobs.ts");
}

// A job: [kind, eight arguments, rows, count, out stride, a stride, b stride], and the control area holds it as it
// is, then the size of its chunks and the number of its first chunk. The kind is the number of a line of WORK below,
// which says what the eight arguments are to it and how its rows are run.
// count > 1 (T.108): the same rows for count tokens, whose out, a and b are that many bytes apart. The rows then go in
// blocks small enough to stay in the cache while every token uses them: each (row, token) is the one kernel call it
// is for a single token, so the numbers are the same as one token at a time. A kind that has a tile takes the count
// tokens in one call instead (a and b are then one frame apart, the same stride). Attention is always one token a
// job, and so is the delta rule (a Gated DeltaNet layer's state after a token is what the next token reads).
export const ROWS = 9, COUNT = 10, SIZE = 14, FIRST = 15;
const BLOCK_BYTES = 16384;
// T.231: the mask the ternary kernels take as an argument (kernels/ternary.ts: a constant written there would be made
// again at every use)
const THREE = 3;

/** T.101: which arguments of each kernel are addresses (usize in kernels/*.ts; tests/forward-check.ts holds this
 * table to the source). On a 64-bit memory the kernels take them as BigInt. */
export const ADDRESSES = {
  matmul_f32: [0, 1, 2], quantize_x: [0, 1, 2], quantize6_x: [0, 1, 2], ternary_x: [0, 1, 2], interleave: [0, 1],
  matmul_t2: [0, 1, 2, 3, 4], matmul_t2r: [0, 1, 2, 3, 4], six_sums: [0, 1], int8_sums: [0, 1], widen_bf16: [0, 1], widen_q8_0: [0, 1], widen_pq2_0: [0, 1], widen_ptq1_0: [0, 1], matmul_q8: [0, 1, 2, 3, 4],
  matmul_q6: [0, 1, 2, 3, 4], rmsnorm: [0, 1, 2], rope: [0, 1, 2], attention: [0, 1, 2, 3, 4], attention_f16: [0, 1, 2, 3, 4],
  to_f16: [0, 1], from_f16: [0, 1], finite_f16: [0], layernorm: [0, 1, 2, 3], gelu: [0, 1, 2], swiglu: [0, 1, 2], geglu: [0, 1, 2], geglu_erf: [0, 1, 2], add_columns: [0, 1, 2], add_inplace: [0, 1],
  gate: [0, 1, 2], convolve: [0, 1, 2], short_conv: [0, 1, 2, 3], delta_rule: [0, 1, 2, 3, 4], rotate: [0, 1, 2], unrotate: [0, 1, 2],
  argmax: [0], penalize: [0, 1], sample: [0, 5, 6], matmul_q8r: [0, 1, 2, 3, 4, 5], matmul_q6r: [0, 1, 2, 3, 4, 5],
  matmul_q8r_tile: [0, 1, 2, 3, 4, 5], matmul_t2r_tile: [0, 1, 2, 3, 4],
  // T.363.4: the kernels of llama.cpp's blocks (kernels/blocks.ts and blocks_relaxed.ts: modules of their own,
  // blockkernel_*.wasm, which addressed() takes as it takes the others, and an engine has beside the two modules
  // above only for a model with such rows: T.363.5)
  sb_tables: [0, 1], sums_x: [0, 1], sb_row: [0, 1, 2], matmul_sb: [0, 1, 2, 3, 4], matmul_sbr: [0, 1, 2, 3, 4],
  matmul_sbr_tile: [0, 1, 2, 3, 4], matmul_sbw_tile: [0, 1, 2, 3, 4, 12], signed_dots: [0, 1, 2],
};
// (T.368.1, the types. A kernel takes numbers, its addresses as BigInt on a 64-bit memory, and what it returns is for
// its caller to know: most return nothing. The kernels are the names of ADDRESSES and no other: a kernel that is
// called and is not in that table is a type error, where it was a Number for an address on a 64-bit memory.)
export type Kernel = (...args: (number | bigint)[]) => unknown;
// T.477: each kernel's arguments by their names in kernels/*.ts (an address a Number here: addressed() makes it a BigInt
// on a 64-bit memory). A second copy of the source's, which tests/forward-check.ts holds to it name by name; a call
// of a kernel with an argument too many or too few is a type error
type Arguments = {
  rmsnorm: [out: number, x: number, w: number, n: number, eps: number];
  layernorm: [out: number, x: number, w: number, b: number, n: number];
  gelu: [out: number, x: number, b: number, n: number];
  swiglu: [out: number, h1: number, h3: number, n: number];
  geglu: [out: number, h1: number, h3: number, n: number];
  geglu_erf: [out: number, h1: number, h3: number, n: number];
  add_inplace: [x: number, y: number, n: number];
  rotate: [out: number, x: number, signs: number, n: number, block: number];
  unrotate: [out: number, z: number, signs: number, n: number, block: number];
  rope: [v: number, fcr: number, fci: number, nh: number, hs: number, rot: number];
  attention: [out: number, q: number, kc: number, vc: number, att: number, pos: number, nh: number, nkv: number, hs: number, h0: number, h1: number];
  attention_f16: [out: number, q: number, kc: number, vc: number, att: number, pos: number, nh: number, nkv: number, hs: number, h0: number, h1: number];
  to_f16: [out: number, x: number, n: number];
  from_f16: [out: number, x: number, n: number];
  finite_f16: [x: number, n: number];
  matmul_f32: [xout: number, x: number, w: number, n: number, r0: number, r1: number];
  six_sums: [out: number, w: number, groups: number];
  int8_sums: [out: number, w: number, groups: number];
  matmul_q8: [xout: number, xq: number, xs: number, wq: number, ws: number, n: number, r0: number, r1: number];
  interleave: [xq: number, xs: number, n: number];
  matmul_t2: [xout: number, xq: number, xs: number, wq: number, ws: number, n: number, r0: number, r1: number, three: number];
  matmul_q6: [xout: number, xq: number, xs: number, wq: number, ws: number, n: number, r0: number, r1: number];
  add_columns: [y: number, rows: number, a: number, count: number, n: number];
  widen_bf16: [out: number, raw: number, n: number];
  widen_q8_0: [out: number, raw: number, n: number];
  widen_pq2_0: [out: number, raw: number, n: number];
  widen_ptq1_0: [out: number, raw: number, n: number];
  quantize_x: [xq: number, xs: number, x: number, n: number, bias: number];
  quantize6_x: [out: number, xs: number, x: number, n: number];
  ternary_x: [out: number, xs: number, x: number, n: number];
  argmax: [x: number, n: number];
  penalize: [logits: number, tokens: number, count: number, penalty: number, presence: number];
  sample: [logits: number, n: number, temperature: number, topp: number, random: number, probs: number, index: number, topk: number, minp: number];
  gate: [out: number, x: number, g: number, n: number];
  convolve: [out: number, taps: number, rows: number, n: number, count: number];
  short_conv: [out: number, taps: number, rows: number, b: number, n: number, count: number];
  delta_rule: [out: number, state: number, next: number, c: number, work: number, kh: number, kd: number, vd: number, vh: number, h0: number, h1: number];
  matmul_q8r: [xout: number, xq: number, xs: number, wq: number, ws: number, wc: number, n: number, r0: number, r1: number];
  matmul_q8r_tile: [xout: number, xq: number, xs: number, wq: number, ws: number, wc: number, n: number, r0: number, r1: number, count: number, os: number, frame: number];
  matmul_q6r: [xout: number, xq: number, xs: number, wq: number, ws: number, wc: number, n: number, r0: number, r1: number];
  matmul_t2r: [xout: number, xq: number, xs: number, wq: number, ws: number, n: number, r0: number, r1: number, three: number];
  matmul_t2r_tile: [xout: number, xq: number, xs: number, wq: number, ws: number, n: number, r0: number, r1: number, count: number, os: number, frame: number, three: number];
  sb_row: [out: number, w: number, tab: number, n: number, type: number];
  matmul_sbr_tile: [xout: number, xq: number, xs: number, w: number, tab: number, n: number, r0: number, r1: number, count: number, os: number, frame: number, type: number];
  signed_dots: [out: number, a: number, b: number];
  matmul_sbw_tile: [xout: number, xq: number, xs: number, w: number, tab: number, n: number, r0: number, r1: number, count: number, os: number, frame: number, type: number, scratch: number];
  matmul_sb: [xout: number, xq: number, xs: number, w: number, tab: number, n: number, r0: number, r1: number, type: number];
  matmul_sbr: [xout: number, xq: number, xs: number, w: number, tab: number, n: number, r0: number, r1: number, type: number];
  sums_x: [xs: number, xq: number, n: number];
  sb_tables: [area: number, codebooks: number];
};
export type Kernels = { [name in keyof typeof ADDRESSES]: (...args: Arguments[name]) => unknown };
// a job as it is run: an array, or its view of the control area's table
export type Job = Iterable<number>;
/** A kernel module's exports as they are, or on a 64-bit memory (wide) with the addresses made BigInt on the way
 * in: the rest of the code keeps its addresses in Numbers (exact up to 2^53). */
export function addressed(exports: WebAssembly.Exports, wide: boolean): Kernels {
  if (!wide) return exports as Kernels;
  const out = { ...exports } as Record<string, Kernel>;
  for (const [name, at] of Object.entries(ADDRESSES)) {
    const kernel = exports[name] as Kernel | undefined;
    if (kernel) {
      out[name] = (...args) => {
        for (const i of at) args[i] = BigInt(args[i]);
        return kernel(...args);
      };
    }
  }
  return out as Kernels;
}


// ---- T.375.4: the kinds of jobs, one line each. A kind's number, which a job begins with, is its place here (KIND).
// A line says:
//   from      whose kernels it calls: the plain module's, or the relaxed one's (a thread has none where the browser
//             lacks relaxed SIMD or the model does not run on int8)
//   one       the call for one token's rows r0 to r1, of a job's eight arguments and its rows
//   tile      (some) count tokens in one call: os and as, the bytes from a token's out and a to the next token's; a8,
//             the job's last argument (thread: which of a memory's threads calls, the coordinator 0, T.363.5)
//   rowBytes  (the kinds whose rows are a matrix's) the bytes of a row n wide, for the blocks count tokens go in
//             (a6: the job's sixth argument, where the kind's rows differ by it)
//   tileRows  (a tile's kinds) the rows its tile takes at once: a prompt's chunks are cut in as many (T.620; 1 where
//             it is not said)
//   warm      a tiny call of each of its kernels on the scratch (warmUp())
// runner() and warmUp() read every line once as a thread is made; a job then costs a look into an array by its
// number. A new kind is a line here, at the end (the numbers before it stay), its kernels' names in ADDRESSES, and
// whoever builds its jobs writing KIND.<its name> (AGENTS.md, "カーネルを登録簿から引くとき").
type Call = (out: number, a: number, b: number, a4: number, a5: number, a6: number, a7: number, a8: number, rows: number, r0: number, r1: number) => unknown;
type Tile = (out: number, a: number, b: number, a4: number, a5: number, a6: number, a7: number, r0: number, r1: number, count: number, os: number, as: number, a8: number) => unknown;
type Scratch = { xq: number; xs: number; w: number; s: number; c: number; out: number };
export type Work = {
  from: "plain" | "relaxed"; one(k: Kernels): Call; tile?(k: Kernels, thread: number): Tile; rowBytes?(n: number, a6: number): number;
  tileRows?: number; warm(k: Kernels, at: Scratch): void;
};
const kinds = <Name extends string>(table: Record<Name, Work>) => table;
// ---- T.363.5: llama.cpp's blocks (kernels/blocks.ts; kernels/README.md, "llama.cpp's blocks"): 256 weights of a row in
// a block of its type's bytes, the scales inside. A type is its number in a GGUF, which the kernels take as an argument:
// BLOCK_TYPES has each by its name, with a block's bytes.
// (T.628.3: Python's, convert/superblocks.py's SUPERBLOCKS, written by python_facts.py)
export const BLOCK_TYPES = PYTHON_BLOCKS;
/** the weights of a block */
export const BLOCK = 256;
/** the bytes of the tables the blocks' kernels read (sb_tables() writes them, of the checkpoint's codebooks) */
export const BLOCK_TABLES = 34992;
// a block's bytes by its type's number
const blockBytes: number[] = [];
for (const [type, bytes] of Object.values(BLOCK_TYPES)) blockBytes[type] = bytes;
const blockRow = (n: number, type: number) => (n / BLOCK) * blockBytes[type];
/** matmul_sbw_tile spreads a row once, as signed weights, into bytes of the calling thread's own: so many for a row n
 * wide, and room for this many threads (a thread past them takes matmul_sbr_tile, whose numbers are the same) */
export const spreadBytes = (n: number) => Math.ceil((n + n / 4 + n / 64) / 64) * 64;
export const SPREAD_THREADS = 64;
// (the blocks' kernels are not warmed: the tables they read are an engine's, made of its model's codebooks)
const unwarmed = () => {};
const blocksTile = (k: Kernels): Tile => (out, a, b, a4, a5, a6, a7, r0, r1, count, os, as) => k.matmul_sbr_tile(out, a, b, a4, a5, a7, r0, r1, count, os, as, a6);
export const WORK = kinds({
  // out, xq, xs, w, scales, corrections, n: int8 weights by 7-bit activations. T.159, its tile: four rows by four
  // tokens, each number matmul_q8r's to the bit
  matmul_q8r: {
    from: "relaxed", rowBytes: (n) => n, tileRows: 4,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_q8r(out, a, b, a4, a5, a6, a7, r0, r1),
    tile: (k) => (out, a, b, a4, a5, a6, a7, r0, r1, count, os, as) => k.matmul_q8r_tile(out, a, b, a4, a5, a6, a7, r0, r1, count, os, as),
    warm: (k, { xq, xs, w, s, c, out }) => {
      k.matmul_q8r(out, xq, xs, w, s, c, 32, 0, 1);
      k.matmul_q8r_tile(out, xq, xs, w, s, c, 32, 0, 4, 4, 0, 0);  // one tile: four rows and four tokens, all the same
    },
  },
  // out, xq, xs, w, scales, -, n: the same without relaxed SIMD
  matmul_q8: {
    from: "plain", rowBytes: (n) => n,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_q8(out, a, b, a4, a5, a7, r0, r1),
    warm: (k, { xq, xs, w, s, out }) => { k.matmul_q8(out, xq, xs, w, s, 32, 0, 1); },
  },
  // out, x, -, w, -, -, n
  matmul_f32: {
    from: "plain", rowBytes: (n) => 4 * n,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_f32(out, a, a4, a7, r0, r1),
    warm: (k, { w, out }) => { k.matmul_f32(out, out + 64, w, 8, 0, 1); },
  },
  // out, q, keys, values, scores, pos, kv heads, head size; its rows are the heads (T.109)
  attention: {
    from: "plain",
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.attention(out, a, b, a4, a5, a6, rows, a7, a8, r0, r1),
    warm: (k, { xq, w, c, out }) => { k.attention(out, xq, w, w, c, 0, 1, 1, 4, 0, 1); },  // one head of 4 at position 0
  },
  // the same over a cache of float16 (T.110)
  attention_f16: {
    from: "plain",
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.attention_f16(out, a, b, a4, a5, a6, rows, a7, a8, r0, r1),
    warm: (k, { xq, w, c, out }) => { k.attention_f16(out, xq, w, w, c, 0, 1, 1, 4, 0, 1); },
  },
  // as matmul_q8r, on int6 weights (T.98): 24 bytes for 32 weights (T.620: they were counted as int8's, a byte a weight)
  matmul_q6r: {
    from: "relaxed", rowBytes: (n) => (n * 3) / 4,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_q6r(out, a, b, a4, a5, a6, a7, r0, r1),
    warm: (k, { xq, xs, w, s, c, out }) => { k.matmul_q6r(out, xq, xs, w, s, c, 32, 0, 1); },
  },
  // as matmul_q8, on int6 weights
  matmul_q6: {
    from: "plain", rowBytes: (n) => (n * 3) / 4,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_q6(out, a, b, a4, a5, a7, r0, r1),
    warm: (k, { xq, xs, w, s, out }) => { k.matmul_q6(out, xq, xs, w, s, 32, 0, 1); },
  },
  // out, state, next, q k and v, work, key heads, key size, value size; its rows are the value heads (T.229)
  delta_rule: {
    from: "plain",
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.delta_rule(out, a, b, a4, a5, a6, a7, a8, rows, r0, r1),
    // one value head, a state of 4 by 4 (c: its work, 6 floats)
    warm: (k, { xq, w, s, c, out }) => { k.delta_rule(out, w, s, xq, c, 1, 4, 4, 1, 0, 1); },
  },
  // out, xq, xs, w, scales, -, n: ternary weights (T.231), two bits each, the activations interleaved. Its tile: a row
  // against four tokens, each number matmul_t2r's to the bit
  matmul_t2r: {
    from: "relaxed", rowBytes: (n) => n / 4,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_t2r(out, a, b, a4, a5, a7, r0, r1, THREE),
    tile: (k) => (out, a, b, a4, a5, a6, a7, r0, r1, count, os, as) => k.matmul_t2r_tile(out, a, b, a4, a5, a7, r0, r1, count, os, as, THREE),
    warm: (k, { xq, xs, w, s, out }) => {
      k.matmul_t2r(out, xq, xs, w, s, 128, 0, 1, THREE);
      k.matmul_t2r_tile(out, xq, xs, w, s, 128, 0, 1, 4, 0, 0, THREE);  // one row and four tokens, all the same
    },
  },
  // as matmul_t2r, without relaxed SIMD
  matmul_t2: {
    from: "plain", rowBytes: (n) => n / 4,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_t2(out, a, b, a4, a5, a7, r0, r1, THREE),
    // one group of 128 (the scratch's first 160 bytes are its activations and weights)
    warm: (k, { xq, xs, w, s, out }) => { k.matmul_t2(out, xq, xs, w, s, 128, 0, 1, THREE); },
  },
  // out, xq, xs, w, tables, type, n: llama.cpp's blocks (T.363.5) by signed 8-bit activations and their sums (sums_x).
  // Its tile: a row read once against four tokens, each number matmul_sbr's to the bit
  matmul_sbr: {
    from: "relaxed", rowBytes: blockRow,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_sbr(out, a, b, a4, a5, a7, r0, r1, a6),
    tile: blocksTile, warm: unwarmed,
  },
  // the same numbers without relaxed SIMD (no tile: a token at a time)
  matmul_sb: {
    from: "plain", rowBytes: blockRow,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_sb(out, a, b, a4, a5, a7, r0, r1, a6),
    warm: unwarmed,
  },
  // as matmul_sbr, for an engine whose relaxed dot product multiplies signed bytes by signed bytes exactly (arm64's:
  // the engine asks signed_dots as it is made): its tile spreads a row once into the thread's own bytes of the scratch
  // at the job's last argument, the same numbers again
  matmul_sbw: {
    from: "relaxed", rowBytes: blockRow,
    one: (k) => (out, a, b, a4, a5, a6, a7, a8, rows, r0, r1) => k.matmul_sbr(out, a, b, a4, a5, a7, r0, r1, a6),
    tile: (k, thread) => (thread < SPREAD_THREADS
      ? (out, a, b, a4, a5, a6, a7, r0, r1, count, os, as, a8) => k.matmul_sbw_tile(out, a, b, a4, a5, a7, r0, r1, count, os, as, a6, a8 + thread * spreadBytes(a7))
      : blocksTile(k)),
    warm: unwarmed,
  },
});
export type Kind = keyof typeof WORK;
/** the number of each kind of job: its place in WORK */
export const KIND = Object.fromEntries(Object.keys(WORK).map((name, number) => [name, number] as const)) as Record<Kind, number>;
/** T.620: the rows a kind's tile takes at once, by the kind's number (1: a row at a time, or no tile) */
export const TILE_ROWS = Object.values(WORK).map((work: Work) => work.tileRows ?? 1);

// ---- T.375.4: how the weights of a matrix are packed in a checkpoint, as the kernels take them: one line for each
// way a quantized tensor is stored (contracts/plan.ts's Stored; float32 is no packing: the engine multiplies by it as
// it is, a job of the kind matmul_f32). A line says:
//   rowBytes  the bytes of a row n wide in the checkpoint
//   scale     the bytes of the scale that follows a group's values there (a float32; llama.cpp's blocks 0: inside)
//   value     weight i of the tensor whose values start at byte at of U, as the int8 it is (what JavaScript reads one
//             at a time: an embedding's row, a tensor widened to float32); or
//   row       (llama.cpp's blocks, of which no weight is read alone) what writes a row n wide at w as float32 to out
//   jobs      the kind of the job that multiplies by it, with relaxed SIMD and without (signed: with a relaxed dot
//             product that multiplies signed bytes by signed bytes, where the packing has a kernel for that)
//   tabled    (llama.cpp's blocks) its kernels take the engine's tables and this number, the type's, where the others
//             take a matrix's scales and corrections; the engine has their modules for such a model alone
//   takes     the form its kernels take the activations in, by a name: matrices whose packings say the same name
//             read what one ready() made (the engine makes one for a name)
//   sums      the kernel that makes the corrections relaxed SIMD needs (-64 times the sum of every group: its 7-bit
//             activations have a bias of 64); null: its relaxed kernel takes none
//   ready     the activations as its kernels take them, chosen once for an engine: from n float32 at x to xq and xs
//   xs        the bytes a group of 32 activations takes in xs
//   group     (T.375.5) the weights one float32 scale is of, which follows its group's values in the checkpoint
//   gpu       (T.375.5) how the GPU holds weights so packed, null: not at all. as: the packing they are in there (int6 is
//             widened to int8, T.155; ternary as it is, T.232); alone: a model of them may be on the GPU alone (T.156; not
//             six bits, which are for a device short of memory: T.155's review); dot: its shaders need WGSL's packed
//             int8 dot product
// The engine reads a line as it makes a matrix (forward/engine.ts) and keeps what it chose with the matrix: a token
// compares no name. What sizes a model before it is loaded reads the same lines (forward/memory.ts's footprint(),
// forward/alone.ts). A new packing is a line here, the kinds of its jobs above and its kernels' names in ADDRESSES.
export type Ready = (xq: number, xs: number, x: number, n: number) => void;
export type Packing<Name extends string = string> = {
  rowBytes(n: number): number; scale: number; value?(U: Uint8Array, at: number, i: number): number;
  row?(k: Kernels, tables: number): (out: number, w: number, n: number) => void;
  jobs: { relaxed: number; plain: number; signed?: number }; tabled?: number; takes: string;
  sums: "int8_sums" | "six_sums" | null; ready(k: Kernels, relaxed: boolean): Ready; xs: number;
  group: number; gpu: { as: Name; alone: boolean; dot: boolean } | null;
};
const signed = (byte: number) => (byte << 24) >> 24;
// (int8 and int6: relaxed SIMD multiplies by 7-bit unsigned activations with a bias of 64, plain by signed 8-bit ones)
const biased = (k: Kernels, relaxed: boolean): Ready => {
  const bias = relaxed ? 64 : 0;
  return (xq, xs, x, n) => { k.quantize_x(xq, xs, x, n, bias); };
};
const packings = <Name extends string>(table: Record<Name, Packing<NoInfer<Name>>>) => table;
// T.363.5: llama.cpp's blocks, a line each, alike but for the type: the activations signed in all 8 bits (-127 to 127),
// and after the scales of their groups the int16 sums of each 16 and each 8 of them (sums_x: 16 bytes a group of 32 in
// all). No corrections (a group's sum is an exact integer before it meets a float), and none of it on the GPU but
// (T.547.2.3) Q4_K's, as it is (the GPU's worker keeps its blocks: gpu/packings.ts), by the packed int8 dot
const blocks = ([type, bytes]: readonly [number, number]): Packing<never> => ({
  rowBytes: (n) => (n / BLOCK) * bytes, scale: 0,
  row: (k, tables) => (out, w, n) => { k.sb_row(out, w, tables, n, type); },
  jobs: { relaxed: KIND.matmul_sbr, plain: KIND.matmul_sb, signed: KIND.matmul_sbw }, tabled: type, takes: "summed",
  sums: null, xs: 16,
  ready: (k) => (xq, xs, x, n) => {
    k.quantize_x(xq, xs, x, n, 0);
    k.sums_x(xs, xq, n);
  },
  group: BLOCK, gpu: null,
});
export const PACKED = packings({
  int8: {
    rowBytes: (n) => n, scale: 4, value: (U, at, i) => signed(U[at + i]), takes: "biased",
    jobs: { relaxed: KIND.matmul_q8r, plain: KIND.matmul_q8 }, sums: "int8_sums", ready: biased, xs: 4,
    group: 32, gpu: { as: "int8", alone: true, dot: false },
  },
  // T.98: six bits a weight, 32 of them in 24 bytes, the layout of llama2_numpy.pack6 (six bits, then two zero bits)
  int6: {
    rowBytes: (n) => n / 32 * 24, scale: 4, takes: "biased",
    value: (U, at, i) => {
      const group = at + ((i / 32) | 0) * 24, j = i % 32;
      const low = j < 16 ? U[group + j] & 15 : U[group + j - 16] >> 4;
      const top = (U[group + 16 + (j % 8)] >> (2 * ((j / 8) | 0))) & 3;
      return signed((low | (top << 4)) << 2);
    },
    jobs: { relaxed: KIND.matmul_q6r, plain: KIND.matmul_q6 }, sums: "six_sums", ready: biased, xs: 4,
    group: 32, gpu: { as: "int8", alone: false, dot: false },
  },
  // T.230: two bits a weight less one, the layout of llama2_numpy.pack_ternary. T.231: its kernels take the activations
  // signed, in all 8 bits, and laid out as the weights' planes are; after the scales of the activations' groups come
  // as many int32, minus each group's sum (interleave), so they take no corrections. T.533: its scales are float16s
  // (scale: 2), PQ2_0's own
  ternary: {
    rowBytes: (n) => n / 4, scale: 2, value: (U, at, i) => ((U[at + (i >> 2)] >> (2 * (i & 3))) & 3) - 1, takes: "interleaved",
    jobs: { relaxed: KIND.matmul_t2r, plain: KIND.matmul_t2 }, sums: null, xs: 8,
    ready: (k) => (xq, xs, x, n) => {
      k.quantize_x(xq, xs, x, n, 0);
      k.interleave(xq, xs, n);
    },
    group: 128, gpu: { as: "ternary", alone: true, dot: true },
  },
  Q2_K: { ...blocks(BLOCK_TYPES.Q2_K), gpu: { as: "Q2_K", alone: false, dot: true } }, Q4_K: { ...blocks(BLOCK_TYPES.Q4_K), gpu: { as: "Q4_K", alone: false, dot: true } }, IQ2_XXS: { ...blocks(BLOCK_TYPES.IQ2_XXS), gpu: { as: "IQ2_XXS", alone: false, dot: true } }, IQ2_XS: { ...blocks(BLOCK_TYPES.IQ2_XS), gpu: { as: "IQ2_XS", alone: false, dot: true } },
  IQ3_XXS: { ...blocks(BLOCK_TYPES.IQ3_XXS), gpu: { as: "IQ3_XXS", alone: false, dot: true } }, IQ1_S: { ...blocks(BLOCK_TYPES.IQ1_S), gpu: { as: "IQ1_S", alone: false, dot: true } }, IQ3_S: { ...blocks(BLOCK_TYPES.IQ3_S), gpu: { as: "IQ3_S", alone: false, dot: true } }, IQ2_S: { ...blocks(BLOCK_TYPES.IQ2_S), gpu: { as: "IQ2_S", alone: false, dot: true } },
  IQ4_XS: { ...blocks(BLOCK_TYPES.IQ4_XS), gpu: { as: "IQ4_XS", alone: false, dot: true } }, IQ1_M: { ...blocks(BLOCK_TYPES.IQ1_M), gpu: { as: "IQ1_M", alone: false, dot: true } },
});
export type Packed = keyof typeof PACKED;
/** the packing of a tensor stored as stored, or null: float32 and float16 are not packed */
export const packingOf = (stored: string | undefined): Packing<Packed> | null => (stored !== undefined && Object.hasOwn(PACKED, stored) ? PACKED[stored as Packed] : null);
/** (T.375.5) the bytes a weight takes in a checkpoint so packed: a group's values and its scale, over the
 * group's weights (int8 1.125, int6 0.875, ternary 0.265625; T.363.5: a block's bytes over its 256) */
export const weightBytes = ({ rowBytes, scale, group }: Packing) => (rowBytes(group) + scale) / group;

/** runRows(job, r0, r1) on these kernels: k, the plain module's exports; r, the relaxed module's or null. job is
 * an array or a view of the control area. Every kind of WORK is read here once: a job is run by its number.
 * thread (T.363.5): which of the threads of a memory this one is (the coordinator 0, a helper its share), for a kind
 * whose tile keeps bytes of the calling thread's own. */
export function runner(k: Kernels, r: Kernels | null, thread = 0) {
  const works = Object.entries(WORK) as [Kind, Work][];
  const none = (name: Kind): Call => () => { throw new Error(`A job of the kind ${name} takes the relaxed kernels, and this thread has none.`); };
  const ones = works.map(([name, { from, one }]) => (from === "plain" ? one(k) : r ? one(r) : none(name)));
  const tiles = works.map(([, { from, tile }]) => {
    const kernels = from === "plain" ? k : r;
    return tile && kernels ? tile(kernels, thread) : null;
  });
  const bytes = works.map(([, { rowBytes }]) => rowBytes ?? null);
  return (job: Job, r0: number, r1: number) => {
    const [kind, out, a, b, a4, a5, a6, a7, a8, rows, count, os, as, bs] = job;
    const one = ones[kind];
    if (count === 1) return one(out, a, b, a4, a5, a6, a7, a8, rows, r0, r1);
    const tile = tiles[kind];
    if (tile) return tile(out, a, b, a4, a5, a6, a7, r0, r1, count, os, as, a8);
    const step = Math.max(1, Math.floor(BLOCK_BYTES / bytes[kind]!(a7, a6)));
    for (let r = r0; r < r1; r += step) {
      const end = Math.min(r + step, r1);
      for (let t = 0; t < count; t++) one(out + t * os, a + t * as, b + t * bs, a4, a5, a6, a7, a8, rows, r, end);
    }
  };
}

/** the kernels warmed up: a new thread runs them unoptimized at first, and a measurement would take that for the
 * speed of the count (T.93). Tiny calls on the scratch, many times; what they compute is thrown away. */
export function warmUp(k: Kernels, r: Kernels | null) {
  const at = { xq: SCRATCH, xs: SCRATCH + 64, w: SCRATCH + 128, s: SCRATCH + 192, c: SCRATCH + 256, out: SCRATCH + 320 };
  const warms = (Object.values(WORK) as Work[]).flatMap(({ from, warm }) => {
    const kernels = from === "plain" ? k : r;
    return kernels ? [() => warm(kernels, at)] : [];
  });
  for (let i = 0; i < 4000; i++) for (const warm of warms) warm();
}
