// T363.1: the matrix product on the weights of Underdog Saluki 27B as its GGUF holds them (llama.cpp's blocks of 256
// weights), in the forms of tests/saluki-forms.ts, beside the kernels of kernels/ on the same shapes (t2r: matmul_t2r
// on ternary weights, which is how Ternary Bonsai 2 27B runs; q8r: matmul_q8r on int8) and a loop that only reads
// (kernels/ceilings.ts's read: the memory's ceiling). One process, the forms taking turns in an order that changes
// every turn, on 1 and 4 threads of one shared memory, the rows in chunks taken in turn as forward/threads.ts's
// phase() cuts them. docs/notes/t363.1-saluki-kernel-forms-2026-10-13.md has what each form is, the table and the
// choice. Nothing here is of the page.
//
// The matrices are read from memory, as a model's are (each read once a token): copies of each past every cache
// (--megabytes a kind of weights, default 512), a call on each in turn (T159's review: one matrix timed again and
// again stays in the caches). The blocks are random bytes with finite scales: every byte string is a block.
//
// Before anything is timed, every form is held to the converter's own reading of the blocks (T362:
// tests/saluki_blocks.py runs convert/superblocks.py on the first rows; the sums are float64, the activations
// rounded as the form takes them), the forms that are one arithmetic to one another to the bit on every row, and each
// to itself on 1 and 4 threads to the bit. A form that differs stops the run.
//
//   node tests/saluki-kernel-bench.mjs [--rounds 2] [--turns 5] [--megabytes 512] [--shapes 17408x5120,5120x17408]
//        [--forms rb,ra,t2r] [--no-prompt]
//
// Then a prompt of 16 tokens through the same matrices: IQ2_XXS's tile (a group's weights in registers for four
// tokens), its rows widened to int8 for int8's tile, and the token's kernel for every token; beside the ternary and
// the int8 tiles. And at the end what the speeds make of a token of the 27B (an estimate: the types that have no
// form here are counted at the speed of the nearest that has).
//
// G weights/s is rows x n a second; GB/s counts what a row reads of the weights. The shapes are the 27B's (5120 wide,
// its FFN 17408). Needs `npm ci` (AssemblyScript; compiles into .tmp/saluki-kernel-bench/) and a python3 with NumPy
// (PYTHON=<path> for another).
//
//   CI: node tests/ci.mjs run tests.yml only_extra=true extra="pip install numpy && npm ci && node tests/saluki-kernel-bench.mjs" --ref <branch>
//       (and runner=ubuntu-24.04-arm)
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Worker, isMainThread, workerData } from "node:worker_threads";
import { leave } from "./leave.mjs";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..") + "/";
const work = root + ".tmp/saluki-kernel-bench/";
// the numbers every thread reads
const FORM = 0, FIRST = 1, STRIDE = 2, COPIES = 3, START = 4, N = 5, ROWS = 6, THREADS = 7, OUT = 8, XQ = 9, XS = 10, CALLS = 11,
  COUNT = 12, FRAME = 13, OUT_FRAME = 14, TAB = 15, SCRATCH = 16, AREA = 17;
const GO = 0, DONE = 1, NEXT = 2;
const SCRATCH_BYTES = 131072;  // a thread's own: four rows widened to int8 with their scales and corrections at most

// The bytes 256 weights take in each kind of weights (values, scales and corrections)
const BYTES = { IQ2_XXS: 66, IQ3_S: 110, Q2_K: 84, IQ1_M: 56, IQ2_XS: 74, IQ2_S: 82, IQ3_XXS: 98, IQ1_S: 50, IQ4_XS: 136, Q4_K: 144, ternary: 72, int8: 320 };
const KINDS = Object.keys(BYTES), BLOCKS = KINDS.filter((kind) => kind !== "ternary" && kind !== "int8");
// T363.4: the kernels of the page (kernels/blocks.ts, blocks_relaxed.ts), by the type's number in a GGUF: "p.<TYPE>" is
// matmul_sbr on rows of the type, "pp.<TYPE>" matmul_sb (no relaxed SIMD). Their tables are sb_tables's, at c[AREA]
const TYPE = { Q2_K: 10, Q4_K: 12, IQ2_XXS: 16, IQ2_XS: 17, IQ3_XXS: 18, IQ1_S: 19, IQ3_S: 21, IQ2_S: 22, IQ4_XS: 23, IQ1_M: 29 };
const product = (kernel, module, of) => ({ module, of, activations: "eight",
  run: (k, c, w, r0, r1) => k[kernel](c[OUT], c[XQ], c[XS], w, c[AREA], c[N], r0, r1, TYPE[of]) });
const g128 = (c) => (c[ROWS] * c[N]) / 128;
// a form of tests/saluki-forms.ts: (xout, xq, xs, w, tab, n, r0, r1, ...more)
const block = (kernel, of, activations, more = () => []) => ({ module: "forms", of, activations,
  run: (k, c, w, r0, r1, scratch) => k[kernel](c[OUT], c[XQ], c[XS], w, c[TAB], c[N], r0, r1, ...more(scratch)) });
const FORMS = {
  rb: block("matmul_iq2xxs", "IQ2_XXS", "eight"),
  rbp: block("matmul_iq2xxs_plain", "IQ2_XXS", "eight"),
  rb2: block("matmul_iq2xxs_once", "IQ2_XXS", "eight"),
  rb3: block("matmul_iq2xxs_lanes", "IQ2_XXS", "eight"),
  any: block("matmul_any", "IQ2_XXS", "eight", () => [0]),
  ra: block("matmul_iq2xxs_ra", "IQ2_XXS", "seven"),
  sb: block("matmul_iq2xxs_sb", "IQ2_XXS", "eight", (scratch) => [scratch]),
  sa: block("matmul_iq2xxs_sa", "IQ2_XXS", "seven", (scratch) => [scratch]),
  iq3s: block("matmul_iq3s", "IQ3_S", "eight"),
  q2k: block("matmul_q2k", "Q2_K", "eight"),
  iq1m: block("matmul_iq1m", "IQ1_M", "eight"),
  t2r: { module: "tree", of: "ternary", activations: "interleaved",
    run: (k, c, w, r0, r1) => k.matmul_t2r(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, c[N], r0, r1, 3) },
  q8r: { module: "tree", of: "int8", activations: "seven",
    run: (k, c, w, r0, r1) => k.matmul_q8r(c[OUT], c[XQ], c[XS], w, w + g128(c) * 128, w + g128(c) * 144, c[N], r0, r1) },
  ...Object.fromEntries(BLOCKS.map((of) => [`p.${of}`, product("matmul_sbr", "blocksr", of)])),
  "pp.IQ2_XXS": product("matmul_sb", "blocks", "IQ2_XXS"),
  "pp.IQ3_S": product("matmul_sb", "blocks", "IQ3_S"),
  // the rows' bytes of IQ2_XXS read and nothing else (in 64s: a row's last bytes may be the next call's)
  read: { module: "ceilings", of: "IQ2_XXS", activations: "eight", run: (k, c, w, r0, r1) => {
    const row = (c[N] / 256) * 66, from = (w + r0 * row) & ~63;
    k.read(from, ((w + r1 * row) & ~63) - from, 1);
  } },
};
// the forms that are one arithmetic: the same numbers to the bit
const SAME = [["rb", "rbp"], ["rb", "any"], ["rb", "rb2"], ["rb", "rb3"], ["rb", "p.IQ2_XXS"], ["rb", "pp.IQ2_XXS"], ["iq3s", "p.IQ3_S"],
  ["p.IQ3_S", "pp.IQ3_S"], ["q2k", "p.Q2_K"], ["iq1m", "p.IQ1_M"]];
// A prompt's tokens (T108) through the same matrix. A kernel without a tile takes the rows in blocks that stay in
// the first cache, every token on each (jobs.ts's way)
const blockOf = (n, bytes) => Math.max(1, Math.floor(16384 / ((n / 256) * bytes)));
const byToken = (form) => (k, c, w, r0, r1, scratch) => {
  const saved = [c[OUT], c[XQ], c[XS]];
  for (let r = r0, step = blockOf(c[N], BYTES[FORMS[form].of]); r < r1; r += step) {
    for (let t = 0; t < c[COUNT]; t++) {
      c[OUT] = saved[0] + t * c[OUT_FRAME]; c[XQ] = saved[1] + t * c[FRAME]; c[XS] = saved[2] + t * c[FRAME];
      FORMS[form].run(k, c, w, r, Math.min(r + step, r1), scratch);
    }
  }
  [c[OUT], c[XQ], c[XS]] = saved;
};
const PROMPTS = {
  "rb, token by token": { module: "forms", of: "IQ2_XXS", activations: "eight", run: byToken("rb") },
  "rb tile": { module: "forms", of: "IQ2_XXS", activations: "eight",
    run: (k, c, w, r0, r1) => k.matmul_iq2xxs_tile(c[OUT], c[XQ], c[XS], w, c[TAB], c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME]) },
  // (rw and sw: only where the engine's relaxed dot product is of signed bytes by signed bytes, arm64)
  "rw tile": { module: "forms", of: "IQ2_XXS", activations: "eight", signed: true,
    run: (k, c, w, r0, r1) => k.matmul_iq2xxs_tile_signed(c[OUT], c[XQ], c[XS], w, c[TAB], c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME]) },
  "sb tile": { module: "forms", of: "IQ2_XXS", activations: "eight",
    run: (k, c, w, r0, r1, scratch) => k.matmul_iq2xxs_sb_tile(c[OUT], c[XQ], c[XS], w, c[TAB], c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME], scratch) },
  "sw tile": { module: "forms", of: "IQ2_XXS", activations: "eight", signed: true,
    run: (k, c, w, r0, r1, scratch) => k.matmul_iq2xxs_sw_tile(c[OUT], c[XQ], c[XS], w, c[TAB], c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME], scratch) },
  // T363.4: the page's tiles: matmul_sbr_tile, and matmul_sbw_tile where the engine multiplies signed by signed
  ...Object.fromEntries(["IQ2_XXS", "IQ3_S", "IQ2_S", "IQ1_M", "Q2_K"].flatMap((of) => [
    [`p.${of}, token by token`, { module: "blocksr", of, activations: "eight", run: byToken(`p.${of}`), base: true }],
    [`p.${of} tile`, { module: "blocksr", of, activations: "eight", by: `p.${of}, token by token`,
      run: (k, c, w, r0, r1) => k.matmul_sbr_tile(c[OUT], c[XQ], c[XS], w, c[AREA], c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME], TYPE[of]) }],
    ...(of === "Q2_K" ? [] : [[`pw.${of} tile`, { module: "blocksr", of, activations: "eight", signed: true, by: `p.${of}, token by token`,
      run: (k, c, w, r0, r1, scratch) => k.matmul_sbw_tile(c[OUT], c[XQ], c[XS], w, c[AREA], c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME], TYPE[of], scratch) }]]),
  ])),
  "sa, token by token": { module: "forms", of: "IQ2_XXS", activations: "seven", run: byToken("sa"), untimed: true },
  "sa tile": { module: "forms", of: "IQ2_XXS", activations: "seven",
    run: (k, c, w, r0, r1, scratch) => k.matmul_iq2xxs_sa_tile(c[OUT], c[XQ], c[XS], w, c[TAB], c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME], scratch) },
  "t2r tile": { module: "tree", of: "ternary", activations: "interleaved",
    run: (k, c, w, r0, r1) => k.matmul_t2r_tile(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME], 3) },
  "q8r tile": { module: "tree", of: "int8", activations: "seven",
    run: (k, c, w, r0, r1) => k.matmul_q8r_tile(c[OUT], c[XQ], c[XS], w, w + g128(c) * 128, w + g128(c) * 144, c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME]) },
};
const NAMES = Object.keys(FORMS), ALL = { ...FORMS, ...PROMPTS }, EVERY = Object.keys(ALL);
const modules = { forms: "forms.wasm", tree: "relaxed.wasm", plain: "plain.wasm", ceilings: "ceilings.wasm", blocks: "blocks.wasm", blocksr: "blocksr.wasm" };
const instances = (memory) => Object.fromEntries(Object.entries(modules).map(([name, file]) =>
  [name, new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(work + file)), { env: { memory } }).exports]));
// forward/threads.ts's phase(): chunks of a quarter of a thread's share (T93), taken in turn; one thread the whole matrix
// (a prompt's chunks in fours of rows, as phase() cuts them for the tiles)
const chunkOf = (rows, threads, quad = 1) => (threads === 1 ? rows : quad * Math.ceil(rows / (threads * 4 * quad)));
function runCalls(kernels, sync, c, index) {
  const form = ALL[EVERY[c[FORM]]], k = kernels[form.module], scratch = c[SCRATCH] + index * SCRATCH_BYTES;
  const threads = c[THREADS], rows = c[ROWS], calls = c[CALLS];
  const size = chunkOf(rows, threads, c[COUNT] > 1 ? 4 : 1), chunks = Math.ceil(rows / size);
  const weights = (call) => c[FIRST] + ((c[START] + call) % c[COPIES]) * c[STRIDE];
  if (threads === 1) {
    if (index === 0) for (let call = 0; call < calls; call++) form.run(k, c, weights(call), 0, rows, scratch);
    return;
  }
  // (a thread's own copy of the numbers: a prompt's token by token moves three of them)
  const own = Float64Array.from(c);
  for (let at = Atomics.add(sync, NEXT, 1); at < calls * chunks; at = Atomics.add(sync, NEXT, 1)) {
    const r0 = (at % chunks) * size;
    form.run(k, own, weights(Math.floor(at / chunks)), r0, Math.min(r0 + size, rows), scratch);
  }
}

if (!isMainThread) {
  const { memory, control, numbers, index } = workerData;
  const sync = new Int32Array(control), c = new Float64Array(numbers);
  const kernels = instances(memory);
  let seen = 0;
  for (;;) {
    Atomics.wait(sync, GO, seen);
    seen = Atomics.load(sync, GO);
    if (seen < 0) break;
    if (index < c[THREADS]) runCalls(kernels, sync, c, index);
    Atomics.add(sync, DONE, 1);
    Atomics.notify(sync, DONE);
  }
  process.exit(0);
}

const args = process.argv.slice(2);
const text = (name, value) => (args.includes(name) ? args[args.indexOf(name) + 1] : value);
const rounds = Number(text("--rounds", 2)), turns = Number(text("--turns", 5));
const shapes = text("--shapes", "17408x5120,5120x17408").split(",").map((shape) => shape.split("x").map(Number));
const timed = text("--forms", NAMES.join(",")).split(",");
for (const form of timed) if (!FORMS[form]) throw new Error(`no form called ${form}: ${NAMES.join(", ")}`);
fs.mkdirSync(work, { recursive: true });
const asc = ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1",
  "--sharedMemory", "--maximumMemory", "65536"];
for (const [source, out, features] of [["tests/saluki-forms.ts", "forms.wasm", "simd,relaxed-simd,threads"],
  ["kernels/kernel_relaxed.ts", "relaxed.wasm", "simd,relaxed-simd,threads"], ["kernels/kernel.ts", "plain.wasm", "simd,threads"],
  ["kernels/ceilings.ts", "ceilings.wasm", "simd,threads"], ["kernels/blocks.ts", "blocks.wasm", "simd,threads"],
  ["kernels/blocks_relaxed.ts", "blocksr.wasm", "simd,relaxed-simd,threads"]]) {
  execFileSync("npx", [...asc, root + source, "-o", work + out, "--enable", features], { cwd: root, stdio: "inherit" });
}
const cpu = () => {
  const model = os.cpus()[0]?.model;
  if (model && model !== "unknown") return model;
  const part = (fs.existsSync("/proc/cpuinfo") ? fs.readFileSync("/proc/cpuinfo", "utf8") : "").match(/CPU part\s*:\s*(0x[0-9a-f]+)/)?.[1];
  return part ? `arm64 ${{ "0xd0c": "Neoverse-N1", "0xd49": "Neoverse-N2", "0xd40": "Neoverse-V1", "0xd4f": "Neoverse-V2" }[part] ?? part}` : "unknown";
};
const PAGE = 65536, align = (bytes, to = 4096) => Math.ceil(bytes / to) * to;
const largest = Math.max(...shapes.map(([rows, n]) => rows * n));
const bytesOf = (kind, rows, n) => align(((rows * n) / 256) * BYTES[kind] + 64);
const prompts = !args.includes("--no-prompt");
const used = KINDS.filter((kind) => timed.some((form) => FORMS[form].of === kind) || (prompts && Object.values(PROMPTS).some((prompt) => prompt.of === kind)));
const small = 24 * 2 ** 20;  // the tables, the scratch areas, the activations, the outputs
// (as many megabytes of copies a kind as a 32-bit memory holds of all the kinds that are timed, 512 at most)
const megabytes = Number(text("--megabytes", Math.min(512, Math.floor(3600 / used.length / 32) * 32)));
console.log(`saluki-kernel-bench: ${cpu()}, ${os.cpus().length} logical cores, Node ${process.version}, matrices read from ${megabytes} MB of copies a kind of weights (${used.length} kinds)`);
const region = (kind) => Math.max(megabytes * 2 ** 20, 2 * bytesOf(kind, 1, largest));
const total = small + used.reduce((sum, kind) => sum + align(region(kind), PAGE), 0);
if (total > 65535 * PAGE) throw new Error(`${(total / 2 ** 30).toFixed(1)} GiB of copies do not fit a 32-bit memory: fewer --megabytes`);
const memory = new WebAssembly.Memory({ initial: Math.ceil(total / PAGE), maximum: 65536, shared: true });
const control = new SharedArrayBuffer(16), sync = new Int32Array(control);
const numbers = new SharedArrayBuffer(32 * 8), c = new Float64Array(numbers);
const kernels = instances(memory);
const signedDot = kernels.forms.signedDot() === 1;
console.log(`saluki-kernel-bench: this engine's relaxed dot product ${signedDot ? "multiplies signed bytes by signed bytes: the rw and sw tiles are timed" : "takes its second operand as unsigned: no rw or sw tile"}`);
const workers = Array.from({ length: 4 }, (_, index) => new Worker(new URL(import.meta.url), { workerData: { memory, control, numbers, index } }));
let gen = 0;
function go() {
  Atomics.store(sync, DONE, 0);
  Atomics.store(sync, NEXT, 0);
  Atomics.store(sync, GO, ++gen);
  Atomics.notify(sync, GO);
  for (let done = Atomics.load(sync, DONE); done < workers.length; done = Atomics.load(sync, DONE)) Atomics.wait(sync, DONE, done, 1000);
}
const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
const U = new Uint8Array(memory.buffer), I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer), N32 = new Int32Array(memory.buffer),
  N16 = new Int16Array(memory.buffer), W32 = new Uint32Array(memory.buffer);
const firsts = {};
used.reduce((at, kind) => { firsts[kind] = at; return at + align(region(kind), PAGE); }, small);
let seed = 7;
const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
// the order of a turn: another every turn
const shuffled = (list) => {
  const out = list.slice();
  for (let i = out.length - 1; i > 0; i--) { const j = next() % (i + 1); [out[i], out[j]] = [out[j], out[i]]; }
  return out;
};
const python = process.env.PYTHON ?? "python3";
const CHECKED = 24;  // the rows held to the converter's reading

// ---- the tables (tests/saluki-forms.ts has the layout)
const TABLES = 21504 + 160;
function tables(at) {
  for (const kind of BLOCKS) fs.rmSync(`${work}${kind}.blocks`, { force: true });
  execFileSync(python, [root + "tests/saluki_blocks.py", work], { stdio: "inherit" });
  const grids = new Int8Array(fs.readFileSync(work + "grids.bin"));
  if (grids.length !== 2048 + 2048 + 16384) throw new Error("grids.bin is not the three grids");
  U.set(grids.subarray(0, 2048), at);
  for (let i = 0; i < 128; i++) {
    let bits = i, ones = 0;
    for (let b = 0; b < 7; b++) ones += (i >> b) & 1;
    bits |= (ones & 1) << 7;
    for (let j = 0; j < 8; j++) U[at + 2048 + 8 * i + j] = (bits >> j) & 1 ? 255 : 0;
  }
  U.set(grids.subarray(2048, 4096), at + 3072);
  for (let i = 0; i < 16384; i++) U[at + 5120 + i] = grids[4096 + i] + 1;
  const v = at + 21504;
  for (let j = 0; j < 16; j++) {
    U[v + j] = 1 << (j & 7);
    U[v + 16 + j] = j >> 3;
    U[v + 32 + j] = 2 + (j >> 3);
    U[v + 48 + j] = 3;
    U[v + 64 + j] = 15;
    U[v + 80 + j] = j >> 2;
  }
  for (let j = 0; j < 8; j++) { N16[(v + 96) / 2 + j] = j & 1 ? 0x80 : 0x08; N16[(v + 112) / 2 + j] = -7; }
  N32.set([512, 64, 8, 1, 7, 7, 7, 7], (v + 128) / 4);
}

const table = [], promptTable = [];
for (const [rows, n] of shapes) {
  if (n % 256) throw new Error(`rows of ${n} are not whole blocks of 256`);
  const groups = (rows * n) / 128, ng = n / 32, blocks = (rows * n) / 256;
  let top = PAGE;
  const take = (bytes) => { const at = top; top = align(at + bytes, 64); return at; };
  const tab = take(TABLES), scratch = take(4 * SCRATCH_BYTES), area = take(34992), booksAt = take(33808);
  tables(tab);
  U.set(fs.readFileSync(work + "codebooks.bin"), booksAt);
  if (kernels.blocks.sb_tables(area, booksAt) !== 34992) throw new Error("sb_tables wrote another number of bytes than 34992");
  // ---- each kind's first copy. A block of the four types is random bytes with a finite float16 d (and dmin)
  const half = () => 0x2800 + (next() & 0x3ff);
  const d = new Float32Array(groups), tw = used.includes("ternary") || used.includes("int8") ? new Int8Array(rows * n) : null;
  if (tw) {
    for (let i = 0; i < tw.length; i++) tw[i] = (next() % 3) - 1;
    for (let g = 0; g < groups; g++) d[g] = Math.fround(0.005 + (next() % 4096) / 163840);
  }
  const random = (at, bytes) => {
    let x = (seed | 1) >>> 0;
    for (let i = Math.floor(at / 4), end = Math.ceil((at + bytes) / 4); i < end; i++) { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; W32[i] = x; }
    seed = x;
  };
  const put16 = (at, value) => { U[at] = value & 255; U[at + 1] = value >> 8; };
  const writers = {
    IQ2_XXS(at) { random(at, blocks * 66); for (let b = 0; b < blocks; b++) put16(at + b * 66, half()); },
    IQ3_S(at) { random(at, blocks * 110); for (let b = 0; b < blocks; b++) put16(at + b * 110, half()); },
    Q2_K(at) { random(at, blocks * 84); for (let b = 0; b < blocks; b++) { put16(at + b * 84 + 80, half()); put16(at + b * 84 + 82, half()); } },
    IQ2_XS(at) { random(at, blocks * 74); for (let b = 0; b < blocks; b++) put16(at + b * 74, half()); },
    IQ2_S(at) { random(at, blocks * 82); for (let b = 0; b < blocks; b++) put16(at + b * 82, half()); },
    IQ3_XXS(at) { random(at, blocks * 98); for (let b = 0; b < blocks; b++) put16(at + b * 98, half()); },
    IQ1_S(at) { random(at, blocks * 50); for (let b = 0; b < blocks; b++) put16(at + b * 50, half()); },
    IQ4_XS(at) { random(at, blocks * 136); for (let b = 0; b < blocks; b++) put16(at + b * 136, half()); },
    Q4_K(at) { random(at, blocks * 144); for (let b = 0; b < blocks; b++) { put16(at + b * 144, half()); put16(at + b * 144 + 2, half()); } },
    IQ1_M(at) {  // its d is the top four bits of its last four words
      random(at, blocks * 56);
      for (let b = 0; b < blocks; b++) {
        const value = half();
        for (let k = 0; k < 4; k++) { const byte = at + b * 56 + 49 + 2 * k; U[byte] = (U[byte] & 15) | (((value >> (4 * k)) & 15) << 4); }
      }
    },
    ternary(at) {
      const scales = at + groups * 32;
      U.fill(0, at, scales);
      for (let j = 0; j < tw.length; j++) U[at + (j >> 2)] |= (tw[j] + 1) << (2 * (j & 3));
      F.set(d, scales / 4);
    },
    int8(at) {  // -127, 0, 127 and float32(d / 127) a group of 32
      const scales = at + groups * 128, corrections = scales + groups * 16;
      for (let j = 0; j < tw.length; j++) I[at + j] = 127 * tw[j];
      for (let g = 0; g < groups * 4; g++) {
        F[scales / 4 + g] = Math.fround(d[g >> 2] / 127);
        let sum = 0;
        for (let j = g * 32; j < g * 32 + 32; j++) sum += 127 * tw[j];
        N32[corrections / 4 + g] = -64 * sum;
      }
    },
  };
  const copies = {};
  for (const kind of used) {
    const stride = bytesOf(kind, rows, n), first = firsts[kind];
    writers[kind](first);
    copies[kind] = megabytes ? Math.max(2, Math.floor(region(kind) / stride)) : 1;
    for (let copy = 1; copy < copies[kind]; copy++) U.copyWithin(first + copy * stride, first, first + stride);
  }
  // ---- the converter's reading of the first rows of the four types
  const checked = Math.min(CHECKED, rows), read = {};
  for (const kind of BLOCKS.filter((name) => used.includes(name))) {
    fs.writeFileSync(`${work}${kind}.blocks`, U.subarray(firsts[kind], firsts[kind] + checked * (n / 256) * BYTES[kind]));
  }
  execFileSync(python, [root + "tests/saluki_blocks.py", work], { stdio: "inherit" });
  for (const kind of BLOCKS.filter((name) => used.includes(name))) {
    const bytes = fs.readFileSync(`${work}${kind}.f32`);
    read[kind] = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    if (read[kind].length !== checked * n) throw new Error(`${kind}.f32 is not ${checked} rows of ${n}`);
  }
  // ---- the activations: one vector, quantized as each kind of kernel takes it, in groups of 32
  const x = new Float32Array(n);
  for (let j = 0; j < n; j++) x[j] = ((next() % 2001) - 1000) / 250 * (1 + (next() % 7 === 0 ? 3 : 0));
  const quantized = (most, bias) => {
    const q = new Int32Array(n), scales = new Float32Array(ng);
    for (let g = 0; g < ng; g++) {
      let high = 0;
      for (let j = g * 32; j < g * 32 + 32; j++) high = Math.max(high, Math.abs(x[j]));
      scales[g] = Math.fround(high / most);
      for (let j = g * 32; j < g * 32 + 32; j++) q[j] = Math.round(x[j] / scales[g]) + bias;
    }
    return { q, scales, bias };
  };
  const eight = quantized(127, 0), seven = quantized(63, 64);
  // a token's frame: its n activations, their n / 32 scales, and what its kind of kernel has after them
  const frame = align(n + n / 2, 64);
  const turned = ({ q, scales }, t) => ({ q: Int32Array.from({ length: n }, (_, j) => q[(j + 32 * t) % n]), scales: Float32Array.from({ length: ng }, (_, g) => scales[(g + t) % ng]) });
  const frames = {
    seven(at, t) { const { q, scales } = turned(seven, t); I.set(q, at); F.set(scales, (at + n) / 4); },
    // the sums of each 16 activations, then of each 8, as int16 (what a kernel's "ready" would leave, once a token)
    eight(at, t) {
      const { q, scales } = turned(eight, t);
      I.set(q, at);
      F.set(scales, (at + n) / 4);
      const sums16 = (at + n + ng * 4) / 2, sums8 = sums16 + n / 16;
      for (let g = 0; g < n / 8; g++) N16[sums8 + g] = q.subarray(g * 8, g * 8 + 8).reduce((a, b) => a + b, 0);
      for (let g = 0; g < n / 16; g++) N16[sums16 + g] = N16[sums8 + 2 * g] + N16[sums8 + 2 * g + 1];
    },
    interleaved(at, t) { const { q, scales } = turned(eight, t); I.set(q, at); F.set(scales, (at + n) / 4); kernels.plain.interleave(at, at + n, n); },
  };
  const places = {};
  for (const kind of Object.keys(frames)) { const at = take(frame); frames[kind](at, 0); places[kind] = { xq: at, xs: at + n }; }
  const outs = [take(rows * 4), take(rows * 4), take(rows * 4), take(rows * 4)];
  // ---- the float64 sums of the first rows: the weights as the converter reads them, the activations as rounded
  const weightOf = {
    ternary: (i, j) => tw[i * n + j] * d[(i * n + j) >> 7],
    int8: (i, j) => 127 * tw[i * n + j] * Math.fround(d[(i * n + j) >> 7] / 127),
  };
  for (const kind of Object.keys(read)) weightOf[kind] = (i, j) => read[kind][i * n + j];
  const reference = (form) => {
    const { q, scales, bias } = FORMS[form].activations === "seven" ? seven : eight, weight = weightOf[FORMS[form].of], out = new Float64Array(checked);
    for (let i = 0; i < checked; i++) {
      for (let g = 0; g < ng; g++) {
        let dot = 0;
        for (let j = g * 32; j < g * 32 + 32; j++) dot += weight(i, j) * (q[j] - bias);
        out[i] += dot * scales[g];
      }
    }
    return out;
  };
  Object.assign(c, { [N]: n, [ROWS]: rows, [TAB]: tab, [SCRATCH]: scratch, [AREA]: area });
  const set = (form) => Object.assign(c, { [FORM]: EVERY.indexOf(form), [FIRST]: firsts[FORMS[form].of], [STRIDE]: bytesOf(FORMS[form].of, rows, n),
    [COPIES]: copies[FORMS[form].of], [XQ]: places[FORMS[form].activations].xq, [XS]: places[FORMS[form].activations].xs, [COUNT]: 1 });
  const once = (form, threads, out) => {
    set(form);
    Object.assign(c, { [THREADS]: threads, [CALLS]: 1, [START]: copies[FORMS[form].of] - 1, [OUT]: out });
    F.fill(-7, out / 4, out / 4 + rows);
    go();
  };
  const errors = {};
  for (const form of timed.filter((name) => name !== "read")) {
    once(form, 1, outs[0]);
    once(form, 4, outs[1]);
    const sums = reference(form), most = sums.reduce((a, b) => Math.max(a, Math.abs(b)), 0);
    let worst = 0;
    for (let i = 0; i < rows; i++) {
      if (!Object.is(F[outs[0] / 4 + i], F[outs[1] / 4 + i])) throw new Error(`${rows} x ${n}, ${form}: 4 threads differ from 1 at row ${i}`);
      if (i < checked) worst = Math.max(worst, Math.abs(F[outs[0] / 4 + i] - sums[i]));
    }
    errors[form] = worst / most;
    if (!(errors[form] < 2e-6)) throw new Error(`${rows} x ${n}, ${form}: ${errors[form].toExponential(2)} of the largest output away from the converter's reading`);
  }
  for (const [one, other] of SAME.filter((pair) => pair.every((form) => timed.includes(form)))) {
    once(one, 1, outs[0]);
    once(other, 1, outs[1]);
    for (let i = 0; i < rows; i++) if (!Object.is(F[outs[0] / 4 + i], F[outs[1] / 4 + i])) throw new Error(`${rows} x ${n}: ${other} differs from ${one} at row ${i}`);
  }
  console.log(`saluki-kernel-bench: ${rows} x ${n}: of the largest |output| of the first ${checked} rows, away from the converter's reading: ` +
    `${Object.entries(errors).map(([form, error]) => `${form} ${error.toExponential(1)}`).join(", ")}; each the same to the bit on 1 and 4 threads, ` +
    `and ${SAME.filter((pair) => pair.every((form) => timed.includes(form))).map((pair) => pair.join(" = ")).join(", ")} to the bit on every row`);
  // ---- the times, the forms taking turns
  for (const threads of [1, 4]) {
    c[THREADS] = threads;
    c[OUT] = outs[2];
    const calls = Math.max(megabytes ? 2 : 1, Math.round((threads === 1 ? 3e8 : 9e8) / (rows * n)));
    c[CALLS] = calls;
    const starts = Object.fromEntries(timed.map((form) => [form, 0]));
    const time = (form) => {
      set(form);
      Object.assign(c, { [THREADS]: threads, [CALLS]: calls, [OUT]: outs[2], [START]: starts[form] });
      starts[form] = (starts[form] + calls) % copies[FORMS[form].of];
      const t0 = performance.now();
      go();
      return (performance.now() - t0) / calls;
    };
    for (const form of timed) time(form);
    const speeds = Object.fromEntries(timed.map((form) => [form, []]));
    for (let r = 0; r < rounds; r++) {
      const ms = Object.fromEntries(timed.map((form) => [form, []]));
      for (let t = 0; t < turns; t++) for (const form of shuffled(timed)) ms[form].push(time(form));
      for (const form of timed) speeds[form].push((rows * n) / median(ms[form]) / 1e6);
    }
    const cells = timed.map((form) => {
      const g = median(speeds[form]), bytes = BYTES[FORMS[form].of] / 256;
      table.push({ rows, n, threads, form, g, low: Math.min(...speeds[form]), high: Math.max(...speeds[form]) });
      return `${form} ${g.toFixed(2)} (${(g * bytes).toFixed(2)} GB/s)`;
    });
    console.log(`saluki-kernel-bench: ${rows} x ${n}, ${threads} thread${threads > 1 ? "s" : ""}, G weights/s: ${cells.join(" | ")}`);
  }
  if (!prompts) continue;
  // ---- a prompt of 16 tokens: a frame a token, the vector above turned by a group a token
  {
    const count = 16, outFrame = align(rows * 4, 64);
    const tokens = {};
    for (const kind of Object.keys(frames)) {
      tokens[kind] = take(count * frame);
      for (let t = 0; t < count; t++) frames[kind](tokens[kind] + t * frame, t);
    }
    const tokenOuts = [take(count * outFrame), take(count * outFrame)];
    if (top > small) throw new Error("the small arrays do not fit their place");
    const setPrompt = (name) => {
      const { of, activations } = PROMPTS[name];
      Object.assign(c, { [FORM]: EVERY.indexOf(name), [FIRST]: firsts[of], [STRIDE]: bytesOf(of, rows, n), [COPIES]: copies[of],
        [XQ]: tokens[activations], [XS]: tokens[activations] + n, [COUNT]: count, [FRAME]: frame, [OUT_FRAME]: outFrame });
    };
    // a tile's numbers are its token's kernel's, to the bit
    for (const [by, tile] of [["rb, token by token", "rb tile"], ["rb, token by token", "sb tile"], ["sa, token by token", "sa tile"],
      ...(signedDot ? [["rb, token by token", "rw tile"], ["rb, token by token", "sw tile"]] : []), ["rb, token by token", "p.IQ2_XXS, token by token"],
      ...Object.entries(PROMPTS).filter(([, prompt]) => prompt.by && (signedDot || !prompt.signed)).map(([name, prompt]) => [prompt.by, name])]) {
      for (const [slot, name] of [[0, by], [1, tile]]) {
        setPrompt(name);
        Object.assign(c, { [THREADS]: 4, [CALLS]: 1, [START]: 0, [OUT]: tokenOuts[slot] });
        go();
      }
      for (let t = 0; t < count; t++) {
        for (let i = 0; i < rows; i++) {
          const at = t * outFrame / 4 + i;
          if (!Object.is(F[tokenOuts[0] / 4 + at], F[tokenOuts[1] / 4 + at])) throw new Error(`${rows} x ${n}: the ${tile} differs from the token's kernel at token ${t}, row ${i}`);
        }
      }
    }
    console.log(`saluki-kernel-bench: ${rows} x ${n}: the rb and sb tiles${signedDot ? ", the rw and sw tiles" : ""} give rb's numbers and the sa tile sa's to the bit, ${count} tokens, every row; ` +
      `and the page's tiles (p.<TYPE> tile${signedDot ? ", pw.<TYPE> tile" : ""}) its token's kernel's`);
    const names = Object.keys(PROMPTS).filter((name) => !PROMPTS[name].untimed && (signedDot || !PROMPTS[name].signed));
    for (const threads of [1, 4]) {
      const calls = Math.max(2, Math.round((threads === 1 ? 6e8 : 18e8) / (rows * n * count)));
      const starts = {};
      const time = (name) => {
        setPrompt(name);
        Object.assign(c, { [THREADS]: threads, [CALLS]: calls, [OUT]: tokenOuts[0], [START]: starts[name] ?? 0 });
        starts[name] = ((starts[name] ?? 0) + calls) % c[COPIES];
        const t0 = performance.now();
        go();
        return (performance.now() - t0) / calls;
      };
      for (const name of names) time(name);
      const speeds = Object.fromEntries(names.map((name) => [name, []]));
      for (let r = 0; r < rounds; r++) {
        const ms = Object.fromEntries(names.map((name) => [name, []]));
        for (let t = 0; t < turns; t++) for (const name of shuffled(names)) ms[name].push(time(name));
        for (const name of names) speeds[name].push((rows * n * count) / median(ms[name]) / 1e6);
      }
      const cells = names.map((name) => {
        promptTable.push({ rows, n, threads, name, g: median(speeds[name]) });
        return `${name} ${median(speeds[name]).toFixed(2)}`;
      });
      console.log(`saluki-kernel-bench: a prompt of ${count} tokens, ${rows} x ${n}, ${threads} thread${threads > 1 ? "s" : ""}, G weights/s: ${cells.join(" | ")}`);
    }
  }
}
// ---- every form against rb, the geometric mean over the shapes
const geo = (list) => Math.exp(list.reduce((a, b) => a + Math.log(b), 0) / list.length);
for (const threads of [1, 4]) {
  const of = (form) => table.filter((row) => row.threads === threads && row.form === form);
  const base = of(timed.includes("rb") ? "rb" : timed[0]);
  console.log(`saluki-kernel-bench: ${threads} thread${threads > 1 ? "s" : ""}, all the shapes, G weights/s (against ${base[0].form}): ` +
    timed.map((form) => `${form} ${geo(of(form).map((row) => row.g)).toFixed(2)} (${geo(of(form).map((row, i) => row.g / base[i].g)).toFixed(2)}x)`).join(" | "));
}
for (const threads of prompts ? [1, 4] : []) {
  const of = (name) => promptTable.filter((row) => row.threads === threads && row.name === name).map((row) => row.g);
  const names = Object.keys(PROMPTS).filter((name) => !PROMPTS[name].untimed && (signedDot || !PROMPTS[name].signed));
  const base = (name) => of(PROMPTS[name].by ?? (PROMPTS[name].base ? name : names[0]));  // (a page's tile against its own type's token by token)
  console.log(`saluki-kernel-bench: a prompt of 16 tokens, ${threads} thread${threads > 1 ? "s" : ""}, all the shapes, G weights/s (against the token's kernel): ` +
    names.map((name) => `${name} ${geo(of(name)).toFixed(2)} (${geo(of(name).map((g, i) => g / base(name)[i])).toFixed(2)}x)`).join(" | "));
}
// ---- a token of the 27B from these speeds. The weights a token reads, by type (docs/notes/t361-saluki-2026-10-13.md:
// every matrix but the embedding); a type with no form here goes at the speed of the nearest that has one. 21.2% of a
// token's weights are in rows of 17408 (ffn_down), the rest in rows of 5120 or 6144
const VALUES = { rb: 9201254400 + 2715811840 + 2160066560, iq3s: 2901934080 + 1399848960, q2k: 2721054720 + 230686720 + 104857600, iq1m: 1572864000 + 2589982720 };
const WIDE = 0.212, ALL_VALUES = 25598361600;
if (["rb", "iq3s", "q2k", "iq1m"].every((form) => timed.includes(form)) && shapes.some(([, n]) => n === 17408) && shapes.some(([, n]) => n === 5120)) {
  for (const threads of [1, 4]) {
    const speed = (form, wide) => geo(table.filter((row) => row.threads === threads && row.form === form && (row.n === 17408) === wide).map((row) => row.g)) * 1e9;
    const seconds = (form, values) => values * (WIDE / speed(form, true) + (1 - WIDE) / speed(form, false));
    const saluki = Object.entries(VALUES).reduce((sum, [form, values]) => sum + seconds(form, values), 0);
    const others = ["t2r", "q8r"].filter((form) => timed.includes(form)).map((form) => `all of them as ${form} ${seconds(form, ALL_VALUES).toFixed(2)} s (${(1 / seconds(form, ALL_VALUES)).toFixed(2)} a second)`);
    console.log(`saluki-kernel-bench: the matrix products of a token of the 27B (${(ALL_VALUES / 1e9).toFixed(1)} G weights), ${threads} thread${threads > 1 ? "s" : ""}, an estimate: ` +
      `Saluki's types ${saluki.toFixed(2)} s (${(1 / saluki).toFixed(2)} a second: IQ2 ${seconds("rb", VALUES.rb).toFixed(2)}, IQ3 ${seconds("iq3s", VALUES.iq3s).toFixed(2)}, ` +
      `K ${seconds("q2k", VALUES.q2k).toFixed(2)}, IQ1 ${seconds("iq1m", VALUES.iq1m).toFixed(2)})${others.length ? `; ${others.join("; ")}` : ""}`);
  }
}
// T363.4: the same from the page's kernels, every type at its own speed (a token reads every matrix but the embedding:
// IQ1_M's 2,844,262,400 less the embedding's 1,271,398,400)
const READ = { IQ2_XXS: 9201254400, IQ3_S: 2901934080, Q2_K: 2721054720, IQ2_S: 2715811840, IQ2_XS: 2160066560, IQ1_M: 1572864000, IQ3_XXS: 1399848960,
  IQ1_S: 2589982720, IQ4_XS: 230686720, Q4_K: 104857600 };
if (BLOCKS.every((of) => timed.includes(`p.${of}`)) && shapes.some(([, n]) => n === 17408) && shapes.some(([, n]) => n === 5120)) {
  for (const threads of [1, 4]) {
    const speed = (form, wide) => geo(table.filter((row) => row.threads === threads && row.form === form && (row.n === 17408) === wide).map((row) => row.g)) * 1e9;
    const seconds = (of) => READ[of] * (WIDE / speed(`p.${of}`, true) + (1 - WIDE) / speed(`p.${of}`, false));
    const all = BLOCKS.reduce((sum, of) => sum + seconds(of), 0);
    console.log(`saluki-kernel-bench: the matrix products of a token of the 27B on the page's kernels, ${threads} thread${threads > 1 ? "s" : ""}, an estimate: ${all.toFixed(2)} s ` +
      `(${(1 / all).toFixed(2)} a second: ${BLOCKS.map((of) => `${of} ${seconds(of).toFixed(3)}`).join(", ")})`);
  }
}
Atomics.store(sync, GO, -1);
Atomics.notify(sync, GO);
await leave(0);
