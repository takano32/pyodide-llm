// T231: the matrix product of a token on ternary weights: the kernels of kernels/ (t2r: matmul_t2r, t2: matmul_t2 without
// relaxed SIMD; q8r and q8: the int8 kernels on the same weights widened to int8, which is how T235 ran a ternary
// model) and the forms of tests/ternary-forms.ts that were set against them (bc, bs, bx, c, a), in one process, taking turns, on 1 and 4 threads of one shared memory, the rows in chunks taken in turn as forward.js's
// phase() cuts them (TODO.md's T231 has what each form is and the table).
//
// The matrices are read from memory, as a model's are (each read once a token): copies of each past every cache
// (--megabytes a form, default 256), a call on each in turn (T159's review: one matrix timed again and again stays in
// the caches). Every form is held to the float64 sums of its own integers and scales, and to itself on 1 and 4 threads,
// to the bit. G weights/s is rows x n a second; GB/s counts what a row reads of the weights (their scales and
// corrections too).
//
//   node tests/ternary-bench.mjs [--rounds 2] [--turns 5] [--megabytes 256] [--shapes 2048x2048,17408x5120] [--forms t2r,q8r] [--no-prompt]
//
// Then a prompt of 16 tokens through the same matrices: the ternary tile (matmul_t2r_tile, a row against four tokens)
// against the token's kernel for every token, and int8's tile (T159) on the widened weights.
//
// The shapes are those of Ternary Bonsai 1.7B (2048 wide, its FFN 6144) and of the 27B (5120 wide, its FFN 17408).
// Compiles with AssemblyScript into .tmp/ternary-bench/ (needs `npm ci`).
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { Worker, isMainThread, workerData } from "node:worker_threads";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..") + "/";
const work = root + ".tmp/ternary-bench/";
// the numbers every thread reads: 0 the form, 1 the first copy, 2 a copy's bytes, 3 the copies, 4 the copy to start at,
// 5 n, 6 rows, 7 threads, 8 out, 9 the activations, 10 their scales, 11 the calls
// (and for a prompt: 12 its tokens, 13 the bytes from a token's frame to the next, 14 from its outputs to the next's)
const FORM = 0, FIRST = 1, STRIDE = 2, COPIES = 3, START = 4, N = 5, ROWS = 6, THREADS = 7, OUT = 8, XQ = 9, XS = 10, CALLS = 11,
  COUNT = 12, FRAME = 13, OUT_FRAME = 14;
const GO = 0, DONE = 1, NEXT = 2;

// Every form: its module, the bytes a group of 128 weights takes (values, scales, corrections), which activations it
// reads, and the call on rows r0..r1 of the copy at w (its values, then its scales, then its corrections)
const g128 = (c) => (c[ROWS] * c[N]) / 128;
const FORMS = {
  t2r: { module: "tree", bytes: [32, 4, 0], activations: "interleaved",
    run: (k, c, w, r0, r1) => k.matmul_t2r(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, c[N], r0, r1, 3) },
  t2: { module: "plain", bytes: [32, 4, 0], activations: "interleaved",
    run: (k, c, w, r0, r1) => k.matmul_t2(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, c[N], r0, r1, 3) },
  bc: { module: "forms", bytes: [32, 4, 0], activations: "interleaved",
    run: (k, c, w, r0, r1) => k.matmul_bc(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, c[N], r0, r1) },
  bs: { module: "forms", bytes: [32, 4, 0], activations: "interleaved",
    run: (k, c, w, r0, r1) => k.matmul_bs(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, c[N], r0, r1) },
  bx: { module: "forms", bytes: [32, 4, 0], activations: "interleaved",
    run: (k, c, w, r0, r1) => k.matmul_bx(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, c[N], r0, r1) },
  bxa: { module: "forms", bytes: [32, 4, 0], activations: "interleaved",
    run: (k, c, w, r0, r1) => k.matmul_bxa(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, c[N], r0, r1, 3) },
  c: { module: "forms", bytes: [32, 4, 16], activations: "seven",
    run: (k, c, w, r0, r1) => k.matmul_c(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, w + g128(c) * 36, c[N], r0, r1) },
  a: { module: "forms", bytes: [28, 4, 0], activations: "base3",
    run: (k, c, w, r0, r1) => k.matmul_a(c[OUT], c[XQ], c[XS], w, w + g128(c) * 28, c[N], r0, r1) },
  q8r: { module: "tree", bytes: [128, 16, 16], activations: "seven",
    run: (k, c, w, r0, r1) => k.matmul_q8r(c[OUT], c[XQ], c[XS], w, w + g128(c) * 128, w + g128(c) * 144, c[N], r0, r1) },
  q8: { module: "plain", bytes: [128, 16, 0], activations: "eight",
    run: (k, c, w, r0, r1) => k.matmul_q8(c[OUT], c[XQ], c[XS], w, w + g128(c) * 128, c[N], r0, r1) },
};
// A prompt's tokens (T108) through the same matrix: the token's kernel for every token, the rows in blocks that stay in
// the first cache (jobs.js's way for a kernel without a tile), the ternary tile (matmul_t2r_tile), and int8's (T159) on
// the same weights widened. of: whose copies of the weights it reads
const blockOf = (n) => Math.max(1, Math.floor(16384 / (n / 4)));
const PROMPTS = {
  "t2r, token by token": { module: "tree", of: "t2r", activations: "interleaved", run: (k, c, w, r0, r1) => {
    for (let r = r0, step = blockOf(c[N]); r < r1; r += step) {
      for (let t = 0; t < c[COUNT]; t++) {
        k.matmul_t2r(c[OUT] + t * c[OUT_FRAME], c[XQ] + t * c[FRAME], c[XS] + t * c[FRAME], w, w + g128(c) * 32, c[N], r, Math.min(r + step, r1), 3);
      }
    }
  } },
  "t2r tile": { module: "tree", of: "t2r", activations: "interleaved",
    run: (k, c, w, r0, r1) => k.matmul_t2r_tile(c[OUT], c[XQ], c[XS], w, w + g128(c) * 32, c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME], 3) },
  "q8r tile": { module: "tree", of: "q8r", activations: "seven",
    run: (k, c, w, r0, r1) => k.matmul_q8r_tile(c[OUT], c[XQ], c[XS], w, w + g128(c) * 128, w + g128(c) * 144, c[N], r0, r1, c[COUNT], c[OUT_FRAME], c[FRAME]) },
};
const NAMES = Object.keys(FORMS), ALL = { ...FORMS, ...PROMPTS }, EVERY = Object.keys(ALL);
const modules = { forms: "forms.wasm", tree: "relaxed.wasm", plain: "plain.wasm" };
const instances = (memory) => Object.fromEntries(Object.entries(modules).map(([name, file]) =>
  [name, new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(work + file)), { env: { memory } }).exports]));
// forward.js's phase(): chunks of a quarter of a thread's share (T93), taken in turn; one thread the whole matrix
// (a prompt's chunks in fours of rows, as phase() cuts them for the tiles)
const chunkOf = (rows, threads, quad = 1) => (threads === 1 ? rows : quad * Math.ceil(rows / (threads * 4 * quad)));
function runCalls(kernels, sync, c, index) {
  const form = ALL[EVERY[c[FORM]]], k = kernels[form.module];
  const threads = c[THREADS], rows = c[ROWS], calls = c[CALLS];
  const size = chunkOf(rows, threads, c[COUNT] > 1 ? 4 : 1), chunks = Math.ceil(rows / size);
  const weights = (call) => c[FIRST] + ((c[START] + call) % c[COPIES]) * c[STRIDE];
  if (threads === 1) {
    if (index === 0) for (let call = 0; call < calls; call++) form.run(k, c, weights(call), 0, rows);
    return;
  }
  for (let at = Atomics.add(sync, NEXT, 1); at < calls * chunks; at = Atomics.add(sync, NEXT, 1)) {
    const r0 = (at % chunks) * size;
    form.run(k, c, weights(Math.floor(at / chunks)), r0, Math.min(r0 + size, rows));
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
const rounds = Number(text("--rounds", 2)), turns = Number(text("--turns", 5)), megabytes = Number(text("--megabytes", 256));
const shapes = text("--shapes", "2048x2048,6144x2048,2048x6144,5120x5120,17408x5120,5120x17408").split(",").map((shape) => shape.split("x").map(Number));
const timed = text("--forms", NAMES.join(",")).split(",");
fs.mkdirSync(work, { recursive: true });
const asc = ["asc", "-O3", "--noAssert", "--runtime", "stub", "--importMemory", "--noExportMemory", "--initialMemory", "1",
  "--sharedMemory", "--maximumMemory", "65536"];
for (const [source, out, features] of [["tests/ternary-forms.ts", "forms.wasm", "simd,relaxed-simd,threads"],
  ["kernels/kernel_relaxed.ts", "relaxed.wasm", "simd,relaxed-simd,threads"], ["kernels/kernel.ts", "plain.wasm", "simd,threads"]]) {
  execFileSync("npx", [...asc, root + source, "-o", work + out, "--enable", features], { cwd: root, stdio: "inherit" });
}
const cpu = () => {
  const model = os.cpus()[0]?.model;
  if (model && model !== "unknown") return model;
  const part = (fs.existsSync("/proc/cpuinfo") ? fs.readFileSync("/proc/cpuinfo", "utf8") : "").match(/CPU part\s*:\s*(0x[0-9a-f]+)/)?.[1];
  return part ? `arm64 ${{ "0xd0c": "Neoverse-N1", "0xd49": "Neoverse-N2", "0xd40": "Neoverse-V1", "0xd4f": "Neoverse-V2" }[part] ?? part}` : "unknown";
};
console.log(`ternary-bench: ${cpu()}, ${os.cpus().length} logical cores, matrices read from ${megabytes} MB of copies a form`);

const PAGE = 65536, align = (bytes, to = 4096) => Math.ceil(bytes / to) * to;
const largest = Math.max(...shapes.map(([rows, n]) => rows * n));
const bytesOf = (form, rows, n) => align(((rows * n) / 128) * FORMS[form].bytes.reduce((a, b) => a + b) + 64);
// the small arrays (the activations of each kind, the outputs), then every form's copies
const small = align(PAGE + 64 * Math.max(...shapes.map(([rows, n]) => 2 * n + rows * 4)), PAGE);
const region = (form) => Math.max(megabytes * 2 ** 20, 2 * bytesOf(form, 1, largest));
const total = small + NAMES.reduce((sum, form) => sum + align(region(form), PAGE), 0);
if (total > 65536 * PAGE) throw new Error(`${(total / 2 ** 30).toFixed(1)} GiB of copies do not fit a 32-bit memory: fewer --megabytes`);
const memory = new WebAssembly.Memory({ initial: Math.ceil(total / PAGE), maximum: 65536, shared: true });
const control = new SharedArrayBuffer(16), sync = new Int32Array(control);
const numbers = new SharedArrayBuffer(16 * 8), c = new Float64Array(numbers);
const kernels = instances(memory);
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
const U = new Uint8Array(memory.buffer), I = new Int8Array(memory.buffer), F = new Float32Array(memory.buffer), N32 = new Int32Array(memory.buffer);
const firsts = {};
NAMES.reduce((at, form) => { firsts[form] = at; return at + align(region(form), PAGE); }, small);

const table = [], promptTable = [], prompts = !args.includes("--no-prompt");
for (const [rows, n] of shapes) {
  if (n % 128) throw new Error(`rows of ${n} are not whole groups of 128`);
  const groups = (rows * n) / 128, ng = n / 32;
  let seed = 7;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
  // the ternary weights (a third each, as the files have them) and their scales
  const tw = new Int8Array(rows * n), d = new Float32Array(groups);
  for (let i = 0; i < tw.length; i++) tw[i] = (next() % 3) - 1;
  for (let g = 0; g < groups; g++) d[g] = Math.fround(0.005 + (next() % 4096) / 163840);
  // ---- each form's copy of the weights
  const writers = {
    t2r(at) {
      const scales = at + groups * 32;
      U.fill(0, at, scales);
      for (let j = 0; j < tw.length; j++) U[at + (j >> 2)] |= (tw[j] + 1) << (2 * (j & 3));
      F.set(d, scales / 4);
    },
    c(at) {
      const scales = at + groups * 32, corrections = scales + groups * 4;
      U.fill(0, at, scales);
      // a block of 64 in 16 bytes: byte c holds weights c, c + 16, c + 32, c + 48
      for (let j = 0; j < tw.length; j++) U[at + (j >> 6) * 16 + (j & 15)] |= (tw[j] + 1) << (2 * ((j >> 4) & 3));
      F.set(d, scales / 4);
      for (let g = 0; g < groups * 4; g++) {
        let sum = 0;
        for (let j = g * 32; j < g * 32 + 32; j++) sum += tw[j];
        N32[corrections / 4 + g] = -64 * sum;
      }
    },
    a(at) {
      const scales = at + groups * 28;
      const byte = (digits) => Math.ceil((digits.reduce((v, digit) => v * 3 + digit, 0) * 256) / 243);
      for (let g = 0; g < groups; g++) {
        const w = (j) => tw[g * 128 + j] + 1, block = at + g * 28;
        for (let m = 0; m < 16; m++) U[block + m] = byte([0, 1, 2, 3, 4].map((digit) => w(16 * digit + m)));
        for (let m = 0; m < 8; m++) U[block + 16 + m] = byte([0, 1, 2, 3, 4].map((digit) => w(80 + 8 * digit + m)));
        for (let m = 0; m < 2; m++) U[block + 24 + m] = byte([0, 1, 2, 3].map((digit) => w(120 + 2 * digit + m)).concat(0));
        U[block + 26] = U[block + 27] = 255;  // where the file has the scale: never read as weights
      }
      F.set(d, scales / 4);
    },
    q8r(at) {  // T235's int8: -127, 0, 127 and float32(d / 127) a group of 32
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
  writers.t2 = writers.bc = writers.bs = writers.bx = writers.bxa = writers.t2r;
  writers.q8 = writers.q8r;
  const copies = {};
  for (const form of NAMES) {
    const stride = bytesOf(form, rows, n), first = firsts[form];
    writers[form](first);
    copies[form] = megabytes ? Math.max(2, Math.floor(region(form) / stride)) : 1;
    for (let copy = 1; copy < copies[form]; copy++) U.copyWithin(first + copy * stride, first, first + stride);
  }
  // ---- the activations: one vector, quantized as each kind of kernel takes it, in groups of 32
  const x = new Float32Array(n);
  for (let j = 0; j < n; j++) x[j] = ((next() % 2001) - 1000) / 250 * (1 + (next() % 7 === 0 ? 3 : 0));
  const quantized = (most, bias) => {
    const q = new Int32Array(n), scales = new Float32Array(ng);
    for (let g = 0; g < ng; g++) {
      let top = 0;
      for (let j = g * 32; j < g * 32 + 32; j++) top = Math.max(top, Math.abs(x[j]));
      scales[g] = Math.fround(top / most);
      for (let j = g * 32; j < g * 32 + 32; j++) q[j] = Math.round(x[j] / scales[g]) + bias;
    }
    return { q, scales, bias };
  };
  const eight = quantized(127, 0), seven = quantized(63, 64);
  const sums = (q) => Array.from({ length: ng }, (_, g) => -q.slice(g * 32, g * 32 + 32).reduce((a, b) => a + b, 0));
  let top = PAGE;
  const take = (bytes) => { const at = top; top = align(at + bytes, 64); return at; };
  const places = {};
  const put = (kind, bytes, scales, more = []) => {
    const xq = take(bytes.length), xs = take((scales.length + more.length) * 4);
    I.set(bytes, xq);
    F.set(scales, xs / 4);
    N32.set(more, xs / 4 + scales.length);
    places[kind] = { xq, xs };
  };
  put("seven", seven.q, seven.scales);
  put("eight", eight.q, eight.scales);
  // interleaved in blocks of 64: byte 16 p + c is activation 4 c + p; after the scales, minus each group's sum
  put("interleaved", Int8Array.from({ length: n }, (_, at) => eight.q[(at & ~63) + 4 * (at & 15) + ((at >> 4) & 3)]), eight.scales, sums(eight.q));
  // PTQ1_0's order: 160 bytes a block of 128 (tests/ternary-forms.ts)
  const base3 = new Int8Array((n / 128) * 160);
  for (let b = 0; b < n / 128; b++) {
    const q = (j) => eight.q[b * 128 + j];
    for (let j = 0; j < 80; j++) base3[b * 160 + j] = q(j);
    for (let digit = 0; digit < 5; digit++) {
      for (let m = 0; m < 8; m++) base3[b * 160 + 80 + 16 * digit + m] = q(80 + 8 * digit + m);
      if (digit < 4) for (let m = 0; m < 2; m++) base3[b * 160 + 80 + 16 * digit + 8 + m] = q(120 + 2 * digit + m);
    }
  }
  put("base3", base3, eight.scales, sums(eight.q));
  {
    // interleave() itself (kernel.ts), on a copy of the int8 activations: the bytes and sums written above
    const { xq, xs } = places.interleaved, own = take(n), ownScales = take(ng * 8);
    I.set(eight.q, own);
    kernels.plain.interleave(own, ownScales, n);
    for (let j = 0; j < n; j++) if (I[own + j] !== I[xq + j]) throw new Error(`interleave() differs at byte ${j}`);
    for (let g = 0; g < ng; g++) if (N32[(ownScales + ng * 4) / 4 + g] !== N32[(xs + ng * 4) / 4 + g]) throw new Error(`interleave()'s sum of group ${g} differs`);
  }
  const outs = [take(rows * 4), take(rows * 4), take(rows * 4)];
  if (top > small) throw new Error("the small arrays do not fit their place");
  // ---- the float64 sums of each kind's integers and scales
  const reference = (kind, weightScale, weight) => {
    const { q, scales, bias } = kind, out = new Float64Array(rows);
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      for (let g = 0; g < ng; g++) {
        let dot = 0;
        for (let j = g * 32; j < g * 32 + 32; j++) dot += weight * tw[i * n + j] * (q[j] - bias);
        sum += dot * scales[g] * weightScale(d[(i * n + g * 32) >> 7]);
      }
      out[i] = sum;
    }
    return out;
  };
  const whole = (scale) => scale, over127 = (scale) => Math.fround(scale / 127);
  const references = { t2r: reference(eight, whole, 1), c: reference(seven, whole, 1), q8r: reference(seven, over127, 127), q8: reference(eight, over127, 127) };
  references.t2 = references.bc = references.bs = references.bx = references.bxa = references.a = references.t2r;
  Object.assign(c, { [N]: n, [ROWS]: rows });
  const set = (form) => Object.assign(c, { [FORM]: NAMES.indexOf(form), [FIRST]: firsts[form], [STRIDE]: bytesOf(form, rows, n),
    [COPIES]: copies[form], [XQ]: places[FORMS[form].activations].xq, [XS]: places[FORMS[form].activations].xs, [COUNT]: 1 });
  const errors = {};
  for (const form of timed) {
    set(form);
    for (const [slot, threads] of [[0, 1], [1, 4]]) {
      Object.assign(c, { [THREADS]: threads, [CALLS]: 1, [START]: copies[form] - 1, [OUT]: outs[slot] });
      F.fill(-7, outs[slot] / 4, outs[slot] / 4 + rows);
      go();
    }
    const most = references[form].reduce((a, b) => Math.max(a, Math.abs(b)), 0);
    let worst = 0;
    for (let i = 0; i < rows; i++) {
      if (!Object.is(F[outs[0] / 4 + i], F[outs[1] / 4 + i])) throw new Error(`${rows} x ${n}, ${form}: 4 threads differ from 1 at row ${i}`);
      worst = Math.max(worst, Math.abs(F[outs[0] / 4 + i] - references[form][i]));
    }
    errors[form] = worst / most;
    if (!(errors[form] < 2e-6)) throw new Error(`${rows} x ${n}, ${form}: ${errors[form].toExponential(2)} of the largest output away from the float64 sums`);
  }
  console.log(`ternary-bench: ${rows} x ${n}: every form within ${Math.max(...Object.values(errors)).toExponential(2)} of its float64 sums (of the largest |output|), and the same to the bit on 1 and 4 threads`);
  // ---- the times, the forms taking turns
  for (const threads of [1, 4]) {
    c[THREADS] = threads;
    c[OUT] = outs[2];
    const calls = Math.max(megabytes ? 2 : 1, Math.round((threads === 1 ? 3e8 : 9e8) / (rows * n)));
    c[CALLS] = calls;
    const starts = Object.fromEntries(timed.map((form) => [form, 0]));
    const time = (form) => {
      set(form);
      c[START] = starts[form];
      starts[form] = (starts[form] + calls) % copies[form];
      const t0 = performance.now();
      go();
      return (performance.now() - t0) / calls;
    };
    for (const form of timed) time(form);
    const speeds = Object.fromEntries(timed.map((form) => [form, []]));
    for (let r = 0; r < rounds; r++) {
      const ms = Object.fromEntries(timed.map((form) => [form, []]));
      for (let t = 0; t < turns; t++) for (const form of timed) ms[form].push(time(form));
      for (const form of timed) speeds[form].push((rows * n) / median(ms[form]) / 1e6);
    }
    const cells = timed.map((form) => {
      const g = median(speeds[form]), bytes = FORMS[form].bytes.reduce((a, b) => a + b) / 128;
      table.push({ rows, n, threads, form, g, low: Math.min(...speeds[form]), high: Math.max(...speeds[form]) });
      return `${form} ${g.toFixed(1)} (${(g * bytes).toFixed(1)} GB/s)`;
    });
    console.log(`ternary-bench: ${rows} x ${n}, ${threads} thread${threads > 1 ? "s" : ""}, G weights/s: ${cells.join(" | ")}`);
  }
  if (!prompts) continue;
  // ---- a prompt of 16 tokens: a frame a token (its activations, then their scales and sums), the vector above turned
  // by a group a token
  {
    const count = 16, frame = align(n + ng * 8, 64), outFrame = align(rows * 4, 64);
    const frames = {};
    for (const [kind, { q, scales }] of [["interleaved", eight], ["seven", seven]]) {
      frames[kind] = take(count * frame);
      for (let t = 0; t < count; t++) {
        const at = frames[kind] + t * frame;
        for (let j = 0; j < n; j++) I[at + j] = q[(j + 32 * t) % n];
        for (let g = 0; g < ng; g++) F[(at + n) / 4 + g] = scales[(g + t) % ng];
        if (kind === "interleaved") kernels.plain.interleave(at, at + n, n);
      }
    }
    const tokenOuts = [take(count * outFrame), take(count * outFrame)];
    if (top > small) throw new Error("the small arrays do not fit their place");
    const setPrompt = (name) => {
      const { of, activations } = PROMPTS[name];
      Object.assign(c, { [FORM]: EVERY.indexOf(name), [FIRST]: firsts[of], [STRIDE]: bytesOf(of, rows, n), [COPIES]: copies[of],
        [XQ]: frames[activations], [XS]: frames[activations] + n, [COUNT]: count, [FRAME]: frame, [OUT_FRAME]: outFrame });
    };
    // the tile's numbers are the token kernel's, to the bit
    for (const [slot, name] of [[0, "t2r, token by token"], [1, "t2r tile"]]) {
      setPrompt(name);
      Object.assign(c, { [THREADS]: 4, [CALLS]: 1, [START]: 0, [OUT]: tokenOuts[slot] });
      go();
    }
    for (let i = 0; i < (count * outFrame) / 4; i++) {
      if (!Object.is(F[tokenOuts[0] / 4 + i], F[tokenOuts[1] / 4 + i])) throw new Error(`${rows} x ${n}: the tile differs from the token's kernel at ${i}`);
    }
    for (const threads of [1, 4]) {
      c[THREADS] = threads;
      const calls = Math.max(2, Math.round((threads === 1 ? 6e8 : 18e8) / (rows * n * count)));
      const starts = {};
      const time = (name) => {
        setPrompt(name);
        Object.assign(c, { [CALLS]: calls, [OUT]: tokenOuts[0], [START]: starts[name] ?? 0 });
        starts[name] = ((starts[name] ?? 0) + calls) % c[COPIES];
        const t0 = performance.now();
        go();
        return (performance.now() - t0) / calls;
      };
      for (const name of Object.keys(PROMPTS)) time(name);
      const speeds = Object.fromEntries(Object.keys(PROMPTS).map((name) => [name, []]));
      for (let r = 0; r < rounds; r++) {
        const ms = Object.fromEntries(Object.keys(PROMPTS).map((name) => [name, []]));
        for (let t = 0; t < turns; t++) for (const name of Object.keys(PROMPTS)) ms[name].push(time(name));
        for (const name of Object.keys(PROMPTS)) speeds[name].push((rows * n * count) / median(ms[name]) / 1e6);
      }
      const cells = Object.keys(PROMPTS).map((name) => {
        promptTable.push({ n, threads, name, g: median(speeds[name]) });
        return `${name} ${median(speeds[name]).toFixed(1)}`;
      });
      console.log(`ternary-bench: a prompt of ${count} tokens, ${rows} x ${n}, ${threads} thread${threads > 1 ? "s" : ""}, G weights/s: ${cells.join(" | ")}`);
    }
  }
}
// ---- every form against the int8 kernel that runs a ternary model today (q8r), the geometric mean over the shapes
const geo = (list) => Math.exp(list.reduce((a, b) => a + Math.log(b), 0) / list.length);
for (const threads of [1, 4]) {
  const of = (form) => table.filter((row) => row.threads === threads && row.form === form);
  const base = of(timed.includes("q8r") ? "q8r" : timed[0]);
  for (const width of [...new Set(shapes.map(([, n]) => n))].concat("all")) {
    const pick = (list) => list.filter((row) => width === "all" || row.n === width);
    const line = timed.map((form) => `${form} ${geo(pick(of(form)).map((row) => row.g)).toFixed(1)} (${geo(pick(of(form)).map((row, i) => row.g / pick(base)[i].g)).toFixed(2)}x)`);
    console.log(`ternary-bench: ${threads} thread${threads > 1 ? "s" : ""}, rows of ${width}, G weights/s (against ${base[0].form}): ${line.join(" | ")}`);
  }
}
for (const threads of prompts ? [1, 4] : []) {
  const of = (name) => promptTable.filter((row) => row.threads === threads && row.name === name).map((row) => row.g);
  const base = of("t2r, token by token");
  console.log(`ternary-bench: a prompt of 16 tokens, ${threads} thread${threads > 1 ? "s" : ""}, all the shapes, G weights/s (against the token's kernel): ` +
    Object.keys(PROMPTS).map((name) => `${name} ${geo(of(name)).toFixed(1)} (${geo(of(name).map((g, i) => g / base[i])).toFixed(2)}x)`).join(" | "));
}
Atomics.store(sync, GO, -1);
Atomics.notify(sync, GO);
process.exit(0);
