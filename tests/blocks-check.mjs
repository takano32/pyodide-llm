// blocks-check.mjs (T363.4): the kernels of llama.cpp's blocks (kernels/blocks.ts, blocks_relaxed.ts: the weights of
// Underdog Saluki 27B as its GGUF holds them) against the same arithmetic written out value by value
// (tests/blocks-model.mjs), to the bit. Node, the compiled kernels and one call of Python (the converter's own reader
// of the blocks, T362: tests/saluki_blocks.py), a few seconds: the light suite runs it. All eight builds: with and
// without relaxed SIMD, on a 32-bit and a 64-bit memory, plain and shared, and the 64-bit ones above 4 GiB too; and
// each of them as V8's Liftoff alone compiles it and as TurboFan alone does (the check runs itself again with the flags).
//
//   the reading: for each of the ten types, rows of random bytes (every byte string is a block; the float16 scales
//     alone are made finite) and rows of all zeros and all ones (the largest indices and scales). The model's float32
//     value of every weight is the converter's reader's, to the bit; so the integers the kernels sum are that reader's.
//   sb_tables: the tables it writes of the checkpoint's row of codebooks, and that it writes no byte more.
//   sums_x: the sums of each 16 and each 8 activations, at int8's ends too, and nothing else written.
//   sb_row: a row as float32 is the reader's values, to the bit.
//   matmul_sb and matmul_sbr: the model's number for every row, to the bit (so the two are each other's); for ranges of
//     rows that begin and end anywhere, with nothing written outside the range; rows of 1 to 3 blocks (a kernel that
//     took a block's scale or its activations' sums from the neighbour fails at two). And beside the float64 sum of the
//     reader's values times the activations, within 1e-5 of the largest output (float32's own rounding of 8 to 24
//     terms was 2.1e-6 at most here; T363.1's line for rows of 5120 was 2e-6. One weight of 256 read wrongly is
//     percents).
//   matmul_sbr_tile: a prompt of 1 to 9, 12 and 16 tokens (every remainder of four, and more than one four), each
//     token its own activations, a frame and an output stride that are not the vectors' lengths: every (row, token)
//     is the token's kernel's number to the bit.
//   matmul_sbw_tile: the same, where this engine's relaxed dot product multiplies signed bytes by signed bytes
//     (signedDots() asks it of every pair of bytes: arm64 does, x86-64 does not), and that it writes no byte past its
//     scratch. Where the engine does not, that is said and the kernel is not called.
//
//   node tests/blocks-check.mjs        (PYTHON=<a python with NumPy>; after `make kernels`)
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { blocksModel } from "./blocks-model.mjs";
import { built, runtimeUrl } from "./tree.mjs";
const { addressed } = await import(runtimeUrl("jobs.ts"));

const root = new URL("../", import.meta.url).pathname;
const work = path.join(root, ".tmp/blocks-check/");
const model = blocksModel(), NAMES = Object.keys(model.TYPES);
let checked = 0;
let seed = 3634;
const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;

// ---- the rows of every type: for each number of blocks a row (1, 2, 3), ROWS rows, of which the last two are all
// zeros and all ones; their scales made finite float16s
const ROWS = 6, WIDTHS = [1, 2, 3], MOST = 3 * 256;
const finite = (bits) => (bits & 0x7c00) === 0x7c00 ? bits & ~0x0400 : bits;
function blocksOf(name, count, fill) {
  const size = model.TYPES[name][1], raw = Uint8Array.from({ length: count * size }, () => fill ?? next() & 255);
  const fix = (at) => { const bits = finite(raw[at] | (raw[at + 1] << 8)); raw[at] = bits & 255; raw[at + 1] = bits >> 8; };
  for (let b = 0; b < count; b++) {
    const p = b * size;
    if (name === "IQ1_M") raw[p + 55] &= ~0x40;  // its d is the top four bits of its last four words: the exponent's top bit cleared
    else if (name === "Q2_K") { fix(p + 80); fix(p + 82); }
    else if (name === "Q4_K") { fix(p); fix(p + 2); }
    else fix(p);
  }
  return raw;
}
const matrices = {};  // name -> [{ blocks, at (in the type's file), raw }]
fs.mkdirSync(work, { recursive: true });
for (const name of NAMES) {
  let at = 0;
  const parts = [];
  matrices[name] = WIDTHS.map((blocks) => {
    const raw = new Uint8Array(ROWS * blocks * model.TYPES[name][1]), row = raw.length / ROWS;
    raw.set(blocksOf(name, (ROWS - 2) * blocks));
    raw.set(blocksOf(name, blocks, 0), (ROWS - 2) * row);
    raw.set(blocksOf(name, blocks, 255), (ROWS - 1) * row);
    parts.push(raw);
    const matrix = { blocks, at, raw };
    at += ROWS * blocks * 256;
    return matrix;
  });
  fs.writeFileSync(`${work}${name}.blocks`, Buffer.concat(parts));
}
execFileSync(process.env.PYTHON ?? "python3", [path.join(root, "tests/saluki_blocks.py"), work], { stdio: "inherit" });
const books = new Int8Array(fs.readFileSync(`${work}codebooks.bin`));
if (books.length !== model.BOOKS.bytes) throw new Error(`the codebooks are ${books.length} bytes, not ${model.BOOKS.bytes}`);
const floats = (file) => { const bytes = fs.readFileSync(file); return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)); };
const same = (got, want, what) => {
  if (got.length !== want.length) throw new Error(`${what}: ${got.length} values, not ${want.length}`);
  for (let i = 0; i < want.length; i++) if (!Object.is(got[i], want[i])) throw new Error(`${what}: value ${i} is ${got[i]}, not ${want[i]}`);
  checked += 1;
};

// ---- the reading: the model's values are the converter's reader's
const read = {};
for (const name of NAMES) {
  read[name] = floats(`${work}${name}.f32`);
  for (const { blocks, at, raw } of matrices[name]) {
    same(model.values(name, raw, 0, blocks * 256, ROWS, books), read[name].subarray(at, at + ROWS * blocks * 256), `the model's reading of ${name} beside the converter's, rows of ${blocks} blocks`);
  }
}

// ---- the activations of a token: int8 from -127 to 127 with both ends in them, a float32 scale for each 32
const activations = (n, kind) => Int8Array.from({ length: n }, (_, j) => (kind === "high" ? 127 : kind === "low" ? -127 : j % 61 === 7 ? 127 : j % 67 === 11 ? -127 : (next() % 255) - 127));
const scales = (n) => Float32Array.from({ length: n / 32 }, () => Math.fround(1e-3 * (1 + (next() % 4000))));

// where the weights of a model past 4 GiB lie (the 27B's do): every address a kernel takes is above it there
const HIGH = 4 * 2 ** 30 + 2 * 65536;
// Which of V8's compilers runs a kernel changes as it is called (Liftoff first, TurboFan milliseconds later), and the two
// have differed above 4 GiB (T233: Liftoff on arm64 reads v128.load32_splat and load32_lane there at the low 32 bits of
// the address; these kernels use neither, but load64_zero, load32_zero and load64_lane). So the whole check runs three
// times: as V8 pleases, with Liftoff alone, and without it.
const tier = process.env.BLOCKS_TIER ?? "";

let signedSaid = false;
for (const [wide, shared, base] of [[false, false, 0], [false, true, 0], [true, false, 0], [true, true, 0], [true, false, HIGH]]) {
  let memory;
  const pages = Math.ceil((base + 6 * 2 ** 20) / 65536);
  try {
    memory = wide ? new WebAssembly.Memory({ initial: BigInt(pages), ...(shared ? { maximum: BigInt(pages), shared } : {}), address: "i64" })
      : new WebAssembly.Memory({ initial: pages, ...(shared ? { maximum: pages, shared } : {}) });
  } catch (error) {
    console.log(`blocks-check: no ${shared ? "shared " : ""}64-bit memory of ${(base / 2 ** 30).toFixed(1)} GiB and more in this Node: skipped (${error.message})`);
    continue;
  }
  const kind = `${shared ? "shared" : "plain"}${wide ? "64" : ""}`;
  const load = (name) => addressed(new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(built(`${name}_${kind}.wasm`))), { env: { memory } }).exports, wide);
  const k = load("blockkernel"), r = load("blockkernel_relaxed");
  const I = new Int8Array(memory.buffer), U = new Uint8Array(memory.buffer), F = new Float32Array(memory.buffer), N = new Int32Array(memory.buffer),
    H = new Int16Array(memory.buffer);
  const where = `${shared ? "shared, " : ""}${wide ? (base ? "64-bit memory, above 4 GiB" : "64-bit memory") : "32-bit memory"}`;

  // ---- the places: the tables, a matrix, a prompt's frames (a token's activations, then its scales and sums), the
  // outputs, the scratch
  const TOKENS = 16, frame = MOST + MOST / 2 + 96, os = ROWS * 4 + 24;
  const tab = base + 65536, bookAt = tab + 65536, w = bookAt + 65536, xq = w + 65536, xs = xq + MOST, outs = xq + TOKENS * frame + 4096,
    floatsAt = outs + TOKENS * os + 4096, scratch = floatsAt + ROWS * MOST * 4 + 4096, end = scratch + 2 * MOST + 4096;
  if (end > memory.buffer.byteLength) throw new Error("blocks-check's memory is too small");

  // ---- sb_tables
  U.fill(0xa5, tab, tab + model.TABLES + 64);
  I.set(books, bookAt);
  if (k.sb_tables(tab, bookAt) !== model.TABLES) throw new Error(`sb_tables (${where}) says it wrote another number of bytes than ${model.TABLES}`);
  for (let i = 0; i < model.BOOKS.IQ4NL; i++) {
    if (I[tab + i] !== books[i] + (i >= model.BOOKS.IQ1S ? 1 : 0)) throw new Error(`sb_tables (${where}): byte ${i} of the grids is ${I[tab + i]}`);
  }
  for (let i = 0; i < 128; i++) {
    let ones = 0;
    for (let b = 0; b < 7; b++) ones += (i >> b) & 1;
    for (let j = 0; j < 8; j++) {
      const minus = j < 7 ? (i >> j) & 1 : ones & 1;
      if (U[tab + 33792 + 8 * i + j] !== (minus ? 255 : 0)) throw new Error(`sb_tables (${where}): the sign ${j} of index ${i} is ${U[tab + 33792 + 8 * i + j]}`);
    }
  }
  for (let j = 0; j < 16; j++) {
    const value = books[model.BOOKS.IQ4NL + j];
    if (I[tab + 34816 + j] !== value || I[tab + 34832 + j] !== Math.abs(value) || U[tab + 34848 + j] !== (value < 0 ? 255 : 0)) throw new Error(`sb_tables (${where}): IQ4_XS's value ${j}`);
  }
  for (let i = model.TABLES; i < model.TABLES + 64; i++) if (U[tab + i] !== 0xa5) throw new Error(`sb_tables (${where}) wrote past its ${model.TABLES} bytes`);
  checked += 1;

  // ---- whether matmul_sbw_tile may be called here
  const signed = model.signedDots(r.signed_dots, I, N, scratch);
  if (!signedSaid && !tier) console.log(`blocks-check: this engine's relaxed dot product ${signed ? "multiplies signed bytes by signed bytes: matmul_sbw_tile is checked" : "does not multiply signed bytes by signed bytes: matmul_sbw_tile is not called"}`);
  signedSaid = true;

  // a token's frame: its activations at xq + t * frame, and at xs + t * frame their scales, then what sums_x leaves
  const tokens = (n, how) => Array.from({ length: TOKENS }, (_, t) => {
    const a = activations(n, how?.[t]), sx = scales(n);
    I.set(a, xq + t * frame);
    U.fill(0x5a, xs + t * frame, xs + t * frame + n / 2 + 64);
    F.set(sx, (xs + t * frame) / 4);
    k.sums_x(xs + t * frame, xq + t * frame, n);
    return { a, sx };
  });

  // ---- sums_x
  for (const n of [32, 256, 768]) {
    const [{ a, sx }] = tokens(n, ["mixed"]);
    const want = model.sums(a), at = xs + n / 8;
    for (let i = 0; i < want.length; i++) if (H[at / 2 + i] !== want[i]) throw new Error(`sums_x (${where}): sum ${i} of ${n} activations is ${H[at / 2 + i]}, not ${want[i]}`);
    for (let g = 0; g < n / 32; g++) if (F[xs / 4 + g] !== sx[g]) throw new Error(`sums_x (${where}) wrote over a scale`);
    for (let i = 0; i < 64; i++) if (U[xs + n / 2 + i] !== 0x5a) throw new Error(`sums_x (${where}) wrote past the sums of ${n} activations`);
    checked += 1;
  }
  for (const how of ["high", "low"]) {  // the largest sums: 16 x 127
    const [{ a }] = tokens(256, [how]);
    same(H.slice((xs + 32) / 2, (xs + 32) / 2 + 48), model.sums(a), `sums_x (${where}) of activations all ${how}`);
  }

  for (const name of NAMES) {
    const [type, size] = model.TYPES[name];
    for (const { blocks, at, raw } of matrices[name]) {
      const n = blocks * 256, what = `${name}, rows of ${blocks} blocks (${where})`;
      U.set(raw, w);
      U.fill(0xc3, w + raw.length, w + raw.length + 256);

      // ---- sb_row
      for (let i = 0; i < ROWS; i++) {
        F.fill(-7, floatsAt / 4, floatsAt / 4 + n + 16);
        k.sb_row(floatsAt, w + i * blocks * size, tab, n, type);
        same(F.slice(floatsAt / 4, floatsAt / 4 + n), read[name].subarray(at + i * n, at + (i + 1) * n), `sb_row of row ${i} of ${what}`);
        if (F[floatsAt / 4 + n] !== -7) throw new Error(`sb_row of ${what} wrote past its row`);
      }

      // ---- one token: every row, ranges of rows, activations at their ends
      const prompt = tokens(n, ["mixed", "high", "low"]);
      const wanted = prompt.map(({ a, sx }) => model.expected(name, raw, 0, n, ROWS, books, a, sx));
      for (const [kernel, call] of [["matmul_sb", k.matmul_sb], ["matmul_sbr", r.matmul_sbr]]) {
        for (let t = 0; t < 4; t++) {
          for (const [r0, r1] of t ? [[0, ROWS]] : [[0, ROWS], [0, 1], [ROWS - 1, ROWS], [1, 4], [2, 2]]) {
            F.fill(-7, outs / 4, outs / 4 + ROWS + 2);
            call(outs, xq + t * frame, xs + t * frame, w, tab, n, r0, r1, type);
            const want = Float32Array.from({ length: ROWS + 2 }, (_, i) => (i >= r0 && i < r1 ? wanted[t][i] : -7));
            same(F.slice(outs / 4, outs / 4 + ROWS + 2), want, `${kernel} of ${what}, token ${t}, rows ${r0} to ${r1}`);
          }
        }
      }
      // beside the converter's reading, in float64
      {
        const { a, sx } = prompt[0];
        let most = 0;
        const sums = Array.from({ length: ROWS - 2 }, (_, i) => {  // (the random rows: a row of all ones is one sign and float32's own rounding)
          let sum = 0;
          for (let j = 0; j < n; j++) sum += read[name][at + i * n + j] * a[j] * sx[j >> 5];
          most = Math.max(most, Math.abs(sum));
          return sum;
        });
        sums.forEach((sum, i) => {
          if (!(Math.abs(sum - wanted[0][i]) <= 1e-5 * most)) throw new Error(`matmul_sb of ${what}: row ${i} is ${wanted[0][i]}, and the reader's values sum to ${sum}`);
        });
        checked += 1;
      }

      // ---- a prompt's tokens
      const tiles = [["matmul_sbr_tile", (...args) => r.matmul_sbr_tile(...args, type)]];
      if (signed) tiles.push(["matmul_sbw_tile", (...args) => r.matmul_sbw_tile(...args, type, scratch)]);
      const room = n + n / 4 + n / 64;  // the scratch matmul_sbw_tile is given
      for (const [kernel, call] of tiles) {
        for (const count of [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16]) {
          for (const [r0, r1] of count === 5 ? [[0, ROWS], [1, 4], [ROWS - 1, ROWS]] : [[0, ROWS]]) {
            F.fill(-7, outs / 4, (outs + TOKENS * os) / 4);
            U.fill(0x3c, scratch, scratch + room + 64);
            call(outs, xq, xs, w, tab, n, r0, r1, count, os, frame);
            for (let t = 0; t < TOKENS; t++) {
              const want = Float32Array.from({ length: os / 4 }, (_, i) => (t < count && i >= r0 && i < r1 ? wanted[t][i] : -7));
              same(F.slice((outs + t * os) / 4, (outs + (t + 1) * os) / 4), want, `${kernel} of ${what}, ${count} tokens, rows ${r0} to ${r1}: token ${t}`);
            }
            for (let i = 0; i < 64; i++) if (U[scratch + room + i] !== 0x3c) throw new Error(`${kernel} of ${what} wrote past its scratch of ${room} bytes`);
          }
        }
      }
      // the tokens' own vectors are as they were
      prompt.forEach(({ a, sx }, t) => {
        for (let j = 0; j < n; j++) if (I[xq + t * frame + j] !== a[j]) throw new Error(`a kernel of ${what} wrote over token ${t}'s activations`);
        for (let g = 0; g < n / 32; g++) if (F[(xs + t * frame) / 4 + g] !== sx[g]) throw new Error(`a kernel of ${what} wrote over token ${t}'s scales`);
      });
      for (let i = 0; i < 256; i++) if (U[w + raw.length + i] !== 0xc3) throw new Error(`a kernel of ${what} wrote past the weights`);
    }
  }
}
if (!tier) {
  for (const flag of ["--liftoff-only", "--no-liftoff"]) {
    execFileSync(process.execPath, [flag, new URL(import.meta.url).pathname], { stdio: "inherit", env: { ...process.env, BLOCKS_TIER: flag } });
  }
  console.log(`blocks-check: ok (${checked} comparisons to the bit, three times over (V8's tiers as they come, Liftoff alone, TurboFan alone): ${NAMES.length} types, with and without relaxed SIMD, 32-bit and 64-bit memories, plain and shared, above 4 GiB)`);
}
