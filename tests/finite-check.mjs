// finite-check.mjs (T243's review): finite_f16 (kernels/kernel.ts), the look forward.js takes at the float16 keys and
// values a GPU wrote back before any of them goes into the cache (stagingFinite): 1 where all n halves are finite
// numbers, 0 where one has every bit of its exponent set (a NaN or an infinity, of either sign), which the kernels'
// halves4 and half would read as a finite number (65536 and more). Node and the compiled kernels alone, a few seconds:
// the light suite runs it (tests/smoke.mjs holds the Emscripten build to NumPy's isfinite; this is the four builds
// forward.js and the software threads instantiate, called as forward.js calls them).
//   - every one of the 65536 halves alone, at every place in a 16-byte line (the loads are not aligned), against
//     Float16Array's own Number.isFinite;
//   - runs of 0 to 70 halves and of the sizes where the eights end and begin again (127 to 129, 255 to 257, ...), at four
//     byte offsets, a bad half at every place of a short run and at the edges of a long one, of each of six kinds (+inf,
//     -inf, a quiet NaN, a NaN of the smallest mantissa, one of every bit, a negative NaN); and halves right before and
//     after the run that are all bad, which a look past its n would find: none is to be (n = 0: always 1);
//   - the blocks a prompt brings back (786,432 halves: llm-jp-3 150M's 64 tokens; 6,291,456: llm-jp-3.1 1.8B's) and the
//     rest after the eights (n + 1 to n + 7: every real kvDim is a multiple of 8, so no run of the engine's has one, and
//     a kernel that left the rest unlooked at would pass forward.js's checks), a bad half at the first and the last
//     places, at each end of the last parts of eight, and at random;
//   - on a 64-bit memory the same, with the address as forward.js hands it (a Number that the wrapper makes a BigInt:
//     jobs.js's ADDRESSES), and, where this Node can make one, an address above 4 GiB.
//   node tests/finite-check.mjs [--kernels <the folder of the .wasm files, public/ if not said>]
//   (--kernels is for the broken builds a review makes: kernels/build.py's plain_module() into a folder of its own)
import fs from "node:fs";
import path from "node:path";
import { built as kernelsFolder, runtimeUrl } from "./tree.mjs";
const { addressed } = await import(runtimeUrl("jobs.js"));

const root = new URL("../", import.meta.url).pathname;
const args = process.argv.slice(2);
const folder = path.resolve(args.includes("--kernels") ? args[args.indexOf("--kernels") + 1] : kernelsFolder());
const BAD = [0x7c00, 0xfc00, 0x7e00, 0x7c01, 0xffff, 0xfe01];
const names = { 0x7c00: "+inf", 0xfc00: "-inf", 0x7e00: "a quiet NaN", 0x7c01: "a NaN of the smallest mantissa", 0xffff: "a NaN of every bit", 0xfe01: "a negative NaN" };
const finiteHalf = (h) => (h & 0x7c00) !== 0x7c00;
let seed = 243;
const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
// a finite half of any size (the largest of either sign, zeros and subnormals among them) from a 16-bit draw
const finite = () => {
  const draw = (next() ^ (next() << 5)) & 0xffff;
  return finiteHalf(draw) ? draw : draw ^ 0x4000;  // (every bit of the exponent is set: bit 14 cleared, the exponent is 15)
};
const PAGES = 256;  // 16 MiB: the 12.6 MB of a block of 1.8B's with room round it
let checked = 0, built = 0;
const fail = (message) => {
  console.error(`finite-check: ${message}`);
  process.exit(1);
};

for (const [build, shared, wide] of [["plain", false, false], ["shared", true, false], ["plain64", false, true], ["shared64", true, true]]) {
  const file = `${folder}/simdkernel_${build}.wasm`;
  if (!fs.existsSync(file)) fail(`no ${file}: make kernels`);
  let memory, exports;
  try {
    memory = new WebAssembly.Memory({
      initial: wide ? BigInt(PAGES) : PAGES, ...(shared ? { maximum: wide ? BigInt(PAGES) : PAGES, shared: true } : {}), ...(wide ? { address: "i64" } : {}),
    });
    exports = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(file)), { env: { memory } }).exports;
  } catch (error) {
    if (!wide) throw error;
    console.log(`finite-check: no 64-bit memory for ${build} in this Node: skipped (${error.message})`);
    continue;
  }
  if (typeof exports.finite_f16 !== "function") fail(`${build}: the kernels have no finite_f16`);
  const k = addressed(exports, wide);
  const H = new Uint16Array(memory.buffer);
  const call = (at, n) => k.finite_f16(at, n);
  const where = (what) => `${build}: ${what}`;
  built++;

  // ---- every half alone, at every place of a 16-byte line
  const oracle = new Float16Array(1), as = new Uint16Array(oracle.buffer);
  for (let offset = 0; offset < 16; offset += 2) {
    const at = 4096 + offset;
    for (let h = 0; h < 65536; h++) {
      H[at / 2] = h;
      if (offset === 0) {
        as[0] = h;
        if (finiteHalf(h) !== Number.isFinite(oracle[0])) fail(`this check's own test of a half is wrong for ${h.toString(16)}`);
      }
      const got = call(at, 1);
      if (got !== (finiteHalf(h) ? 1 : 0)) fail(where(`the half ${h.toString(16)} alone at byte ${offset}: ${got}`));
      checked++;
    }
  }

  // ---- runs: no half outside [at, at + n) looked at, a bad half inside it found wherever it is
  const sizes = [...Array(71).keys(), 127, 128, 129, 255, 256, 257, 1000, 1023, 1024, 1025, 4095, 4096, 4097];
  for (const n of sizes) {
    for (const offset of [0, 2, 6, 14]) {
      const at = 8192 + offset;
      const fillRun = () => {
        for (let i = 0; i < n; i++) H[at / 2 + i] = finite();
        for (let i = 1; i <= 20; i++) {  // all bad, right before and right after
          H[at / 2 - i] = BAD[i % BAD.length];
          H[at / 2 + n + i - 1] = BAD[(i + 3) % BAD.length];
        }
      };
      fillRun();
      if (call(at, n) !== 1) fail(where(`a run of ${n} finite halves at byte ${offset} with bad ones around it: ${call(at, n)} (a look past the run?)`));
      checked++;
      const places = n <= 40 ? [...Array(n).keys()] : [0, 1, 6, 7, 8, 9, n - 9, n - 8, n - 7, n - 2, n - 1, next() % n, next() % n, next() % n];
      for (const p of new Set(places)) {
        for (const bad of BAD) {
          H[at / 2 + p] = bad;
          if (call(at, n) !== 0) fail(where(`${names[bad]} at place ${p} of a run of ${n} at byte ${offset} was not found`));
          fillRun();
          checked++;
        }
      }
    }
  }

  // ---- the blocks of a prompt, and what is left after their eights (and bad halves where the block ends, which are to
  // be left alone)
  const at = 4096;
  for (const words of [786432, 6291456]) {
    for (let i = 0; i < words + 16; i++) H[at / 2 + i] = finite();
    for (let rest = 0; rest < 8; rest++) {
      const n = words + rest;
      for (let i = n; i < n + 8; i++) H[at / 2 + i] = BAD[i % BAD.length];
      if (call(at, n) !== 1) fail(where(`a block of ${n} finite halves with bad ones just after it: ${call(at, n)} (a look past the block?)`));
      for (let i = n; i < n + 8; i++) H[at / 2 + i] = finite();
      if (call(at, n) !== 1) fail(where(`a block of ${n} finite halves: ${call(at, n)}`));
      checked += 2;
      const places = new Set([0, 1, 7, 8, n - 1, n - 2, n - 8, n - 9, words - 1, words, next() % n, next() % n, next() % n].filter((p) => p >= 0 && p < n));
      for (let i = 0; i < rest; i++) places.add(words + i);  // every place of the rest
      for (const p of places) {
        const kept = H[at / 2 + p];
        for (const bad of [BAD[p % BAD.length], BAD[(p + 2) % BAD.length]]) {
          H[at / 2 + p] = bad;
          if (call(at, n) !== 0) fail(where(`${names[bad]} at place ${p} of a block of ${n} (${rest} after the eights) was not found`));
          checked++;
        }
        H[at / 2 + p] = kept;
      }
    }
  }
}
// ---- an address above 4 GiB, on a 64-bit memory (forward.js's staging is there where the checkpoint takes the room below)
{
  const high = 4 * 2 ** 30 + 2 * 65536, pages = high / 65536 + 4;
  let memory;
  try {
    memory = new WebAssembly.Memory({ initial: BigInt(pages), address: "i64" });
  } catch (error) {
    console.log(`finite-check: no 64-bit memory above 4 GiB in this Node: skipped (${error.message})`);
  }
  if (memory) {
    const exports = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${folder}/simdkernel_plain64.wasm`)), { env: { memory } }).exports;
    const k = addressed(exports, true);
    const H = new Uint16Array(memory.buffer, high, 64);
    for (let n = 0; n <= 40; n++) {
      for (let i = 0; i < 64; i++) H[i] = 0x3c00;
      if (k.finite_f16(high, n) !== 1) fail(`above 4 GiB: a run of ${n} finite halves`);
      for (let p = 0; p < n; p++) {
        H[p] = 0x7c01;
        if (k.finite_f16(high, n) !== 0) fail(`above 4 GiB: a NaN at place ${p} of a run of ${n} was not found`);
        H[p] = 0x3c00;
        checked += 2;
      }
    }
  }
}
console.log(`ok: finite_f16 finds a NaN or an infinity at every place and looks at nothing outside its n, in the ${built} builds (${checked} calls)`);
