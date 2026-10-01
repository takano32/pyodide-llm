// rotate-check.mjs (T237): the two kernels of the rotated basis (kernels/kernel.ts: rotate, unrotate) against the
// same arithmetic written out here in the order llama2_numpy.hadamard() computes in (times 1 / sqrt(block), then the
// sums and differences of values 1, 2, 4, ... apart), to the bit: every number is one float32 product, sum or
// difference, which Math.fround follows. Node and the compiled kernels alone, under a second: the light suite runs it
// (tests/smoke.mjs holds the kernels to NumPy itself, on Pyodide).
//   every block from 1 to 4096, one, two, three and five blocks of it (and lengths that are no multiple of four, for
//   blocks of 1 and 2), into another place and in place, on a 32-bit and on a 64-bit memory; nothing is written past
//   the n values, and the input of a call that writes elsewhere is left as it was.
// It also says how long a block of 1024 takes (T237: one token of the 27B turns 2,122 of them).
import fs from "node:fs";

const root = new URL("../", import.meta.url).pathname;
const f = Math.fround;
let seed = 237;
const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
const random = () => f((next() % 20001) / 10000 - 1);

// the transform of block values of v from at, in place: llama2_numpy.hadamard()'s passes
function butterflies(v, at, block) {
  for (let half = 1; half < block; half *= 2) {
    for (let i = 0; i < block; i += 2 * half) {
      for (let j = 0; j < half; j++) {
        const a = v[at + i + j], b = v[at + i + half + j];
        v[at + i + j] = f(a + b);
        v[at + i + half + j] = f(a - b);
      }
    }
  }
}
function rotated(x, signs, block) {
  const scale = f(1 / Math.sqrt(block));
  const out = Float32Array.from(x, (value, i) => f(f(value * signs[i]) * scale));
  for (let b = 0; b < x.length; b += block) butterflies(out, b, block);
  return out;
}
function unrotated(z, signs, block) {
  const scale = f(1 / Math.sqrt(block));
  const out = Float32Array.from(z, (value) => f(value * scale));
  for (let b = 0; b < z.length; b += block) butterflies(out, b, block);
  return out.map((value, i) => f(value * signs[i]));
}

let checked = 0;
for (const [file, wide] of [["simdkernel_plain.wasm", false], ["simdkernel_plain64.wasm", true]]) {
  if (!fs.existsSync(`${root}public/${file}`)) throw new Error(`no public/${file}: make kernels`);
  let memory;
  try {
    memory = new WebAssembly.Memory(wide ? { initial: 8n, address: "i64" } : { initial: 8 });
  } catch {
    console.log("rotate-check: no 64-bit memory in this Node, the 32-bit kernels alone");
    continue;
  }
  const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(`${root}public/${file}`)), { env: { memory } }).exports;
  const at = (address) => (wide ? BigInt(address) : address);
  const F = new Float32Array(memory.buffer), bits = new Uint32Array(memory.buffer);
  for (let block = 1; block <= 4096; block *= 2) {
    const lengths = [1, 2, 3, 5].map((blocks) => blocks * block).concat(block < 4 ? [7 * block, 9 * block] : []);
    for (const n of lengths) {
      const X = 4096, S = X + (n + 4) * 4, OUT = S + (n + 4) * 4;
      const scale = f(1 / Math.sqrt(block));
      const x = Float32Array.from({ length: n }, random);
      const signs = Float32Array.from({ length: n }, () => (next() & 1 ? 1 : -1));
      for (const [name, kernel, want] of [["rotate", k.rotate, rotated(x, signs, block)], ["unrotate", k.unrotate, unrotated(x, signs, block)]]) {
        for (const inPlace of [false, true]) {
          F.fill(-7, X / 4, (OUT + (n + 4) * 4) / 4);
          F.set(x, X / 4);
          F.set(signs.map((sign) => f(sign * scale)), S / 4);
          const out = inPlace ? X : OUT;
          kernel(at(out), at(X), at(S), n, block);
          const expected = new Uint32Array(want.buffer);
          for (let i = 0; i < n; i++) {
            if (bits[out / 4 + i] !== expected[i]) {
              throw new Error(`${name} differs at value ${i} (block ${block}, ${n} values${inPlace ? ", in place" : ""}, ${file}): ${F[out / 4 + i]} against ${want[i]}`);
            }
          }
          for (let i = n; i < n + 4; i++) if (F[out / 4 + i] !== -7) throw new Error(`${name} wrote past its ${n} values (block ${block})`);
          if (!inPlace) for (let i = 0; i < n; i++) if (F[X / 4 + i] !== x[i]) throw new Error(`${name} changed its input (block ${block})`);
          checked += 1;
        }
      }
      // and one undoes the other: R^-1 R x is x again, to the rounding of the sums
      F.set(x, X / 4);
      F.set(signs.map((sign) => f(sign * scale)), S / 4);
      k.rotate(at(OUT), at(X), at(S), n, block);
      k.unrotate(at(OUT), at(OUT), at(S), n, block);
      for (let i = 0; i < n; i++) {
        if (Math.abs(F[OUT / 4 + i] - x[i]) > 2e-5) throw new Error(`unrotate does not undo rotate (block ${block}, value ${i})`);
      }
    }
  }
  if (!wide) {
    // the time of a block of 1024 (5 blocks: a vector of the 27B's residual stream), the fastest of 20 rounds of 2000
    const n = 5120, X = 4096, S = X + n * 4, OUT = S + n * 4;
    F.set(Float32Array.from({ length: n }, random), X / 4);
    F.fill(1 / 32, S / 4, S / 4 + n);
    let best = Infinity;
    for (let round = 0; round < 20; round++) {
      const began = performance.now();
      for (let i = 0; i < 2000; i++) k.rotate(OUT, X, S, n, 1024);
      best = Math.min(best, (performance.now() - began) / 2000);
    }
    console.log(`rotate-check: rotate of 5 blocks of 1024 takes ${(best * 1000).toFixed(1)} µs ` +
      `(${(best * 1e6 / n).toFixed(2)} ns a value, ${(best * 1e6 / 5).toFixed(0)} ns a block)`);
  }
}
console.log(`ok: rotate and unrotate are the same float32 arithmetic as llama2_numpy.hadamard(), to the bit (${checked} calls)`);
