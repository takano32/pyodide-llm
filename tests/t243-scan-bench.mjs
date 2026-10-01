// T243: the cost of scanning the float16 keys and values of a request for a NaN or an infinity, in JavaScript
import os from "node:os";
const shapes = [["llm-jp-3.1 1.8B / Pythia 1.4B, a block of 64", 64, 24, 2048], ["the same, 4 steps", 4, 24, 2048],
  ["llm-jp-3 150M, a block of 64", 64, 12, 512], ["Llama 3.2 1B, a block of 64", 64, 16, 512]];
const memory = new WebAssembly.Memory({ initial: 1024, maximum: 1024, shared: true });
const H = new Uint16Array(memory.buffer), W = new Uint32Array(memory.buffer);
const halfBits = () => ((Math.random() < 0.5 ? 0x8000 : 0) | ((10 + ((Math.random() * 8) | 0)) << 10) | ((Math.random() * 1024) | 0));
for (let i = 0; i < 16 << 20; i++) H[i] = halfBits();
const each = (at, n) => { for (let i = at, end = at + n; i < end; i++) if ((H[i] & 0x7c00) === 0x7c00) return true; return false; };
const pairs = (at, n) => { let bad = 0; for (let i = at >> 1, end = (at + n) >> 1; i < end; i++) bad |= ((W[i] & 0x7c007c00) + 0x04000400) & 0x80008000; return bad !== 0; };
const median = (f) => { const t = []; for (let r = 0; r < 15; r++) { const b = performance.now(); f(); t.push(performance.now() - b); } return t.sort((a, b) => a - b)[7]; };
console.log(`${os.cpus()[0].model || "unknown CPU"}, ${os.arch()}, node ${process.version}, load ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}`);
for (const [name, count, layers, kvDim] of shapes) {
  const stride = 64 * kvDim, words = count * layers * 2 * kvDim;
  const scan = (f) => () => { let bad = false; for (let s = 0; s < 2 * layers; s++) bad = f(s * stride, count * kvDim) || bad; if (bad) throw new Error("bad"); };
  scan(each)(); scan(pairs)();
  console.log(`${name}: ${words} words; one at a time ${median(scan(each)).toFixed(3)} ms, two at a time ${median(scan(pairs)).toFixed(3)} ms`);
}
// both find each of the three, in either half of a 32-bit word
for (const bits of [0x7c00, 0xfc00, 0x7e00, 0x7fff, 0xffff]) for (const at of [1000, 1001]) {
  const was = H[at]; H[at] = bits;
  if (!each(992, 16) || !pairs(992, 16)) throw new Error(`missed ${bits.toString(16)} at ${at}`);
  H[at] = was;
}
for (const bits of [0x7bff, 0xfbff, 0x0000, 0x8000, 0x03ff, 0x7800]) { H.fill(bits, 992, 1008); if (each(992, 16) || pairs(992, 16)) throw new Error(`took ${bits.toString(16)} as not finite`); }
console.log("both scans find 7c00 fc00 7e00 7fff ffff in either half of a word, and pass 7bff fbff 0 8000 3ff 7800");
// the kernel's scan (finite_f16) and today's write-back of the same words (from_f16 into a float32 cache, copyWithin into a float16 one)
const fs = await import("node:fs");
const imports = { env: { memory, abort: () => { throw new Error("abort"); } } };
const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync("public/simdkernel_shared.wasm")), imports).exports;
const U8 = new Uint8Array(memory.buffer);
for (let i = 0; i < 16 << 20; i++) H[i] = halfBits();
for (const [name, count, layers, kvDim] of shapes) {
  const stride = 64 * kvDim * 2, n = count * kvDim, parts = 2 * layers, cacheAt = 36 << 20;
  const kernel = () => { let fine = 1; for (let s = 0; s < parts; s++) fine &= k.finite_f16(s * stride, n); if (!fine) throw new Error("bad"); };
  const widen = () => { for (let s = 0; s < parts; s++) for (let t = 0; t < count; t++) k.from_f16(cacheAt + (s * count + t) * kvDim * 4, s * stride + t * kvDim * 2, kvDim); };
  const copy = () => { for (let s = 0; s < parts; s++) for (let t = 0; t < count; t++) U8.copyWithin(cacheAt + (s * count + t) * kvDim * 2, s * stride + t * kvDim * 2, s * stride + (t + 1) * kvDim * 2); };
  kernel();
  console.log(`${name}: the kernel's scan ${median(kernel).toFixed(3)} ms; from_f16 of the same ${median(widen).toFixed(3)} ms, copyWithin of the same ${median(copy).toFixed(3)} ms`);
}
for (const bits of [0x7c00, 0xfc00, 0x7e00, 0xffff]) for (const at of [0, 7, 8, 1000, 1023, 1024, 1026]) {
  H.fill(0x3c00, 0, 1040); H[at] = bits;
  if (k.finite_f16(0, 1027) !== 0) throw new Error(`the kernel missed ${bits.toString(16)} at ${at}`);
  if (k.finite_f16(0, at) !== 1 || k.finite_f16((at + 1) * 2, 1027 - at - 1) !== 1) throw new Error(`the kernel read past its words at ${at}`);
}
console.log("the kernel finds 7c00 fc00 7e00 ffff in the eights and in the rest, and reads no word outside its n");
console.log(`load ${os.loadavg().map((l) => l.toFixed(2)).join(" ")}`);
