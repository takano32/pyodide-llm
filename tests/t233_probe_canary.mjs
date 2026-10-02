// probe-canary.mjs (T233 review): which loads of a 64-bit memory this engine reads wrongly above 4 GiB, in which tier.
//   node [--liftoff-only | --no-liftoff] .tmp/probe-canary.mjs [shared] [calls]
// A module of one function per operation: f(address: i64) -> i32, the operation's lane 0 (or the scalar). Two values are
// written: at the address (the right one) and at address - 2^32 (where an address that lost its upper 32 bits reads).
const GiB = 2 ** 30, args = process.argv.slice(2);
const shared = args.includes("shared");
const calls = Number(args.find((a) => /^\d+$/.test(a)) ?? 1);
const leb = (n) => { n = BigInt(n); const out = []; do { let b = Number(n & 0x7fn); n >>= 7n; if (n) b |= 0x80; out.push(b); } while (n); return out; };
const vec = (items) => [...leb(items.length), ...items.flat()];
const section = (id, bytes) => [id, ...leb(bytes.length), ...bytes];
const name = (s) => [...leb(s.length), ...Buffer.from(s)];

// every op: [label, the instructions that take local 0 (the i64 address) to an i32, memarg offset]
const OPS = [
  ["i32.load", (o) => [0x28, 2, ...leb(o)]],
  ["f32.load", (o) => [0x2a, 2, ...leb(o), 0xbc]],  // i32.reinterpret_f32 (0xbc)
  ["v128.load", (o) => [0xfd, 0x00, 4, ...leb(o), 0xfd, 0x1b, 0]],
  ["v128.load8_splat", (o) => [0xfd, 0x07, 0, ...leb(o), 0xfd, 0x1b, 0]],
  ["v128.load16_splat", (o) => [0xfd, 0x08, 1, ...leb(o), 0xfd, 0x1b, 0]],
  ["v128.load32_splat", (o) => [0xfd, 0x09, 2, ...leb(o), 0xfd, 0x1b, 0]],
  ["v128.load64_splat", (o) => [0xfd, 0x0a, 3, ...leb(o), 0xfd, 0x1b, 0]],
  ["v128.load32_zero", (o) => [0xfd, 0x5c, 2, ...leb(o), 0xfd, 0x1b, 0]],
  ["v128.load64_zero", (o) => [0xfd, 0x5d, 3, ...leb(o), 0xfd, 0x1b, 0]],
  ["v128.load8x8_s", (o) => [0xfd, 0x01, 3, ...leb(o), 0xfd, 0x1b, 0]],
  ["v128.load32x2_u", (o) => [0xfd, 0x06, 3, ...leb(o), 0xfd, 0x1b, 0]],
];
// load32_lane / load8_lane need a vector first: local.get 0 (address) is on the stack already, then a v128.const
const WITH_VECTOR = [
  ["v128.load32_lane(0)", (o) => [0xfd, 0x0c, ...new Array(16).fill(0), 0xfd, 0x56, 2, ...leb(o), 0, 0xfd, 0x1b, 0]],
  ["v128.load32_lane(1) lane 1", (o) => [0xfd, 0x0c, ...new Array(16).fill(0), 0xfd, 0x56, 2, ...leb(o), 1, 0xfd, 0x1b, 1]],
  ["v128.load8_lane(0)", (o) => [0xfd, 0x0c, ...new Array(16).fill(0), 0xfd, 0x54, 0, ...leb(o), 0, 0xfd, 0x1b, 0]],
];
function build(code, wideShared, max) {
  const type = section(1, vec([[0x60, ...vec([[0x7e]]), ...vec([[0x7f]])]]));
  const limits = wideShared ? [0x07, ...leb(1), ...leb(max)] : [0x04, ...leb(1)];
  const imports = section(2, vec([[...name("env"), ...name("memory"), 0x02, ...limits]]));
  const functions = section(3, vec([[0]]));
  const exports = section(7, vec([[...name("f"), 0x00, 0]]));
  const body = [0, 0x20, 0, ...code, 0x0b];
  const codes = section(10, vec([[...leb(body.length), ...body]]));
  return Uint8Array.from([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, ...type, ...imports, ...functions, ...exports, ...codes]);
}
const high = 4 * GiB + 2 * 65536, at = high + 4096;
const pages = Math.ceil((high + 4 * 2 ** 20) / 65536);
const memory = new WebAssembly.Memory(shared ? { initial: BigInt(pages), maximum: BigInt(pages + 16), shared: true, address: "i64" } : { initial: BigInt(pages), address: "i64" });
const U = new Uint8Array(memory.buffer);
const right = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x01];
const wrong = right.map((b) => 0xff - b);
const at2 = [at, at + 4096];
for (const place of at2) {
  U.set(right, place);
  U.set(wrong, place - 2 ** 32);
}
const view = (place, length) => Buffer.from(U.slice(place, place + length));
console.log(`${process.arch} V8 ${process.versions.v8} ${process.execArgv.join(" ") || "(default flags)"}; ${shared ? "shared" : "non-shared"} 64-bit memory of ${(pages / 16384).toFixed(3)} GiB; ${calls} call(s) each`);
const results = [];
for (const [label, make] of [...OPS, ...WITH_VECTOR]) {
  for (const offset of [0, 4096, 2 ** 32 + 8]) {
    // offset 2^32 + 8 is no address of this memory unless the address register is smaller: base = at - offset
    const base = offset === 2 ** 32 + 8 ? at - offset : offset === 4096 ? at - 4096 : at;
    if (base < 0) continue;
    let instance;
    try {
      instance = new WebAssembly.Instance(new WebAssembly.Module(build(make(offset), shared, pages + 16)), { env: { memory } });
    } catch (error) {
      results.push(`${label} +${offset}: cannot build (${error.message.slice(0, 80)})`);
      continue;
    }
    // what is right: the op at address `at`; computed on a copy of the memory below by the same op at a low address
    const want = (() => {
      // the same bytes at a low place of a memory under 4 GiB
      const small = new WebAssembly.Memory(shared ? { initial: 4n, maximum: 8n, shared: true, address: "i64" } : { initial: 4n, address: "i64" });
      new Uint8Array(small.buffer).set(right, 4096 + 8);
      const low = new WebAssembly.Instance(new WebAssembly.Module(build(make(offset === 2 ** 32 + 8 ? 0 : offset), shared, 8)), { env: { memory: small } }).exports.f;
      return low(BigInt(4096 + 8 - (offset === 2 ** 32 + 8 ? 0 : offset)));
    })();
    const f = instance.exports.f;
    let wrongAt = -1;
    for (let i = 0; i < calls; i++) {
      const got = f(BigInt(base));
      if (got !== want) { wrongAt = i; var last = got; break; }
    }
    results.push(`${label.padEnd(28)} offset ${String(offset).padEnd(11)} ${wrongAt < 0 ? "right" : `WRONG at call ${wrongAt} (got ${(last >>> 0).toString(16)}, want ${(want >>> 0).toString(16)})`}`);
  }
}
console.log(results.join("\n"));
