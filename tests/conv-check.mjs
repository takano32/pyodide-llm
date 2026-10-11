// conv-check.mjs (T260): the kernel of an LFM2's convolution layer (kernels/kernel.ts: short_conv) against the same
// arithmetic written out here and against llama2_numpy.py's own, to the bit. Node and the compiled kernels alone, under
// a second: the light suite runs it.
//   short_conv(out, taps, rows, b, n, count): the newest row is written B * z, and out is C times the sum of taps *
//   rows, the oldest token first. Every number is a float32 product or sum in an order the kernel fixes, which
//   Math.fround follows here and NumPy's float32 arrays follow in short_convolution(): no number may differ. Channels
//   of 1 to 64 (the fours and what is left after them: a real model's 1024 and 2048 are whole fours), 2 to 5 taps, and
//   what the kernel must leave alone (the older rows, what the matrix in gave, the floats past its n channels).
// The arithmetic here is written from the kernel, and a kernel and its copy could be wrong the same way; NumPy's is the
// definition (a native Python: PYTHON, else python3; NumPy is there where pytest is), which transformers agrees with on
// the real model (tests/reference_lfm2.py).
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import { built, python as sources } from "./tree.mjs";

const root = new URL("../", import.meta.url).pathname;
const kernels = (pages) => {
  const memory = new WebAssembly.Memory({ initial: pages });
  const k = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(built("simdkernel_plain.wasm"))), { env: { memory } }).exports;
  return [k, new Float32Array(memory.buffer)];
};
const f = Math.fround;
let seed = 260;
const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) >>> 8;
const random = () => f((next() % 20001) / 10000 - 1);  // -1 .. 1
let checked = 0;

// ---- against the same arithmetic in JavaScript
{
  const [k, F] = kernels(16);
  for (const n of [1, 2, 3, 4, 5, 7, 8, 13, 32, 64]) {
    for (const count of [2, 3, 4, 5]) {
      const taps = 4096, rows = taps + count * n * 4, b = rows + count * n * 4, out = b + 3 * n * 4, end = out + (n + 1) * 4;
      for (let i = taps / 4; i < end / 4; i++) F[i] = random();
      for (let c = 0; c < n; c++) F[(rows + (count - 1) * n * 4) / 4 + c] = -7;  // the newest row: the kernel's to write
      for (let i = 0; i <= n; i++) F[out / 4 + i] = -7;
      const before = F.slice(taps / 4, end / 4);
      const at = (name) => ({ taps: 0, rows: count * n, b: 2 * count * n, out: 2 * count * n + 3 * n })[name];
      const want = before.slice();
      for (let c = 0; c < n; c++) {
        const now = f(before[at("b") + c] * before[at("b") + 2 * n + c]);  // B * z
        want[at("rows") + (count - 1) * n + c] = now;
        let sum = f(before[at("taps") + c] * before[at("rows") + c]);
        for (let j = 1; j < count; j++) {
          const row = j === count - 1 ? now : before[at("rows") + j * n + c];
          sum = f(sum + f(before[at("taps") + j * n + c] * row));
        }
        want[at("out") + c] = f(before[at("b") + n + c] * sum);  // C * the convolution
      }
      k.short_conv(out, taps, rows, b, n, count);
      const got = F.subarray(taps / 4, end / 4);
      for (let i = 0; i < got.length; i++) {
        if (!Object.is(got[i], want[i])) {
          throw new Error(`short_conv differs at float ${i} (${n} channels, ${count} taps): ${got[i]} against ${want[i]}`);
        }
      }
      checked++;
    }
  }
}

// ---- against llama2_numpy.py's own: a made-up layer's state moved on by a few tokens, the bits of every output
const python = `
import json, sys
import numpy as np
sys.path.insert(0, ${JSON.stringify(sources())})
import llama2_numpy as L

rng = np.random.default_rng(260)
cases = []
for dim, taps, tokens in [(1024, 3, 5), (2048, 3, 4), (64, 4, 6), (96, 2, 3), (6, 3, 4), (5, 5, 7)]:
    llama = L.Llama.__new__(L.Llama)
    llama.dim = dim
    llama.conv = rng.uniform(-1, 1, (1, taps, dim)).astype(np.float32)
    llama.conv_state = np.zeros((1, taps - 1, dim), dtype=np.float32)
    # the two matrices of the layer as the identity: what short_convolution() computes between them
    llama.win, llama.wout = [np.eye(3 * dim, dtype=np.float32)], [np.eye(dim, dtype=np.float32)]
    mixed = rng.uniform(-2, 2, (tokens, 3 * dim)).astype(np.float32)
    outs = [llama.short_convolution(0, row) for row in mixed]
    hexes = lambda values: np.ascontiguousarray(values, dtype="<f4").tobytes().hex()
    cases.append(dict(dim=dim, taps=taps, weights=hexes(llama.conv[0]), mixed=[hexes(row) for row in mixed],
                      outs=[hexes(out) for out in outs], state=hexes(llama.conv_state[0])))
print(json.dumps(cases))
`;
const numpy = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c", python], { cwd: root, maxBuffer: 1 << 28 }).toString());
const floats = (hex) => new Float32Array(new Uint8Array(Buffer.from(hex, "hex")).buffer);
let againstNumpy = 0;
{
  const [k, F] = kernels(64);
  for (const c of numpy) {
    const n = c.dim, taps = 4096, rows = taps + c.taps * n * 4, b = rows + c.taps * n * 4, out = b + 3 * n * 4;
    F.set(floats(c.weights), taps / 4);
    F.fill(0, rows / 4, b / 4);  // position 0: no token before it
    c.mixed.forEach((mixed, t) => {
      // forward/engine.js's shortConvolution(): the rows move up by one, the kernel writes this token's
      F.copyWithin(rows / 4, (rows + n * 4) / 4, b / 4);
      F.set(floats(mixed), b / 4);
      k.short_conv(out, taps, rows, b, n, c.taps);
      const want = floats(c.outs[t]), got = F.subarray(out / 4, out / 4 + n);
      for (let i = 0; i < n; i++) {
        if (got[i] !== want[i]) throw new Error(`short_conv differs from NumPy's short_convolution() at channel ${i} of ${n}, token ${t} (${c.taps} taps): ${got[i]} against ${want[i]}`);
      }
      againstNumpy++;
    });
    // and the state after the last token is NumPy's: the last taps - 1 rows
    const state = floats(c.state), held = F.subarray((rows + n * 4) / 4, b / 4);
    for (let i = 0; i < state.length; i++) {
      if (held[i] !== state[i]) throw new Error(`the rows after ${c.mixed.length} tokens differ from NumPy's state at ${i} (${n} channels, ${c.taps} taps)`);
    }
  }
}
console.log(`ok: short_conv to the bit, against JavaScript's arithmetic (${checked} cases) and against NumPy's short_convolution() (${againstNumpy} tokens)`);
