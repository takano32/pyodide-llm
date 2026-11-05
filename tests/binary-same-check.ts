// binary-same-check.ts (T634): the same 1-bit model as the binary dtype and as the ternary one (the signs of every
// weight and one scale of its group of 128, as the 1-bit Bonsai's Q1_0 has them, which the page held as ternary before),
// both on the page's forward pass (src/runtime/forward.ts): the logits are the same to the bit, at every position, a
// token at a time and in blocks (forward_many), with relaxed SIMD and without. kernels/binary.ts says why: a binary
// weight's code is a ternary one's, so every group's integer sum is the same integer and every float32 operation after
// it the same in the same order.
//
//   node tests/binary-same-check.ts <out of tests/make_ternary.py: ternary ... signs> <the same model's: binary> [--positions 64] [--apart]
//
// --apart (T634.2: the 27B, 7.2 and 3.9 GB, which a runner of 16 GB does not hold twice) reads each model in a process of
// its own (this script again, with --side) and compares the logits they leave in files beside the binary checkpoint.
//
// The light suite runs it on tests/make_ternary.py's made-up models (a Qwen3 and a hybrid in a rotated basis).
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { pyodideWithEngine } from "./engine.ts";

const args = process.argv.slice(2);
const [ternary, binary] = args.filter((a, i) => !a.startsWith("--") && !(args[i - 1] ?? "").startsWith("--"));
const positions = args.includes("--positions") ? Number(args[args.indexOf("--positions") + 1]) : 64;
const of = (out) => ({ file: path.resolve(`${out}.bin`), options: JSON.parse(fs.readFileSync(`${out}.json`, "utf8")) });
const valued = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : undefined);
const side = valued("--side");
const CASES = [["relaxed SIMD", []], ["without relaxed SIMD", ["relaxed"]]];
const leftAt = (dtype, c) => `${binary}.logits-${dtype}-${c}`;

// one model with one case: its backend, the tokens, the logits at every position and after a block (null without one)
async function logitsOf(dtype, disable) {
  const given = of(dtype === "ternary" ? ternary : binary);
  // (a 64-bit memory where the ternary checkpoint passes 3 GiB, as the page gives the 1-bit Bonsai 27B)
  const { pyodide: py } = await pyodideWithEngine({ wide: fs.statSync(of(ternary).file).size > 3 * 2 ** 30 });
  py.FS.writeFile("tokenizer.bin", fs.readFileSync(`${ternary}.tokenizer.bin`));
  py.globals.set("GIVEN", py.toPy(given));
  py.globals.set("DISABLE", py.toPy(disable));
  py.globals.set("POSITIONS", positions);
  const [backend, tokens, rows, block] = py.runPython(`
import numpy as np
assert GIVEN["options"]["dtype"] == ${JSON.stringify(dtype)}, GIVEN["options"]["dtype"]
m = kernel_llama_file(GIVEN["file"], open("tokenizer.bin", "rb").read(), **dict(GIVEN["options"], disable=tuple(DISABLE)))
tokens = [m.bos] + np.random.default_rng(634).integers(3, m.vocab_size, POSITIONS - 1).tolist()
rows = np.stack([np.asarray(m.forward(token, pos), dtype=np.float32) for pos, token in enumerate(tokens)])
block = None
if m.forward_many is not None:
    m.forward_many(tokens[:-1], 0)
    block = np.asarray(m.forward(tokens[-1], len(tokens) - 1), dtype=np.float32).tobytes()
(m.backend, tokens, rows.tobytes(), block)
`).toJs();
  return { backend, tokens, rows: new Float32Array(rows.slice().buffer), block: block ? new Float32Array(block.slice().buffer) : null };
}

if (side !== undefined) {
  // a process of --apart's: one model's logits into files
  const c = Number(valued("--case"));
  const got = await logitsOf(side, CASES[c][1]);
  const at = leftAt(side, c);
  fs.writeFileSync(`${at}.json`, JSON.stringify({ backend: got.backend, tokens: got.tokens, block: got.block !== null }));
  fs.writeFileSync(`${at}.bin`, Buffer.from(got.rows.buffer));
  if (got.block) fs.writeFileSync(`${at}.block`, Buffer.from(got.block.buffer));
  process.exit(0);
}

const same = (x, y) => x.length === y.length && new Uint32Array(x.buffer, x.byteOffset, x.length).every((v, i) => v === new Uint32Array(y.buffer, y.byteOffset, y.length)[i]);
const most = (x, y) => x.reduce((m, v, i) => Math.max(m, Math.abs(v - y[i])), 0);
for (const [c, [label, disable]] of CASES.entries()) {
  const got = [];
  for (const dtype of ["ternary", "binary"]) {
    if (args.includes("--apart")) {
      const child = spawnSync(process.execPath, [...process.execArgv, process.argv[1], ternary, binary, "--positions", String(positions), "--side", dtype, "--case", String(c)], { stdio: "inherit" });
      if (child.status !== 0) throw new Error(`binary-same-check: the ${dtype} model's process (${label}) ended with ${child.status ?? child.signal}`);
      const at = leftAt(dtype, c);
      const said = JSON.parse(fs.readFileSync(`${at}.json`, "utf8"));
      const read = (file) => { const bytes = fs.readFileSync(file); return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4); };
      got.push({ ...said, rows: read(`${at}.bin`), block: said.block ? read(`${at}.block`) : null });
      for (const end of [".json", ".bin", ".block"]) fs.rmSync(`${at}${end}`, { force: true });
    } else got.push(await logitsOf(dtype, disable));
  }
  const [t, n] = got;
  if (!n.backend.includes("binary")) throw new Error(`binary-same-check: the binary model ran on ${n.backend}`);
  if (JSON.stringify(t.tokens) !== JSON.stringify(n.tokens)) throw new Error(`binary-same-check: ${label}: the two models read different tokens`);
  const vocabulary = n.rows.length / n.tokens.length;
  for (let pos = 0; pos < n.tokens.length; pos++) {
    const x = t.rows.subarray(pos * vocabulary, (pos + 1) * vocabulary), y = n.rows.subarray(pos * vocabulary, (pos + 1) * vocabulary);
    if (!same(x, y)) throw new Error(`binary-same-check: ${label}: the logits at position ${pos} differ, by ${most(x, y)} at most`);
  }
  const blocks = t.block !== null && n.block !== null;
  if (blocks && !same(t.block, n.block)) throw new Error(`binary-same-check: ${label}: the logits after a block differ, by ${most(t.block, n.block)} at most`);
  const greedy = [...Array(Math.min(8, n.tokens.length)).keys()].map((pos) => { const row = n.rows.subarray(pos * vocabulary, (pos + 1) * vocabulary); return row.reduce((best, v, i) => (v > row[best] ? i : best), 0); });
  console.log(`binary-same-check: ${n.backend} and ${t.backend} (${label}): the logits the same to the bit at ${n.tokens.length} positions${blocks ? ` and after a block of ${n.tokens.length - 1}` : ""} (the most likely at the first eight: ${JSON.stringify(greedy)})${args.includes("--apart") ? ", each model in a process of its own" : ""}`);
}
