// T.244: keys and values that were no finite numbers in a float16 cache. The kernels' to_f16 writes a NaN or an infinity
// of float32's as an infinity, which the reads (halves4, half) take for 65536: a float32 cache keeps the NaN, and the
// logits of every token that reads it are not finite (T.195's NOT_FINITE stops there), where a float16 one's were. So
// to_f16 says how many it was handed, the cache keeps the first position they were at, and the logits of a token at or
// after it are made NaN until that position is written anew. Node and the plain kernels alone, a second or two:
//   - to_f16 counts a NaN, +inf and -inf of float32's, and nothing of a finite number however large (70000, which it
//     writes as an infinity too: a float32 cache keeps it finite, and so do the logits);
//   - forward.js's engine (a made-up llama, int8, every weight 0: every logit 0) on the real kernels, where the kernels'
//     to_f16 is handed a NaN key at a chosen position (wrap): the tokens before it finite, it and those after it NaN, one
//     at a time and in a prompt's block; the position written anew from it (or from before it) finite again, from
//     after it not; a float32 cache (the NaN is not made here: nothing is asked of to_f16) none of this.
// The GPU's steps that would take the CPU's keys and values up from there are refused: tests/gpu-part-check.ts.
//   node tests/unfinite-check.ts     (PYTHON=.venv/bin/python)
import assert from "node:assert/strict";
import fs from "node:fs";
import { plans } from "./plans.ts";
import { built, runtime } from "./tree.ts";

const { compileKernels, createForward } = await import(runtime("forward.js"));
const { plansOf, planOf, FORM } = plans(new URL("..", import.meta.url));
const PAGE = 65536;
const kernels = compileKernels(fs.readFileSync(built("simdkernel_plain.wasm")), fs.readFileSync(built("simdkernel_relaxed_plain.wasm")));

// ---- the kernel: what it counts
{
  const memory = new WebAssembly.Memory({ initial: 1 });
  const k = new WebAssembly.Instance(kernels.plain, { env: { memory } }).exports;
  const F = new Float32Array(memory.buffer);
  const counted = (values) => {
    F.set(values, 0);
    return k.to_f16(4096, 0, values.length);
  };
  assert.equal(counted([1, -2, 0.5, 65504, -65504, 70000, -1e30, 1e-30]), 0, "finite numbers, the too large among them: none");
  assert.equal(counted([NaN, 1, Infinity, -Infinity, 70000]), 3, "a NaN, +inf and -inf of float32's");
  assert.equal(counted([]), 0);
}
console.log("unfinite-check: to_f16 counts the NaNs and infinities of float32's it is handed, and no finite number");

// ---- the engine
const [p] = plansOf([{ header: [64, 128, 2, 4, 4, 64, 32], form: FORM, dtype: "int8" }]);
function engine({ half, bad }) {
  const memory = new WebAssembly.Memory({ initial: Math.ceil((64 + p.size) / PAGE) + 64 });
  // (the key handed to to_f16 at the bad position's: NaN, as a broken weight or an overflow would make it)
  let pos = -1;
  const wrap = (exports) => ({ ...exports, to_f16(out, x, n) {
    if (pos === bad.at) new Float32Array(memory.buffer)[x / 4] = NaN;
    return exports.to_f16(out, x, n);
  } });
  const made = createForward({ memory, base: 64, size: p.size, kernels, plan: planOf(p), halfKeys: half, wrap });
  const logits = () => [...made.logits()];
  return {
    one(at) { pos = at; made.forward(1, at); pos = -1; },
    finite: () => logits().every(Number.isFinite),
    unfinite: () => logits().every(Number.isNaN),
  };
}

// one token at a time
{
  const bad = { at: 2 };
  const e = engine({ half: true, bad });
  const seen = [0, 1, 2, 3, 4].map((at) => (e.one(at), e.finite() ? "finite" : e.unfinite() ? "NaN" : "mixed"));
  assert.deepEqual(seen, ["finite", "finite", "NaN", "NaN", "NaN"], "the logits from the position of the NaN on");
  bad.at = -1;
  e.one(3);
  assert.ok(e.unfinite(), "written anew from after it: the NaN is still read");
  e.one(2);
  assert.ok(e.finite(), "written anew from it: finite again");
  e.one(3);
  assert.ok(e.finite());
  bad.at = 3;
  e.one(3);
  bad.at = -1;
  e.one(0);
  assert.ok(e.finite(), "a generation begun again from position 0");
  e.one(1);
  assert.ok(e.finite());
}
console.log("unfinite-check: a float16 cache's NaN makes the logits of its position and after it NaN, until it is written anew");

// a prompt's block, then a token
{
  const memory = new WebAssembly.Memory({ initial: Math.ceil((64 + p.size) / PAGE) + 64 });
  let calls = 0;
  // (the 2 layers put a key and a value of each of a block's tokens in turn: the third key of the first layer is
  // position 2's)
  const wrap = (exports) => ({ ...exports, to_f16(out, x, n) {
    if (calls++ === 4) new Float32Array(memory.buffer)[x / 4] = NaN;
    return exports.to_f16(out, x, n);
  } });
  const made = createForward({ memory, base: 64, size: p.size, kernels, plan: planOf(p), halfKeys: true, wrap });
  made.forwardMany([1, 2, 3, 4], 0);
  made.forward(5, 4);
  assert.ok([...made.logits()].every(Number.isNaN), "a NaN put in a prompt's block: the token after it");
  made.forwardMany([1, 2, 3], 0);
  made.forward(4, 3);
  assert.ok([...made.logits()].every(Number.isFinite), "the prompt again from position 0: finite");
}
console.log("unfinite-check: and in a prompt's block, read by the token after it");

// a float32 cache: to_f16 is not asked, and the engine makes nothing NaN
{
  const memory = new WebAssembly.Memory({ initial: Math.ceil((64 + p.size) / PAGE) + 64 });
  let asked = 0;
  const wrap = (exports) => ({ ...exports, to_f16(out, x, n) { asked += 1; return exports.to_f16(out, x, n); } });
  const made = createForward({ memory, base: 64, size: p.size, kernels, plan: planOf(p), halfKeys: false, wrap });
  for (let at = 0; at < 4; at++) made.forward(1, at);
  assert.equal(asked, 0, "a float32 cache asks nothing of to_f16");
  assert.ok([...made.logits()].every(Number.isFinite));
}
console.log("unfinite-check: a float32 cache: nothing of it");
