// plan-facts-check.mjs (T375.2)
// forward.ts reads what a model's layers are from the facts of Python's plan, and from no name of an architecture:
// its own createForward() on the kernels (simdkernel_plain.wasm), for small made-up float32 models, against the
// reference in NumPy (the native Python: tests/plan_facts.py makes each model, its plan and the reference's logits).
// No Pyodide and no browser: a few seconds, in the light suite, where the comparison of forward.ts with NumPy on the
// site's models (tests/forward-check.mjs) is the full suite's.
//
// The models: the five architectures there are (a Llama with layers RoPE leaves alone, a Qwen2's biases and a Qwen3's
// norms of the heads, a GPT-2, a GPT-NeoX, a Qwen3.5, an LFM2; a Llama and a Qwen3.5 in a rotated basis, whose signs
// are found by the plan's widths), and three whose facts are no architecture's pair: RMSNorms with GPT-2's FFN,
// LayerNorms with a gated FFN (both with a parallel residual), and a Llama no layer of which RoPE turns. Every architecture of the table has LayerNorms with GPT-2's FFN or RMSNorms with a gated one, so an engine
// that read one of the two facts for the other would answer right for all of them; and the two made-up ones have
// names forward.ts has never heard of.
//
// For every model: the logits of every token one at a time are NumPy's (within 1e-3, and the same largest: forward-check's
// line for float32), and the same again where the first tokens went through the layers as a prompt's block.
//
//   node tests/plan-facts-check.mjs [--forward <another forward.ts, to see a broken one fail>]      (PYTHON=.venv/bin/python)
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { built, runtime } from "./tree.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : runtime("forward.js");
const { createForward } = await import(pathToFileURL(forwardFile).href);
const kernels = { plain: new WebAssembly.Module(fs.readFileSync(built("simdkernel_plain.wasm"))), relaxed: null };

fs.mkdirSync(path.join(root, ".tmp"), { recursive: true });
const folder = fs.mkdtempSync(path.join(root, ".tmp", "plan-facts-"));
let failed = 0;
try {
  const cases = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", [path.join(root, "tests/plan_facts.py"), root, folder],
    { maxBuffer: 1 << 28, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).toString());
  const bytes = (name) => new Uint8Array(fs.readFileSync(path.join(folder, name)));
  const PAGE = 65536, BASE = 64;
  for (const { name, plan, tokens, checkpoint, logits, facts } of cases) {
    const file = bytes(checkpoint), vocab = plan.vocab_size;
    const wanted = new Float32Array(bytes(logits).buffer);
    assert.equal(wanted.length, tokens.length * vocab, `${name}: the reference's logits`);
    assert.deepEqual([plan.layer_norm, plan.gated_ffn], [facts.layer_norm, facts.gated_ffn], `${name}: the plan says the facts the reference computed by`);
    const engine = () => {
      const memory = new WebAssembly.Memory({ initial: Math.ceil((BASE + file.length) / PAGE) + 1 });
      new Uint8Array(memory.buffer).set(file, BASE);
      const made = createForward({ memory, base: BASE, size: file.length, kernels,
        plan: { ...plan, derived: Object.fromEntries(Object.entries(plan.derived).map(([table, at]) => [table, bytes(at)])) } });
      const out = new Float32Array(vocab);
      made.bind(out);
      return { made, out };
    };
    // how far forward.ts's logits at a position are from the reference's: [the largest difference, whether the largest logit is the same one]
    const against = (out, pos) => {
      let worst = 0, mine = 0, theirs = 0;
      for (let v = 0; v < vocab; v++) {
        const want = wanted[pos * vocab + v];
        if (!(Math.abs(out[v] - want) <= worst)) worst = Math.abs(out[v] - want);  // (a NaN is the worst there is)
        if (out[v] > out[mine]) mine = v;
        if (want > wanted[pos * vocab + theirs]) theirs = v;
      }
      return [worst, mine === theirs];
    };
    let worst = 0, same = true;
    const look = (out, pos) => {
      const [far, top] = against(out, pos);
      if (!(far <= worst)) worst = far;
      same &&= top;
    };
    // every token on its own
    const single = engine();
    tokens.forEach((token, pos) => { single.made.forward(token, pos, true); look(single.out, pos); });
    single.made.release();
    // and the first of them as a prompt's block, the rest after it
    const block = engine(), first = tokens.length - 3;
    block.made.forwardMany(tokens.slice(0, first), 0);
    tokens.slice(first).forEach((token, i) => { block.made.forward(token, first + i, true); look(block.out, first + i); });
    block.made.release();
    const right = worst <= 1e-3 && same;
    if (!right) failed++;
    console.log(`${right ? "ok" : "WRONG"}: ${name} (${plan.layer_norm ? "LayerNorms" : "RMSNorms"}, ${plan.gated_ffn ? "a gated FFN" : "GPT-2's FFN"}, ` +
      `${plan.layers.map(({ kind, rope }) => (kind === "attention" ? (rope ? "A" : "a") : kind[0])).join("")}${plan.rotated ? ", rotated" : ""}): forward.ts's logits are within ${worst.toExponential(2)} of NumPy's over ` +
      `${tokens.length} tokens, one at a time and after a block${same ? "" : ", and NOT the same largest one"}`);
  }
  // a plan that says nothing of its layers, or of a kind of layer this engine has no step for, is refused as the engine is made
  const { plan, checkpoint } = cases[0], file = bytes(checkpoint);
  const refused = (changed, words) => {
    const memory = new WebAssembly.Memory({ initial: Math.ceil((BASE + file.length) / PAGE) + 1 });
    assert.throws(() => createForward({ memory, base: BASE, size: file.length, kernels, plan: { ...plan, derived: {}, ...changed } }), words);
  };
  refused({ layers: undefined }, /the facts of no layers/);
  refused({ layers: plan.layers.slice(1) }, /the facts of 3 layers, and the model has 4/);
  refused({ layers: plan.layers.map((layer, l) => (l === 2 ? { ...layer, kind: "recurrent" } : layer)) }, /no layer of the kind "recurrent"/);
  refused({ layers: plan.layers.map((layer, l) => (l === 2 ? { ...layer, kind: "linear" } : layer)) }, /not the numbers of the layers of its kind/);
  console.log("ok: a plan without its layers' facts, with fewer than the model has layers, or with a kind of layer the engine has no step for is refused as the engine is made");
} finally {
  fs.rmSync(folder, { recursive: true, force: true });
}
console.log(failed ? `plan-facts-check: FAILED (${failed})` : "plan-facts-check: ok");
process.exit(failed ? 1 : 0);
