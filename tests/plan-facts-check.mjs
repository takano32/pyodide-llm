// plan-facts-check.mjs (T.375.2)
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
// T.375.3: and the table of the kinds of layers is open. An LFM2's plan whose convolution layers are said to be of a
// kind forward.ts has not heard of, with that kind written here (outside the product, on what the engine hands a kind:
// forward/kinds.ts's Core) and handed in beside the product's attention: NumPy's logits again, and the engine's own
// rule for a kind that keeps a state (the positions in turn).
//
// For every model: the logits of every token one at a time are NumPy's (within LINE, and the same largest), and the same
// again where the first tokens went through the layers as a prompt's block. T.484: LINE is 1e-5, not forward-check's
// 1e-3 for float32: these made-up models are small, the right answers differ by at most 4.4e-7 (of 33 cases), and a
// fact taken wrong moved them by less than 1e-3 (a RoPE of the wrong layers: AGENTS.md), which 1e-3 let through.
//
//   node tests/plan-facts-check.mjs [--forward <another forward.ts, to see a broken one fail>]      (PYTHON=.venv/bin/python)
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { built, runtime } from "./tree.mjs";

const LINE = 1e-5;
const root = fileURLToPath(new URL("..", import.meta.url));
const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : runtime("forward.js");
const { createForward, LAYER_KINDS, footprint, external, tableHole, placer } = await import(pathToFileURL(forwardFile).href);
const kernels = { plain: new WebAssembly.Module(fs.readFileSync(built("simdkernel_plain.wasm"))), relaxed: null };

fs.mkdirSync(path.join(root, ".tmp"), { recursive: true });
const folder = fs.mkdtempSync(path.join(root, ".tmp", "plan-facts-"));
let failed = 0;
try {
  const cases = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", [path.join(root, "tests/plan_facts.py"), root, folder],
    { maxBuffer: 1 << 28, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).toString());
  const bytes = (name) => new Uint8Array(fs.readFileSync(path.join(folder, name)));
  const PAGE = 65536, BASE = 64;
  // T.375.3: a kind of layer of this file's own: an LFM2's short convolution (llama2_numpy's short_convolution() has the
  // rule), with nothing of forward/convolution.ts: its rows, its array of a token's frame, its step
  const OUTSIDE = "a kind written outside";
  let cleared = 0;
  const outside = {
    // (T.375.5: what it says of a model's sizes, to the engine of the plan and to footprint() of the form: its array of
    // a token's frame, its rows, and which layers of a form are its own)
    frame: (sizes) => ({ arrays: [["gated", 3 * sizes.dim * 4]] }),
    state: (sizes, count) => [count * sizes.numbers.convolution.taps * sizes.dim * 4],
    layers: ({ convolution }, count) => (convolution ? Array.from({ length: count }, (_, l) => convolution.layers[l] === "c") : null),
    make({ plan, sizes, k, room, floats, matrix, at, S, matmuls }, count) {
      const { dim } = plan, { taps } = plan.convolution, bytes = taps * dim * 4, [rows] = outside.state(sizes, count).map((each) => room.alloc(each));
      const win = matrix("win"), wout = matrix("wout"), weights = floats("conv");
      return {
        clear() {
          cleared += 1;
          room.F.fill(0, rows / 4, (rows + count * bytes) / 4);
        },
        layer: (a) => (tokens) => {
          const mine = rows + a * bytes;
          matmuls(at.xb, tokens, [[win, at.gated, S, a]]);
          for (let t = 0; t < tokens; t++) {
            room.F.copyWithin(mine / 4, (mine + dim * 4) / 4, (mine + bytes) / 4);
            k.short_conv(at.xb + t * S, weights + a * bytes, mine, at.gated + t * S, dim, taps);
          }
          matmuls(at.xb, tokens, [[wout, at.xb2, S, a]]);
        },
      };
    },
  };
  const lfm2 = cases.find(({ name }) => name === "lfm2");
  assert.ok(lfm2, "the cases have an LFM2");
  const renamed = { ...lfm2, name: "lfm2, its convolution layers of a kind handed in from outside", kinds: { attention: LAYER_KINDS.attention, [OUTSIDE]: outside },
    plan: { ...lfm2.plan, layers: lfm2.plan.layers.map((layer) => (layer.kind === "convolution" ? { ...layer, kind: OUTSIDE } : layer)) } };
  for (const { name, plan, tokens, checkpoint, logits, facts, kinds } of [...cases, renamed]) {
    const file = bytes(checkpoint), vocab = plan.vocab_size;
    const wanted = new Float32Array(bytes(logits).buffer);
    assert.equal(wanted.length, tokens.length * vocab, `${name}: the reference's logits`);
    assert.deepEqual([plan.layer_norm, plan.gated_ffn, plan.activation], [facts.layer_norm, facts.gated_ffn, facts.activation], `${name}: the plan says the facts the reference computed by`);
    // (T.261: and which layers are local ones, each by its two facts)
    assert.deepEqual(plan.layers.map(({ window, local }, l) => [l, window, local]).filter(([, window, local]) => window || local),
      facts.local.map((l) => [l, facts.window, true]), `${name}: the plan says the local layers the reference computed by`);
    const engine = () => {
      const memory = new WebAssembly.Memory({ initial: Math.ceil((BASE + file.length) / PAGE) + 1 });
      new Uint8Array(memory.buffer).set(file, BASE);
      const made = createForward({ memory, base: BASE, size: file.length, kernels, kinds,
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
    const single = engine(), alone = [];
    tokens.forEach((token, pos) => { single.made.forward(token, pos, true); look(single.out, pos); alone.push(single.out.slice()); });
    single.made.release();
    // T.329.2.2: a table of every layer's inputs not in the memory (the rest of the checkpoint after it as much lower),
    // its rows read from the file: the same logits to the bit, and every row read is the token's
    const hole = tableHole(plan.tensors);
    if (hole) {
      const kept = new Uint8Array([...file.subarray(0, hole[0]), ...file.subarray(hole[1])]), read = [];
      const memory = new WebAssembly.Memory({ initial: Math.ceil((BASE + kept.length) / PAGE) + 1 });
      new Uint8Array(memory.buffer).set(kept, BASE);
      const rows = (offset, length, to) => {
        read.push(offset);
        new Uint8Array(memory.buffer).set(file.subarray(offset, offset + length), to);
      };
      const made = external({ memory, base: BASE, size: file.length, kernels, apart: { place: placer([hole]), stored: kept.length, rows } })
        .start({ ...plan, derived: Object.fromEntries(Object.entries(plan.derived).map(([table, at]) => [table, bytes(at)])) });
      const out = new Float32Array(vocab), table = plan.tensors.per_layer_table, row = table.shape[1] * 4;
      made.bind(out);
      tokens.forEach((token, pos) => {
        made.forward(token, pos, true);
        assert.deepEqual(out, alone[pos], `${name}: the logits with the table read from the file, at position ${pos}`);
      });
      made.release();
      assert.deepEqual(read, tokens.map((token) => table.offset + token * row), `${name}: the rows read from the file are the tokens'`);
    }
    // and the first of them as a prompt's block, the rest after it
    const block = engine(), first = tokens.length - 3;
    block.made.forwardMany(tokens.slice(0, first), 0);
    tokens.slice(first).forEach((token, i) => { block.made.forward(token, first + i, true); look(block.out, first + i); });
    // (T.375.3) a kind that keeps a state is cleared by position 0 and takes the positions in turn, whoever wrote it
    if (kinds) {
      // (T.375.5) footprint() asked with the same table sizes what this engine took: what the kind said of the form is
      // what it said of the plan, to the byte of the product's own convolution kind, and holds what was placed
      const header = [plan.dim, plan.hidden_dim, plan.n_layers, plan.n_heads, plan.n_kv_heads, plan.vocab_size, plan.seq_len];
      const form = { dtype: "float32", arch: plan.arch, head_dim: plan.head_size, convolution: plan.convolution };
      const bound = footprint(header, file.length, form, kinds), placed = block.made.memoryBytes() - BASE - file.length;
      assert.equal(bound, footprint(header, file.length, form), `${name}: footprint() counts the kind handed in as it counts the product's convolution layers`);
      assert.ok(footprint(header, file.length, { ...form, convolution: null }, kinds) > bound, `${name}: and other bytes without the kind's layers (keys and values in their place)`);
      assert.ok(placed <= bound && bound - placed < 2 ** 20 + plan.seq_len * plan.n_layers * 2 * plan.widths.kv * 4, `${name}: the engine placed ${placed} bytes after the checkpoint, footprint() counted ${bound}`);
      assert.equal(cleared, 2, `${name}: position 0 cleared the kind's state, once an engine`);
      assert.throws(() => block.made.forward(tokens[0], 3, true), /position 10 comes next/, `${name}: a position out of turn`);
      block.made.forward(tokens[0], 0, true);
      assert.equal(cleared, 3, `${name}: and position 0 begins again`);
    }
    block.made.release();
    const right = worst <= LINE && same;
    if (!right) failed++;
    console.log(`${right ? "ok" : "WRONG"}: ${name} (${plan.layer_norm ? "LayerNorms" : "RMSNorms"}, ${plan.gated_ffn ? "a gated FFN" : "GPT-2's FFN"}, ` +
      `${plan.layers.map(({ kind, rope }) => (kind === "attention" ? (rope ? "A" : "a") : kind === OUTSIDE ? "o" : kind[0])).join("")}${plan.rotated ? ", rotated" : ""}): forward.ts's logits are within ${worst.toExponential(2)} of NumPy's over ` +
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
  refused({ layers: plan.layers.map((layer, l) => (l === 2 ? { ...layer, kind: "convolution" } : layer)) }, /not the numbers of the layers of its kind/);
  // (a name every object has is no kind of the table's)
  refused({ layers: plan.layers.map((layer, l) => (l === 2 ? { ...layer, kind: "toString" } : layer)) }, /no layer of the kind "toString"/);
  console.log("ok: a plan without its layers' facts, with fewer than the model has layers, or with a kind of layer the engine has no step for is refused as the engine is made");
} finally {
  fs.rmSync(folder, { recursive: true, force: true });
}
console.log(failed ? `plan-facts-check: FAILED (${failed})` : "plan-facts-check: ok");
// (T.575: T.384's stand-still in process.exit() right after heavy work: 2 of 250 runs at once on the development machine,
// one of them in a run of the light suite; a moment first, as worker-fetches-check's)
setTimeout(() => process.stdout.write("", () => process.exit(failed ? 1 : 0)), 200);
