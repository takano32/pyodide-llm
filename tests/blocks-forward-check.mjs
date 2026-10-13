// blocks-forward-check.mjs (T363.5)
// The page's forward pass on models whose matrices are kept as llama.cpp's blocks (a type for each layer of a stack:
// Underdog Saluki 27B's shape in small, and blocks beside rows of other kinds), on the real kernels: simdkernel_* and,
// on the same memory, blockkernel_*. No Pyodide and no browser: the native Python makes the models, their plans (as
// Llama(external=) hands them to forward.ts) and the reference's logits (tests/blocks_forward.py), and Node runs
// forward.ts's createForward() on them.
//
// For every model:
//   - every call of a kernel of the blocks is held to the plan and to the model of the kernels (tests/blocks-model.mjs,
//     which tests/blocks-check.mjs holds to the converter's readers): the matrix is a layer of a stack the plan has
//     where the plan has it, of the type the plan says, with the engine's tables; and the product is the model's of the
//     activations the kernel was handed, to the bit. Every layer of every stack of blocks is multiplied for every
//     token, and a token's row of the embedding is the model's reading of it. (This is the tight check: the engine
//     rounds the activations to 8 bits, and what one rounding differs by is in every logit of a deep model, so that
//     no reference that rounds alike stays nearer the logits than about a hundredth of the largest: measured.)
//   - the logits are within LOOSE of the engine in NumPy's (float32 activations), and no nearer than that to it than
//     NumPy's with the activations rounded as the kernels round them is (said, not held).
//   - a plan that is wrong in a way no refusal catches fails one of the two: a layer taken for another type and the
//     embedding taken for another (the calls), the output matrix read in the other order of the value heads and the
//     codebooks read a byte off (the logits, several times LOOSE).
//   - every other way the engine runs gives the same logits to the bit: without relaxed SIMD, with the tile of the
//     signed dot product and without, a prompt's block before the last tokens, a 64-bit memory, a shared memory with
//     1 to 4 software threads (a block and each tile there too), and a thread that stops in the middle of its chunk.
//   - footprint() counts what the engine placed after the checkpoint, of the header, the form and the weights Python
//     counted.
//
//   node tests/blocks-forward-check.mjs [--forward <another forward.ts, to see a broken one fail>]      (PYTHON=.venv/bin/python)
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { Worker } from "node:worker_threads";
import { fileURLToPath, pathToFileURL } from "node:url";
import { softwareThread } from "./helper-node.mjs";
import { leave } from "./leave.mjs";
import { built, runtime, runtimeUrl } from "./tree.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const args = process.argv.slice(2);
const forwardFile = args.includes("--forward") ? path.resolve(args[args.indexOf("--forward") + 1]) : runtime("forward.js");
const { createForward, compileKernels, compileBlocks, weightsMemory, footprint, takesBlocks } = await import(pathToFileURL(forwardFile).href);
const { WAKE, COUNTER, ACTIVE, CONTROL_BYTES, KIND, PACKED } = await import(runtimeUrl("jobs.ts"));

import { blocksModel } from "./blocks-model.mjs";
const { BLOCK_TYPES } = await import(runtimeUrl("jobs.ts"));
const model = blocksModel();
// the line, as a share of the largest logit of the reference: forward.ts against NumPy's float32 activations where
// every matrix is blocks (measured: 1.1e-2 to 1.6e-2; with the activations rounded alike 3e-7 to 1.3e-2), and where
// some are int8 and six bits, whose relaxed kernels take the activations in 7 bits (measured: 4.0e-2)
const LOOSE = { whole: 0.05, mixed: 0.15 };
const read = (name) => fs.readFileSync(built(name));
// the kernels of one build, with those of llama.cpp's blocks beside them
const build = (variant) => ({ ...compileKernels(read(`simdkernel_${variant}.wasm`), read(`simdkernel_relaxed_${variant}.wasm`), variant.endsWith("64")),
  blocks: compileBlocks(read(`blockkernel_${variant}.wasm`), read(`blockkernel_relaxed_${variant}.wasm`)) });
const KERNELS = Object.fromEntries(["plain", "shared", "plain64", "shared64"].map((variant) => [variant, build(variant)]));
const spawn = (data) => new Promise((resolve) => {
  const worker = softwareThread();
  worker.once("message", () => resolve({ terminate: () => worker.terminate() }));
  worker.postMessage(data);
});
// T120: a software thread that takes a chunk and ends without counting it (tests/threads-check.mjs's)
const dying = ({ memory, share }) => new Promise((resolve) => {
  const worker = new Worker(`
    const { parentPort, workerData: { memory, share, WAKE, COUNTER, ACTIVE, CONTROL_BYTES } } = require("node:worker_threads");
    const ctl = new Int32Array(memory.buffer, 0, CONTROL_BYTES / 4);
    parentPort.postMessage("ready");
    for (let gen = Atomics.load(ctl, WAKE + share); ; gen = Atomics.load(ctl, WAKE + share)) {
      Atomics.wait(ctl, WAKE + share, gen);
      if (Atomics.load(ctl, WAKE + share) & 1) continue;
      Atomics.add(ctl, ACTIVE, 1);
      Atomics.add(ctl, COUNTER, 1);
      process.exit(0);
    }`, { eval: true, workerData: { memory, share, WAKE, COUNTER, ACTIVE, CONTROL_BYTES } });
  worker.once("message", () => resolve({ terminate: () => worker.terminate() }));
});

fs.mkdirSync(path.join(root, ".tmp"), { recursive: true });
const folder = fs.mkdtempSync(path.join(root, ".tmp", "blocks-forward-"));
let failed = 0;
const say = (right, line) => {
  if (!right) failed++;
  console.log(`${right ? "ok" : "WRONG"}: ${line}`);
};
try {
  const cases = JSON.parse(execFileSync(process.env.PYTHON ?? "python3", [path.join(root, "tests/blocks_forward.py"), root, folder],
    { maxBuffer: 1 << 28, env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" } }).toString());
  const bytes = (name) => new Uint8Array(fs.readFileSync(path.join(folder, name)));
  // whether this engine's relaxed dot product multiplies signed bytes by signed bytes (arm64's does): the engine asks
  // the same of the same kernel, and takes the signed tile where it does
  let signed = null;
  for (const [index, { name, plan: said, tokens, checkpoint, logits, whole, header, form, weights }] of cases.entries()) {
    // (the second model is the first but for the order of the value heads: what is slow, the calls followed in
    // JavaScript and the software threads, is the first's and the others')
    const again = index === 1;
    const file = bytes(checkpoint), vocab = said.vocab_size;
    const plan = { ...said, derived: Object.fromEntries(Object.entries(said.derived).map(([table, at]) => [table, bytes(at)])) };
    const wanted = Object.fromEntries(Object.entries(logits).map(([answer, at]) => [answer, new Float32Array(bytes(at).buffer)]));
    assert.equal(wanted.rounded.length, tokens.length * vocab, `${name}: the reference's logits`);
    assert.ok(plan.own_kinds && takesBlocks(form.kinds), `${name}: the plan says its rows have kinds of their own, and the form's are blocks`);
    // an engine on a memory of one build's kernels; its logits for every token, one at a time or after a block of `block`
    const engine = async ({ variant = "plain", changed = {}, threads = 1, starts = spawn, watch = null, ...more } = {}) => {
      const { memory, base } = weightsMemory(file.length, { shared: variant.startsWith("shared"), wide: variant.endsWith("64") });
      new Uint8Array(memory.buffer, base, file.length).set(file);
      if (watch) Object.assign(watch.at, { memory, base });
      const made = createForward({ ...(watch ? { wrap: watch.wrap } : {}), memory, base, size: file.length, kernels: KERNELS[variant], plan: { ...plan, ...changed }, halfKeys: false, spawn: starts, ...more });
      if (threads > 1) assert.equal(await made.setThreads(threads), threads, `${name}: ${threads} threads`);
      return { made, memory, base };
    };
    // T363.5: every call of the blocks' kernels of an engine made with wrap: watching(the plan that is right), held to it
    const watching = () => {
      const at = {}, memoryOf = () => at;
      const seen = { wrong: [], multiplied: new Map(), rows: 0, tables: 0 };
      const layersOf = Object.entries(plan.tensors).flatMap(([stack, t]) => (t.layers ?? [t]).map((layer, l) => ({ stack, l, ...layer, shape: t.shape.slice(t.layers ? 1 : 0) })))
        .filter(({ kind, stack }) => PACKED[kind]?.tabled !== undefined && !(stack === "wcls" && plan.shared_classifier));
      const wrong = (what) => { if (seen.wrong.length < 5) seen.wrong.push(what); };
      const product = (name, [xout, xq, xs, w, tab, n, r0, r1, count, os, frame, type], call) => {
        const { memory, base } = memoryOf(), U = new Uint8Array(memory.buffer);
        // (the embedding is multiplied by where it is the classifier too)
        const layer = layersOf.find(({ offset, stack }) => base + offset === w && (stack !== "token_embedding_table" || plan.shared_classifier));
        if (!layer) return wrong(`${name} was handed a matrix at ${w - base} of the file, where the plan has no layer of blocks`);
        const [number] = BLOCK_TYPES[layer.kind], [rows, width] = layer.shape.slice(-2);
        if (type !== number || n !== width || tab !== seen.tables || !(r0 >= 0 && r0 < r1 && r1 <= rows)) {
          return wrong(`${name} of ${layer.stack}[${layer.l}] (${layer.kind}): type ${type} for ${number}, n ${n} for ${width}, tables at ${tab} for ${seen.tables}, rows ${r0} to ${r1} of ${rows}`);
        }
        call();
        const books = new Int8Array(memory.buffer, base + plan.tensors.codebooks.offset, model.BOOKS.bytes), bytes = PACKED[layer.kind].rowBytes(n);
        for (let t = 0; t < count; t++) {
          const a = new Int8Array(memory.buffer, xq + t * frame, n), scales = new Float32Array(memory.buffer.slice(xs + t * frame, xs + t * frame + n / 8));
          const wanted = model.expected(layer.kind, U, w + r0 * bytes, n, r1 - r0, books, a, scales), got = new Float32Array(memory.buffer.slice(xout + t * os + r0 * 4, xout + t * os + r1 * 4));
          if (!wanted.every((value, i) => Object.is(value, got[i]))) wrong(`${name} of ${layer.stack}[${layer.l}] (${layer.kind}), rows ${r0} to ${r1}, token ${t}: not the model's product of what it was handed`);
        }
        const key = `${layer.stack}[${layer.l}]`;
        seen.multiplied.set(key, (seen.multiplied.get(key) ?? 0) + count * (r1 - r0) / rows);
      };
      const wrap = (k) => ({ ...k,
        sb_tables: (area, codebooks) => { seen.tables = area; return k.sb_tables(area, codebooks); },
        matmul_sb: (...args) => product("matmul_sb", [...args.slice(0, 8), 1, 0, 0, args[8]], () => k.matmul_sb(...args)),
        ...(k.matmul_sbr ? {
          matmul_sbr: (...args) => product("matmul_sbr", [...args.slice(0, 8), 1, 0, 0, args[8]], () => k.matmul_sbr(...args)),
          matmul_sbr_tile: (...args) => product("matmul_sbr_tile", args, () => k.matmul_sbr_tile(...args)),
          matmul_sbw_tile: (...args) => product("matmul_sbw_tile", args, () => k.matmul_sbw_tile(...args)),
        } : {}),
        sb_row: (out, w, tab, n, type) => {
          const { memory, base } = memoryOf(), table = plan.tensors.token_embedding_table, bytes = PACKED[table.kind].rowBytes(n);
          k.sb_row(out, w, tab, n, type);
          const token = (w - base - table.offset) / bytes;
          if (type !== BLOCK_TYPES[table.kind][0] || n !== plan.dim || tab !== seen.tables || !Number.isInteger(token) || token < 0 || token >= plan.vocab_size) {
            return wrong(`sb_row: type ${type} for ${BLOCK_TYPES[table.kind][0]}, n ${n}, at ${w - base} of the file`);
          }
          const books = new Int8Array(memory.buffer, base + plan.tensors.codebooks.offset, model.BOOKS.bytes);
          const wanted = model.values(table.kind, new Uint8Array(memory.buffer), w, n, 1, books), got = new Float32Array(memory.buffer.slice(out, out + n * 4));
          if (!wanted.every((value, i) => Object.is(value, got[i]))) wrong(`sb_row of token ${token}: not the model's reading of its row`);
          seen.rows += 1;
        },
      });
      return { at, seen, wrap, layers: layersOf.filter(({ stack }) => stack !== "token_embedding_table" || plan.shared_classifier) };
    };
    // (upTo: the tokens run, all of them unless said)
    const run = async ({ upTo = tokens.length, ...options } = {}, block = 0) => {
      const { made, base } = await engine(options);
      const out = [];
      if (block) made.forwardMany(tokens.slice(0, block), 0);
      for (let pos = block; pos < upTo; pos++) {
        made.forward(tokens[pos], pos, true);
        out.push(made.logits().slice());
      }
      const placed = made.memoryBytes() - base - file.length, lost = made.lostThreads, backend = made.backend;
      await made.release();
      return { out, placed, lost, backend, from: block };
    };
    // how far logits are from a reference's, as a share of the reference's largest: the worst of every position
    const scale = wanted.rounded.reduce((most, value) => Math.max(most, Math.abs(value)), 0);
    const far = ({ out, from }, reference) => {
      let worst = 0;
      out.forEach((row, i) => row.forEach((value, v) => {
        const apart = Math.abs(value - reference[(from + i) * vocab + v]);
        if (!(apart <= worst)) worst = apart;  // (a NaN is the worst there is)
      }));
      return worst / scale;
    };
    // (to the bit, at every position both have logits of: at one at least)
    const same = (a, b) => {
      const shared = a.out.map((row, i) => [row, b.out[a.from + i - b.from]]).filter(([, other]) => other);
      return shared.length > 0 && shared.every(([row, other]) => row.every((value, v) => Object.is(value, other[v])));
    };

    // ---- the numbers
    const first = await run();
    assert.match(first.backend, /^SIMD kernels, int8, relaxed SIMD$/, `${name}: what the status line says of it is what it says of the file's dtype`);
    const line = whole ? LOOSE.whole : LOOSE.mixed, loose = far(first, wanted.floats);
    say(loose <= line, `${name}: forward.ts's logits are within ${loose.toExponential(2)} of the largest of NumPy's (the line ${line}; ` +
      `${far(first, wanted.rounded).toExponential(2)} of NumPy's with the activations of the blocks rounded as the kernels round them)`);
    // every call of the blocks' kernels, one token at a time and a block, with relaxed SIMD and without
    // (the model of the kernels follows every weight in JavaScript: a few tokens are watched, every row of every layer)
    const watched = async (changed, block, upTo) => {
      const watch = watching();
      const ran = await run({ watch, changed, upTo }, block);
      const tokensRun = upTo, missing = watch.layers.filter(({ stack, l }) => Math.abs((watch.seen.multiplied.get(`${stack}[${l}]`) ?? 0) - tokensRun * (stack === "wcls" || stack === "token_embedding_table" ? (tokensRun - block) / tokensRun : 1)) > 1e-9);
      return { ran, wrong: [...watch.seen.wrong, ...missing.slice(0, 3).map(({ stack, l }) => `${stack}[${l}] was multiplied ${watch.seen.multiplied.get(`${stack}[${l}]`) ?? 0} times for ${tokensRun} tokens`)],
        calls: [...watch.seen.multiplied.values()].length, rows: watch.seen.rows, layers: watch.layers.length };
    };
    const blocksEmbedding = PACKED[plan.tensors.token_embedding_table.kind]?.tabled !== undefined;
    for (const [how, changed, block, upTo] of again ? [] : [["a token", {}, 0, 1], ["a block of 5 and a token", {}, 5, 6], ["without relaxed SIMD, a block of 2 and a token", { relaxed: false }, 2, 3]]) {
      const { ran, wrong, rows, layers } = await watched(changed, block, upTo);
      say(!wrong.length && (same(first, ran) || (!whole && changed.relaxed === false)),
        `${name}, ${how}: every call of the blocks' kernels is of a layer the plan has, of its type, and gives the model's product to the bit: ${layers} layers of blocks, each for every token` +
        `${blocksEmbedding ? `, and ${rows} rows of the embedding` : ""}${wrong.length ? `: ${wrong.join("; ")}` : ""}`);
      if (blocksEmbedding && rows !== upTo) say(false, `${name}, ${how}: ${rows} rows of the embedding were read for ${upTo} tokens`);
    }

    // ---- what must not pass
    if (whole) {
      const other = (kind) => (kind === "IQ1_S" ? "IQ1_M" : "IQ1_S");
      const [stack, t] = Object.entries(plan.tensors).find(([, tensor]) => tensor.layers);
      const embedding = plan.tensors.token_embedding_table;
      // (the calls say these two: what a layer of another type does to the logits is no more than the rounding does)
      for (const [what, changed] of [
        [`layer 1 of ${stack} taken for ${other(t.layers[1].kind)} in place of ${t.layers[1].kind}`, { tensors: { ...plan.tensors, [stack]: { ...t, layers: t.layers.map((layer, l) => (l === 1 ? { ...layer, kind: other(layer.kind) } : layer)) } } }],
        [`the embedding taken for ${other(embedding.kind)} in place of ${embedding.kind}`, { tensors: { ...plan.tensors, token_embedding_table: { ...embedding, kind: other(embedding.kind) } } }],
      ]) {
        const { wrong } = await watched(changed, 0, 1);
        say(wrong.length > 0, `${name}: ${what} is seen in the calls (${wrong[0] ?? "NOT seen"})`);
      }
      // (and the logits these: several times the line)
      for (const [what, changed] of [
        ...(plan.linear ? [["the output matrix read in the other order of the value heads", { tiled: !plan.tiled }]] : []),
        ["the codebooks read a byte off", { tensors: { ...plan.tensors, codebooks: { ...plan.tensors.codebooks, offset: plan.tensors.codebooks.offset - 1 } } }],
      ]) {
        const apart = far(await run({ changed }), wanted.floats);
        say(!(apart <= line) && !(apart <= 3 * loose), `${name}: ${what} is ${apart.toExponential(2)} off (the line ${line}, and the right plan ${loose.toExponential(2)})`);
      }
    }

    // ---- the same numbers to the bit, however the engine runs
    signed ??= askedSigned();
    // (int8 and six bits take the activations in 7 bits with relaxed SIMD and in 8 without: other numbers, as ever)
    const ways = [
      ["without relaxed SIMD", { changed: { relaxed: false } }, 0],
      ["without the tile of the signed dot product", { trying: { signed: false } }, 9],
      ...(signed ? [["with the tile of the signed dot product", { trying: { signed: true } }, 9]] : []),
      ["a prompt's block of 9 before the last tokens", {}, 9],
      ["a block without relaxed SIMD", { changed: { relaxed: false } }, 9],
      ["a 64-bit memory", { variant: "plain64" }, 0],
      ["a block on a 64-bit memory", { variant: "plain64" }, 9],
      ["a shared memory, 1 thread", { variant: "shared" }, 0],
      ["2 threads", { variant: "shared", threads: 2 }, 0],
      ["a block on 4 threads", { variant: "shared", threads: 4 }, 9],
      ["a block on 4 threads without the signed tile", { variant: "shared", threads: 4, trying: { signed: false } }, 9],
      ["a block on 3 threads without relaxed SIMD", { variant: "shared", threads: 3, changed: { relaxed: false } }, 9],
      ["4 threads on a shared 64-bit memory, a block", { variant: "shared64", threads: 4 }, 9],
    ];
    const differing = [];
    // (a thread takes a third of a second to start and warm up: the models with every kind of layer run on them)
    const taken = ways.filter(([what, { threads }]) => (whole || !what.includes("without relaxed")) && (!threads || (plan.linear && !again)));
    for (const [what, options, block] of taken) {
      if (!same(first, await run(options, block))) differing.push(what);
    }
    say(!differing.length, `${name}: the same logits to the bit ${taken.length} other ways (${taken.map(([what]) => what).join("; ")})${differing.length ? `: DIFFERS ${differing.join("; ")}` : ""}`);
    // T120: a software thread that stops in the middle of its chunk is given up, and the phase run again gives the same
    if (plan.linear && !again) {
      const warned = console.warn;
      console.warn = () => {};
      const stopped = await run({ variant: "shared", threads: 2, starts: dying, stalledMs: 300 }, 9);
      console.warn = warned;
      say(stopped.lost && same(first, stopped), `${name}: a software thread that stops mid-chunk is given up, and the logits are the same to the bit`);
    }

    // ---- the memory: what footprint() counts of the header, the form and Python's count of the dtype's own weights
    const options = { dtype: "int8", ...form, int8: true, relaxed: true, halfKV: false, shared: false, weights };
    const bound = footprint(header, file.length, options), unsaid = footprint(header, file.length, { ...options, weights: undefined });
    const cache = said.seq_len * said.layers.filter(({ kind }) => kind === "attention").length * 2 * said.widths.kv * 4;
    // (a form that gives rows kinds that are no blocks is no conversion's: what the engine widens of those is not counted)
    say(!whole || (first.placed <= bound && bound - first.placed < 2 ** 20 + 2 ** 17 && unsaid >= bound),
      `${name}: the engine placed ${first.placed} bytes after the checkpoint (${cache} of them keys and values), footprint() counted ${bound}` +
      ` (${unsaid} without Python's count of the int8 weights)`);
  }
  // what the engine chooses of the jobs' kinds is read off a job: none is built here, so the table is asked
  assert.ok(KIND.matmul_sbr > KIND.matmul_t2 && KIND.matmul_sb > KIND.matmul_t2 && KIND.matmul_sbw > KIND.matmul_t2, "the blocks' kinds of jobs come after the kinds there were");
  console.log(`(this engine's relaxed dot product ${signed ? "multiplies signed bytes by signed bytes: the signed tile is the engine's own choice" : "takes the second byte for unsigned: the signed tile is not taken, and not tried"})`);

  // the same question the engine asks (forward/engine.ts's signedDots()), of the kernel itself
  function askedSigned() {
    const memory = new WebAssembly.Memory({ initial: 1 });
    const { signed_dots: dots } = new WebAssembly.Instance(KERNELS.plain.blocks.relaxed, { env: { memory } }).exports;
    const I8 = new Int8Array(memory.buffer), I32 = new Int32Array(memory.buffer), at = 1024;
    for (let x = -128; x < 128; x++) {
      for (let y0 = -128; y0 < 128; y0 += 16) {
        let want = 0;
        for (let j = 0; j < 16; j++) {
          I8[at + j] = x;
          I8[at + 16 + j] = y0 + j;
          want += x * (y0 + j);
        }
        dots(at + 32, at, at + 16);
        if (I32[at / 4 + 8] + I32[at / 4 + 9] + I32[at / 4 + 10] + I32[at / 4 + 11] !== want) return false;
      }
    }
    return true;
  }
} finally {
  fs.rmSync(folder, { recursive: true, force: true });
}
console.log(failed ? `blocks-forward-check: FAILED (${failed})` : "blocks-forward-check: ok");
await leave(failed ? 1 : 0);
