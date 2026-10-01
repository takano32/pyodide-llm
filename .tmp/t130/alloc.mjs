// T130's review: what forward.js's createForward() really allocates after the checkpoint (its own alloc(), with kernels
// that do nothing: the engine is built, not run) at the end of the whole context, against footprint(), for every listed
// model, as int8 and int6, with and without relaxed SIMD, on a shared memory (base 8192 and the shared rule's type for
// the keys and values) and on a plain one (base 64 and the plain rule's), the plain one as a shared memory refused
// leaves it (sized by the shared estimate, type from keysInHalf on the plain memory).
//   node --expose-gc .tmp/t130/alloc.mjs [id-substring ...]
import fs from "node:fs";
import { createForward, footprint, keysInHalf, needsWide } from "../../public/forward.js";
import { CONTROL_BYTES } from "../../public/jobs.js";

const PAGE = 65536, GiB = 2 ** 30, MiB = 2 ** 20;
const plans = JSON.parse(fs.readFileSync(new URL("plans.json", import.meta.url), "utf8"));
const only = process.argv.slice(2);
const empty = new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
const noop = () => 0;
const stubs = () => new Proxy({}, { get: (_, name) => (name === "then" ? undefined : noop) });

function planOf(p, relaxed, quantized) {
  const [dim, hidden, layers, heads, kvHeads, signedVocab, seqLen] = p.header;
  const headSize = p.head_size, derived = {};
  for (const [name, bytes] of Object.entries(p.derived)) derived[name] = new Uint8Array(bytes);
  return {
    arch: p.form.arch, dim, hidden_dim: hidden, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: headSize,
    vocab_size: Math.abs(signedVocab), seq_len: seqLen, rotary: headSize, parallel_residual: p.form.arch === "neox",
    kv_start: seqLen, rms_norm_eps: 1e-5, shared_classifier: signedVocab > 0, int8: p.keep_int8, relaxed,
    tensors: p.tensors, derived,
    // the final norm's weight has outliers: the engine is told 8 channels (footprint counts them for every quantized model)
    outliers: p.keep_int8 ? [0, 1, 2, 3, 4, 5, 6, 7].filter((c) => c < dim) : [],
    half_kv: p.keep_int8,
  };
}

const results = [];
let failures = 0;
for (const p of plans) {
  if (only.length && !only.some((s) => p.id.includes(s))) continue;
  if (p.dtype === "float16" && !process.env.F16) continue;  // widening every weight of a float16 file touches 4 bytes of RAM for each
  process.stderr.write(`${p.id} ${p.dtype}\n`);
  const quantized = p.dtype === "int8" || p.dtype === "int6";
  for (const relaxed of process.env.RELAXED ? [true] : [true, false]) {
    const options = { ...p.form, dtype: p.dtype, int8: true, relaxed, halfKV: p.keep_int8 && quantized, outliers: 8, gpu: false };
    const sharedAfter = footprint(p.header, p.size, { ...options, shared: true });
    const wide = needsWide(p.size, sharedAfter);
    for (const scenario of process.env.SCENARIO ? [process.env.SCENARIO] : ["shared", "plain", "refused"]) {
      const shared = scenario === "shared";
      const base = shared ? CONTROL_BYTES : 64;
      const sized = scenario === "plain" ? footprint(p.header, p.size, { ...options, shared: false }) : sharedAfter;
      const bound = footprint(p.header, p.size, { ...options, shared });
      const halfKeys = keysInHalf(p.header, p.size, { ...options, shared });
      const initial = Math.ceil((base + p.size) / PAGE) + 1;
      let memory;
      try {
        memory = wide ? new WebAssembly.Memory({ initial: BigInt(initial), address: "i64" }) : new WebAssembly.Memory({ initial });
      } catch (error) {
        results.push({ id: p.id, dtype: p.dtype, relaxed, scenario, error: `memory: ${error.message}` });
        continue;
      }
      let line;
      try {
        const engine = createForward({ memory, base, size: p.size, kernels: { plain: empty, relaxed: relaxed ? empty : null, wide },
          plan: planOf(p, relaxed, quantized), halfKeys, wrap: stubs });
        const used = engine.memoryBytes() - base - p.size;
        engine.release?.();
        line = { id: p.id, dtype: p.dtype, relaxed, scenario, wide, half: halfKeys, used, bound, margin: bound - used,
          total: base + p.size + used, sizedTotal: base + p.size + sized };
      } catch (error) {
        line = { id: p.id, dtype: p.dtype, relaxed, scenario, error: `engine: ${error.stack?.split("\n")[0] ?? error}` };
      }
      memory = undefined;
      globalThis.gc?.();
      if (line.error || line.used > line.bound) failures++;
      results.push(line);
    }
  }
}
fs.writeFileSync(new URL("alloc.json", import.meta.url), JSON.stringify(results, null, 1));
const bad = results.filter((r) => r.error || r.used > r.bound);
const margins = results.filter((r) => !r.error).map((r) => r.margin);
console.log(`${results.length} engines built, ${bad.length} past footprint() or failed`);
for (const r of bad) console.log(r.error ? `ERROR ${r.id} ${r.dtype} relaxed=${r.relaxed} ${r.scenario}: ${r.error}` : `PAST ${r.id} ${r.dtype} relaxed=${r.relaxed} ${r.scenario}: used ${(r.used / MiB).toFixed(1)} MiB > bound ${(r.bound / MiB).toFixed(1)} MiB`);
console.log(`margin (bound - used): min ${(Math.min(...margins) / MiB).toFixed(2)} MiB, max ${(Math.max(...margins) / MiB).toFixed(2)} MiB`);
// the engine of a refused shared memory: does it still fit the width the shared estimate chose?
const over = results.filter((r) => !r.error && r.scenario === "refused" && r.total > (r.wide ? 262144 : 65536) * PAGE);
console.log(`refused-shared engines past their memory's width: ${over.length}`);
for (const r of over) console.log("  OVER", r.id, r.dtype, `relaxed=${r.relaxed}`, `total ${(r.total / GiB).toFixed(3)} GiB`);
const fits = results.filter((r) => !r.error && !r.wide && r.total > 65536 * PAGE);
console.log(`32-bit engines past 4 GiB (any scenario): ${fits.length}`);
for (const r of fits) console.log("  OVER32", r.id, r.dtype, `relaxed=${r.relaxed}`, r.scenario, `total ${(r.total / GiB).toFixed(3)} GiB`);
