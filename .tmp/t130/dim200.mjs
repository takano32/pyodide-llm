// the dim-200 shape of memory-check.mjs on a forward.js: what footprint() counts and what createForward allocates
import { execFileSync } from "node:child_process";
import path from "node:path";
const file = path.resolve(process.argv[2]);
const { createForward, footprint, keysInHalf } = await import(file);
const root = new URL("../../", import.meta.url).pathname;
const PAGE = 65536, MiB = 2 ** 20;
const python = `
import json, sys
import numpy as np
sys.path.insert(0, "public")
import llama2_convert, llama2_numpy as L
header, form, dtype = [200, 400, 4, 4, 4, 20000, 2048], {"bias": False, "arch": "llama", "qk_norm": False, "head_dim": 0}, "int8"
probe = L.Llama.__new__(L.Llama)
(probe.dim, probe.hidden_dim, probe.n_layers, probe.n_heads, probe.n_kv_heads, vocab, probe.seq_len) = header
probe.vocab_size = abs(vocab)
probe.head_size = probe.dim // probe.n_heads
probe.q_dim = probe.n_heads * probe.head_size
kv_dim = probe.n_kv_heads * probe.head_size
probe.arch, probe.rotary = "llama", probe.head_size
places = L.Places(np.int8, False)
probe.llama_tensors(places.take, vocab > 0, False, kv_dim, False, places.dtype, lambda w: np.zeros(w // 2), False)
tensors = {name: getattr(probe, name).plan() for name in L.TENSOR_NAMES if isinstance(getattr(probe, name, None), L.Tensor)}
table = probe.seq_len * (probe.head_size // 2) * 4
print(json.dumps({"size": places.offset, "tensors": tensors, "derived": {"freq_cis_real": table, "freq_cis_imag": table}, "head_size": probe.head_size}))
`;
const p = JSON.parse(execFileSync("python3", ["-c", python], { cwd: root }).toString());
const header = [200, 400, 4, 4, 4, 20000, 2048], [dim, hidden, layers, heads, kvHeads, vocab, seqLen] = header;
const empty = new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
const plan = { arch: "llama", dim, hidden_dim: hidden, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: p.head_size, vocab_size: vocab,
  seq_len: seqLen, rotary: p.head_size, parallel_residual: false, kv_start: seqLen, rms_norm_eps: 1e-5, shared_classifier: true, int8: false, relaxed: true,
  tensors: p.tensors, derived: Object.fromEntries(Object.entries(p.derived).map(([n, b]) => [n, new Uint8Array(b)])), outliers: [], half_kv: false };
for (const shared of [true, false]) {
  const options = { dtype: "int8", int8: true, relaxed: true, halfKV: true, outliers: 8, gpu: false, shared };
  const halfKeys = keysInHalf(header, p.size, options), bound = footprint(header, p.size, options);
  const base = shared ? 8192 : 64, memory = new WebAssembly.Memory({ initial: Math.ceil((base + p.size) / PAGE) + 1 });
  const engine = createForward({ memory, base, size: p.size, kernels: { plain: empty, relaxed: empty, wide: false }, plan, halfKeys,
    wrap: () => new Proxy({}, { get: () => () => 0 }) });
  const used = engine.memoryBytes() - base - p.size;
  console.log(`${shared ? "shared" : "plain "}: halfKeys ${halfKeys}, footprint ${(bound / MiB).toFixed(2)} MiB, created ${(used / MiB).toFixed(2)} MiB`);
}
