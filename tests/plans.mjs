// plans.mjs
// What createForward() is handed for a made-up header, without Pyodide and without built kernels: the plan Python
// makes (llama2_numpy's own Places and tensor order, by the native Python) and kernels that do nothing. For the checks
// that run forward.js on its own: tests/memory-check.mjs (T130) and tests/unchanged-calls.mjs (T346).
//   const { plansOf, planOf, FORM, empty, nothing } = plans(<the repository's root, a URL or a path>)
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

// the plan Python hands createForward (llama2_numpy's own Places and tensor order), for a header, a form and a dtype:
// where every tensor is, and the bytes of the RoPE tables Python computes (the checkpoint's size is where the places end)
const python = `
import json, sys
import numpy as np
sys.path.insert(0, "public")
import llama2_convert, llama2_numpy as L

def plan_of(header, form, dtype):
    packing = dtype if dtype in L.PACKED else None
    npdtype = np.dtype(np.int8 if dtype == "int8" or packing else dtype)
    probe = L.Llama.__new__(L.Llama)
    (probe.dim, probe.hidden_dim, probe.n_layers, probe.n_heads, probe.n_kv_heads, vocab, probe.seq_len) = header
    probe.vocab_size = abs(vocab)
    probe.head_size = form["head_dim"] or probe.dim // probe.n_heads
    probe.q_dim = probe.n_heads * probe.head_size
    kv_dim = probe.n_kv_heads * probe.head_size
    probe.arch, probe.rotary = form["arch"], probe.head_size
    probe.rope_magnitude = 1.0  # the places, not the values (as external_tensors() sets it)
    # T229: a Qwen3.5's linear-attention layers, and RoPE over a quarter of a head
    probe.linear = L.linear_form(form["linear"])
    if probe.linear is not None:
        probe.slots = L.layer_slots(probe.n_layers, probe.linear)
        probe.rotary = probe.head_size // 4
    # T260: an LFM2's convolution layers
    probe.convolution = L.convolution_form(form["convolution"], probe.n_layers)
    # the engine's own condition for keeping int8: the int8 kernels work on groups of 32 only (T229: and the rows of a
    # linear-attention layer's output matrix, its value heads together)
    keep = npdtype == np.int8 and all(n % 32 == 0 for n in (probe.dim, probe.q_dim, kv_dim, probe.hidden_dim)) \
        and (probe.linear is None or L.linear_widths(probe.linear)[2] % 32 == 0)
    places = L.Places(npdtype, packing)
    freq = lambda width: np.zeros(width // 2)
    if form["arch"] in ("gpt2", "neox"):
        probe.gpt2_tensors(places.take, vocab > 0, keep, kv_dim, places.dtype, freq)
    elif form["arch"] == "qwen35":
        probe.qwen35_tensors(places.take, vocab > 0, keep, kv_dim, places.dtype, freq)
    elif form["arch"] == "lfm2":
        probe.lfm2_tensors(places.take, vocab > 0, keep, kv_dim, places.dtype, freq)
    else:
        probe.llama_tensors(places.take, vocab > 0, keep, kv_dim, form["bias"], places.dtype, freq, form["qk_norm"])
    tensors = {name: getattr(probe, name).plan() for name in L.TENSOR_NAMES if isinstance(getattr(probe, name, None), L.Tensor)}
    table = probe.seq_len * (probe.head_size // 2) * 4
    derived = {"freq_cis_real": table, "freq_cis_imag": table} if form["arch"] == "gpt2" or npdtype != np.float32 else {}
    assert places.offset == llama2_convert.checkpoint_size(header, dtype, form), "the places do not end where the file does"
    return {"header": header, "form": form, "dtype": dtype, "size": places.offset, "tensors": tensors, "derived": derived,
            "keep_int8": bool(keep), "head_size": probe.head_size}

print(json.dumps([plan_of(**s) for s in json.loads(sys.stdin.read())]))
`;
const plansIn = (root) => (shapes) => JSON.parse(execFileSync(process.env.PYTHON ?? "python3", ["-c", python],
  { cwd: typeof root === "string" ? root : fileURLToPath(root), input: JSON.stringify(shapes), maxBuffer: 1 << 28 }).toString());
export const FORM = { bias: false, arch: "llama", qk_norm: false, head_dim: 0, linear: null, rotated: null, convolution: null };

// what createForward is handed, from a plan of Python's: kv_start where the cache starts, and the outlier channels of the
// final norm there are (footprint() counts the most there can be for every int8 model: the second part asks for them)
export const empty = new WebAssembly.Module(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0]));
export function planOf(p, { relaxed = true, kvStart = p.header[6], outliers = 0 } = {}) {
  const [dim, hidden, layers, heads, kvHeads, signedVocab, seqLen] = p.header;
  // T237's review: a rotated basis (the form's) is a block and a float32 for every value of every width a matrix reads: the
  // residual stream, an attention's output, the FFN's inside, a linear layer's output (what Python's plan holds in derived)
  const rotated = p.form.rotated ? p.form.rotated.block : 0, linear = p.form.linear;
  const signs = rotated ? [...new Set([dim, heads * p.head_size, hidden, ...(linear ? [linear.value_heads * linear.value_dim] : [])])] : [];
  return {
    arch: p.form.arch, dim, hidden_dim: hidden, n_layers: layers, n_heads: heads, n_kv_heads: kvHeads, head_size: p.head_size,
    vocab_size: Math.abs(signedVocab), seq_len: seqLen, rotary: p.form.linear ? p.head_size / 4 : p.head_size, linear: p.form.linear,
    convolution: p.form.convolution,
    // (T255: the layers RoPE leaves alone; none of these made-up models has any. tests/plan-keys-check.mjs holds these keys to Python's)
    unturned: [],
    parallel_residual: p.form.arch === "neox", rotated,
    kv_start: kvStart, rms_norm_eps: 1e-5, shared_classifier: signedVocab > 0, int8: p.keep_int8, relaxed, tensors: p.tensors,
    derived: Object.fromEntries([...Object.entries(p.derived).map(([name, bytes]) => [name, new Uint8Array(bytes)]),
      ...signs.map((width) => [`signs.${width}`, new Uint8Array(width * 4)])]),
    outliers: p.keep_int8 ? Array.from({ length: Math.min(outliers, dim) }, (_, c) => c) : [], half_kv: p.keep_int8,
  };
}
// (the kernels do nothing unless a part says what one does: wrap is how the profiler swaps them, tests/profile.mjs)
export const nothing = () => new Proxy({}, { get: () => () => 0 });

export const plans = (root) => ({ plansOf: plansIn(root), planOf, FORM, empty, nothing });
