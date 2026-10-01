"""The plan forward.js's createForward() is handed (llama2_numpy.Llama.external_forward) for every listed model, from the
engine's own Places and tensor order (no checkpoint needed): tensors, and the sizes of the arrays Python hands over as
`derived` (the RoPE tables of a file that leaves them out)."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "public"))
import llama2_numpy as L

rows = json.load(open(os.path.join(HERE, "sizes.json")))


def plan_of(header, form, dtype):
    six = dtype == "int6"
    npdtype = np.dtype(np.int8 if dtype in ("int8", "int6") else dtype)
    probe = L.Llama.__new__(L.Llama)
    (probe.dim, probe.hidden_dim, probe.n_layers, probe.n_heads, probe.n_kv_heads, vocab_size, probe.seq_len) = header
    probe.vocab_size = abs(vocab_size)
    probe.head_size = form["head_dim"] or probe.dim // probe.n_heads
    probe.q_dim = probe.n_heads * probe.head_size
    kv_dim = probe.n_kv_heads * probe.head_size
    probe.arch = form["arch"]
    probe.rotary = probe.head_size
    # the engine's own condition for keeping int8 (the int8 kernels work on groups of 32 only)
    suitable = npdtype != np.int8 or (probe.dim % 32 == 0 and probe.q_dim % 32 == 0 and kv_dim % 32 == 0 and probe.hidden_dim % 32 == 0)
    keep_int8 = suitable and npdtype == np.int8
    places = L.Places(npdtype, six)
    frequencies = lambda width: np.zeros(width // 2)
    if form["arch"] in ("gpt2", "neox"):
        probe.gpt2_tensors(places.take, vocab_size > 0, keep_int8, kv_dim, places.dtype, frequencies)
    else:
        probe.llama_tensors(places.take, vocab_size > 0, keep_int8, kv_dim, form["bias"], places.dtype, frequencies, form["qk_norm"])
    tensors = {name: getattr(probe, name).plan() for name in L.TENSOR_NAMES if isinstance(getattr(probe, name, None), L.Tensor)}
    # Llama.__init__ after the tensors: GPT-2 has zero RoPE tables, a file that is not float32 has them computed
    table = probe.seq_len * (probe.head_size // 2) * 4
    derived = {}
    if form["arch"] == "gpt2" or npdtype != np.float32:
        derived = {"freq_cis_real": table, "freq_cis_imag": table}
    return {"tensors": tensors, "derived": derived, "keep_int8": bool(keep_int8), "end": places.offset, "head_size": probe.head_size,
            "shared_classifier": vocab_size > 0}


out = []
for row in rows:
    dtypes = ["float16"] if row["id"].endswith("-f16") else ["float32"] if row["id"].endswith("-f32") or row["id"] in ("stories260K", "stories3_5M") else ["int8", "int6"]
    for dtype in dtypes:
        plan = plan_of(row["header"], row["form"], dtype)
        size = row[f"size_{dtype}"]
        assert plan["end"] == size, (row["id"], dtype, plan["end"], size)  # the places end where the file does
        out.append({"id": row["id"], "dtype": dtype, "header": row["header"], "form": row["form"], "size": size, **plan})
json.dump(out, open(os.path.join(HERE, "plans.json"), "w"))
print(len(out), "plans")
