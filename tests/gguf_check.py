# gguf_check.py
# The check T74 needs before a GGUF reader is written: a mistake there does not raise, it only makes the text
# worse (the way T72 broke), so what "right" means is fixed first, on a real model, and the reader is held to it.
#
# The GGUF reading here is a reference of its own, written from the format's description and kept apart from
# public/llama2_convert.py on purpose: the reader that goes into the page must not share code with what checks it.
#
#   python3 tests/gguf_check.py tensors <model.gguf> <directory of the same model: config.json, model.safetensors>
#       Every tensor of the GGUF against the Hugging Face one: its name, shape and relative error, and for the
#       q and k matrices which order the GGUF holds (Hugging Face's, or turned the way llama.cpp and llama2.c turn
#       them). For Q8_0, also how many int8 values equal what llama2_convert.quantize() makes of the original.
#       The metadata against config.json, and the vocabulary against tokenizer.json.
#   python3 tests/gguf_check.py logits <out A> <out B> <text file> [tokens = 300]
#       Two converted checkpoints (the <out> of tests/perplexity_prepare.py) on the same text: the largest logit
#       difference, how often the most likely token agrees, and the perplexity of each. The acceptance of T74 is
#       this between the int8 made from the GGUF and the int8 made from safetensors.
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))

# ------------------------------------------------------------------------------------------- the reference reader
# GGUF v2/v3: "GGUF", version u32, tensor count u64, metadata count u64, the metadata (key, type u32, value), the
# tensor infos (name, n_dims u32, dims u64 innermost first, type u32, offset u64), then the data, aligned to
# general.alignment (32 when absent). Offsets count from the start of the data.
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9
F32, F16, Q8_0, BF16 = 0, 1, 8, 30
TYPE_NAMES = {F32: "F32", F16: "F16", Q8_0: "Q8_0", BF16: "BF16"}


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def take(self, fmt):
        (value,) = struct.unpack_from(fmt, self.data, self.at)
        self.at += struct.calcsize(fmt)
        return value

    def string(self):
        size = self.take("<Q")
        self.at += size
        return bytes(self.data[self.at - size:self.at]).decode("utf-8", errors="replace")

    def value(self, kind):
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            item, count = self.take("<I"), self.take("<Q")
            return [self.value(item) for _ in range(count)]
        return self.take(SCALARS[kind])


def read_gguf(path):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    r = Reader(data)
    assert bytes(data[:4]) == b"GGUF", "not a GGUF file"
    r.at = 4
    version, tensors, entries = r.take("<I"), r.take("<Q"), r.take("<Q")
    metadata = {}
    for _ in range(entries):
        key = r.string()
        metadata[key] = r.value(r.take("<I"))
    infos = {}
    for _ in range(tensors):
        name = r.string()
        dims = [r.take("<Q") for _ in range(r.take("<I"))]
        infos[name] = {"shape": tuple(reversed(dims)), "type": r.take("<I"), "offset": r.take("<Q")}
    alignment = metadata.get("general.alignment", 32)
    base = (r.at + alignment - 1) // alignment * alignment
    return version, metadata, infos, data, base


def tensor(info, data, base):
    """float32 values, and for Q8_0 also the int8 values and the float16 scales as stored."""
    count = math.prod(info["shape"])
    start = base + info["offset"]
    if info["type"] == F32:
        return np.frombuffer(data, np.float32, count, start).reshape(info["shape"]), None
    if info["type"] == F16:
        return np.frombuffer(data, np.float16, count, start).astype(np.float32).reshape(info["shape"]), None
    if info["type"] == BF16:
        wide = np.frombuffer(data, np.uint16, count, start).astype(np.uint32) << 16
        return wide.view(np.float32).reshape(info["shape"]), None
    if info["type"] == Q8_0:
        # blocks of 32: a float16 scale, then 32 int8
        blocks = np.frombuffer(data, np.uint8, count // 32 * 34, start).reshape(-1, 34)
        scales = blocks[:, :2].copy().view(np.float16).reshape(-1)
        values = blocks[:, 2:].copy().view(np.int8)
        return (values * scales.astype(np.float32)[:, None]).reshape(info["shape"]), (values, scales)
    raise ValueError(f"type {info['type']} is not one T74 takes (F32, F16, BF16, Q8_0)")


# ------------------------------------------------------------------------------------------- names and orders
def hugging_face_name(name):
    """blk.3.attn_q.weight -> model.layers.3.self_attn.q_proj.weight (Llama; T74 starts with it)."""
    fixed = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
             "output.weight": "lm_head.weight"}
    if name in fixed:
        return fixed[name]
    _, layer, rest = name.split(".", 2)
    parts = {"attn_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
             "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
             "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
             "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
             "ffn_down.weight": "mlp.down_proj.weight"}
    return f"model.layers.{layer}.{parts[rest]}"


def turned(w, heads):
    """What llama.cpp's convert_hf_to_gguf.py does to q and k (the same turn as llama2.c's export): the two
    halves of each head, as Hugging Face keeps them for rotate_half, back to adjacent pairs. Written out here
    rather than taken from llama2_convert.permute_heads, which is what is being checked."""
    rows = w.shape[0] // heads
    return w.reshape(heads, 2, rows // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# ------------------------------------------------------------------------------------------- the two checks
def check_tensors(gguf_path, directory):
    from llama2_convert import Safetensors, quantize

    version, metadata, infos, data, base = read_gguf(gguf_path)
    config = json.loads((directory / "config.json").read_text())
    hf_data = np.memmap(directory / "model.safetensors", dtype=np.uint8, mode="r")
    hf = Safetensors(lambda offset, length: hf_data[offset:offset + length])
    arch = metadata["general.architecture"]
    print(f"GGUF v{version}, {len(infos)} tensors, architecture {arch}, "
          f"types {sorted({TYPE_NAMES.get(i['type'], i['type']) for i in infos.values()})}")

    print("\n| metadata | GGUF | config.json |\n|---|---|---|")
    pairs = [("block_count", "num_hidden_layers"), ("embedding_length", "hidden_size"),
             ("feed_forward_length", "intermediate_size"), ("attention.head_count", "num_attention_heads"),
             ("attention.head_count_kv", "num_key_value_heads"), ("context_length", "max_position_embeddings"),
             ("rope.freq_base", "rope_theta"), ("attention.layer_norm_rms_epsilon", "rms_norm_eps")]
    mismatched = 0
    for key, name in pairs:
        ours, theirs = metadata.get(f"{arch}.{key}"), config.get(name)
        same = ours is not None and theirs is not None and math.isclose(float(ours), float(theirs), rel_tol=1e-6)
        mismatched += not same
        print(f"| {arch}.{key} | {ours} | {name} = {theirs}{'' if same else ' **differs**'} |")

    tokens = metadata.get("tokenizer.ggml.tokens", [])
    tokenizer = directory / "tokenizer.json"
    if tokenizer.exists():
        parsed = json.loads(tokenizer.read_text())
        vocab = {**parsed["model"]["vocab"], **{t["content"]: t["id"] for t in parsed.get("added_tokens", [])}}
        by_id = sorted(vocab, key=vocab.get)
        differ = sum(a != b for a, b in zip(tokens, by_id))
        merges = parsed["model"].get("merges", [])
        print(f"\nvocabulary: GGUF {len(tokens)} pieces ({metadata.get('tokenizer.ggml.model')}), tokenizer.json "
              f"{len(by_id)}; {differ} differ at the same id. merges: GGUF "
              f"{len(metadata.get('tokenizer.ggml.merges', []))}, tokenizer.json {len(merges)}")
        mismatched += differ > 0 or len(tokens) != len(by_id)

    heads, kv_heads = config["num_attention_heads"], config.get("num_key_value_heads", config["num_attention_heads"])
    print("\n| tensor | type | shape | relative error | order | int8 equal to quantize() |\n|---|---|---|---:|---|---:|")
    worst, orders = 0.0, set()
    for name, info in infos.items():
        values, raw = tensor(info, data, base)
        target = hugging_face_name(name)
        if target not in hf:
            print(f"| {name} | {TYPE_NAMES.get(info['type'])} | {info['shape']} | | no {target} in safetensors | |")
            mismatched += 1
            continue
        shape = tuple(hf.shape(target))
        original = hf.rows(target, 0, shape[0]).reshape(shape).astype(np.float32)
        order = ""
        if name.endswith(("attn_q.weight", "attn_k.weight")):
            n = heads if "attn_q" in name else kv_heads
            as_is, turn = relative(values, original), relative(values, turned(original, n))
            order = "turned (llama2.c order)" if turn < as_is else "as Hugging Face"
            orders.add(order)
            if turn < as_is:
                original = turned(original, n)
        error = relative(values, original) if values.shape == original.shape else float("nan")
        worst = max(worst, error) if not math.isnan(error) else math.inf
        equal = ""
        if raw is not None and values.shape == original.shape:
            ours, _ = quantize(original.reshape(-1, original.shape[-1]))
            equal = f"{(ours.reshape(-1) == raw[0].reshape(-1)).mean() * 100:.2f}%"
        print(f"| {name} | {TYPE_NAMES.get(info['type'])} | {info['shape']} | {error:.5f} | {order} | {equal} |")
    print(f"\nworst relative error {worst:.5f}; q and k are stored {' and '.join(sorted(orders)) or '(none)'}; "
          f"{mismatched} mismatches")
    return mismatched == 0 and worst < 0.02


def check_logits(a, b, text_file, count):
    from llama2_numpy import Llama

    def load(out):
        return Llama(np.memmap(f"{out}.bin", dtype=np.uint8, mode="r"), Path(f"{out}.tokenizer.bin").read_bytes(),
                     kernels=None, **json.loads(Path(f"{out}.json").read_text()))

    first, second = load(a), load(b)
    tokens = first.tokenizer.encode(Path(text_file).read_text())[:count]
    assert tokens == second.tokenizer.encode(Path(text_file).read_text())[:count], "the two tokenize differently"
    tokens = [first.bos] + tokens
    largest, agree, nll = 0.0, 0, [0.0, 0.0]
    for pos in range(len(tokens) - 1):
        one = np.asarray(first.forward(tokens[pos], pos), dtype=np.float64)
        two = np.asarray(second.forward(tokens[pos], pos), dtype=np.float64)
        largest = max(largest, float(np.abs(one - two).max()))
        agree += int(one.argmax() == two.argmax())
        for i, logits in enumerate((one, two)):
            shifted = logits - logits.max()
            nll[i] -= shifted[tokens[pos + 1]] - math.log(np.exp(shifted).sum())
    n = len(tokens) - 1
    print(json.dumps({"a": Path(a).name, "b": Path(b).name, "tokens": n, "largest logit difference": largest,
                      "top-1 agreement": agree / n, "perplexity a": math.exp(nll[0] / n),
                      "perplexity b": math.exp(nll[1] / n)}))


if __name__ == "__main__":
    if sys.argv[1] == "tensors":
        sys.exit(0 if check_tensors(sys.argv[2], Path(sys.argv[3])) else 1)
    elif sys.argv[1] == "logits":
        check_logits(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 300)
