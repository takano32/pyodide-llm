# gguf_check.py
# The check T74 needs before a GGUF reader is written: a mistake there does not raise, it only makes the text
# worse (the way T72 broke), so what "right" means is fixed first, on a real model, and the reader is held to it.
#
# The GGUF reading here is a reference of its own, written from the format's description and kept apart from
# public/llama2_convert.py on purpose: the reader that goes into the page must not share code with what checks it.
#
#   python3 tests/gguf_check.py tensors <model.gguf> <directory of the same model: config.json, model.safetensors
#                                       or its shards and model.safetensors.index.json>
#       Every tensor of the GGUF against the Hugging Face one: its name, shape and relative error, and for the
#       q and k matrices which order the GGUF holds (Hugging Face's, or turned the way llama.cpp and llama2.c turn
#       them). For Q8_0, also how many int8 values equal what llama2_convert.quantize() makes of the original.
#       The metadata against config.json, and the vocabulary against tokenizer.json.
#   python3 tests/gguf_check.py logits <out A> <out B> <text file> [tokens = 300]
#       Two converted checkpoints (the <out> of tests/perplexity_prepare.py) on the same text: the largest logit
#       difference, how often the most likely token agrees, and the perplexity of each. The acceptance of T74 is
#       this between the int8 made from the GGUF and the int8 made from safetensors: top-1 agreement of 99% and
#       perplexity within 0.2% (exit 1 otherwise).
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
# T98: the 4- and 5-bit types of the GGUF files that are about int4, read only to measure them (widened to float32)
Q4_0, Q4_1, Q5_0, Q4_K, Q6_K = 2, 3, 6, 12, 14
TYPE_NAMES = {F32: "F32", F16: "F16", Q8_0: "Q8_0", BF16: "BF16", Q4_0: "Q4_0", Q4_1: "Q4_1", Q5_0: "Q5_0",
              Q4_K: "Q4_K", Q6_K: "Q6_K"}
# bytes per value: a block of 32 values (or a super-block of 256) and its scales
BYTES = {F32: 4, F16: 2, BF16: 2, Q8_0: 34 / 32, Q4_0: 18 / 32, Q4_1: 20 / 32, Q5_0: 22 / 32, Q4_K: 144 / 256, Q6_K: 210 / 256}


def half(raw):
    return raw.copy().view(np.float16).astype(np.float32)


def low_high(qs):
    """16 bytes of 4-bit values per block: the low halves are values 0..15, the high halves 16..31 (ggml's order)."""
    return np.concatenate([qs & 0x0F, qs >> 4], axis=1)


def widen_q4_0(raw):  # a float16 scale, 16 bytes: (q - 8) * d
    blocks = raw.reshape(-1, 18)
    return (low_high(blocks[:, 2:]).astype(np.float32) - 8) * half(blocks[:, :2])


def widen_q4_1(raw):  # a float16 scale and a float16 minimum, 16 bytes: q * d + m
    blocks = raw.reshape(-1, 20)
    return low_high(blocks[:, 4:]).astype(np.float32) * half(blocks[:, :2]) + half(blocks[:, 2:4])


def widen_q5_0(raw):  # a float16 scale, 32 fifth bits (u32), 16 bytes: (q | fifth << 4) - 16, times d
    blocks = raw.reshape(-1, 22)
    fifth = blocks[:, 2:6].copy().view("<u4")
    bits = (fifth >> np.arange(32, dtype=np.uint32)) & 1
    return ((low_high(blocks[:, 6:]) | (bits << 4)).astype(np.float32) - 16) * half(blocks[:, :2])


def widen_q4_k(raw):
    """Super-blocks of 256: float16 d and dmin, 12 bytes of eight 6-bit scales and eight 6-bit minimums, 128 bytes
    of 4-bit values. Sub-block j of 32: value * d * scale[j] - dmin * min[j]; the values go 64 at a time, the low
    halves of 32 bytes first, then their high halves."""
    blocks = raw.reshape(-1, 144)
    d, dmin, packed, qs = half(blocks[:, 0:2]), half(blocks[:, 2:4]), blocks[:, 4:16].astype(np.int32), blocks[:, 16:]
    scales, minimums = np.empty((len(blocks), 8), np.int32), np.empty((len(blocks), 8), np.int32)
    scales[:, :4], minimums[:, :4] = packed[:, 0:4] & 63, packed[:, 4:8] & 63
    scales[:, 4:] = (packed[:, 8:12] & 0x0F) | ((packed[:, 0:4] >> 6) << 4)
    minimums[:, 4:] = (packed[:, 8:12] >> 4) | ((packed[:, 4:8] >> 6) << 4)
    values = qs.reshape(-1, 4, 32)
    values = np.stack([values & 0x0F, values >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)
    return (values * (d * scales)[:, :, None] - (dmin * minimums)[:, :, None]).reshape(-1, 256)


def widen_q6_k(raw):
    """Super-blocks of 256: 128 bytes of low 4 bits, 64 bytes of high 2 bits, 16 int8 scales (one per 16 values),
    a float16 d. Two halves of 128; in each, value l (0..31) of the four quarters is made of ql[l], ql[l + 32]
    (low and high nibbles) and the four 2-bit fields of qh[l], minus 32."""
    blocks = raw.reshape(-1, 210)
    ql, qh = blocks[:, :128].reshape(-1, 2, 64), blocks[:, 128:192].reshape(-1, 2, 32)
    scales, d = blocks[:, 192:208].copy().view(np.int8).astype(np.float32), half(blocks[:, 208:210])
    quarters = [(ql[:, :, :32] & 0x0F) | (((qh >> 0) & 3) << 4), (ql[:, :, 32:] & 0x0F) | (((qh >> 2) & 3) << 4),
                (ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4), (ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4)]
    values = np.stack(quarters, axis=2).astype(np.float32) - 32  # (blocks, half, quarter, 32)
    scale = scales.reshape(-1, 2, 4, 2).repeat(16, axis=3)  # one per 16 values
    return (values * scale * d[:, :, None, None]).reshape(-1, 256)


WIDEN = {Q4_0: (32, 18, widen_q4_0), Q4_1: (32, 20, widen_q4_1), Q5_0: (32, 22, widen_q5_0),
         Q4_K: (256, 144, widen_q4_k), Q6_K: (256, 210, widen_q6_k)}


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
    assert version in (2, 3), f"GGUF version {version}: only 2 and 3 count tensors in 64 bits (1 used 32)"
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


def tensor(info, data, base, first=0, last=None):
    """float32 values of rows first..last, and for Q8_0 also the int8 values and the float16 scales as stored."""
    shape = info["shape"]
    last = shape[0] if last is None else last
    row = math.prod(shape[1:]) if len(shape) > 1 else shape[0]
    if len(shape) > 1:
        shape = (last - first, *shape[1:])
    count = math.prod(shape)
    size = BYTES.get(info["type"], 0)
    start = base + info["offset"] + int(first * row * size)
    if info["type"] == F32:
        return np.frombuffer(data, np.float32, count, start).reshape(shape), None
    if info["type"] == F16:
        return np.frombuffer(data, np.float16, count, start).astype(np.float32).reshape(shape), None
    if info["type"] == BF16:
        wide = np.frombuffer(data, np.uint16, count, start).astype(np.uint32) << 16
        return wide.view(np.float32).reshape(shape), None
    if info["type"] == Q8_0:
        # blocks of 32: a float16 scale, then 32 int8
        blocks = np.frombuffer(data, np.uint8, count // 32 * 34, start).reshape(-1, 34)
        scales = blocks[:, :2].copy().view(np.float16).reshape(-1)
        values = blocks[:, 2:].copy().view(np.int8)
        return (values * scales.astype(np.float32)[:, None]).reshape(shape), (values, scales)
    if info["type"] in WIDEN:
        values, block_bytes, widen = WIDEN[info["type"]]
        raw = np.frombuffer(data, np.uint8, count // values * block_bytes, start)
        return widen(raw).reshape(shape), None
    raise ValueError(f"type {info['type']} is not one this reads ({', '.join(TYPE_NAMES.values())})")


# ------------------------------------------------------------------------------------------- names and orders
def hugging_face_name(name):
    """blk.3.attn_q.weight -> model.layers.3.self_attn.q_proj.weight (Llama; T74 starts with it)."""
    fixed = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
             "output.weight": "lm_head.weight"}
    if name in fixed:
        return fixed[name]
    _, layer, rest = name.split(".", 2)
    parts = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj",
             "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
             "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
    tensor, kind = rest.rsplit(".", 1)  # .weight, or .bias (Qwen2's q, k and v)
    return f"model.layers.{layer}.{parts[tensor]}.{kind}"


def turned(w, heads):
    """What llama.cpp's convert_hf_to_gguf.py does to q and k (the same turn as llama2.c's export): the two
    halves of each head, as Hugging Face keeps them for rotate_half, back to adjacent pairs. Written out here
    rather than taken from llama2_convert.permute_heads, which is what is being checked."""
    rows = w.shape[0] // heads
    return w.reshape(heads, 2, rows // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# ------------------------------------------------------------------------------------------- the two checks
BLOCK = 8 << 20  # values; larger tensors are compared in blocks of rows


def large(info, data, base, hf, target, shape, quantize):
    """relative error and int8 equality of a large matrix, a block of rows at a time."""
    rows = max(1, BLOCK // math.prod(shape[1:]))
    difference = total = same = count = 0.0
    for first in range(0, shape[0], rows):
        last = min(first + rows, shape[0])
        values, raw = tensor(info, data, base, first, last)
        original = hf.rows(target, first, last).reshape(last - first, *shape[1:]).astype(np.float32)
        difference += float(((values - original) ** 2).sum())
        total += float((original ** 2).sum())
        if raw is not None:
            ours, _ = quantize(original.reshape(-1, original.shape[-1]))
            same += float((ours.reshape(-1) == raw[0].reshape(-1)).sum())
            count += ours.size
    return math.sqrt(difference / max(total, 1e-30)), f"{same / count * 100:.2f}%" if count else ""


def check_tensors(gguf_path, directory):
    from llama2_convert import Safetensors, Shards, quantize

    version, metadata, infos, data, base = read_gguf(gguf_path)
    config = json.loads((directory / "config.json").read_text())
    index = directory / "model.safetensors.index.json"
    if not (directory / "model.safetensors").exists() and index.exists():
        # a model split over several files (T136: Qwen2.5 3B and 7B): every shard as one source, as convert_hf.py reads it
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        maps = [np.memmap(directory / name, dtype=np.uint8, mode="r") for name in files]
        hf = Shards([Safetensors(lambda offset, length, data=data: data[offset:offset + length]) for data in maps])
    else:
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
        # llama.cpp fills the vocabulary up to the rows of the embedding with pieces of its own ([PAD151665] ...,
        # token type 5, unused), where tokenizer.json ends earlier (T136: Qwen2.5's 151665 of 151936): no difference
        kinds = metadata.get("tokenizer.ggml.token_type", [])
        padding = len(tokens) > len(by_id) == len(set(by_id)) and len(tokens) == config.get("vocab_size") \
            and all(kind == 5 for kind in kinds[len(by_id):]) and len(kinds) == len(tokens)
        print(f"\nvocabulary: GGUF {len(tokens)} pieces ({metadata.get('tokenizer.ggml.model')}), tokenizer.json "
              f"{len(by_id)}; {differ} differ at the same id. merges: GGUF "
              f"{len(metadata.get('tokenizer.ggml.merges', []))}, tokenizer.json {len(merges)}"
              + (f"; the GGUF's last {len(tokens) - len(by_id)} are its padding (unused) up to vocab_size" if padding else ""))
        mismatched += differ > 0 or (len(tokens) != len(by_id) and not padding)

    heads, kv_heads = config["num_attention_heads"], config.get("num_key_value_heads", config["num_attention_heads"])
    print("\n| tensor | type | shape | relative error | order | int8 equal to quantize() |\n|---|---|---|---:|---|---:|")
    worst, orders = 0.0, set()
    for name, info in infos.items():
        target = hugging_face_name(name)
        if target not in hf:
            print(f"| {name} | {TYPE_NAMES.get(info['type'])} | {info['shape']} | | no {target} in safetensors | |")
            mismatched += 1
            continue
        shape = tuple(hf.shape(target))
        if math.prod(shape) > BLOCK and len(shape) > 1 and tuple(info["shape"]) == shape:
            # a large matrix (Qwen's embedding is 545 MB as float32) is compared a block of rows at a time: two
            # whole copies side by side are what took this machine down
            error, equal = large(info, data, base, hf, target, shape, quantize)
            worst = max(worst, error)
            print(f"| {name} | {TYPE_NAMES.get(info['type'])} | {info['shape']} | {error:.5f} | | {equal} |")
            continue
        values, raw = tensor(info, data, base)
        original = hf.rows(target, 0, shape[0]).reshape(shape).astype(np.float32)
        order = ""
        if name.endswith(("attn_q.weight", "attn_k.weight", "attn_q.bias", "attn_k.bias")):
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

    # One model at a time: without the kernels an int8 model is widened to float32 (SmolLM2 135M: 540 MB each),
    # and two of them side by side took this 6.6 GB machine down twice. The logits of the first are kept
    # (count x vocab_size float32, 59 MB for SmolLM2) and the model is let go before the second is loaded.
    import gc

    def logits_of(out, tokens=None):
        llama = load(out)
        encoded = llama.tokenizer.encode(Path(text_file).read_text())[:count]
        if tokens is not None:
            assert encoded == tokens[1:], "the two tokenize differently"
        tokens = [llama.bos] + encoded
        rows = np.stack([np.asarray(llama.forward(tokens[pos], pos), dtype=np.float32) for pos in range(len(tokens) - 1)])
        del llama
        gc.collect()
        return tokens, rows

    tokens, first = logits_of(a)
    _, second = logits_of(b, tokens)
    largest, agree, nll = 0.0, 0, [0.0, 0.0]
    for pos in range(len(tokens) - 1):
        one, two = first[pos].astype(np.float64), second[pos].astype(np.float64)
        largest = max(largest, float(np.abs(one - two).max()))
        agree += int(one.argmax() == two.argmax())
        for i, logits in enumerate((one, two)):
            shifted = logits - logits.max()
            nll[i] -= shifted[tokens[pos + 1]] - math.log(np.exp(shifted).sum())
    n = len(tokens) - 1
    first_ppl, second_ppl = math.exp(nll[0] / n), math.exp(nll[1] / n)
    # The line T74 is held to (Fable, 2026-09-24): the two int8 differ only by the rounding of the scales
    # (float16 in Q8_0, float32 here), so they must agree on the most likely token 99 times in 100 and be within
    # 0.2% of perplexity. The float32 against int8 of the same model measures 96.3% and 0.6%: a mistake in the
    # reader (a wrong order, a wrong scale) lands far outside this.
    ok = agree / n >= 0.99 and abs(second_ppl / first_ppl - 1) <= 0.002
    print(json.dumps({"a": Path(a).name, "b": Path(b).name, "tokens": n, "largest logit difference": largest,
                      "top-1 agreement": agree / n, "perplexity a": first_ppl, "perplexity b": second_ppl,
                      "within the line": ok}))
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "tensors":
        sys.exit(0 if check_tensors(sys.argv[2], Path(sys.argv[3])) else 1)
    elif sys.argv[1] == "logits":
        sys.exit(0 if check_logits(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 300) else 1)
