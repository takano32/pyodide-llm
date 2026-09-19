# make_hf_fixture.py
# The files Hugging Face would publish for stories260K, made from the llama2.c checkpoint and tokenizer of this
# directory: model.safetensors, config.json, tokenizer.model (sentencepiece) and settings.json. tests/e2e.mjs opens
# them through the folder button, and the page has to convert them back and write the same story.
#
#   python3 tests/make_hf_fixture.py <directory> [F32|BF16]
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "public"))
from llama2_convert import layout  # noqa: E402


def varint(value):
    out = b""
    while True:
        out += bytes([value & 0x7F | (0x80 if value > 0x7F else 0)])
        value >>= 7
        if not value:
            return out


def field(number, payload):
    """One protobuf field: bytes are length-delimited, an int is a varint, a float is 32 bits."""
    if isinstance(payload, bytes):
        return varint(number << 3 | 2) + varint(len(payload)) + payload
    if isinstance(payload, float):
        return varint(number << 3 | 5) + struct.pack("<f", payload)
    return varint(number << 3) + varint(payload)


def sentencepiece_model(tokenizer, vocab_size):
    NORMAL, UNKNOWN, CONTROL, BYTE, BPE = 1, 2, 3, 6, 2
    model, offset = b"", 4
    for id in range(vocab_size):
        score, length = struct.unpack_from("<fi", tokenizer, offset)
        text = tokenizer[offset + 8:offset + 8 + length].decode("utf-8")
        offset += 8 + length
        is_byte = len(text) == 6 and text.startswith("<0x") and text.endswith(">")
        kind = UNKNOWN if id == 0 else CONTROL if id in (1, 2) else BYTE if is_byte else NORMAL
        model += field(1, field(1, text.replace(" ", "▁").encode("utf-8")) + field(2, float(score)) + field(3, kind))
    return model + field(2, field(3, BPE)) + field(3, field(1, b"identity"))


if __name__ == "__main__":
    out, stored = Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "F32"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = (ROOT / "stories260K.bin").read_bytes()
    header = struct.unpack_from("<7i", checkpoint, 0)
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header
    head_size = dim // n_heads
    names = ["model.embed_tokens", "input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "post_attention_layernorm", "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj", "model.norm",
             None, None, "lm_head"]

    def permute(w, heads):  # llama2.c's adjacent pairs -> Hugging Face's [first halves, second halves]
        return w.reshape(heads, head_size // 2, 2, w.shape[1]).transpose(0, 2, 1, 3).reshape(w.shape)

    tensors, offset = {}, 28
    for name, (shape, _) in zip(names, layout(*header)):
        tensor = np.frombuffer(checkpoint, dtype=np.float32, count=int(np.prod(shape)), offset=offset).reshape(shape)
        offset += tensor.nbytes
        if name is None:
            continue  # the RoPE tables: the converter computes them
        if name.startswith(("model.", "lm_head")):
            tensors[f"{name}.weight"] = tensor
            continue
        for layer in range(n_layers):
            heads = {"self_attn.q_proj": n_heads, "self_attn.k_proj": n_kv_heads}.get(name)
            tensors[f"model.layers.{layer}.{name}.weight"] = permute(tensor[layer], heads) if heads else tensor[layer]

    index, data = {"__metadata__": {"format": "pt"}}, b""
    for name, tensor in tensors.items():
        tensor = np.ascontiguousarray(tensor)
        raw = (tensor.view(np.uint32) >> 16).astype(np.uint16).tobytes() if stored == "BF16" else tensor.tobytes()
        index[name] = {"dtype": stored, "shape": list(tensor.shape), "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    encoded = json.dumps(index).encode()
    (out / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (out / "config.json").write_text(json.dumps(dict(
        model_type="llama", hidden_size=dim, intermediate_size=hidden_dim, num_hidden_layers=n_layers,
        num_attention_heads=n_heads, num_key_value_heads=n_kv_heads, vocab_size=abs(vocab_size),
        max_position_embeddings=seq_len, rope_theta=10000.0, tie_word_embeddings=vocab_size > 0,
        hidden_act="silu", bos_token_id=1, eos_token_id=2), indent=2))
    (out / "tokenizer.model").write_bytes(sentencepiece_model((ROOT / "tok512.bin").read_bytes(), abs(vocab_size)))
    (out / "settings.json").write_text(json.dumps(dict(
        name="stories260K from Hugging Face files", conversion=dict(dtype="float32"), options=dict(stop_tokens=[1]),
        generation=dict(steps=256, temperature=0.0), prompt="Once upon a time")))
    print(f"{out}: {', '.join(sorted(p.name for p in out.iterdir()))}")
