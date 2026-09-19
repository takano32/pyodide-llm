# convert_hf.py
# Convert a Hugging Face Llama checkpoint into what llama2_numpy.py loads, with nothing but NumPy:
#   <out>.bin            llama2.c "legacy" checkpoint (7 int header, then the tensors), float32 or float16
#   <out>.tokenizer.bin  llama2.c tokenizer format, from the sentencepiece model
# It runs when the site is deployed, so no converted binary has to live in the repository.
#
#   python3 convert_hf.py <directory with config.json, pytorch_model.bin | model.safetensors,
#                          spiece.model | tokenizer.model | tokenizer.json> <out> [float32|float16] [max seq_len]
import json
import pickle
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------------------------------------ weights
def bfloat16(raw):
    # NumPy has no bfloat16, but a bfloat16 is exactly the upper half of a float32: widening is a shift
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def load_safetensors(path):
    with open(path, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_size))
    # memory-mapped: a tensor is only read when it is written out
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_size)
    readers = {"F32": lambda raw: np.frombuffer(raw, dtype=np.float32),
               "F16": lambda raw: np.frombuffer(raw, dtype=np.float16), "BF16": bfloat16}
    class Tensors(dict):
        def __getitem__(self, name):
            info = dict.__getitem__(self, name)
            begin, end = info["data_offsets"]
            return readers[info["dtype"]](data[begin:end]).reshape(info["shape"])

    return Tensors({name: info for name, info in header.items() if name != "__metadata__"})


def load_torch_pickle(path):
    """Read a PyTorch zip checkpoint without PyTorch. Only tensor-rebuilding globals are allowed to unpickle."""
    archive = zipfile.ZipFile(path)
    prefix = next(name for name in archive.namelist() if name.endswith("/data.pkl"))[:-len("data.pkl")]
    readers = {"FloatStorage": lambda raw: np.frombuffer(raw, dtype=np.float32),
               "HalfStorage": lambda raw: np.frombuffer(raw, dtype=np.float16), "BFloat16Storage": bfloat16}

    def rebuild_tensor(storage, storage_offset, size, stride, *unused):
        array = storage[storage_offset:storage_offset + int(np.prod(size))].reshape(size)
        assert tuple(stride) == tuple(s // array.itemsize for s in array.strides), "non-contiguous tensor"
        return array

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("collections", "OrderedDict"):
                return dict
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild_tensor
            if module == "torch" and name in readers:
                return readers[name]
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

        def persistent_load(self, pid):
            _, reader, key, _, _ = pid
            return reader(archive.read(f"{prefix}data/{key}"))

    return Unpickler(archive.open(f"{prefix}data.pkl")).load()


def convert_weights(directory, out_path, dtype, max_seq_len):
    config = json.loads((directory / "config.json").read_text())
    assert config["model_type"] == "llama" and not config.get("rope_scaling"), "only plain Llama models"
    dim, n_layers = config["hidden_size"], config["num_hidden_layers"]
    n_heads, n_kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    # the KV cache grows with seq_len, so a long context can be cut down for the browser
    head_size, seq_len = dim // n_heads, min(config["max_position_embeddings"], max_seq_len)
    safetensors = directory / "model.safetensors"
    tensors = load_safetensors(safetensors) if safetensors.exists() else load_torch_pickle(directory / "pytorch_model.bin")

    def permute_reverse(w, heads):
        # Hugging Face stores each head of wq/wk as [first halves, second halves] (rotate_half);
        # llama2.c rotates adjacent pairs, so interleave the two halves again
        return w.reshape(heads, 2, head_size // 2, w.shape[1]).transpose(0, 2, 1, 3).reshape(w.shape)

    def layers(name, transform=lambda w: w):
        # callables, so that only one converted tensor is in memory at a time
        return [lambda i=i: transform(tensors[f"model.layers.{i}.{name}.weight"]) for i in range(n_layers)]

    shared_classifier = config.get("tie_word_embeddings", False) or "lm_head.weight" not in tensors
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    frequencies = 1.0 / config.get("rope_theta", 10000.0) ** (np.arange(0, head_size, 2, dtype=np.float64) / head_size)
    ordered = [
        lambda: tensors["model.embed_tokens.weight"],
        *layers("input_layernorm"),
        *layers("self_attn.q_proj", lambda w: permute_reverse(w, n_heads)),
        *layers("self_attn.k_proj", lambda w: permute_reverse(w, n_kv_heads)),
        *layers("self_attn.v_proj"),
        *layers("self_attn.o_proj"),
        *layers("post_attention_layernorm"),
        *layers("mlp.gate_proj"),
        *layers("mlp.down_proj"),
        *layers("mlp.up_proj"),
        lambda: tensors["model.norm.weight"],
        lambda: np.cos(positions * frequencies),
        lambda: np.sin(positions * frequencies),
    ]
    if not shared_classifier:
        ordered.append(lambda: tensors["lm_head.weight"])

    vocab_size = config["vocab_size"]
    with open(out_path, "wb") as f:
        # negative vocab size is llama2.c's way of signaling an unshared classifier
        f.write(struct.pack("<7i", dim, config["intermediate_size"], n_layers, n_heads, n_kv_heads,
                            vocab_size if shared_classifier else -vocab_size, seq_len))
        for tensor in ordered:
            f.write(np.ascontiguousarray(tensor(), dtype=dtype).tobytes())
    return vocab_size


# ---------------------------------------------------------------------------------------------- tokenizer
def protobuf_fields(data):
    """Yield (field number, value) of one protobuf message; nested messages come back as bytes."""
    i = 0

    def varint():
        nonlocal i
        value = shift = 0
        while True:
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    while i < len(data):
        key = varint()
        field, wire_type = key >> 3, key & 7
        if wire_type == 0:
            yield field, varint()
        elif wire_type == 1:
            yield field, data[i:i + 8]
            i += 8
        elif wire_type == 2:
            size = varint()
            yield field, data[i:i + size]
            i += size
        elif wire_type == 5:
            yield field, data[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")


def sentencepiece_pieces(model_path):
    NORMAL, USER_DEFINED = 1, 4
    for field, value in protobuf_fields(model_path.read_bytes()):
        if field == 1:  # ModelProto.pieces
            piece = dict(protobuf_fields(value))
            score = struct.unpack("<f", piece[2])[0] if 2 in piece else 0.0
            yield piece.get(1, b"").decode("utf-8"), score, piece.get(3, NORMAL) in (NORMAL, USER_DEFINED)


def tokenizer_json_pieces(json_path):
    tokenizer = json.loads(json_path.read_text())
    assert tokenizer["model"]["type"] == "Unigram", "only unigram tokenizer.json files"
    special = {token["content"] for token in tokenizer["added_tokens"] if token["special"]}
    for id, (text, score) in enumerate(tokenizer["model"]["vocab"]):
        is_byte = len(text) == 6 and text.startswith("<0x") and text.endswith(">")
        yield text, score, not (is_byte or text in special or id == tokenizer["model"].get("unk_id"))


def convert_tokenizer(directory, out_path, vocab_size):
    model = next((p for p in (directory / "spiece.model", directory / "tokenizer.model") if p.exists()), None)
    pieces = []
    for text, score, matchable in sentencepiece_pieces(model) if model else tokenizer_json_pieces(directory / "tokenizer.json"):
        # control, unknown and byte pieces must never match user text: llama2_numpy.py skips such scores
        pieces.append((score if matchable else -1e9, text.replace("\u2581", " ").encode("utf-8")))
    # a model can have more embedding rows than the tokenizer has pieces
    pieces += [(-1e9, b"")] * (vocab_size - len(pieces))
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", max(len(text) for _, text in pieces)))
        for score, text in pieces:
            f.write(struct.pack("<fi", score, len(text)) + text)


if __name__ == "__main__":
    directory, out = Path(sys.argv[1]), sys.argv[2]
    dtype = np.dtype(sys.argv[3] if len(sys.argv) > 3 else "float32")
    max_seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    vocab_size = convert_weights(directory, f"{out}.bin", dtype, max_seq_len)
    convert_tokenizer(directory, f"{out}.tokenizer.bin", vocab_size)
    print(f"{out}.bin: {Path(f'{out}.bin').stat().st_size:,} bytes ({dtype}), vocabulary {vocab_size}")
