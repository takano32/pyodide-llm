# tests/gguf_fetch.py (T357.6; out of .github/workflows/gguf.yml, where two jobs each had a copy of it in a here-document)
# The files tests/gguf_check.py tensors compares, into one folder: an original's config.json, its tokenizer and its
# safetensors (one file, or the shards its index names) from huggingface.co, and a GGUF as model.gguf.
#
#   python3 tests/gguf_fetch.py <repo>@<revision> <repo>@<revision>/<file>.gguf <folder> [--either-tokenizer]
#
# Each by tests/fetching.py's download(): held to its length and asked for again where it stopped short (T307: of 80
# GGUFs in one run, six came short), where the connection broke and where the server failed, four times in all; each
# try that failed is a line. --either-tokenizer: tokenizer.json and tokenizer.model, whichever the original has (a
# candidate of stage 2: each is compared); without it tokenizer.json, which must be there.
# T517: an original with no safetensors at all (LINE's japanese-large-lm 1.7B instruction-sft has only PyTorch's
# pytorch_model.bin) has its pickle read as convert_hf.py reads it (no PyTorch; only the globals that rebuild tensors)
# and written beside it as model.safetensors, the tensors as they are stored, so that gguf_check.py reads one kind.
# HF_ENDPOINT: another hub than huggingface.co (tests/fetching-check.mjs's, on this machine).
import json
import os
import struct
import sys
import urllib.error
from pathlib import Path

from fetching import download

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def fetch(original, gguf, out, either=False):
    (repo, revision), (gguf_repo, rest) = original.split("@"), gguf.split("@")
    gguf_revision, gguf_file = rest.split("/", 1)
    at = lambda name: f"{HUB}/{repo}/resolve/{revision}/{name}"

    def get(url, name):
        again = lambda error: print(name, "is asked for again:", error, flush=True)
        return download(url, out / name, tries=4, timeout=300, said=again)

    get(at("config.json"), "config.json")
    if either:
        for name in ("tokenizer.json", "tokenizer.model"):
            try:
                get(at(name), name)
            except urllib.error.HTTPError as error:
                print(name, error.code)
    else:
        get(at("tokenizer.json"), "tokenizer.json")
    try:
        get(at("model.safetensors"), "model.safetensors")
    except urllib.error.HTTPError:
        try:
            index = json.loads(get(at("model.safetensors.index.json"), "model.safetensors.index.json").read_text())
        except urllib.error.HTTPError:
            pickled(get(at("pytorch_model.bin"), "pytorch_model.bin"), out / "model.safetensors")
        else:
            for shard in sorted(set(index["weight_map"].values())):
                get(at(shard), shard)
    get(f"{HUB}/{gguf_repo}/resolve/{gguf_revision}/{gguf_file}", "model.gguf")
    print(sorted(os.listdir(out)))


def pickled(path, target):
    """PyTorch's pickle at path as a safetensors file at target, each tensor in the type it is read as (F16 or F32)."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from convert_hf import load_torch_pickle
    tensors = {name: np.ascontiguousarray(tensor) for name, tensor in load_torch_pickle(path).items()}
    kinds = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}  # (a bfloat16 is read as float32)
    header, at = {"__metadata__": {"format": "pt"}}, 0
    for name, tensor in tensors.items():
        header[name] = {"dtype": kinds[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [at, at + tensor.nbytes]}
        at += tensor.nbytes
    encoded = json.dumps(header).encode()
    with open(target, "wb") as file:
        file.write(struct.pack("<Q", len(encoded)) + encoded)
        for tensor in tensors.values():
            file.write(tensor.tobytes())
    path.unlink()


if __name__ == "__main__":
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    if len(arguments) != 3 or set(sys.argv[1:]) - set(arguments) - {"--either-tokenizer"}:
        raise SystemExit("usage: gguf_fetch.py <repo>@<revision> <repo>@<revision>/<file>.gguf <folder> [--either-tokenizer]")
    fetch(arguments[0], arguments[1], Path(arguments[2]), either="--either-tokenizer" in sys.argv[1:])
