# tests/gguf_fetch.py (T.357.6; out of .github/workflows/gguf.yml, where two jobs each had a copy of it in a here-document)
# The files tests/gguf_check.py tensors compares, into one folder: an original's config.json, its tokenizer and its
# safetensors (one file, or the shards its index names) from huggingface.co, and a GGUF as model.gguf.
#
#   python3 tests/gguf_fetch.py <repo>@<revision> <repo>@<revision>/<file>.gguf <folder> [--either-tokenizer]
#
# Each by tests/fetching.py's download(): held to its length and asked for again where it stopped short (T.307: of 80
# GGUFs in one run, six came short), where the connection broke and where the server failed, four times in all; each
# try that failed is a line. --either-tokenizer: tokenizer.json and tokenizer.model, whichever the original has (a
# candidate of stage 2: each is compared); without it tokenizer.json, which must be there.
# T.517: an original with no safetensors at all (LINE's japanese-large-lm 1.7B instruction-sft has only PyTorch's
# pytorch_model.bin) has its pickle read without PyTorch (only the globals that rebuild tensors, as convert_hf.py) and
# written beside it as model.safetensors, the tensors as they are stored, so that gguf_check.py reads one kind.
# HF_ENDPOINT: another hub than huggingface.co (tests/fetching-check.ts's, on this machine).
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


def torch_tensors(path):
    """{name: array} of PyTorch's zip checkpoint, without PyTorch: only the globals that rebuild float32 and float16
    tensors unpickle. convert_hf.py's load_torch_pickle() in the form of LINE's 1.7B instruction-sft, which it does not
    read: saved by DeepSpeed, every tensor is a view of one storage of 3.3 GB, which that reads again for each tensor
    (a runner of 16 GB was shut down, runs 38731171773 and 38732582106), and its state dict carries attributes (the
    _metadata of torch.save), which a plain dict cannot take (run 38735291103)."""
    import pickle
    import zipfile
    import numpy as np
    archive = zipfile.ZipFile(path)
    prefix = next(name for name in archive.namelist() if name.endswith("/data.pkl"))[:-len("data.pkl")]
    readers = {"FloatStorage": np.float32, "HalfStorage": np.float16}
    storages = {}

    class Held(dict):
        """an OrderedDict with the attributes pickle sets on it"""

    def rebuild_tensor(storage, storage_offset, size, stride, *unused):
        array = storage[storage_offset:storage_offset + int(np.prod(size))].reshape(size)
        assert tuple(stride) == tuple(s // array.itemsize for s in array.strides), "non-contiguous tensor"
        return array

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("collections", "OrderedDict"):
                return Held
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild_tensor
            if module == "torch" and name in readers:
                return readers[name]
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

        def persistent_load(self, pid):
            _, kind, key, _, _ = pid
            if key not in storages:
                storages[key] = np.frombuffer(archive.read(f"{prefix}data/{key}"), dtype=kind)
            return storages[key]

    return Unpickler(archive.open(f"{prefix}data.pkl")).load()


def pickled(path, target):
    """PyTorch's pickle at path as a safetensors file at target, each tensor in the type it is stored in (F16 or F32)."""
    import numpy as np
    tensors = {name: np.ascontiguousarray(tensor) for name, tensor in torch_tensors(path).items()}
    kinds = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}
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
