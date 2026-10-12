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
# HF_ENDPOINT: another hub than huggingface.co (tests/fetching-check.mjs's, on this machine).
import json
import os
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
        index = json.loads(get(at("model.safetensors.index.json"), "model.safetensors.index.json").read_text())
        for shard in sorted(set(index["weight_map"].values())):
            get(at(shard), shard)
    get(f"{HUB}/{gguf_repo}/resolve/{gguf_revision}/{gguf_file}", "model.gguf")
    print(sorted(os.listdir(out)))


if __name__ == "__main__":
    arguments = [argument for argument in sys.argv[1:] if not argument.startswith("--")]
    if len(arguments) != 3 or set(sys.argv[1:]) - set(arguments) - {"--either-tokenizer"}:
        raise SystemExit("usage: gguf_fetch.py <repo>@<revision> <repo>@<revision>/<file>.gguf <folder> [--either-tokenizer]")
    fetch(arguments[0], arguments[1], Path(arguments[2]), either="--either-tokenizer" in sys.argv[1:])
