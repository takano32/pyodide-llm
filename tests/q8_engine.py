# q8_engine.py (the review of T245 and T247, a probe for CI, not for main): the engine (NumPy, float32) on the 4B's
# GGUF as the page reads it, for the logits of the 96 positions tests/reference_qwen35.py holds it to transformers at,
# and the negative log likelihood of the first tokens of a text, saved for tests/q8_compare.py to put against
# tests/q8_lab.py's (transformers on the original, and on its weights rounded to Q8_0 in Hugging Face's order).
#
#   python tests/q8_engine.py <directory> --text <file> --out <prefix> [--source gguf|safetensors] [--first 192] [--positions 96]
#
# The float32 checkpoint is 17 GB: a memory map, pinned as far as the memory allows (reference_qwen35.pinned).
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import reference_qwen35 as ref  # noqa: E402
from llama2_numpy import Llama  # noqa: E402

say = lambda *parts: print("Q8ENGINE", *parts, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", default="gguf")
    parser.add_argument("--first", type=int, default=192)
    parser.add_argument("--positions", type=int, default=96)
    parser.add_argument("--model", default="4B")
    arguments = parser.parse_args()
    import tokenizers

    directory, model = Path(arguments.directory), ref.LARGE[arguments.model]
    repo, revision = model["repo"], model["revision"]
    for file in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        ref.fetch(file, directory, repo, revision)
    if arguments.source == "gguf":
        folder = directory / "gguf"
        folder.mkdir(exist_ok=True)
        for file in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if not (folder / file).exists():
                (folder / file).symlink_to((directory / file).resolve())
        ref.fetch(model["gguf"][2], folder, model["gguf"][0], model["gguf"][1])
        source, out = folder, directory / "gguf-float32"
    else:
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        for shard in sorted(set(index["weight_map"].values())):
            ref.fetch(shard, directory, repo, revision)
        source, out = directory, directory / "float32"
    options = ref.prepared(source, out)
    data = np.memmap(f"{out}.bin", dtype=np.uint8, mode="r")
    llama = Llama(data, Path(f"{out}.tokenizer.bin").read_bytes(), kernels=None,
                  **{**options, "specials": ref.SPECIALS, "stop_tokens": [248044, 248046]})
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    bos = ref.BOS
    sentence = ([bos] + tokenizer.encode(ref.TEXT, add_special_tokens=False).ids)[:arguments.positions]
    tokens = tokenizer.encode(Path(arguments.text).read_text(), add_special_tokens=False).ids[:1500]
    window = [bos] + tokens[:ref.WINDOW - 1]
    row = window[:arguments.first + 1]
    held = ref.pinned(data)
    say(f"{arguments.source}: {held} of the checkpoint's {len(data)} bytes pinned; bos {llama.bos}")
    began = time.perf_counter()
    logits = []
    for pos, token in enumerate(sentence):
        logits.append(llama.forward(token, pos).copy())
        if pos % 16 == 15:
            say(f"{arguments.source}: the sentence at position {pos + 1}, {time.perf_counter() - began:.0f} s")
    np.save(f"{arguments.out}-sentence-logits.npy", np.stack(logits))
    began, nll = time.perf_counter(), []
    for pos in range(len(row) - 1):
        values = np.asarray(llama.forward(row[pos], pos), dtype=np.float64)
        values -= values.max()
        nll.append(-(values[row[pos + 1]] - math.log(np.exp(values).sum())))
        if pos % 32 == 31:
            say(f"{arguments.source}: the text at position {pos + 1}, {time.perf_counter() - began:.0f} s")
    np.save(f"{arguments.out}-nll.npy", np.array(nll))
    say(f"{arguments.source}: perplexity of the first {len(nll)} tokens {math.exp(sum(nll) / len(nll)):.3f} "
        f"({time.perf_counter() - began:.0f} s)")


if __name__ == "__main__":
    main()
