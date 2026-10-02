# by_layer_logits.py (the review of T253): the page's engine against transformers over the whole depth of a model that no
# runner holds in float32 (Granite 4.2 8B: 35 GB). tests/logits.mjs saves the page's engine's logits (int8, 7-bit
# activations, a token at a time) at the positions of a text; this computes transformers' for the very same ids a layer at
# a time (reference_llama.ByLayer, in float32) and holds the two to each other: the largest and the mean difference of a
# logit, how many positions have the same most likely token, and the Kullback-Leibler distance of the engine's
# next-token distribution from transformers' (nats a position). A conversion that is wrong somewhere in the depth shows
# here as a whole as the quantization noise (what the 3B, which fits, shows) is a few tenths of a logit.
#
#   (in CI: tests.yml's extra=, after tests/write.sh with WRITER=logits.mjs has saved the engine's logits)
#   OUT=.tmp/engine TEXT=.tmp/ja.txt TOKENS=96 WRITER=logits.mjs bash tests/write.sh hf-granite-4.2-8b
#   bash tests/reference_llama.sh --run tests/by_layer_logits.py hf-granite-4.2-8b .tmp/engine
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
import format_check  # noqa: E402
import reference_llama  # noqa: E402
from reference_llama import say  # noqa: E402


def log_softmax(rows):
    rows = np.asarray(rows, dtype=np.float64)
    rows = rows - rows.max(axis=1, keepdims=True)
    return rows - np.log(np.exp(rows).sum(axis=1, keepdims=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("engine", help="the OUT of tests/logits.mjs: OUT.ids.json and OUT.logits.f32")
    parser.add_argument("--directory", default=str(HERE.parent / ".tmp" / "by-layer"))
    args = parser.parse_args()
    entry = next(entry for entry in format_check.entries() if entry["id"] == args.model)
    source = entry["hf"].get("vocabulary") or entry["hf"]
    original = Path(subprocess.check_output([sys.executable, str(HERE / "hf_fetch.py"), f"hf:{source['repo']}@{source['revision']}",
                                             str(Path(args.directory) / "weights")], text=True).splitlines()[-1])
    ids = json.loads(Path(f"{args.engine}.ids.json").read_text())
    engine = np.fromfile(f"{args.engine}.logits.f32", dtype=np.float32)
    engine = engine.reshape(len(ids), -1)
    say(f"{args.model}: {len(ids)} ids ({ids[:8]} …), the engine's logits {engine.shape}, {source['repo']}@{source['revision'][:8]}")
    by = reference_llama.ByLayer(original)
    theirs = by.logits(by.hidden([ids])[0]).numpy()
    assert theirs.shape == engine.shape, (theirs.shape, engine.shape)
    largest, mean, same, margin = reference_llama.differences(engine, theirs)
    ours_log, theirs_log = log_softmax(engine), log_softmax(theirs)
    kl = (np.exp(theirs_log) * (theirs_log - ours_log)).sum(axis=1)
    say(f"{args.model}: logits of transformers (by layer): min {theirs.min():.3f}, max {theirs.max():.3f}, mean {theirs.mean():.4f}, std {theirs.std():.4f}")
    say(f"{args.model}: logits of the page's engine: min {engine.min():.3f}, max {engine.max():.3f}, mean {engine.mean():.4f}, std {engine.std():.4f}")
    say(f"{args.model}: the page's engine against transformers over {len(ids)} positions: largest difference {largest:.3e}, mean {mean:.3e}, the same "
        f"most likely token at {same} of {len(ids)} positions (the largest gap between transformers' best two where it is not: {margin:.3e}); "
        f"the engine's distribution from transformers' (nats a position): mean {kl.mean():.3e}, largest {kl.max():.3e} (position {int(kl.argmax())})")
    # how the difference grows with the position: the first and the last 16 positions (a layer wrong deep in the model, or a position-dependent
    # error, shows as a trend)
    gap = np.abs(engine.astype(np.float64) - theirs.astype(np.float64)).mean(axis=1)
    say(f"{args.model}: mean difference of a position's logits, first 16 positions {gap[:16].mean():.3e}, last 16 {gap[-16:].mean():.3e}")


if __name__ == "__main__":
    main()
