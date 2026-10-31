# make_ternary.py
# T.230, T.231: made-up ternary models as files, for the checks of forward.js that take a model by its files
# (tests/forward-check.ts, tests/threads-check.ts: <out>.bin, <out>.tokenizer.bin and <out>.json, as
# tests/perplexity_prepare.py writes them). The real ternary models are far too large to be built with the site, so
# these stand in: random Hugging Face tensors whose matrices are made ternary (every group of 128 along a row is -d, 0
# and +d of one d of its own, about a third each, as Prism ML's Ternary Bonsai models are), converted the way the page
# converts them (llama2_convert.Conversion), to the ternary dtype or to any other of the same values.
#
#   python tests/make_ternary.py <out> [ternary | int8 | float32 | int6] [qwen3 | own | hybrid | rotated | rotated-tied | large]
#
# qwen3 (the default): the shape of Ternary Bonsai 1.7B in small, a Qwen3 (norms of the heads of q and k, heads that
#   do not fill dim, grouped-query attention, the classifier shared with the embedding).
# own: the same with a classifier of its own and a final norm with outlier channels (T.92), which a ternary classifier
#   multiplies apart as an int8 one does.
# hybrid: the shape of Ternary Bonsai 2 27B in small, a Qwen3.5 (T.229: Gated DeltaNet layers, three value heads to a
#   key head) with a classifier of its own.
# rotated: hybrid said to be in a rotated basis (T.237: blocks of 128 with random signs), as the 27B's file is: the
#   engine turns every matrix's input before it is quantized and laid out for the ternary kernels, and the
#   embedding's rows back.
# large (T.620): qwen3 a model's width, for timing (tests/threads-compare.ts --prompt): dim 1024, an FFN of 3072, 8 layers,
#   16 heads of 64 and 8 of keys, a classifier of its own (about 100 million values, 26 MB ternary). Its numbers
#   are no model's: only how long its matrices take.
# rotated-tied: the same with the embedding as the classifier too (no output matrix: what version 2 of Prism ML's
#   rotated basis is for, `tied_output`, as the smaller models of a Qwen3.5 have it): one ternary table whose rows
#   the engine turns back where it embeds and multiplies by the turned input where it classifies (T.237's review).
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from made_up import basis, files, qwen35_model, qwen3, synthetic_weights  # noqa: E402
import llama2_convert  # noqa: E402

VOCAB = 384


def ternarized(tensors, seed=5):
    """The tensors with every matrix made ternary: each group of 128 along a row keeps the signs of its larger values
    (about two thirds of them) times one scale of its own, the rest are 0. The rows of every matrix are whole groups."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, tensor in tensors.items():
        if tensor.ndim == 2 and tensor.shape[1] % 128 == 0:
            groups = tensor.reshape(-1, 128)
            cut = np.quantile(np.abs(groups), 1 / 3, axis=1, keepdims=True)
            scales = (np.abs(groups).mean(axis=1, keepdims=True) * rng.uniform(0.5, 1.5, (len(groups), 1))).astype(np.float16)
            tensor = (np.sign(groups) * (np.abs(groups) > cut) * scales.astype(np.float32)).astype(np.float32).reshape(tensor.shape)
        out[name] = tensor
    return out


def model(kind):
    """(the Hugging Face tensors, config.json, the context, the header's metadata)"""
    if kind in ("hybrid", "rotated", "rotated-tied"):
        tensors, config = qwen35_model(dim=128, hidden_dim=256, n_layers=8, every=4, n_heads=4, n_kv_heads=2, head_dim=64,
                                       key_heads=2, value_heads=6, key_dim=64, value_dim=64, vocab_size=VOCAB, seq_len=1024,
                                       shared=kind == "rotated-tied")
        # (the widths a matrix reads: the residual stream, an attention's output and the FFN's inside, a linear layer's output)
        said = basis(128, {128, 256, 384})[0] if kind.startswith("rotated") else None
        return tensors, config, 1024, said and {llama2_convert.ROTATED: json.dumps(said)}
    if kind == "large":
        settings, weights = synthetic_weights(dim=1024, hidden_dim=3072, n_layers=8, n_heads=16, n_kv_heads=8, vocab_size=VOCAB,
                                              seq_len=1024, shared=False, head_size=64)
        tensors, config = qwen3(settings, weights, False)
        return tensors, {**config, "rms_norm_eps": 1e-6}, 1024, None
    settings, weights = synthetic_weights(dim=128, hidden_dim=384, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=VOCAB,
                                          seq_len=1024, shared=kind != "own", head_size=64)
    if kind == "own":
        # a final norm with outlier channels (llama2_numpy.OUTLIER_RATIO: the largest 4 times the median and more)
        weights["rms_final_weight"][[3, 40, 77]] *= 9
    tensors, config = qwen3(settings, weights, kind != "own")
    return tensors, {**config, "rms_norm_eps": 1e-6}, 1024, None


def main():
    out, dtype = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ternary"
    tensors, config, seq_len, metadata = model(sys.argv[3] if len(sys.argv) > 3 else "qwen3")
    files(out, ternarized(tensors), config, dtype, seq_len, VOCAB, metadata=metadata)


if __name__ == "__main__":
    main()
