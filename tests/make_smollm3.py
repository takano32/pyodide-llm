# make_smollm3.py
# T255: a made-up SmolLM3 (a Llama some of whose layers RoPE leaves alone) as files, for the checks of forward.js that
# take a model by its files (tests/forward-check.mjs: <out>.bin, <out>.tokenizer.bin and <out>.json, as
# tests/perplexity_prepare.py writes them). The real one is too large to be built with the site (3.3 GB as int8), so
# this one stands in: random Hugging Face tensors (tests/made_up.py's synthetic_weights) converted the way the page
# converts them (llama2_convert.Conversion), with rows of whole groups of 32 for the int8 kernels and grouped keys and
# values. Eight layers, every fourth left alone, as the published 3B's 36.
#
#   python tests/make_smollm3.py <out> [int8 | float32 | float16 | int6]
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from made_up import files, hugging_face, synthetic_weights  # noqa: E402

VOCAB, CONTEXT = 320, 1024


def main():
    out, dtype = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "int8"
    settings, weights = synthetic_weights(dim=64, hidden_dim=192, n_layers=8, n_heads=4, n_kv_heads=2, vocab_size=VOCAB,
                                          seq_len=CONTEXT)
    tensors, config = hugging_face(settings, weights, True)
    config = {**config, "model_type": "smollm3", "no_rope_layer_interval": 4, "use_sliding_window": False}
    files(out, tensors, config, dtype, CONTEXT, VOCAB)


if __name__ == "__main__":
    main()
