# make_spark.py
# T342: a made-up Spark-X2.5 (three local layers of every four that turn their whole heads by a base of their own, the
# others a quarter of each head; a gate of each head's output; the exact GELU in the FFN's gate) as files, for the checks
# of forward.js that take a model by its files (tests/forward-check.mjs, tests/threads-check.mjs,
# tests/gpu-hybrid-check.mjs: <out>.bin, <out>.tokenizer.bin and <out>.json, as tests/perplexity_prepare.py writes them).
# The real ones are too large to be built with the site (the smaller is 1.8 GB as int8), so this one stands in: random
# Hugging Face tensors (tests/made_up.py's spark_model) converted the way the page converts them (llama2_convert.Conversion),
# with rows of whole groups of 32 for the int8 kernels, and values a tenth of a standard deviation (a stream through 12
# layers and no norm behind them grows past where the line of a made-up model, a relative 0.5, holds int8 to NumPy).
#
#   python tests/make_spark.py <out> [int8 | float32 | float16 | int6] [small | own | long]
#
# small (the default): the 1.7B's order of layers (three local ones, then one that attends all) in 12 layers, eight
# heads of 32 in a dim of 64 (q twice as wide as the stream, as the 4B's) and two heads of keys and values, of which RoPE
# turns 8 values in the full-attention layers, a window of 48 positions (a check of 128 positions is past it for 80 of
# them, and a prompt's block of 16 straddles its edge), a context of 1024.
# own: a classifier of its own, heads that fill the width (dim 96, heads of 24: the kernels' rows of fours and what is
# left of them; half of each turned), keys and values for every head, a window of 5 (shorter than a prompt's block).
# long: a window of 300 positions, past the 256 the keys and values begin with room for, in a context of 4096: the
# cache has grown before a layer first leaves a position out.
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from made_up import files, spark_model  # noqa: E402

VOCAB = 320
SHAPES = {"small": dict(dim=64, hidden_dim=128, kinds="ssslssslsssl", n_heads=8, n_kv_heads=2, head_dim=32, window=48,
                        seq_len=1024, local_theta=10000.0, spread=0.1),
          "own": dict(dim=96, hidden_dim=160, kinds="sslssl", n_heads=4, n_kv_heads=4, head_dim=24, window=5, seq_len=1024,
                      shared=False, part=0.5, spread=0.1),
          "long": dict(dim=128, hidden_dim=256, kinds="ssslsssl", n_heads=8, n_kv_heads=2, head_dim=32, window=300,
                       seq_len=4096, local_theta=10000.0, spread=0.1)}


def main():
    out, dtype = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "int8"
    shape = SHAPES[sys.argv[3] if len(sys.argv) > 3 else "small"]
    tensors, config = spark_model(vocab_size=VOCAB, **shape)
    files(out, tensors, config, dtype, shape["seq_len"], VOCAB)


if __name__ == "__main__":
    main()
