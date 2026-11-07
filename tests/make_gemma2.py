# make_gemma2.py
# T612: a made-up Gemma 2 (a Gemma 3's layers without the norms of the heads, one local layer of every two, its
# attention's scores and its logits capped by a tanh) as files, for the checks of forward.js that take a model by its
# files (tests/forward-check.ts, tests/threads-check.ts: <out>.bin, <out>.tokenizer.bin and <out>.json, as
# tests/perplexity_prepare.py writes them). The real ones are too large to be built with the site (the 2B is 2.6 GB as
# int8), so this one stands in: random Hugging Face tensors (tests/made_up.py's gemma2_model) converted the way the page
# converts them (llama2_convert.Conversion), with rows of whole groups of 32 for the int8 kernels. Its caps are small (4
# of the scores, 6 of the logits), so that the tanh bites on these numbers as the real 50 and 30 do on the real ones.
#
#   python tests/make_gemma2.py <out> [int8 | float32 | float16 | int6] [small | own]
#
# small (the default): ten layers, four heads of 32 in a dim of 64 and two heads of keys and values (a grouped-query
# model: its keys and values float32 on a shared memory), a window of 48 positions (a check of 128 positions is past
# it for 80 of them), q multiplied by the root of 32 / 40 (query_pre_attn_scalar 40), a context of 1024.
# own: a classifier of its own, heads that fill the width (dim 96, heads of 24: the kernels' rows of fours and what is
# left of them), keys and values for every head (float16 on a shared memory: the capped kernel of float16 keys), a
# window of 5 (shorter than a prompt's block).
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from made_up import files, gemma2_model  # noqa: E402

VOCAB = 320
CAPS = dict(scores_cap=4.0, logits_cap=6.0)
SHAPES = {"small": dict(dim=64, hidden_dim=128, kinds="slslslslsl", n_heads=4, n_kv_heads=2, head_dim=32, window=48,
                        seq_len=1024, scalar=40),
          "own": dict(dim=96, hidden_dim=160, kinds="slslsl", n_heads=4, n_kv_heads=4, head_dim=24, window=5, seq_len=1024,
                      shared=False)}


def main():
    out, dtype = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "int8"
    shape = SHAPES[sys.argv[3] if len(sys.argv) > 3 else "small"]
    tensors, config = gemma2_model(vocab_size=VOCAB, **CAPS, **shape)
    files(out, tensors, config, dtype, shape["seq_len"], VOCAB)


if __name__ == "__main__":
    main()
