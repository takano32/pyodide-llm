# make_gemma4.py
# T331.2: a made-up Gemma 4 (its language model: a Gemma 3's layers, the full-attention ones with heads of a size of
# their own turned in part by the angles of the whole head, the heads of v normalized, a number every layer ends by
# multiplying the stream by, the logits capped) as files, for the checks of forward.js that take a model by its files
# (tests/forward-check.mjs, tests/threads-check.mjs: <out>.bin, <out>.tokenizer.bin and <out>.json, as
# tests/perplexity_prepare.py writes them). The real ones are too large to be built with the site (E2B is 5 GB as
# int8), so this one stands in: random Hugging Face tensors (tests/made_up.py's gemma4_model) converted the way the page
# converts them (llama2_convert.Conversion), with rows of whole groups of 32 for the int8 kernels.
#
#   python tests/make_gemma4.py <out> [int8 | float32 | float16 | int6] [small | own | long]
#
# small (the default): E2B's order of layers (four sliding ones, then one that attends all) in 12 layers, four heads of
# 32 in a dim of 64 and the full ones' of 64, one head of keys and values, a window of 48 positions (a check of 128
# positions is past it for 80 of them, and a prompt's block of 16 straddles its edge), a context of 1024.
# own: a classifier of its own, keys and values for every head, heads of 24 and 48 (the kernels' rows of fours and what
# is left of them), a window of 5 (shorter than a prompt's block).
# long: a window of 300 positions, past the 256 the keys and values begin with room for, in a context of 4096: the
# cache has grown before a layer first leaves a position out.
# shared (T330.2): small's widths in 14 layers (two full ones before the shared), its last five layers read the keys and values of the last of their kind before them, and
# have an FFN twice as wide (E2B's).
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from made_up import files, gemma4_model  # noqa: E402

VOCAB = 320
SHAPES = {"small": dict(dim=64, hidden_dim=128, kinds="sssslsssslsl", n_heads=4, n_kv_heads=1, head_dim=32, wide_head=64,
                        window=48, seq_len=1024, whole=True),
          "own": dict(dim=96, hidden_dim=160, kinds="sslssl", n_heads=4, n_kv_heads=4, head_dim=24, wide_head=48, window=5,
                      seq_len=1024, shared=False),
          "shared": dict(dim=64, hidden_dim=128, kinds="sslsslssssslsl", n_heads=4, n_kv_heads=1, head_dim=32, wide_head=64,
                         window=48, seq_len=1024, sharing=5, double=True),
          "long": dict(dim=128, hidden_dim=256, kinds="sslsslsl", n_heads=4, n_kv_heads=2, head_dim=32, wide_head=64,
                       window=300, seq_len=4096)}


def main():
    out, dtype = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "int8"
    shape = SHAPES[sys.argv[3] if len(sys.argv) > 3 else "small"]
    tensors, config = gemma4_model(vocab_size=VOCAB, **shape)
    files(out, tensors, config, dtype, shape["seq_len"], VOCAB)


if __name__ == "__main__":
    main()
