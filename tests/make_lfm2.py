# make_lfm2.py
# T260: a made-up LFM2 (convolution layers among attention layers) as files, for the checks of forward.js that take a
# model by its files (tests/forward-check.mjs, tests/threads-check.mjs, tests/gpu-hybrid-check.mjs: <out>.bin,
# <out>.tokenizer.bin and <out>.json, as tests/perplexity_prepare.py writes them). The real ones are too large to be
# built with the site (the smallest is 0.26 GB as int8), so this one stands in: random Hugging Face tensors
# (tests/conftest.py's lfm2_model) converted the way the page converts them (llama2_convert.Conversion), with rows of
# whole groups of 32 for the int8 kernels and grouped keys and values.
#
#   python tests/make_lfm2.py <out> [int8 | float32 | float16 | int6] [small | own | four]
#
# small (the default): the 350M's order of layers in 16 layers, a context of 1024.
# own: a classifier of its own, the 230M's order of layers, keys and values for every head, a dim that is no multiple
# of 4 times a head (dim 96, heads of 24): the kernel's rows of fours and what is left of them.
# four: four taps (no published model has them), a convolution layer first and last, a context of 4096.
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from tree import python_folder
sys.path.insert(0, python_folder(HERE.parent))
sys.path.insert(0, str(HERE))
from conftest import lfm2_model  # noqa: E402
from test_convert import safetensors_file  # noqa: E402
import llama2_convert  # noqa: E402

VOCAB = 320
SHAPES = {"small": dict(dim=64, block_ff_dim=192, kinds="ccaccaccacacacac", n_heads=4, n_kv_heads=2, seq_len=1024),
          "own": dict(dim=96, block_ff_dim=160, adjust=False, kinds="ccacacacacacac", n_heads=4, n_kv_heads=4, seq_len=1024,
                      shared=False),
          "four": dict(dim=128, block_ff_dim=384, kinds="cacccaccac", n_heads=8, n_kv_heads=2, taps=4, seq_len=4096)}


def main():
    out, dtype = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "int8"
    shape = SHAPES[sys.argv[3] if len(sys.argv) > 3 else "small"]
    tensors, config = lfm2_model(vocab_size=VOCAB, **shape)
    file = safetensors_file(tensors)
    size = struct.unpack("<Q", file[:8])[0]
    vocabulary = json.dumps({"added_tokens": [], "model": {"type": "Unigram", "unk_id": 0,
                             "vocab": [[f"w{i}", -float(i)] for i in range(VOCAB)]}}).encode()
    conversion = llama2_convert.Conversion(file[8:8 + size].decode(), 8 + size, json.dumps(config), vocabulary,
                                           "tokenizer.json", dtype=dtype, max_seq_len=shape["seq_len"], start=8 + size)
    conversion.feed(file[8 + size:])
    conversion.finish()
    Path(f"{out}.bin").write_bytes(bytes(conversion.checkpoint))
    Path(f"{out}.tokenizer.bin").write_bytes(conversion.tokenizer)
    Path(f"{out}.json").write_text(json.dumps(conversion.options))
    print(f"{out}.bin: {len(conversion.checkpoint)} bytes ({dtype}), {json.dumps(conversion.options)}")


if __name__ == "__main__":
    main()
