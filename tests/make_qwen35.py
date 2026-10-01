# make_qwen35.py
# T229: a made-up Qwen3.5 (hybrid attention) as files, for the checks of forward.js that take a model by its files
# (tests/forward-check.mjs, tests/threads-check.mjs: <out>.bin, <out>.tokenizer.bin and <out>.json, as
# tests/perplexity_prepare.py writes them). No real model of the kind is small enough to be built with the site, so
# this one stands in: random Hugging Face tensors (tests/conftest.py's qwen35_model) converted the way the page
# converts them (llama2_convert.Conversion), with rows of whole groups of 32 for the int8 kernels, two value heads to
# a key head, heads that do not fill dim, and RoPE over a quarter of a head.
#
#   python tests/make_qwen35.py <out> [int8 | float32 | float16 | int6] [small | state | wide]
#
# small (the default): a context of 1024, heads and states of a few kilobytes: for the numbers.
# state: a context of 4096 and value heads as large as the real models' (16 of 128 by 128), so that the state of the
# linear-attention layers (6.3 MB, held twice) and the keys and values of the two attending layers (8.4 MB in
# float32) outweigh the rest: for what forward.js puts after the checkpoint against footprint().
# wide (the review of T229): heads of 256, as every real Qwen3.5 has.
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
from conftest import qwen35_model  # noqa: E402
from test_convert import safetensors_file  # noqa: E402
import llama2_convert  # noqa: E402

VOCAB = 320
SHAPES = {"small": dict(dim=64, hidden_dim=128, n_heads=4, n_kv_heads=2, head_dim=32, key_heads=2, value_heads=4,
                        key_dim=16, value_dim=16, seq_len=1024),
          "state": dict(dim=256, hidden_dim=512, n_heads=4, n_kv_heads=2, head_dim=64, key_heads=8, value_heads=16,
                        key_dim=128, value_dim=128, seq_len=4096),
          # heads of 256 as every real Qwen3.5 has (the others' heads are 32 and 64), a key and value head of 32: the
          # attention kernels on heads that wide, in float16 too (a 4B or a 9B keeps float16 keys and values past 4 GiB)
          "wide": dict(dim=256, hidden_dim=512, n_heads=2, n_kv_heads=1, head_dim=256, key_heads=2, value_heads=4,
                       key_dim=32, value_dim=32, seq_len=1024)}


def main():
    out, dtype = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "int8"
    shape = SHAPES[sys.argv[3] if len(sys.argv) > 3 else "small"]
    tensors, config = qwen35_model(n_layers=8, every=4, vocab_size=VOCAB, **shape)
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
