# perplexity_prepare.py
# A Hugging Face model converted the way the page converts it (llama2_convert.Conversion, fed the file in order),
# for tests/perplexity.mjs and tests/perplexity_native.py (T85): <out>.bin, <out>.tokenizer.bin and <out>.json,
# the options the page would give Llama(). convert_hf.py does not say those options; the page's path does.
#
#   python3 tests/perplexity_prepare.py <directory with config.json, model.safetensors and the tokenizer> <out> [int8|float32]
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "public"))
from llama2_convert import Conversion  # noqa: E402

CHUNK = 8 << 20

directory, out = Path(sys.argv[1]), sys.argv[2]
dtype = sys.argv[3] if len(sys.argv) > 3 else "int8"
tokenizer = next(p for p in (directory / n for n in ("tokenizer.json", "spiece.model", "tokenizer.model")) if p.exists())
data = np.memmap(directory / "model.safetensors", dtype=np.uint8, mode="r")
size = struct.unpack("<Q", bytes(data[:8]))[0]
conversion = Conversion(bytes(data[8:8 + size]).decode(), 8 + size, (directory / "config.json").read_text(),
                        tokenizer.read_bytes(), tokenizer.name, dtype=dtype)
for start in range(0, len(data), CHUNK):
    conversion.feed(bytes(data[start:start + CHUNK]))
conversion.finish()
Path(f"{out}.bin").write_bytes(conversion.checkpoint)
Path(f"{out}.tokenizer.bin").write_bytes(conversion.tokenizer)
options = {key: value for key, value in conversion.options.items() if key != "template"}
Path(f"{out}.json").write_text(json.dumps(options))
print(f"{out}.bin: {Path(f'{out}.bin').stat().st_size:,} bytes, options {options}")
