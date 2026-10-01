# t236_numpy_rows.py (T236's review, a probe for CI, not for main): the NumPy row of tests/perplexity.mjs (the int8
# weights widened to float32, activations not quantized) for several texts, the checkpoint loaded once. The rows of
# tests/t236_rows.mjs are held against these.
#
#   python tests/t236_numpy_rows.py <out of tests/perplexity_prepare.py, int8> <name> <tokens> <text file> ...
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "public"))
sys.path.insert(0, str(HERE))
from llama2_numpy import Llama  # noqa: E402
from perplexity import perplexity  # noqa: E402

out, name, count, *texts = sys.argv[1:]
llama = Llama(np.memmap(f"{out}.bin", dtype=np.uint8, mode="r"), Path(f"{out}.tokenizer.bin").read_bytes(), kernels=None,
              **json.loads(Path(f"{out}.json").read_text()))
for path in texts:
    tokens = llama.tokenizer.encode(Path(path).read_text())[:int(count)]
    began = time.perf_counter()
    value, n = perplexity(llama, tokens, min(512, llama.seq_len))
    print("T236ROWS", json.dumps({"path": name, "row": "NumPy int8", "text": Path(path).name, "tokens": n, "perplexity": value,
                                  "seconds": round(time.perf_counter() - began)}), flush=True)
