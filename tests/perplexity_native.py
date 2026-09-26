# perplexity_native.py
# The NumPy row of tests/perplexity.mjs in native Python, for checkpoints too big for Pyodide on a small machine
# (a float32 original is read through a memory map, not copied). Only NumPy computes here: the kernels need Pyodide.
# The float32 rows of NumPy and of the kernels agree to the last digit (AGENTS.md), so this is the original's value.
#
#   python3 tests/perplexity_native.py <out of tests/perplexity_prepare.py> <tokens> <text file> [--eps 1e-6]
#
# --eps: the epsilon of every RMSNorm in place of the one of the options (T124: to measure what Qwen3's 1e-6 against
# the 1e-5 the engine had until then was worth)
import json
import sys
import time
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "public"))
sys.path.insert(0, str(here))
from llama2_numpy import Llama  # noqa: E402
from perplexity import perplexity  # noqa: E402

args = sys.argv[1:]
eps = None
if "--eps" in args:
    at = args.index("--eps")
    eps = float(args[at + 1])
    del args[at:at + 2]
out, count, text = args[0], int(args[1]), Path(args[2]).read_text()
llama = Llama(np.memmap(f"{out}.bin", dtype=np.uint8, mode="r"), Path(f"{out}.tokenizer.bin").read_bytes(), kernels=None,
              **{**json.loads(Path(f"{out}.json").read_text()), **({"rms_norm_eps": eps} if eps else {})})
tokens = llama.tokenizer.encode(text)[:count]
started = time.perf_counter()
value, n = perplexity(llama, tokens, min(512, llama.seq_len))
print(json.dumps({"model": Path(out).name, "backend": llama.backend, "eps": llama.rms_norm_eps, "perplexity": value, "tokens": n,
                  "seconds": time.perf_counter() - started}))
