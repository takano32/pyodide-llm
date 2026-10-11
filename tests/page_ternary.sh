#!/usr/bin/env bash
# tests/page_ternary.sh (T230, T231): a real ternary model of the list on the page's forward pass (public/forward.js),
# as the ternary dtype (its weights as they are, on the ternary kernels) and as int8 (the same weights widened, on the
# int8 kernels: how T235 ran it), for CI (tests.yml's extra=, after `make kernels` and with the suite's pyodide: the
# development machine runs no model of gigabytes):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/page_ternary.sh" --ref <branch> --grep "ternary|perplexity|thread|converted"
#   bash tests/page_ternary.sh [<id of the list> = hf-ternary-bonsai-1.7b] [en | ja] [the parts, e.g. "convert perplexity"]
#
# The parts (all of them by default):
#   convert     the page's conversion of the GGUF to either dtype in Pyodide, under cProfile (tests/profile-convert.mjs)
#   ptq1_0      the same of a PTQ1_0 file of the same weights, made here, and the checkpoint the same byte for byte
#   perplexity  on 1500 tokens of Wikipedia (T85's articles, English or Japanese: the first 1500 tokens of the three
#               joined, which is the beginning of the first article, Mount Fuji or 富士山: the review of T230 ran them
#               one by one and got this run's numbers for the first): the float32 of the file's values and
#               the int8 weights in native NumPy (widened to float32 they are past Pyodide), then both dtypes on the
#               kernels (tests/perplexity.mjs --file). The ternary dtype in NumPy is the float32 row: its first logits
#               are held to the float32 file's to the bit here
#   compare     the logits of the two dtypes on the same 256 tokens (tests/ternary-compare.mjs)
#   threads     the logits the same to the bit with 1, 2, 4 and 8 threads, and the tokens a second of each count, the
#               two dtypes one after the other in one process (tests/threads-check.mjs)
set -euo pipefail
id="${1:-hf-ternary-bonsai-1.7b}"
language="${2:-en}"
parts="${3:-convert ptq1_0 perplexity compare threads}"
dir="${RUNNER_TEMP:-.tmp}/ternary"
has() { [[ " $parts " == *" $1 "* ]]; }
model=$(python tests/hf_fetch.py "$id" "$dir" | tail -1)
node tests/wikipedia.mjs "$language" "$dir/$language.txt"
text="$dir/$language.txt"
echo "ternary: $id, $(node -p 'require("os").cpus()[0].model') ($(uname -m)), Wikipedia ($language)"
if has convert; then
  for dtype in ternary int8; do
    echo "ternary: the conversion to $dtype"
    node tests/profile-convert.mjs "$model" "$dtype" | head -12 | cut -c1-200
  done
fi
dtypes="int8 ternary"
if has perplexity; then dtypes="float32 int8 ternary"; fi
for dtype in $dtypes; do
  python tests/perplexity_prepare.py "$model" "$dir/$dtype" "$dtype" 2>&1 | tail -1 | cut -c1-200
  echo "ternary: $dtype is $(stat -c %s "$dir/$dtype.bin") bytes"
done
if has ptq1_0; then
  # the same weights as a PTQ1_0 file (tests/make_ptq1_0.py: no model this small is published as one): the page's
  # conversion of it under cProfile, and the checkpoint it makes against the one of the PQ2_0 file
  gguf=$(ls "$model"/*.gguf | head -1)
  mkdir -p "$dir/ptq"
  for file in "$model"/*; do
    case "$file" in *.gguf) ;; *) ln -sf "$(realpath "$file")" "$dir/ptq/" ;; esac
  done
  python tests/make_ptq1_0.py "$gguf" "$dir/ptq/model-PTQ1_0.gguf"
  echo "ternary: the conversion of the PTQ1_0 file to ternary"
  node tests/profile-convert.mjs "$dir/ptq" ternary | head -12 | cut -c1-200
  python tests/perplexity_prepare.py "$dir/ptq" "$dir/from-ptq" ternary 2>&1 | tail -1 | cut -c1-200
  cmp "$dir/from-ptq.bin" "$dir/ternary.bin"
  echo "ternary: the PTQ1_0 file of the same weights converts to the same ternary checkpoint, byte for byte ($(stat -c %s "$dir/ptq/model-PTQ1_0.gguf") bytes of GGUF)"
  rm "$dir/from-ptq.bin" "$dir/ptq/model-PTQ1_0.gguf"
fi
if has perplexity; then
  # the ternary dtype in NumPy is the float32 file's values: the same logits to the bit, a few positions of each
  python - "$dir" <<'PYTHON'
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "tests")
from tree import python_folder
sys.path.insert(0, python_folder())
from llama2_numpy import Llama
out, rows = sys.argv[1], {}
for dtype in ("float32", "ternary"):
    llama = Llama(np.memmap(f"{out}/{dtype}.bin", dtype=np.uint8, mode="r"), Path(f"{out}/{dtype}.tokenizer.bin").read_bytes(),
                  kernels=None, **json.loads(Path(f"{out}/{dtype}.json").read_text()))
    token, rows[dtype] = llama.bos, []
    for pos in range(6):
        rows[dtype].append(np.array(llama.forward(token, pos)))
        token = int(rows[dtype][-1].argmax())
    del llama
same = all(np.array_equal(a, b) for a, b in zip(rows["float32"], rows["ternary"]))
print(f"ternary: the ternary dtype in NumPy against the float32 file, 6 positions: {'the same logits to the bit' if same else 'DIFFERENT — FAILED'}")
sys.exit(0 if same else 1)
PYTHON
  float32=$(python tests/perplexity_native.py "$dir/float32" 1500 "$text")
  echo "ternary: perplexity of the float32 of the file's values (and so of the ternary dtype), NumPy: $float32"
fi
rm -f "$dir/float32.bin"
if has perplexity; then
  int8=$(python tests/perplexity_native.py "$dir/int8" 1500 "$text")
  echo "ternary: perplexity of the int8 weights, NumPy: $int8"
  node tests/perplexity.mjs "$dir/int8" 1500 "$text" --file "--numpy=$(node -p "JSON.parse(process.argv[1]).perplexity" "$int8")"
  node tests/perplexity.mjs "$dir/ternary" 1500 "$text" --file "--numpy=$(node -p "JSON.parse(process.argv[1]).perplexity" "$float32")"
fi
if has compare; then
  node tests/ternary-compare.mjs "$dir/ternary" "$dir/int8" "$text" 256
fi
if has threads; then
  echo "ternary: the software threads, the ternary dtype and then int8"
  node tests/threads-check.mjs "$dir/ternary" "$dir/int8" --rounds 3 --positions 32
fi
