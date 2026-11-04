#!/usr/bin/env bash
# tests/page_binary.sh (T.634.2): a 1-bit Bonsai of the list (its Q1_0 GGUF) converted as the page converts it, to the
# ternary dtype (what the list kept it as before) and to binary (what it keeps now), for CI (tests.yml's extra=, after
# `make kernels`: the development machine runs no model of gigabytes):
#
#   node tests/ci.ts run tests.yml only_extra=true minutes=120 extra="pip install -q numpy jinja2 && npm ci && make kernels && bash tests/page_binary.sh hf-bonsai-1.7b" --ref <branch>
#   bash tests/page_binary.sh [<id of the list> = hf-bonsai-1.7b] [positions = 64]
#
# What it says: the seconds of the conversion in Pyodide of each (tests/page-27b.ts convert: by the conduct of a
# conversion answered from the folder, into a memory of forward.js, as the worker converts it), the two checkpoints'
# bytes, and the logits of the two on the page's forward pass the same to the bit, a token at a time and after a block,
# with relaxed SIMD and without (tests/binary-same-check.ts): the list's answers are the ones they were, greedy or
# sampled with the same seed.
set -euo pipefail
id="${1:-hf-bonsai-1.7b}"
positions="${2:-64}"
dir="${RUNNER_TEMP:-.tmp}/binary"
model=$(python tests/hf_fetch.py "$id" "$dir" | tail -1)
echo "binary: $id, $(node -p 'require("os").cpus()[0].model') ($(uname -m)); $(df -h "$dir" | tail -1)"
for dtype in ternary binary; do
  echo "binary: the conversion to $dtype in Pyodide"
  node tests/page-27b.ts "$dir/none" convert "$model" --entry "$id" --dtype "$dtype" 2>&1 | grep "^pyodide:" | cut -c1-300
done
for dtype in ternary binary; do
  python tests/perplexity_prepare.py "$model" "$dir/$dtype" "$dtype" --entry "$id" 2>&1 | tail -1 | cut -c1-200
  echo "binary: $dtype is $(stat -c %s "$dir/$dtype.bin") bytes"
done
echo "binary: $(df -h "$dir" | tail -1)"
node tests/binary-same-check.ts "$dir/ternary" "$dir/binary" --positions "$positions"
