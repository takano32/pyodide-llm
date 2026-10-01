#!/usr/bin/env bash
# tests/page_qwen35.sh (T229): Qwen/Qwen3.5-0.8B on the page's forward pass (public/forward.js on the int8 kernels),
# for CI (tests.yml's extra=, after `make kernels` and with the suite's pyodide: the development machine runs no
# model of gigabytes):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/page_qwen35.sh" --ref <branch> --grep "qwen35|perplexity|thread"
#
# The model converted the way the page converts it (tests/perplexity_prepare.py), then
#   - its perplexity on 1500 tokens of English Wikipedia (T85's text): the float32 original and the int8 weights in
#     native NumPy (the int8 weights widened to float32 are 3 GB, past Pyodide), and the int8 weights on the kernels
#     with 7-bit and 8-bit activations (tests/perplexity.mjs --file);
#   - the software threads: the logits of a greedy run and of a prompt in blocks the same to the bit with 1, 2, 4 and
#     8 threads, and after a thread that stops in the middle of its work (tests/threads-check.mjs).
#
# T236: with an id of the list (bash tests/page_qwen35.sh hf-qwen3.5-0.8b), that entry as the page takes it instead of
# the original's safetensors: the GGUF's weights with the original's vocabulary and config.json (tests/hf_fetch.py).
# The float32 row is then the GGUF's values, as Q8_0 rounded them.
set -euo pipefail
dir="${RUNNER_TEMP:-.tmp}/qwen35"
if [ -n "${1:-}" ]; then
  model=$(python tests/hf_fetch.py "$1" "$dir" | tail -1)
  what="the float32 of the list's $1"
else
  python tests/reference_qwen35.py "$dir" --only=fetch
  model="$dir"
  what="the float32 original"
fi
node tests/wikipedia.mjs en "$dir/en.txt"
for dtype in float32 int8; do
  python tests/perplexity_prepare.py "$model" "$dir/$dtype" "$dtype" 2>&1 | tail -1 | cut -c1-200
done
echo "qwen35: perplexity of $what, NumPy: $(python tests/perplexity_native.py "$dir/float32" 1500 "$dir/en.txt")"
rm "$dir/float32.bin"
numpy=$(python tests/perplexity_native.py "$dir/int8" 1500 "$dir/en.txt")
echo "qwen35: perplexity of the int8 weights, NumPy: $numpy"
node tests/perplexity.mjs "$dir/int8" 1500 "$dir/en.txt" --file "--numpy=$(node -p "JSON.parse(process.argv[1]).perplexity" "$numpy")"
echo "qwen35: the software threads"
node tests/threads-check.mjs "$dir/int8" --rounds 1 --positions 32
