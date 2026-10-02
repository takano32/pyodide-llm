#!/usr/bin/env bash
# t272_kernels_ci.sh (the review of T260, T272; a throwaway branch's): the probe's NumPy copy of the activations' rounding
# against the real kernels, on one text in one job. The 350M (as the page converts the list's entry), the text of
# Wikipedia, then
#   - tests/t272_probe.py's "all": every input 8-bit and 7-bit, in NumPy (the copy);
#   - tests/perplexity_native.py: the int8 weights in native NumPy, the activations not rounded (the copy's base);
#   - tests/perplexity.mjs --file: the same on the kernels, 7-bit (matmul_q8r) and 8-bit (matmul_q8).
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/t272_kernels_ci.sh en 1500" --ref t260-review-probe --grep "t272:"
set -euo pipefail
language="$1"; tokens="$2"
pip install --quiet numpy tokenizers regex sentencepiece pytest
npm ci --silent
make kernels > /dev/null
mkdir -p .tmp/t272
echo "t272: $(node -p 'require("os").cpus()[0].model') ($(nproc) logical cores)"
bash tests/gpu-real.sh hf-lfm2.5-350m 2>&1 | tail -2 | cut -c1-200 | sed 's/^/t272: /'
node tests/wikipedia.mjs "$language" ".tmp/t272/$language.txt"
python tests/t272_probe.py .tmp/real/hf-lfm2.5-350m ".tmp/t272/$language.txt" "$tokens" all
numpy=$(python tests/perplexity_native.py .tmp/real/hf-lfm2.5-350m "$tokens" ".tmp/t272/$language.txt")
echo "t272: native NumPy, the int8 weights, the activations not rounded: $numpy"
node tests/perplexity.mjs .tmp/real/hf-lfm2.5-350m "$tokens" ".tmp/t272/$language.txt" --file "--numpy=$(node -p 'JSON.parse(process.argv[1]).perplexity' "$numpy")" | sed 's/^/t272: /'
