#!/usr/bin/env bash
# t272_ci.sh (the review of T260, T272; a throwaway branch's): the probe of tests/t272_probe.py on the real LFM2.5 350M, for CI
# (tests.yml's extra=; with only_extra=true the runner has no models and no kernels, and needs none):
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/t272_ci.sh en 1500 all roles" --ref t260-review-probe --grep "t272:"
#
# $1 the language of the text (en | ja), $2 the tokens, the rest the experiments of the probe.
set -euo pipefail
language="$1"; tokens="$2"; shift 2
pip install --quiet numpy
mkdir -p .tmp/t272
echo "t272: $(node -p 'require("os").cpus()[0].model') ($(nproc) logical cores)"
free -m | sed 's/^/t272: /'
bash tests/gpu-real.sh hf-lfm2.5-350m 2>&1 | tail -2 | cut -c1-200 | sed 's/^/t272: /'
node tests/wikipedia.mjs "$language" ".tmp/t272/$language.txt"
python tests/t272_probe.py .tmp/real/hf-lfm2.5-350m ".tmp/t272/$language.txt" "$tokens" "$@"
