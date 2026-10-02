#!/usr/bin/env bash
# t272_ci.sh (the review of T260, T272; a throwaway branch's): the probe of tests/t272_probe.py on a real LFM2 of the list, for CI
# (tests.yml's extra=; with only_extra=true the runner has no models and no kernels, and needs none):
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/t272_ci.sh en 1500 all roles" --ref t260-review-probe --grep "t272:"
#   ... extra="T272_MODEL=hf-lfm2.5-1.2b-jp bash tests/t272_ci.sh en 1500 all positions"
#   ... extra="T272_ORIGINAL=1 bash tests/t272_ci.sh en 1500 hybrid"      (the 350M's float32 original too)
#
# $1 the language of the text (en | ja), $2 the tokens, the rest the experiments of the probe. T272_MODEL: the entry (the 350M).
set -euo pipefail
language="$1"; tokens="$2"; shift 2
model="${T272_MODEL:-hf-lfm2.5-350m}"
pip install --quiet numpy
mkdir -p .tmp/t272
echo "t272: $(node -p 'require("os").cpus()[0].model') ($(nproc) logical cores), $model"
free -m | sed 's/^/t272: /'
bash tests/gpu-real.sh "$model" 2>&1 | tail -2 | cut -c1-200 | sed 's/^/t272: /'
if [ -n "${T272_ORIGINAL:-}" ]; then
  # the float32 original of the 350M, as ?hf= would convert its safetensors
  source=$(python3 tests/hf_fetch.py "hf:LiquidAI/LFM2.5-350M@9e6c6ccf47cd318696e137d381a7ded8fe4df09f" .tmp/downloads-original | tail -1)
  python3 tests/perplexity_prepare.py "$source" .tmp/real/original float32 2>&1 | tail -1 | cut -c1-200 | sed 's/^/t272: /'
  export T272_ORIGINAL=.tmp/real/original
fi
node tests/wikipedia.mjs "$language" ".tmp/t272/$language.txt"
python tests/t272_probe.py ".tmp/real/$model" ".tmp/t272/$language.txt" "$tokens" "$@"
