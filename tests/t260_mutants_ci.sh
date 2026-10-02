#!/usr/bin/env bash
# t260_mutants_ci.sh (the review of T260; a throwaway branch's): what tests/t260_mutants.py needs, and the run itself (tests.yml's
# extra= with only_extra=true):
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=120 extra="bash tests/t260_mutants_ci.sh js" --ref t260-review-probe --grep "mutants:"
set -euo pipefail
pip install --quiet numpy pytest tokenizers regex sentencepiece
npm ci --silent
make kernels > /dev/null
echo "mutants: $(node -p 'require("os").cpus()[0].model') ($(nproc) logical cores)"
python tests/t260_mutants.py "$@"
