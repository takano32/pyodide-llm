#!/usr/bin/env bash
# tests/reference_qwen35.sh (T229): what tests/reference_qwen35.py needs, and the run itself, for CI (tests.yml's
# extra=; the development machine has no PyTorch):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_qwen35.sh" --ref <branch> --grep "qwen35"
#
# transformers at the commit the engine's formulas were read from (llama2_numpy.py, above linear_form()); numpy,
# tokenizers and pytest (the unit tests' helpers) are the workflow's own.
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet safetensors "transformers @ https://github.com/huggingface/transformers/archive/7fb5bcd1d4b8a5c225a2c33429b2e9e023dd61ae.tar.gz"
python tests/reference_qwen35.py "${RUNNER_TEMP:-.tmp}/qwen35" "$@"
