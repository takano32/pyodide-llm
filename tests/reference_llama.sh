#!/usr/bin/env bash
# tests/reference_llama.sh (T253, T254): what tests/reference_llama.py needs, and the run itself, for CI (tests.yml's
# extra=; the development machine has no PyTorch):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_llama.sh hf-granite-4.2-3b" --ref <branch> --grep "reference:"
#   (no id: the made-up Granites alone; the options of reference_llama.py follow the ids: --weak, --layers=4)
#   bash tests/reference_llama.sh --run tests/chat_nll.py <arguments>
#   (the review of T253: another tool that needs torch and this transformers, with what they need, as python gets it)
#
# transformers at the commit the Granite's attention was read from (llama2_convert.py, above query_scale()); numpy,
# tokenizers and pytest (the unit tests' helpers) are the workflow's own. The downloads and the float32 checkpoint go
# to /mnt where the runner has it (Granite 4.2 3B: 7.3 GB of safetensors and 14.6 GB of float32), else to the runner's
# temporary directory.
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet safetensors jinja2 "transformers @ https://github.com/huggingface/transformers/archive/7cd73d9df0c14b151c684b708a9f27d8d0349dfe.tar.gz"
directory="${RUNNER_TEMP:-.tmp}/reference"
if [ -d /mnt ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p /mnt/reference && sudo chown "$USER" /mnt/reference
  directory=/mnt/reference
fi
df -h "$(dirname "$directory")" | sed 's/^/reference: /'
free -m | sed 's/^/reference: /'
if [ "${1:-}" = "--run" ]; then
  shift
  python "$@"
elif [ "$#" -eq 0 ]; then
  python tests/reference_llama.py "$directory" --only=made-up
else
  python tests/reference_llama.py "$directory" "$@"
fi
