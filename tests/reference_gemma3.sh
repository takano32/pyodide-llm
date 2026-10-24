#!/usr/bin/env bash
# tests/reference_gemma3.sh (T.261): what tests/reference_gemma3.py needs, and the run itself, for CI (tests.yml's extra=;
# the development machine has no PyTorch):
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/reference_gemma3.sh" --ref <branch> --grep "gemma3:"
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/reference_gemma3.sh --only=real --model=270M,1B" --ref <branch> --grep "gemma3:"
#
# With only_extra=true it runs alone (no models of the site, no kernels, no suite): it installs what it needs.
# transformers at the commit the engine's formulas were read from (engine/layers.py, the comment on Gemma 3), and numpy,
# tokenizers and pytest (the unit tests' helpers). The downloads and the float32 checkpoint go to /mnt where the runner
# has it (the 1B: 2.0 GB of safetensors and 4.0 GB of float32), else to the runner's temporary directory.
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy tokenizers pytest safetensors jinja2 "transformers @ https://github.com/huggingface/transformers/archive/28034545c35acac05793695bd5c878f66e6b544e.tar.gz"
directory="${RUNNER_TEMP:-.tmp}/gemma3"
if [ -d /mnt ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p /mnt/gemma3 && sudo chown "$USER" /mnt/gemma3
  directory=/mnt/gemma3
fi
df -h "$(dirname "$directory")" | sed 's/^/gemma3: /'
free -m | sed 's/^/gemma3: /'
python tests/reference_gemma3.py "$directory" "$@"
