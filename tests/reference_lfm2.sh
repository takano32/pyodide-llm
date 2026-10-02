#!/usr/bin/env bash
# tests/reference_lfm2.sh (T260): what tests/reference_lfm2.py needs, and the run itself, for CI (tests.yml's extra=;
# the development machine has no PyTorch):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_lfm2.sh" --ref <branch> --grep "lfm2:"
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/reference_lfm2.sh --only=real --model=350M,1.2B-JP" --ref <branch> --grep "lfm2:"
#
# With only_extra=true it runs alone (no models of the site, no kernels, no suite): it installs what it needs.
# transformers at the commit the engine's formulas were read from (llama2_numpy.py, above convolution_form()), and
# numpy, tokenizers and pytest (the unit tests' helpers). The downloads and the float32 checkpoint go to /mnt where the
# runner has it (the 1.2B: 2.3 GB of safetensors and 4.7 GB of float32), else to the runner's temporary directory.
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy tokenizers pytest safetensors jinja2 "transformers @ https://github.com/huggingface/transformers/archive/7cd73d9df0c14b151c684b708a9f27d8d0349dfe.tar.gz"
directory="${RUNNER_TEMP:-.tmp}/lfm2"
if [ -d /mnt ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p /mnt/lfm2 && sudo chown "$USER" /mnt/lfm2
  directory=/mnt/lfm2
fi
df -h "$(dirname "$directory")" | sed 's/^/lfm2: /'
free -m | sed 's/^/lfm2: /'
python tests/reference_lfm2.py "$directory" "$@"
