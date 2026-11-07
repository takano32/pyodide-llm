#!/usr/bin/env bash
# tests/reference_gemma2.sh (T612): what tests/reference_gemma2.py needs, and the run itself, for CI (tests.yml's extra=;
# the development machine has no PyTorch):
#
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_gemma2.sh" --ref <branch> --grep "gemma2:"
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_gemma2.sh --only=real" --ref <branch> --grep "gemma2:"
#
# With only_extra=true it runs alone (no models of the site, no kernels, no suite): it installs what it needs.
# transformers at the commit the engine's Gemma 3 was read from (engine/layers.py), which has Gemma 2 too, and numpy,
# tokenizers and pytest (the unit tests' helpers). --only=real: the downloads and the float32 checkpoint (5.2 GB and
# 10.5 GB) go to /mnt where the runner has it, else to the runner's temporary directory.
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy tokenizers pytest safetensors jinja2 "transformers @ https://github.com/huggingface/transformers/archive/28034545c35acac05793695bd5c878f66e6b544e.tar.gz"
directory="${RUNNER_TEMP:-.tmp}/gemma2"
if [ -d /mnt ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p /mnt/gemma2 && sudo chown "$USER" /mnt/gemma2
  directory=/mnt/gemma2
fi
df -h "$(dirname "$directory")" | sed 's/^/gemma2: /'
free -m | sed 's/^/gemma2: /'
python tests/reference_gemma2.py "$directory" "$@"
