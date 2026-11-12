#!/usr/bin/env bash
# tests/reference_phi3.sh (T258): what tests/reference_phi3.py needs, and the run itself, for CI (tests.yml's extra=;
# the development machine has no PyTorch):
#
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_phi3.sh" --ref <branch> --grep "phi3:"
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_phi3.sh --only=made-up" --ref <branch> --grep "phi3:"
#
# With only_extra=true it runs alone (no models of the site, no kernels, no suite): it installs what it needs.
# transformers of the version whose Phi-3 the engine was written from (5.16.1: models/phi3), and numpy, tokenizers and
# pytest (the unit tests' helpers). The download (7.6 GB a model) and the converted files go to /mnt where the runner
# has it, else to the runner's temporary directory. transformers holds the first layers in float32 (8 of 32: about
# 4.4 GB) and the engine's NumPy forward pass the same: one after the other, never both.
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy tokenizers pytest safetensors jinja2 "transformers==5.16.1"
directory="${RUNNER_TEMP:-.tmp}/phi3"
if [ -d /mnt ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p /mnt/phi3 && sudo chown "$USER" /mnt/phi3
  directory=/mnt/phi3
fi
df -h "$(dirname "$directory")" | sed 's/^/phi3: /'
free -m | sed 's/^/phi3: /'
python tests/reference_phi3.py "$directory" "$@"
