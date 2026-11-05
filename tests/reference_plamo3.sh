#!/usr/bin/env bash
# tests/reference_plamo3.sh (T649.2): what tests/reference_plamo3.py needs, and the run itself, for CI (tests.yml's
# extra=; the development machine has no PyTorch):
#
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_plamo3.sh" --ref <branch> --grep "plamo3:"
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_plamo3.sh --only=real --model=2B --no-faults" --ref <branch> --grep "plamo3:"
#
# With only_extra=true it runs alone (no models of the site, no kernels, no suite): it installs what it needs.
# transformers of the version the model's own code was written for (its config.json's transformers_version: the code is
# the model's, modeling_plamo.py, and transformers has no class of it), numba (its tokenization_plamo.py), and numpy,
# tokenizers and pytest (the unit tests' helpers). The downloads and the float32 checkpoint go to /mnt where the runner
# has it (the 0.8B: 1.8 GB of safetensors and 3.6 GB of float32), else to the runner's temporary directory.
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy numba tokenizers pytest safetensors jinja2 "transformers==4.57.1"
directory="${RUNNER_TEMP:-.tmp}/plamo3"
if [ -d /mnt ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p /mnt/plamo3 && sudo chown "$USER" /mnt/plamo3
  directory=/mnt/plamo3
fi
df -h "$(dirname "$directory")" | sed 's/^/plamo3: /'
free -m | sed 's/^/plamo3: /'
python tests/reference_plamo3.py "$directory" "$@"
