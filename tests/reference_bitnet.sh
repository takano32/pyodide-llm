#!/usr/bin/env bash
# tests/reference_bitnet.sh (T283): what tests/reference_bitnet.py needs, and the run itself, for CI (tests.yml's extra=;
# the development machine has no PyTorch):
#
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_bitnet.sh" --ref <branch> --grep "bitnet:"
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_bitnet.sh --only=made-up" --ref <branch> --grep "bitnet:"
#
# With only_extra=true it runs alone (no models of the site, no kernels, no suite): it installs what it needs.
# transformers of the version whose BitNet the engine was written from (5.16.1: models/bitnet, integrations/bitnet.py),
# accelerate (its BitNet quantizer asks for it), and numpy, tokenizers and pytest (the unit tests' helpers). The
# download (1.2 GB) and the converted files (1.9 GB of float32) go to /mnt where the runner has it, else to the
# runner's temporary directory. transformers holds the 2B unpacked in float32 (about 10 GB) and the engine's NumPy
# forward pass its matrices widened (about 10 GB): one after the other, never both.
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy tokenizers pytest safetensors jinja2 accelerate "transformers==5.16.1"
directory="${RUNNER_TEMP:-.tmp}/bitnet"
if [ -d /mnt ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p /mnt/bitnet && sudo chown "$USER" /mnt/bitnet
  directory=/mnt/bitnet
fi
df -h "$(dirname "$directory")" | sed 's/^/bitnet: /'
free -m | sed 's/^/bitnet: /'
python tests/reference_bitnet.py "$directory" "$@"
