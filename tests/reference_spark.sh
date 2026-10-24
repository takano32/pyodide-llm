#!/usr/bin/env bash
# tests/reference_spark.sh (T.342): what tests/reference_spark.py needs, and the run itself, for CI (tests.yml's extra=;
# the development machine has no PyTorch):
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/reference_spark.sh" --ref <branch> --grep "spark:"
#   node tests/ci.mjs run tests.yml only_extra=true extra="bash tests/reference_spark.sh --only=real --model=1.7B" --ref <branch> --grep "spark:"
#
# With only_extra=true it runs alone (no models of the site, no kernels, no suite): it installs what it needs.
# transformers of the version the model's own code was written for (its config.json's transformers_version: the code is
# the model's, modeling_spark.py, and transformers has no class of it), and numpy, tokenizers and pytest (the unit tests'
# helpers). The downloads and the float32 checkpoint go to /mnt where the runner has it (the 1.7B: 3.4 GB of
# safetensors and 7 GB of float32), else to the runner's temporary directory.
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy tokenizers pytest safetensors jinja2 "transformers==4.57.1"
directory="${RUNNER_TEMP:-.tmp}/spark"
if [ -d /mnt ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p /mnt/spark && sudo chown "$USER" /mnt/spark
  directory=/mnt/spark
fi
df -h "$(dirname "$directory")" | sed 's/^/spark: /'
free -m | sed 's/^/spark: /'
python tests/reference_spark.py "$directory" "$@"
