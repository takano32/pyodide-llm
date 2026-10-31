#!/usr/bin/env bash
# tests/reference_gemma4.sh (T.329.3): what tests/reference_gemma4.py needs, and the run itself, for CI (tests.yml's
# extra=; the development machine has no PyTorch):
#
#   node tests/ci.ts run tests.yml only_extra=true minutes=150 extra="bash tests/reference_gemma4.sh" --ref <branch> --grep "gemma4:"
#   node tests/ci.ts run tests.yml only_extra=true extra="bash tests/reference_gemma4.sh --only=made-up" --ref <branch> --grep "gemma4:"
#
# With only_extra=true it runs alone (no models of the site, no kernels, no suite): it installs what it needs.
# transformers at the commit the engine's formulas were read from (engine/layout.py, the comment on Gemma 4), and
# numpy, tokenizers and pytest (the unit tests' helpers). The E2B's 10.2 GB of safetensors and its float32 checkpoint of
# 18.6 GB go to /mnt where the runner has it, else to the runner's temporary directory; transformers holds 9.3 GB as
# bfloat16 and the engine maps the checkpoint, so 16 GB of swap there too (what a peak past the memory is paged to,
# not a place to compute in).
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy tokenizers pytest safetensors jinja2 "transformers @ https://github.com/huggingface/transformers/archive/2126f65b0ab267e113e78a002447fc7094de83d5.tar.gz"
directory="${RUNNER_TEMP:-.tmp}/gemma4"
if [ -d /mnt ] && sudo -n true 2>/dev/null; then
  sudo mkdir -p /mnt/gemma4 && sudo chown "$USER" /mnt/gemma4
  directory=/mnt/gemma4
  case " $* " in
    *" --only=made-up "*) ;;
    *) sudo fallocate -l 16G /mnt/gemma4.swap && sudo chmod 600 /mnt/gemma4.swap && sudo mkswap -q /mnt/gemma4.swap && sudo swapon /mnt/gemma4.swap ;;
  esac
fi
mkdir -p "$directory"
df -h "$(dirname "$directory")" | sed 's/^/gemma4: /'
free -m | sed 's/^/gemma4: /'
echo "gemma4: $(lscpu | sed -n 's/^Model name: *//p' | head -1) ($(uname -m)), $(nproc) logical cores"
python tests/reference_gemma4.py "$directory" "$@"
