#!/usr/bin/env bash
# tests/reference_qwen35.sh (T229): what tests/reference_qwen35.py needs, and the run itself, for CI (tests.yml's
# extra=; the development machine has no PyTorch):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/reference_qwen35.sh" --ref <branch> --grep "qwen35"
#
# transformers at the commit the engine's formulas were read from (llama2_numpy.py, above linear_form()); numpy,
# tokenizers and pytest (the unit tests' helpers) are the workflow's own.
#
# T245: with --model=4B [--from=gguf], a model of 17 GB as float32 on a runner of 16 GB of memory: the files go to
# /mnt (the original's 9 GB, a float32 checkpoint of 17 GB or two, the GGUF's 4.5 GB), transformers' float32 gets 16 GB
# more of swap, and the engine may pin what it maps (reference_qwen35.py's pinned()).
set -euo pipefail
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet safetensors "transformers @ https://github.com/huggingface/transformers/archive/7fb5bcd1d4b8a5c225a2c33429b2e9e023dd61ae.tar.gz"
dir="${RUNNER_TEMP:-.tmp}/qwen35"
case " $* " in
  *" --model="*)
    if [ -n "${RUNNER_TEMP:-}" ] && [ -d /mnt ]; then
      dir=/mnt/qwen35
      sudo mkdir -p "$dir" && sudo chown "$USER" "$dir"
      sudo fallocate -l 16G /mnt/qwen35.swap && sudo chmod 600 /mnt/qwen35.swap
      sudo mkswap -q /mnt/qwen35.swap && sudo swapon /mnt/qwen35.swap
      sudo prlimit --pid $$ --memlock=unlimited:unlimited
      df -h / /mnt | sed 's/^/qwen35: /'
      free -m | sed 's/^/qwen35: /'
    fi
    mkdir -p "$dir"
    node tests/wikipedia.mjs en "$dir/en.txt"
    python tests/reference_qwen35.py "$dir" "--text=$dir/en.txt" "$@"
    ;;
  *)
    python tests/reference_qwen35.py "$dir" "$@"
    ;;
esac
