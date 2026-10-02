#!/usr/bin/env bash
# t245_review.sh (the review of T245 and T247, a probe for CI, not for main): tests.yml's extra= with only_extra=true.
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=120 extra="bash tests/t245_review.sh <stage> [args]" \
#     --grep "Q8LAB|T245" --minutes 130 --ref <branch>
#
# stages:
#   lab-tiny        tests/q8_lab.py --tiny: the lab's own checks on a made-up model (a few minutes)
#   lab [jobs]      tests/q8_lab.py on the real 4B: Q8_0 of the original's weights in transformers, by group, by BOS, shifted,
#                   and the eight tensors of the value heads read in llama.cpp's order one at a time
set -euo pipefail
cd "$(dirname "$0")/.."
stage=${1:?a stage}
shift || true
sudo mkdir -p /mnt/t245 && sudo chown "$USER" /mnt/t245
dir=/mnt/t245
echo "runner: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) logical cores, $(free -g | awk '/Mem:/{print $2}') GB of memory, stage $stage"
df -h / /mnt | sed 's/^/runner: /'
pip install --quiet numpy pytest tokenizers regex sentencepiece

reference_tools() {
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet safetensors huggingface_hub
}
transformers_of_t229() {
  pip install --quiet "transformers @ https://github.com/huggingface/transformers/archive/7fb5bcd1d4b8a5c225a2c33429b2e9e023dd61ae.tar.gz"
}
swap() {  # 16 GB more of swap, as tests/reference_qwen35.sh has it for a model that is more than the memory
  sudo fallocate -l 16G /mnt/t245.swap && sudo chmod 600 /mnt/t245.swap
  sudo mkswap -q /mnt/t245.swap && sudo swapon /mnt/t245.swap
  sudo prlimit --pid $$ --memlock=unlimited:unlimited
  free -m | sed 's/^/runner: /'
}
article() {  # <language> <out> <title>: Wikipedia's API timed out from a runner once, so more than once
  local attempt
  for attempt in 1 2 3 4 5 6; do
    if node tests/wikipedia.mjs "$@" 2> "$dir/wikipedia.err" && [ -s "$2" ]; then
      echo "T245 article $3 ($1): $(wc -c < "$2") bytes, sha256 $(sha256sum "$2" | cut -c1-12)"
      return 0
    fi
    echo "T245 article $3: attempt $attempt failed ($(head -c 200 "$dir/wikipedia.err" | tr '\n' ' ')); waiting"
    sleep 30
  done
  return 1
}
llamacpp_file() {  # llama.cpp's conversion/qwen.py at the commit the converter's reading was written from
  curl -sSL -f --retry 5 -o "$dir/qwen.py" \
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/dcd387a412ca54e172a8d60eb71ef6753850c8ca/conversion/qwen.py"
  echo "T245 llama.cpp's qwen.py: $(wc -l < "$dir/qwen.py") lines, sha256 $(sha256sum "$dir/qwen.py" | cut -c1-12)"
}

case "$stage" in
  lab-tiny)
    reference_tools
    transformers_of_t229
    llamacpp_file
    LLAMACPP_QWEN="$dir/qwen.py" python tests/q8_lab.py --tiny
    ;;
  lab)  # lab [jobs]
    reference_tools
    transformers_of_t229
    swap
    llamacpp_file
    article en "$dir/en.txt"  # T85's three articles, as tests/page_qwen35.sh has them
    LLAMACPP_QWEN="$dir/qwen.py" python tests/q8_lab.py "$dir/orig" --text "$dir/en.txt" --jobs "${1:-all}" --save "$dir/lab"
    ;;
  engine)  # the engine on the 4B's GGUF against transformers on the original, and on the original rounded to Q8_0
    reference_tools
    transformers_of_t229
    swap
    article en "$dir/en.txt"
    python tests/q8_lab.py "$dir/orig" --text "$dir/en.txt" --jobs v0,v1 --save "$dir/lab"
    df -h /mnt | sed 's/^/runner: /'
    python tests/q8_engine.py "$dir/orig" --text "$dir/en.txt" --out "$dir/engine" --source gguf
    python tests/q8_compare.py "$dir/lab" "$dir/engine"
    ;;
  *)
    echo "no stage $stage"
    exit 2
    ;;
esac
free -m | sed 's/^/runner: memory after: /'
