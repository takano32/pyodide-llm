#!/usr/bin/env bash
# tests/page_gemma3.sh (T261): a Gemma 3 on the page's forward pass (src/runtime/forward.js on the int8 kernels), for
# CI (tests.yml's extra=, after `make kernels` and with the suite's pyodide) and, for the 270M, the development machine:
#
#   node tests/ci.mjs run tests.yml extra="bash tests/page_gemma3.sh" --ref <branch> --grep "gemma3:|perplexity|thread"
#   node tests/ci.mjs run tests.yml extra="bash tests/page_gemma3.sh hf:unsloth/gemma-3-1b-it@5b11413a10db4e486ef16a20101fd028f8f2499c" runner=ubuntu-24.04-arm --ref <branch> ...
#
# The model as ?hf= opens it (tests/hf_fetch.py: the repository's safetensors, config.json and tokenizer; an id of the
# list once the list has one), converted the way the page converts it (tests/perplexity_prepare.py), then
#   - its perplexity on 1500 tokens of English Wikipedia and of Japanese Wikipedia (T85's articles; three times the
#     window of 512 positions its local layers attend): the float32 and the int8 weights in native NumPy, and the int8
#     weights on the kernels with 7-bit and 8-bit activations (tests/perplexity.mjs --file);
#   - the software threads: the logits of a greedy run and of a prompt in blocks the same to the bit with 1, 2, 4 and
#     8 threads, after a thread that stops in the middle of its work, and the tokens a second of each count
#     (tests/threads-check.mjs).
# A second word says which parts to run, "float32 int8 kernels threads" when left out; a third the languages, "en ja";
# a fourth the number of tokens (1500).
set -euo pipefail
id="${1:-hf:unsloth/gemma-3-270m-it@23cf460f6bb16954176b3ddcc8d4f250501458a9}"
parts=" ${2:-float32 int8 kernels threads} "
languages="${3:-en ja}"
tokens="${4:-1500}"
has() { case "$parts" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
dir="${RUNNER_TEMP:-.tmp}/gemma3-page"
if [ -n "${RUNNER_TEMP:-}" ] && [ -d /mnt ]; then
  dir=/mnt/gemma3-page
  sudo mkdir -p "$dir" && sudo chown "$USER" "$dir"
fi
mkdir -p "$dir"
echo "gemma3: $id on $(node -p 'require("os").cpus()[0].model') ($(nproc) logical cores)"
model=$(python tests/hf_fetch.py "$id" "$dir" | tail -1)
for language in $languages; do node tests/wikipedia.mjs "$language" "$dir/$language.txt"; done
if has float32; then
  python tests/perplexity_prepare.py "$model" "$dir/float32" float32 2>&1 | tail -1 | cut -c1-200
  for language in $languages; do
    echo "gemma3: perplexity of the float32 of $id, NumPy, $language: $(python tests/perplexity_native.py "$dir/float32" "$tokens" "$dir/$language.txt")"
  done
  rm "$dir/float32.bin"
fi
python tests/perplexity_prepare.py "$model" "$dir/int8" int8 2>&1 | tail -1 | cut -c1-200
ls -l "$dir/int8.bin" | awk '{ print "gemma3: the int8 checkpoint is " $5 " bytes" }'
for language in $languages; do
  numpy=
  if has int8; then
    numpy=$(python tests/perplexity_native.py "$dir/int8" "$tokens" "$dir/$language.txt")
    echo "gemma3: perplexity of the int8 weights of $id, NumPy, $language: $numpy"
    # (a fault of the int8 path is 10% or more; 7 bits cost other models 0.2 to 3.4%)
    numpy="--numpy=$(node -p "JSON.parse(process.argv[1]).perplexity" "$numpy") --max-change=6"
  fi
  if has kernels; then
    echo "gemma3: the kernels, $language"
    node tests/perplexity.mjs "$dir/int8" "$tokens" "$dir/$language.txt" --file $numpy
  fi
done
if has threads; then
  echo "gemma3: the software threads of $id"
  node tests/threads-check.mjs "$dir/int8" --rounds 3 --positions 64
fi
