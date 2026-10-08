#!/usr/bin/env bash
# tests/page_lfm2.sh (T260): an LFM2 of the list on the page's forward pass (public/forward.js on the int8 kernels),
# for CI (tests.yml's extra=, after `make kernels` and with the suite's pyodide: the development machine runs no model
# of gigabytes):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/page_lfm2.sh hf-lfm2.5-350m" --ref <branch> --grep "lfm2:|perplexity|thread"
#   node tests/ci.mjs run tests.yml extra="bash tests/page_lfm2.sh hf-lfm2.5-1.2b-jp" runner=ubuntu-24.04-arm --ref <branch> ...
#
# The entry as the page takes it (tests/hf_fetch.py: the GGUF's weights with the original's vocabulary and
# config.json), converted the way the page converts it (tests/perplexity_prepare.py), then
#   - its perplexity on 1500 tokens of English Wikipedia and of Japanese Wikipedia (T85's articles): the float32 (the
#     GGUF's values, as Q8_0 rounded them) and the int8 weights in native NumPy, and the int8 weights on the kernels
#     with 7-bit and 8-bit activations (tests/perplexity.mjs --file);
#   - the software threads: the logits of a greedy run and of a prompt in blocks the same to the bit with 1, 2, 4 and
#     8 threads, after a thread that stops in the middle of its work, and the tokens a second of each count
#     (tests/threads-check.mjs).
# A second word says which parts to run, "float32 int8 kernels threads" when left out; a third the languages, "en ja";
# a fourth the number of tokens (1500). --original as the second word's first: the original's safetensors instead of
# the GGUF (what Q8_0's rounding costs is the difference of the two float32 rows).
set -euo pipefail
id="$1"
parts=" ${2:-float32 int8 kernels threads} "
languages="${3:-en ja}"
tokens="${4:-1500}"
has() { case "$parts" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
dir="${RUNNER_TEMP:-.tmp}/lfm2-page"
if [ -n "${RUNNER_TEMP:-}" ] && [ -d /mnt ]; then
  dir=/mnt/lfm2-page
  sudo mkdir -p "$dir" && sudo chown "$USER" "$dir"
fi
mkdir -p "$dir"
echo "lfm2: $id on $(node -p 'require("os").cpus()[0].model') ($(nproc) logical cores)"
if has --original; then
  # the repository and revision the entry's vocabulary comes from: its safetensors, as ?hf= opens it
  source=$(node -e "import('./src/models.js').then(({ MODELS }) => { const { repo, revision } = MODELS.find((m) => m.id === '$id').hf.vocabulary; console.log('hf:' + repo + '@' + revision); })")
  model=$(python tests/hf_fetch.py "$source" "$dir" | tail -1)
  what="the float32 original of $id"
else
  model=$(python tests/hf_fetch.py "$id" "$dir" | tail -1)
  what="the float32 of the list's $id (its GGUF's values)"
fi
for language in $languages; do node tests/wikipedia.mjs "$language" "$dir/$language.txt"; done
if has float32; then
  python tests/perplexity_prepare.py "$model" "$dir/float32" float32 2>&1 | tail -1 | cut -c1-200
  for language in $languages; do
    echo "lfm2: perplexity of $what, NumPy, $language: $(python tests/perplexity_native.py "$dir/float32" "$tokens" "$dir/$language.txt")"
  done
  rm "$dir/float32.bin"
fi
python tests/perplexity_prepare.py "$model" "$dir/int8" int8 2>&1 | tail -1 | cut -c1-200
for language in $languages; do
  numpy=
  if has int8; then
    numpy=$(python tests/perplexity_native.py "$dir/int8" "$tokens" "$dir/$language.txt")
    echo "lfm2: perplexity of the int8 weights of $id, NumPy, $language: $numpy"
    # (a fault of the int8 path is 10% or more; 7 bits cost other models 0.2 to 3.4%)
    numpy="--numpy=$(node -p "JSON.parse(process.argv[1]).perplexity" "$numpy") --max-change=6"
  fi
  if has kernels; then
    echo "lfm2: the kernels, $language"
    node tests/perplexity.mjs "$dir/int8" "$tokens" "$dir/$language.txt" --file $numpy
  fi
done
if has threads; then
  echo "lfm2: the software threads of $id"
  node tests/threads-check.mjs "$dir/int8" --rounds 3 --positions 64
fi
