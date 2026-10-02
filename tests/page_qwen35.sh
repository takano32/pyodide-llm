#!/usr/bin/env bash
# tests/page_qwen35.sh (T229): Qwen/Qwen3.5-0.8B on the page's forward pass (public/forward.js on the int8 kernels),
# for CI (tests.yml's extra=, after `make kernels` and with the suite's pyodide: the development machine runs no
# model of gigabytes):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/page_qwen35.sh" --ref <branch> --grep "qwen35|perplexity|thread"
#
# The model converted the way the page converts it (tests/perplexity_prepare.py), then
#   - its perplexity on 1500 tokens of English Wikipedia (T85's text): the float32 original and the int8 weights in
#     native NumPy (the int8 weights widened to float32 are 3 GB, past Pyodide), and the int8 weights on the kernels
#     with 7-bit and 8-bit activations (tests/perplexity.mjs --file);
#   - the software threads: the logits of a greedy run and of a prompt in blocks the same to the bit with 1, 2, 4 and
#     8 threads, and after a thread that stops in the middle of its work (tests/threads-check.mjs).
#
# T236: with an id of the list (bash tests/page_qwen35.sh hf-qwen3.5-0.8b), that entry as the page takes it instead of
# the original's safetensors: the GGUF's weights with the original's vocabulary and config.json (tests/hf_fetch.py).
# The float32 row is then the GGUF's values, as Q8_0 rounded them. T247's review: and with the entry's own options
# (perplexity_prepare.py --entry), its BOS above all: <|im_start|>, where the converter's is <|endoftext|>. The rows of T229,
# T236 and T247 before the review ran with <|endoftext|> in front of every window, a way the list's entries never run:
# a Qwen3.5 reads a text 18% (2B) to 45% (4B) worse with it, and its perplexity then moves by up to 3% under any rounding
# of its weights (docs/quantization.md).
#
# T247: a second word says which of the four parts to run, "float32 int8 kernels threads" when left out. NumPy's two
# rows widen the weights to float32, which a runner's 16 GB holds for the 2B (7.5 GB) and not for the 4B (16.8 GB) or
# the 9B (35.8 GB): those run "kernels threads" (bash tests/page_qwen35.sh hf-qwen3.5-4b "kernels threads"), and the
# 2B's rows take two runs of the job's 90 minutes. On a runner the files go to /mnt, which has the room. A third word
# is the number of tokens (1500 when left out).
set -euo pipefail
parts=" ${2:-float32 int8 kernels threads} "
tokens="${3:-1500}"
has() { case "$parts" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
dir="${RUNNER_TEMP:-.tmp}/qwen35"
if [ -n "${RUNNER_TEMP:-}" ] && [ -d /mnt ]; then
  dir=/mnt/qwen35
  sudo mkdir -p "$dir" && sudo chown "$USER" "$dir"
fi
if [ -n "${1:-}" ]; then
  model=$(python tests/hf_fetch.py "$1" "$dir" | tail -1)
  what="the float32 of the list's $1"
else
  python tests/reference_qwen35.py "$dir" --only=fetch
  model="$dir"
  what="the float32 original"
fi
node tests/wikipedia.mjs en "$dir/en.txt"
if has float32; then
  python tests/perplexity_prepare.py "$model" "$dir/float32" float32 ${1:+--entry "$1"} 2>&1 | tail -1 | cut -c1-200
  echo "qwen35: perplexity of $what, NumPy: $(python tests/perplexity_native.py "$dir/float32" "$tokens" "$dir/en.txt")"
  rm "$dir/float32.bin"
fi
python tests/perplexity_prepare.py "$model" "$dir/int8" int8 ${1:+--entry "$1"} 2>&1 | tail -1 | cut -c1-200
# a 64-bit memory where the checkpoint and what forward.js puts after it pass 4 GiB (the 4B's 4.7 GB, the 9B's 10.1 GB)
wide=
if [ "$(stat -c %s "$dir/int8.bin")" -gt 3400000000 ]; then wide=--wide; fi
numpy=
if has int8; then
  numpy=$(python tests/perplexity_native.py "$dir/int8" "$tokens" "$dir/en.txt")
  echo "qwen35: perplexity of the int8 weights, NumPy: $numpy"
  # (--max-change: the review of T229. 7 bits cost +1.86% on x86-64 and +2.29% on arm64, 8 bits -0.36% and -0.45%; a fault of the
  # int8 path is 10% or more)
  numpy="--numpy=$(node -p "JSON.parse(process.argv[1]).perplexity" "$numpy") --max-change=5"
fi
if has kernels; then
  if [ "$(stat -c %s "$dir/int8.bin")" -gt 6000000000 ]; then
    # a process for each row (7-bit and 8-bit activations): two memories of the 9B at once are more than a runner has
    node tests/perplexity.mjs "$dir/int8" "$tokens" "$dir/en.txt" --file $numpy $wide --rows=0
    node tests/perplexity.mjs "$dir/int8" "$tokens" "$dir/en.txt" --file $numpy $wide --rows=2
  else
    node tests/perplexity.mjs "$dir/int8" "$tokens" "$dir/en.txt" --file $numpy $wide
  fi
fi
if has threads; then
  echo "qwen35: the software threads"
  node tests/threads-check.mjs "$dir/int8" --rounds 1 --positions 32 $wide
fi
