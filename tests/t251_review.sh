#!/usr/bin/env bash
# t251_review.sh (the review of T251, T252 and T262, a probe for CI, not for main): tests.yml's extra= with only_extra=true.
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=120 extra="bash tests/t251_review.sh <stage> [args]" \
#     --grep "t251_review|start_check|chat_nll|chat_fluency|degenerate|answers" --minutes 130 --ref <branch>
#
# stages (every one prints lines that begin with the tool's own name, and "t251_review:" for the stage's):
#   starts <id> <tokens> <window> <bos> [also] [--stored]
#                   the same English text under each start, on the original (tests/start_check.py: transformers) and on the page's
#                   engine (tests/start_check.mjs: int8, 7-bit activations), so that the two perplexities are of one text and one set
#                   of targets; the page's is of the converted GGUF
#   chat <repository@revision> <bos> <end of a turn> [--stored] [--fidelity=N]
#                   tests/chat_nll.py: the first token in chat form on text no model wrote and on the model's own answers (transformers)
#   fluency <id> <end of a turn> [PICK]
#                   tests/chat_fluency.mjs: the same hand-written answers on the page's engine (a 7B has no transformers on a runner)
#   degenerate      tests/degenerate.mjs: how often the models of the English ladder's small end come apart as the page samples them,
#                   with GPT-2 124M and Pythia 70M as the ones that were already in the list
#   think <id> [tokens]
#                   tests/answers.mjs: a model that thinks, to the end of its thought, as the page samples it
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tmp
stage=${1:?a stage}
shift || true
sudo mkdir -p /mnt/t251 && sudo chown "$USER" /mnt/t251
dir=/mnt/t251
echo "t251_review: runner $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) logical cores, $(free -g | awk '/Mem:/{print $2}') GB of memory, stage $stage $*"
df -h / /mnt | sed 's/^/t251_review: /'
pip install --quiet numpy pytest tokenizers regex sentencepiece

reference_tools() {  # as tests/reference_llama.sh does (once in a run that has several stages)
  python -c "import torch, transformers" 2> /dev/null && return 0
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet safetensors jinja2 protobuf "transformers @ https://github.com/huggingface/transformers/archive/7cd73d9df0c14b151c684b708a9f27d8d0349dfe.tar.gz"
}
page_tools() {
  [ -f public/simdkernel.so ] && return 0
  npm ci --silent
  make kernels > /dev/null
}
english() {  # T85's three English articles, the page's own text of tests/start_check.mjs's English runs (Wikipedia timed out from a runner once)
  local attempt
  for attempt in 1 2 3 4 5 6; do
    if node tests/wikipedia.mjs en "$dir/en.txt" 2> "$dir/wikipedia.err" && [ -s "$dir/en.txt" ]; then
      echo "t251_review: English text $(wc -c < "$dir/en.txt") bytes, sha256 $(sha256sum "$dir/en.txt" | cut -c1-12)"
      return 0
    fi
    echo "t251_review: Wikipedia, attempt $attempt failed ($(head -c 200 "$dir/wikipedia.err" | tr '\n' ' ')); waiting"
    sleep 30
  done
  return 1
}

case "$stage" in
  starts)
    id=${1:?a model id}; tokens=${2:?tokens}; window=${3:?window}; bos=${4:?the BOS}; also=${5:-}; stored=${6:-}
    [ "$also" = "--stored" ] && { stored=--stored; also=; }
    reference_tools
    page_tools
    english
    echo "t251_review: $id on the original, held by transformers"
    python tests/start_check.py "$id" --text "$dir/en.txt" --tokens "$tokens" --window "$window" --directory "$dir/sc" ${also:+--also $also} $stored
    rm -rf "$dir/sc"
    echo "t251_review: $id on the page's engine"
    STARTS="none,$bos${also:+,$also}" TEXT="$dir/en.txt" WINDOW="$window" WRITER=start_check.mjs TOKENS="$tokens" bash tests/write.sh "$id"
    ;;
  chat)
    target=${1:?owner/repository@revision}; bos=${2:?the BOS}; end=${3:?the end of a turn}; shift 3
    reference_tools
    python tests/chat_nll.py "$target" "$bos" "$end" "$@"
    ;;
  fluency)
    id=${1:?a model id}; end=${2:?the end of a turn}; pick=${3:-}
    page_tools
    PICK="$pick" END="$end" WRITER=chat_fluency.mjs bash tests/write.sh "$id"
    ;;
  degenerate)
    page_tools
    prompts='["Once upon a time", "The history of the city", "My favorite food is"]'
    seeds=$(seq -s ' ' 1 64)
    for id in "$@"; do
      SEEDS="$seeds" PROMPTS="$prompts" WRITER=degenerate.mjs TOKENS=200 bash tests/write.sh "$id"
    done
    ;;
  bits)
    # what the 7-bit activations of the relaxed kernels cost a model whose final norm is near the outlier gate (3.0 to 3.8 against the gate's 4.0):
    # the same English text and targets on the page's engine, once as the page runs it (relaxed: 7 bits) and once with matmul_q8 (8 bits: what
    # Safari runs, ?without=relaxed). Only the activations differ: the weights are the same int8
    id=${1:?a model id}; tokens=${2:?tokens}; window=${3:?window}; bos=${4:?the BOS}
    page_tools
    english
    room=/mnt/t251/bits
    mkdir -p "$room"
    model=$(python3 tests/hf_fetch.py "$id" "$room/downloads" | tail -1)
    python3 tests/perplexity_prepare.py "$model" "$room/$id" int8 | cut -c1-160
    python3 tests/write_options.py "$id" "$room/small" > "$room/$id.page.json"
    rm -rf "$room/downloads"
    python3 - "$room/$id.page.json" <<'PY'
import json, sys
page = json.load(open(sys.argv[1]))
json.dump(page, open(sys.argv[1].replace(".page.json", ".7bit.json"), "w"))
page["options"]["disable"] = ["relaxed"]
json.dump(page, open(sys.argv[1].replace(".page.json", ".8bit.json"), "w"))
PY
    for variant in 7bit 8bit; do
      echo "t251_review: $id, activations of the $variant variant"
      node tests/start_check.mjs "$room/$id" "$room/$id.$variant.json" "$tokens" --starts "$bos" --window "$window" --text "$dir/en.txt"
    done
    rm -rf "$room/$id".*
    ;;
  loss)
    # where a small model's loss against the original comes from: tests/perplexity.mjs's three rows on the same English text, the int8 weights
    # with 8-bit activations (matmul_q8), with 7-bit ones (matmul_q8r, as the page runs them) and widened to float32 with float32 activations (NumPy)
    page_tools
    english
    room=/mnt/t251/loss
    mkdir -p "$room"
    for id in "$@"; do
      model=$(python3 tests/hf_fetch.py "$id" "$room/downloads" | tail -1)
      python3 tests/perplexity_prepare.py "$model" "$room/$id" int8 | cut -c1-160
      rm -rf "$room/downloads"
      echo "t251_review: $id, three rows"
      node tests/perplexity.mjs "$room/$id" 1024 "$dir/en.txt"
      rm -rf "$room/$id".*
    done
    ;;
  think)
    id=${1:?a model id}; tokens=${2:-1800}
    page_tools
    PROMPTS='["What is 17 times 24? Think first.", "Give me three tips for sleeping better."]' SEEDS="1" WRITER=answers.mjs TOKENS="$tokens" bash tests/write.sh "$id"
    ;;
  *)
    echo "t251_review: no stage $stage" >&2
    exit 2
    ;;
esac
echo "t251_review: stage $stage done"
