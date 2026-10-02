#!/usr/bin/env bash
# T230's review (a probe, not for main): what the ternary kernels' 8-bit activations cost against the int8 path's 7 bits
# (the page before T230) and 8 bits (matmul_q8, relaxed SIMD off), on the same texts, for CI (tests.yml's extra=):
#
#   bash tests/t230_review_perplexity.sh <id of the list> <articles | whole> <en | ja | both> [--wide]
#
# articles: each of T85's three articles of the language on its own (about 1500 tokens each, 512-token windows);
# whole: the three together, as T230 measured the 1.7B. One row each: the int8 weights' 7-bit and 8-bit activations and the
# ternary weights' 8-bit (matmul_t2r). Wikipedia is fetched when it runs; nothing of it is stored.
set -euo pipefail
id="$1"
mode="${2:-articles}"
languages="${3:-both}"
wide="${4:-}"
dir="${RUNNER_TEMP:-.tmp}/review"
mkdir -p "$dir"
model=$(python tests/hf_fetch.py "$id" "$dir" | tail -1)
echo "review: $id, $(node -p 'require("os").cpus()[0].model') ($(uname -m)), $mode, $languages"
python tests/perplexity_prepare.py "$model" "$dir/int8" int8 2>&1 | tail -1 | cut -c1-200
python tests/perplexity_prepare.py "$model" "$dir/ternary" ternary 2>&1 | tail -1 | cut -c1-200
echo "review: int8 $(stat -c %s "$dir/int8.bin") bytes, ternary $(stat -c %s "$dir/ternary.bin") bytes"
run() {
  local label="$1" text="$2"
  echo "review: $label"
  node tests/perplexity.mjs "$dir/ternary" 1500 "$text" --file $wide --rows=0 | grep '^| ' | grep -v 'computation' | sed 's/^/review:   /'
  node tests/perplexity.mjs "$dir/int8" 1500 "$text" --file $wide --rows=0,2 | grep '^| ' | grep -v 'computation' | sed 's/^/review:   /'
}
for language in en ja; do
  if [[ "$languages" != both && "$languages" != "$language" ]]; then continue; fi
  if [[ "$mode" == whole ]]; then
    node tests/wikipedia.mjs "$language" "$dir/$language.txt"
    run "$language, the three articles" "$dir/$language.txt"
  else
    case "$language" in
      en) titles=("Mount Fuji" "Natsume Sōseki" "Shinkansen") ;;
      ja) titles=("富士山" "夏目漱石" "新幹線") ;;
    esac
    for title in "${titles[@]}"; do
      node tests/wikipedia.mjs "$language" "$dir/one.txt" "$title"
      run "$language, $title" "$dir/one.txt"
    done
  fi
done
