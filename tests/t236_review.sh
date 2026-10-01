#!/usr/bin/env bash
# t236_review.sh (the review of T236 and T246, a probe for CI, not for main): tests.yml's extra= with only_extra=true.
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=120 extra="bash tests/t236_review.sh <stage> [args]" \
#     --grep "T236|T246" --minutes 130 --ref <branch>
#
# stages:
#   tiny            transformers' side of tests/t236_bos.py on a made-up model (the API calls, in a few minutes)
#   bos             tests/t236_bos.py: the first token in chat form and in plain text (transformers float32, and the page's engine)
#   gates           tests/t236_gates.py: Q8_0 on the two gate matrices, in KL, against the original (NumPy float32)
#   rows            tests/t236_rows.mjs and tests/t236_numpy_rows.py: the kernel rows and the NumPy row of the GGUF's path and the
#                   original's, on 3 English and 3 Japanese articles
#   loops <id> <penalty|none>   tests/t236_loops.mjs: 12 prompts through the page's engine, to the end of the context
#   scan <4B|8B> <revision>     tests/t246_scan.py: blocks with two magnitudes in the -unpacked originals
#   yarn <size> <revision> [by-layer]   tests/t246_chat_reference.py: yarn against a plain RoPE on texts and chat turns
set -euo pipefail
cd "$(dirname "$0")/.."
stage=${1:?a stage}
shift || true
sudo mkdir -p /mnt/t236 && sudo chown "$USER" /mnt/t236
dir=/mnt/t236
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
page_tools() {
  npm ci --silent
  make kernels > /dev/null
}
entries() {
  node -e "import('./src/models.js').then(({ MODELS }) => console.log(JSON.stringify(MODELS.filter((m) => m.id.startsWith('hf-qwen3.5-0.8b')))))" > "$dir/entries.json"
}
article() {  # <language> <out> <title>: Wikipedia's API timed out from a runner once, so more than once
  local attempt
  for attempt in 1 2 3 4 5 6; do
    if node tests/wikipedia.mjs "$@" 2> "$dir/wikipedia.err" && [ -s "$2" ]; then
      echo "T236 article $3 ($1): $(wc -c < "$2") bytes, sha256 $(sha256sum "$2" | cut -c1-12)"
      return 0
    fi
    echo "T236 article $3: attempt $attempt failed ($(head -c 200 "$dir/wikipedia.err" | tr '\n' ' ')); waiting"
    sleep 30
  done
  return 1
}
articles() {  # T85's three, English and Japanese: the first of each article, plain text
  article en "$dir/en1.txt" "Mount Fuji"
  article en "$dir/en2.txt" "Natsume Sōseki"
  article en "$dir/en3.txt" "Shinkansen"
  article ja "$dir/ja1.txt" 富士山
  article ja "$dir/ja2.txt" 夏目漱石
  article ja "$dir/ja3.txt" 新幹線
}
original() {  # the original's files (tests/reference_qwen35.py fetches them at its revision)
  python tests/reference_qwen35.py "$dir/orig" --only=fetch
}
gguf() {  # the list's GGUF with the original's vocabulary and config.json, as the page reads it
  python tests/hf_fetch.py hf-qwen3.5-0.8b "$dir/hf" | tail -1
}
prepare() {  # <source> <out name> <dtype>
  python tests/perplexity_prepare.py "$1" "$dir/$2" "$3" 2>&1 | tail -1 | cut -c1-300
}

case "$stage" in
  tiny)
    reference_tools
    transformers_of_t229
    python tests/t236_bos.py --tiny
    ;;
  bos)
    reference_tools
    transformers_of_t229
    original
    source=$(gguf)
    prepare "$source" g32 float32
    entries
    articles
    python tests/t236_bos.py --original "$dir/orig" --gguf "$dir/g32" --entries "$dir/entries.json" --tokens 40 --plain-tokens 512 \
      --texts "$dir/en1.txt" "$dir/en2.txt" "$dir/ja1.txt" "$dir/ja2.txt"
    ;;
  gates)
    original
    source=$(gguf)
    prepare "$dir/orig" o32 float32
    prepare "$source" g32 float32
    articles
    python tests/t236_gates.py "$dir/o32" "$dir/g32" 512 "$dir/en1.txt" "$dir/en2.txt" "$dir/ja1.txt" "$dir/ja2.txt"
    ;;
  rows)
    page_tools
    original
    source=$(gguf)
    prepare "$dir/orig" o8 int8
    prepare "$source" g8 int8
    articles
    node tests/t236_rows.mjs "{\"count\": 1022, \"paths\": [{\"name\": \"gguf\", \"out\": \"$dir/g8\"}, {\"name\": \"original\", \"out\": \"$dir/o8\"}], \"texts\": [\"$dir/en1.txt\", \"$dir/en2.txt\", \"$dir/en3.txt\", \"$dir/ja1.txt\", \"$dir/ja2.txt\", \"$dir/ja3.txt\"]}"
    python tests/t236_numpy_rows.py "$dir/g8" gguf 1022 "$dir"/en1.txt "$dir"/en2.txt "$dir"/en3.txt "$dir"/ja1.txt "$dir"/ja2.txt "$dir"/ja3.txt
    python tests/t236_numpy_rows.py "$dir/o8" original 1022 "$dir"/en1.txt "$dir"/en2.txt "$dir"/en3.txt "$dir"/ja1.txt "$dir"/ja2.txt "$dir"/ja3.txt
    ;;
  tiny-node)
    page_tools
    python tests/make_qwen35.py "$dir/tiny8" int8 | cut -c1-120
    printf 'w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20' > "$dir/tiny.txt"
    node tests/t236_rows.mjs "{\"count\": 16, \"paths\": [{\"name\": \"tiny\", \"out\": \"$dir/tiny8\"}], \"texts\": [\"$dir/tiny.txt\"]}"
    node tests/t236_loops.mjs "{\"out\": \"$dir/tiny8\", \"id\": \"hf-qwen3.5-0.8b-thinking\", \"tiny\": true, \"prompts\": [\"w1 w2\", \"w3\"], \"seed\": 1}"
    ;;
  loops)
    id=${1:?an entry id}
    penalty=${2:-none}
    page_tools
    source=$(gguf)
    prepare "$source" g8 int8
    spec=$(node -e '
      const fs = require("node:fs");
      const [out, id, penalty] = process.argv.slice(1);
      console.log(JSON.stringify({ out, id, prompts: JSON.parse(fs.readFileSync("tests/t236_prompts.json", "utf8")), seed: 1000,
        penalty: penalty === "none" ? null : Number(penalty) }));' "$dir/g8" "$id" "$penalty")
    node tests/t236_loops.mjs "$spec"
    ;;
  scan)
    size=${1:?4B or 8B}
    revision=${2:?a revision}
    base="https://huggingface.co/prism-ml/Ternary-Bonsai-$size-unpacked/resolve/$revision"
    mkdir -p "$dir/scan-$size"
    curl -sSL -f --retry 5 "$base/model.safetensors.index.json" -o "$dir/scan-$size/index.json"
    for shard in $(python3 -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" "$dir/scan-$size/index.json"); do
      curl -sSL -f --retry 5 --retry-delay 10 -C - -o "$dir/scan-$size/$shard" "$base/$shard"
      echo "T246SCAN fetched $shard: $(stat -c %s "$dir/scan-$size/$shard") bytes"
    done
    python tests/t246_scan.py "$size" "$dir/scan-$size"
    ;;
  yarn)
    size=${1:?a size}
    revision=${2:?a revision}
    mode=${3:-}
    reference_tools
    pip install --quiet transformers==4.57.6
    article ja "$dir/ja-tokyo.txt" 東京都
    article en "$dir/en-tokyo.txt" "Tokyo"
    flags=()
    if [ "$mode" = by-layer ]; then flags=(--by-layer); fi
    python tests/t246_chat_reference.py "$dir" "$size" "$revision" "${flags[@]}" --chat tests/t246_chat.jsonl "$dir/ja-tokyo.txt:1024" "$dir/en-tokyo.txt:1024"
    ;;
  *)
    echo "no stage $stage"
    exit 2
    ;;
esac
free -m | sed 's/^/runner: memory after: /'
