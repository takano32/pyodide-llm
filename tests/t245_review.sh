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
#   engine          the engine (NumPy, float32) on the 4B's GGUF against transformers on the original and on its Q8_0
#   bits6 <id> [n]  a real Qwen3.5 in six bits on the page's forward pass, against its int8
#   ja-rows <id> <articles> [labels]   the kernel rows (7-bit, 8-bit activations) on Japanese Wikipedia, an article each
#   loops-entry <id> <penalty|none> [prompts]   the page's thinking form to the end of the context (tests/t236_loops.mjs)
#   formats         tests/format_check.py on the Qwen3.5 entries, with two more prompts
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
page_tools() {
  npm ci --silent
  make kernels > /dev/null
}
prepare() {  # <source> <out name> <dtype>
  python tests/perplexity_prepare.py "$1" "$dir/$2" "$3" 2>&1 | tail -1 | cut -c1-300
}
article() {  # <language> <out> [title]: Wikipedia's API timed out from a runner once, so more than once
  local attempt
  for attempt in 1 2 3 4 5 6; do
    if node tests/wikipedia.mjs "$@" 2> "$dir/wikipedia.err" && [ -s "$2" ]; then
      echo "T245 article ${3:-the three of T85} ($1): $(wc -c < "$2") bytes, sha256 $(sha256sum "$2" | cut -c1-12)"
      return 0
    fi
    echo "T245 article ${3:-}: attempt $attempt failed ($(head -c 200 "$dir/wikipedia.err" | tr '\n' ' ')); waiting"
    sleep 30
  done
  return 1
}
llamacpp_file() {  # llama.cpp's conversion/qwen.py at the commit the converter's reading was written from
  curl -sSL -f --retry 5 -o "$dir/qwen.py" \
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/dcd387a412ca54e172a8d60eb71ef6753850c8ca/conversion/qwen.py"
  echo "T245 llama.cpp's qwen.py: $(wc -l < "$dir/qwen.py") lines, sha256 $(sha256sum "$dir/qwen.py" | cut -c1-12)"
}
big() {  # whether a converted checkpoint wants a 64-bit memory (tests/page_qwen35.sh's size)
  if [ "$(stat -c %s "$1")" -gt 3400000000 ]; then echo true; else echo false; fi
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
  lab-model)  # lab-model <0.8B|2B> [jobs]: the same on a model whose value heads are not tiled (as many to a key head)
    reference_tools
    transformers_of_t229
    swap
    article en "$dir/en.txt"
    python tests/q8_lab.py "$dir/orig" --text "$dir/en.txt" --model "${1:?a size}" --jobs "${2:-v0,v1,only-,all-but-}" --save "$dir/lab"
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
  bits6)  # bits6 <id> [tokens]
    id=${1:?an entry id}
    tokens=${2:-1500}
    page_tools
    model=$(python tests/hf_fetch.py "$id" "$dir/hf" | tail -1)
    article en "$dir/en.txt"
    prepare "$model" i8 int8
    prepare "$model" i6 int6
    wide8=; if [ "$(big "$dir/i8.bin")" = true ]; then wide8=--wide; fi
    wide6=; if [ "$(big "$dir/i6.bin")" = true ]; then wide6=--wide; fi
    echo "T245 bits: int8 $(stat -c %s "$dir/i8.bin") bytes, int6 $(stat -c %s "$dir/i6.bin") bytes"
    echo "T245 bits: int8 rows (7-bit, 8-bit)"
    node tests/perplexity.mjs "$dir/i8" "$tokens" "$dir/en.txt" --file $wide8 --rows=0,2
    echo "T245 bits: int6 rows (7-bit with relaxed SIMD, 8-bit without)"
    node tests/perplexity.mjs "$dir/i6" "$tokens" "$dir/en.txt" --file $wide6 --rows=0,2
    echo "T245 bits: the software threads on the six bits"
    node tests/threads-check.mjs "$dir/i6" --rounds 1 --positions 32 $wide6
    ;;
  ja-rows)  # ja-rows <id> <articles, e.g. 1,2,3> [labels, e.g. 7-bit,8-bit]
    id=${1:?an entry id}
    list=${2:-1,2,3}
    labels=${3:-7-bit,8-bit}
    page_tools
    model=$(python tests/hf_fetch.py "$id" "$dir/hf" | tail -1)
    article ja "$dir/ja1.txt" 富士山
    article ja "$dir/ja2.txt" 夏目漱石
    article ja "$dir/ja3.txt" 新幹線
    prepare "$model" g8 int8
    wide=$(big "$dir/g8.bin")
    texts=
    for n in ${list//,/ }; do texts="$texts${texts:+, }\"$dir/ja$n.txt\""; done
    texts="[$texts]"
    for label in ${labels//,/ }; do
      echo "T245 ja: $id, $label, articles $list"
      node tests/t236_rows.mjs "{\"count\": 1022, \"paths\": [{\"name\": \"$id\", \"out\": \"$dir/g8\"}], \"texts\": $texts, \"labels\": [\"$label\"], \"wide\": $wide}"
    done
    ;;
  loops-entry)  # loops-entry <id> <penalty|none> [prompt numbers, e.g. 0,2,4]
    id=${1:?an entry id}
    penalty=${2:-none}
    picked=${3:-0,1,2,3,4,5,6,7,8,9,10,11}
    page_tools
    model=$(python tests/hf_fetch.py "$id" "$dir/hf" | tail -1)
    prepare "$model" g8 int8
    python tests/write_options.py "$id" "$dir/small" > "$dir/page.json"
    wide=$(big "$dir/g8.bin")
    spec=$(node -e '
      const fs = require("node:fs");
      const [out, id, penalty, page, picked, wide] = process.argv.slice(1);
      const all = JSON.parse(fs.readFileSync("tests/t236_prompts.json", "utf8"));
      console.log(JSON.stringify({ out, id, page, prompts: picked.split(",").map((i) => all[Number(i)]), seed: 1000, wide: wide === "true",
        penalty: penalty === "none" ? null : Number(penalty) }));' "$dir/g8" "$id" "$penalty" "$dir/page.json" "$picked" "$wide")
    node tests/t236_loops.mjs "$spec"
    ;;
  layerwise)  # layerwise <size> <entry id> [check]: transformers a layer at a time against the page's forward pass on the GGUF as int8
    size=${1:?a size}
    id=${2:?an entry id}
    reference_tools
    transformers_of_t229
    page_tools
    article en "$dir/en.txt"
    python tests/layerwise_qwen35.py "$dir/orig" --model "$size" --out "$dir/lw" --text "$dir/en.txt" ${3:+--check}
    df -h /mnt | sed 's/^/runner: /'
    model=$(python tests/hf_fetch.py "$id" "$dir/hf" | tail -1)
    prepare "$model" g8 int8
    wide=$(big "$dir/g8.bin")
    for label in 7-bit 8-bit; do
      node tests/q8_page_logits.mjs "{\"out\": \"$dir/g8\", \"ids\": $(cat "$dir/lw-sentence-ids.json"), \"labels\": [\"$label\"], \"save\": \"$dir/page\", \"wide\": $wide}"
    done
    python tests/q8_page_compare.py "$dir/lw" "$dir/page"
    wideflag=; if [ "$wide" = true ]; then wideflag=--wide; fi
    echo "T245 layerwise: the page's perplexity rows (8-bit, then 7-bit) on the same 1500 tokens"
    node tests/perplexity.mjs "$dir/g8" 1500 "$dir/en.txt" --file $wideflag --rows=2
    node tests/perplexity.mjs "$dir/g8" 1500 "$dir/en.txt" --file $wideflag --rows=0
    ;;
  llamacpp)  # llama.cpp's own converter on made-up Qwen3.5 models of two and three value heads to a key head
    reference_tools
    transformers_of_t229
    pip install --quiet tqdm pyyaml requests sentencepiece protobuf
    sha=dcd387a412ca54e172a8d60eb71ef6753850c8ca
    curl -sSL -f --retry 5 -o "$dir/llama.cpp.tar.gz" "https://github.com/ggml-org/llama.cpp/archive/$sha.tar.gz"
    tar -xzf "$dir/llama.cpp.tar.gz" -C "$dir"
    python tests/llamacpp_tiny.py "$dir/llama.cpp-$sha" "$dir/tiny" --outtypes "${1:-f32,q8_0}"
    ;;
  chat-nll)  # chat-nll <owner/repository@revision>: tests/chat_nll.py (the first token in chat form, T236's review) on another size
    reference_tools
    transformers_of_t229
    python tests/chat_nll.py "${1:?owner/repository@revision}" ${2:-} ${3:-}
    ;;
  formats)  # the Qwen3.5 entries' formats against transformers' own, for every prompt, strictly
    pip install --quiet "transformers==5.16.1" "tokenizers==0.23.1" "jinja2==3.1.6" "sentencepiece==0.2.2" protobuf
    python tests/format_check.py --prompt "<think>" --prompt "<|im_start|>assistant" "$dir/formats" \
      hf-qwen3.5-0.8b hf-qwen3.5-0.8b-thinking hf-qwen3.5-2b hf-qwen3.5-2b-thinking hf-qwen3.5-4b hf-qwen3.5-4b-thinking \
      hf-qwen3.5-9b hf-qwen3.5-9b-thinking
    ;;
  *)
    echo "no stage $stage"
    exit 2
    ;;
esac
free -m | sed 's/^/runner: memory after: /'
