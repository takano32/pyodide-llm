#!/usr/bin/env bash
# t246_run.sh (T246, a probe for CI, not for main): tests.yml's extra= for a Ternary Bonsai.
#
#   bash tests/t246_run.sh engine <model id> [tokens of each Wikipedia text]     the page's forward pass (tests/t246_page.mjs)
#   bash tests/t246_run.sh reference <1.7B | 4B | 8B> <revision of the -unpacked repo> [by-layer]  transformers (tests/t246_reference.py)
set -euo pipefail
cd "$(dirname "$0")/.."
kind=$1
sudo mkdir -p /mnt/t246 && sudo chown "$USER" /mnt/t246
df -h / /mnt | sed 's/^/T246 disk: /'
free -m | sed 's/^/T246 memory: /'
# the texts: the two of T235 (fixed files) and Wikipedia's, fetched now (their sha256 is in each tool's log)
node tests/wikipedia.mjs ja /mnt/t246/ja-fuji.txt 富士山
node tests/wikipedia.mjs ja /mnt/t246/ja-soseki.txt 夏目漱石
node tests/wikipedia.mjs en /mnt/t246/en-3.txt
if [ "$kind" = engine ]; then
  id=$2 count=${3:-1024}
  began=$SECONDS
  source=$(python3 tests/hf_fetch.py "$id" /mnt/t246/downloads)
  fetched=$SECONDS
  python3 tests/perplexity_prepare.py "$source" "/mnt/t246/$id" int8
  echo "T246 $id: seconds of fetching $((fetched - began)), of converting $((SECONDS - fetched)) (native Python)"
  ls -l "/mnt/t246/$id.bin"
  texts=()
  if [ "$count" != 0 ]; then
    texts=(--ppl "/mnt/t246/ja-fuji.txt:$count" "/mnt/t246/ja-soseki.txt:$count" "/mnt/t246/en-3.txt:$count" tests/fixtures/t235/text.txt:256)
  fi
  node tests/t246_page.mjs "/mnt/t246/$id" "$id" --threads 1,2,4 --tokens 64 --rounds 3 "${texts[@]}"
  free -m | sed 's/^/T246 memory after: /'
else
  size=$2 revision=$3
  pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet transformers==4.57.6
  mode=()
  if [ "${4:-}" = by-layer ]; then mode=(--by-layer); fi
  python3 tests/t246_reference.py /mnt/t246 "$size" "$revision" "${mode[@]}" --greedy 24 /mnt/t246/ja-fuji.txt:1024 /mnt/t246/ja-soseki.txt:1024 \
    /mnt/t246/en-3.txt:1024 tests/fixtures/t235/text.txt:256
  free -m | sed 's/^/T246 memory after: /'
fi
