#!/usr/bin/env bash
# tests/write.sh (T249): what Hugging Face models of the list write for their own prompts on the page's engine, greedy,
# for a person to read. For CI (tests.yml's extra=, after `make kernels`: the development machine runs no model of
# gigabytes):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/write.sh hf-japanese-gpt2-xsmall hf-eurollm-1.7b-instruct" --grep "written by|write.sh" --ref <branch>
#
# Each model is fetched at the revision the list pins (tests/hf_fetch.py), converted to int8 as the page converts it
# (tests/perplexity_prepare.py) and given the options and the format the page gives it (tests/write_options.py); then
# tests/write.mjs writes TOKENS tokens (32). What was fetched and converted goes before the next model (an 8B's GGUF
# and its int8 are 18 GB, so on /mnt where a runner has one). A model that fails says so and the next one runs; the
# exit status is the number that failed.
set -uo pipefail
cd "$(dirname "$0")/.."
room=".tmp/write"
if [ -d /mnt ] && [ -n "${GITHUB_ACTIONS:-}" ]; then
  sudo mkdir -p /mnt/write && sudo chown "$USER" /mnt/write
  room=/mnt/write
fi
mkdir -p "$room"
failed=0
for id in "$@"; do
  began=$SECONDS
  (
    set -e
    model=$(python3 tests/hf_fetch.py "$id" "$room/downloads" | tail -1)
    fetched=$SECONDS
    python3 tests/perplexity_prepare.py "$model" "$room/$id" int8 | cut -c1-160
    python3 tests/write_options.py "$id" "$room/small" > "$room/$id.page.json"
    converted=$SECONDS
    rm -rf "$room/downloads"
    node tests/write.mjs "$room/$id" "$room/$id.page.json" "${TOKENS:-32}"
    echo "write.sh $id: fetching $((fetched - began)) s, converting $((converted - fetched)) s, writing $((SECONDS - converted)) s"
  )
  # (not `( … ) || …`: bash ignores set -e in everything a || tests)
  if [ $? -ne 0 ]; then
    failed=$((failed + 1))
    echo "write.sh $id: FAILED"
  fi
  rm -rf "$room/downloads" "$room/$id".*
done
echo "write.sh: $# models, $failed failed"
exit "$failed"
