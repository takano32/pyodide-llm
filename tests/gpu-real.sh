#!/usr/bin/env bash
# gpu-real.sh (T183): models of src/models.js for tests/gpu-check.mjs, fetched from Hugging Face at the revisions the
# list pins (tests/hf_fetch.py) and converted to int8 as the page converts them (tests/perplexity_prepare.py; T232: to
# ternary where the list says the model's weights are that, as the page asks for them: src/models.js's weights), into
# .tmp/real/<id>.bin, .tokenizer.bin and .json. gpu-prompt.yml's input real= runs it; so can a person, where WebGPU's
# tests run (AGENTS.md: not on the development machine). The seconds of each step go to the log.
#
#   bash tests/gpu-real.sh <model id> ...        then: node tests/gpu-check.mjs .tmp/real/<model id>.json ...
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tmp/real .tmp/downloads
for id in "$@"; do
  began=$SECONDS
  source=$(python3 tests/hf_fetch.py "$id" .tmp/downloads)
  fetched=$SECONDS
  dtype=$(node --input-type=module -e 'import { MODELS } from "./src/models.js"; console.log(MODELS.find((m) => m.id === process.argv[1])?.weights ?? "int8")' "$id")
  python3 tests/perplexity_prepare.py "$source" ".tmp/real/$id" "$dtype"
  echo "seconds of $id's preparation: fetching $((fetched - began)), converting to $dtype $((SECONDS - fetched))"
done
