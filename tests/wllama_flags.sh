#!/usr/bin/env bash
# T519 review: wllama under V8 flags (a control: is its wasm stuck in the baseline tier, or slowed by bounds checks?)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tmp/keep
for flags in "" "--js-flags=--no-liftoff" "--js-flags=--no-wasm-bounds-checks"; do
  echo "#### flags: [$flags]"
  WLLAMA_CHROMIUM_ARGS="$flags" python tests/wllama_compare.py run hf-smollm2-360m-instruct --rounds 3 --tokens 64 --results ".tmp/keep/flags.jsonl" 2>&1 | grep -v "^$" | grep "wllama-in-browser\|FAILED\|^- \|Error" | cut -c1-240
done
