#!/usr/bin/env bash
# T230's review (a probe): the code 3 in the PQ2_0 files of the three Ternary Bonsai models of the list
set -euo pipefail
dir="${RUNNER_TEMP:-.tmp}/codes"
for id in hf-ternary-bonsai-1.7b hf-ternary-bonsai-4b hf-ternary-bonsai-8b; do
  model=$(python tests/hf_fetch.py "$id" "$dir/$id" | tail -1)
  python tests/t230_probe_codes.py "$model"/*.gguf
  rm -rf "$dir/$id"
done
