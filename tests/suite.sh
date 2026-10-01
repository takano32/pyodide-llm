#!/usr/bin/env bash
# tests/suite.sh (T193): the tests of deploy.yml and tests.yml, in one place so that a branch's tests.yml runs the very
# steps the deploy runs (the deploy runs on main only).
#
#   bash tests/suite.sh light   what keeps a broken page off the site: the engine's pytest, the page's modules in Node,
#                               the smoke test on the latest Pyodide (the deploy runs this one, and tests.yml by default)
#   bash tests/suite.sh full    and the heavier checks: forward.js against NumPy (shared, --plain, --wide), the default
#                               choice of the GPU or the CPU (gpu-default-check) and the software threads (tests.yml
#                               every night, and with full=true)
#
# The owner, 2026-09-27: "たまーにフルテストする感じでコミットたくさんあるときは軽くしたい". Needs what `make models
# kernels` makes and the Python packages of the workflows; the build is the workflow's own step.
set -euo pipefail
suite=${1:-}
case "$suite" in
  light | full) ;;
  *) echo "usage: bash tests/suite.sh light|full (not '$suite')" >&2; exit 2 ;;
esac
echo "suite: $suite, starting"
# each part's name and seconds, for the log (tests/ci.mjs reads the lines that start with "suite:" and "--- ")
part() {
  local name=$1 began=$SECONDS
  shift
  echo "--- $name"
  "$@"
  echo "--- $name: $((SECONDS - began)) s"
}

part "unit tests of the engine" python -m pytest tests -q

# Node alone: the benchmark's tables (T45), the table CI writes (T82), the list, the kept models, the service worker...
page_modules() {
  node tests/bench.mjs
  node tests/page-memory.mjs
  node tests/wake.mjs
  node tests/summary-check.mjs
  node tests/models-check.mjs
  node tests/ladder-check.mjs
  node tests/kept-check.mjs
  node tests/coi-js-check.mjs
  node tests/gpu-choice-check.mjs
  # T225's review: /benchmark/'s layer check against devices that round the cache's float16 as WGSL lets them (about 10 s)
  node tests/layer-check.mjs
  node tests/worker-sink-check.mjs
  # T129: where fetching and loading meet in worker.js, on made-up fetches and a fast clock (a few seconds)
  node tests/worker-check.mjs
  # T130's review, on forward.js's createForward() with kernels that do nothing: the KV cache grows in place without losing a
  # byte, and footprint() holds what is allocated (a few seconds); T223's: the search for the software threads on noisy times
  node tests/memory-check.mjs
  node tests/thread-search-check.mjs
}
part "unit tests of the page's modules" page_modules

# the page resolves the latest Pyodide release when it is opened, so test against that one
part "the latest Pyodide" npm install --no-save pyodide@latest
part "smoke test" node tests/smoke.mjs
# T229: the kernels of Qwen3.5's linear attention against the same arithmetic in JavaScript (under a second)
part "the delta rule's kernels" node tests/delta-check.mjs
# T237: the kernels of the rotated basis against the same arithmetic in JavaScript, to the bit (under a second)
part "the rotated basis's kernels" node tests/rotate-check.mjs
# T217 (the review of T201): attention's softmax where its largest score decides something (two positions far above
# the rest): a largest that leaves positions out, which forward-check's line cannot see (under a second)
part "attention's largest score" node tests/attention-check.mjs

if [ "$suite" = full ]; then
  # T93: the forward pass of public/forward.js against NumPy's, on the site's models (the line is for 128 positions)
  part "forward.js against NumPy" node tests/forward-check.mjs --rounds 1 --positions 128
  part "forward.js against NumPy, not shared" node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --plain
  # T130: a shared memory refused for a model whose float32 keys and values would not fit: float16 on a plain memory
  part "forward.js against NumPy, float16 on a plain memory" node tests/forward-check.mjs tiny-lm llm-jp-3-150m --rounds 1 --positions 128 --plain --half-keys
  # T101: the 64-bit memory and its kernels
  part "forward.js against NumPy, 64-bit" node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --wide
  # T229: a made-up Qwen3.5 (hybrid attention; no real one is small enough for the build): float32 to NumPy's numbers,
  # int8 within the line of a made-up model, one with a state as large as a real model's for the memory after the
  # checkpoint against footprint(); on a shared memory, a plain one and a 64-bit one
  made_up_qwen35() {
    mkdir -p .tmp
    python tests/make_qwen35.py .tmp/made-up-qwen35-float32 float32
    python tests/make_qwen35.py .tmp/made-up-qwen35-int8 int8
    python tests/make_qwen35.py .tmp/made-up-qwen35-state int8 state
    for memory in "" --plain --wide; do
      node tests/forward-check.mjs .tmp/made-up-qwen35-float32 .tmp/made-up-qwen35-int8 .tmp/made-up-qwen35-state --rounds 1 --positions 128 $memory
    done
  }
  part "forward.js against NumPy, a made-up Qwen3.5" made_up_qwen35
  # T237: the same made-up Qwen3.5 folded into a rotated basis (Ternary Bonsai 2 27B's): forward.js turns every
  # matrix's input and the embedding's rows back, to NumPy's numbers
  made_up_rotated() {
    python tests/make_qwen35.py .tmp/made-up-rotated-float32 float32 small rotated
    python tests/make_qwen35.py .tmp/made-up-rotated-int8 int8 small rotated
    python tests/make_qwen35.py .tmp/made-up-rotated-state int8 state rotated
    for memory in "" --plain --wide; do
      node tests/forward-check.mjs .tmp/made-up-rotated-float32 .tmp/made-up-rotated-int8 .tmp/made-up-rotated-state --rounds 1 --positions 128 $memory
    done
  }
  part "forward.js against NumPy, a made-up Qwen3.5 in a rotated basis" made_up_rotated
  # T148: the default choice of the GPU or the CPU for a prompt's blocks, with a made-up GPU's worker
  part "the GPU or the CPU by default" node tests/gpu-default-check.mjs
  part "the software threads" node tests/threads-check.mjs
  # T229: the value heads of a linear-attention layer's delta rule shared out among the threads, to the bit
  part "the software threads, a made-up Qwen3.5" node tests/threads-check.mjs .tmp/made-up-qwen35-float32 .tmp/made-up-qwen35-int8 --rounds 1
  part "the software threads, a made-up Qwen3.5 in a rotated basis" node tests/threads-check.mjs .tmp/made-up-rotated-float32 .tmp/made-up-rotated-int8 --rounds 1
  # T206: the pre-tokenizers against the real ones at every code point (about 90 s, too long for the deploy)
  part "the pre-tokenizers at every code point" env EVERY_CODE_POINT=1 python -m pytest tests/test_bytebpe.py -q -k every_character
fi
echo "suite: $suite passed in $SECONDS s"
