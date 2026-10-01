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
# T229 (the review): a Qwen3.5 is not put on a GPU where an adapter is there: forward.js's gpuUnfit says so first, and no
# adapter of CI's reaches that line (a few seconds)
hybrid_stays_on_the_cpu() {
  mkdir -p .tmp
  python tests/make_qwen35.py .tmp/made-up-qwen35-int8 int8
  node tests/gpu-hybrid-check.mjs .tmp/made-up-qwen35-int8
}
part "a Qwen3.5 where a GPU adapter is" hybrid_stays_on_the_cpu
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
  # (the review: and the float16 and six-bit files, which the page opens too: a Safari keeps a 4B in six bits)
  made_up_qwen35() {
    mkdir -p .tmp
    python tests/make_qwen35.py .tmp/made-up-qwen35-float32 float32
    python tests/make_qwen35.py .tmp/made-up-qwen35-float16 float16
    python tests/make_qwen35.py .tmp/made-up-qwen35-int8 int8
    python tests/make_qwen35.py .tmp/made-up-qwen35-int6 int6
    python tests/make_qwen35.py .tmp/made-up-qwen35-state int8 state
    # heads of 256, as every real Qwen3.5 has (the others' are 32 and 64); a plain memory with float16 keys and values too, which
    # a 4B or a 9B keeps past 4 GiB: the attention kernels on heads that wide
    python tests/make_qwen35.py .tmp/made-up-qwen35-wide-float32 float32 wide
    python tests/make_qwen35.py .tmp/made-up-qwen35-wide-int8 int8 wide
    for memory in "" --plain --wide; do
      node tests/forward-check.mjs .tmp/made-up-qwen35-float32 .tmp/made-up-qwen35-float16 .tmp/made-up-qwen35-int8 .tmp/made-up-qwen35-int6 .tmp/made-up-qwen35-state .tmp/made-up-qwen35-wide-float32 .tmp/made-up-qwen35-wide-int8 --rounds 1 --positions 128 $memory
    done
    node tests/forward-check.mjs .tmp/made-up-qwen35-wide-int8 --rounds 1 --positions 128 --plain --half-keys
  }
  part "forward.js against NumPy, a made-up Qwen3.5" made_up_qwen35
  # T148: the default choice of the GPU or the CPU for a prompt's blocks, with a made-up GPU's worker
  part "the GPU or the CPU by default" node tests/gpu-default-check.mjs
  part "the software threads" node tests/threads-check.mjs
  # T229: the value heads of a linear-attention layer's delta rule shared out among the threads, to the bit, and a thread
  # that stops in the middle of one (the review: its state is written beside the old one, so the phase can be run again)
  part "the software threads, a made-up Qwen3.5" node tests/threads-check.mjs .tmp/made-up-qwen35-float32 .tmp/made-up-qwen35-int8 .tmp/made-up-qwen35-state --rounds 1
  # and with every address above 4 GiB (--wide --high), where a Qwen3.5 4B or 9B keeps its states and its tensors of the linear layers
  part "the software threads, a made-up Qwen3.5 above 4 GiB" node tests/threads-check.mjs .tmp/made-up-qwen35-int8 .tmp/made-up-qwen35-state --wide --high --rounds 1 --positions 24
  # T206: the pre-tokenizers against the real ones at every code point (about 90 s, too long for the deploy)
  part "the pre-tokenizers at every code point" env EVERY_CODE_POINT=1 python -m pytest tests/test_bytebpe.py -q -k every_character
fi
echo "suite: $suite passed in $SECONDS s"
