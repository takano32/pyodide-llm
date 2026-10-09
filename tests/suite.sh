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
# T260: the kernel of an LFM2's convolution layer against the same arithmetic in JavaScript and in NumPy, to the bit
# (under a second)
part "the convolution layer's kernel" node tests/conv-check.mjs
# T230, T231: the kernels of the ternary weights against the same arithmetic in JavaScript, to the bit (under a second)
part "the ternary weights' kernels" node tests/ternary-check.mjs
# T237: the kernels of the rotated basis against the same arithmetic in JavaScript, to the bit (under a second)
part "the rotated basis's kernels" node tests/rotate-check.mjs
# T243 (the review): the look at the GPU's float16 keys and values, finite_f16, on the four builds of the kernels forward.js
# instantiates: every half at every place of a 16-byte line, the rest after the eights (no engine's kvDim leaves one), a bad
# half just outside the run, an address above 4 GiB (a few seconds)
part "the finite look at float16" node tests/finite-check.mjs
# T229 (the review): a Qwen3.5 is not put on a GPU where an adapter is there: forward.js's gpuUnfit says so first, and no
# adapter of CI's reaches that line (a few seconds)
hybrid_stays_on_the_cpu() {
  mkdir -p .tmp
  python tests/make_qwen35.py .tmp/made-up-qwen35-int8 int8
  node tests/gpu-hybrid-check.mjs .tmp/made-up-qwen35-int8
}
part "a Qwen3.5 where a GPU adapter is" hybrid_stays_on_the_cpu
# T260: nor is an LFM2, whose convolution layers have no shader: the next line of gpuUnfit (a few seconds)
lfm2_stays_on_the_cpu() {
  python tests/make_lfm2.py .tmp/made-up-lfm2-int8 int8
  node tests/gpu-hybrid-check.mjs .tmp/made-up-lfm2-int8
}
part "an LFM2 where a GPU adapter is" lfm2_stays_on_the_cpu
# T217 (the review of T201): attention's softmax where its largest score decides something (two positions far above
# the rest): a largest that leaves positions out, which forward-check's line cannot see (under a second)
part "attention's largest score" node tests/attention-check.mjs

if [ "$suite" = full ]; then
  # T262's review: every workflow job has a time limit and every step that installs a browser or runs apt has its own (and
  # apt's time-outs set first): the rule that was a sentence in AGENTS.md. Not the deploy's: a missing limit cannot break the
  # page, and the deploy's suite is what keeps a broken page off the site (T193); the next full suite says it (under a second)
  part "the workflows' time limits" node tests/workflows-check.mjs
  # T346: the files past the size a file should have, said every night (never a failure)
  part "the sizes of the files" node tests/unchanged.mjs sizes
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
    # Safari's path, with no relaxed SIMD (8-bit activations: matmul_q8 and matmul_q6), where a 4B is held in six bits
    node tests/forward-check.mjs .tmp/made-up-qwen35-int8 .tmp/made-up-qwen35-int6 .tmp/made-up-qwen35-state --without relaxed --rounds 1 --positions 128
  }
  part "forward.js against NumPy, a made-up Qwen3.5" made_up_qwen35
  # T230, T231: made-up ternary models (the real ones are too large for the build): the shape of Ternary Bonsai 1.7B,
  # the same with a classifier of its own and outlier channels, a hybrid one as Ternary Bonsai 2 27B is, and that one
  # in a rotated basis (T237), as the 27B's file is, and that with the embedding as the classifier (version 2 of the
  # basis: the review of T237); on a
  # shared memory, a plain one and a 64-bit one, and without relaxed SIMD (matmul_t2, the same numbers to the bit)
  made_up_ternary() {
    mkdir -p .tmp
    for kind in qwen3 own hybrid rotated rotated-tied; do python tests/make_ternary.py .tmp/made-up-ternary-$kind ternary $kind; done
    for memory in "" --plain --wide "--without relaxed"; do
      node tests/forward-check.mjs .tmp/made-up-ternary-qwen3 .tmp/made-up-ternary-own .tmp/made-up-ternary-hybrid .tmp/made-up-ternary-rotated .tmp/made-up-ternary-rotated-tied --rounds 1 --positions 128 $memory
    done
  }
  part "forward.js against NumPy, made-up ternary models" made_up_ternary
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
  # T260: a made-up LFM2 (convolution layers among attention layers; no real one is small enough for the build): float32
  # and float16 to NumPy's numbers, int8 and six bits within the line of a made-up model, the prompt in blocks to the
  # bit, a position out of turn refused, the memory after the checkpoint against footprint(); the 350M's order of
  # layers, the 230M's with a classifier of its own, and four taps; on a shared memory, a plain one and a 64-bit one,
  # and without relaxed SIMD (Safari's path)
  made_up_lfm2() {
    mkdir -p .tmp
    for dtype in float32 float16 int8 int6; do python tests/make_lfm2.py .tmp/made-up-lfm2-$dtype $dtype; done
    for shape in own four; do
      python tests/make_lfm2.py .tmp/made-up-lfm2-$shape-float32 float32 $shape
      python tests/make_lfm2.py .tmp/made-up-lfm2-$shape-int8 int8 $shape
    done
    for memory in "" --plain --wide; do
      node tests/forward-check.mjs .tmp/made-up-lfm2-float32 .tmp/made-up-lfm2-float16 .tmp/made-up-lfm2-int8 .tmp/made-up-lfm2-int6 .tmp/made-up-lfm2-own-float32 .tmp/made-up-lfm2-own-int8 .tmp/made-up-lfm2-four-float32 .tmp/made-up-lfm2-four-int8 --rounds 1 --positions 128 $memory
    done
    node tests/forward-check.mjs .tmp/made-up-lfm2-int8 .tmp/made-up-lfm2-int6 .tmp/made-up-lfm2-own-int8 .tmp/made-up-lfm2-four-int8 --without relaxed --rounds 1 --positions 128
  }
  part "forward.js against NumPy, a made-up LFM2" made_up_lfm2
  # T255: a made-up SmolLM3 (every fourth layer's q and k not turned by RoPE; the real one is too large for the build):
  # float32 and float16 to NumPy's numbers, int8 and six bits within the line of a made-up model, the prompt in blocks
  # to the bit; on a shared memory, a plain one and a 64-bit one, and without relaxed SIMD (Safari's path)
  made_up_smollm3() {
    mkdir -p .tmp
    for dtype in float32 float16 int8 int6; do python tests/make_smollm3.py .tmp/made-up-smollm3-$dtype $dtype; done
    for memory in "" --plain --wide; do
      node tests/forward-check.mjs .tmp/made-up-smollm3-float32 .tmp/made-up-smollm3-float16 .tmp/made-up-smollm3-int8 .tmp/made-up-smollm3-int6 --rounds 1 --positions 128 $memory
    done
    node tests/forward-check.mjs .tmp/made-up-smollm3-int8 .tmp/made-up-smollm3-int6 --without relaxed --rounds 1 --positions 128
  }
  part "forward.js against NumPy, a made-up SmolLM3" made_up_smollm3
  # T148: the default choice of the GPU or the CPU for a prompt's blocks, with a made-up GPU's worker
  part "the GPU or the CPU by default" node tests/gpu-default-check.mjs
  part "the software threads" node tests/threads-check.mjs
  # T229: the value heads of a linear-attention layer's delta rule shared out among the threads, to the bit, and a thread
  # that stops in the middle of one (the review: its state is written beside the old one, so the phase can be run again)
  part "the software threads, a made-up Qwen3.5" node tests/threads-check.mjs .tmp/made-up-qwen35-float32 .tmp/made-up-qwen35-int8 .tmp/made-up-qwen35-state --rounds 1
  # and with every address above 4 GiB (--wide --high), where a Qwen3.5 4B or 9B keeps its states and its tensors of the linear layers
  part "the software threads, a made-up Qwen3.5 above 4 GiB" node tests/threads-check.mjs .tmp/made-up-qwen35-int8 .tmp/made-up-qwen35-state --wide --high --rounds 1 --positions 24
  part "the software threads, made-up ternary models" node tests/threads-check.mjs .tmp/made-up-ternary-qwen3 .tmp/made-up-ternary-rotated --rounds 1
  part "the software threads, a made-up Qwen3.5 in a rotated basis" node tests/threads-check.mjs .tmp/made-up-rotated-float32 .tmp/made-up-rotated-int8 --rounds 1
  # T260: an LFM2's matrices shared out among the threads, to the bit, a thread that stops in the middle of a phase (the
  # convolution's rows are this thread's alone, outside every phase), and with every address above 4 GiB
  part "the software threads, a made-up LFM2" node tests/threads-check.mjs .tmp/made-up-lfm2-float32 .tmp/made-up-lfm2-int8 .tmp/made-up-lfm2-four-int8 --rounds 1
  part "the software threads, a made-up LFM2 above 4 GiB" node tests/threads-check.mjs .tmp/made-up-lfm2-int8 .tmp/made-up-lfm2-four-int8 --wide --high --rounds 1 --positions 24
  # T233's review: the tool that holds Ternary Bonsai 2 27B to its references (tests/page_27b.sh) is run when the model's
  # computation may have changed, which is seldom, and a tool that is seldom run goes out of step with forward.js unseen. Its
  # dry stage runs a made-up model of the 27B's kind through the comparison, the long prompt, the speed and the memory, the
  # engine broken on purpose where it must be caught (a minute, no download)
  part "the 27B's tool, on a made-up model" env STAGES=dry bash tests/page_27b.sh
  # T206: the pre-tokenizers against the real ones at every code point (about 90 s, too long for the deploy)
  part "the pre-tokenizers at every code point" env EVERY_CODE_POINT=1 python -m pytest tests/test_bytebpe.py -q -k every_character
fi
echo "suite: $suite passed in $SECONDS s"
