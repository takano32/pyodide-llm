#!/usr/bin/env bash
# tests/suite.sh (T.193): the tests of deploy.yml and tests.yml, in one place so that a branch's tests.yml runs the very
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
# T.384: no test waits forever. One run of the full suite stood still for 88 minutes after a test's last line with nothing
# in the log to say where (run 38024999470). Every node of the suite has a limit (TEST_LIMIT seconds, 30 minutes: the
# longest test takes about ten); at the limit tests/limited.mjs prints where the test stood and what kept it alive, and
# kills it, which fails the suite.
node() {
  # (T.584: with RUNTIME=bun, the tests Bun cannot run are left out, each with a line that says so: TASKS.md's T.585)
  if [ "${RUNTIME:-node}" = bun ] && [ -n "${BUN_LEAVES[$1]:-}" ]; then echo "$1: left out with Bun (${BUN_LEAVES[$1]})"; return 0; fi
  command "${RUNTIME:-node}" tests/limited.mjs "${TEST_LIMIT:-1800}" "$@"
}
# T.584: what Bun (1.4.4) cannot run, and why
declare -A BUN_LEAVES=(
  [tests/memory-check.mjs]="its WebAssembly memory grows by a copy, about a second a time for a 1.8 GB model: past the limit"
  [tests/blocks-forward-check.mjs]="its 64-bit WebAssembly memory is off (BUN_JSC_useWasmMemory64), and on a shared one posted to a worker arrives 32-bit: oven-sh/bun#45739"
)
# each part's name and seconds, for the log (tests/ci.mjs reads the lines that start with "suite:" and "--- ")
part() {
  local name=$1 began=$SECONDS
  shift
  echo "--- $name"
  "$@"
  echo "--- $name: $((SECONDS - began)) s"
}

part "unit tests of the engine" python -m pytest tests -q
# the ledger: TASKS.md keeps the tasks still moving and a table of contents made from their headings, docs/tasks/ the done ones
part "the ledger's tables of contents" python tests/ledger.py --check

# Node alone: the benchmark's tables (T.45), the table CI writes (T.82), the list, the kept models, the service worker...
page_modules() {
  node tests/bench.mjs
  node tests/page-memory.mjs
  node tests/wake.mjs
  node tests/summary-check.mjs
  node tests/models-check.mjs
  # T.367.1: where in a tree what is (tests/tree.mjs, tests/tree.py and their one rule), on made-up trees of every layout
  # the tools are to read; and the one reader of how a file takes its neighbours' names (tests/imports.mjs), on made-up texts
  node tests/tree-check.mjs
  node tests/imports-check.mjs
  # T.391.6: the server of the site and the end of a tool, which every tool that drives a browser shares (tests/serving.mjs,
  # tests/leave.mjs): no browser, a made-up folder, a close that never returns (a second or so)
  node tests/serving-check.mjs
  # T.357.5: the stand-ins the checks below run the product on (tests/fakes/: the Cache API, fetch, a Worker's face, the
  # clock of a token, Pyodide's proxies and a generator's face), each by itself: what each does as the thing it stands
  # for, which a check that uses it leans on without saying (under a second)
  node tests/fakes-check.mjs
  # T.357.6: the Node tools' one place for fetching (tests/fetching.mjs: a file held to its length, asked for again where
  # another try may mend it, not where the server refused) on a server of this machine; and through it and Python's
  # (tests/fetching.py) the text of Wikipedia and the files gguf.yml compares (a second or so; nothing leaves the machine)
  node tests/fetching-check.mjs
  # T.368.1: the type check of what a browser runs (tsc --noEmit on three programs: the pages' scripts, the runtime's
  # TypeScript, the Service Worker as the JavaScript it is), that a type's import is said as one, and the count of the
  # anys that are written (T.473: no more than there are now); T.368.8: and that a name a file of the runtime uses without
  # declaring or importing it is an ambient name the runtime means (the one thing of a wrong name that tsc lets pass).
  # Then the tool that says whether two builds are the same code (tests/dist-same.mjs, the proof of a step that only
  # adds types) and what blanks a .ts file's types for the tools, on made-up builds and texts; and (T.481) the tool that
  # says whether the sources of two commits are the same code by their syntax trees (tests/same-syntax.mjs: what a
  # bundler prints away, a shorthand opened, a number written otherwise), on made-up texts
  node tests/types.mjs
  node tests/dist-same-check.mjs
  node tests/same-syntax-check.mjs
  # T.368.2: the shapes that cross a line as plain objects are types (src/runtime/contracts/): what reads them for the
  # checks that hold each to its two ends (plan-keys-check, worker-modules-check and worker-sink-check, below), on
  # made-up types; that those files hold types alone and are read with `import type`; and the conduct's requests held
  # to the tuples conduct.py yields and to the answerers
  node tests/contracts-check.mjs
  # T.367.3.1: what is asked of a build (tests/dist-check.mjs, run on the build itself after `npm run build`) and how a
  # tool reads a module of a build (tests/bundle.mjs), on made-up builds
  node tests/dist-rules-check.mjs
  # T.367.2: the site's Python as two archives (python_archive.py): what goes in and that the same sources give the same
  # bytes, that they hold what the tools walk and place, and the worker's way with them (a stand-in of Pyodide, and Pyodide)
  node tests/python-archive-check.mjs
  # T.367.4: what says "the site was updated while this page was open", with stand-ins of a fetch (404, the 500s, a line
  # that dropped, no answer), and, in the sources, what becomes of each file that is asked for later (under a second)
  # T.380.1: how a failure is carried and told apart (src/runtime/failure.ts): a Failure and its kinds, what is read of
  # whatever else is thrown (Python's exceptions as Pyodide hands them over, a memory that ran out in its three shapes,
  # a thing that is no error), and that the sources put no mark on an Error in another way (under a second)
  node tests/failure-check.mjs
  # T.380.1: the table of why a model is not on the GPU (src/runtime/forward/reasons.ts): every sentence whole, one
  # cause one sentence, written nowhere else, and read by forward.ts's parts, the GPU's worker and the pages alone
  node tests/reasons-check.mjs
  # T.477: forward/paths.ts, what /benchmark/ times of the model page's own path, alone: a stand-in of the engine and
  # a made-up clock (the sides in turn, the warm-up left out, a GPU's side not taken whole, the threads' search) (under a second)
  node tests/paths-check.mjs
  # T.380.2: every place that catches a failure and tells nobody (tests/silent.mjs reads them off the syntax trees),
  # held to a table of why it may; one that is written without a row fails, and is told the ways out (under a second)
  node tests/silent-check.mjs
  node tests/updated-check.mjs
  # T.439: what the worker tells the page where it goes on, slower, without a file of its build that is gone (stand-ins)
  node tests/worker-outdated-check.mjs
  node tests/ladder-check.mjs
  node tests/kept-check.mjs
  node tests/coi-js-check.mjs
  node tests/gpu-choice-check.mjs
  # T.366: the device's key holds the WGSL a device is given and no function's source: one character more in any piece of
  # the engine's shaders moves it, and a formatter's, a minifier's or a bundler's way with the file does not (about 15 s)
  node tests/device-key-check.mjs
  # T.351: what tsc does not hold of the names shaders.ts (the window) and its modules (src/runtime/shaders/) hand one
  # another: each module comes after those it takes from, and the window exports every name the modules export where
  # they are declared (the tools read the window as JavaScript: a name dropped is undefined there without a word)
  node tests/shaders-modules-check.mjs
  # T.379.1: the catalogue of the engine's shaders (shaders/engine.ts) by itself: its rows in the recorded order (the
  # order is the device's key), every combination of each maker's choices, no text twice, ENGINE the rows by name, and
  # the key the fold of the rows (under a second)
  node tests/shaders-catalogue-check.mjs
  # T.225's review: /benchmark/'s layer check against devices that round the cache's float16 as WGSL lets them (about 10 s)
  node tests/layer-check.mjs
  node tests/worker-sink-check.mjs
  # T.129: where fetching and loading meet in worker.js, on made-up fetches and a fast clock (a few seconds)
  node tests/worker-check.mjs
  # T.464 (out of it): the place of the weights, /benchmark/'s shared memory, what is thrown and is no Error, a stop during
  # a run and the GPU lost under one, on the same harness (a few seconds)
  node tests/worker-memory-check.mjs
  # T.357: what a conversion asks of huggingface.co, in order, for every kind of source the page converts (one file, shards, a
  # GGUF alone and with another repository's vocabulary, the tokenizer's candidates, the optional files), on a made-up hub
  # and a converter's stand-in, against tests/fixtures/conversion-fetches.json (T.374.2.1: by the real conduct in Pyodide
  # and the worker's loop that answers it: about five seconds)
  node tests/worker-fetches-check.mjs
  # T.374.2.1: the worker's loop that answers a conversion's conduct, alone (a stand-in generator, no Pyodide): every kind
  # of request, a file that is not there, failures and cancelled loads, every proxy let go, and T.403's file handle
  node tests/worker-conduct-check.mjs
  # T.374.4: what the Node tools that convert a model answer the same conduct with (tests/conducting.mjs), alone: a
  # stand-in generator, a folder under .tmp/, no Pyodide (under a second)
  node tests/conducting-check.mjs
  # T.350, T.378.3: what tsc does not hold of the worker and its modules (src/runtime/worker/): no imported name hidden
  # by a local, no top-level name twice (one name in the one context the two checks above run them in); T.368.2: the
  # messages' types have nothing that nobody writes or reads, and every message posted says its type to tsc
  node tests/worker-modules-check.mjs
  # T.378.3: what the worker's modules hold has owners: request.js (what the page asked for with a load) as the module
  # it is, and loaded.js (the model loaded now) on a stand-in of Pyodide and of a model: no model, no network
  node tests/worker-parts-check.mjs
  # T.462: the worker's start (runtime.ts's init()) run whole: the version of Pyodide, its loader by its URL, engine.zip,
  # the kernels for Python, forward.ts's kernels, and NumPy where forward.ts does not come (a made-up Pyodide; a second)
  node tests/worker-init-check.mjs
  # T.349: what tsc does not hold of what forward.ts and its modules (src/runtime/forward/) hand one another: no ring of
  # imports, a part takes nothing createForward() never hands it, every key made of shape, weights, … is taken by a
  # part, and what is read of the two parts' state is their getters
  node tests/forward-modules-check.mjs
  # T.357: the plan Python hands forward.js and the plan forward.js hands the GPU's worker, each held to both of its ends:
  # a key written and read nowhere, or read and never written (undefined without a word), for one made-up model of every
  # layout and dtype (under a second); T.375.2: and the one plan written by hand, /benchmark/'s made-up
  # model's (src/runtime/benchmark/madeup.ts), held to the plan Python makes for the same header
  node tests/plan-keys-check.mjs
  # T.622: a token's frame as createForward() lays it out from the plan, held array by array to the one footprint()
  # counts from the header and the form (forward/memory.ts's frameOf()), every architecture and kind of layer (a second)
  node tests/frame-check.mjs
  # T.355: the names the model page's script (src/pages/index.astro and src/page/) and /benchmark/'s use: declared,
  # imported, or a page's globals (the .astro scripts are in no program of tsc's, the build does not say a name that
  # is nowhere, and tsc takes a page's own names for the modules'); T.378.5: what the parts hold has owners (model.ts,
  # answer.ts, running.ts): no imported name hidden by a local, no top-level name twice, and the owners read nothing
  # of a page
  node tests/page-modules-check.mjs
  # T.378.5: the owners of the pages' state each by itself, imported by Node as the modules they are (model.ts, answer.ts,
  # running.ts: what each event changes and nothing else), and the model a visit opens with (opening.ts, a process a visit)
  node tests/page-parts-check.mjs
  # T.556: what an answer's Markdown makes with the latest marked (from jsDelivr, as the page takes it), before DOMPurify:
  # what is drawn, and what is said as text (the model's HTML, links, images). The browser's part: markdown-in-browser.mjs
  node tests/markdown-check.mjs
  # T.505.3: which file of a repository nobody listed is its model (src/page/repository.ts: safetensors, else the best
  # GGUF by the order of the quantizations, else the owner's sentence), on made-up lists and on real ones' names
  node tests/repository-check.mjs
  # T.353: what tsc does not hold of /benchmark/'s GPU worker (src/runtime/benchmark/gpu.ts) and its modules: no ring of
  # imports, every field of the object the steps share used, every message posted says its type, a message as the
  # window's onmessage is set answered once; and the names the tables' modules (src/bench/, JavaScript) use
  node tests/bench-modules-check.mjs
  # T.352, T.378.4: what tsc does not hold of the model's GPU worker (src/runtime/gpu.ts) and its modules (src/runtime/gpu/):
  # its state's owners (model, stopping and lost are hidden nowhere and read where a function runs; every field of the
  # model's record is set in its owner's file and read somewhere), every message posted says its type, and messages as
  # the window's receiver is set played once, in order (the worker runs only where there is a GPU)
  node tests/gpu-modules-check.mjs
  # T.378.4: the parts of that worker that hold its state, each by itself on a WebGPU that writes down what it is asked
  # and computes nothing: device.js's record and its end, weights.js's (and a checkpoint that comes in stretches, for a
  # model on the GPU alone), block.js's (the cache grown, a block's dispatches and answer) (under a second)
  node tests/gpu-worker-parts-check.mjs
  # T.465.1: forms.ts choosing a prompt's attention and tiled forms on that WebGPU, whose checks' dispatches are answered
  # as the shaders would (tests/fakes/gpu-answers.ts): every form right, and each check refusing an answer a little off
  # (about 15 s)
  node tests/gpu-forms-check.ts
  # T.357.3: the parts of the tool that checks the GPU in CI (tests/gpu-check.mjs, which runs there alone), each by itself
  # on made-up numbers: its measures, which made-up models an engine runs, a model's verdict and its steps' for runs as
  # they should be and wrong in one thing each, the tables, that the harness is a module whose imports are files of this
  # tree, the references' two classes in the native Python (under a second)
  node tests/gpu-check-parts-check.mjs
  # T.130's review, on forward.js's createForward() with kernels that do nothing: the KV cache grows in place without losing a
  # byte, and footprint() holds what is allocated (a few seconds); T.223's: the search for the software threads on noisy times
  node tests/memory-check.mjs
  node tests/thread-search-check.mjs
  # T.378.1: the software threads' part (forward/threads.js) by itself, with stand-ins for its neighbours: what the engine
  # tells it (a count to use, a search, a generation, a token, a stop) changes what it says, in any order (under a second)
  node tests/threads-part-check.mjs
  # T.378.2: the GPU's side (forward/gpuside.js) by itself the same way: a stand-in of the GPU's worker that answers on this
  # thread, a clock the test moves, two pages of memory: what the engine tells it (a generation, a prompt, a step, where
  # the CPU wrote, a side, a let-go) and every way a request is refused (under a second)
  node tests/gpu-part-check.mjs
  # T.450: an engine on the GPU alone made in Node (kernels that do nothing, the GPU's stand-in): it starts no software
  # thread whatever is asked, its blocks and steps go to the GPU, and a token on the CPU loses it (under two seconds)
  node tests/gpu-alone-check.mjs
  # T.375.2: forward.ts reads what a model's layers are from the facts of Python's plan and from no architecture's name:
  # its createForward() on the kernels against the reference in NumPy (the native Python), for small float32 models of
  # the five architectures and of three whose facts are no architecture's pair (RMSNorms with GPT-2's FFN, LayerNorms
  # with a gated one, a Llama RoPE turns no layer of); and a plan without its layers' facts is refused (under a second)
  node tests/plan-facts-check.mjs
  # T.375.3: the parts the engine chooses as it is made, each by itself with stand-ins: the three kinds of layers, the two
  # FFNs and the cache of the keys and values (what goes out of a step, in what order and where; what position 0 clears)
  node tests/layer-parts-check.mjs
  # T.375.4: the two tables of jobs.ts (the kinds of jobs; how a matrix's weights are packed) by themselves, with kernels
  # that write down their calls: every kind's call, tile and blocks, the warm-up, and each packing against Python's
  # registry of the dtypes (under a second)
  node tests/kernel-registry-check.mjs
  # T.375.5: what sizes a model before it is loaded, by the same tables as the engine: the architectures' lines, a
  # model's sizes of its header and form, footprint() with a table of kinds written in the test, and the GPU's side of
  # it (stand-ins alone; under a second)
  node tests/sizes-check.mjs
}
part "unit tests of the page's modules" page_modules

# the page resolves the latest Pyodide release when it is opened, so test against that one
part "the latest Pyodide" npm install --no-save pyodide@latest
part "smoke test" node tests/smoke.mjs
# T.229: the kernels of Qwen3.5's linear attention against the same arithmetic in JavaScript (under a second)
part "the delta rule's kernels" node tests/delta-check.mjs
# T.260: the kernel of an LFM2's convolution layer against the same arithmetic in JavaScript and in NumPy, to the bit
# (under a second)
part "the convolution layer's kernel" node tests/conv-check.mjs
# T.230, T.231: the kernels of the ternary weights against the same arithmetic in JavaScript, to the bit (under a second)
part "the ternary weights' kernels" node tests/ternary-check.mjs
# T.363.4: the kernels of llama.cpp's blocks (Underdog Saluki's weights as its GGUF has them) against the same arithmetic
# in JavaScript, to the bit, and the reading against the converter's own (one call of Python); every build, and each as
# Liftoff alone and TurboFan alone compile it (about 4 seconds)
part "the kernels of llama.cpp's blocks" node tests/blocks-check.mjs
# T.363.5: the page's forward pass on small models whose matrices are kept as llama.cpp's blocks (a type for each layer:
# Underdog Saluki's shape, and blocks beside rows of other kinds), on the real kernels: every call of the blocks' kernels
# held to the plan and to the model of the kernels, the logits to the engine in NumPy's, and the same logits to the bit
# without relaxed SIMD, with either tile, after a block, on a 64-bit memory and on software threads (about 14 seconds)
part "the forward pass on llama.cpp's blocks" node tests/blocks-forward-check.mjs
# T.237: the kernels of the rotated basis against the same arithmetic in JavaScript, to the bit (under a second)
part "the rotated basis's kernels" node tests/rotate-check.mjs
# T.243 (the review): the look at the GPU's float16 keys and values, finite_f16, on the four builds of the kernels forward.js
# instantiates: every half at every place of a 16-byte line, the rest after the eights (no engine's kvDim leaves one), a bad
# half just outside the run, an address above 4 GiB (a few seconds)
part "the finite look at float16" node tests/finite-check.mjs
part "keys and values that were not finite, in a float16 cache" node tests/unfinite-check.mjs
# T.229 (the review): a Qwen3.5 was not put on a GPU where an adapter is there (no adapter of CI's reaches gpuUnfit's lines);
# T.271.2: its prompts are the GPU's now, its worker started with a plan of its linear-attention layers (a few seconds)
qwen35_goes_to_the_gpu() {
  mkdir -p .tmp
  python tests/make_qwen35.py .tmp/made-up-qwen35-int8 int8
  node tests/gpu-hybrid-check.mjs .tmp/made-up-qwen35-int8
  # (T.568.4: and one whose key heads are wider than the GPU's delta rule takes stays on the CPU, as its kind says)
  python tests/make_qwen35.py .tmp/made-up-qwen35-past int8 past
  node tests/gpu-hybrid-check.mjs .tmp/made-up-qwen35-past
}
part "a Qwen3.5 where a GPU adapter is" qwen35_goes_to_the_gpu
# T.260: nor was an LFM2, whose convolution layers had no shader; T.545: its prompts are the GPU's now, its worker started with
# a plan of its layers' kinds and places, its taps and rows (a few seconds)
lfm2_goes_to_the_gpu() {
  python tests/make_lfm2.py .tmp/made-up-lfm2-int8 int8
  node tests/gpu-hybrid-check.mjs .tmp/made-up-lfm2-int8
}
part "an LFM2 where a GPU adapter is" lfm2_goes_to_the_gpu
# T.261: a Gemma 3 was kept off the GPU here; T.261.4: its prompts are the GPU's now, its worker started with a plan of what
# it has besides (a few seconds)
gemma3_goes_to_the_gpu() {
  python tests/make_gemma3.py .tmp/made-up-gemma3-int8 int8
  node tests/gpu-hybrid-check.mjs .tmp/made-up-gemma3-int8
}
part "a Gemma 3 where a GPU adapter is" gemma3_goes_to_the_gpu
# T.342: nor is a Spark-X2.5, whose gate of the heads and exact GELU no shader has: gpuUnfit's line of the pair of norms
# and FFN (a few seconds)
spark_stays_on_the_cpu() {
  python tests/make_spark.py .tmp/made-up-spark-int8 int8
  node tests/gpu-hybrid-check.mjs .tmp/made-up-spark-int8
}
part "a Spark-X2.5 where a GPU adapter is" spark_stays_on_the_cpu
# T.357.3: the tool that checks the GPU in CI (tests/gpu-check.mjs), run from its arguments to its exit with a stand-in
# for Dawn that gives no device: NumPy's answers, the plans, the server, the worker threads, the harness's runs on the
# CPU, the verdicts and the tables, on three made-up models. No shader runs (about ten seconds, Pyodide's start)
part "the GPU's check without a GPU" node tests/gpu-check-dry-check.mjs
# T.357.4: the three tools that hold the engine to transformers in CI (tests/reference_llama.py, reference_qwen35.py,
# reference_lfm2.py), run from their arguments to their exit with stand-ins for torch and transformers that answer with
# the engine itself: the made-up models, the conversions, the lines, the states, the long text, every fault, and that a
# transformers the engine is not within the line of is said to be so. Nothing is held to transformers (about 20 seconds)
part "the reference tools without torch" python tests/reference_dry.py
# T.217 (the review of T.201): attention's softmax where its largest score decides something (two positions far above
# the rest): a largest that leaves positions out, which forward-check's line cannot see (under a second)
part "attention's largest score" node tests/attention-check.mjs
# T.458: the engine tells the GPU's side what happened (run() where the CPU wrote, forward() each step timed, a generation
# begun) on forward.js's real CPU pass of llm-jp-3 150M with a made-up GPU's worker: a GPU far faster, its prompts and its
# steps (gpu-default-check --quick; the whole check is the full suite's), about 20 seconds. Each of the four ways to
# forget failed only the full check before
part "what the engine tells the GPU's side" node tests/gpu-default-check.mjs --quick
# T.483: the three windows of a worker (the model's, its GPU's, /benchmark/'s GPU's) await nothing before their receiver
# is set: a module worker's first message is lost where they do (under a second)
part "the workers' receivers come first" node tests/receivers-check.mjs

if [ "$suite" = full ]; then
  # T.262's review: every workflow job has a time limit and every step that installs a browser or runs apt has its own (and
  # apt's time-outs set first): the rule that was a sentence in AGENTS.md. Not the deploy's: a missing limit cannot break the
  # page, and the deploy's suite is what keeps a broken page off the site (T.193); the next full suite says it (under a second)
  part "the workflows' time limits" node tests/workflows-check.mjs
  # T.346: the files past the size a file should have, said every night (never a failure)
  part "the sizes of the files" node tests/unchanged.mjs sizes
  # T.438: a built script that changes is named again by every file that names it, to the page's scripts (what T.367.4's
  # "the site was updated" rests on): the site built once, and once with each named script touched (about 3 minutes)
  part "a changed file's namers named again" node tests/names-follow-check.mjs
  # T.93: the forward pass of src/runtime/forward.js against NumPy's, on the site's models (the line is for 128 positions)
  part "forward.js against NumPy" node tests/forward-check.mjs --rounds 1 --positions 128
  part "forward.js against NumPy, not shared" node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --plain
  # T.130: a shared memory refused for a model whose float32 keys and values would not fit: float16 on a plain memory
  part "forward.js against NumPy, float16 on a plain memory" node tests/forward-check.mjs tiny-lm llm-jp-3-150m --rounds 1 --positions 128 --plain --half-keys
  # T.101: the 64-bit memory and its kernels
  part "forward.js against NumPy, 64-bit" node tests/forward-check.mjs stories260K tiny-lm --rounds 1 --positions 128 --wide
  # T.229: a made-up Qwen3.5 (hybrid attention; no real one is small enough for the build): float32 to NumPy's numbers,
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
  # T.230, T.231: made-up ternary models (the real ones are too large for the build): the shape of Ternary Bonsai 1.7B,
  # the same with a classifier of its own and outlier channels, a hybrid one as Ternary Bonsai 2 27B is, and that one
  # in a rotated basis (T.237), as the 27B's file is, and that with the embedding as the classifier (version 2 of the
  # basis: the review of T.237); on a
  # shared memory, a plain one and a 64-bit one, and without relaxed SIMD (matmul_t2, the same numbers to the bit)
  made_up_ternary() {
    mkdir -p .tmp
    for kind in qwen3 own hybrid rotated rotated-tied; do python tests/make_ternary.py .tmp/made-up-ternary-$kind ternary $kind; done
    for memory in "" --plain --wide "--without relaxed"; do
      node tests/forward-check.mjs .tmp/made-up-ternary-qwen3 .tmp/made-up-ternary-own .tmp/made-up-ternary-hybrid .tmp/made-up-ternary-rotated .tmp/made-up-ternary-rotated-tied --rounds 1 --positions 128 $memory
    done
  }
  part "forward.js against NumPy, made-up ternary models" made_up_ternary
  # T.237: the same made-up Qwen3.5 folded into a rotated basis (Ternary Bonsai 2 27B's): forward.js turns every
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
  # T.260: a made-up LFM2 (convolution layers among attention layers; no real one is small enough for the build): float32
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
  # T.261: a made-up Gemma 3 (a norm behind the attention and the FFN, a GELU in the gate, local layers; no real one is
  # small enough for the build): float32 and float16 to NumPy's numbers, int8 and six bits within the line of a made-up
  # model, the prompt in blocks to the bit, the memory after the checkpoint against footprint(); a window of 48
  # positions and one of 5 within the 128 compared, and one of 300 past where the cache first grows (400 positions); on a
  # shared memory, a plain one and a 64-bit one, and without relaxed SIMD (Safari's path). T.510.1: the local layers keep
  # their window alone, in a ring of the window and a block, 316 positions for the window of 300 (21 for the one of 5,
  # six times round in 128): on the shared memory to 960 positions, three times round, with blocks across its seam
  made_up_gemma3() {
    mkdir -p .tmp
    for dtype in float32 float16 int8 int6; do python tests/make_gemma3.py .tmp/made-up-gemma3-$dtype $dtype; done
    for shape in own long; do
      python tests/make_gemma3.py .tmp/made-up-gemma3-$shape-float32 float32 $shape
      python tests/make_gemma3.py .tmp/made-up-gemma3-$shape-int8 int8 $shape
    done
    for memory in "" --plain --wide; do
      node tests/forward-check.mjs .tmp/made-up-gemma3-float32 .tmp/made-up-gemma3-float16 .tmp/made-up-gemma3-int8 .tmp/made-up-gemma3-int6 .tmp/made-up-gemma3-own-float32 .tmp/made-up-gemma3-own-int8 --rounds 1 --positions 128 $memory
      node tests/forward-check.mjs .tmp/made-up-gemma3-long-float32 .tmp/made-up-gemma3-long-int8 --rounds 1 --positions "$([ -z "$memory" ] && echo 960 || echo 400)" $memory
    done
    node tests/forward-check.mjs .tmp/made-up-gemma3-int8 .tmp/made-up-gemma3-int6 .tmp/made-up-gemma3-own-int8 --without relaxed --rounds 1 --positions 128
  }
  part "forward.js against NumPy, a made-up Gemma 3" made_up_gemma3
  # T.331.2: a made-up Gemma 4 (a Gemma 3's layers, the full-attention ones with heads twice as wide, turned in part by the
  # whole head's angles; the heads of v normalized, a number every layer ends with, the logits capped): as the Gemma 3
  # above, each shape of tests/make_gemma4.py (T.330.2: shared, its last layers reading others' keys and values; T.329.2:
  # embedded, and an embedding of every layer, in every dtype)
  made_up_gemma4() {
    mkdir -p .tmp
    for dtype in float32 float16 int8 int6; do python tests/make_gemma4.py .tmp/made-up-gemma4-$dtype $dtype; done
    for shape in own long shared; do
      python tests/make_gemma4.py .tmp/made-up-gemma4-$shape-float32 float32 $shape
      python tests/make_gemma4.py .tmp/made-up-gemma4-$shape-int8 int8 $shape
    done
    for dtype in float32 float16 int8 int6; do python tests/make_gemma4.py .tmp/made-up-gemma4-embedded-$dtype $dtype embedded; done
    for memory in "" --plain --wide; do
      node tests/forward-check.mjs .tmp/made-up-gemma4-float32 .tmp/made-up-gemma4-float16 .tmp/made-up-gemma4-int8 .tmp/made-up-gemma4-int6 .tmp/made-up-gemma4-own-float32 .tmp/made-up-gemma4-own-int8 .tmp/made-up-gemma4-shared-float32 .tmp/made-up-gemma4-shared-int8 --rounds 1 --positions 128 $memory
      node tests/forward-check.mjs .tmp/made-up-gemma4-embedded-float32 .tmp/made-up-gemma4-embedded-float16 .tmp/made-up-gemma4-embedded-int8 .tmp/made-up-gemma4-embedded-int6 --rounds 1 --positions 128 $memory
      node tests/forward-check.mjs .tmp/made-up-gemma4-long-float32 .tmp/made-up-gemma4-long-int8 --rounds 1 --positions "$([ -z "$memory" ] && echo 960 || echo 400)" $memory
    done
    node tests/forward-check.mjs .tmp/made-up-gemma4-int8 .tmp/made-up-gemma4-int6 .tmp/made-up-gemma4-own-int8 .tmp/made-up-gemma4-shared-int8 .tmp/made-up-gemma4-embedded-int8 --without relaxed --rounds 1 --positions 128
  }
  part "forward.js against NumPy, a made-up Gemma 4" made_up_gemma4
  # T.342: a made-up Spark-X2.5 (local layers that turn their whole heads, the others a quarter of each; a gate of each
  # head's output; the exact GELU): as the Gemma 3 above, each shape of tests/make_spark.py
  made_up_spark() {
    mkdir -p .tmp
    for dtype in float32 float16 int8 int6; do python tests/make_spark.py .tmp/made-up-spark-$dtype $dtype; done
    for shape in own long; do
      python tests/make_spark.py .tmp/made-up-spark-$shape-float32 float32 $shape
      python tests/make_spark.py .tmp/made-up-spark-$shape-int8 int8 $shape
    done
    for memory in "" --plain --wide; do
      node tests/forward-check.mjs .tmp/made-up-spark-float32 .tmp/made-up-spark-float16 .tmp/made-up-spark-int8 .tmp/made-up-spark-int6 .tmp/made-up-spark-own-float32 .tmp/made-up-spark-own-int8 --rounds 1 --positions 128 $memory
      node tests/forward-check.mjs .tmp/made-up-spark-long-float32 .tmp/made-up-spark-long-int8 --rounds 1 --positions "$([ -z "$memory" ] && echo 960 || echo 400)" $memory
    done
    node tests/forward-check.mjs .tmp/made-up-spark-int8 .tmp/made-up-spark-int6 .tmp/made-up-spark-own-int8 --without relaxed --rounds 1 --positions 128
  }
  part "forward.js against NumPy, a made-up Spark-X2.5" made_up_spark
  # T.255: a made-up SmolLM3 (every fourth layer's q and k not turned by RoPE; the real one is too large for the build):
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
  # T.148: the default choice of the GPU or the CPU for a prompt's blocks, with a made-up GPU's worker
  part "the GPU or the CPU by default" node tests/gpu-default-check.mjs
  part "the software threads" node tests/threads-check.mjs
  # T.229: the value heads of a linear-attention layer's delta rule shared out among the threads, to the bit, and a thread
  # that stops in the middle of one (the review: its state is written beside the old one, so the phase can be run again)
  part "the software threads, a made-up Qwen3.5" node tests/threads-check.mjs .tmp/made-up-qwen35-float32 .tmp/made-up-qwen35-int8 .tmp/made-up-qwen35-state --rounds 1
  # and with every address above 4 GiB (--wide --high), where a Qwen3.5 4B or 9B keeps its states and its tensors of the linear layers
  part "the software threads, a made-up Qwen3.5 above 4 GiB" node tests/threads-check.mjs .tmp/made-up-qwen35-int8 .tmp/made-up-qwen35-state --wide --high --rounds 1 --positions 24
  part "the software threads, made-up ternary models" node tests/threads-check.mjs .tmp/made-up-ternary-qwen3 .tmp/made-up-ternary-rotated --rounds 1
  part "the software threads, a made-up Qwen3.5 in a rotated basis" node tests/threads-check.mjs .tmp/made-up-rotated-float32 .tmp/made-up-rotated-int8 --rounds 1
  # T.260: an LFM2's matrices shared out among the threads, to the bit, a thread that stops in the middle of a phase (the
  # convolution's rows are this thread's alone, outside every phase), and with every address above 4 GiB
  part "the software threads, a made-up LFM2" node tests/threads-check.mjs .tmp/made-up-lfm2-float32 .tmp/made-up-lfm2-int8 .tmp/made-up-lfm2-four-int8 --rounds 1
  part "the software threads, a made-up LFM2 above 4 GiB" node tests/threads-check.mjs .tmp/made-up-lfm2-int8 .tmp/made-up-lfm2-four-int8 --wide --high --rounds 1 --positions 24
  # T.261: and a made-up Gemma 3, whose local layers hand each token's heads the cache from its window's first position on
  part "the software threads, a made-up Gemma 3" node tests/threads-check.mjs .tmp/made-up-gemma3-float32 .tmp/made-up-gemma3-int8 .tmp/made-up-gemma3-own-int8 --rounds 1
  # T.331.2: and a made-up Gemma 4, whose layers' heads and FFN are of two widths
  part "the software threads, a made-up Gemma 4" node tests/threads-check.mjs .tmp/made-up-gemma4-float32 .tmp/made-up-gemma4-int8 .tmp/made-up-gemma4-own-int8 .tmp/made-up-gemma4-shared-int8 .tmp/made-up-gemma4-embedded-int8 --rounds 1
  part "the software threads, a made-up Spark-X2.5" node tests/threads-check.mjs .tmp/made-up-spark-float32 .tmp/made-up-spark-int8 .tmp/made-up-spark-own-int8 --rounds 1
  # T.233's review: the tool that holds Ternary Bonsai 2 27B to its references (tests/page_27b.sh) is run when the model's
  # computation may have changed, which is seldom, and a tool that is seldom run goes out of step with forward.js unseen. Its
  # dry stage runs a made-up model of the 27B's kind through the comparison, the long prompt, the speed and the memory, the
  # engine broken on purpose where it must be caught (a minute, no download)
  part "the 27B's tool, on a made-up model" env STAGES=dry bash tests/page_27b.sh
  # T.364: the same for Underdog Saluki 27B's tool (tests/page_saluki.sh), which shares tests/page-27b.mjs with it: a made-up
  # model of llama.cpp's blocks through the reference's NumPy pass, both conversions and every mode (a minute)
  part "Saluki's tool, on a made-up model" env STAGES=dry bash tests/page_saluki.sh
  # T.206: the pre-tokenizers against the real ones at every code point (about 90 s, too long for the deploy)
  part "the pre-tokenizers at every code point" env EVERY_CODE_POINT=1 python -m pytest tests/test_bytebpe.py -q -k every_character
fi
echo "suite: $suite passed in $SECONDS s"
