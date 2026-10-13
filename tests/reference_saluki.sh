#!/usr/bin/env bash
# tests/reference_saluki.sh (T361): the reference for Underdog Saluki 27B, on a CI runner (never on the development
# machine: it fetches 7.9 GB and runs a 27B). Unlike Ternary Bonsai 2 27B (tests/reference_27b.sh) this model runs in
# stock llama.cpp, so the reference is llama.cpp itself (MIT, a pinned commit of ggml-org/llama.cpp), built here for
# the CPU, with tests/reference_27b_fork.cpp against its libraries (the driver uses llama.h's public API alone and
# builds against upstream as it is): the prompts' ids as llama.cpp tokenizes them, 16 greedy tokens of each and the
# logits of every position, both one token at a time and as one batch, and its tokens a second (the only speed on a
# CPU anyone has for this file). docs/notes/t361-saluki-2026-10-13.md has what the file is and how it is to be held.
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=40 extra="bash tests/reference_saluki.sh" \
#     --grep "^(llama|saluki|runner):" --minutes 50 --ref <branch>
# Five to seven minutes on a runner of 4 logical cores and 16 GB, x86-64 or arm64 (runner=ubuntu-24.04-arm): the file
# in 40 s, the build in 70 s, llama.cpp at 0.93 to 1.35 tokens a second (T361's runs).
#
# STAGES (default "head fetch llama agree keep"):
#   dry    no model and no llama.cpp (a minute, 12 MiB of the file by a Range request): the head of the file held to
#          what the notes say of it, and files of the driver's form with made-up logits through agree and keep, one
#          that is right and one that agree must refuse
#   head   the head of the fetched file (or, alone, of the file where it is) held to the notes
#   fetch  the file, to its length and its sha256
#   llama  llama.cpp built and run: SECONDS_FOR_LLAMA (3000) is when it stops writing; TEXTS: the first so many prompts
#   f32    (T364.1, not in the default) llama.cpp with float32 activations (tests/reference_saluki_patch.py; float32
#          keys and values, the weights not repacked): the tokens of the llama stage again, one at a time, into
#          f32/fork-<i>.single. It widens every row it multiplies by (20 s a token on 4 cores): SECONDS_FOR_F32 (1500)
#          is when a text stops being written, and one that is not finished keeps the rows it has. F32_PICK: the texts
#          ("0 1")
#   numpy  (T364.1, not in the default) the engine's NumPy forward pass over the GGUF (tests/reference_saluki_numpy.py):
#          float32, with the activations rounded as the page's kernels round them and as llama.cpp rounds them, and
#          broken on purpose; against llama.cpp as it is and with float32 activations (TIGHT_LINES holds it to that).
#          SAVE_LOGITS is $work/saved unless said: what tests/page_saluki.sh compares the page's forward pass with.
#          NUDGED="0,2": for these texts two more runs that round as the page does numbers a millionth off
#   agree  the driver's files: whole, finite, greedy, and how far llama.cpp's two paths are from each other
#   keep   what a later run compares with, into .tmp/keep/reference-saluki (tests.yml keeps it a week with the run)
# REFERENCES=<a run's id>: the files that run kept, in place of everything else (then agree alone).
# What llama.cpp computes here is not the model in float32: it rounds the activations of every matrix of a k- or
# i-quantized type to 8 bits (Q8_K: 256 values to a scale). Its logits move by 0.3 to 0.8 (now and then by 10) from one
# of its two paths to the other and from one CPU to another, and the text it writes differs between CPUs (T361's
# runs): no narrow line holds anything to these logits (the notes' section on the reference).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/.."
llama_commit=4413a192410fad1f054e9216114087a3e01527a0
file=Underdog-Saluki-27B-1.0-IQ2-mix.gguf
stages=${STAGES:-head fetch llama agree keep}
has() { [[ " $stages " == *" $1 "* ]]; }
threads=$(nproc)
python -c "import numpy" 2>/dev/null || python -m pip install -q numpy

if has dry; then
  dry=.tmp/reference-saluki-dry
  rm -rf "$dry/right" "$dry/wrong" "$dry/keep"
  python tests/reference_saluki.py head "$dry"
  python tests/reference_saluki.py made-up "$dry"
  python tests/reference_saluki.py agree "$dry/right" "$dry/keep"
  if python tests/reference_saluki.py agree "$dry/wrong" > "$dry/wrong.log"; then
    cat "$dry/wrong.log"
    echo "saluki: dry: agree let through a written token that is not its row's largest logit"
    exit 1
  fi
  grep -q "WRONG: text 0: wrote" "$dry/wrong.log" || { cat "$dry/wrong.log"; echo "saluki: dry: agree refused for another reason"; exit 1; }
  [ -s "$dry/keep/summary.json" ] || { echo "saluki: dry: no summary was written"; exit 1; }
  echo "saluki: dry: the head is the notes', agree takes the right files and refuses the wrong ones"
  exit 0
fi

echo "runner: $(lscpu | sed -n 's/^Model name: *//p' | head -1) ($(uname -m)), $threads logical cores, $(free -g | awk '/Mem:/{print $2}') GB of memory"
df -h / /mnt 2>/dev/null | sed 's/^/runner: /' || true

if [ -n "${REFERENCES:-}" ]; then
  gh run download "$REFERENCES" --repo "${GITHUB_REPOSITORY:-takano32/pyodide-llm}" --name kept --dir ".tmp/kept/$REFERENCES"
  python tests/reference_saluki.py agree ".tmp/kept/$REFERENCES/reference-saluki"
  exit $?
fi

# where 8 GB of model and the logits go: the first of these with 14 GB free
work=
for candidate in "$here/../.tmp" /mnt; do
  [ -d "$candidate" ] || mkdir -p "$candidate" 2>/dev/null || sudo mkdir -p "$candidate"
  free_kb=$(df -Pk "$candidate" | awk 'NR==2{print $4}')
  if [ "$free_kb" -gt $((14 * 1024 * 1024)) ]; then
    work="$candidate/reference-saluki"
    [ -w "$candidate" ] || { sudo mkdir -p "$work"; sudo chown "$(id -u):$(id -g)" "$work"; }
    mkdir -p "$work"
    break
  fi
done
[ -n "$work" ] || { echo "runner: no disk with 14 GB free"; exit 1; }
work=$(cd "$work" && pwd)
echo "runner: working in $work ($(df -Ph "$work" | awk 'NR==2{print $4}') free)"

if has fetch; then
  began=$SECONDS
  python tests/reference_saluki.py fetch "$work"
  echo "runner: fetched and hashed $file in $((SECONDS - began)) s"
fi
if has head; then
  if [ -f "$work/$file" ]; then python tests/reference_saluki.py head "$work" "$work/$file"; else python tests/reference_saluki.py head "$work"; fi
fi

# the prompts: tests/reference_27b.sh's four (the same base model and the same chat template, Qwen3.8-27B's: two bare
# texts, a turn that answers at once, and the turn the template writes with nothing said, whose system turn it adds)
prompts=("The capital of Japan is"
         "日本でいちばん高い山は"
         $'<|im_start|>user\nWhat is 17 times 24?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
         $'<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nWhat is 17 times 24?<|im_end|>\n<|im_start|>assistant\n<think>\n')
prompts=("${prompts[@]:0:${TEXTS:-${#prompts[@]}}}")

built() {  # llama.cpp at the pinned commit, its libraries and the driver against them (once in a work directory)
  [ -x "$work/reference_saluki_llama" ] && return 0
  local began=$SECONDS
  if [ ! -d "$work/llama.cpp/.git" ]; then
    git init -q "$work/llama.cpp"
    git -C "$work/llama.cpp" remote add origin https://github.com/ggml-org/llama.cpp
  fi
  git -C "$work/llama.cpp" fetch -q --depth 1 origin "$llama_commit"
  git -C "$work/llama.cpp" checkout -q FETCH_HEAD
  echo "llama: $(git -C "$work/llama.cpp" rev-parse HEAD), $(head -3 "$work/llama.cpp/LICENSE" | tr '\n' ' ')"
  # the libraries alone (no tools, no server, no examples)
  cmake -S "$work/llama.cpp" -B "$work/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=ON > "$work/cmake.log" 2>&1 \
    || { tail -40 "$work/cmake.log"; echo "llama: cmake failed"; exit 1; }
  cmake --build "$work/llama.cpp/build" --target llama -j "$threads" > "$work/build.log" 2>&1 \
    || { tail -60 "$work/build.log"; echo "llama: the build failed"; exit 1; }
  libraries=$(find "$work/llama.cpp/build" -name 'lib*.so' | sort)
  echo "llama: built $(echo "$libraries" | xargs -n1 basename | tr '\n' ' ')in $((SECONDS - began)) s"
  directories=$(echo "$libraries" | xargs -n1 dirname | sort -u)
  rpath=$(echo "$directories" | sed 's/^/-Wl,-rpath,/' | tr '\n' ' ')
  # shellcheck disable=SC2086
  g++ -std=c++17 -O2 "$here/reference_27b_fork.cpp" -I "$work/llama.cpp/include" -I "$work/llama.cpp/ggml/include" \
    -DREFERENCE_NO_REPACK $libraries $rpath -o "$work/reference_saluki_llama" \
    || { echo "llama: the driver did not build"; exit 1; }
}

if has llama; then
  rm -f "$work"/prompt-*.txt "$work"/fork-*
  for index in "${!prompts[@]}"; do printf '%s' "${prompts[$index]}" > "$work/prompt-$index.txt"; done
  built
  began=$SECONDS
  # (the lines come as the driver writes them: a run the job's limit ends has its progress in the log. The driver says
  # "fork:" of whatever llama.cpp it was built against)
  set +e
  "$work/reference_saluki_llama" "$work/$file" "$work" "$threads" "${SECONDS_FOR_LLAMA:-3000}" 16 "${prompts[@]}" 2>&1 \
    | tee "$work/llama.log" | grep -a --line-buffered -E '^fork:|eval time|load time|total time' \
    | grep -a --line-buffered -v 'ptq1_0 activations' | sed -u -e 's/^fork: //' -e 's/^/llama: /'
  code=${PIPESTATUS[0]}
  set -e
  [ "$code" = 0 ] || { tail -60 "$work/llama.log"; echo "llama: the run failed ($code)"; exit 1; }
  echo "llama: ran in $((SECONDS - began)) s"
  ls -la "$work"/fork-* | sed 's/^/llama: /'
fi

status=0
if has f32; then
  # (the texts of F32_PICK, each by itself: the driver replays the ids of <a directory>/fork-<i>.ids for i from 0, so a
  # text that is not the first is handed to it as fork-0.ids of a directory of its own)
  built
  began=$SECONDS
  python tests/reference_saluki_patch.py "$work/llama.cpp" | sed 's/^reference_saluki_patch: /f32: /'
  if cmake --build "$work/llama.cpp/build" --target llama -j "$threads" > "$work/build-f32.log" 2>&1; then
    echo "f32: built again in $((SECONDS - began)) s"
    mkdir -p "$work/f32"
    for index in ${F32_PICK:-0 1}; do
      [ -f "$work/fork-$index.ids" ] || { echo "f32: no ids of text $index (the llama stage, or REFERENCES, leaves them)"; status=1; continue; }
      began=$SECONDS
      rm -rf "$work/f32-one"
      mkdir -p "$work/f32-one/ids" "$work/f32-one/out"
      cp "$work/fork-$index.ids" "$work/f32-one/ids/fork-0.ids"
      set +e
      env BLOCKS_F32_ACTIVATIONS=1 NO_REPACK=1 KV_F32=1 REPLAY_FROM="$work/f32-one/ids" \
        "$work/reference_saluki_llama" "$work/$file" "$work/f32-one/out" "$threads" "${SECONDS_FOR_F32:-1500}" 16 "${prompts[$index]}" 2>&1 \
        | tee "$work/f32-$index.log" | grep -a --line-buffered -E '^fork:|eval time|total time' | sed -u -e 's/^fork: //' -e "s/^/f32: text $index: /"
      code=${PIPESTATUS[0]}
      set -e
      if [ "$code" = 0 ] && [ -s "$work/f32-one/out/fork-0.single" ]; then
        mv "$work/f32-one/out/fork-0.single" "$work/f32/fork-$index.single"
      else
        tail -40 "$work/f32-$index.log"; echo "f32: text $index: the run failed ($code)"; status=1
      fi
      echo "f32: text $index ran in $((SECONDS - began)) s"
    done
    ls -la "$work"/f32/fork-* 2>/dev/null | sed 's/^/f32: /' || true
  else
    tail -60 "$work/build-f32.log"
    echo "f32: llama.cpp with the patch did not build — FAILED (the other stages go on)"
    status=1
  fi
fi
if has numpy; then
  began=$SECONDS
  SAVE_LOGITS="${SAVE_LOGITS:-$work/saved}" python tests/reference_saluki_numpy.py "$work/$file" "$work" || status=1
  echo "reference: ran in $((SECONDS - began)) s"
fi
if has agree; then
  python tests/reference_saluki.py agree "$work" "$work" || status=1
fi

if has keep; then
  # what a later comparison reads (0.25 GB: a row of logits is 1 MB), kept with the run
  keep=.tmp/keep/reference-saluki
  mkdir -p "$keep"
  cp "$work"/fork-*.ids "$work"/fork-*.logits "$work"/fork-*.single "$work"/prompt-*.txt "$keep/"
  cp "$work/summary.json" "$work/llama.log" "$keep/" 2>/dev/null || true
  # (T364.1: and what the page's comparison reads: llama.cpp with float32 activations, the engine's NumPy runs)
  for more in f32 saved; do
    if [ -d "$work/$more" ]; then mkdir -p "$keep/$more"; cp "$work/$more"/* "$keep/$more/" 2>/dev/null || true; fi
  done
  echo "saluki: the reference is kept with this run: $(du -sh "$keep" | cut -f1) (REFERENCES=<this run's id> takes it)"
fi
exit $status
