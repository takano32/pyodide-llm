#!/usr/bin/env bash
# tests/reference_27b.sh (T238): a reference for Ternary Bonsai 2 27B, on a CI runner (never on the development machine:
# it fetches 5.95 GB). Widened to float32 the model is 107 GB, so the usual NumPy reference cannot hold it. Two
# references, and the comparison of the two:
#   (1) Prism ML's fork of llama.cpp (MIT, a pinned commit), built here for the CPU, with tests/reference_27b_fork.cpp
#       against its libraries: the prompts' ids as the fork tokenizes them, 16 greedy tokens and the logits of every
#       position, and its tokens/s (the only speed on a CPU anyone has for this model);
#   (2) tests/reference_27b.py: the engine's own NumPy forward pass (src/python/llama2_numpy.py) over the GGUF read by
#       memory map, a matrix widened at a time.
#
# And, since the review of T237, a third, between the two: the fork again, patched (tests/reference_27b_patch.py) to
# multiply its ternary matrices by float32 activations where it rounds them to Q8_0, over the very tokens of the first run
# (a replay) and with float32 keys and values, so that what stands between it and the engine's float32 forward pass is
# float32's rounding and not the rounding of the activations (a few tenths of a logit: the fork's own paths differ by that).
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=150 extra="bash tests/reference_27b.sh" \
#     --grep "^(fork|f32|reference|runner):" --minutes 170 --ref <branch>
#
# STAGES (default "fork f32 numpy"): which of the three run; SECONDS_FOR_FORK, SECONDS_FOR_F32: after how long the fork
# stops writing (the float32 one is slower: it widens every row it multiplies by); BATCH_F32=1: and the replayed tokens as
# one batch too (the fork's other path, for how far it is from itself in float32). TEXTS: how many of the four texts.
# STAGES=none fetches the file and writes the texts, no more (T233: tests/page_27b.sh works in the same directory).
# STAGES=long (T233's review): a text of 5,987 tokens through the fork, past the 4096 positions of the list's context:
# the first of its own, see the comment at its stage (LONG_ROWS, LONG_CTX, LONG_TOKENS, SECONDS_FOR_LONG).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
fork_commit=88c4bc60b9c9578f134385be9535e853f2db9b9f
model_revision=b072e1d3b35a0a630cece372c2127528e0994386
file=Ternary-Bonsai-2-27B-PTQ1_0.gguf
bytes=5946648928
sha256=53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3
stages=${STAGES:-fork f32 numpy}
threads=$(nproc)

echo "runner: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $threads logical cores, $(free -g | awk '/Mem:/{print $2}') GB of memory"
df -h / /mnt 2>/dev/null | sed 's/^/runner: /' || true

# where 6 GB of model and the logits go: the first of these with 12 GB free
work=
for candidate in "$here/../.tmp" /mnt; do
  [ -d "$candidate" ] || mkdir -p "$candidate" 2>/dev/null || sudo mkdir -p "$candidate"
  free_kb=$(df -Pk "$candidate" | awk 'NR==2{print $4}')
  if [ "$free_kb" -gt $((12 * 1024 * 1024)) ]; then
    work="$candidate/reference-27b"
    [ -w "$candidate" ] || { sudo mkdir -p "$work"; sudo chown "$(id -u):$(id -g)" "$work"; }
    mkdir -p "$work"
    break
  fi
done
[ -n "$work" ] || { echo "runner: no disk with 12 GB free"; exit 1; }
work=$(cd "$work" && pwd)
echo "runner: working in $work ($(df -Ph "$work" | awk 'NR==2{print $4}') free)"

began=$SECONDS
if [ ! -f "$work/$file" ] || [ "$(stat -c %s "$work/$file")" != "$bytes" ]; then
  curl -sS -L -f --retry 5 --retry-delay 10 -C - -o "$work/$file" \
    "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/$model_revision/$file"
fi
[ "$(stat -c %s "$work/$file")" = "$bytes" ] || { echo "runner: $file has $(stat -c %s "$work/$file") bytes, not $bytes"; exit 1; }
echo "runner: fetched $file in $((SECONDS - began)) s"
# (the size of a download that was resumed is not yet the file: HF's own sha256 of the revision's file, 25 s or so)
began=$SECONDS
got=$(sha256sum "$work/$file" | cut -d' ' -f1)
[ "$got" = "$sha256" ] || { echo "runner: $file has the sha256 $got, not $sha256"; exit 1; }
echo "runner: the sha256 of $file is the revision's ($((SECONDS - began)) s)"

# the prompts: two bare texts, a turn of a chat in the form that answers at once (no system turn), and one in the form that
# thinks, which has a system turn the template adds by itself ("Reasoning effort is set to xhigh": T228, what jinja2 renders
# from the GGUF's chat_template with nothing said), a longer text than the others (60 tokens or so)
prompts=("The capital of Japan is"
         "日本でいちばん高い山は"
         $'<|im_start|>user\nWhat is 17 times 24?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
         $'<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nWhat is 17 times 24?<|im_end|>\n<|im_start|>assistant\n<think>\n')

# TEXTS (T233): the first so many of them alone (a run that needs the first text only: 20 positions instead of 155)
prompts=("${prompts[@]:0:${TEXTS:-${#prompts[@]}}}")
rm -f "$work"/prompt-*.txt
for index in "${!prompts[@]}"; do printf '%s' "${prompts[$index]}" > "$work/prompt-$index.txt"; done

# the fork, built here, for the stages that run it (T233: the long one too)
case " $stages " in *" fork "*|*" long "*)
  began=$SECONDS
  if [ ! -d "$work/fork/.git" ]; then
    git init -q "$work/fork"
    git -C "$work/fork" remote add origin https://github.com/PrismML-Eng/llama.cpp
  fi
  git -C "$work/fork" fetch -q --depth 1 origin "$fork_commit"
  git -C "$work/fork" checkout -q FETCH_HEAD
  echo "fork: $(git -C "$work/fork" rev-parse HEAD), $(head -3 "$work/fork/LICENSE" | tr '\n' ' ')"
  case " $stages " in *" f32 "*) python "$here/reference_27b_patch.py" "$work/fork" | sed 's/^reference_27b_patch: /fork: /' ;; esac
  # the libraries alone (no tools, no server, no examples: they pull in what this needs none of)
  cmake -S "$work/fork" -B "$work/fork/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DGGML_NATIVE=ON > "$work/cmake.log" 2>&1 \
    || { tail -40 "$work/cmake.log"; echo "fork: cmake failed"; exit 1; }
  cmake --build "$work/fork/build" --target llama -j "$threads" > "$work/build.log" 2>&1 \
    || { tail -60 "$work/build.log"; echo "fork: the build failed"; exit 1; }
  libraries=$(find "$work/fork/build" -name 'lib*.so' | sort)
  echo "fork: built $(echo "$libraries" | xargs -n1 basename | tr '\n' ' ')in $((SECONDS - began)) s"
  directories=$(echo "$libraries" | xargs -n1 dirname | sort -u)
  rpath=$(echo "$directories" | sed 's/^/-Wl,-rpath,/' | tr '\n' ' ')
  # shellcheck disable=SC2086
  g++ -std=c++17 -O2 "$here/reference_27b_fork.cpp" -I "$work/fork/include" -I "$work/fork/ggml/include" \
    $libraries $rpath -o "$work/reference_27b_fork" \
    || { echo "fork: the driver did not build"; exit 1; }
;; esac

case " $stages " in *" fork "*)
  began=$SECONDS
  "$work/reference_27b_fork" "$work/$file" "$work" "$threads" "${SECONDS_FOR_FORK:-2400}" 16 "${prompts[@]}" \
    > "$work/fork.log" 2>&1 || { tail -60 "$work/fork.log"; echo "fork: the run failed"; exit 1; }
  grep -a -E '^fork:|eval time|load time|total time' "$work/fork.log" | sed 's/^\([^f]\)/fork: \1/'
  echo "fork: ran in $((SECONDS - began)) s"
  ls -la "$work"/fork-* | sed 's/^/fork: /'
;; esac

# T233's review: a text of about 6,000 tokens (tests/fixtures/long-27b.txt: English and Japanese prose of this repository's
# documents, 5,987 tokens), past the 4096 positions the list's context has, through the fork in batches of 256 with a
# context of LONG_CTX (8192): the logits of some positions only (LONG_ROWS: the first ones are the floor, a few are around
# 4096, the last ones are the longest context) and the 16 tokens it writes after it. At the fork's speed on a CPU of the
# runner (0.64 to 1.13 tokens a second) it takes 1.5 to 2.7 hours; SECONDS_FOR_LONG is when it stops (a prompt that is not
# whole keeps its rows and writes no tokens). tests/page_27b.sh's `long` stage holds the page's forward pass to it
case " $stages " in *" long "*)
  began=$SECONDS
  cp "$here/fixtures/long-27b.txt" "$work/long.txt"
  rows=${LONG_ROWS:-20,40,60,80,100,127,200,400,700,1000,1400,1800,2200,2600,3000,3400,3800,4000,4080,4090,4094,4095,4100,4101,4104,4300,4500,4700,4900,5100,5300,5500,5700,5900,5970,5980,5981,5982,5983,5984,5985}
  # (the lines come as the fork writes them, so that a run the job's limit ends has its progress in the log)
  set +e
  env LONG_TEXT="$work/long.txt" LONG_ROWS="$rows" LONG_CTX="${LONG_CTX:-8192}" ${LONG_TOKENS:+LONG_TOKENS=$LONG_TOKENS} \
    "$work/reference_27b_fork" "$work/$file" "$work" "$threads" "${SECONDS_FOR_LONG:-18000}" 16 unused 2>&1 \
    | tee "$work/fork-long.log" | grep -a --line-buffered -E '^fork: (CPU|loaded|keys|long|out of time)|eval time|load time|total time'
  code=${PIPESTATUS[0]}
  set -e
  [ "$code" = 0 ] || { tail -40 "$work/fork-long.log"; echo "fork: the long run failed ($code)"; exit 1; }
  echo "fork: the long text ran in $((SECONDS - began)) s"
  ls -la "$work"/fork-long.* | sed 's/^/fork: /'
;; esac

status=0
case " $stages " in *" f32 "*)
  began=$SECONDS
  mkdir -p "$work/f32"
  # the very tokens of the first run, one at a time (and with BATCH_F32=1 as one batch too), float32 keys and values and
  # float32 activations for the ternary matrices; it widens every row it multiplies by, so it is slower than the first
  env PTQ1_0_F32_ACTIVATIONS=1 KV_F32=1 REPLAY_FROM="$work" ${BATCH_F32:+REPLAY_BATCH=1} \
    "$work/reference_27b_fork" "$work/$file" "$work/f32" "$threads" "${SECONDS_FOR_F32:-3000}" 16 "${prompts[@]}" \
    > "$work/f32.log" 2>&1 || { tail -60 "$work/f32.log"; echo "f32: the run failed"; status=1; }
  grep -a -E '^fork:' "$work/f32.log" | sed 's/^fork: /f32: /' || true
  echo "f32: ran in $((SECONDS - began)) s"
  ls -la "$work"/f32/fork-* 2>/dev/null | sed 's/^/f32: /' || true
;; esac

case " $stages " in *" numpy "*)
  began=$SECONDS
  python -m pip install -q numpy
  python "$here/reference_27b.py" "$work/$file" "$work" || status=1
  echo "reference: ran in $((SECONDS - began)) s"
;; esac
exit $status
