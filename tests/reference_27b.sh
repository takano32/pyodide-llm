#!/usr/bin/env bash
# tests/reference_27b.sh (T238): a reference for Ternary Bonsai 2 27B, on a CI runner (never on the development machine:
# it fetches 5.95 GB). Widened to float32 the model is 107 GB, so the usual NumPy reference cannot hold it. Two
# references, and the comparison of the two:
#   (1) Prism ML's fork of llama.cpp (MIT, a pinned commit), built here for the CPU, with tests/reference_27b_fork.cpp
#       against its libraries: the prompts' ids as the fork tokenizes them, 16 greedy tokens and the logits of every
#       position, and its tokens/s (the only speed on a CPU anyone has for this model);
#   (2) tests/reference_27b.py: the engine's own NumPy forward pass (public/llama2_numpy.py) over the GGUF read by
#       memory map, a matrix widened at a time.
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=150 extra="bash tests/reference_27b.sh" \
#     --grep "^(fork|reference|runner):" --minutes 170 --ref <branch>
#
# STAGES (default "fork numpy"): which of the two run; SECONDS_FOR_FORK: after how long the fork stops writing.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
fork_commit=88c4bc60b9c9578f134385be9535e853f2db9b9f
model_revision=b072e1d3b35a0a630cece372c2127528e0994386
file=Ternary-Bonsai-2-27B-PTQ1_0.gguf
bytes=5946648928
stages=${STAGES:-fork numpy}
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

# the prompts: two bare texts and a turn of a chat in the form that answers at once (no system turn)
prompts=("The capital of Japan is"
         "日本でいちばん高い山は"
         $'<|im_start|>user\nWhat is 17 times 24?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')

for index in "${!prompts[@]}"; do printf '%s' "${prompts[$index]}" > "$work/prompt-$index.txt"; done

case " $stages " in *" fork "*)
  began=$SECONDS
  if [ ! -d "$work/fork/.git" ]; then
    git init -q "$work/fork"
    git -C "$work/fork" remote add origin https://github.com/PrismML-Eng/llama.cpp
  fi
  git -C "$work/fork" fetch -q --depth 1 origin "$fork_commit"
  git -C "$work/fork" checkout -q FETCH_HEAD
  echo "fork: $(git -C "$work/fork" rev-parse HEAD), $(head -3 "$work/fork/LICENSE" | tr '\n' ' ')"
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
  began=$SECONDS
  "$work/reference_27b_fork" "$work/$file" "$work" "$threads" "${SECONDS_FOR_FORK:-2400}" 16 "${prompts[@]}" \
    > "$work/fork.log" 2>&1 || { tail -60 "$work/fork.log"; echo "fork: the run failed"; exit 1; }
  grep -a -E '^fork:|eval time|load time|total time' "$work/fork.log" | sed 's/^\([^f]\)/fork: \1/'
  echo "fork: ran in $((SECONDS - began)) s"
  ls -la "$work"/fork-* | sed 's/^/fork: /'
;; esac

case " $stages " in *" numpy "*)
  began=$SECONDS
  python -m pip install -q numpy
  python "$here/reference_27b.py" "$work/$file" "$work"
  echo "reference: ran in $((SECONDS - began)) s"
;; esac
