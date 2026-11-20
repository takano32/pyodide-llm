#!/usr/bin/env bash
# T519 review: the native llama.cpp b12211 (the release wllama 3.13.0 is built from) on the same GGUF files, the CPU
# alone, as an anchor for the browser comparison (llama-bench: prompt processing "pp" and generation "tg", the
# same lengths as the browser run). Everything goes under .tmp/native/.
set -euo pipefail
cd "$(dirname "$0")/.."
work=.tmp/native
mkdir -p "$work"
case "$(uname -m)" in
  x86_64) asset=llama-b12211-bin-ubuntu-x64.tar.gz ;;
  aarch64) asset=llama-b12211-bin-ubuntu-arm64.tar.gz ;;
  *) echo "no asset for $(uname -m)"; exit 1 ;;
esac
[ -f "$work/$asset" ] || curl -fsSL --retry 3 -o "$work/$asset" "https://github.com/ggml-org/llama.cpp/releases/download/b12211/$asset"
mkdir -p "$work/bin"
tar xzf "$work/$asset" -C "$work/bin"
bench=$(find "$work/bin" -name llama-bench -type f | head -1)
echo "bench: $bench"; ldd "$bench" | head -3 || true
export LD_LIBRARY_PATH="$(dirname "$bench")"
grep -m1 "model name" /proc/cpuinfo || true
grep -o -m1 "avx[0-9a-z_]*\|neon\|asimd[a-z]*" /proc/cpuinfo | sort -u | tr '\n' ' ' || true; echo
get() { [ -f "$work/$2" ] || curl -fsSL --retry 3 -o "$work/$2" "$1"; }
get "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/smollm2-360m-instruct-q8_0.gguf" smollm2.gguf
get "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q8_0.gguf" qwen3.gguf
get "https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF/resolve/232fbd7916519edecddfdae0beae7e76fe70453f/plamo-3-translate-2b-Q4_K_M.gguf" plamo.gguf
run() { # model prompt threads
  echo "== $1 pp$2 tg128 threads $3"
  "$bench" -m "$work/$1.gguf" -p "$2" -n 128 -t "$3" -ngl 0 -r 3 -o md
}
for t in 2 1; do
  run smollm2 46 $t
  run qwen3 30 $t
done
run plamo 96 2
echo "== SmolLM2 with the wasm build's own limits: no AVX (GGML_CPU is chosen at run time; the backend files in the folder)"
ls "$(dirname "$bench")" | head -40
