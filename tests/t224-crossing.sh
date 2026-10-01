#!/bin/bash
# T224's review (throwaway): tests.yml's extra=. The engine's generation on lavapipe from positions that put a run, or
# the run after it, across the place where flash_attn_vec takes more parts a head (64 -> 65, 128 -> 129, 256 -> 257):
# gpu-check's synthetic model with a prompt of T224_COUNT tokens; the rows of the generation by each attention.
set -u
sudo apt-get update -qq > /dev/null && sudo apt-get install -y -qq mesa-vulkan-drivers > /dev/null
mkdir -p .tmp/dawn && (cd .tmp/dawn && npm init -y > /dev/null && npm install webgpu@0.6.1 > /dev/null 2>&1)
export VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1)
echo "cpu: $(node -p 'require("os").cpus()[0].model'), icd $VK_ICD_FILENAMES, commit $(git rev-parse --short HEAD)"
status=0
for width in 512 128; do
  for count in 61 125 127 253; do
    repeat=3; [ "$count" -gt 150 ] && repeat=6
    echo "=== gpu-check synthetic, LP_NATIVE_VECTOR_WIDTH=$width, a prompt of $count tokens (text x$repeat)"
    LP_NATIVE_VECTOR_WIDTH=$width T224_COUNT=$count T224_REPEAT=$repeat timeout 1500 node tests/gpu-check.mjs synthetic --engine dawn --webgpu .tmp/dawn/node_modules/webgpu > .tmp/gpu-check-crossing.log 2>&1
    check=$?
    grep -E "^synthetic|a token by.*its attention by|FAILED|^- [A-C]:" .tmp/gpu-check-crossing.log | cut -c1-330 | head -40
    echo "=== exit $check"
    [ "$check" = 0 ] || status=1
  done
done
exit $status
