#!/bin/bash
# T224's review (throwaway): tests.yml's extra=. Where flash_attn_vec's subgroup form is wrong at lavapipe's subgroup of 32.
set -u
sudo apt-get update -qq > /dev/null && sudo apt-get install -y -qq mesa-vulkan-drivers > /dev/null
mkdir -p .tmp/dawn && (cd .tmp/dawn && npm init -y > /dev/null && npm install webgpu@0.6.1 > /dev/null 2>&1)
export VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1)
echo "cpu: $(node -p 'require("os").cpus()[0].model'), icd $VK_ICD_FILENAMES, commit $(git rev-parse --short HEAD)"
status=0
for width in 128 256 512 1024; do
  echo "=== subgroup operations, LP_NATIVE_VECTOR_WIDTH=$width"
  LP_NATIVE_VECTOR_WIDTH=$width node tests/t224-subgroup-ops.mjs .tmp/dawn/node_modules/webgpu || status=1
done
echo "=== gpu-check synthetic, LP_NATIVE_VECTOR_WIDTH=1024 (the prompt's and the token's shaders with subgroups of 32)"
LP_NATIVE_VECTOR_WIDTH=1024 timeout 2400 node tests/gpu-check.mjs synthetic --engine dawn --webgpu .tmp/dawn/node_modules/webgpu > .tmp/gpu-check-1024.log 2>&1
check=$?
grep -E "gpu: the matrices|gpu: a token|its attention by|FAILED|^synthetic|^- [A-M]:|subgroup" .tmp/gpu-check-1024.log | cut -c1-420 | head -120
echo "=== gpu-check exit $check"
[ "$check" = 0 ] || status=1
exit $status
