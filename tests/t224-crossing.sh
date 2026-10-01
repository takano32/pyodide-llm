#!/bin/bash
# T224's review (throwaway): tests.yml's extra=. The engine's generation on lavapipe (subgroups of 16) from positions that
# put a run, or the run after it, across the places where flash_attn_vec takes more parts a head: 256 -> 257 (4 -> 8)
# and 512 -> 513 (8 -> 16), gpu-check's synthetic model with a longer context (seq_len 1024) and a prompt of T224_COUNT.
set -u
sudo apt-get update -qq > /dev/null && sudo apt-get install -y -qq mesa-vulkan-drivers > /dev/null
mkdir -p .tmp/dawn && (cd .tmp/dawn && npm init -y > /dev/null && npm install webgpu@0.6.1 > /dev/null 2>&1)
export VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1)
echo "cpu: $(node -p 'require("os").cpus()[0].model'), icd $VK_ICD_FILENAMES, commit $(git rev-parse --short HEAD)"
sed -i 's/seq_len=256/seq_len=1024/' tests/gpu-check.mjs
status=0
for spec in "253 8" "509 16" "257 8"; do
  set -- $spec
  count=$1; repeat=$2
  echo "=== gpu-check synthetic, LP_NATIVE_VECTOR_WIDTH=512, a prompt of $count tokens (text x$repeat), seq_len 1024"
  LP_NATIVE_VECTOR_WIDTH=512 T224_COUNT=$count T224_REPEAT=$repeat timeout 2400 node tests/gpu-check.mjs synthetic --engine dawn --webgpu .tmp/dawn/node_modules/webgpu > .tmp/gpu-check-crossing.log 2>&1
  check=$?
  grep -E "^synthetic|a token by.*its attention by|FAILED|^- [A-C]:" .tmp/gpu-check-crossing.log | cut -c1-330 | head -40
  echo "--- the tail of the log"
  tail -5 .tmp/gpu-check-crossing.log | cut -c1-300
  echo "=== exit $check"
  [ "$check" = 0 ] || status=1
done
exit $status
