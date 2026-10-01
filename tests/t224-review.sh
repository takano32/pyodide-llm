#!/bin/bash
# T224's review (throwaway, not for main): tests.yml's extra=. The probe on Dawn + lavapipe at subgroup widths 4, 8, 16, 32.
set -u
sudo apt-get update -qq > /dev/null && sudo apt-get install -y -qq mesa-vulkan-drivers > /dev/null
mkdir -p .tmp/dawn && (cd .tmp/dawn && npm init -y > /dev/null && npm install webgpu@0.6.1 > /dev/null 2>&1)
export VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1)
echo "cpu: $(node -p 'require("os").cpus()[0].model'), icd $VK_ICD_FILENAMES, commit $(git rev-parse --short HEAD)"
status=0
for width in 128 512 256 1024; do
  echo "=== t224 probe, LP_NATIVE_VECTOR_WIDTH=$width"
  LP_NATIVE_VECTOR_WIDTH=$width timeout 1500 node tests/t224-review-probe.mjs .tmp/dawn/node_modules/webgpu || status=1
done
exit $status
