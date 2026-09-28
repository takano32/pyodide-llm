#!/bin/bash
# The review of T219 (2) on CI (throwaway, branch t219-review-probe): the sampler's check with the review's cases on
# the branch and on the variants (lavapipe widths 4 and 16), then the variants timed in turn with main's.
sudo apt-get update -qq >/dev/null && sudo apt-get install -y -qq mesa-vulkan-drivers >/dev/null
mkdir -p .tmp/dawn && (cd .tmp/dawn && npm init -y >/dev/null && npm install webgpu@0.6.1 >/dev/null 2>&1)
export VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1)
echo "cpu: $(node -p 'require("os").cpus()[0].model')"
node tests/t219-review-probe.mjs --write .tmp/variants
cp public/shaders.js .tmp/variants/kept.js
for v in branch noAny noReturn signedMag noMag; do
  cp .tmp/variants/$v.js public/shaders.js
  for w in 128 512; do
    echo "== $v, lavapipe width $w: the sampling's verdicts with the review's cases"
    LP_NATIVE_VECTOR_WIDTH=$w timeout 900 node tests/bench-dawn.mjs .tmp/dawn/node_modules/webgpu public check 16 "sampling,sampling in chunks"
  done
done
cp .tmp/variants/kept.js public/shaders.js
echo "== timing (default width)"
timeout 1200 node tests/t219-review-probe.mjs .tmp/dawn/node_modules/webgpu --rounds 9
echo "== timing (width 4)"
LP_NATIVE_VECTOR_WIDTH=128 timeout 1200 node tests/t219-review-probe.mjs .tmp/dawn/node_modules/webgpu --rounds 9
echo "== done"
