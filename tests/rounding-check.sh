#!/usr/bin/env bash
# T225's review: tests/rounding-check.mjs where only the repository is (tests.yml's extra=, on a branch): installs Mesa's
# lavapipe and Dawn (the npm package webgpu, under .tmp/dawn, as gpu-prompt.yml's Dawn job does) first, then runs it:
#   node tests/ci.mjs run tests.yml extra="bash tests/rounding-check.sh toward-zero away everything nearest" --ref <branch>
set -euo pipefail
if [ ! -d .tmp/dawn/node_modules/webgpu ]; then
  sudo apt-get update -qq > /dev/null && sudo apt-get install -y -qq mesa-vulkan-drivers > /dev/null
  mkdir -p .tmp/dawn && (cd .tmp/dawn && npm init -y > /dev/null && npm install webgpu@0.6.1 > /dev/null 2>&1)
fi
export VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1)
node tests/rounding-check.mjs .tmp/dawn/node_modules/webgpu "$@"
