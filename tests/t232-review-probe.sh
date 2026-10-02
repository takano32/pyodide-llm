#!/bin/bash
# T232's review (throwaway, not for main): what the made-up ternary models of dim 128 and the tests on lavapipe's
# defaults cannot see. The Dawn job of gpu-prompt.yml runs this in place of its gpu-check step (t232-review-probe's
# workflow). Each experiment prints the lines that say what happened; a failure of gpu-check is data here, not an error.
#   bash tests/t232-review-probe.sh [all | mutants]
set -u
export VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1)
DAWN=.tmp/dawn/node_modules/webgpu
OLD="synthetic-ternary synthetic-ternary-calm synthetic-ternary-wide"
only=${1:-all}
echo "cpu: $(node -p 'require("os").cpus()[0].model'), icd $VK_ICD_FILENAMES, commit $(git rev-parse --short HEAD), $only"

# what a run says: the first two of the engine's own choices ("gpu:" lines), each model's line, the runs' lines and the
# lines of every failure (the other "gpu:" lines of the engine are the same again and again: they filled the 70 lines
# of the first round)
summary() {
  { grep -m2 -E '^\[log\] gpu: (the matrices|a token)' "$1"
    grep -v -E '^\[log\] gpu: (the matrices|a token)' "$1" | grep -E '^suite|^synthetic[a-z0-9-]* \(|wrong:|did not take|unusable|FAILED|^ *- |NOT |on the GPU alone|a GPU that|a token by|a token:|^ok|Error|error'
  } | cut -c1-430 | head -${2:-90}
}
experiment() {  # a name, then the command; its log in .tmp/probe-<name>.log
  local name=$1; shift
  echo; echo "=================== $name"
  "$@" > ".tmp/probe-$name.log" 2>&1
  local code=$?
  summary ".tmp/probe-$name.log" 70
  echo "=== $name: exit $code"
  return $code
}
check() { node tests/gpu-check.mjs "$@" --nan none --engine dawn --webgpu $DAWN; }

# 0. the new model on the branch as it is
[ "$only" = other ] || experiment untied check synthetic-ternary-untied

if [ "$only" = all ] || [ "$only" = other ]; then
  # 1. subgroup size: what lavapipe reports under LP_NATIVE_VECTOR_WIDTH, and the ternary and int8 models at 16
  for width in 128 256 512; do
    LP_NATIVE_VECTOR_WIDTH=$width node tests/t232-subgroup.mjs $DAWN
  done
  LP_NATIVE_VECTOR_WIDTH=512 experiment width16 check synthetic-ternary synthetic-ternary-untied synthetic

  # 2. float32 to float16 rounding the other ways (Direct3D rounds toward zero): the engine's own checks ("gpu:" lines)
  for how in toward-zero away everything; do
    GPU_ROUNDING=$how experiment "round-$how" check synthetic-ternary synthetic-ternary-untied synthetic
  done

  # 3. a browser without the packed int8 dot (Safari, Firefox: unverified) and the int8 model beside it
  GPU_HIDE=packed_4x8_integer_dot_product experiment nopacked check synthetic-ternary synthetic-ternary-untied synthetic
fi

[ "$only" = other ] && exit 0

# 4. mutants the dim 128 models could not see: the old three must pass them (a hole), the new one must not
git checkout -q -- public
for mutant in $(node tests/t232-review-mutants.mjs); do
  git checkout -q -- public
  node tests/t232-review-mutants.mjs "$mutant" > /dev/null
  experiment "old-$mutant" check $OLD > /dev/null 2>&1
  old=$?
  experiment "new-$mutant" check synthetic-ternary-untied > /dev/null 2>&1
  new=$?
  echo "### MUTANT $mutant: the three of dim 128 exit $old ($([ $old = 0 ] && echo PASSED || echo failed)), the untied one exit $new ($([ $new = 0 ] && echo PASSED || echo failed))"
  grep -E 'wrong:|did not take|FAILED|^- [A-I]:.*of its line' .tmp/probe-new-$mutant.log | cut -c1-300 | head -6 | sed 's/^/    new: /'
  grep -E 'wrong:|did not take|FAILED' .tmp/probe-old-$mutant.log | cut -c1-300 | head -4 | sed 's/^/    old: /'
done
git checkout -q -- public
exit 0
