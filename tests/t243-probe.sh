#!/bin/bash
# T243 (a throwaway branch): the fix broken on purpose, each way through the tests that are to catch it, and the cost
# of the look at the keys and values on this runner.
#   node tests/ci.mjs run tests.yml extra="bash tests/t243-probe.sh" --ref t243-broken
set -u
mkdir -p .tmp/t243
echo "T243 probe: $(node -e 'const os = require("os"); console.log((os.cpus()[0].model || "unknown CPU") + ", " + os.arch() + ", " + os.cpus().length + " cores")')"

echo "T243 probe: the cost of the look"
node tests/t243-scan-bench.mjs 2>&1 | sed 's/^/T243 cost: /'

# name, a Python expression that takes forward.js's text s to the broken one
broken() {
  local name="$1" change="$2" file="public/forward-t243-broken.js"
  python3 - "$change" "$file" <<'EOF'
import sys
s = open("public/forward.js").read()
t = eval(sys.argv[1])
assert t != s, "the change changed nothing"
open(sys.argv[2], "w").write(t)
EOF
  [ $? -eq 0 ] || { echo "T243 mutation: $name: NOT MADE"; return; }
  node tests/gpu-default-check.mjs --forward "$file" > .tmp/t243/out.log 2>&1
  local code=$?
  local lines
  lines=$(grep -c '^- T243' .tmp/t243/out.log)
  local others
  others=$(grep '^- ' .tmp/t243/out.log | grep -vc '^- T243')
  if [ $code -ne 0 ] && [ "$lines" -gt 0 ]; then
    echo "T243 mutation: $name: caught ($lines of T243's checks failed, $others others)"
  else
    echo "T243 mutation: $name: NOT CAUGHT (exit $code)"
  fi
  grep '^- T243' .tmp/t243/out.log | cut -c1-330 | sed 's/^/T243 mutation:   /'
  rm -f "$file"
}

echo "T243 probe: forward.js as it is"
node tests/gpu-default-check.mjs > .tmp/t243/out.log 2>&1
echo "T243 mutation: none: exit $? ($(tail -1 .tmp/t243/out.log))"
grep '^T243' .tmp/t243/out.log | cut -c1-400 | sed 's/^/T243 said: /'

broken "the look removed (a block and the steps)" 's.replace("!direct && !stagingFinite(", "false && !stagingFinite(")'
broken "a block's look removed" 's.replace("!direct && !stagingFinite(count)", "false && !stagingFinite(count)")'
broken "the steps' look removed" 's.replace("!direct && !stagingFinite(sampled)", "false && !stagingFinite(sampled)")'
broken "only the keys looked at" 's.replace("for (let part = 0; part < 2 * layers; part++) {\n      if (!k.finite_f16", "for (let part = 0; part < layers; part++) {\n      if (!k.finite_f16")'
broken "only the values looked at" 's.replace("for (let part = 0; part < 2 * layers; part++) {\n      if (!k.finite_f16", "for (let part = layers; part < 2 * layers; part++) {\n      if (!k.finite_f16")'
broken "only the first position looked at" 's.replace("GPU_BLOCK * kvDim * 2, count * kvDim)) return false", "GPU_BLOCK * kvDim * 2, kvDim)) return false")'
broken "only the first layer looked at" 's.replace("if (!k.finite_f16(staging + part * GPU_BLOCK", "if (!k.finite_f16(staging + (part < layers ? 0 : layers) * GPU_BLOCK")'
broken "the GPU not stopped, the request only given back" 's.replace("stopGpu(notFiniteKV(", "void (notFiniteKV(")'

# the kernel: an infinity not counted (a NaN alone: the exponent's bits set and a mantissa), then only the eights
# looked at; the kernels built again each time, and put back
kernel() {
  local name="$1" change="$2"
  cp kernels/kernel.ts .tmp/t243/kernel.ts
  python3 - "$change" <<'EOF'
import sys
s = open("kernels/kernel.ts").read()
t = eval(sys.argv[1])
assert t != s, "the change changed nothing"
open("kernels/kernel.ts", "w").write(t)
EOF
  [ $? -eq 0 ] || { echo "T243 mutation: $name: NOT MADE"; return; }
  python3 kernels/build.py public > .tmp/t243/build.log 2>&1 || { echo "T243 mutation: $name: NOT BUILT"; tail -5 .tmp/t243/build.log; }
  node tests/gpu-default-check.mjs > .tmp/t243/out.log 2>&1
  local code=$?
  echo "T243 mutation: $name: gpu-default-check $([ $code -ne 0 ] && echo caught || echo 'NOT CAUGHT') ($(grep -c '^- T243' .tmp/t243/out.log) of T243's checks failed, $(grep '^- ' .tmp/t243/out.log | grep -vc '^- T243') others)"
  grep '^- T243' .tmp/t243/out.log | cut -c1-330 | sed 's/^/T243 mutation:   /'
  node tests/smoke.mjs > .tmp/t243/smoke.log 2>&1
  code=$?
  echo "T243 mutation: $name: smoke $([ $code -ne 0 ] && echo caught || echo 'NOT CAUGHT') ($(grep -m1 -o 'AssertionError.*' .tmp/t243/smoke.log | cut -c1-200))"
  cp .tmp/t243/kernel.ts kernels/kernel.ts
}
SIMD='i16x8.eq(v128.and(v128.load(x + (<usize>i << 1)), exponent), exponent)'
REST='(<i32>load<u16>(x + (<usize>i << 1)) & 0x7c00) == 0x7c00'
kernel "an infinity not counted" "s.replace('$SIMD', 'i16x8.gt_s(v128.and(v128.load(x + (<usize>i << 1)), i16x8.splat(0x7fff)), exponent)').replace('$REST', '(<i32>load<u16>(x + (<usize>i << 1)) & 0x7fff) > 0x7c00')"
kernel "a NaN not counted (the infinities alone)" "s.replace('$SIMD', 'i16x8.eq(v128.and(v128.load(x + (<usize>i << 1)), i16x8.splat(0x7fff)), exponent)').replace('$REST', '(<i32>load<u16>(x + (<usize>i << 1)) & 0x7fff) == 0x7c00')"
kernel "the rest after the eights not looked at" "s.replace('for (; i < n; i++) rest |= <i32>((<i32>load<u16>', 'for (; i < 0; i++) rest |= <i32>((<i32>load<u16>')"
python3 kernels/build.py public > .tmp/t243/build.log 2>&1
echo "T243 probe: done"
