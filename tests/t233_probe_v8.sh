#!/usr/bin/env bash
# T233 review probe: V8's Liftoff and TurboFan on a 64-bit memory above 4 GiB (what the canary of tests/ternary-check.mjs
# says, and what the real ternary kernel does from its first call), on this runner's CPU
cd "$(dirname "$0")/.."
echo "probe: $(node --version), V8 $(node -p process.versions.v8), $(uname -m), $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
for flags in "" "--liftoff-only" "--no-liftoff"; do
  for shared in "" shared; do
    node $flags tests/t233_probe_canary.mjs $shared 2>&1 | grep -E 'V8|WRONG|splat   ' | sed 's/^/probe: /'
  done
done
node tests/t233_probe_setflag.mjs 2>&1 | sed 's/^/probe: /'
echo "probe: the real matmul_t2r, its first call below 4 GiB then above (what the model does), 3072 rows"
node tests/t233_probe_kernel.mjs rows=3072 calls=60 2>&1 | sed 's/^/probe: /'
node tests/t233_probe_kernel.mjs shared rows=3072 calls=60 2>&1 | sed 's/^/probe: /'
echo "probe: its first call above 4 GiB (what a model that began there would do)"
node tests/t233_probe_kernel.mjs coldhigh rows=3072 calls=5 2>&1 | sed 's/^/probe: /'
echo "probe: calls of 64 rows (a short matrix: the budget of tier-up is slow to run out)"
node tests/t233_probe_kernel.mjs rows=64 calls=400 2>&1 | sed 's/^/probe: /'
echo "probe: with Liftoff only, and with TurboFan only"
node --liftoff-only tests/t233_probe_kernel.mjs rows=3072 calls=20 2>&1 | sed 's/^/probe: /'
node --no-liftoff tests/t233_probe_kernel.mjs coldhigh rows=3072 calls=20 2>&1 | sed 's/^/probe: /'
echo "probe: how often the first high call (the second call of a fresh process, after a cold call below 4 GiB of N rows) is right"
for rows in 256 1024 3072 10240; do
  right=0
  for i in $(seq 1 20); do
    if node tests/t233_probe_kernel.mjs rows=$rows calls=1 2>&1 | grep -q '^call 0 .*: right'; then right=$((right + 1)); fi
  done
  echo "probe: a cold call of $rows rows below 4 GiB, then one above: right in $right of 20 fresh processes"
done
