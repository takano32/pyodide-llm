#!/usr/bin/env bash
# T359 review: the page's conversion speed, this tree against origin/main's, in one job, the order of the two swapped
# every round (A B, B A, A B, ...), for each dtype: tests/profile-convert.mjs on the same model directory.
#   bash tests/abba-convert.sh <model directory> [rounds] [dtypes...]
# The base tree is made on the runner's own clone (git worktree add .tmp/base FETCH_HEAD), with this tree's node_modules
# and kernels. PROFILE=0 times both without cProfile.
# T374.2.1: this tree's tool converts by the conduct of a conversion (as the worker's loop: the parts to the feed the
# request of a stream brings), and a base of before T374.2.1 calls the converter itself: against such a base, the
# difference is what the conduct adds to a conversion's parts. Prints one "base|now <dtype> <MB/s>" line per run and the medians and ranges at the end.
set -u
dir=$1; rounds=${2:-8}; shift 2 || true; dtypes=${*:-int8 float32}
if [ ! -d .tmp/base ]; then
  git fetch -q --depth 1 origin main && git worktree add -q .tmp/base FETCH_HEAD
  cp -r node_modules .tmp/base/ && cp public/simdkernel* .tmp/base/public/
fi
# PROFILE=0 (the time without cProfile): the base tree's tool may not know it (run the profiled comparison first)
if [ "${PROFILE:-}" = 0 ]; then sed -i 's/began = time.perf_counter(); profiler.enable()")/profiler.enable(); profiler.disable(); began = time.perf_counter()")/' .tmp/base/tests/profile-convert.mjs; fi
mkdir -p .tmp/abba; : > .tmp/abba/lines
run() { # tree name dtype
  (cd "$1" && node tests/profile-convert.mjs "$OLDPWD/$dir" "$3" 2>&1 | sed -n 's/.*(\([0-9]*\) MB\/s).*/\1/p' | head -1 | sed "s|^|$2 $3 |") >> .tmp/abba/lines
}
for r in $(seq 1 "$rounds"); do
  for d in $dtypes; do
    if [ $((r % 2)) = 1 ]; then run . now "$d"; run .tmp/base base "$d"; else run .tmp/base base "$d"; run . now "$d"; fi
  done
done
cat .tmp/abba/lines
node -e '
const rows = require("fs").readFileSync(".tmp/abba/lines", "utf8").trim().split("\n").map((l) => l.split(" "));
for (const key of new Set(rows.map((r) => r[0] + " " + r[1]))) {
  const v = rows.filter((r) => r[0] + " " + r[1] === key).map((r) => +r[2]).sort((a, b) => a - b);
  console.log(key, "median", v[v.length >> 1], "min", v[0], "max", v[v.length - 1], "n", v.length);
}'
