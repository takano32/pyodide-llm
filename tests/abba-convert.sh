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
  # (the built kernels are where each tree keeps them: tests/tree.mjs)
  cp -r node_modules .tmp/base/ && mkdir -p "$(node tests/tree.mjs --root .tmp/base built)" &&
    cp "$(node tests/tree.mjs built)"/simdkernel* "$(node tests/tree.mjs --root .tmp/base built)"/
fi
# T374.2.1: every run gets a layout of its own (LAYOUT: see tests/profile-convert.mjs): where the buffers lie is the
# same in every run of one tool, and differs between two tools or trees. A base tree whose tool does not know LAYOUT
# (before T374.2.1) is given the two lines. The last lines say now against base with one standard error of the runs of
# this job; between jobs the same comparison moved by more than that (TODO.md's T374.2.1): one job is not a verdict.
if ! grep -q LAYOUT .tmp/base/tests/profile-convert.mjs; then
  sed -i -e 's|^const convert = py.pyimport("llama2_convert");$|py.runPython(`layout_before = bytes(${Number((process.env.LAYOUT ?? "0,0").split(",")[0])})`);\n&|' \
    -e 's|^py.globals.set("conversion", conversion);$|py.runPython(`layout_beside = bytes(${Number((process.env.LAYOUT ?? "0,0").split(",")[1])})`);\n&|' .tmp/base/tests/profile-convert.mjs
  grep -c layout_ .tmp/base/tests/profile-convert.mjs | grep -qx 2 || { echo "abba-convert: the base tree's tool could not be given a layout"; exit 1; }
fi
# PROFILE=0 (the time without cProfile): the base tree's tool may not know it (run the profiled comparison first)
if [ "${PROFILE:-}" = 0 ]; then sed -i 's/began = time.perf_counter(); profiler.enable()")/profiler.enable(); profiler.disable(); began = time.perf_counter()")/' .tmp/base/tests/profile-convert.mjs; fi
mkdir -p .tmp/abba; : > .tmp/abba/lines
run() { # tree name dtype
  (cd "$1" && LAYOUT=$(( (RANDOM * 32768 + RANDOM) % 2097152 + 1 )),$(( (RANDOM * 32768 + RANDOM) % 2097152 + 1 )) node tests/profile-convert.mjs "$OLDPWD/$dir" "$3" 2>&1 | sed -n 's/.*(\([0-9]*\) MB\/s).*/\1/p' | head -1 | sed "s|^|$2 $3 |") >> .tmp/abba/lines
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
}
// now against base, by the means, with one standard error of the difference within this job
for (const d of new Set(rows.map((r) => r[1]))) {
  const stat = (name) => { const v = rows.filter((r) => r[0] === name && r[1] === d).map((r) => +r[2]), m = v.reduce((a, b) => a + b, 0) / v.length; return [m, v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1) / v.length]; };
  const [bm, bv] = stat("base"), [nm, nv] = stat("now");
  console.log(d, "now against base:", ((nm / bm - 1) * 100).toFixed(2) + "%", "+-", (Math.sqrt(bv + nv) / bm * 100).toFixed(2) + "%");
}'
