#!/bin/bash
# tests/compare-builds.sh (T.367.5): what a change did to the forward pass as a visitor's browser runs it, with the
# measure of no change beside it every time. Three builds: another commit's (main by default), this tree's (dist/, which
# `npm run build` wrote), and this tree's built once more. Then tests/compare-engines.mjs through the bundles, `times`
# times over:
#   the other commit's build against this tree's          what is asked
#   this tree's second build against this tree's          the control: the same code on both sides
# A ratio of the first that is inside what the control says of no difference is no difference (T.367.3.2: two builds of
# one code were 1.4% apart on arm64 with 7 rounds; T.437: 15 rounds or more, and the control every time).
#
#   bash tests/compare-builds.sh [<commit> = origin/main] [<rounds> = 21] [<times> = 2] [model ids = tiny-lm llm-jp-3-150m]
#   THREADS=2,4 bash tests/compare-builds.sh …    T.378.1, T.447: after each of those, the same two pairs WITH software threads
#                                                 (tests/threads-compare.mjs: each build's forward and its own helper, that
#                                                 many threads). compare-engines runs no software thread: a change of
#                                                 forward/threads.js, helper.js or jobs.js is measured only here.
#
# CI's (the speed of a development machine decides nothing): tests.yml's extra=, on x86-64 and runner=ubuntu-24.04-arm,
# after the suite (the models, the kernels and dist/ are there then). Both sides read this tree's Python and kernels:
# what differs is forward.js as each build wrote it. Read the CPU's name with the numbers (compare-engines says it).
set -euo pipefail
cd "$(dirname "$0")/.."
ref="${1:-origin/main}"; rounds="${2:-21}"; times="${3:-2}"; shift $(( $# < 3 ? $# : 3 ))
[ -d dist/_astro ] || { echo "compare-builds: no dist/_astro here: npm run build"; exit 1; }
other=$(node -e "import('./tests/other-tree.mjs').then((m) => process.stdout.write(m.otherTree(process.argv[1]).folder))" "$ref")
# the other commit's build: its sources, this tree's packages and built files (kernels, archives: not what is measured;
# a commit with T.433's plugin makes its archives again of its own sources as its build starts)
[ -e "$other/node_modules" ] || ln -s "$PWD/node_modules" "$other/node_modules"
mkdir -p "$other/src/runtime/built" && cp src/runtime/built/* "$other/src/runtime/built/"
(cd "$other" && rm -rf dist && npx astro build > build.txt 2>&1) || { tail -20 "$other/build.txt"; echo "compare-builds: the build of $ref failed"; exit 1; }
again=.tmp/compare-builds/again
rm -rf "$again" && mkdir -p .tmp/compare-builds && npx astro build --outDir "$again" > .tmp/compare-builds/build.txt 2>&1 || { tail -20 .tmp/compare-builds/build.txt; echo "compare-builds: the second build of this tree failed"; exit 1; }
name() { basename "$(ls "$1"/_astro/forward.*.js)"; }
echo "compare-builds: forward as built: $ref $(name "$other/dist") ($(sha256sum "$other"/dist/_astro/forward.*.js | cut -c1-12)), this tree $(name dist) ($(sha256sum dist/_astro/forward.*.js | cut -c1-12)), this tree again $(name "$again") ($(sha256sum "$again"/_astro/forward.*.js | cut -c1-12))"
for time in $(seq 1 "$times"); do
  echo "compare-builds: $ref's build (old) against this tree's (new), $rounds rounds, time $time of $times"
  node tests/compare-engines.mjs "$@" --rounds "$rounds" --bundle-old "$other/dist" --bundle dist
  echo "compare-builds: the control, this tree built again (old) against this tree's build (new), $rounds rounds, time $time of $times"
  node tests/compare-engines.mjs "$@" --rounds "$rounds" --bundle-old "$again" --bundle dist
  [ -n "${THREADS:-}" ] || continue
  echo "compare-builds: with $THREADS software threads, $ref's build (old) against this tree's (new), $rounds rounds, time $time of $times"
  node tests/threads-compare.mjs "$@" --rounds "$rounds" --threads "$THREADS" --old "$other/dist" --new dist
  echo "compare-builds: with $THREADS software threads, the control, this tree built again (old) against this tree's build (new), $rounds rounds, time $time of $times"
  node tests/threads-compare.mjs "$@" --rounds "$rounds" --threads "$THREADS" --old "$again" --new dist
done
