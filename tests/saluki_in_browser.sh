#!/usr/bin/env bash
# tests/saluki_in_browser.sh (T.364.3): the list's entry of Underdog Saluki 27B opened by a real browser from the site of
# this checkout, built and served on the runner: for a branch, which no deployed site has. On a CI runner only (it
# fetches 7.9 GB, and the page takes 8 GiB of memory and keeps 7.96 GB in the browser's profile):
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=100 extra="bash tests/saluki_in_browser.sh chromium" \
#     --grep "^(saluki|ready|generated|again|ok|FAILED|-|isolated|wide|plain|no-relaxed|widened|chromium|firefox)" --minutes 110 --ref <branch>
#
# Once the entry is deployed, models.yml does the same against the site (models=hf-underdog-saluki-27b twice=true
# timeout=4200), and browsers.yml's group saluki-27b every week.
#
# First the small model of the same kind (tests/blocks-in-browser.mjs, a minute: the bundled worker, the kernels of the
# blocks from _astro/, every kind of memory), so that a broken path fails before the large download. Then
# tests/e2e.mjs with a profile on disk (E2E_TWICE: the conversion kept in the origin private file system and loaded
# from it on the second visit), and what its status line must say of this model.
#   $1        the browser (chromium; firefox)
#   ENTRY     the entry (hf-underdog-saluki-27b, which answers at once; -thinking)
#   SMALL     the visits of the small model ("" for all; none)
#   E2E_TIMEOUT, E2E_QUERY  as tests/e2e.mjs takes them (4200 s; nothing)
set -euo pipefail
cd "$(dirname "$0")/.."
browser=${1:-chromium}
entry=${ENTRY:-hf-underdog-saluki-27b}
echo "saluki: $entry in $browser on $(lscpu | sed -n 's/^Model name: *//p' | head -1) ($(uname -m)), $(nproc) logical cores, $(free -g | awk '/Mem:/{print $2}') GB of memory, $(df -BG --output=avail . | tail -1 | tr -d ' ') of disk"
began=$SECONDS
python -c "import numpy, pytest, jinja2" 2>/dev/null || python -m pip install -q numpy pytest jinja2
[ -d node_modules/pyodide ] || npm ci > /dev/null
make kernels > /dev/null
npm run build > /dev/null
bash tests/apt-timeouts.sh
timeout 1200 npx playwright-core install --with-deps "$browser" > /dev/null
echo "saluki: the site built and $browser installed in $((SECONDS - began)) s"

if [ "${SMALL-}" != none ]; then
  # shellcheck disable=SC2086
  node tests/blocks-in-browser.mjs "$browser" ${SMALL-}
fi

out=.tmp/keep/saluki-browser
mkdir -p "$out"
status=0
E2E_TWICE=1 E2E_TIMEOUT=${E2E_TIMEOUT:-4200} E2E_RESULTS="$out/results.jsonl" E2E_ARTIFACTS="$out/artifacts" E2E_CONSOLE="$out/console.txt" \
  node tests/e2e.mjs "$entry" "$browser" || status=1
# what the page said and measured, as the run's own lines
node -e '
  const fs = require("fs");
  const lines = fs.existsSync(process.argv[1]) ? fs.readFileSync(process.argv[1], "utf8").trim().split("\n").filter(Boolean).map((line) => JSON.parse(line)) : [];
  let failed = lines.length === 0;
  for (const run of lines) {
    console.log(`saluki: status line: ${run.backend}`);
    console.log(`saluki: ready in ${run.readySeconds} s (${JSON.stringify(run.load)}), ${run.tokPerSecond} tok/s, heap ${run.heapMB} MB, isolated ${run.isolated}, not kept: ${run.notKept}; again: ${JSON.stringify(run.again)}`);
    // (no apostrophe in here, a dot in its place: this is inside the quotes of the shell)
    const wanted = /· SIMD kernels, llama\.cpp.s blocks(, relaxed SIMD)?, 64-bit memory(, \d+ threads)?(, [^·]+)? · Pyodide /;
    if (!wanted.test(run.backend ?? "")) { console.log("saluki: the status line does not say the blocks of llama.cpp on a 64-bit memory — FAILED"); failed = true; }
    if (!run.again?.fromCache) { console.log("saluki: the second visit did not come from what the browser kept — FAILED"); failed = true; }
  }
  process.exit(failed ? 1 : 0);
' "$out/results.jsonl" || status=1
grep -E "gpu:|threads:|kept:|memory:|kernels:" "$out/console.txt" | cut -c1-300 | sed 's/^/saluki: console: /' | head -40 || true
exit $status
