#!/usr/bin/env bash
# tests/gemma4_in_browser.sh (T.329.3): a real Gemma 4 (E2B, 10.2 GB of safetensors) opened by ?hf= in a real browser,
# from the site of this checkout, built and served on the runner: for a branch, which no deployed site has. On a CI
# runner only:
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=120 extra="bash tests/gemma4_in_browser.sh chromium firefox" \
#     --grep "^(gemma4|ready|opened|again|ok|FAILED|-|chromium|firefox|webkit)" --minutes 130 --ref <branch>
#   node tests/ci.mjs run tests.yml only_extra=true runner=macos-latest minutes=120 extra="SITE=https://takano32.github.io/pyodide-llm/ bash tests/gemma4_in_browser.sh webkit" ...
#
# First the small made-up Gemma 4 (tests/gemma4-in-browser.mjs: the table of every layer's input read from the kept
# file, a second visit, a second tab), so that a broken path fails before the large download. Then tests/e2e.mjs with
# a profile on disk (E2E_TWICE: the conversion kept in the origin private file system as it comes, and the second
# visit loading it and reading the table from it), and what the page said and measured.
#   $@          the browsers (chromium)
#   MODEL       the model, as tests/e2e.mjs takes it (Google's E2B at a fixed revision)
#   SMALL       none: not the made-up model first (it runs in Chromium alone: the tool answers huggingface.co by
#               Playwright's route, which a service worker's requests go past in the other browsers)
#   SITE        a site to open instead of building this checkout's (the deployed one, where the product is the same:
#               a macOS runner's make is too old for the Makefile)
#   E2E_TIMEOUT as tests/e2e.mjs takes it (3600 s)
set -euo pipefail
cd "$(dirname "$0")/.."
model=${MODEL:-hf:google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7}
[ $# -gt 0 ] || set -- chromium
echo "gemma4: $model in $* on $( (lscpu 2>/dev/null | sed -n 's/^Model name: *//p'; sysctl -n machdep.cpu.brand_string 2>/dev/null) | head -1) ($(uname -m)), $(getconf _NPROCESSORS_ONLN) logical cores"
began=$SECONDS
python -c "import numpy" 2>/dev/null || python -m pip install -q numpy
[ -d node_modules/pyodide ] || npm ci > /dev/null
if [ -z "${SITE-}" ]; then
  make kernels > /dev/null
  npm run build > /dev/null
fi
if [ "$(uname)" = Linux ]; then
  bash tests/apt-timeouts.sh
  timeout 1200 npx playwright-core install --with-deps "$@" > /dev/null
else
  npx playwright-core install "$@" > /dev/null
fi
echo "gemma4: the site built and $* installed in $((SECONDS - began)) s"
out=.tmp/keep/gemma4-browser
mkdir -p "$out"
status=0
for browser in "$@"; do
  if [ "${SMALL-}" != none ] && [ -z "${SITE-}" ] && [ "$browser" = chromium ]; then
    PYTHON=python node tests/gemma4-in-browser.mjs "$browser" 2>&1 | sed 's/^/gemma4: small: /' || status=1
  fi
  log="$out/$browser.txt"
  E2E_TWICE=1 E2E_TIMEOUT=${E2E_TIMEOUT:-3600} E2E_RESULTS="$out/results-$browser.jsonl" E2E_ARTIFACTS="$out/artifacts-$browser" \
    E2E_CONSOLE="$out/console-$browser.txt" node tests/e2e.mjs "$model" "$browser" ${SITE:+"$SITE"} 2>&1 | tee "$log" || status=1
  node -e '
    const fs = require("fs");
    const file = process.argv[1], browser = process.argv[2];
    const lines = fs.existsSync(file) ? fs.readFileSync(file, "utf8").trim().split("\n").filter(Boolean).map((line) => JSON.parse(line)) : [];
    for (const run of lines) {
      console.log(`gemma4: ${browser}: status line: ${run.backend}`);
      console.log(`gemma4: ${browser}: ready in ${run.readySeconds} s (${JSON.stringify(run.load)}), ${run.tokPerSecond} tok/s, heap ${run.heapMB} MB, isolated ${run.isolated}, not kept: ${run.notKept}; again: ${JSON.stringify(run.again)}`);
    }
    if (!lines.length) console.log(`gemma4: ${browser}: no result recorded`);
  ' "$out/results-$browser.jsonl" "$browser"
  grep -E "per-layer|table|rows|kept:|memory:|threads:|gpu:" "$out/console-$browser.txt" 2>/dev/null | cut -c1-300 | sed "s/^/gemma4: $browser: console: /" | head -30 || true
done
exit $status
