#!/usr/bin/env bash
# tests/repository_in_browser.sh (T505.3): a repository of GGUFs alone opened by ?hf= in real browsers, from the site of
# this checkout, built and served on the runner: for a branch, which no deployed site has, and for the browsers the
# development machine does not run (Firefox, WebKit). What it sees that nothing else does: whether huggingface.co lets
# a page of each browser read the list of a repository's files (its API, under the service worker), and that the page
# then takes the best GGUF there, converts it, answers, and loads it from what the browser kept on the second visit.
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=40 extra="bash tests/repository_in_browser.sh firefox webkit" \
#     --grep "^(repository|ready|opened|status|again|ok|FAILED|-|chromium|firefox|webkit)" --minutes 50 --ref <branch>
#
#   $@          the browsers (chromium firefox webkit)
#   REPOSITORY  the repository (a small one whose GGUFs open alone: LiquidAI/LFM2.5-230M-GGUF, 247 MB as Q8_0)
#   TAKES       what the page must say it opened it by (its Q8_0)
set -euo pipefail
cd "$(dirname "$0")/.."
repository=${REPOSITORY:-LiquidAI/LFM2.5-230M-GGUF}
takes=${TAKES:-LFM2.5-230M-Q8_0.gguf}
[ $# -gt 0 ] || set -- chromium firefox webkit
began=$SECONDS
[ -d node_modules/pyodide ] || npm ci > /dev/null
make kernels > /dev/null
npm run build > /dev/null
bash tests/apt-timeouts.sh
timeout 1200 npx playwright-core install --with-deps "$@" > /dev/null
echo "repository: the site built and $* installed in $((SECONDS - began)) s"
out=.tmp/keep/repository-browser
mkdir -p "$out"
status=0
for browser in "$@"; do
  log="$out/$browser.txt"
  E2E_TWICE=1 E2E_TIMEOUT=${E2E_TIMEOUT:-900} E2E_RESULTS="$out/results.jsonl" E2E_ARTIFACTS="$out/artifacts" E2E_CONSOLE="$out/console-$browser.txt" \
    node tests/e2e.mjs "hf:$repository" "$browser" 2>&1 | tee "$log" || status=1
  if ! grep -q "^opened: .*$takes" "$log"; then
    echo "repository: $browser did not open $repository by $takes — FAILED"
    status=1
  fi
  if ! grep -q "^again: .*kept in" "$log"; then
    echo "repository: the second visit of $browser did not come from what it kept — FAILED"
    status=1
  fi
done
exit $status
