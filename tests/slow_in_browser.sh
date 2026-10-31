#!/usr/bin/env bash
# tests/slow_in_browser.sh (T.538): the page on a slow line and on one that stops (tests/slow-check.ts), from the site of
# this checkout built on the runner, and from that of another commit (BEFORE) on the same runner by turns: a branch's
# check before and after, which slow.yml (the deployed site) cannot give. After tests.yml has made the models, the
# kernels and the build (not only_extra):
#
#   node tests/ci.ts run tests.yml minutes=120 extra="bash tests/slow_in_browser.sh" \
#     --grep "^(slow|stall|the |through|console|ok|FAILED|---)" --minutes 130 --ref <branch>
#
#   CASES   scenario:rate:queue:model, spaced (default: 0.15 Mbps as one deep queue, which fell back before T.538 (the
#           review of T.129), and as the oldest connection first; 0.4 Mbps shared by turns; and a stop)
#   BEFORE  the commit whose site goes first in each case (default origin/main; none: this checkout's alone)
set -euo pipefail
cd "$(dirname "$0")/.."
cases=${CASES:-"slow:18750:fifo:stories260K slow:18750:lane:stories260K slow:50000:fair:stories260K stall:50000:lane:stories260K"}
before=${BEFORE-origin/main}
began=$SECONDS
[ -d dist ] || npm run build > /dev/null
sites=("after=dist")
if [ -n "$before" ]; then
  tree=.tmp/slow-before
  git fetch --quiet origin
  rm -rf "$tree"
  git worktree add --quiet --detach "$tree" "$before"
  ln -s "$PWD/public/models" "$tree/public/models"
  (cd "$tree" && npm ci > /dev/null && make kernels > /dev/null && npm run build > /dev/null)
  sites=("before=$tree/dist" "after=dist")
fi
bash tests/apt-timeouts.sh
timeout 1200 npx playwright-core install --with-deps chromium > /dev/null
echo "slow: the sites built and Chromium installed in $((SECONDS - began)) s"
status=0
for one in $cases; do
  IFS=: read -r scenario rate queue model <<< "$one"
  for site in "${sites[@]}"; do
    echo "--- ${site%%=*}: $scenario at $rate bytes a second, $queue, $model"
    # (the exit of the site after alone: the one before is what is compared with)
    if ! timeout 1500 node tests/slow-check.ts "$scenario" --dist "${site#*=}" --rate "$rate" --queue "$queue" --model "$model"; then
      [ "${site%%=*}" = before ] || status=1
    fi
  done
done
exit $status
