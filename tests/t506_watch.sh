#!/usr/bin/env bash
# tests/t506_watch.sh (T506, a tool of the search: taken away once the task is done): one model opened in a browser
# from the deployed site while the runner's memory is written down every two seconds, then what the kernel said of
# processes it ended.   bash tests/t506_watch.sh <model> <browser> [E2E_QUERY]
set -uo pipefail
cd "$(dirname "$0")/.."
model=$1 browser=$2 query=${3-}
name="$browser-$model-${query:-plain}"
out=.tmp/keep/t506/$name
mkdir -p "$out"
began=$SECONDS
( while true; do
    echo "mem: $name $((SECONDS - began))s used $(free -m | awk '/Mem:/{print $3" available "$7}') swap $(free -m | awk '/Swap:/{print $3}') MB; largest: $(ps -eo rss,comm --sort=-rss | sed -n 2,4p | awk '{printf "%d MB %s, ", $1/1024, $2}')"
    sleep 2
  done ) > "$out/memory.txt" &
watcher=$!
E2E_QUERY=$query E2E_ARTIFACTS=$out E2E_RESULTS=$out/results.jsonl E2E_TIMEOUT=${E2E_TIMEOUT:-240} \
  node tests/e2e.mjs "$model" "$browser" https://takano32.github.io/pyodide-llm/ 2>&1 | cut -c1-500 | sed "s/^/e2e: $name: /"
kill $watcher
# every fifth line, and all of the last twenty
awk 'NR % 5 == 1' "$out/memory.txt"
tail -20 "$out/memory.txt" | sed 's/^mem:/mem (end):/'
cat "$out"/*.log 2>/dev/null | tail -12 | cut -c1-300 | sed "s/^/kept: $name: /"
sudo dmesg 2>/dev/null | grep -i -E "out of memory|killed process|oom" | tail -5 | cut -c1-300 | sed "s/^/dmesg: $name: /"
exit 0
