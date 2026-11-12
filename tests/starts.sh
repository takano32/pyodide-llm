#!/usr/bin/env bash
# tests/starts.sh (T399): the three readings of the token in front of a chat entry (AGENTS.md: "頭のトークンの効きは、素の文・
# 手書きの答え・モデル自身の答えの 3 つで測る"), one model after another on one runner. For CI (tests.yml's extra=, after
# `make kernels`; torch and transformers are needed: the development machine loads no model of gigabytes):
#
#   node tests/ci.ts run tests.yml minutes=360 extra="bash tests/starts.sh '<ids>;<chat_nll's arguments>;<starts>' …" --ref <branch>
#
# Each argument is one model: <ids> the list's entries that share its conversion, joined by + (a Qwen3's at-once entry
# and its thinking one: each is read by tests/answer_check.ts with its own format); <chat_nll's arguments> what
# tests/chat_nll.py is given (the original as owner/repository@revision, the BOS and the end of a turn, its flags), or
# empty for none; <starts> tests/start_check.ts's --starts on Japanese plain text (none,151643), or empty for none.
# The steps go one after another in their own processes (T283: transformers in float32 and the page's engine in one
# process lost a runner), and what each fetched is gone before the next. ANSWER_TOKENS: answer_check's length (60).
# A step that fails says so and the rest go on; the exit status is the number that failed.
set -uo pipefail
cd "$(dirname "$0")/.."
room=".tmp/starts"
if [ -d /mnt ] && [ -n "${GITHUB_ACTIONS:-}" ]; then
  sudo mkdir -p /mnt/starts && sudo chown "$USER" /mnt/starts
  room=/mnt/starts
fi
mkdir -p "$room"
export HF_HOME="$room/hf"
failed=0
step() {  # a step's name, then its command: run, and count it if it fails
  local name=$1
  shift
  "$@" || { failed=$((failed + 1)); echo "starts.sh $name: FAILED"; }
}
for model in "$@"; do
  IFS=';' read -r ids original starts <<< "$model"
  first=${ids%%+*}
  began=$SECONDS
  if [ -n "$original" ]; then
    # (the arguments are words: the flags hold no space, --then's text writes one as \x20 if it must)
    read -ra given <<< "$original"
    step "$first chat_nll" python3 tests/chat_nll.py "${given[@]}"
    rm -rf "$HF_HOME"
  fi
  IFS='+' read -ra entries <<< "$ids"
  for id in "${entries[@]}"; do
    python3 tests/write_options.py "$id" "$room/small" > "$room/$id.page.json" || { failed=$((failed + 1)); echo "starts.sh $id: FAILED"; rm -f "$room/$id.page.json"; }
  done
  # the dtype the page keeps the model in, as write_options.py says it (a Bonsai's binary or ternary, not int8: the
  # engine refuses a checkpoint that is not what its options say)
  dtype=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["options"].get("dtype", "int8"))' "$room/$first.page.json") &&
    fetched=$(python3 tests/hf_fetch.py "$first" "$room/downloads" | tail -1) &&
    python3 tests/perplexity_prepare.py "$fetched" "$room/$first" "$dtype" | cut -c1-160
  if [ $? -ne 0 ]; then
    failed=$((failed + 1))
    echo "starts.sh $first: FAILED to fetch or convert"
  else
    rm -rf "$room/downloads"
    if [ -n "$starts" ]; then
      echo "start_check $first: Japanese"
      step "$first start_check" node tests/start_check.ts "$room/$first" "$room/$first.page.json" --starts "$starts" --tokens 1500
    fi
    for id in "${entries[@]}"; do
      [ -f "$room/$id.page.json" ] && step "$id answer_check" node tests/answer_check.ts "$room/$first" "$room/$id.page.json" "${ANSWER_TOKENS:-60}"
    done
  fi
  echo "starts.sh $ids: $((SECONDS - began)) s"
  rm -rf "$room/downloads" "$room/$first".* "$room"/*.page.json "$HF_HOME"
done
echo "starts.sh: $# models, $failed steps failed"
exit "$failed"
