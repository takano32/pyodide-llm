#!/usr/bin/env bash
# tests/small_models.sh (T.511.5): the small models a phone can hold, side by side by one measure, for the choice of the
# page's default model (T.511.6). For each model of the list named:
#   1. the nats per character of the same Japanese text (tests/start_check.mjs, the page's engine, int8, from the page's
#      BOS, windows of 512): the beginning of Japanese Wikipedia's 富士山, CHARACTERS characters of it (3000), the same
#      characters for every model, so that models of other vocabularies are held to the same text (a perplexity is not:
#      a token is another length in each);
#   2. the dozen questions of tests/answers.mjs, greedy and as the entry samples (SEEDS, "1 2"), TOKENS tokens (400).
# A model of this site (tiny-lm, llm-jp-3-150m) is read from the checkpoint `make models` made; a Hugging Face model is
# fetched at the revision the list pins and converted as the page converts it (tests/write.sh's way). For CI:
#
#   node tests/ci.mjs run tests.yml only_extra=true extra="npm ci && make kernels python && bash tests/small_models.sh hf-lfm2.5-230m" --grep "small_models|start_check|answers" --ref <branch>
#
# The text is fetched when this runs (nothing of Wikipedia is kept in this repository); its length and sha256 are
# printed, so that two runs can be seen to have read the same text. Exit: the number of models that failed.
set -uo pipefail
cd "$(dirname "$0")/.."
room=".tmp/small-models"
if [ -d /mnt ] && [ -n "${GITHUB_ACTIONS:-}" ]; then
  sudo mkdir -p /mnt/small-models && sudo chown "$USER" /mnt/small-models
  room=/mnt/small-models
fi
mkdir -p "$room"
text="$room/fuji.txt"
node tests/wikipedia.mjs ja "$room/fuji-whole.txt" 富士山
node -e 'const fs = require("fs"); const [from, to, n] = process.argv.slice(1); fs.writeFileSync(to, fs.readFileSync(from, "utf8").slice(0, Number(n)));' \
  "$room/fuji-whole.txt" "$text" "${CHARACTERS:-3000}"
echo "small_models: the text is $(node -e 'console.log([...require("fs").readFileSync(process.argv[1], "utf8")].length)' "$text") characters, sha256 $(sha256sum "$text" | cut -c1-16)"
failed=0
for id in "$@"; do
  began=$SECONDS
  (
    set -e
    if [ -f "$id.bin" ]; then
      # a model of this site: the checkpoint `make models` made, and the page's JSON from the list (answers.mjs's input)
      out="$room/$id"
      ln -sf "$PWD/$id.bin" "$out.bin"
      ln -sf "$PWD/$id.tokenizer.bin" "$out.tokenizer.bin"
      node -e '
        const fs = require("fs");
        import("./src/models.ts").then(({ MODELS }) => {
          const [id, file] = process.argv.slice(1), entry = MODELS.find((m) => m.id === id), head = Buffer.alloc(28), fd = fs.openSync(`${id}.bin`, "r");
          fs.readSync(fd, head, 0, 28, 0);
          const header = Array.from({ length: 7 }, (_, i) => head.readInt32LE(4 * i));
          fs.writeFileSync(`${file}.page.json`, JSON.stringify({ id, name: entry.name, options: entry.options, template: entry.template ?? null,
            prompt: entry.prompt, generation: entry.generation, header }));
          fs.writeFileSync(`${file}.json`, JSON.stringify(entry.options));  // (what start_check.mjs takes an <out> by)
        });' "$id" "$out"
    else
      model=$(python3 tests/hf_fetch.py "$id" "$room/downloads" | tail -1)
      python3 tests/perplexity_prepare.py "$model" "$room/$id" int8 | cut -c1-160
      python3 tests/write_options.py "$id" "$room/small" > "$room/$id.page.json"
      rm -rf "$room/downloads"
    fi
    prepared=$SECONDS
    node tests/start_check.mjs "$room/$id" "$room/$id.page.json" --starts bos --tokens 100000 --window 512 --text "$text"
    measured=$SECONDS
    GREEDY=1 SEEDS="${SEEDS:-1 2}" node tests/answers.mjs "$room/$id" "$room/$id.page.json" "${TOKENS:-400}"
    echo "small_models $id: preparing $((prepared - began)) s, the text $((measured - prepared)) s, the answers $((SECONDS - measured)) s"
  )
  if [ $? -ne 0 ]; then
    failed=$((failed + 1))
    echo "small_models $id: FAILED"
  fi
  rm -rf "$room/downloads" "$room/$id".*
done
echo "small_models: $# models, $failed failed"
exit "$failed"
