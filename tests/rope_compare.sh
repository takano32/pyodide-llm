#!/usr/bin/env bash
# tests/rope_compare.sh (T235): a model of the list whose entry says another RoPE scaling than its conversion does (the
# three Ternary Bonsai: rope_scaling: {}, a plain RoPE, over config.json's yarn), both ways on the page's forward pass
# (src/runtime/forward.js on the kernels, the dtype the page keeps the model in): the perplexity on Wikipedia with each,
# and what each writes for the entry's own prompt. One conversion, as the page makes it; the two differ in the options
# alone, as on the page (the engine makes the RoPE tables from them). For CI (tests.yml's extra=, after `make kernels`:
# the development machine runs no model of gigabytes):
#
#   node tests/ci.mjs run tests.yml extra="bash tests/rope_compare.sh hf-ternary-bonsai-1.7b" --ref <branch> --grep "^rope|written by|kernels, "
#   bash tests/rope_compare.sh [<id of the list> = hf-ternary-bonsai-1.7b] [tokens = 1500] [languages = "ja en"]
#
# The perplexity is tests/perplexity.mjs's first row (the page's own path) on the first <tokens> tokens of T85's
# articles (the beginning of the first: Mount Fuji, 富士山), a context of 512 begun anew from the entry's BOS. The
# writing is tests/write.mjs's: greedy, then as the page samples with seeds 1 and 2 (WRITE_TOKENS tokens, 48).
set -euo pipefail
cd "$(dirname "$0")/.."
id="${1:-hf-ternary-bonsai-1.7b}"
tokens="${2:-1500}"
languages="${3:-ja en}"
dir="${RUNNER_TEMP:-.tmp}/rope"
mkdir -p "$dir"
echo "rope: $id, $(node -p 'require("os").cpus()[0].model') ($(uname -m))"
dtype=$(node -e "import('./src/models.js').then(({ MODELS }) => console.log(MODELS.find((m) => m.id === process.argv[1]).weights ?? 'int8'))" "$id")
model=$(python3 tests/hf_fetch.py "$id" "$dir" | tail -1)
python3 tests/perplexity_prepare.py "$model" "$dir/made" "$dtype" | cut -c1-200
python3 tests/write_options.py "$id" "$dir/small" > "$dir/page.json"
# the two sets of options: the page's (the conversion's under the entry's) and the same with the conversion's own
# scaling put back, each as tests/perplexity.mjs reads it (<out>.json beside <out>.bin) and as tests/write.mjs does
python3 - "$dir" <<'PYTHON'
import json, os, sys
from pathlib import Path
room = Path(sys.argv[1])
made, page = json.loads((room / "made.json").read_text()), json.loads((room / "page.json").read_text())
theirs, ours = made.get("rope_scaling"), page["options"].get("rope_scaling")
if not theirs or ours == theirs:
    sys.exit(f"rope: the entry and its conversion say the same RoPE scaling ({theirs}): nothing to compare")
for name, scaling, said in (("entry", ours, "the entry's"), ("conversion", theirs, "the conversion's")):
    options = {**page["options"], "dtype": made["dtype"], "rope_scaling": scaling}
    (room / f"{name}.json").write_text(json.dumps(options))
    (room / f"{name}.page.json").write_text(json.dumps({**page, "id": f"{page['id']} with {said} RoPE", "options": options}, ensure_ascii=False))
    for end in ("bin", "tokenizer.bin"):
        link = room / f"{name}.{end}"
        link.unlink(missing_ok=True)
        os.symlink((room / f"made.{end}").resolve(), link)
    print(f"rope: {said} options say rope_scaling {json.dumps(scaling)}, bos {options['bos']}, dtype {options['dtype']}")
PYTHON
for language in $languages; do
  node tests/wikipedia.mjs "$language" "$dir/$language.txt"
  for which in conversion entry; do
    node tests/perplexity.mjs "$dir/$which" "$tokens" "$dir/$language.txt" --file --rows=0 | tee "$dir/$which.$language.txt"
  done
  node - "$id" "$language" "$dir" <<'NODE'
const fs = require("node:fs");
const [id, language, room] = process.argv.slice(2);
const read = (which) => Number(/^\| kernels[^|]*\| ([0-9.]+) \|/m.exec(fs.readFileSync(`${room}/${which}.${language}.txt`, "utf8"))[1]);
const [theirs, ours] = [read("conversion"), read("entry")];
console.log(`rope: ${id}, ${language}: perplexity ${theirs.toFixed(3)} with the conversion's scaling, ${ours.toFixed(3)} with the entry's: ` +
  `the conversion's is ${Math.abs((theirs / ours - 1) * 100).toFixed(2)}% the ${theirs > ours ? "higher" : "lower"}`);
NODE
done
for which in conversion entry; do
  SEEDS="${SEEDS:-1 2}" node tests/write.mjs "$dir/$which" "$dir/$which.page.json" "${WRITE_TOKENS:-48}"
done
rm -rf "$dir"
echo "rope: $id done"
