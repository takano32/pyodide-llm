#!/usr/bin/env bash
# tests/page_gemma4.sh (T329.3): a Gemma 4 on the page's forward pass (src/runtime/forward.js on the int8 kernels, a
# 64-bit memory: the E2B's int8 is 5.2 GB), for CI (tests.yml's extra=, after `make kernels` and with the suite's
# pyodide). The development machine runs none of it (10.2 GB of safetensors):
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=170 extra="bash tests/page_gemma4.sh" --ref <branch> --grep "gemma4:|Perplexity|kernels,|page:|answers"
#   node tests/ci.mjs run tests.yml only_extra=true runner=ubuntu-24.04-arm minutes=170 extra="bash tests/page_gemma4.sh '' 'speed answers'" --ref <branch> ...
#
# With only_extra=true it installs what it needs (numpy, jinja2: the converter reads the template with it; the npm
# packages, the kernels).
#
# The model as ?hf= opens it (tests/hf_fetch.py: the repository's safetensors, config.json, the tokenizer and the chat
# template; no entry of the list has it yet), converted the way the page converts it (tests/perplexity_prepare.py, the
# converter's options alone), then the parts the second word names ("perplexity speed answers float32" when left out):
#   perplexity  the int8 weights on the kernels with 7-bit and 8-bit activations, 1500 tokens of English and of Japanese
#               Wikipedia (T85's articles: past the window of 512 of its local layers), read from the file into
#               forward.js's memory (tests/perplexity.mjs --file: the table of every layer's input in that memory too)
#   speed       tokens a second with 1, 2 and 4 threads, the logits the same to the bit with each, a prompt in blocks,
#               what forward.js placed against footprint(), and that no GPU is started (tests/page-27b.mjs speed)
#   answers     twelve questions (six Japanese, six English), greedy and with two seeds as generation_config.json samples
#               (temperature 1, top-p 0.95, top-k 64), up to TOKENS tokens (300) in the format of the model's template
#               (tests/answers.mjs): what it writes, whether it stops, the language of the answer
#   float32     the float32 conversion in native NumPy (tests/perplexity_native.py, 18.6 GB mapped): the original's
#               perplexity, which transformers computes (tests/reference_gemma4.py holds the two together)
#   against     (T529, not of the default parts) the model against the original's safetensors (AGAINST, hf:<repository>@
#               <revision>; E2B's where none is said), both converted to int8: the page's engine's logits at the same
#               ids (tests/logits.mjs, AGAINST_TOKENS of each language, 96), the largest difference, the most likely token,
#               the KL and the perplexity of each
# The first word is the model (hf:<repository>@<revision>, or an entry of the list: hf-gemma-4-e2b takes a GGUF's
# weights from T529), the third the languages ("en ja").
#
#   node tests/ci.mjs run tests.yml only_extra=true minutes=170 extra="bash tests/page_gemma4.sh hf-gemma-4-e2b 'against perplexity answers'" --ref <branch> --grep "gemma4:|Perplexity|kernels,|page:|answers"
set -euo pipefail
id="${1:-hf:google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7}"
[ -n "$id" ] || id="hf:google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7"
parts=" ${2:-perplexity speed answers float32} "
languages="${3:-en ja}"
tokens=1500
status=0
# a tool that fails says why on its last lines (Node prints Pyodide's whole source line first, longer than a log keeps)
said() { "$@" 2> "$dir/stderr.txt" || { status=1; awk 'length($0) < 2000' "$dir/stderr.txt" | tail -30 | sed 's/^/gemma4: stderr: /'; echo "gemma4: FAILED: $*"; }; }
has() { case "$parts" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
dir="${RUNNER_TEMP:-.tmp}/gemma4-page"
if [ -n "${RUNNER_TEMP:-}" ] && [ -d /mnt ] && sudo -n true 2>/dev/null; then
  dir=/mnt/gemma4-page
  sudo mkdir -p "$dir" && sudo chown "$USER" "$dir"
fi
mkdir -p "$dir"
python -c "import numpy, jinja2" 2>/dev/null || python -m pip install -q numpy jinja2
[ -d node_modules/pyodide ] || npm ci > /dev/null
make kernels > /dev/null
echo "gemma4: $id on $(lscpu | sed -n 's/^Model name: *//p' | head -1) ($(uname -m)), $(nproc) logical cores, $(free -g | awk '/Mem:/{print $2}') GB of memory, $(df -BG --output=avail "$dir" | tail -1 | tr -d ' ') of disk"
began=$SECONDS
model=$(python tests/hf_fetch.py "$id" "$dir" | tail -1)
echo "gemma4: fetched in $((SECONDS - began)) s: $(du -shL "$model" | cut -f1)"
for language in $languages; do node tests/wikipedia.mjs "$language" "$dir/$language.txt"; done
began=$SECONDS
# (an entry of the list: its options under the converter's, as the worker merges them; T529)
entry=(); case "$id" in hf:*) ;; *) entry=(--entry "$id") ;; esac
python tests/perplexity_prepare.py "$model" "$dir/int8" int8 "${entry[@]}" 2>&1 | tail -1 | cut -c1-400
echo "gemma4: converted to int8 in $((SECONDS - began)) s (native Python); the checkpoint is $(stat -c %s "$dir/int8.bin") bytes"
if has perplexity; then
  for language in $languages; do
    echo "gemma4: the kernels, $language"
    said node tests/perplexity.mjs "$dir/int8" "$tokens" "$dir/$language.txt" --file --wide
  done
fi
if has speed; then
  echo "gemma4: the speed of $id"
  said node tests/page-27b.mjs "$dir/int8" speed --threads 1,2,4 --positions 24 --rounds 2 --entry none
fi
if has answers; then
  # the page's options and the model's format (the converter's template, else the one the template writes without thinking)
  python - "$dir" <<'EOF'
import json, struct, sys
from pathlib import Path
out = Path(sys.argv[1])
options = json.loads((out / "int8.json").read_text())
template = options.pop("template", None) or "<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
header = list(struct.unpack("<7i", (out / "int8.bin").open("rb").read(28)))
generation = {"temperature": 1.0, "topp": 0.95, "top_k": 64, "repetition_penalty": 1.0}
(out / "page.json").write_text(json.dumps({"id": "gemma-4-e2b", "name": "Gemma 4 E2B", "options": options, "template": template,
                                           "prompt": "", "generation": generation, "header": header}, ensure_ascii=False))
print(f"gemma4: the format {json.dumps(template)}, bos {options['bos']}, stops {options['stop_tokens']}")
EOF
  # (Node 24's V8 reads a 64-bit memory above 4 GiB wrongly in Liftoff's code on arm64: AGENTS.md, the pitfalls of the
  # kernels. page-27b.mjs sets --no-liftoff itself where it must; answers.mjs does not, and its logits were NaN there)
  liftoff=; [ "$(uname -m)" = aarch64 ] && liftoff=--no-liftoff
  GREEDY=1 SEEDS="1 2" said node $liftoff tests/answers.mjs "$dir/int8" "$dir/page.json" "${TOKENS:-300}"
fi
if has against; then
  # T529: the model of the first word (an entry that takes a GGUF's weights) against the original's safetensors
  # converted the same way (AGAINST: the original, hf:<repository>@<revision>): the page's engine's logits at the same
  # ids, one token at a time (tests/logits.mjs, 7-bit activations), the largest difference, how often the most likely
  # token agrees and the mean KL. Two int8 of the same model differ by the rounding of Q8_0's scales (float16) alone
  against="${AGAINST:-hf:google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7}"
  began=$SECONDS
  original=$(python tests/hf_fetch.py "$against" "$dir/original" | tail -1)
  echo "gemma4: the original fetched in $((SECONDS - began)) s: $(du -shL "$original" | cut -f1)"
  python tests/perplexity_prepare.py "$original" "$dir/original-int8" int8 2>&1 | tail -1 | cut -c1-200
  for out in int8 original-int8; do
    python - "$dir" "$out" <<'EOF'
import json, struct, sys
from pathlib import Path
out, name = Path(sys.argv[1]), sys.argv[2]
options = json.loads((out / f"{name}.json").read_text())
header = list(struct.unpack("<7i", (out / f"{name}.bin").open("rb").read(28)))
(out / f"{name}.page.json").write_text(json.dumps({"id": name, "options": options, "header": header}, ensure_ascii=False))
EOF
    for language in $languages; do
      TEXT="$dir/$language.txt" TOKENS="${AGAINST_TOKENS:-96}" OUT="$dir/$out.$language" said node tests/logits.mjs "$dir/$out" "$dir/$out.page.json"
    done
  done
  for language in $languages; do
    python - "$dir" "$language" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
out, language = Path(sys.argv[1]), sys.argv[2]
ids = [json.loads((out / f"{name}.{language}.ids.json").read_text()) for name in ("int8", "original-int8")]
assert ids[0] == ids[1], "the two tokenize differently"
a, b = (np.fromfile(out / f"{name}.{language}.logits.f32", dtype=np.float32).reshape(len(ids[0]), -1).astype(np.float64)
        for name in ("int8", "original-int8"))
log = lambda x: x - x.max(axis=1, keepdims=True) - np.log(np.exp(x - x.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
p, q = log(b), log(a)
kl = (np.exp(p) * (p - q)).sum(axis=1)
nll = lambda x: -float(np.mean([x[i, ids[0][i + 1]] for i in range(len(ids[0]) - 1)]))
print(f"gemma4: against the original, {language}: {len(ids[0])} positions, largest logit difference "
      f"{np.abs(a - b).max():.4f} (logits up to {np.abs(b).max():.2f}), the most likely token the same at "
      f"{int((a.argmax(axis=1) == b.argmax(axis=1)).sum())}/{len(ids[0])}, KL mean {kl.mean():.2e} largest {kl.max():.2e}, "
      f"perplexity {np.exp(nll(q)):.3f} (GGUF) and {np.exp(nll(p)):.3f} (safetensors)")
EOF
  done
  rm -f "$dir/original-int8.bin"
fi
rm -f "$dir/int8.bin"
if has float32; then
  began=$SECONDS
  python tests/perplexity_prepare.py "$model" "$dir/float32" float32 2>&1 | tail -1 | cut -c1-200
  echo "gemma4: converted to float32 in $((SECONDS - began)) s"
  for language in $languages; do
    echo "gemma4: perplexity of the float32 of $id, NumPy, $language: $(python tests/perplexity_native.py "$dir/float32" "$tokens" "$dir/$language.txt")"
  done
  rm -f "$dir/float32.bin"
fi
exit $status
